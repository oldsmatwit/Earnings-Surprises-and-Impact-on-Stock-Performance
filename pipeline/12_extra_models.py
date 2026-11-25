"""
12_extra_models.py
OPTIONAL extension: richer features + more models, evaluated on the SAME chronological split
as 07/09/10 (same --cutoff, same 30-day embargo, everything learned from training rows only).
Does not overwrite any earlier result file.

NEW FEATURES (all use only information available BEFORE the event's reaction day)
  company history   hist_abs_car1d, hist_mean_car1d   mean |CAR| / CAR of the company's previous 8 events
                    hist_slope                         company's own earnings-response slope (CAR on SUE, prev 8 events)
                    exp_reaction                       hist_slope * sue
  surprise shape    abs_sue, sue_x_vol
  market regime     mkt_ret_5d, mkt_ret_20d, mkt_vol_20d, mkt_dd   S&P 500 up to the day BEFORE the reaction day
                    (needs data/benchmark_gspc.csv from 04_collect_prices.py)
  recent reactions  mkt_recent_car1d, sector_recent_car1d   mean 1-day CAR of events with a STRICTLY EARLIER
                    reaction date in the last 30 days (all / same sector); mkt_beat_share_30d
                    sue_vs_sector_90d                    SUE minus the sector's mean SUE over the prior 90 days

NEW MODELS
  Histogram gradient boosting (sklearn), with a monotonic constraint (higher SUE never lowers the
  prediction), hyper-parameters chosen by time-series CV on the training period only
  Equal-weight ENSEMBLE of Logistic / RF / HistGB / NN (extended features)

DESIGN: every model is fit on BOTH feature sets ("base" = the 07 features, "extended") so the
output shows whether the NEW FEATURES or the NEW MODELS (or neither) change anything.
EXTRA TASK: absolute move |CAR| (volatility of the reaction) and "big move" classification.
EXTRA METRIC: top-minus-bottom decile spread of realised CAR (bp), no transaction costs.

Outputs: data/model_dataset_extended.csv, data/extra_models_results.csv, data/extra_abs_move_results.csv,
         data/extra_perm_importance_1d.csv, data/extra_hgb_params.json, figures/extra_models_auc.png
"""
import argparse
import json
import os
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import model_zoo as mz

m07 = mz.m07
m10 = importlib.import_module("10_tune_compare")          # re-use its time-series CV folds
DATA, BENCH = "data/model_dataset.csv", "data/benchmark_gspc.csv"
RS, EMB = mz.RANDOM_STATE, pd.Timedelta(days=mz.EMBARGO_DAYS)
EXT_NEW = ["hist_abs_car1d", "hist_mean_car1d", "hist_slope", "exp_reaction", "abs_sue", "sue_x_vol",
           "mkt_ret_5d", "mkt_ret_20d", "mkt_vol_20d", "mkt_dd", "mkt_recent_car1d",
           "sector_recent_car1d", "mkt_beat_share_30d", "sue_vs_sector_90d"]
HGB_GRID = [dict(learning_rate=lr, max_depth=d, min_samples_leaf=m)
            for lr in (0.03, 0.08) for d in (3, 5) for m in (50, 200)]
HGB_DEFAULT = dict(learning_rate=0.05, max_depth=4, min_samples_leaf=100)
os.makedirs("figures", exist_ok=True)


# ------------------------------------------------------------------ feature engineering
def _roll_time(s, window, min_periods):
    """Mean over events with a STRICTLY earlier reaction date inside `window` (s indexed by react_date)."""
    return s.rolling(window, closed="left", min_periods=min_periods).mean().to_numpy()


def build_features(df, bench_csv=BENCH):
    d = df.sort_values(["react_date", "symbol"]).reset_index(drop=True)
    d["abs_sue"] = d["sue"].abs()
    d["sue_x_vol"] = d["sue"] * d["vol_prior_20d"]

    # ---- company history: only PREVIOUS events of the same company (shift(1)); their 1-day
    #      window ended on their own reaction day, which is before this event's reaction day.
    for c in ("hist_abs_car1d", "hist_mean_car1d", "hist_slope"):
        d[c] = np.nan
    for _, g in d.groupby("symbol"):
        x, y = g["sue"].shift(1), g["car_1d"].shift(1)
        d.loc[g.index, "hist_abs_car1d"] = y.abs().rolling(8, min_periods=3).mean().to_numpy()
        d.loc[g.index, "hist_mean_car1d"] = y.rolling(8, min_periods=3).mean().to_numpy()
        mx, my = x.rolling(8, min_periods=4).mean(), y.rolling(8, min_periods=4).mean()
        cov = (x * y).rolling(8, min_periods=4).mean() - mx * my
        var = (x * x).rolling(8, min_periods=4).mean() - mx ** 2
        d.loc[g.index, "hist_slope"] = (cov / var.where(var > 1e-6)).clip(-0.2, 0.2).to_numpy()
    d["exp_reaction"] = d["hist_slope"] * d["sue"]

    # ---- recent reactions across the market / sector (strictly earlier reaction dates)
    s = d.set_index("react_date")
    d["mkt_recent_car1d"] = _roll_time(s["car_1d"], "30D", 20)
    d["mkt_beat_share_30d"] = _roll_time((s["surprise"] > 0).astype(float), "30D", 20)
    d["sector_recent_car1d"], d["sue_vs_sector_90d"] = np.nan, np.nan
    for _, g in d.groupby("sector"):
        gs = g.set_index("react_date")
        d.loc[g.index, "sector_recent_car1d"] = _roll_time(gs["car_1d"], "30D", 5)
        d.loc[g.index, "sue_vs_sector_90d"] = (gs["sue"].to_numpy() - _roll_time(gs["sue"], "90D", 5))

    # ---- market regime from the S&P 500, value through the day BEFORE the reaction day
    for c in ("mkt_ret_5d", "mkt_ret_20d", "mkt_vol_20d", "mkt_dd"):
        d[c] = np.nan
    if os.path.exists(bench_csv):
        b = pd.read_csv(bench_csv, parse_dates=["date"]).set_index("date")["Close"].sort_index()
        b = b[~b.index.duplicated()]
        f = pd.DataFrame({"mkt_ret_5d": b.pct_change(5), "mkt_ret_20d": b.pct_change(20),
                          "mkt_vol_20d": b.pct_change().rolling(20).std(),
                          "mkt_dd": b / b.rolling(252, min_periods=60).max() - 1}).shift(1)
        f = f.reindex(pd.DatetimeIndex(d["react_date"]), method="ffill")
        for c in f.columns:
            d[c] = f[c].to_numpy()
    else:
        print(f"WARNING: {bench_csv} not found - market-regime features skipped")
    return d, [c for c in EXT_NEW if d[c].notna().any()]


# ------------------------------------------------------------------ models
def make_prep(num):
    n = Pipeline([("win", m07.Winsorizer(1, 99)), ("imp", SimpleImputer(strategy="median")),
                  ("sc", StandardScaler())])
    c = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                  ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", n, num), ("cat", c, m07.CAT)])


def pipe2(model, num):
    return Pipeline([("prep", make_prep(num)), ("model", model)])


def X_of(d, num):
    return d[num + m07.CAT].assign(quarter=d["quarter"].astype(str))


class _HGBBase(BaseEstimator):
    def __init__(self, mono_idx=None, learning_rate=0.05, max_depth=4, min_samples_leaf=100,
                 max_iter=200, l2_regularization=1.0, random_state=RS):
        self.mono_idx, self.learning_rate, self.max_depth = mono_idx, learning_rate, max_depth
        self.min_samples_leaf, self.max_iter = min_samples_leaf, max_iter
        self.l2_regularization, self.random_state = l2_regularization, random_state

    def _kw(self, X):
        cst = np.zeros(X.shape[1], int)
        if self.mono_idx is not None:
            cst[self.mono_idx] = 1            # num block comes first in the ColumnTransformer output
        return dict(learning_rate=self.learning_rate, max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf, max_iter=self.max_iter,
                    l2_regularization=self.l2_regularization, monotonic_cst=cst,
                    early_stopping=False, random_state=self.random_state)


class HGBReg(RegressorMixin, _HGBBase):
    def fit(self, X, y):
        self.m_ = HistGradientBoostingRegressor(**self._kw(X)).fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X)


class HGBClf(ClassifierMixin, _HGBBase):
    def fit(self, X, y):
        self.m_ = HistGradientBoostingClassifier(**self._kw(X)).fit(X, y)
        self.classes_ = self.m_.classes_
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X)

    def predict(self, X):
        return self.m_.predict(X)


def tune_hgb(train, k, task, num, n_folds):
    car, lab = f"car_{k}d", f"positive_car_{k}d"
    d = train.dropna(subset=[car]).sort_values("date").reset_index(drop=True)
    folds, best, best_s = m10.time_folds(d, n_folds), HGB_DEFAULT, -np.inf
    for params in HGB_GRID:
        sc = []
        for tr_m, va_m in folds:
            tr, va = d[tr_m], d[va_m]
            if len(tr) < mz.MIN_TRAIN or len(va) < mz.MIN_TEST:
                continue
            idx = num.index("sue")
            if task == "regression":
                lo, hi = tr[car].quantile([0.01, 0.99])
                ytr = tr[car].clip(lo, hi)
                p = pipe2(HGBReg(mono_idx=idx, **params), num).fit(X_of(tr, num), ytr).predict(X_of(va, num))
                sc.append(m07.reg_metrics(va[car], p, ytr.mean())["r2_oos"])
            elif va[lab].nunique() > 1:
                p = pipe2(HGBClf(mono_idx=idx, **params), num).fit(X_of(tr, num), tr[lab].astype(int))
                sc.append(roc_auc_score(va[lab].astype(int), p.predict_proba(X_of(va, num))[:, 1]))
        if sc and np.mean(sc) > best_s:
            best, best_s = params, float(np.mean(sc))
    return best


# ------------------------------------------------------------------ evaluation helpers
def spread_bp(score, y, q=0.1):
    lo, hi = np.quantile(score, [q, 1 - q])
    top, bot = y[score >= hi], y[score <= lo]
    return float((top.mean() - bot.mean()) * 1e4) if len(top) and len(bot) else np.nan


def boot_auc(y, probs, ref, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in probs}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        for m, p in probs.items():
            draws[m].append(roc_auc_score(y[i], p[i]))
    ci = {m: np.percentile(v, [2.5, 97.5]) for m, v in draws.items()}
    diff = {m: np.percentile(np.array(v) - np.array(draws[ref]), [2.5, 97.5])
            for m, v in draws.items() if m != ref}
    return ci, diff


def run_window(tr, te, k, ext_num, hp, want_perm):
    car, lab = f"car_{k}d", f"positive_car_{k}d"
    tr, te = tr.dropna(subset=[car]), te.dropna(subset=[car])
    if len(tr) < mz.MIN_TRAIN or len(te) < mz.MIN_TEST:
        return [], None
    lo, hi = tr[car].quantile([0.01, 0.99])
    ytr, ytr_c = tr[car].clip(lo, hi), tr[lab].astype(int)
    yte, yc = te[car].to_numpy(), te[lab].astype(int).to_numpy()
    hr, hc = hp[f"{k}d|regression"], hp[f"{k}d|classification"]
    reg, clf, perm = {}, {}, None
    for fs, num in (("base", m07.NUM), ("extended", ext_num)):
        i = num.index("sue")
        Xtr, Xte = X_of(tr, num), X_of(te, num)
        regs = {"Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=25, max_features=0.5,
                                                       n_jobs=-1, random_state=RS),
                "HistGB (mono SUE)": HGBReg(mono_idx=i, **hr)}
        clfs = {"Logistic Regression": LogisticRegression(max_iter=2000),
                "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=25, max_features="sqrt",
                                                        n_jobs=-1, random_state=RS),
                "HistGB (mono SUE)": HGBClf(mono_idx=i, **hc)}
        if fs == "extended":
            regs["Neural Net"] = mz.MLPEnsembleRegressor(**mz.NN_PARAMS)
            clfs["Neural Net"] = mz.MLPEnsembleClassifier(**mz.NN_PARAMS)
        for n, m in regs.items():
            reg[f"{n} [{fs}]"] = pipe2(m, num).fit(Xtr, ytr).predict(Xte)
        for n, m in clfs.items():
            p = pipe2(m, num).fit(Xtr, ytr_c)
            clf[f"{n} [{fs}]"] = p.predict_proba(Xte)[:, 1]
            if want_perm and k == 1 and fs == "extended" and n == "HistGB (mono SUE)":
                r = permutation_importance(p, Xte, yc, scoring="roc_auc", n_repeats=3, random_state=RS, n_jobs=-1)
                perm = pd.DataFrame({"feature": Xte.columns, "auc_drop": r.importances_mean}).sort_values(
                    "auc_drop", ascending=False)
    ext_keys = [n for n in clf if n.endswith("[extended]")]
    clf["Ensemble [extended]"] = np.mean([clf[n] for n in ext_keys], axis=0)
    reg["Ensemble [extended]"] = np.mean([reg[n] for n in reg if n.endswith("[extended]")], axis=0)

    ref = "Logistic Regression [base]"
    ci, diff = boot_auc(yc, clf, ref)
    meta = {"window": f"{k}d", "n_train": len(tr), "n_test": len(te)}
    rows = []
    for n, p in reg.items():
        rows.append({**meta, "task": "regression", "model": n, **m07.reg_metrics(pd.Series(yte), p, ytr.mean()),
                     "spread_bp": spread_bp(p, yte)})
    for n, p in clf.items():
        r = {**meta, "task": "classification", "model": n, "auc": roc_auc_score(yc, p),
             "accuracy": float(((p >= 0.5).astype(int) == yc).mean()), "ci_lo": ci[n][0], "ci_hi": ci[n][1],
             "spread_bp": spread_bp(p, yte)}
        if n in diff:
            r["diff_vs_logit_base_lo"], r["diff_vs_logit_base_hi"] = diff[n]
        rows.append(r)
    return rows, perm


def run_abs(tr, te, k, ext_num, hp):
    """|CAR| (size of the reaction) and 'big move' (|CAR| above the training median)."""
    car = f"car_{k}d"
    tr, te = tr.dropna(subset=[car]), te.dropna(subset=[car])
    if len(tr) < mz.MIN_TRAIN or len(te) < mz.MIN_TEST:
        return []
    atr, ate = tr[car].abs(), te[car].abs().to_numpy()
    atr = atr.clip(upper=atr.quantile(0.99))
    big_tr, big_te = (atr > atr.median()).astype(int), (ate > atr.median()).astype(int)
    i = ext_num.index("abs_sue")
    Xtr, Xte = X_of(tr, ext_num), X_of(te, ext_num)
    meta = {"window": f"{k}d", "n_train": len(tr), "n_test": len(te)}
    rows = []
    for n, m in {"Baseline (train mean)": DummyRegressor(), "Linear Regression": LinearRegression(),
                 "HistGB (mono |SUE|)": HGBReg(mono_idx=i, **hp[f"{k}d|regression"])}.items():
        p = pipe2(m, ext_num).fit(Xtr, atr).predict(Xte)
        sse, sst = ((ate - p) ** 2).sum(), ((ate - atr.mean()) ** 2).sum()
        rows.append({**meta, "task": "abs_move_regression", "model": n, "r2_oos": 1 - sse / sst,
                     "spearman": float(pd.Series(ate).corr(pd.Series(p), method="spearman"))
                     if np.std(p) > 1e-12 else np.nan})
    for n, m in {"Logistic Regression": LogisticRegression(max_iter=2000),
                 "HistGB (mono |SUE|)": HGBClf(mono_idx=i, **hp[f"{k}d|classification"])}.items():
        p = pipe2(m, ext_num).fit(Xtr, big_tr).predict_proba(Xte)[:, 1]
        rows.append({**meta, "task": "big_move_classification", "model": n, "auc": roc_auc_score(big_te, p)})
    return rows


def plot_auc(res):
    d = res[res["task"] == "classification"]
    fig, ax = plt.subplots(figsize=(10, 5))
    wins = list(dict.fromkeys(d["window"]))
    for j, (m, g) in enumerate(d.groupby("model")):
        g = g.set_index("window").reindex(wins)
        x = np.arange(len(wins)) + (j - len(d["model"].unique()) / 2) * 0.07
        ax.errorbar(x, g["auc"], yerr=[g["auc"] - g["ci_lo"], g["ci_hi"] - g["auc"]], marker="o",
                    ms=4, lw=1, capsize=2, label=m, ls="--" if "[base]" in m else "-")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(wins)), wins)
    ax.set(title="Direction AUC: base vs extended features, more models (95% bootstrap CI)", ylabel="AUC")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig("figures/extra_models_auc.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", help="same split date as 07/09/10")
    ap.add_argument("--windows", type=int, nargs="+", default=list(m07.WINDOWS))
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--skip-tune", action="store_true")
    ap.add_argument("--skip-perm", action="store_true")
    args = ap.parse_args()

    raw = pd.read_csv(DATA, parse_dates=["date", "react_date"])
    df, ext_new = build_features(raw)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv("data/model_dataset_extended.csv", index=False)
    ext_num = m07.NUM + ext_new
    print(f"{len(ext_new)} new features: {ext_new}")

    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else df["date"].iloc[int(0.75 * len(df))]
    train, test = df[df["date"] < cutoff - EMB], df[df["date"] >= cutoff]
    print(f"Cutoff {cutoff.date()} | train {len(train)} | test {len(test)}")

    hp = {f"{k}d|{t}": (HGB_DEFAULT if args.skip_tune else tune_hgb(train, k, t, ext_num, args.n_folds))
          for k in args.windows for t in ("regression", "classification")}
    json.dump(hp, open("data/extra_hgb_params.json", "w"), indent=1)
    print("HistGB params:", hp)

    rows, abs_rows, perms = [], [], []
    for k in args.windows:
        print(f"  {k}d ...")
        r, perm = run_window(train, test, k, ext_num, hp, not args.skip_perm)
        rows += r
        abs_rows += run_abs(train, test, k, ext_num, hp)
        if perm is not None:
            perms.append(perm)
    res = pd.DataFrame(rows)
    res.to_csv("data/extra_models_results.csv", index=False)
    pd.DataFrame(abs_rows).to_csv("data/extra_abs_move_results.csv", index=False)
    if perms:
        perms[0].to_csv("data/extra_perm_importance_1d.csv", index=False)

    c = res[res["task"] == "classification"]
    print("\n=== Direction AUC (95% CI) and decile spread ===")
    print(c[["window", "model", "auc", "ci_lo", "ci_hi", "diff_vs_logit_base_lo", "diff_vs_logit_base_hi",
             "spread_bp"]].round(4).to_string(index=False))
    print("\n=== Return regression ===")
    print(res[res["task"] == "regression"][["window", "model", "r2_oos", "spearman", "dir_acc", "spread_bp"]]
          .round(4).to_string(index=False))
    if abs_rows:
        print("\n=== Size of the move (|CAR|) and big-move classification ===")
        print(pd.DataFrame(abs_rows).drop(columns=["n_train"]).round(4).to_string(index=False))
    if perms:
        print("\nPermutation importance, 1d HistGB (AUC drop when the column is shuffled):")
        print(perms[0].head(12).round(4).to_string(index=False))
    plot_auc(res)

    best = c[c["window"] == "1d"]["auc"].max() if (c["window"] == "1d").any() else np.nan
    print(f"\nBest 1-day AUC: {best:.3f}")
    if best > 0.70:
        print("!! A 1-day AUC this high is suspicious for earnings data. Before trusting it, check for "
              "look-ahead leakage (any feature built with the event's own return or later events).")
    print("Saved data/extra_*.csv, data/model_dataset_extended.csv, figures/extra_models_auc.png")