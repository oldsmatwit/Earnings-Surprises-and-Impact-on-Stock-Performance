"""
10_tune_compare.py
Week 5: tune the Random Forest, compare it with the neural net, and check how robust the
comparison is.

  A  TUNING (training data only). Random-Forest hyper-parameters (and, with --tune-nn, the NN
     architecture) are chosen by EXPANDING-WINDOW time-series cross-validation inside the
     training period, with the same 30-day embargo as 07. The test set is never used to pick
     anything. -> data/tuning_cv.csv, data/tuned_params.json
  B  COMPARISON on the same chronological split as 07/09: baseline | linear/logistic |
     RF (default, as in 07) | RF (tuned) | Neural Net. Bootstrap 95% CIs on AUC and R2, and a
     PAIRED bootstrap of (Neural Net - tuned RF).  -> data/tuned_comparison.csv,
     figures/tuned_comparison.png
  C  CUTOFF SENSITIVITY: repeat the pre-specified models for several train/test cut dates.
     The test sets are nested (everything after the cutoff), so the rows are NOT independent
     replications; read them as "does the conclusion flip?".  -> data/cutoff_sensitivity.csv,
     figures/cutoff_sensitivity.png
  D  BREAKDOWNS of the main-split test predictions by sector and by Beat / Miss / Meet.
     -> data/breakdowns.csv, figures/breakdowns_1d.png

Run 07 (and 09) first. Use the SAME --cutoff everywhere. After the 2008 back-fill the default
cutoff (75th percentile of event dates) moves, so pin it, e.g. --cutoff 2023-01-01.
"""
import argparse
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

import model_zoo as mz

m07 = mz.m07
DATA = "data/model_dataset.csv"
WINDOWS = m07.WINDOWS
RS = mz.RANDOM_STATE
EMB = pd.Timedelta(days=mz.EMBARGO_DAYS)

RF_GRID = {"min_samples_leaf": [10, 25, 50, 100], "max_features": [0.3, 0.5, "sqrt"]}
NN_GRID = {"hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)], "alpha": [1e-2, 1e-1]}
RF_DEFAULT_REG = dict(min_samples_leaf=25, max_features=0.5)       # exactly the 07 settings
RF_DEFAULT_CLF = dict(min_samples_leaf=25, max_features="sqrt")
N_BOOT = 300
os.makedirs("figures", exist_ok=True)


# ----------------------------------------------------------------------------- factories
def make_rf(task, params, n_estimators=300):
    cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
    return cls(n_estimators=n_estimators, n_jobs=-1, random_state=RS, **params)


def make_nn(task, params):
    cls = mz.MLPEnsembleRegressor if task == "regression" else mz.MLPEnsembleClassifier
    return cls(**params)


def configs(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())]


# ----------------------------------------------------------------------------- tuning CV
def time_folds(d, n_folds=4, first=0.4):
    """Expanding-window folds on a date-sorted frame. Validation blocks tile the last
    (1-first) share of the training dates; each fold trains on events before the block start
    minus the embargo, so a 20-day window can never reach into the validation block."""
    edges = d["date"].quantile(np.linspace(first, 1.0, n_folds + 1)).to_numpy()
    out = []
    for i in range(n_folds):
        a, b = edges[i], edges[i + 1]
        last = i == n_folds - 1
        va = (d["date"] >= a) & ((d["date"] <= b) if last else (d["date"] < b))
        tr = d["date"] < a - EMB
        out.append((tr.to_numpy(), va.to_numpy()))
    return out


def cv_score(d, task, k, build, n_folds):
    """Mean validation score over the folds: R2 vs fold-train mean (regression) or AUC."""
    car, lab = f"car_{k}d", f"positive_car_{k}d"
    scores = []
    for tr_m, va_m in time_folds(d, n_folds):
        tr, va = d[tr_m], d[va_m]
        if len(tr) < mz.MIN_TRAIN or len(va) < mz.MIN_TEST:
            continue
        if task == "regression":
            lo, hi = tr[car].quantile([0.01, 0.99])
            ytr = tr[car].clip(lo, hi)
            pred = m07.pipe(build()).fit(mz._X(tr), ytr).predict(mz._X(va))
            scores.append(m07.reg_metrics(va[car], pred, ytr.mean())["r2_oos"])
        else:
            if va[lab].nunique() < 2:
                continue
            p = m07.pipe(build()).fit(mz._X(tr), tr[lab].astype(int))
            scores.append(roc_auc_score(va[lab].astype(int), p.predict_proba(mz._X(va))[:, 1]))
    return (float(np.mean(scores)), float(np.std(scores)), len(scores)) if scores else (np.nan, np.nan, 0)


def tune(train, k, task, family, grid, n_folds, log):
    car = f"car_{k}d"
    d = train.dropna(subset=[car]).sort_values("date").reset_index(drop=True)
    best, best_s = None, -np.inf
    for params in configs(grid):
        build = ((lambda: make_rf(task, params, 200)) if family == "RF"
                 else (lambda: make_nn(task, params)))
        mean, sd, n = cv_score(d, task, k, build, n_folds)
        log.append({"window": f"{k}d", "task": task, "family": family, "params": json.dumps(params, default=str),
                    "cv_score": mean, "cv_sd": sd, "n_folds": n,
                    "metric": "r2_oos" if task == "regression" else "auc"})
        if n and mean > best_s:
            best, best_s = params, mean
    return best


def load_or_tune(train, windows, n_folds, tune_nn, cache="data/tuned_params.json", reuse=False):
    if reuse and os.path.exists(cache):
        print(f"Re-using {cache}")
        raw = json.load(open(cache))
        for v in raw.values():                                   # JSON turns tuples into lists
            if "hidden_layer_sizes" in v.get("nn", {}):
                v["nn"]["hidden_layer_sizes"] = tuple(v["nn"]["hidden_layer_sizes"])
        return raw
    hp, log = {}, []
    for k in windows:
        for task in ("regression", "classification"):
            key = f"{k}d|{task}"
            rf = tune(train, k, task, "RF", RF_GRID, n_folds, log) or (
                RF_DEFAULT_REG if task == "regression" else RF_DEFAULT_CLF)
            nn = (tune(train, k, task, "NN", NN_GRID, n_folds, log) if tune_nn else None) or mz.NN_PARAMS
            hp[key] = {"rf": rf, "nn": nn}
            print(f"  tuned {key}: RF {rf} | NN {nn}")
    pd.DataFrame(log).to_csv("data/tuning_cv.csv", index=False)
    json.dump(hp, open(cache, "w"), indent=1, default=str)
    return hp


# ----------------------------------------------------------------------------- bootstrap
def _r2(y, p, mu):
    return 1 - ((y - p) ** 2).sum() / ((y - mu) ** 2).sum()


def boot(y, preds, mu, task, n_boot=N_BOOT, seed=0):
    """preds: {name: array}. Returns per-model 95% CI and paired CI of NN - tuned RF."""
    rng = np.random.default_rng(seed)
    n, draws = len(y), {m: [] for m in preds}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if task == "classification" and len(np.unique(y[i])) < 2:
            continue
        for m, p in preds.items():
            draws[m].append(roc_auc_score(y[i], p[i]) if task == "classification" else _r2(y[i], p[i], mu))
    ci = {m: np.percentile(v, [2.5, 97.5]) for m, v in draws.items() if v}
    diff = None
    if "Neural Net" in draws and "RF (tuned)" in draws:
        dd = np.array(draws["Neural Net"]) - np.array(draws["RF (tuned)"])
        diff = np.percentile(dd, [2.5, 97.5])
    return ci, diff


# ----------------------------------------------------------------------------- B: compare
def predict_all(train, te, k, hp):
    car, lab = f"car_{k}d", f"positive_car_{k}d"
    tr, te = train.dropna(subset=[car]), te.dropna(subset=[car])
    Xtr, Xte = mz._X(tr), mz._X(te)
    lo, hi = tr[car].quantile([0.01, 0.99])
    ytr = tr[car].clip(lo, hi)
    hr, hc = hp[f"{k}d|regression"], hp[f"{k}d|classification"]
    regs = {"Baseline (train mean)": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "RF (default)": make_rf("regression", RF_DEFAULT_REG),
            "RF (tuned)": make_rf("regression", hr["rf"]),
            "Neural Net": make_nn("regression", hr["nn"])}
    clfs = {"Baseline (majority class)": DummyClassifier(strategy="prior"),
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "RF (default)": make_rf("classification", RF_DEFAULT_CLF),
            "RF (tuned)": make_rf("classification", hc["rf"]),
            "Neural Net": make_nn("classification", hc["nn"])}
    reg = {n: m07.pipe(m).fit(Xtr, ytr).predict(Xte) for n, m in regs.items()}
    ytr_c = tr[lab].astype(int)
    clf = {n: m07.pipe(m).fit(Xtr, ytr_c).predict_proba(Xte)[:, 1] for n, m in clfs.items()}
    return tr, te, ytr.mean(), reg, clf


def compare(train, test, windows, hp):
    rows, bd_rows, store = [], [], {}
    for k in windows:
        print(f"  comparing {k}d ...")
        tr, te, mu, reg, clf = predict_all(train, test, k, hp)
        if len(tr) < mz.MIN_TRAIN or len(te) < mz.MIN_TEST:
            continue
        store[k] = (te, mu, reg, clf)
        meta = {"window": f"{k}d", "n_train": len(tr), "n_test": len(te)}
        y = te[f"car_{k}d"].to_numpy()
        yc = te[f"positive_car_{k}d"].astype(int).to_numpy()
        ci_r, d_r = boot(y, {n: p for n, p in reg.items() if not n.startswith("Baseline")}, mu, "regression")
        ci_c, d_c = boot(yc, {n: p for n, p in clf.items() if not n.startswith("Baseline")}, mu, "classification")
        for n, p in reg.items():
            r = {**meta, "task": "regression", "model": n, **m07.reg_metrics(te[f"car_{k}d"], p, mu)}
            if n in ci_r:
                r["ci_lo"], r["ci_hi"] = ci_r[n]
            rows.append(r)
        for n, p in clf.items():
            r = {**meta, "task": "classification", "model": n,
                 **mz._clf_metrics(te[f"positive_car_{k}d"].astype(int), (p >= 0.5).astype(int), p)}
            if n in ci_c:
                r["ci_lo"], r["ci_hi"] = ci_c[n]
            rows.append(r)
        for task, d in (("regression", d_r), ("classification", d_c)):
            if d is not None:
                rows.append({**meta, "task": task, "model": "DIFF: Neural Net - RF (tuned)",
                             "ci_lo": d[0], "ci_hi": d[1]})
    out = pd.DataFrame(rows)
    out.to_csv("data/tuned_comparison.csv", index=False)
    return out, store


def plot_compare(out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a, task, metric, ttl, ref in [(ax[0], "classification", "auc", "Direction AUC (out-of-sample)", 0.5),
                                      (ax[1], "regression", "r2_oos", "Return R2 vs train mean", 0.0)]:
        d = out[(out["task"] == task) & ~out["model"].str.startswith(("DIFF", "Baseline"))]
        for j, (m, g) in enumerate(d.groupby("model")):
            x = np.arange(len(g)) + (j - 1.5) * 0.08
            a.errorbar(x, g[metric], yerr=[g[metric] - g["ci_lo"], g["ci_hi"] - g[metric]],
                       marker="o", capsize=2, lw=1, label=m)
        a.set_xticks(np.arange(g.shape[0]), g["window"].tolist())
        a.axhline(ref, color="k", lw=0.8, ls="--")
        a.set(title=ttl)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/tuned_comparison.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- C: cutoffs
def cutoff_sensitivity(df, quantiles, windows):
    rows = []
    for q in quantiles:
        cutoff = df["date"].iloc[int(q * len(df))]
        tr, te = df[df["date"] < cutoff - EMB], df[df["date"] >= cutoff]
        print(f"  cutoff {cutoff.date()} (train {len(tr)}, test {len(te)})")
        for k in windows:
            rows += mz.fit_eval(tr, te, k, meta={"cutoff": str(cutoff.date()), "train_share": q})
    out = pd.DataFrame(rows)
    out.to_csv("data/cutoff_sensitivity.csv", index=False)
    return out


def plot_cutoffs(out):
    if out.empty:
        return
    w = "1d" if (out["window"] == "1d").any() else out["window"].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, task, metric, ref in [(ax[0], "classification", "auc", 0.5), (ax[1], "regression", "r2_oos", 0.0)]:
        d = out[(out["task"] == task) & (out["window"] == w) & ~out["model"].str.startswith(("Baseline", "Rule"))]
        for m, g in d.groupby("model"):
            a.plot(g["cutoff"], g[metric], marker="o", label=m)
        a.axhline(ref, color="k", lw=0.8, ls="--")
        a.set(title=f"{metric} by train/test cutoff ({w} window)", xlabel="cutoff date")
        a.tick_params(axis="x", rotation=25)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/cutoff_sensitivity.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- D: breakdowns
def breakdowns(store, min_n=50):
    rows = []
    for k, (te, mu, reg, clf) in store.items():
        car, lab = f"car_{k}d", f"positive_car_{k}d"
        direction = pd.Series(np.select([te["surprise"] > 0, te["surprise"] < 0], ["Beat", "Miss"], "Meet"),
                              index=te.index)
        for dim, col in (("sector", te["sector"]), ("direction", direction)):
            for lvl in sorted(col.dropna().unique()):
                m = (col == lvl).to_numpy()
                if m.sum() < min_n:
                    continue
                y, yc = te[car].to_numpy()[m], te[lab].astype(int).to_numpy()[m]
                base = {"window": f"{k}d", "dimension": dim, "level": lvl, "n": int(m.sum()),
                        "share_positive": float(yc.mean())}
                for name, p in reg.items():
                    r = m07.reg_metrics(pd.Series(y), p[m], mu)
                    rows.append({**base, "task": "regression", "model": name,
                                 "r2_oos": r["r2_oos"], "spearman": r["spearman"], "dir_acc": r["dir_acc"]})
                for name, p in clf.items():
                    c = mz._clf_metrics(pd.Series(yc), (p[m] >= 0.5).astype(int), p[m])
                    rows.append({**base, "task": "classification", "model": name,
                                 "auc": c["auc"], "accuracy": c["accuracy"]})
    out = pd.DataFrame(rows)
    out.to_csv("data/breakdowns.csv", index=False)
    return out


def plot_breakdowns(bd):
    d = bd[(bd["window"] == "1d") & (bd["task"] == "classification")
           & bd["model"].isin(["Logistic Regression", "RF (tuned)", "Neural Net"])]
    if d.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    for a, dim in zip(ax, ("sector", "direction")):
        piv = d[d["dimension"] == dim].pivot_table(index="level", columns="model", values="auc")
        if piv.empty:
            continue
        piv.plot.bar(ax=a, width=0.8)
        a.axhline(0.5, color="k", lw=0.8, ls="--")
        a.set(title=f"1-day direction AUC by {dim} (main split, test set)", xlabel="")
        a.tick_params(axis="x", rotation=40)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/breakdowns_1d.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", help="same split date as 07/09 (strongly recommended)")
    ap.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS))
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--tune-nn", action="store_true", help="also tune NN size/alpha by CV (fairer vs tuned RF)")
    ap.add_argument("--reuse-tuning", action="store_true", help="load data/tuned_params.json instead of re-tuning")
    ap.add_argument("--cutoff-quantiles", type=float, nargs="+", default=[0.60, 0.70, 0.75, 0.80])
    ap.add_argument("--cutoff-windows", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--skip-cutoff", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else df["date"].iloc[int(0.75 * len(df))]
    train, test = df[df["date"] < cutoff - EMB], df[df["date"] >= cutoff]
    print(f"Cutoff {cutoff.date()} | train {len(train)} | test {len(test)}")

    print("\nA. Tuning (time-series CV inside the training period only)")
    hp = load_or_tune(train, args.windows, args.n_folds, args.tune_nn, reuse=args.reuse_tuning)

    print("\nB. Comparison on the main split")
    comp, store = compare(train, test, args.windows, hp)
    show = ["window", "model", "r2_oos", "spearman", "dir_acc", "accuracy", "auc", "ci_lo", "ci_hi"]
    for task in ("regression", "classification"):
        print(f"\n=== {task} ===")
        print(comp[comp["task"] == task][[c for c in show if c in comp]].dropna(axis=1, how="all").round(4).to_string(index=False))
    plot_compare(comp)

    print("\nD. Breakdowns by sector and by Beat/Miss/Meet")
    plot_breakdowns(breakdowns(store))

    if not args.skip_cutoff:
        print("\nC. Cutoff sensitivity (pre-specified settings, nested test sets)")
        plot_cutoffs(cutoff_sensitivity(df, args.cutoff_quantiles, args.cutoff_windows))

    print("\nSaved data/tuning_cv.csv, tuned_params.json, tuned_comparison.csv, breakdowns.csv, "
          "cutoff_sensitivity.csv and figures/tuned_comparison|breakdowns_1d|cutoff_sensitivity.png")