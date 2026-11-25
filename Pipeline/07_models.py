"""
07_models.py
First modeling results for the midterm.

Split: CHRONOLOGICAL. Events before the cutoff train, events after test. Default cutoff =
the date of the 75th-percentile event (or pass --cutoff YYYY-MM-DD). Training events within
EMBARGO_DAYS before the cutoff are dropped so a 20-day return window can never overlap the
test period.

For each window (1, 3, 5, 20 days):
  Regression      target car_kd            baseline (train mean) | Linear Regression | Random Forest
  Classification  target positive_car_kd   baseline (majority)   | rule "beat = up" | Logistic | Random Forest

Everything that learns from data (winsorizing, imputing, scaling, encoding) is fit on the
TRAINING rows only, inside the sklearn Pipeline.

Outputs: data/model_results.csv, data/rf_feature_importance.csv, figures/model_comparison.png,
         figures/rf_importance_1d.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "data/model_dataset.csv"
WINDOWS = (1, 3, 5, 20)
EMBARGO_DAYS = 30
RANDOM_STATE = 42

NUM = ["surprise_pct", "sue", "surprise_dir", "prior_surprise_pct", "beats_last_4",
       "ret_prior_20d", "ret_prior_60d", "vol_prior_20d", "volume_ratio_5d_vs_60d"]
CAT = ["sector", "cap_tier", "freq_tier", "age_tier", "quarter", "report_timing"]

os.makedirs("figures", exist_ok=True)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to percentile cutoffs learned on the training data only."""
    def __init__(self, lower=1.0, upper=99.0):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_ = np.nanpercentile(X, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_preprocessor():
    num = Pipeline([("win", Winsorizer(1, 99)),
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])


def pipe(model):
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def reg_metrics(y, pred, train_mean):
    sse, sst = ((y - pred) ** 2).sum(), ((y - train_mean) ** 2).sum()
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2_oos": float(1 - sse / sst),                      # vs. the training-mean forecast
            "spearman": float(spearmanr(y, pred)[0]) if np.std(pred) > 1e-12 else np.nan,
            "dir_acc": float((np.sign(pred) == np.sign(y)).mean())}


def clf_metrics(y, pred, score):
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auc": float(roc_auc_score(y, score))}


def run(df, cutoff):
    train_all = df[df["date"] < cutoff - pd.Timedelta(days=EMBARGO_DAYS)]
    test_all = df[df["date"] >= cutoff]
    print(f"Cutoff {cutoff.date()}  |  train events: {len(train_all)}  test events: {len(test_all)}")
    print(f"Train period {train_all['date'].min().date()} to {train_all['date'].max().date()}; "
          f"test period {test_all['date'].min().date()} to {test_all['date'].max().date()}\n")

    results, importances = [], []
    for k in WINDOWS:
        car, lab = f"car_{k}d", f"positive_car_{k}d"
        tr, te = train_all.dropna(subset=[car]), test_all.dropna(subset=[car])
        if len(tr) < 200 or len(te) < 50:
            print(f"{k}d: not enough data, skipping")
            continue
        meta = {"window": f"{k}d", "n_train": len(tr), "n_test": len(te)}

        # ---------------- regression ----------------
        lo, hi = tr[car].quantile([0.01, 0.99])                  # train-only target clipping
        ytr, yte = tr[car].clip(lo, hi), te[car]
        regs = {"Baseline (train mean)": DummyRegressor(strategy="mean"),
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=25,
                                                       max_features=0.5, n_jobs=-1,
                                                       random_state=RANDOM_STATE)}
        for name, m in regs.items():
            p = pipe(m).fit(tr[NUM + CAT].assign(quarter=tr["quarter"].astype(str)), ytr)
            pred = p.predict(te[NUM + CAT].assign(quarter=te["quarter"].astype(str)))
            results.append({**meta, "task": "regression", "model": name,
                            **reg_metrics(yte, pred, ytr.mean())})

        # ---------------- classification ----------------
        ytr_c, yte_c = tr[lab].astype(int), te[lab].astype(int)
        base_acc = max(yte_c.mean(), 1 - yte_c.mean())
        clfs = {"Baseline (majority class)": DummyClassifier(strategy="prior"),
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=25,
                                                        max_features="sqrt", n_jobs=-1,
                                                        random_state=RANDOM_STATE)}
        Xtr = tr[NUM + CAT].assign(quarter=tr["quarter"].astype(str))
        Xte = te[NUM + CAT].assign(quarter=te["quarter"].astype(str))
        for name, m in clfs.items():
            p = pipe(m).fit(Xtr, ytr_c)
            score = p.predict_proba(Xte)[:, 1]
            results.append({**meta, "task": "classification", "model": name,
                            **clf_metrics(yte_c, p.predict(Xte), score), "majority_acc": base_acc})
            if name == "Random Forest":
                names = p.named_steps["prep"].get_feature_names_out()
                imp = pd.Series(p.named_steps["model"].feature_importances_, index=names)
                importances.append(imp.rename(f"{k}d"))
        rule = (te["surprise"] > 0).astype(int)                  # naive rule: beat -> up, else down
        results.append({**meta, "task": "classification", "model": "Rule: beat = up",
                        **clf_metrics(yte_c, rule, te["surprise_dir"]), "majority_acc": base_acc})

    res = pd.DataFrame(results)
    res.to_csv("data/model_results.csv", index=False)
    if importances:
        imp = pd.concat(importances, axis=1)
        imp.index = [i.replace("num__", "").replace("cat__", "") for i in imp.index]
        imp.sort_values(imp.columns[0], ascending=False).round(4).to_csv("data/rf_feature_importance.csv")
    return res, (pd.concat(importances, axis=1) if importances else None)


def plot_results(res, imp):
    reg, clf = res[res["task"] == "regression"], res[res["task"] == "classification"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, g in clf.groupby("model"):
        ax[0].plot(g["window"], g["auc"], marker="o", label=model)
    ax[0].axhline(0.5, color="k", lw=0.8, ls="--")
    ax[0].set(title="Direction classification: out-of-sample AUC", ylabel="AUC (0.5 = coin flip)")
    ax[0].legend(fontsize=8)
    for model, g in reg.groupby("model"):
        ax[1].plot(g["window"], g["r2_oos"], marker="o", label=model)
    ax[1].axhline(0, color="k", lw=0.8, ls="--")
    ax[1].set(title="Return regression: out-of-sample R² vs train mean", ylabel="R²")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/model_comparison.png", dpi=150)
    plt.close(fig)

    if imp is not None:
        s = imp["1d"].copy()
        s.index = [i.replace("num__", "").replace("cat__", "") for i in s.index]
        s = s.sort_values().tail(12)
        fig, ax = plt.subplots(figsize=(7, 5))
        s.plot.barh(ax=ax, color="#4C72B0")
        ax.set(title="Random Forest feature importance (1-day direction)", xlabel="importance")
        fig.tight_layout()
        fig.savefig("figures/rf_importance_1d.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", help="train/test split date, e.g. 2023-01-01")
    args = ap.parse_args()

    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    cutoff = (pd.Timestamp(args.cutoff) if args.cutoff
              else df["date"].iloc[int(0.75 * len(df))])
    res, imp = run(df, cutoff)

    show = [c for c in ["window", "model", "n_test", "rmse", "mae", "r2_oos", "spearman", "dir_acc",
                        "accuracy", "f1", "auc", "majority_acc"] if c in res.columns]
    for task in ("regression", "classification"):
        print(f"\n=== {task} ===")
        print(res[res["task"] == task][show].dropna(axis=1, how="all").round(4).to_string(index=False))
    plot_results(res, imp)
    print("\nSaved data/model_results.csv, data/rf_feature_importance.csv, figures/*.png")