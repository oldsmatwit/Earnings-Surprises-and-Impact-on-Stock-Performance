"""
08_stress_test.py
Stress test of the earnings-surprise results in the 2008 and 2020 crises.

PART A  Do the relationships change in a crisis?  (descriptive, no model)
        Per regime (GFC_2008 / COVID_2020 / Normal) and window: mean and spread of the
        market-adjusted CAR, beat-minus-miss gap (Welch t-test), Spearman(SUE, CAR).
PART B  Do the models break in a crisis?  (Linear/Logistic, Random Forest, Neural Net)
        Two ways to build the training set, because they answer different questions:
          forward    train ONLY on events before the crisis (minus embargo)  -> "would a model
                     trained in calm times have survived the crisis?"  This is the honest
                     forecasting test, but it needs enough pre-crisis history. For 2008 that
                     usually does not exist in this dataset, so it is skipped automatically.
          leave-out  train on all NON-crisis events (before and after, minus embargo around
                     every crisis window), test on the crisis. Uses future data, so it is NOT a
                     forecasting test; it tests whether calm-period relationships transfer
                     to crisis conditions. Always possible if the crisis has events.
        The calm hold-out results from 07_models.py (if present) are added as a reference.

Crisis windows are by REACTION date and are easy to change in CRISES below.

Outputs: data/stress_descriptives.csv, data/stress_results.csv,
         figures/stress_descriptives.png, figures/stress_model_performance.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import model_zoo as mz

DATA = "data/model_dataset.csv"
CALM_RESULTS = "data/model_results.csv"
WINDOWS = mz.m07.WINDOWS
EMB = pd.Timedelta(days=mz.EMBARGO_DAYS)
MIN_FORWARD_TRAIN = 1000          # below this a "forward" model is not meaningful

CRISES = {   # name: (start, end) of reaction dates
    "GFC_2008":   (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-06-30")),  # Lehman -> post-trough
    "COVID_2020": (pd.Timestamp("2020-02-19"), pd.Timestamp("2020-06-30")),  # peak -> rebound
}

os.makedirs("figures", exist_ok=True)


def regime_of(d: pd.Series) -> pd.Series:
    out = pd.Series("Normal", index=d.index)
    for name, (s, e) in CRISES.items():
        out[(d >= s) & (d <= e)] = name
    return out


# ------------------------------------------------------------------- Part A
def descriptives(df):
    rows = []
    for reg, g in df.groupby("regime"):
        for k in WINDOWS:
            c = f"car_{k}d"
            d = g.dropna(subset=[c, "sue"])
            if len(d) < 30:
                continue
            beat, miss = d.loc[d["surprise"] > 0, c], d.loc[d["surprise"] < 0, c]
            t, p = stats.ttest_ind(beat, miss, equal_var=False) if min(len(beat), len(miss)) > 2 else (np.nan, np.nan)
            rho, p_rho = stats.spearmanr(d["sue"], d[c])
            rows.append({"regime": reg, "window": f"{k}d", "n": len(d),
                         "mean_car": d[c].mean(), "sd_car": d[c].std(),
                         "mean_abs_car": d[c].abs().mean(), "share_positive": (d[c] > 0).mean(),
                         "share_beat": (d["surprise"] > 0).mean(),
                         "beat_minus_miss": beat.mean() - miss.mean(), "welch_t": t, "p_welch": p,
                         "spearman_sue": rho, "p_spearman": p_rho})
    out = pd.DataFrame(rows)
    out.to_csv("data/stress_descriptives.csv", index=False)
    print("\n=== Part A: crisis vs normal ===")
    print(out.round(4).to_string(index=False))

    o1 = out[out["window"] == "1d"].set_index("regime")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, col, ttl in zip(ax, ["beat_minus_miss", "sd_car", "spearman_sue"],
                           ["Beat minus miss, mean 1-day CAR", "Std dev of 1-day CAR",
                            "Spearman(SUE, 1-day CAR)"]):
        v = o1[col] * (100 if col != "spearman_sue" else 1)
        a.bar(v.index, v.values, color=["#C44E52" if i != "Normal" else "#8C8C8C" for i in v.index])
        a.set(title=ttl, ylabel="pp" if col != "spearman_sue" else "rho")
        a.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig("figures/stress_descriptives.png", dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------- Part B
def splits_for(df, crisis):
    s, e = CRISES[crisis]
    te = df[(df["react_date"] >= s) & (df["react_date"] <= e)]
    fwd = df[df["date"] < s - EMB]
    near_any = pd.Series(False, index=df.index)
    for (s2, e2) in CRISES.values():
        near_any |= (df["react_date"] >= s2 - EMB) & (df["react_date"] <= e2 + EMB)
    return te, {"forward": fwd, "leave-out": df[~near_any]}


def model_stress(df, windows=WINDOWS):
    rows = []
    for crisis in CRISES:
        te, trains = splits_for(df, crisis)
        print(f"\n--- {crisis}: {len(te)} crisis events ---")
        if len(te) < mz.MIN_TEST:
            print("    not enough events in the crisis window - see note on data coverage")
            continue
        for mode, tr in trains.items():
            if mode == "forward" and len(tr) < MIN_FORWARD_TRAIN:
                print(f"  [forward] only {len(tr)} pre-crisis events -> skipped")
                continue
            print(f"  [{mode}] train {len(tr)}")
            for k in windows:
                rows += mz.fit_eval(tr, te, k, meta={"regime": crisis, "mode": mode})
    res = pd.DataFrame(rows)

    # calm reference from 07 (+09), so crisis numbers can be read against normal times
    if os.path.exists(CALM_RESULTS):
        calm = pd.read_csv(CALM_RESULTS)
        calm.insert(0, "mode", "chronological")
        calm.insert(0, "regime", "Calm hold-out (07)")
        res = pd.concat([res, calm], ignore_index=True)
    res.to_csv("data/stress_results.csv", index=False)
    return res


def plot_model_stress(res):
    if res.empty:
        return
    res = res.copy()
    res["setup"] = res["regime"].str.replace("Calm hold-out (07)", "Calm", regex=False) + "\n" + res["mode"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, task, metric, ttl, ref in [
            (ax[0], "classification", "auc", "Direction AUC (1-day window)", 0.5),
            (ax[1], "regression", "r2_oos", "Out-of-sample R² vs train mean (1-day window)", 0.0)]:
        d = res[(res["task"] == task) & (res["window"] == "1d") & (~res["model"].str.startswith("Baseline"))]
        if d.empty:
            continue
        piv = d.pivot_table(index="setup", columns="model", values=metric)
        piv.plot.bar(ax=a, rot=0, width=0.8)
        a.axhline(ref, color="k", lw=0.8, ls="--")
        a.set(title=ttl, xlabel="")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/stress_model_performance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS),
                    help="return windows for Part B, e.g. --windows 1 5 (faster)")
    args = ap.parse_args()
    MODEL_WINDOWS = tuple(args.windows)
    df = pd.read_csv(DATA, parse_dates=["date", "react_date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["direction"] = np.select([df["surprise"] > 0, df["surprise"] < 0], ["Beat", "Miss"], "Meet")
    df["regime"] = regime_of(df["react_date"])

    print(f"Dataset: {len(df)} events, {df['date'].min().date()} to {df['date'].max().date()}")
    print(df["regime"].value_counts().to_string())
    if (df["regime"] == "GFC_2008").sum() < mz.MIN_TEST:
        print("\n!! Almost no 2008-09 events: the earnings pull does not reach that far back. "
              "Check data/earnings_availability_summary.csv ('earliest') and fix coverage "
              "(Alpha Vantage / other source) before reporting a 2008 stress test.")

    descriptives(df)
    res = model_stress(df, MODEL_WINDOWS)
    plot_model_stress(res)

    show = [c for c in ["regime", "mode", "window", "model", "n_train", "n_test", "r2_oos",
                        "dir_acc", "accuracy", "auc"] if c in res.columns]
    if len(res):
        print("\n=== Part B: model performance ===")
        print(res[show].round(4).to_string(index=False))
    print("\nSaved data/stress_*.csv and figures/stress_*.png")