"""
09_neural_net.py
Neural network vs classical ML, on the SAME chronological split as 07_models.py.

Run 07_models.py first (with the same --cutoff, if you used one). This script
  1. fits the neural net (MLP ensemble, 5 seeds, early stopping) for every window,
  2. appends its rows to data/model_results.csv (replacing any earlier NN rows) and redraws
     figures/model_comparison.png so the NN is a line next to Linear / Logistic / RF,
  3. runs a small architecture sensitivity check -> data/nn_sensitivity.csv. The MAIN model is
     pre-specified as (64, 32); the other sizes are only reported, never used to pick a winner
     on the test set.

Outputs: data/model_results.csv (updated), data/nn_sensitivity.csv, figures/model_comparison.png
"""
import argparse
import os

import pandas as pd

import model_zoo as mz

DATA = "data/model_dataset.csv"
RESULTS = "data/model_results.csv"
SENS_CSV = "data/nn_sensitivity.csv"
WINDOWS = mz.m07.WINDOWS
SENS_CFGS = {"NN (32)": dict(hidden_layer_sizes=(32,), alpha=1e-2),
             "NN (64-32)": dict(hidden_layer_sizes=(64, 32), alpha=1e-2),
             "NN (128-64-32)": dict(hidden_layer_sizes=(128, 64, 32), alpha=1e-2),
             "NN (64-32), alpha=1e-1": dict(hidden_layer_sizes=(64, 32), alpha=1e-1)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", help="same split date you used for 07_models.py")
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else df["date"].iloc[int(0.75 * len(df))]
    emb = pd.Timedelta(days=mz.EMBARGO_DAYS)
    train, test = df[df["date"] < cutoff - emb], df[df["date"] >= cutoff]
    print(f"Cutoff {cutoff.date()} | train {len(train)} | test {len(test)}")

    rows, sens = [], []
    for k in WINDOWS:
        print(f"{k}d ...")
        rows += mz.fit_eval(train, test, k, classic=False)
        if not args.skip_sensitivity:
            for r in mz.fit_eval(train, test, k, classic=False, nn_cfgs=SENS_CFGS):
                sens.append(r)
    nn = pd.DataFrame(rows)

    # ---- merge into the main results table ----
    if os.path.exists(RESULTS):
        res = pd.read_csv(RESULTS)
        res = res[~res["model"].str.startswith("Neural Net")]
        res = pd.concat([res, nn], ignore_index=True)
    else:
        print("WARNING: data/model_results.csv not found - run 07_models.py first for the comparison.")
        res = nn
    res["window"] = pd.Categorical(res["window"], [f"{k}d" for k in WINDOWS], ordered=True)
    res = res.sort_values(["task", "window", "model"]).reset_index(drop=True)
    res["window"] = res["window"].astype(str)
    res.to_csv(RESULTS, index=False)
    if sens:
        pd.DataFrame(sens).to_csv(SENS_CSV, index=False)

    show = [c for c in ["window", "model", "rmse", "r2_oos", "spearman", "dir_acc",
                        "accuracy", "auc"] if c in res.columns]
    for task in ("regression", "classification"):
        print(f"\n=== {task} ===")
        print(res[res["task"] == task][show].dropna(axis=1, how="all").round(4).to_string(index=False))

    mz.m07.plot_results(res, None)       # redraws figures/model_comparison.png (importance fig untouched)
    print("\nUpdated data/model_results.csv, figures/model_comparison.png"
          + ("" if args.skip_sensitivity else ", data/nn_sensitivity.csv"))