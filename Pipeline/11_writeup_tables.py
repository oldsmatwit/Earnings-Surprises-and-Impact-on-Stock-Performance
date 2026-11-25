"""
11_writeup_tables.py
Week 5: collect everything the write-up needs into ONE markdown file (data/results_summary.md)
so the numbers in the report come straight from the pipeline outputs.

Sections (each is skipped with a note if its input file does not exist yet):
  1. Data coverage and survivorship in the crisis samples (how many companies, which sectors /
     size tiers, vs the full sample)  -> supports the survivor-bias discussion for 2008
  2. Crisis vs normal: surprise-return relationship (08, Part A)
  3. Crisis model performance (08, Part B)
  4. ML vs NN on the main split, with bootstrap CIs and the paired NN-minus-RF difference (10)
  5. NN architecture sensitivity (09)
  6. Cutoff sensitivity (10)
  7. Breakdowns by sector and by Beat/Miss (10)
  + automatic "headline checks" (does a CI exclude chance? is the NN-RF difference distinguishable
    from 0? does the ranking of models flip across cutoffs?). These are prompts for the text,
    not conclusions: read the tables before writing the sentence.
"""
import os

import numpy as np
import pandas as pd

import importlib
s08 = importlib.import_module("08_stress_test")      # reuses CRISES / regime_of

OUT = "data/results_summary.md"
parts = []


def md(df, digits=3):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).to_numpy()]
    return "\n".join([head, sep] + body)


def section(title, body):
    parts.append(f"## {title}\n\n{body}\n")


def read(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


# ---------------------------------------------------------------- 1 coverage / survivorship
df = pd.read_csv("data/model_dataset.csv", parse_dates=["date", "react_date"])
df["regime"] = s08.regime_of(df["react_date"])
n_all = df["symbol"].nunique()
cov = (df.groupby("regime").agg(events=("symbol", "size"), companies=("symbol", "nunique"))
         .assign(share_of_companies=lambda t: t["companies"] / n_all).reset_index())
yr = df.groupby(df["date"].dt.year).agg(events=("symbol", "size"), companies=("symbol", "nunique")).reset_index()
yr.columns = ["year", "events", "companies"]
body = (f"Dataset: {len(df)} events, {n_all} companies, {df['date'].min().date()} to {df['date'].max().date()}.\n\n"
        + md(cov) + "\n\nEvents and companies per year:\n\n" + md(yr))
mix = []
for dim in ["sector", "cap_tier", "freq_tier", "age_tier"]:
    full = df.drop_duplicates("symbol")[dim].value_counts(normalize=True).rename("all_companies")
    parts_ = [full]
    for reg in ("GFC_2008", "COVID_2020"):
        g = df[df["regime"] == reg].drop_duplicates("symbol")[dim].value_counts(normalize=True).rename(reg)
        parts_.append(g)
    t = pd.concat(parts_, axis=1).fillna(0)
    t.insert(0, "dimension", dim)
    mix.append(t.rename_axis("level").reset_index())
body += ("\n\nComposition of the companies that have crisis events vs the full company set "
         "(shares of companies). Large gaps = the crisis sample is not representative; the sample is "
         "also limited to today's S&P 500 members (survivorship).\n\n" + md(pd.concat(mix)))
section("1. Data coverage and survivorship", body)

# ---------------------------------------------------------------- 2 / 3 stress test
d = read("data/stress_descriptives.csv")
if d is not None:
    cols = ["regime", "window", "n", "mean_car", "sd_car", "beat_minus_miss", "p_welch", "spearman_sue", "p_spearman"]
    section("2. Crisis vs normal (08, Part A)", md(d[cols], 4))
r = read("data/stress_results.csv")
if r is not None and len(r):
    cols = [c for c in ["regime", "mode", "window", "model", "n_train", "n_test", "r2_oos", "dir_acc", "accuracy", "auc"] if c in r]
    section("3. Crisis model performance (08, Part B)",
            md(r[~r["model"].str.startswith("Baseline")][cols].sort_values(["regime", "mode", "window", "model"]), 4))

# ---------------------------------------------------------------- 4 ML vs NN
c = read("data/tuned_comparison.csv")
if c is not None:
    cols = [x for x in ["window", "task", "model", "r2_oos", "spearman", "dir_acc", "accuracy", "auc", "ci_lo", "ci_hi"] if x in c]
    section("4. ML vs NN, main split (10); CI = bootstrap 95% on AUC / R2; DIFF rows = paired bootstrap",
            md(c[cols], 4))
s = read("data/nn_sensitivity.csv")
if s is not None:
    cols = [x for x in ["window", "task", "model", "r2_oos", "dir_acc", "accuracy", "auc"] if x in s]
    section("5. NN architecture sensitivity (09; reported only, not used to pick a winner)", md(s[cols], 4))
cs = read("data/cutoff_sensitivity.csv")
if cs is not None:
    d_ = cs[~cs["model"].str.startswith(("Baseline", "Rule"))]
    t1 = d_[d_["task"] == "classification"].pivot_table(index=["window", "cutoff"], columns="model", values="auc").reset_index()
    t2 = d_[d_["task"] == "regression"].pivot_table(index=["window", "cutoff"], columns="model", values="r2_oos").reset_index()
    section("6. Cutoff sensitivity (10); pre-specified settings, nested test sets",
            "AUC:\n\n" + md(t1, 4) + "\n\nOut-of-sample R2:\n\n" + md(t2, 4))
b = read("data/breakdowns.csv")
if b is not None:
    k = b[(b["window"] == "1d") & b["model"].isin(["Logistic Regression", "RF (tuned)", "Neural Net"])]
    auc = k[k["task"] == "classification"].pivot_table(index=["dimension", "level", "n", "share_positive"],
                                                       columns="model", values="auc").reset_index()
    section("7. Breakdowns, 1-day direction AUC (10); other windows in data/breakdowns.csv", md(auc, 3))

# ---------------------------------------------------------------- headline checks
chk = []
if c is not None:
    for w, g in c[c["task"] == "classification"].groupby("window"):
        for m in ("RF (tuned)", "Neural Net", "Logistic Regression"):
            x = g[g["model"] == m]
            if len(x) and x["ci_lo"].notna().all():
                chk.append(f"- {w} AUC, {m}: {x['auc'].iloc[0]:.3f} "
                           f"[{x['ci_lo'].iloc[0]:.3f}, {x['ci_hi'].iloc[0]:.3f}] -> "
                           + ("CI excludes 0.5" if x["ci_lo"].iloc[0] > 0.5 else "CI includes 0.5 (no demonstrated skill)"))
    for (w, t), g in c[c["model"].str.startswith("DIFF")].groupby(["window", "task"]):
        lo, hi = g["ci_lo"].iloc[0], g["ci_hi"].iloc[0]
        chk.append(f"- {w} {t}, NN minus tuned RF: [{lo:.3f}, {hi:.3f}] -> "
                   + ("distinguishable from 0" if lo > 0 or hi < 0 else "NOT distinguishable from 0"))
if cs is not None:
    for (w, t), g in cs[cs["task"] == "classification"].groupby(["window", "task"]):
        rk = g[~g["model"].str.startswith(("Baseline", "Rule"))].pivot_table(index="cutoff", columns="model", values="auc").idxmax(axis=1)
        chk.append(f"- {w} best-AUC model by cutoff: " + ", ".join(f"{i}: {v}" for i, v in rk.items())
                   + (" -> ranking changes with the cutoff" if rk.nunique() > 1 else " -> same winner at every cutoff"))
if chk:
    section("Headline checks (prompts for the text, not conclusions)", "\n".join(chk))

open(OUT, "w").write("# Results summary (auto-generated)\n\n" + "\n".join(parts))
print(f"Wrote {OUT} ({len(parts)} sections)")