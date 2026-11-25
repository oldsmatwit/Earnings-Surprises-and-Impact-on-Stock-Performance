"""
06_eda.py
Exploratory statistics and the core visualizations for the midterm.

Tables  -> data/eda_*.csv
Figures -> figures/*.png
(Winsorizing here is for DISPLAY only; modeling clips inside the train-only pipeline.)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA = "data/model_dataset.csv"
FIG_DIR = "figures"
WINDOWS = (1, 3, 5, 20)
CARS = [f"car_{k}d" for k in WINDOWS]
ORDER = ["Miss", "Meet", "Beat"]

os.makedirs(FIG_DIR, exist_ok=True)


def winsor(s: pd.Series, p: float = 1.0) -> pd.Series:
    lo, hi = s.quantile([p / 100, 1 - p / 100])
    return s.clip(lo, hi)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=150)
    plt.close(fig)


def summary_tables(df):
    cols = ["surprise", "surprise_pct", "sue"] + CARS
    df[cols].describe(percentiles=[.05, .25, .5, .75, .95]).T.round(4).to_csv("data/eda_summary_stats.csv")
    df["year"] = df["date"].dt.year
    pd.concat({
        "by_sector": df["sector"].value_counts(),
        "by_timing": df["report_timing"].value_counts(),
        "by_direction": df["direction"].value_counts(),
    }).rename("events").to_csv("data/eda_event_counts.csv")
    print(df[cols].describe().round(4).to_string())
    print("\nEvents by direction:\n", df["direction"].value_counts(normalize=True).round(3).to_string())


def significance_tests(df) -> pd.DataFrame:
    rows = []
    for k in WINDOWS:
        c = f"car_{k}d"
        d = df.dropna(subset=[c, "sue"])
        rho_sue, p_sue = stats.spearmanr(d["sue"], d[c])
        dp = d.dropna(subset=["surprise_pct"])
        rho_pct, p_pct = stats.spearmanr(dp["surprise_pct"], dp[c])
        beat, miss = d.loc[d["surprise"] > 0, c], d.loc[d["surprise"] < 0, c]
        t, p_t = stats.ttest_ind(beat, miss, equal_var=False)
        rows.append({"window": f"{k}d", "n": len(d),
                     "spearman_sue": rho_sue, "p_sue": p_sue,
                     "spearman_surprise_pct": rho_pct, "p_surprise_pct": p_pct,
                     "mean_car_beat": beat.mean(), "mean_car_miss": miss.mean(),
                     "beat_minus_miss": beat.mean() - miss.mean(), "welch_t": t, "p_welch": p_t})
    out = pd.DataFrame(rows)
    out.to_csv("data/eda_significance_tests.csv", index=False)
    print("\nSignificance tests\n", out.round(4).to_string(index=False))
    return out


def plot_distributions(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(winsor(df["surprise_pct"].dropna()) * 100, bins=60, color="#4C72B0")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set(title="EPS surprise (% of estimate, winsorized 1/99)", xlabel="%", ylabel="events")
    ax[1].hist(winsor(df["car_1d"].dropna()) * 100, bins=60, color="#55A868")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set(title="1-day market-adjusted return (winsorized 1/99)", xlabel="%", ylabel="events")
    save(fig, "fig1_distributions")


def plot_scatter(df):
    d = df.dropna(subset=["sue", "car_1d"])
    x, y = winsor(d["sue"]), winsor(d["car_1d"]) * 100
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(x, y, s=6, alpha=0.2, color="#4C72B0")
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + b, color="crimson", lw=2, label=f"OLS slope = {m:.3f} pp per SUE unit")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Standardized surprise (SUE) vs 1-day market-adjusted return",
           xlabel="SUE", ylabel="1-day CAR (%)")
    ax.legend()
    save(fig, "fig2_sue_vs_car1d")


def plot_deciles(df):
    d = df.dropna(subset=["sue"]).copy()
    d["bucket"] = pd.qcut(d["sue"], 10, labels=False, duplicates="drop") + 1
    t = d.groupby("bucket")[CARS].mean() * 100
    t.round(4).to_csv("data/eda_car_by_sue_decile.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, k in zip(CARS, WINDOWS):
        ax.plot(t.index, t[c], marker="o", label=f"{k}-day")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Mean market-adjusted return by SUE decile (1 = worst, 10 = best)",
           xlabel="SUE decile", ylabel="mean CAR (%)")
    ax.legend()
    save(fig, "fig3_car_by_sue_decile")


def plot_direction(df):
    g = df.groupby("direction")[CARS]
    means = (g.mean() * 100).reindex(ORDER)
    sem = (g.sem() * 100).reindex(ORDER)
    x, w = np.arange(len(WINDOWS)), 0.26
    colors = {"Miss": "#C44E52", "Meet": "#8C8C8C", "Beat": "#55A868"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for n, lbl in enumerate(ORDER):
        ax.bar(x + (n - 1) * w, means.loc[lbl].values, w, yerr=sem.loc[lbl].values,
               capsize=3, label=lbl, color=colors[lbl])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{k}-day" for k in WINDOWS])
    ax.set(title="Mean market-adjusted return: beats vs misses (±1 SE)", ylabel="mean CAR (%)")
    ax.legend()
    save(fig, "fig4_car_by_direction")


def plot_sector(df):
    d = df[df["direction"].isin(["Beat", "Miss"])]
    t = (d.groupby(["sector", "direction"])["car_1d"].mean().unstack() * 100).sort_values("Beat")
    t.round(4).to_csv("data/eda_car1d_by_sector.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    t[["Miss", "Beat"]].plot.barh(ax=ax, color=["#C44E52", "#55A868"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(title="Mean 1-day market-adjusted return by sector", xlabel="mean CAR (%)", ylabel="")
    save(fig, "fig5_car1d_by_sector")


def plot_events_per_year(df):
    c = df["date"].dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(c.index.astype(str), c.values, color="#4C72B0")
    ax.set(title="Earnings events per year in the dataset", ylabel="events")
    ax.tick_params(axis="x", rotation=60)
    save(fig, "fig6_events_per_year")


if __name__ == "__main__":
    df = pd.read_csv(DATA, parse_dates=["date"])
    df["direction"] = np.select([df["surprise"] > 0, df["surprise"] < 0], ["Beat", "Miss"], "Meet")

    summary_tables(df)
    significance_tests(df)
    plot_distributions(df)
    plot_scatter(df)
    plot_deciles(df)
    plot_direction(df)
    plot_sector(df)
    plot_events_per_year(df)
    print(f"\nFigures saved to {FIG_DIR}/, tables saved to data/eda_*.csv")