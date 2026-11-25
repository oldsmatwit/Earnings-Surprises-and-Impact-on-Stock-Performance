"""
03_surprise_features.py
Earnings-surprise features (no prices needed). WEEK 3 version: reads the merged,
finalized earnings files and only keeps companies on the FINAL company list.

FEATURE PLAN
  Surprise features (this file)
    surprise            actual EPS - estimated EPS
    surprise_pct        surprise / |estimate|  (NaN when |estimate| < 0.05 to avoid blow-ups)
    sue                 surprise / std of the previous 8 surprises (standardized, no look-ahead)
    surprise_dir        sign of the surprise (+1 / 0 / -1)
    prior_surprise_pct  previous quarter's surprise_pct
    beats_last_4        number of beats in the previous 4 quarters
  Price/market features (05_build_dataset.py)
    ret_prior_20d, ret_prior_60d, vol_prior_20d, volume_ratio_5d_vs_60d
  Company / calendar features
    sector, cap_tier, freq_tier, age_tier, quarter, report_timing
  Targets (05_build_dataset.py)
    car_1d, car_3d, car_5d, car_20d   -> regression
    positive_car_<window>d            -> logistic regression / random forest classifier

NOTE: outlier clipping (winsorizing) uses cutoffs computed on the TRAINING period
only, so it is applied inside the model pipeline (07_models.py), not here.
"""
import glob
import os

import numpy as np
import pandas as pd

FINAL_DIR = "data/earnings_final"
FINAL_COMPANIES_CSV = "data/final_companies.csv"
OUT_CSV = "data/earnings_events_features.csv"


def load_all_earnings(cache_dir=FINAL_DIR, companies_csv=FINAL_COMPANIES_CSV) -> pd.DataFrame:
    keep = set(pd.read_csv(companies_csv)["symbol"])
    files = [f for f in glob.glob(f"{cache_dir}/*.csv")
             if os.path.splitext(os.path.basename(f))[0] in keep]
    frames = [pd.read_csv(f) for f in files]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_surprise_features(earnings: pd.DataFrame) -> pd.DataFrame:
    df = earnings.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = (df.dropna(subset=["epsActual", "epsEstimated"])        # drops future / incomplete rows
            .drop_duplicates(["symbol", "date"], keep="last")
            .sort_values(["symbol", "date"])
            .reset_index(drop=True))

    est = df["epsEstimated"]
    df["surprise"] = df["epsActual"] - est
    df["surprise_pct"] = df["surprise"] / est.abs().where(est.abs() >= 0.05)
    df["surprise_dir"] = np.sign(df["surprise"])

    g = df.groupby("symbol")
    # rolling stats use shift(1) so each event only sees PAST quarters
    df["surprise_std8"] = g["surprise"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=4).std())
    df["sue"] = df["surprise"] / df["surprise_std8"].replace(0, np.nan)
    df["prior_surprise_pct"] = g["surprise_pct"].shift(1)
    df["beats_last_4"] = g["surprise_dir"].transform(
        lambda s: (s.shift(1) > 0).rolling(4, min_periods=2).sum())
    df["quarter"] = df["date"].dt.quarter
    return df


def attach_company_features(events: pd.DataFrame, companies_csv=FINAL_COMPANIES_CSV):
    keep = ["symbol", "sector", "cap_tier", "freq_tier", "age_tier"]
    return events.merge(pd.read_csv(companies_csv)[keep], on="symbol", how="left")


if __name__ == "__main__":
    raw = load_all_earnings()
    events = attach_company_features(build_surprise_features(raw))
    events.to_csv(OUT_CSV, index=False)
    print(events.shape, "events from", events["symbol"].nunique(), "companies")
    print("Report timing mix:\n", events["report_timing"].value_counts(normalize=True).round(3))
    print(events[["surprise", "surprise_pct", "sue"]].describe())