"""
05_build_dataset.py
Align quarterly earnings events with daily prices, clean, and build the modeling table.
One row = one earnings announcement.

TIMING RULE (announcement-day return definition)
  reaction day (day 0) = first trading day on which the market can react
     bmo / dmh -> the announcement date itself (first trading day >= date)
     amc       -> the next trading day            (first trading day >  date)
     unknown   -> event excluded from the main sample
  base price   = close of the trading day BEFORE day 0
  car_kd       = (stock close[day k-1] / base - 1) - (S&P 500 close[day k-1] / base - 1)
                 i.e. market-adjusted cumulative return over k trading days starting at day 0
                 k = 1, 3, 5, 20

PRICE FEATURES use only data up to the day before day 0 (no look-ahead):
  ret_prior_20d / ret_prior_60d, vol_prior_20d (std of daily returns),
  volume_ratio_5d_vs_60d (mean volume last 5 days / mean volume last 60 days)

OVERLAP RULE: if the next earnings reaction day is fewer than 20 trading days away,
the event is flagged (overlap_20d) and its car_20d is set to missing, so a 20-day
window never contains a second announcement for the same company.

Outputs: data/model_dataset.csv, data/cleaning_log.csv, data/timing_validation.csv
"""
from collections import Counter
import os

import numpy as np
import pandas as pd

EVENTS_CSV = "data/earnings_events_features.csv"
PRICE_DIR = "data/prices"
BENCH_CSV = "data/benchmark_gspc.csv"
OUT_CSV = "data/model_dataset.csv"
LOG_CSV = "data/cleaning_log.csv"
TIMING_CSV = "data/timing_validation.csv"

WINDOWS = (1, 3, 5, 20)
KNOWN_TIMING = ["bmo", "amc", "dmh"]
MAX_GAP_DAYS = 5          # reaction day must be within 5 calendar days of the announcement
MIN_PRIOR_DAYS = 61       # trading days of history needed before day 0 (60-day features)


def load_benchmark() -> pd.Series:
    b = pd.read_csv(BENCH_CSV, parse_dates=["date"]).set_index("date")["Close"]
    return b[~b.index.duplicated()].rename("bench")


def load_prices(sym):
    path = f"{PRICE_DIR}/{sym}.csv"
    return pd.read_csv(path, parse_dates=["date"]).set_index("date") if os.path.exists(path) else None


def build_event_table(events: pd.DataFrame, bench: pd.Series):
    rows, drops = [], Counter()
    for sym, ev in events.sort_values(["symbol", "date"]).groupby("symbol"):
        px = load_prices(sym)
        if px is None:
            drops["no price file"] += len(ev)
            continue
        d = px.join(bench, how="inner").dropna(subset=["Close", "bench"])
        d = d[~d.index.duplicated()]
        idx = d.index.values
        close, b, vol = (d["Close"].to_numpy(float), d["bench"].to_numpy(float),
                         d["Volume"].to_numpy(float))
        r = np.full(len(d), np.nan)
        r[1:] = close[1:] / close[:-1] - 1
        br = np.full(len(d), np.nan)
        br[1:] = b[1:] / b[:-1] - 1
        ar = r - br                                           # daily market-adjusted return

        sym_rows = []
        for e in ev.to_dict("records"):
            ed = pd.Timestamp(e["date"])
            side = "right" if e["report_timing"] == "amc" else "left"
            i = int(np.searchsorted(idx, ed.to_datetime64(), side=side))   # day-0 index
            if i >= len(idx):
                drops["no prices after announcement"] += 1
                continue
            if (pd.Timestamp(idx[i]) - ed).days > MAX_GAP_DAYS:
                drops["price gap around announcement"] += 1
                continue
            if i < MIN_PRIOR_DAYS:
                drops["less than 60 days of prior history"] += 1
                continue
            j = int(np.searchsorted(idx, ed.to_datetime64(), side="left"))  # calendar event day
            base = i - 1
            row = dict(e)
            row["react_date"] = pd.Timestamp(idx[i])
            row["i_react"] = i
            # --- price features (information up to the close before day 0) ---
            row["ret_prior_20d"] = close[base] / close[base - 20] - 1
            row["ret_prior_60d"] = close[base] / close[base - 60] - 1
            row["vol_prior_20d"] = np.nanstd(r[i - 20:i], ddof=1)
            v60 = np.nanmean(vol[i - 60:i])
            row["volume_ratio_5d_vs_60d"] = np.nanmean(vol[i - 5:i]) / v60 if v60 > 0 else np.nan
            # --- targets: market-adjusted cumulative returns ---
            for k in WINDOWS:
                end = i + k - 1
                row[f"car_{k}d"] = ((close[end] / close[base] - 1) - (b[end] / b[base] - 1)
                                    if end < len(idx) else np.nan)
            # --- diagnostics for validating the timing labels ---
            row["abs_ar_event_day"] = abs(ar[j])
            row["abs_ar_next_day"] = abs(ar[j + 1]) if j + 1 < len(idx) else np.nan
            sym_rows.append(row)

        if sym_rows:
            ii = np.array([x["i_react"] for x in sym_rows])
            gap = np.r_[np.diff(ii), 10_000]
            for x, g in zip(sym_rows, gap):
                x["trading_days_to_next_event"] = int(g)
                x["overlap_20d"] = bool(g < 20)
                if g < 20:
                    x["car_20d"] = np.nan
            rows += sym_rows
    return pd.DataFrame(rows), drops


def timing_validation(df: pd.DataFrame) -> pd.DataFrame:
    """If the labels are right, 'amc' events should move MORE on the next day than on the
    announcement date, and 'bmo' events the opposite."""
    out = []
    for t, d in df.groupby("report_timing"):
        d = d.dropna(subset=["abs_ar_event_day", "abs_ar_next_day"])
        out.append({"timing": t, "n": len(d),
                    "mean_abs_ar_event_day": d["abs_ar_event_day"].mean(),
                    "mean_abs_ar_next_day": d["abs_ar_next_day"].mean(),
                    "share_next_day_larger": (d["abs_ar_next_day"] > d["abs_ar_event_day"]).mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ev = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
    log = [("Events with EPS actual and estimate", len(ev))]

    ev = ev.dropna(subset=["sue"])
    log.append(("SUE defined (>= 4 prior quarters)", len(ev)))

    n_unknown = int((~ev["report_timing"].isin(KNOWN_TIMING)).sum())
    ev = ev[ev["report_timing"].isin(KNOWN_TIMING)]
    log.append((f"Announcement timing known ({n_unknown} unknown removed)", len(ev)))

    table, drops = build_event_table(ev, load_benchmark())
    for reason, n in drops.items():
        print(f"  dropped {n:5d}  {reason}")
    log.append(("Matched to prices (history, gaps, trailing data OK)", len(table)))

    for k in WINDOWS:
        table[f"positive_car_{k}d"] = (table[f"car_{k}d"] > 0).astype(float).where(table[f"car_{k}d"].notna())
    n_overlap = int(table["overlap_20d"].sum())
    log.append((f"Rows with usable car_20d ({n_overlap} overlap-flagged)", int(table["car_20d"].notna().sum())))

    table = table.drop(columns=["i_react"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    table.to_csv(OUT_CSV, index=False)
    pd.DataFrame(log, columns=["step", "rows"]).to_csv(LOG_CSV, index=False)

    print("\nCleaning log")
    for step, n in log:
        print(f"  {n:7d}  {step}")
    print(f"\nFinal table: {table.shape[0]} events, {table['symbol'].nunique()} companies, "
          f"{table['date'].min().date()} to {table['date'].max().date()}")
    print("Share positive CAR:", {f"{k}d": round(table[f'positive_car_{k}d'].mean(), 3) for k in WINDOWS})

    tv = timing_validation(table)
    tv.to_csv(TIMING_CSV, index=False)
    print("\nTiming validation (amc should show next-day larger, bmo event-day larger)")
    print(tv.round(4).to_string(index=False))

    spot = ["AAPL", "MSFT", "NVDA", "JPM", "WMT", "JNJ"]
    mode = (table[table["symbol"].isin(spot)].groupby("symbol")["report_timing"]
            .agg(lambda s: s.mode().iat[0]))
    print("\nSpot check (expect AAPL/MSFT/NVDA amc; JPM/WMT/JNJ bmo):\n", mode.to_string())