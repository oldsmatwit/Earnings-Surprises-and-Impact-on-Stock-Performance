"""
04_collect_prices.py
Download daily prices for the final companies plus the market benchmark.

Benchmark decision: ^GSPC (S&P 500 index) from yfinance rather than the FRED SP500
series. FRED only carries ~10 years of history, which would cap the usable event
window; ^GSPC goes back far enough to cover every earnings event we collected, and
it comes from the same source/format as the stock prices.
(Stock prices are dividend/split adjusted, ^GSPC is a price index; the difference is
negligible over 1-20 trading day windows.)

Run AFTER 03_surprise_features.py (it uses the earliest event date to pick the start).
Cached per symbol, so reruns only download what is missing. WEEK 5: a cached file is also
re-downloaded if it starts AFTER the company's first earnings event (happens after the 2008
back-fill in 02_earnings_collector.py extends the event history).
"""
import os
import time

import pandas as pd
import yfinance as yf

FINAL_COMPANIES_CSV = "data/final_companies.csv"
EVENTS_CSV = "data/earnings_events_features.csv"
PRICE_DIR = "data/prices"
BENCHMARK = "^GSPC"
BENCH_CSV = "data/benchmark_gspc.csv"
COVERAGE_CSV = "data/price_coverage.csv"
LOOKBACK_DAYS = 150          # calendar days before the first event (for 60-day features)

os.makedirs(PRICE_DIR, exist_ok=True)


def fetch_prices(symbol: str, start: str, retries: int = 3):
    for attempt in range(retries):
        try:
            h = yf.Ticker(symbol).history(start=start, auto_adjust=True)
            if h is None or h.empty:
                return None
            h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
            out = h[["Close", "Volume"]].copy()
            out.index.name = "date"
            return out[~out.index.duplicated(keep="last")]
        except Exception as exc:
            print(f"  {symbol}: attempt {attempt + 1} failed ({str(exc)[:80]})")
            time.sleep(2 * (attempt + 1))
    return None


if __name__ == "__main__":
    events = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
    start = (events["date"].min() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    symbols = sorted(events["symbol"].unique())
    print(f"Prices from {start} for {len(symbols)} companies + {BENCHMARK}")

    bench = fetch_prices(BENCHMARK, start)
    if bench is None:
        raise SystemExit("Could not download the benchmark.")
    bench[["Close"]].to_csv(BENCH_CSV)

    first_ev = events.groupby("symbol")["date"].min()

    def needs_fetch(sym):
        p = f"{PRICE_DIR}/{sym}.csv"
        if not os.path.exists(p):
            return True
        first_px = pd.read_csv(p, usecols=["date"], parse_dates=["date"])["date"].min()
        return pd.notna(first_px) and first_px > first_ev[sym]     # prices start after events

    todo = [s for s in symbols if needs_fetch(s)]
    print(f"{len(symbols) - len(todo)} cached, {len(todo)} to fetch")
    for i, sym in enumerate(todo, 1):
        df = fetch_prices(sym, start)
        if df is not None:
            df.to_csv(f"{PRICE_DIR}/{sym}.csv")
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")
        time.sleep(0.4)

    rows = []
    for sym in symbols:
        p = f"{PRICE_DIR}/{sym}.csv"
        if not os.path.exists(p):
            rows.append({"symbol": sym, "n_days": 0})
            continue
        d = pd.read_csv(p, parse_dates=["date"])
        first_event = events.loc[events["symbol"] == sym, "date"].min()
        rows.append({"symbol": sym, "n_days": len(d), "first_price": d["date"].min(),
                     "last_price": d["date"].max(), "first_event": first_event})
    cov = pd.DataFrame(rows)
    cov.to_csv(COVERAGE_CSV, index=False)
    print("Companies with no price file:", (cov["n_days"] == 0).sum())
    print("Median trading days per company:", cov["n_days"].median())