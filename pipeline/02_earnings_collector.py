"""
02_earnings_collector.py
Collect the FULL historical earnings data for the selected companies and finalize
the company list (WEEK 3 version).

Sources
  Primary : yfinance Ticker.get_earnings_dates()  -> no API key, no daily cap
  Fallback: Alpha Vantage EARNINGS endpoint        -> 25 calls/day, 1 call per company

Pipeline
  1. python 02_earnings_collector.py               # yfinance for every selected company
  2. python 02_earnings_collector.py --alphavantage # gap fill (max 25 calls per run, never
                                                   # re-spends a call on a company already tried)
  3. python 02_earnings_collector.py --finalize     # swap companies with too little data for
                                                   # reserves (same sector, same cell first)
  4. python 02_earnings_collector.py --backfill     # WEEK 5: pull OLD history (Alpha Vantage) for
                                                   # companies whose usable data starts too late
                                                   # to cover the 2008 crisis. 25 calls/day, never
                                                   # re-spends a call; rerun daily until done.
  (--test runs steps 1-2 on 20 sector-spread companies only)

Output columns (all sources): symbol, date, epsActual, epsEstimated, report_timing, source

report_timing labels
  bmo     = before market open   (time stamp before 9:30 ET)
  amc     = after market close   (time stamp at/after 16:00 ET)
  dmh     = during market hours
  unknown = no usable time stamp (midnight / missing)
Timing rule used downstream (see 05_build_dataset.py):
  bmo, dmh -> reaction day = the announcement date itself
  amc      -> reaction day = the NEXT trading day
  unknown  -> excluded from the main sample
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

SELECTED_CSV = "data/selected_companies.csv"
YF_DIR = "data/earnings_yf"
AV_DIR = "data/earnings_av"
FINAL_DIR = "data/earnings_final"            # merged per-company files used by 03
FINAL_COMPANIES_CSV = "data/final_companies.csv"
SUMMARY_CSV = "data/earnings_availability_summary.csv"

YF_LIMIT = 100              # rows requested; Yahoo may return fewer
MIN_USABLE_EVENTS = 20      # ~5 years of quarters with both actual and estimate
KNOWN_TIMING = ["bmo", "amc", "dmh"]

# Optional: paste your Alpha Vantage key here (falls back to the AV_API_KEY env var)
AV_API_KEY = "PASTE_ALPHA_VANTAGE_KEY_HERE"
if AV_API_KEY == "PASTE_ALPHA_VANTAGE_KEY_HERE":
    AV_API_KEY = os.environ.get("AV_API_KEY")
AV_DAILY_BUDGET = 25

for _d in (YF_DIR, AV_DIR, FINAL_DIR):
    os.makedirs(_d, exist_ok=True)


# ------------------------------ timing labels ---------------------------------
def _timing_from_time(ts: pd.Timestamp) -> str:
    if pd.isna(ts) or (ts.hour == 0 and ts.minute == 0):   # midnight = no time supplied
        return "unknown"
    minutes = ts.hour * 60 + ts.minute
    if minutes < 9 * 60 + 30:
        return "bmo"
    if minutes >= 16 * 60:
        return "amc"
    return "dmh"


# ------------------------------ yfinance --------------------------------------
def fetch_yfinance(symbol: str, retries: int = 3):
    for attempt in range(retries):
        try:
            raw = yf.Ticker(symbol).get_earnings_dates(limit=YF_LIMIT)
            if raw is None or raw.empty:
                return None
            raw = raw.reset_index()
            date_col = raw.columns[0]                       # "Earnings Date"
            ts = pd.to_datetime(raw[date_col])              # tz-aware, America/New_York
            out = pd.DataFrame({
                "symbol": symbol,
                "date": ts.dt.tz_localize(None).dt.normalize(),
                "epsActual": pd.to_numeric(raw.get("Reported EPS"), errors="coerce"),
                "epsEstimated": pd.to_numeric(raw.get("EPS Estimate"), errors="coerce"),
                "report_timing": ts.apply(_timing_from_time),
                "source": "yfinance",
            })
            return out.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            print(f"  {symbol}: attempt {attempt + 1} failed ({str(exc)[:80]})")
            time.sleep(2 * (attempt + 1))
    return None


# ------------------------------ Alpha Vantage ---------------------------------
def fetch_alphavantage(symbol: str):
    """Returns a DataFrame, None (no data for this symbol) or "LIMIT" (daily cap hit)."""
    resp = requests.get("https://www.alphavantage.co/query",
                        params={"function": "EARNINGS", "symbol": symbol, "apikey": AV_API_KEY},
                        timeout=30)
    data = resp.json()
    if "Note" in data or "Information" in data:
        print(f"  {symbol}: {str(data)[:120]}")              # rate-limit message
        return "LIMIT"
    if "quarterlyEarnings" not in data:
        return None
    q = pd.DataFrame(data["quarterlyEarnings"])
    timing = {"pre-market": "bmo", "post-market": "amc", "during-market": "dmh"}
    out = pd.DataFrame({
        "symbol": symbol,
        "date": pd.to_datetime(q["reportedDate"], errors="coerce"),
        "epsActual": pd.to_numeric(q["reportedEPS"], errors="coerce"),
        "epsEstimated": pd.to_numeric(q["estimatedEPS"], errors="coerce"),
        "report_timing": (q["reportTime"].map(timing).fillna("unknown")
                          if "reportTime" in q else "unknown"),
        "source": "alphavantage",
    })
    return out.sort_values("date").reset_index(drop=True)


# ------------------------------ merging ---------------------------------------
def merge_sources(sym: str):
    """Pick the source with more usable events as the base; borrow known timing labels
    from the other source where the dates match. Returns None if nothing is cached."""
    srcs = {}
    for name, folder in (("yfinance", YF_DIR), ("alphavantage", AV_DIR)):
        path = f"{folder}/{sym}.csv"
        if os.path.exists(path):
            srcs[name] = pd.read_csv(path, parse_dates=["date"])
    if not srcs:
        return None
    usable = {k: len(v.dropna(subset=["epsActual", "epsEstimated"])) for k, v in srcs.items()}
    base_name = max(usable, key=usable.get)                 # ties -> yfinance
    base = srcs[base_name].copy()
    for name, other in srcs.items():
        if name == base_name or other.empty:
            continue
        known = other[other["report_timing"].isin(KNOWN_TIMING)]
        known = known.drop_duplicates("date").set_index("date")["report_timing"]
        mask = ~base["report_timing"].isin(KNOWN_TIMING)
        filled = base.loc[mask, "date"].map(known)
        base.loc[mask, "report_timing"] = filled.fillna(base.loc[mask, "report_timing"])
        # WEEK 5: also EXTEND the history. Events the base source lacks (no base date within
        # +-5 days) are appended, so e.g. Alpha Vantage can supply pre-2010 quarters on top of
        # a yfinance base. Rows keep their own `source` label.
        have = pd.DatetimeIndex(base["date"].dropna())
        ext = other.dropna(subset=["date", "epsActual", "epsEstimated"])
        if len(have) and len(ext):
            ext = ext[ext["date"].map(lambda d: abs((have - d).days).min() > 5)]
        if len(ext):
            base = pd.concat([base, ext], ignore_index=True)
    return base.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _summarize_one(sym: str) -> dict:
    merged = merge_sources(sym)
    if merged is None or merged.empty:
        return {"symbol": sym, "downloaded": merged is not None, "n_events": 0, "n_usable": 0,
                "earliest": pd.NaT, "latest": pd.NaT, "pct_timing_known": 0.0, "source": None}
    merged.to_csv(f"{FINAL_DIR}/{sym}.csv", index=False)
    usable = merged.dropna(subset=["epsActual", "epsEstimated"])
    return {
        "symbol": sym, "downloaded": True, "n_events": len(merged), "n_usable": len(usable),
        "earliest": usable["date"].min() if len(usable) else pd.NaT,
        "latest": usable["date"].max() if len(usable) else pd.NaT,
        "pct_timing_known": usable["report_timing"].isin(KNOWN_TIMING).mean() if len(usable) else 0.0,
        "source": "+".join(sorted(merged["source"].dropna().unique())),
    }


def summarize_availability(symbols) -> pd.DataFrame:
    out = pd.DataFrame([_summarize_one(s) for s in symbols])
    out["meets_min"] = out["n_usable"] >= MIN_USABLE_EVENTS
    out.to_csv(SUMMARY_CSV, index=False)
    return out


# ------------------------------ collection ------------------------------------
def collect_yfinance(symbols, pause=1.0):
    todo = [s for s in symbols if not os.path.exists(f"{YF_DIR}/{s}.csv")]
    print(f"yfinance: {len(symbols) - len(todo)} cached, {len(todo)} to fetch")
    for i, sym in enumerate(todo, 1):
        df = fetch_yfinance(sym)
        if df is not None:
            df.to_csv(f"{YF_DIR}/{sym}.csv", index=False)
        if i % 20 == 0:
            print(f"  {i}/{len(todo)}")
        time.sleep(pause)


def collect_alphavantage(symbols):
    """Spend the daily calls on companies that are too thin or have unknown timing.
    Companies already tried (a file exists in AV_DIR) are skipped so no call is wasted."""
    if not AV_API_KEY:
        raise SystemExit("Set AV_API_KEY (in this file or as an environment variable).")
    s = summarize_availability(symbols)
    need = s[(~s["meets_min"]) | (s["pct_timing_known"] < 0.5)]
    need = need[~need["symbol"].map(lambda x: os.path.exists(f"{AV_DIR}/{x}.csv"))]
    need = need.sort_values(["meets_min", "pct_timing_known"])      # below-minimum first
    print(f"{len(need)} companies need a fallback; using up to {AV_DAILY_BUDGET} calls")
    for sym in need["symbol"].tolist()[:AV_DAILY_BUDGET]:
        df = fetch_alphavantage(sym)
        if isinstance(df, str):          # daily limit reached
            break
        if df is None:                   # no data: write an empty marker so we do not retry
            df = pd.DataFrame(columns=["symbol", "date", "epsActual", "epsEstimated",
                                       "report_timing", "source"])
        df.to_csv(f"{AV_DIR}/{sym}.csv", index=False)
        time.sleep(12)                   # free tier also limits calls per minute


# ------------------------------ 2008 back-fill (week 5) -------------------------
def check_source_agreement(symbols):
    """Where yfinance and Alpha Vantage overlap, do they agree? Splicing two sources is only
    safe if EPS actuals/estimates are on the same basis (GAAP vs adjusted) and the sign of the
    surprise matches. Writes data/source_agreement.csv."""
    rows = []
    for sym in symbols:
        py, pa = f"{YF_DIR}/{sym}.csv", f"{AV_DIR}/{sym}.csv"
        if not (os.path.exists(py) and os.path.exists(pa)):
            continue
        y = pd.read_csv(py, parse_dates=["date"]).dropna(subset=["epsActual", "epsEstimated"])
        a = pd.read_csv(pa, parse_dates=["date"]).dropna(subset=["date", "epsActual", "epsEstimated"])
        if y.empty or a.empty:
            continue
        m = pd.merge_asof(y.sort_values("date"), a.sort_values("date"), on="date",
                          tolerance=pd.Timedelta(days=3), direction="nearest",
                          suffixes=("_yf", "_av")).dropna(subset=["epsActual_av"])
        if m.empty:
            continue
        sy = np.sign(m["epsActual_yf"] - m["epsEstimated_yf"])
        sa = np.sign(m["epsActual_av"] - m["epsEstimated_av"])
        rows.append({"symbol": sym, "n_overlap": len(m),
                     "actual_match": (abs(m["epsActual_yf"] - m["epsActual_av"]) <= 0.02).mean(),
                     "estimate_match": (abs(m["epsEstimated_yf"] - m["epsEstimated_av"]) <= 0.02).mean(),
                     "surprise_sign_match": (sy == sa).mean()})
    out = pd.DataFrame(rows)
    out.to_csv("data/source_agreement.csv", index=False)
    if len(out):
        print("Source agreement on overlapping events (median across companies): "
              + ", ".join(f"{c}={out[c].median():.2f}" for c in
                          ["actual_match", "estimate_match", "surprise_sign_match"]))
    return out


def collect_backfill(symbols, want_from="2005-01-01"):
    """Spend the daily Alpha Vantage calls on companies that (a) were listed by `want_from`,
    (b) have no usable earnings before want_from + 1 year (SUE needs 4 prior quarters), and
    (c) have not been tried yet. Candidates are ordered round-robin ACROSS SECTORS so a partly
    finished back-fill is still sector-balanced."""
    if not AV_API_KEY:
        raise SystemExit("Set AV_API_KEY (in this file or as an environment variable).")
    want = pd.Timestamp(want_from)
    s = summarize_availability(symbols)
    meta = pd.read_csv(SELECTED_CSV)
    cols = [c for c in ["symbol", "sector", "first_trade_date"] if c in meta.columns]
    c = s.merge(meta[cols], on="symbol", how="left")
    late = c["earliest"].isna() | (c["earliest"] > want + pd.Timedelta(days=365))
    if "first_trade_date" in c:
        listed = pd.to_datetime(c["first_trade_date"], errors="coerce")
        late &= listed.isna() | (listed <= want)
    c = c[late & ~c["symbol"].map(lambda x: os.path.exists(f"{AV_DIR}/{x}.csv"))].copy()
    c["rr"] = c.groupby("sector").cumcount()
    c = c.sort_values(["rr", "sector"])
    print(f"{len(c)} companies still need old history; using up to {AV_DAILY_BUDGET} calls today "
          f"(~{-(-max(len(c) - AV_DAILY_BUDGET, 0) // AV_DAILY_BUDGET)} more day(s) after this)")
    for sym in c["symbol"].tolist()[:AV_DAILY_BUDGET]:
        df = fetch_alphavantage(sym)
        if isinstance(df, str):
            break
        if df is None:
            df = pd.DataFrame(columns=["symbol", "date", "epsActual", "epsEstimated",
                                       "report_timing", "source"])
        df.to_csv(f"{AV_DIR}/{sym}.csv", index=False)
        time.sleep(12)
    s2 = summarize_availability(symbols)                    # rewrites data/earnings_final/*.csv
    crisis = pd.Timestamp("2008-09-01")
    print(f"Companies with usable earnings before {crisis.date()}: "
          f"{int((s2['earliest'] < crisis).sum())} of {len(s2)} "
          f"(was {int((s['earliest'] < crisis).sum())})")
    print("Earliest usable date (median):", s2["earliest"].median())
    check_source_agreement(symbols)


# ------------------------------ finalize list ---------------------------------
def finalize_company_list():
    """Replace companies with too little data using the ranked reserve list.
    Replacement preference: same sector AND same cap x freq x age cell, then reserve rank."""
    sel = pd.read_csv(SELECTED_CSV)
    chosen = sel[sel["selected"]].copy()
    summary = summarize_availability(chosen["symbol"].tolist())
    failing = summary.loc[~summary["meets_min"], "symbol"].tolist()
    print(f"{len(failing)} of {len(chosen)} selected companies are below {MIN_USABLE_EVENTS} usable events")

    reserves = sel[(~sel["selected"]) & sel["reserve_rank"].notna()]
    tried, replacements = set(), []
    unreplaced = []
    for sym in failing:
        row = sel[sel["symbol"] == sym].iloc[0]
        pool = reserves[(reserves["sector"] == row["sector"]) & ~reserves["symbol"].isin(tried)].copy()
        pool["same_cell"] = pool["cell"] == row["cell"]
        pool = pool.sort_values(["same_cell", "reserve_rank"], ascending=[False, True])
        found = None
        for _, cand in pool.iterrows():
            tried.add(cand["symbol"])
            if not os.path.exists(f"{YF_DIR}/{cand['symbol']}.csv"):
                df = fetch_yfinance(cand["symbol"])
                if df is not None:
                    df.to_csv(f"{YF_DIR}/{cand['symbol']}.csv", index=False)
                time.sleep(1)
            if _summarize_one(cand["symbol"])["n_usable"] >= MIN_USABLE_EVENTS:
                found = cand
                break
        if found is None:
            unreplaced.append(sym)
            print(f"  WARNING: no usable reserve found for {sym} ({row['sector']})")
        else:
            replacements.append((found, sym))
            print(f"  {sym} -> {found['symbol']}")

    final = chosen[~chosen["symbol"].isin(failing)].copy()
    final["status"], final["replaces"] = "original", ""
    if replacements:
        rep = pd.DataFrame([c for c, _ in replacements])
        rep["selected"] = True
        rep["status"] = "replacement"
        rep["replaces"] = [s for _, s in replacements]
        final = pd.concat([final, rep], ignore_index=True)
    final.to_csv(FINAL_COMPANIES_CSV, index=False)

    summary = summarize_availability(final["symbol"].tolist())
    print(f"\nFinal list: {len(final)} companies "
          f"({len(replacements)} replacements, {len(unreplaced)} not replaced)")
    print(final["sector"].value_counts().to_string())
    print("Median usable events per company:", summary["n_usable"].median())
    print("Median % of events with known timing:", round(summary["pct_timing_known"].median(), 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="only 20 companies")
    parser.add_argument("--alphavantage", action="store_true", help="fill gaps via Alpha Vantage")
    parser.add_argument("--backfill", action="store_true",
                        help="Alpha Vantage: add old (pre-2010) history for the 2008 stress test")
    parser.add_argument("--want-from", default="2005-01-01",
                        help="--backfill: want usable history starting around this date")
    parser.add_argument("--finalize", action="store_true", help="swap in reserves, write final list")
    args = parser.parse_args()

    if args.finalize:
        finalize_company_list()
        raise SystemExit(0)

    sel = pd.read_csv(SELECTED_CSV)
    symbols = sel.loc[sel["selected"], "symbol"].tolist()
    if args.test:
        symbols = sel[sel["selected"]].groupby("sector").head(2)["symbol"].tolist()[:20]

    if args.backfill:
        sel_final = pd.read_csv(FINAL_COMPANIES_CSV) if os.path.exists(FINAL_COMPANIES_CSV) else sel[sel["selected"]]
        collect_backfill(sel_final["symbol"].tolist(), args.want_from)
        raise SystemExit(0)
    if args.alphavantage:
        collect_alphavantage(symbols)
    else:
        collect_yfinance(symbols)

    summary = summarize_availability(symbols)
    print(summary[["n_usable", "pct_timing_known"]].describe().round(2))
    print("Earliest usable date (median):", summary["earliest"].median())
    print("Companies below the minimum:", (~summary["meets_min"]).sum(), "of", len(summary))