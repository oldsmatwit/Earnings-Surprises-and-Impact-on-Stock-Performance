"""
01_select_companies.py
Build the company list for the earnings-surprise project.

Steps
1. Get the current S&P 500 constituents (symbol, sector, sub-industry).
2. Pull free Yahoo Finance metadata for every constituent (cached, resumable).
3. Derive stratification variables:
     - industry        -> GICS sector
     - capital size    -> market-cap tertile (Small / Mid / Large within the S&P 500)
     - trading freq.   -> share turnover tertile (avg daily volume / shares outstanding)
     - company age     -> years since first trade (<10, 10-25, 25+)
4. Select ~250 companies: sector quotas proportional to the index, then a
   round-robin draw across (cap x turnover x age) cells inside each sector so
   every combination is represented.
5. Save the selected list, reserves (replacements if a company lacks data),
   and balance tables for the report.
"""
import os
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------- configuration ---------------------------------
DATA_DIR = "data"
UNIVERSE_CSV = f"{DATA_DIR}/sp500_universe.csv"
METADATA_CSV = f"{DATA_DIR}/yf_metadata.csv"
SELECTED_CSV = f"{DATA_DIR}/selected_companies.csv"
BALANCE_CSV = f"{DATA_DIR}/selection_balance_summary.csv"

TARGET_N = 250          # companies to include
MIN_YEARS_LISTED = 6    # need several years of quarterly events
RANDOM_STATE = 42       # reproducible selection
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

os.makedirs(DATA_DIR, exist_ok=True)


# ------------------------- 1. S&P 500 universe --------------------------------
def get_sp500_universe() -> pd.DataFrame:
    resp = requests.get(
        WIKI_URL,
        headers={"User-Agent": "Mozilla/5.0 (student research project)"},
        timeout=30,
    )
    resp.raise_for_status()
    df = pd.read_html(StringIO(resp.text))[0]
    df = df.rename(columns={
        "Symbol": "symbol",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "Date added": "date_added",
    })
    # Yahoo / FMP use "BRK-B" rather than "BRK.B"
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    return df[["symbol", "name", "sector", "sub_industry", "date_added"]]


# ------------------------- 2. Yahoo Finance metadata --------------------------
def fetch_one(symbol: str) -> dict:
    t = yf.Ticker(symbol)
    info = t.info or {}

    first = info.get("firstTradeDateMilliseconds") or info.get("firstTradeDateEpochUtc")
    if first:
        first_trade = pd.to_datetime(first, unit="ms" if first > 1e11 else "s")
    else:  # fallback: lightweight monthly history
        h = t.history(period="max", interval="1mo")
        first_trade = pd.Timestamp(h.index[0]).tz_localize(None) if len(h) else pd.NaT

    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    return {
        "symbol": symbol,
        "market_cap": info.get("marketCap"),
        "avg_volume": info.get("averageVolume"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "price": price,
        "yf_sector": info.get("sector"),
        "yf_industry": info.get("industry"),
        "first_trade_date": first_trade,
    }


def get_metadata(symbols, save_every=25, pause=0.5) -> pd.DataFrame:
    """Download metadata for all symbols; cached so reruns only fetch what is missing."""
    done = pd.read_csv(METADATA_CSV) if os.path.exists(METADATA_CSV) else pd.DataFrame()
    have = set(done["symbol"]) if len(done) else set()
    rows = done.to_dict("records")

    todo = [s for s in symbols if s not in have]
    print(f"Metadata: {len(have)} cached, {len(todo)} to fetch")
    for i, sym in enumerate(todo, 1):
        try:
            rows.append(fetch_one(sym))
        except Exception as exc:  # keep going; record the failure
            print(f"  {sym}: failed ({exc})")
            rows.append({"symbol": sym})
        if i % save_every == 0:
            pd.DataFrame(rows).to_csv(METADATA_CSV, index=False)
            print(f"  fetched {i}/{len(todo)}")
        time.sleep(pause)

    meta = pd.DataFrame(rows).drop_duplicates("symbol", keep="last")
    meta.to_csv(METADATA_CSV, index=False)
    return meta


# ------------------------- 3. stratification variables ------------------------
def add_strata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["first_trade_date"] = pd.to_datetime(df["first_trade_date"], errors="coerce")
    df["years_listed"] = (pd.Timestamp.today() - df["first_trade_date"]).dt.days / 365.25
    df["turnover"] = df["avg_volume"] / df["shares_outstanding"]          # trading frequency
    df["dollar_volume"] = df["avg_volume"] * df["price"]                  # liquidity ($)

    ok = df["market_cap"].notna() & df["turnover"].notna() & df["years_listed"].notna()
    df = df[ok].copy()

    df["cap_tier"] = pd.qcut(df["market_cap"], 3, labels=["Small", "Mid", "Large"])
    df["freq_tier"] = pd.qcut(df["turnover"], 3, labels=["Low", "Medium", "High"])
    df["age_tier"] = pd.cut(df["years_listed"], [0, 10, 25, np.inf],
                            labels=["<10 yrs", "10-25 yrs", "25+ yrs"])
    df["cell"] = (df["cap_tier"].astype(str) + "|" + df["freq_tier"].astype(str)
                  + "|" + df["age_tier"].astype(str))
    return df


# ------------------------- 4. selection ---------------------------------------
def allocate_quotas(counts: pd.Series, n: int) -> pd.Series:
    """Proportional allocation by sector using the largest-remainder method."""
    raw = counts / counts.sum() * n
    quota = np.floor(raw).astype(int)
    extra = n - quota.sum()
    quota.loc[(raw - quota).sort_values(ascending=False).index[:extra]] += 1
    return quota.clip(upper=counts)


def round_robin_pick(sector_df: pd.DataFrame, k: int, rng) -> list:
    """Cycle through (cap x freq x age) cells so every combination is represented."""
    pools = {c: list(rng.permutation(g["symbol"].values))
             for c, g in sector_df.groupby("cell")}
    cells = list(rng.permutation(list(pools)))
    picked = []
    while len(picked) < k and any(pools.values()):
        for c in cells:
            if pools[c] and len(picked) < k:
                picked.append(pools[c].pop())
    return picked


def select_companies(df: pd.DataFrame, n: int = TARGET_N) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    eligible = df[df["years_listed"] >= MIN_YEARS_LISTED].copy()
    print(f"Eligible after history filter: {len(eligible)} of {len(df)}")

    quotas = allocate_quotas(eligible["sector"].value_counts(), n)
    selected = []
    for sector, k in quotas.items():
        selected += round_robin_pick(eligible[eligible["sector"] == sector], int(k), rng)

    eligible["selected"] = eligible["symbol"].isin(selected)

    # reserves: random order within sector, used if FMP lacks data for a pick
    eligible["reserve_rank"] = np.nan
    for sector, g in eligible[~eligible["selected"]].groupby("sector"):
        order = rng.permutation(g.index)
        eligible.loc[order, "reserve_rank"] = np.arange(1, len(order) + 1)
    return eligible


# ------------------------- 5. balance report ----------------------------------
def balance_report(universe: pd.DataFrame, sel: pd.DataFrame) -> pd.DataFrame:
    chosen = sel[sel["selected"]]
    parts = []
    for col in ["sector", "cap_tier", "freq_tier", "age_tier"]:
        u = universe[col].value_counts(normalize=True).rename("universe_share")
        s = chosen[col].value_counts(normalize=True).rename("sample_share")
        t = pd.concat([u, s], axis=1).fillna(0).round(3)
        t.insert(0, "dimension", col)
        t.index.name = "level"
        parts.append(t.reset_index())
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    uni = get_sp500_universe()
    uni.to_csv(UNIVERSE_CSV, index=False)
    print(f"S&P 500 constituents: {len(uni)}")

    meta = get_metadata(uni["symbol"].tolist())
    full = add_strata(uni.merge(meta, on="symbol", how="left"))

    result = select_companies(full)
    result.to_csv(SELECTED_CSV, index=False)

    report = balance_report(full, result)
    report.to_csv(BALANCE_CSV, index=False)

    chosen = result[result["selected"]]
    print(f"\nSelected {len(chosen)} companies")
    print(report.to_string(index=False))
    print("\nCap x Frequency x Age cells covered:", chosen["cell"].nunique(), "of 27")