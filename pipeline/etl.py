"""
portfolio-risk-analytics — ETL: parquet/CSV sources -> DuckDB star schema.

Builds dim_ticker, dim_date, fact_daily_prices, fact_returns,
fact_market_index in data/warehouse.duckdb. Fully deterministic: the same
seeded inputs always produce the same warehouse.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Re-export the generator so callers (e.g. the Streamlit app) can build the
# warehouse in one step on first run.
from data_gen.generate import generate_dataframes, write_parquet  # noqa: E402


def build_db(db_path: Path, price_rows: pd.DataFrame, tickers: pd.DataFrame,
             index: pd.DataFrame) -> dict:
    """Load seeded dataframes into a fresh DuckDB star schema; returns counts."""
    if db_path.exists():
        db_path.unlink()  # rebuild from scratch: deterministic pipeline

    con = duckdb.connect(str(db_path))
    con.execute((ROOT / "pipeline" / "schema.sql").read_text())

    # --- dim_ticker -----------------------------------------------------------
    con.register("tickers", tickers)
    con.execute(
        "INSERT INTO dim_ticker SELECT ticker, sector, beta_market, "
        "annualised_drift, shares_outstanding FROM tickers"
    )

    # --- dim_date -------------------------------------------------------------
    all_dates = pd.DataFrame(
        {"date": sorted(set(price_rows["date"]) | set(index["date"]))}
    )
    all_dates["year"] = all_dates["date"].dt.year
    all_dates["month"] = all_dates["date"].dt.month
    all_dates["day"] = all_dates["date"].dt.day
    all_dates["day_of_week"] = all_dates["date"].dt.dayofweek
    all_dates["is_month_end"] = all_dates["date"].dt.is_month_end
    con.register("dates", all_dates)
    con.execute("INSERT INTO dim_date SELECT * FROM dates")

    # --- fact_daily_prices (with market cap) ----------------------------------
    cap = price_rows.merge(tickers[["ticker", "shares_outstanding"]], on="ticker")
    cap["market_cap"] = cap["close"] * cap["shares_outstanding"]
    fact_prices = cap[["date", "ticker", "close", "market_cap"]]
    con.register("fact_prices", fact_prices)
    con.execute(
        "INSERT INTO fact_daily_prices SELECT date, ticker, close, market_cap "
        "FROM fact_prices"
    )

    # --- fact_returns ----------------------------------------------------------
    price_wide = price_rows.pivot(index="date", columns="ticker",
                                  values="close").sort_index()
    log_rets = np.log(price_wide / price_wide.shift(1))
    simple_rets = price_wide.pct_change()
    fr = log_rets.stack().rename("log_return").to_frame()
    fr["simple_return"] = simple_rets.stack()
    fr = fr.dropna().reset_index().rename(columns={"level_1": "ticker"})
    fr = fr[["date", "ticker", "log_return", "simple_return"]]
    con.register("fr_df", fr)
    con.execute("INSERT INTO fact_returns SELECT * FROM fr_df")

    # --- fact_market_index ------------------------------------------------------
    index = index.sort_values("date")
    index["log_return"] = np.log(index["close"] / index["close"].shift(1))
    con.register("index_df", index)
    con.execute("INSERT INTO fact_market_index SELECT date, close, log_return FROM index_df")

    # --- data-quality assertions (fail the load if anything is off) -------------
    (price_count,) = con.execute("SELECT COUNT(*) FROM fact_daily_prices").fetchone()
    (null_close,) = con.execute(
        "SELECT COUNT(*) FROM fact_daily_prices WHERE close IS NULL OR close <= 0"
    ).fetchone()
    (return_rows,) = con.execute("SELECT COUNT(*) FROM fact_returns").fetchone()
    (date_rows,) = con.execute("SELECT COUNT(*) FROM dim_date").fetchone()
    assert null_close == 0, f"{null_close} bad close values"
    assert price_count == return_rows + len(tickers), (
        f"returns should equal prices minus one row per ticker: "
        f"{price_count=} {return_rows=}"
    )
    assert date_rows > 0, "dim_date is empty"

    con.close()
    return {"prices": price_count, "returns": return_rows, "dates": date_rows}


def main() -> None:
    price_rows, meta, index_df = generate_dataframes()
    write_parquet(price_rows, meta, index_df, DATA_DIR)
    counts = build_db(DATA_DIR / "warehouse.duckdb", price_rows, meta, index_df)
    print(f"ETL OK: {counts['prices']:,} price rows, {counts['returns']:,} return rows, "
          f"{counts['dates']} dates → warehouse.duckdb")


if __name__ == "__main__":
    main()
