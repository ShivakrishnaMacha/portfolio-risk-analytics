"""
portfolio-risk-analytics — seeded synthetic daily price generator.

Deterministic (fixed seed): 40 tickers across 5 sectors x ~3 years of daily
closes, generated with geometric Brownian motion layered on a factor
structure (market factor + sector factors + idiosyncratic noise) so
cross-asset correlations look like real equities.

Outputs (under data/):
  - prices.parquet   : date, ticker, close
  - tickers.csv      : ticker, sector, beta_market, drift, shares_outstanding
  - index.parquet    : date, close of the synthetic cap-weighted market index
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_DAYS = 756  # ~3 trading years
END_DATE = "2026-10-16"

SECTORS = ["Technology", "Healthcare", "Financials", "Energy", "Consumer"]
TICKERS_PER_SECTOR = 8

# Sector annualised characteristics (drift, vol of the sector factor,
# and the sector's average market beta).
SECTOR_PARAMS = {
    "Technology": {"drift": 0.12, "sector_vol": 0.18, "beta": 1.15},
    "Healthcare": {"drift": 0.08, "sector_vol": 0.15, "beta": 0.90},
    "Financials": {"drift": 0.07, "sector_vol": 0.16, "beta": 1.05},
    "Energy": {"drift": 0.06, "sector_vol": 0.26, "beta": 0.95},
    "Consumer": {"drift": 0.06, "sector_vol": 0.13, "beta": 0.80},
}

MARKET_DRIFT = 0.09   # annualised market factor drift
MARKET_VOL = 0.16     # annualised market factor vol
IDIO_VOL = 0.22       # annualised idiosyncratic vol
RISK_FREE = 0.02


def generate_dataframes(seed: int = SEED):
    """Build the seeded price/index dataframes (deterministic)."""
    rng = np.random.default_rng(seed)
    n_tickers = len(SECTORS) * TICKERS_PER_SECTOR

    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS)
    dt = 1 / 252  # one trading day in years

    # --- Factor shocks (daily log-return shocks) ---------------------------------
    mkt_shock = rng.normal(0.0, MARKET_VOL * np.sqrt(dt), size=N_DAYS)
    sector_shock = {
        s: rng.normal(0.0, p["sector_vol"] * np.sqrt(dt), size=N_DAYS)
        for s, p in SECTOR_PARAMS.items()
    }

    # --- Per-ticker simulation ---------------------------------------------------
    prices: dict[str, np.ndarray] = {}
    meta_rows = []
    ticker_order: list[str] = []
    betas: list[float] = []

    for sector in SECTORS:
        p = SECTOR_PARAMS[sector]
        for i in range(1, TICKERS_PER_SECTOR + 1):
            ticker = f"{sector[:4].upper()}{i:02d}"  # e.g. TECH01
            ticker_order.append(ticker)

            beta = float(rng.normal(p["beta"], 0.12))
            idio_shock = rng.normal(0.0, IDIO_VOL * np.sqrt(dt), size=N_DAYS)

            drift = (p["drift"] - 0.5 * (p["sector_vol"] ** 2 + IDIO_VOL ** 2)) * dt
            log_rets = (
                drift
                + beta * mkt_shock
                + sector_shock[sector]
                + idio_shock
            )

            start_price = float(rng.uniform(20.0, 400.0))
            prices[ticker] = start_price * np.exp(np.cumsum(log_rets))
            betas.append(beta)

            meta_rows.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "beta_market": round(beta, 4),
                    "annualised_drift": p["drift"],
                    # Synthetic shares outstanding for cap-weighting the index.
                    "shares_outstanding": int(rng.integers(5_000_000, 500_000_000)),
                }
            )

    n_tickers = len(ticker_order)
    price_rows = pd.DataFrame(
        {
            "date": np.repeat(dates.values, n_tickers),
            "ticker": ticker_order * N_DAYS,
            "close": np.concatenate([prices[t] for t in ticker_order])
            .reshape(n_tickers, N_DAYS)
            .T.ravel(),
        }
    )

    # --- Market index: cap-weighted composite of the 40 tickers -------------------
    shares = {r["ticker"]: r["shares_outstanding"] for r in meta_rows}
    market_cap = pd.DataFrame(
        {t: prices[t] * shares[t] for t in ticker_order}, index=dates
    )
    total_cap = market_cap.sum(axis=1)
    index_rets = total_cap.pct_change().fillna(0.0)
    index_level = 1000.0 * (1.0 + index_rets).cumprod()

    index_df = pd.DataFrame({"date": dates.values, "close": index_level.values})

    return price_rows, pd.DataFrame(meta_rows), index_df


def write_parquet(price_rows: pd.DataFrame, meta: pd.DataFrame,
                  index_df: pd.DataFrame, data_dir: Path) -> None:
    """Persist the seeded dataframes (the deterministic build artifacts)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    price_rows.to_parquet(data_dir / "prices.parquet", index=False)
    meta.to_csv(data_dir / "tickers.csv", index=False)
    index_df.to_parquet(data_dir / "index.parquet", index=False)


def main() -> None:
    price_rows, meta, index_df = generate_dataframes()
    data_dir = Path(__file__).resolve().parents[1] / "data"
    write_parquet(price_rows, meta, index_df, data_dir)

    n_tickers = meta["ticker"].nunique()
    dates = price_rows["date"].unique()
    print(f"Wrote {len(price_rows):,} price rows for {n_tickers} tickers "
          f"({min(dates).date()} → {max(dates).date()})")
    print(f"Index start {index_df['close'].iloc[0]:.1f} → "
          f"end {index_df['close'].iloc[-1]:.1f}")


if __name__ == "__main__":
    main()
