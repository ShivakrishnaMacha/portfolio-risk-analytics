"""
portfolio-risk-analytics — risk & performance analytics layer.

Pure pandas/numpy: everything here runs against the return series pulled
from the DuckDB warehouse. All metrics assume daily log returns unless
stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Basic performance
# --------------------------------------------------------------------------- #
def cumulative_returns(log_returns: pd.Series) -> pd.Series:
    """Cumulative simple-return series from daily log returns (starts at 0)."""
    return np.exp(log_returns.cumsum()) - 1.0


def total_return(log_returns: pd.Series) -> float:
    return float(np.exp(log_returns.sum()) - 1.0)


def annualised_return(log_returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    n = len(log_returns)
    if n == 0:
        return 0.0
    return float(np.exp(log_returns.mean() * periods_per_year) - 1.0)


def annualised_volatility(log_returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(log_returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    log_returns: pd.Series,
    risk_free_annual: float = 0.02,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    excess = log_returns - np.log(1 + risk_free_annual) / periods_per_year
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    log_returns: pd.Series,
    risk_free_annual: float = 0.02,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    excess = log_returns - np.log(1 + risk_free_annual) / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
@dataclass
class DrawdownResult:
    series: pd.Series       # drawdown series (<= 0)
    max_drawdown: float     # deepest drawdown, negative fraction
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None  # first date back above prior peak; None if unrecovered
    duration_days: int      # peak -> recovery (or peak -> last date if unrecovered)


def max_drawdown(log_returns: pd.Series) -> DrawdownResult:
    """Underwater series + peak/trough/recovery dates from log returns."""
    if len(log_returns) == 0:
        empty = pd.Series(dtype=float)
        return DrawdownResult(empty, 0.0, None, None, None, 0)

    wealth = np.exp(log_returns.cumsum())
    running_max = wealth.cummax()
    dd = (wealth - running_max) / running_max  # <= 0 everywhere

    trough_idx = dd.idxmin()
    max_dd = float(dd.loc[trough_idx])

    # Peak = the running max level in effect at the trough.
    peak_mask = (wealth[:trough_idx] == running_max.loc[trough_idx])
    peak_date = peak_mask[peak_mask].index[-1] if peak_mask.any() else wealth.index[0]

    # Recovery: first date after the trough where wealth >= pre-drawdown peak.
    peak_level = running_max.loc[trough_idx]
    after = wealth.loc[trough_idx:]
    recovered = after[after >= peak_level]
    recovery_date = recovered.index[0] if len(recovered) > 0 else None

    if recovery_date is not None:
        duration = (recovery_date - peak_date).days
    else:
        duration = (wealth.index[-1] - peak_date).days

    return DrawdownResult(
        series=dd,
        max_drawdown=max_dd,
        peak_date=peak_date,
        trough_date=trough_idx,
        recovery_date=recovery_date,
        duration_days=int(duration),
    )


# --------------------------------------------------------------------------- #
# Tail risk: VaR / CVaR (historical simulation)
# --------------------------------------------------------------------------- #
def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: the (1-confidence) quantile of returns (negative loss number)."""
    return float(returns.quantile(1.0 - confidence, interpolation="linear"))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR (expected shortfall): mean of returns at/below VaR."""
    var = historical_var(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


# --------------------------------------------------------------------------- #
# Correlation & factor exposures
# --------------------------------------------------------------------------- #
def correlation_matrix(returns_wide: pd.DataFrame) -> pd.DataFrame:
    return returns_wide.corr()


def factor_exposures(
    asset_returns: pd.Series, factors: pd.DataFrame
) -> dict[str, float]:
    """
    OLS betas of an asset's returns on factor returns.

    factors: DataFrame of factor daily returns (columns = factor names).
    Returns dict of {factor: beta}; includes an 'alpha' (annualised intercept).
    """
    aligned = pd.concat({"y": asset_returns, **factors}, axis=1).dropna()
    y = aligned["y"].to_numpy()
    X = np.column_stack([np.ones(len(aligned)), aligned[factors.columns].to_numpy()])
    betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    result = {"alpha": float(betas[0] * TRADING_DAYS)}
    for name, beta in zip(factors.columns, betas[1:]):
        result[name] = float(beta)
    return result


def rolling_volatility(log_returns: pd.Series, window: int = 30) -> pd.Series:
    return log_returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


# --------------------------------------------------------------------------- #
# Efficient frontier (Monte Carlo)
# --------------------------------------------------------------------------- #
@dataclass
class FrontierResult:
    weights: np.ndarray       # (n_portfolios, n_assets)
    annual_returns: np.ndarray
    annual_vols: np.ndarray
    sharpes: np.ndarray
    max_sharpe_weights: np.ndarray
    min_variance_weights: np.ndarray


def efficient_frontier(
    returns_wide: pd.DataFrame,
    n_portfolios: int = 2000,
    seed: int = 7,
    risk_free_annual: float = 0.02,
) -> FrontierResult:
    """Monte Carlo random-portfolio frontier from annualised mean/covariance."""
    rng = np.random.default_rng(seed)
    returns_wide = returns_wide.dropna()
    mu = returns_wide.mean().to_numpy() * TRADING_DAYS
    cov = returns_wide.cov(ddof=1).to_numpy() * TRADING_DAYS
    n_assets = returns_wide.shape[1]

    rand = rng.random((n_portfolios, n_assets))
    weights = rand / rand.sum(axis=1, keepdims=True)

    port_rets = weights @ mu
    port_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    port_vols = np.maximum(port_vols, 1e-12)
    sharpes = (port_rets - risk_free_annual) / port_vols

    return FrontierResult(
        weights=weights,
        annual_returns=port_rets,
        annual_vols=port_vols,
        sharpes=sharpes,
        max_sharpe_weights=weights[int(np.argmax(sharpes))],
        min_variance_weights=weights[int(np.argmin(port_vols))],
    )
