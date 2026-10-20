"""
Tests for the portfolio-risk-analytics metrics layer.

Uses small hand-built fixtures with analytically known answers, plus the
seeded synthetic series for distributional properties (VaR quantile, bounds).
"""

import numpy as np
import pandas as pd
import pytest

from metrics.metrics import (
    annualised_return,
    correlation_matrix,
    cumulative_returns,
    efficient_frontier,
    factor_exposures,
    historical_cvar,
    historical_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)

TRADING_DAYS = 252


@pytest.fixture
def fixture_returns() -> pd.Series:
    """Fixed series with a known total return: exp(sum(log r)) - 1."""
    return pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], name="log_ret")


def test_total_return_compounds_exactly(fixture_returns):
    # (1+r) compounding through simple returns must equal the log-based total.
    simple = np.exp(fixture_returns) - 1.0
    compounded = float(np.prod(1.0 + simple) - 1.0)
    assert total_return(fixture_returns) == pytest.approx(compounded, rel=1e-12)


def test_cumulative_returns_endpoints(fixture_returns):
    cum = cumulative_returns(fixture_returns)
    assert cum.iloc[0] == pytest.approx(np.exp(0.01) - 1.0)
    assert cum.iloc[-1] == pytest.approx(total_return(fixture_returns))


def test_sharpe_matches_manual():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0004, 0.012, 500))
    rf_daily = np.log(1 + 0.02) / TRADING_DAYS
    excess = rets - rf_daily
    manual = excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert sharpe_ratio(rets) == pytest.approx(manual, rel=1e-9)


def test_sortino_uses_only_downside():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0005, 0.01, 500))
    rf_daily = np.log(1 + 0.02) / TRADING_DAYS
    excess = rets - rf_daily
    downside = excess[excess < 0]
    manual = excess.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert sortino_ratio(rets) == pytest.approx(manual, rel=1e-9)


def test_drawdown_nonpositive_with_correct_recovery():
    # Wealth: 1 -> 1.5 -> 0.75 (50% drawdown) -> 1.5 (recovered).
    prices = pd.Series([1.0, 1.5, 0.75, 1.5],
                       index=pd.date_range("2024-01-01", periods=4))
    log_rets = np.log(prices / prices.shift(1)).dropna()
    result = max_drawdown(log_rets)

    assert (result.series <= 1e-12).all(), "drawdown must be <= 0 everywhere"
    assert result.max_drawdown == pytest.approx(-0.5, rel=1e-9)
    assert result.trough_date == prices.index[2]
    assert result.recovery_date == prices.index[3]
    assert result.peak_date == prices.index[1]


def test_drawdown_unrecovered_has_no_recovery_date():
    prices = pd.Series([1.0, 2.0, 1.0, 0.5],
                       index=pd.date_range("2024-01-01", periods=4))
    log_rets = np.log(prices / prices.shift(1)).dropna()
    result = max_drawdown(log_rets)
    assert result.max_drawdown == pytest.approx(-0.75, rel=1e-9)
    assert result.recovery_date is None


def test_var_quantile_property_on_seeded_series():
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(0.0005, 0.011, 10_000))
    var95 = historical_var(rets, 0.95)
    breach_rate = float((rets <= var95).mean())
    assert abs(breach_rate - 0.05) < 0.005  # ~5% of returns below the 95% VaR
    cvar95 = historical_cvar(rets, 0.95)
    assert cvar95 < var95  # CVaR is always worse than VaR
    assert historical_var(rets, 0.99) < var95  # higher confidence -> deeper cut


def test_correlation_bounds_and_diagonal():
    rng = np.random.default_rng(3)
    wide = pd.DataFrame(rng.normal(0, 0.01, (200, 5)), columns=list("ABCDE"))
    corr = correlation_matrix(wide)
    assert ((corr >= -1.0 - 1e-12) & (corr <= 1.0 + 1e-12)).all().all()
    assert np.allclose(np.diag(corr.values), 1.0)


def test_factor_exposures_recover_known_betas():
    rng = np.random.default_rng(4)
    mkt = pd.Series(rng.normal(0.0004, 0.01, 1000), name="market")
    smb = pd.Series(rng.normal(0.0001, 0.006, 1000), name="size")
    factors = pd.DataFrame({"market": mkt, "size": smb})
    noise = rng.normal(0, 0.002, 1000)
    asset = 0.0002 + 1.25 * mkt.values + 0.60 * smb.values + noise
    betas = factor_exposures(pd.Series(asset), factors)
    assert betas["market"] == pytest.approx(1.25, abs=0.05)
    assert betas["size"] == pytest.approx(0.60, abs=0.05)


def test_frontier_weights_sum_to_one_and_sharpe_beats_benchmark():
    rng = np.random.default_rng(5)
    wide = pd.DataFrame(
        rng.normal(0.0005, 0.012, (300, 8)) + rng.normal(0.0003, 0.006, (300, 1)),
        columns=[f"T{i}" for i in range(8)],
    )
    fr = efficient_frontier(wide, n_portfolios=2000, seed=7)
    assert np.allclose(fr.weights.sum(axis=1), 1.0)
    assert (fr.annual_vols >= 0).all()
    assert len(fr.max_sharpe_weights) == 8
    # Max-Sharpe random portfolio should beat the equally-weighted benchmark.
    bench_rets = wide.mean(axis=1)
    bench_sharpe = sharpe_ratio(bench_rets)
    assert fr.sharpes.max() >= bench_sharpe
