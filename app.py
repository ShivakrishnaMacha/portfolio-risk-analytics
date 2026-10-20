"""Portfolio risk & performance lab dashboard.

Run with:  streamlit run app.py   (from the project root, after the
quickstart steps in the README have built data/warehouse.duckdb)
"""

from __future__ import annotations

import pathlib

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from metrics.metrics import (
    annualised_return,
    annualised_volatility,
    correlation_matrix,
    cumulative_returns,
    efficient_frontier,
    factor_exposures,
    historical_cvar,
    historical_var,
    max_drawdown,
    rolling_volatility,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)

DB_PATH = pathlib.Path(__file__).resolve().parent / "data" / "warehouse.duckdb"
RISK_FREE = 0.02


@st.cache_data(show_spinner="Loading warehouse…")
def get_data():
    if not DB_PATH.exists():
        # First run (e.g. Streamlit Community Cloud): build the seeded
        # warehouse in place so the demo works with no committed binaries.
        from data_gen.generate import generate_dataframes
        from pipeline.etl import build_db

        with st.spinner("Building seeded demo dataset (one-time, ~10 seconds)…"):
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            build_db(DB_PATH, *generate_dataframes())

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tickers = con.execute("SELECT * FROM dim_ticker").df()
        prices = con.execute(
            "SELECT date, ticker, close FROM fact_daily_prices"
        ).df()
        returns = con.execute(
            "SELECT date, ticker, log_return, simple_return FROM fact_returns"
        ).df()
        index = con.execute(
            "SELECT date, close, log_return FROM fact_market_index"
        ).df()
    finally:
        con.close()

    for df in (prices, returns, index):
        df["date"] = pd.to_datetime(df["date"])

    log_wide = returns.pivot(index="date", columns="ticker",
                             values="log_return").sort_index().dropna()
    simple_wide = returns.pivot(index="date", columns="ticker",
                                values="simple_return").sort_index().dropna()

    # Portfolio = daily-rebalanced equal-weight across the 40 tickers.
    port_log = log_wide.mean(axis=1)
    port_simple = simple_wide.mean(axis=1)

    idx_log = index.set_index("date")["log_return"].dropna()
    idx_simple = (index.set_index("date")["close"].pct_change().dropna())

    # Synthetic proxy factors (labelled as proxies, not Fama-French data).
    factors = build_proxy_factors(log_wide, tickers)

    return {
        "tickers": tickers,
        "port_log": port_log,
        "port_simple": port_simple,
        "idx_log": idx_log,
        "idx_simple": idx_simple,
        "log_wide": log_wide,
        "factors": factors,
        "index": index,
    }


def build_proxy_factors(log_wide: pd.DataFrame, tickers: pd.DataFrame) -> pd.DataFrame:
    """Seeded proxy factors: market (index), size, value, momentum."""
    info = tickers.set_index("ticker")
    caps = info["shares_outstanding"]

    small = caps[caps <= caps.median()].index.intersection(log_wide.columns)
    big = caps[caps > caps.median()].index.intersection(log_wide.columns)
    size = log_wide[small].mean(axis=1) - log_wide[big].mean(axis=1)

    beta = info["beta_market"]
    low_beta = beta[beta <= beta.quantile(0.4)].index.intersection(log_wide.columns)
    high_beta = beta[beta >= beta.quantile(0.6)].index.intersection(log_wide.columns)
    value = log_wide[low_beta].mean(axis=1) - log_wide[high_beta].mean(axis=1)

    prior21 = log_wide.rolling(21).sum().shift(1)
    winners = prior21.apply(
        lambda row: row[row >= row.quantile(0.7)].index, axis=1
    )
    losers = prior21.apply(
        lambda row: row[row <= row.quantile(0.3)].index, axis=1
    )
    mom_rows = []
    for date, row in log_wide.iterrows():
        w, l = winners.loc[date], losers.loc[date]
        mom_rows.append(row[w].mean() - row[l].mean() if len(w) and len(l) else 0.0)
    momentum = pd.Series(mom_rows, index=log_wide.index)

    return pd.DataFrame({
        "size": size,
        "value": value,
        "momentum": momentum,
    })


st.set_page_config(page_title="Portfolio Risk Analytics", layout="wide",
                   page_icon="📉")
st.title("📉 Portfolio Risk & Performance Lab")
st.caption("40 seeded synthetic equities · 5 sectors · DuckDB warehouse · "
           "all figures from deterministic daily data")

data = get_data()
port_log, port_simple = data["port_log"], data["port_simple"]
idx_log, idx_simple = data["idx_log"], data["idx_simple"]

# Market factor for the factor panel = the cap-weighted synthetic index.
factors = data["factors"].copy()
factors["market"] = idx_log.reindex(factors.index)
factors = factors.dropna()

# ------------------------------------------------------------ sidebar --------
st.sidebar.header("Controls")
confidence = st.sidebar.selectbox("VaR/CVaR confidence", ["95%", "99%"],
                                  index=0)
conf_level = 0.95 if confidence == "95%" else 0.99
log_scale = st.sidebar.checkbox("Log scale on equity curves", value=False)
n_portfolios = st.sidebar.slider("Frontier portfolios", 500, 5000, 2000,
                                 step=500)

# ------------------------------------------------------------ KPI row --------
def kpi(label: str, value: float, delta: float | None, fmt: str):
    st.metric(label, fmt.format(value),
              None if delta is None else fmt.format(delta))

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    kpi("Total return (3Y)", total_return(port_log),
        total_return(port_log) - total_return(idx_log), "{:.1%}")
with col2:
    kpi("Annualised return", annualised_return(port_log),
        annualised_return(port_log) - annualised_return(idx_log), "{:.1%}")
with col3:
    kpi("Annualised vol", annualised_volatility(port_log),
        annualised_volatility(port_log) - annualised_volatility(idx_log),
        "{:.1%}")
with col4:
    kpi("Sharpe", sharpe_ratio(port_log, RISK_FREE),
        sharpe_ratio(port_log, RISK_FREE) - sharpe_ratio(idx_log, RISK_FREE),
        "{:.2f}")
with col5:
    kpi(f"VaR {confidence}", historical_var(port_simple, conf_level),
        historical_var(port_simple, conf_level)
        - historical_var(idx_simple, conf_level), "{:.2%}")
with col6:
    dd = max_drawdown(port_log)
    idx_dd = max_drawdown(idx_log)
    kpi("Max drawdown", dd.max_drawdown,
        dd.max_drawdown - idx_dd.max_drawdown, "{:.1%}")

st.caption("Deltas are portfolio minus the cap-weighted benchmark index.")

# ---------------------------------------------------- equity curves --------
st.subheader("Equity curves")
equity = pd.DataFrame({
    "Portfolio (equal-weight)": np.exp(port_log.cumsum()),
    "Benchmark (cap-weighted)": np.exp(idx_log.reindex(port_log.index).cumsum()),
})
fig = go.Figure()
for col in equity.columns:
    fig.add_trace(go.Scatter(x=equity.index, y=equity[col], mode="lines",
                             name=col))
fig.update_layout(yaxis_title="Growth of $1",
                  yaxis_type="log" if log_scale else "linear",
                  hovermode="x unified", height=420)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------- drawdown --------
st.subheader("Drawdown (underwater chart)")
dd_fig = go.Figure()
dd_fig.add_trace(go.Scatter(x=dd.series.index, y=dd.series * 100,
                            fill="tozeroy", name="Drawdown %",
                            line=dict(color="#d62728")))
rec = dd.recovery_date.date() if dd.recovery_date is not None else "not recovered"
dd_fig.update_layout(
    yaxis_title="Drawdown (%)",
    title=(f"Max drawdown {dd.max_drawdown:.1%} · "
           f"trough {dd.trough_date.date()} · recovery {rec} · "
           f"duration {dd.duration_days} days"),
    height=380,
)
st.plotly_chart(dd_fig, use_container_width=True)

# ---------------------------------------------------------- risk panel --------
st.subheader("Tail risk")
rc1, rc2 = st.columns(2)
with rc1:
    var_table = pd.DataFrame({
        "VaR 95%": [historical_var(port_simple, 0.95),
                    historical_var(idx_simple, 0.95)],
        "CVaR 95%": [historical_cvar(port_simple, 0.95),
                     historical_cvar(idx_simple, 0.95)],
        "VaR 99%": [historical_var(port_simple, 0.99),
                    historical_var(idx_simple, 0.99)],
        "CVaR 99%": [historical_cvar(port_simple, 0.99),
                     historical_cvar(idx_simple, 0.99)],
    }, index=["Portfolio", "Benchmark"])
    st.dataframe(var_table.style.format("{:.2%}"), use_container_width=True)
    st.write(f"Sortino (portfolio): **{sortino_ratio(port_log, RISK_FREE):.2f}** "
             f"vs benchmark {sortino_ratio(idx_log, RISK_FREE):.2f}")
with rc2:
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=port_simple * 100, nbinsx=60,
                                name="Daily returns %",
                                marker_color="#1f77b4", opacity=0.75))
    for cl, color in [(0.95, "#d62728"), (0.99, "#7f7f7f")]:
        v = historical_var(port_simple, cl) * 100
        hist.add_vline(x=v, line_dash="dash", line_color=color,
                       annotation_text=f"VaR {int(cl*100)}% {v:.2f}%")
    hist.update_layout(xaxis_title="Daily return (%)", yaxis_title="Days",
                       height=380, showlegend=False)
    st.plotly_chart(hist, use_container_width=True)

st.subheader("Rolling 30-day volatility")
rv = pd.DataFrame({
    "Portfolio": rolling_volatility(port_log, 30),
    "Benchmark": rolling_volatility(idx_log.reindex(port_log.index), 30),
})
st.plotly_chart(px.line(rv, y=["Portfolio", "Benchmark"], height=340),
                use_container_width=True)

# -------------------------------------------- correlation & factors --------
st.subheader("Correlation & factor exposures")
cf1, cf2 = st.columns(2)
with cf1:
    sector = data["tickers"].set_index("ticker")["sector"]
    sector_rets = pd.DataFrame({
        s: data["log_wide"][sector[sector == s].index.intersection(
            data["log_wide"].columns)].mean(axis=1)
        for s in sector.unique()
    })
    corr = correlation_matrix(sector_rets)
    heat = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                colorscale="RdBu", zmin=-1, zmax=1,
                                text=corr.round(2).values,
                                texttemplate="%{text}"))
    heat.update_layout(title="Sector return correlation", height=380)
    st.plotly_chart(heat, use_container_width=True)
with cf2:
    betas = factor_exposures(port_log, factors)
    bdf = pd.DataFrame({"factor": list(betas.keys()),
                        "exposure": list(betas.values)})
    bars = px.bar(bdf, x="factor", y="exposure", height=380,
                  title="Portfolio factor exposures (OLS betas)",
                  color="exposure", color_continuous_scale="RdBu")
    st.plotly_chart(bars, use_container_width=True)
    st.caption("Factors are seeded proxies (size = small−big cap, value = "
               "low−high beta, momentum = winners−losers), not Fama-French data.")

# ----------------------------------------------- efficient frontier --------
st.subheader("Efficient frontier (Monte Carlo)")
with st.spinner(f"Simulating {n_portfolios:,} random portfolios…"):
    fr = efficient_frontier(data["log_wide"], n_portfolios=n_portfolios, seed=7)

bench_rets = data["log_wide"].mean(axis=1)
bench_sharpe = sharpe_ratio(bench_rets, RISK_FREE)

ff = go.Figure()
ff.add_trace(go.Scatter(x=fr.annual_vols * 100, y=fr.annual_returns * 100,
                        mode="markers", name="Random portfolios",
                        marker=dict(color=fr.sharpes, colorscale="Viridis",
                                    showscale=True,
                                    colorbar=dict(title="Sharpe")),
                        opacity=0.55))
ms = fr.max_sharpe_weights
mv = fr.min_variance_weights
for w, name, color in [(ms, "Max Sharpe", "#d62728"), (mv, "Min variance",
                                                       "#2ca02c")]:
    r = float(w @ (data["log_wide"].mean().to_numpy() * 252))
    v = float(np.sqrt(w @ (data["log_wide"].cov(ddof=1).to_numpy() * 252) @ w))
    ff.add_trace(go.Scatter(x=[v * 100], y=[r * 100], mode="markers",
                            name=name, marker=dict(size=14, color=color,
                                                   symbol="star")))
ff.update_layout(xaxis_title="Annualised volatility (%)",
                 yaxis_title="Annualised return (%)",
                 height=460, legend=dict(orientation="h", y=1.08))
st.plotly_chart(ff, use_container_width=True)

wdf = pd.DataFrame({"ticker": data["log_wide"].columns,
                    "max_sharpe_%": ms * 100,
                    "min_variance_%": mv * 100})
wdf = wdf.sort_values("max_sharpe_%", ascending=False)
st.dataframe(wdf.style.format("{:.1f}"), use_container_width=True,
             height=320)
st.caption(f"Equal-weight benchmark Sharpe: {bench_sharpe:.2f} · "
           f"max-Sharpe portfolio Sharpe: {fr.sharpes.max():.2f} · "
           "weights sum to 1 by construction.")
