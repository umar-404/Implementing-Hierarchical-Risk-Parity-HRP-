"""HRP Portfolio Optimizer — Streamlit dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from backtester import (
    cumulative_returns,
    performance_summary,
    walk_forward_backtest,
)
from data_loader import DEFAULT_TICKERS, fetch_prices, compute_log_returns
from hrp_engine import get_corr, plot_dendrogram

st.set_page_config(page_title="HRP Portfolio Optimizer", layout="wide")

st.title("📊 Hierarchical Risk Parity (HRP) Portfolio Optimizer")
st.caption(
    "López de Prado (2016) HRP vs. Equal-Weight (1/N) and Min-Variance, "
    "with a walk-forward backtest."
)

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    tickers = st.multiselect(
        "Universe (tickers)",
        options=sorted(set(DEFAULT_TICKERS + ["BRK-B", "WMT", "KO", "CSCO", "META"])),
        default=DEFAULT_TICKERS,
    )
    if not tickers:
        st.warning("Select at least 2 tickers.")
        st.stop()

    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", pd.Timestamp("2018-01-01"))
    end_date = col2.date_input("End date", pd.Timestamp.today())

    rebalance_days = st.select_slider(
        "Rebalance frequency (trading days)",
        options=[63, 126, 252],
        format_func=lambda d: {63: "Quarterly", 126: "Semi-annual", 252: "Annual"}[d],
    )
    lookback_days = st.select_slider(
        "Lookback estimation window (trading days)",
        options=[252, 378, 504, 756],
        format_func=lambda d: f"{d} days",
    )
    linkage_method = st.selectbox(
        "Linkage method", options=["single", "ward", "average", "complete"],
        index=0,
    )

    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)


# ---------------------------------------------------------------- Helpers
@st.cache_data(show_spinner=False)
def load_returns(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Fetch prices and compute log returns (cached)."""
    prices = fetch_prices(list(tickers), start, end)
    return compute_log_returns(prices)


def fig_to_buffer(fig) -> io.BytesIO:
    """Serialize a matplotlib figure to a PNG buffer for downloads."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------- Run
if run_btn:
    try:
        returns = load_returns(tuple(tickers), str(start_date), str(end_date))
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    if len(returns) <= lookback_days:
        st.error(
            f"Only {len(returns)} return days available — need more than "
            f"{lookback_days} for the lookback window. Shorten the window "
            f"or extend the date range."
        )
        st.stop()

    with st.spinner("Running walk-forward backtest..."):
        try:
            result = walk_forward_backtest(
                returns,
                rebalance_days=rebalance_days,
                lookback_days=lookback_days,
                linkage_method=linkage_method,
            )
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

    port_returns = result["returns"]
    weights_hist = result["weights"]
    cum = cumulative_returns(port_returns)
    summary = performance_summary(port_returns)

    # ------------------------------------------------------------ Metrics
    st.subheader("Performance Metrics")
    st.dataframe(
        summary.style.format("{:.2%}", subset=["Annualized Return",
                                               "Annualized Volatility",
                                               "Max Drawdown"])
        .format("{:.2f}", subset=["Sharpe Ratio"]),
        use_container_width=True,
    )
    csv_metrics = summary.to_csv().encode()
    st.download_button("⬇️ Download metrics (CSV)", csv_metrics,
                       "hrp_metrics.csv", "text/csv")

    # --------------------------------------------------- Cumulative returns
    st.subheader("Cumulative Returns — HRP vs Benchmarks")
    fig_cum, ax_cum = plt.subplots(figsize=(12, 5))
    cum.plot(ax=ax_cum, linewidth=2)
    ax_cum.set_ylabel("Cumulative Growth ($1 invested)")
    ax_cum.set_xlabel("")
    ax_cum.legend(title="Strategy")
    ax_cum.grid(alpha=0.3)
    st.pyplot(fig_cum, use_container_width=True)
    st.download_button("⬇️ Download cumulative chart", fig_to_buffer(fig_cum),
                       "hrp_cumulative_returns.png", "image/png")

    # -------------------------------------------------- HRP latest weights
    st.subheader("Current HRP Weights")
    latest_w = weights_hist["HRP"].iloc[-1].sort_values(ascending=False)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(latest_w.to_frame("Weight").style.format("{:.2%}"))
    with c2:
        fig_w, ax_w = plt.subplots(figsize=(6, 4))
        ax_w.bar(latest_w.index, latest_w.values)
        ax_w.set_ylabel("Weight")
        ax_w.set_title("HRP Allocation (latest rebalance)")
        ax_w.grid(alpha=0.3, axis="y")
        st.pyplot(fig_w, use_container_width=True)

    # ---------------------------------------------------- Dendrogram
    st.subheader("Hierarchical Clustering Dendrogram")
    fig_d, ax_d = plt.subplots(figsize=(12, 5))
    plot_dendrogram(returns, linkage_method=linkage_method, ax=ax_d)
    st.pyplot(fig_d, use_container_width=True)
    st.download_button("⬇️ Download dendrogram", fig_to_buffer(fig_d),
                       "hrp_dendrogram.png", "image/png")

    # ---------------------------------------------------- Correlation heatmap
    st.subheader("Correlation Matrix Heatmap")
    fig_h, ax_h = plt.subplots(figsize=(9, 7))
    sns.heatmap(get_corr(returns), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0, ax=ax_h, square=True)
    st.pyplot(fig_h, use_container_width=True)

    with st.expander("View raw data"):
        tab1, tab2 = st.tabs(["Log Returns", "HRP Weight History"])
        tab1.dataframe(returns, use_container_width=True)
        tab2.dataframe(weights_hist["HRP"], use_container_width=True)
else:
    st.info("Configure the settings in the sidebar and click **Run Backtest**.")

