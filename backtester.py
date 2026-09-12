"""Walk-forward backtesting engine comparing HRP against benchmarks.

Benchmarks: Equal-Weighted (1/N) and a standard Minimum Variance
portfolio. Weights are estimated on a trailing lookback window and
held until the next rebalance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hrp_engine import get_hrp_weights


def equal_weight_weights(asset_names: list[str] | pd.Index) -> pd.Series:
    """Return equal (1/N) portfolio weights.

    Args:
        asset_names: Names of the assets in the universe.

    Returns:
        Series of weights, each equal to ``1 / N``.
    """
    names = list(asset_names)
    return pd.Series(1.0 / len(names), index=names)


def min_variance_weights(returns: pd.DataFrame) -> pd.Series:
    """Compute standard (long-only) minimum variance portfolio weights.

    Solved via the closed-form ``w = inv(Sigma) 1 / (1' inv(Sigma) 1)``
    with negative entries floored at zero and re-normalized (a robust
    heuristic that avoids a scipy/CVXPY dependency).

    Args:
        returns: In-sample window of asset returns.

    Returns:
        Series of long-only weights summing to 1.0.
    """
    cov = returns.cov().values.astype(float)
    n = cov.shape[0]
    # Regularize against singular collinear covariance matrices.
    cov += np.eye(n) * 1e-10 * max(np.trace(cov) / n, 1e-12)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
        return equal_weight_weights(returns.columns)

    ones = np.ones(n)
    w = inv @ ones
    w = np.where(w > 0, w, 0.0)
    if w.sum() <= 0:
        return equal_weight_weights(returns.columns)
    w = w / w.sum()
    return pd.Series(w, index=returns.columns)


def compute_weights(
    method: str,
    returns: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Dispatch weight computation by strategy name.

    Args:
        method: One of ``'HRP'``, ``'Equal-Weight'``, ``'Min-Variance'``.
        returns: In-sample window of asset returns.
        linkage_method: Linkage method passed through to HRP.

    Returns:
        Series of portfolio weights summing to 1.0.

    Raises:
        ValueError: If ``method`` is not recognized.
    """
    if method == "HRP":
        return get_hrp_weights(returns, linkage_method=linkage_method)
    if method == "Equal-Weight":
        return equal_weight_weights(returns.columns)
    if method == "Min-Variance":
        return min_variance_weights(returns)
    raise ValueError(f"Unknown strategy: {method!r}")


def walk_forward_backtest(
    returns: pd.DataFrame,
    rebalance_days: int = 252,
    lookback_days: int = 504,
    strategies: tuple[str, ...] = ("HRP", "Equal-Weight", "Min-Variance"),
    linkage_method: str = "single",
) -> dict[str, object]:
    """Run a walk-forward backtest for multiple strategies.

    Every ``rebalance_days`` trading days, weights are re-estimated from
    the trailing ``lookback_days`` window and held until the next
    rebalance. Daily portfolio returns are tracked from the first day
    where a full lookback window is available.

    Args:
        returns: DataFrame of daily asset returns.
        rebalance_days: Holding period between rebalances (default 252).
        lookback_days: Trailing estimation window length (default 504).
        strategies: Strategy names to simulate.
        linkage_method: Linkage method passed through to HRP.

    Returns:
        Dict with keys ``'returns'`` (DataFrame of daily portfolio
        returns per strategy) and ``'weights'`` (dict mapping strategy
        -> DataFrame of historical weights, one row per rebalance).

    Raises:
        ValueError: If history is shorter than the lookback window.
    """
    n = len(returns)
    if n <= lookback_days:
        raise ValueError(
            f"Not enough history: {n} rows <= lookback of {lookback_days} days."
        )

    start = lookback_days
    all_dates = returns.index[start:]

    port_returns = pd.DataFrame(index=all_dates)
    weight_history: dict[str, list[pd.Series]] = {s: [] for s in strategies}
    weight_dates: dict[str, list[pd.Timestamp]] = {s: [] for s in strategies}

    current_weights: dict[str, pd.Series] = {}
    next_rebal = start

    for i in range(start, n):
        if i >= next_rebal:
            window = returns.iloc[i - lookback_days : i]
            current_weights = {
                s: compute_weights(s, window, linkage_method=linkage_method)
                for s in strategies
            }
            for s in strategies:
                weight_history[s].append(current_weights[s])
                weight_dates[s].append(returns.index[i])
            next_rebal = i + rebalance_days

        daily = returns.iloc[i]
        for s in strategies:
            w = current_weights[s].reindex(daily.index).fillna(0.0)
            port_returns.loc[returns.index[i], s] = float(w @ daily)

    port_returns = port_returns.astype(float)
    weights_dfs = {
        s: pd.DataFrame(weight_history[s], index=weight_dates[s])
        for s in strategies
    }
    return {"returns": port_returns, "weights": weights_dfs}


def cumulative_returns(port_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound daily portfolio returns into a cumulative growth curve.

    Args:
        port_returns: DataFrame of daily portfolio returns per strategy.

    Returns:
        DataFrame of cumulative return indices starting at 1.0.
    """
    return (1.0 + port_returns).cumprod()


def annualized_return(port_returns: pd.Series) -> float:
    """Annualize a series of daily returns via geometric compounding.

    Args:
        port_returns: Daily portfolio returns.

    Returns:
        Annualized return as a decimal.
    """
    total = float((1.0 + port_returns).prod())
    years = len(port_returns) / 252.0
    if years <= 0 or total <= 0:
        return 0.0
    return total ** (1.0 / years) - 1.0


def annualized_volatility(port_returns: pd.Series) -> float:
    """Compute annualized volatility from daily returns.

    Args:
        port_returns: Daily portfolio returns.

    Returns:
        Annualized volatility as a decimal.
    """
    return float(port_returns.std(ddof=1) * np.sqrt(252))


def sharpe_ratio(port_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute the annualized Sharpe ratio (default risk-free rate = 0).

    Args:
        port_returns: Daily portfolio returns.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Sharpe ratio; 0.0 if volatility is negligible.
    """
    vol = annualized_volatility(port_returns)
    if vol < 1e-12:
        return 0.0
    return (annualized_return(port_returns) - risk_free_rate) / vol


def max_drawdown(port_returns: pd.Series) -> float:
    """Compute the maximum drawdown of a daily-return series.

    Args:
        port_returns: Daily portfolio returns.

    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.25 for -25%).
    """
    wealth = (1.0 + port_returns).cumprod()
    running_max = wealth.cummax()
    drawdowns = wealth / running_max - 1.0
    return float(drawdowns.min())


def performance_summary(port_returns: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side performance metrics table.

    Args:
        port_returns: DataFrame of daily portfolio returns per strategy.

    Returns:
        DataFrame with rows = strategies and columns =
        Annualized Return, Annualized Volatility, Sharpe Ratio,
        Max Drawdown.
    """
    return pd.DataFrame(
        {
            "Annualized Return": port_returns.apply(annualized_return),
            "Annualized Volatility": port_returns.apply(annualized_volatility),
            "Sharpe Ratio": port_returns.apply(sharpe_ratio),
            "Max Drawdown": port_returns.apply(max_drawdown),
        }
    )

