"""Quick sanity tests for the HRP engine and backtester.

Run with:  python3 test_hrp.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import compute_log_returns
from backtester import (
    walk_forward_backtest,
    performance_summary,
    cumulative_returns,
    equal_weight_weights,
    min_variance_weights,
)
from hrp_engine import (
    get_corr,
    get_cov,
    get_distance_matrix,
    get_quasi_diag,
    get_rec_bipart,
    get_hrp_weights,
    cluster_correlation,
    plot_dendrogram,
)


def synthetic_returns(n_days: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # 6 assets: 3 highly correlated, 3 independent -> clusters should form.
    common = rng.normal(0, 0.01, n_days)
    data = {
        "A1": common + rng.normal(0, 0.005, n_days),
        "A2": common + rng.normal(0, 0.005, n_days),
        "A3": common + rng.normal(0, 0.005, n_days),
        "B1": rng.normal(0.0003, 0.012, n_days),
        "B2": rng.normal(0.0002, 0.015, n_days),
        "B3": rng.normal(0.0001, 0.018, n_days),
    }
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(data, index=idx)


def test_distance_matrix() -> None:
    r = synthetic_returns(300)
    corr = get_corr(r)
    d = get_distance_matrix(corr)
    assert d.shape == corr.shape
    assert np.allclose(np.diag(d), 0.0)
    assert (d.values >= 0).all()
    # Correlated pair -> much smaller distance than independent pair
    assert d.loc["A1", "A2"] < d.loc["A1", "B2"] * 0.75
    print("test_distance_matrix: OK")


def test_weights_sum_to_one() -> None:
    r = synthetic_returns(600)
    w = get_hrp_weights(r)
    assert np.isclose(w.sum(), 1.0), w.sum()
    assert (w >= 0).all()
    assert len(w) == r.shape[1]
    print("test_weights_sum_to_one: OK")


def test_collinearity_edge_case() -> None:
    # Duplicate asset creates a singular correlation/covariance matrix.
    r = synthetic_returns(400)
    r["A1_dup"] = r["A1"]
    w = get_hrp_weights(r)
    assert np.isclose(w.sum(), 1.0)
    assert np.isfinite(w).all()
    mv = min_variance_weights(r)
    assert np.isclose(mv.sum(), 1.0)
    assert np.isfinite(mv).all()
    print("test_collinearity_edge_case: OK")


def test_quasi_diag() -> None:
    r = synthetic_returns(500)
    link = cluster_correlation(get_distance_matrix(get_corr(r)))
    order = get_quasi_diag(link)
    assert sorted(order) == list(range(r.shape[1]))
    print("test_quasi_diag: OK")


def test_backtest() -> None:
    r = synthetic_returns(1000)
    result = walk_forward_backtest(
        r, rebalance_days=252, lookback_days=504,
        strategies=("HRP", "Equal-Weight", "Min-Variance"),
    )
    pr = result["returns"]
    assert not pr.empty
    assert np.isfinite(pr.values).all()
    summary = performance_summary(pr)
    assert list(summary.index) == ["HRP", "Equal-Weight", "Min-Variance"]
    assert list(summary.columns) == [
        "Annualized Return", "Annualized Volatility",
        "Sharpe Ratio", "Max Drawdown",
    ]
    cum = cumulative_returns(pr)
    assert (cum > 0).all().all()
    # Weights at each rebalance must sum to 1
    for s, wdf in result["weights"].items():
        assert np.allclose(wdf.sum(axis=1), 1.0), s
    print(summary.round(4))
    print("test_backtest: OK")


def test_real_data_smoke() -> None:
    try:
        from data_loader import fetch_data
        rets = fetch_data(["AAPL", "MSFT", "JPM"], "2023-01-01", "2024-06-01")
    except Exception as exc:  # network may be unavailable
        print(f"test_real_data_smoke: SKIPPED ({exc})")
        return
    w = get_hrp_weights(rets)
    assert np.isclose(w.sum(), 1.0)
    ax = plot_dendrogram(rets)
    assert ax is not None
    print("test_real_data_smoke: OK")


if __name__ == "__main__":
    test_distance_matrix()
    test_weights_sum_to_one()
    test_collinearity_edge_case()
    test_quasi_diag()
    test_backtest()
    test_real_data_smoke()
    print("\nAll tests passed.")
