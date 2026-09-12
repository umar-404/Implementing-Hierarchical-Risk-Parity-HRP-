# Implementing Hierarchical Risk Parity (HRP)

![alt text](<Screenshot from 2026-09-13 02-45-49.png>)

A robust, production-ready Python implementation of Marcos López de Prado's **Hierarchical Risk Parity (HRP)** portfolio optimization algorithm. This project leverages graph theory, agglomerative hierarchical clustering, and recursive bisection to construct diversified asset allocations that avoid the mathematical instabilities of traditional mean-variance optimization. Includes a modular backtesting engine and an interactive Streamlit dashboard for performance visualization and dendrogram analysis.

> **This is an implementation project** — a faithful, working build of the HRP algorithm described in:
> López de Prado, M. (2016). *"Building Diversified Portfolios that Outperform Out-of-Sample."* Journal of Portfolio Management.

## Features

- **Core HRP engine** (`hrp_engine.py`): full 5-step pipeline
  1. Covariance (Σ) & Pearson correlation (ρ) matrices from daily log returns
  2. Correlation distance matrix: `D_ij = sqrt(0.5 * (1 - ρ_ij))`
  3. Agglomerative hierarchical clustering (`scipy.cluster.hierarchy`)
  4. Quasi-diagonalization (reordering assets so correlated assets are adjacent)
  5. Recursive bisection top-down weight allocation
- **Backtesting engine** (`backtester.py`): walk-forward simulation (annual / semi-annual / quarterly rebalancing) comparing HRP against **Equal-Weight (1/N)** and a **Min-Variance** benchmark, with Annualized Return, Annualized Volatility, Sharpe Ratio, and Max Drawdown
- **Streamlit dashboard** (`app.py`): interactive ticker/date/rebalance settings, cumulative-return comparison chart, side-by-side metrics table, HRP allocation, clustering dendrogram, and correlation heatmap — all exportable
- **Edge-case robustness**: handles zero-variance assets, collinear (singular) covariance matrices, and missing price data
- **Test suite** (`test_hrp.py`): sanity checks on the math pipeline, weights normalization, collinearity edge cases, and a live-data smoke test

## Project Structure

```
├── app.py            # Streamlit web dashboard
├── backtester.py     # Walk-forward backtesting engine & performance stats
├── data_loader.py    # yfinance data fetching & log-return computation
├── hrp_engine.py     # Core HRP algorithm (clustering -> bisection)
├── test_hrp.py       # Sanity tests & live-data smoke test
└── requirements.txt  # Dependencies
```

## Installation

```bash
git clone https://github.com/umar-404/Implementing-Hierarchical-Risk-Parity-HRP-.git
cd Implementing-Hierarchical-Risk-Parity-HRP-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the dashboard:

```bash
streamlit run app.py
```

Or use the modules programmatically:

```python
from data_loader import fetch_data
from hrp_engine import get_hrp_weights, plot_dendrogram
from backtester import walk_forward_backtest, performance_summary

returns = fetch_data(["AAPL", "MSFT", "JPM", "NVDA", "PG"], "2020-01-01", "2025-01-01")
weights = get_hrp_weights(returns)          # HRP allocation (sums to 1.0)
plot_dendrogram(returns)                   # Visualize asset clusters
result = walk_forward_backtest(returns)    # Walk-forward backtest vs benchmarks
print(performance_summary(result["returns"]))
```

Run the test suite:

```bash
python test_hrp.py
```

## How HRP Works

Traditional mean-variance optimization requires inverting the covariance matrix, which becomes ill-conditioned when assets are highly correlated — leading to unstable, concentrated portfolios. HRP sidesteps matrix inversion entirely:

1. **Cluster** assets by correlation distance so the structure of the market's correlation graph is respected.
2. **Quasi-diagonalize** the covariance matrix using the dendrogram ordering.
3. **Recursively bisect** the ordered list, splitting capital between sub-clusters in proportion to the inverse of their variances.

The result is a well-diversified, closed-form allocation with no dependence on expected returns — empirically more stable out-of-sample than Markowitz or standard risk parity.

## Disclaimer

This project is for **educational and research purposes only** and does not constitute investment advice.

## Author

Built by [umar-404](https://github.com/umar-404)
