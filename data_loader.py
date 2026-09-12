"""Data loading utilities for the HRP portfolio project.

Fetches adjusted close prices from Yahoo Finance via ``yfinance``,
computes daily log returns, and cleans missing values.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

#: Default liquid multi-sector basket used when no tickers are provided.
DEFAULT_TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "JNJ", "XOM", "PG", "V",
]


def fetch_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch adjusted close prices for the given tickers.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date in ``YYYY-MM-DD`` format.
        end_date: End date in ``YYYY-MM-DD`` format.

    Returns:
        DataFrame of adjusted close prices indexed by trading date,
        with missing values forward-filled and rows of all-NaN dropped.

    Raises:
        ValueError: If no price data could be retrieved for any ticker.
    """
    logger.info("Fetching price data for %d tickers: %s", len(tickers), tickers)
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if data is None or data.empty:
        raise ValueError(
            f"No price data retrieved for tickers {tickers} "
            f"between {start_date} and {end_date}."
        )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:  # Single ticker returns a flat column index
        prices = data[["Close"]]
        prices.columns = tickers

    # Keep only successfully downloaded tickers.
    prices = prices.dropna(axis=1, how="all")
    # Forward-fill short gaps, then drop leading rows without history.
    prices = prices.ffill().dropna(how="any")
    if prices.empty:
        raise ValueError("Price data empty after cleaning.")
    logger.info("Retrieved %d trading days of prices.", len(prices))
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price DataFrame.

    Args:
        prices: DataFrame of adjusted close prices.

    Returns:
        DataFrame of daily log returns with leading NaN rows removed.
    """
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="any")


def fetch_data(
    tickers: list[str] | None = None,
    start_date: str = "2018-01-01",
    end_date: str = "2026-12-31",
) -> pd.DataFrame:
    """Fetch prices and return cleaned daily log returns.

    Args:
        tickers: Ticker symbols; defaults to :data:`DEFAULT_TICKERS`.
        start_date: Start date in ``YYYY-MM-DD`` format.
        end_date: End date in ``YYYY-MM-DD`` format.

    Returns:
        DataFrame of daily log returns.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    prices = fetch_prices(tickers, start_date, end_date)
    return compute_log_returns(prices)
