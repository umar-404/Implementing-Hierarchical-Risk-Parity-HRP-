"""Core Hierarchical Risk Parity (HRP) engine.

Implements the algorithm described in:

    López de Prado, M. (2016). "Building Diversified Portfolios that
    Outperform Out-of-Sample." Journal of Portfolio Management.

Pipeline: covariance/correlation -> distance matrix -> hierarchical
clustering -> quasi-diagonalization -> recursive bisection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def get_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the covariance matrix of asset returns.

    Args:
        returns: DataFrame of asset returns (rows = time).

    Returns:
        Covariance matrix as a DataFrame.
    """
    return returns.cov()


def get_corr(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix of asset returns.

    Args:
        returns: DataFrame of asset returns (rows = time).

    Returns:
        Correlation matrix as a DataFrame.
    """
    return returns.corr()


def get_distance_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix into a distance matrix.

    Implements Lopez de Prado's correlation distance:
    ``D_ij = sqrt(0.5 * (1 - rho_ij))``.

    Args:
        corr: Correlation matrix.

    Returns:
        Distance matrix (zeros on the diagonal).
    """
    arr = corr.to_numpy(dtype=float, copy=True)
    dvals = np.sqrt(np.clip(0.5 * (1.0 - arr), 0.0, None))
    np.fill_diagonal(dvals, 0.0)
    return pd.DataFrame(dvals, index=corr.index, columns=corr.columns)


def cluster_correlation(distance_matrix: pd.DataFrame, method: str = "single") -> np.ndarray:
    """Run hierarchical clustering on the distance matrix.

    Args:
        distance_matrix: Condensed-form-ready square distance matrix.
        method: Linkage method (``'single'``, ``'ward'``, ``'average'`` ...).

    Returns:
        The scipy linkage matrix (one row per merge).
    """
    # Clip tiny negatives from floating point before condensing.
    sym = distance_matrix.values.astype(float)
    sym = (sym + sym.T) / 2.0
    np.fill_diagonal(sym, 0.0)
    condensed = squareform(sym, checks=False)
    return linkage(condensed, method=method)


def get_quasi_diag(link: np.ndarray) -> list[int]:
    """Quasi-diagonalize: return the sorted item order from the linkage.

    Items that are highly correlated become adjacent in the ordering.

    Args:
        link: Linkage matrix from :func:`scipy.cluster.hierarchy.linkage`.

    Returns:
        Ordered list of original column indices.
    """

    n_obs = len(link) + 1

    def _get_ivp_order(node: float | int) -> list[int]:
        if int(node) < n_obs:  # Original observation
            return [int(node)]
        row = int(node) - n_obs
        left = int(link[row, 0])
        right = int(link[row, 1])
        return _get_ivp_order(left) + _get_ivp_order(right)

    root = len(link) + n_obs - 1  # Last merge row is the dendrogram root
    return _get_ivp_order(root)


def get_cluster_variance(cov: pd.DataFrame, assets: list[str]) -> float:
    """Compute the variance of a cluster using inverse-variance weights.

    ``cluster_variance = w' Sigma w`` with ``w`` proportional to the
    inverse diagonal of the sub-covariance (IVP within the cluster).

    Args:
        cov: Full covariance matrix.
        assets: Subset of asset names defining the cluster.

    Returns:
        The cluster's variance. Zero-variance assets are handled by
        equal weighting within the cluster.
    """
    sub = cov.loc[assets, assets].values.astype(float)
    diag = np.diag(sub)
    # Handle zero/near-zero variance: fall back to equal weights.
    safe_inv = np.where(diag > 1e-16, 1.0 / diag, 0.0)
    if safe_inv.sum() <= 0:
        w = np.full(len(assets), 1.0 / len(assets))
    else:
        w = safe_inv / safe_inv.sum()
    return float(w @ sub @ w)


def get_rec_bipart(cov: pd.DataFrame, sort_ix: list[str]) -> pd.Series:
    """Recursive bisection weight allocation on a quasi-diagonal order.

    Repeatedly splits the ordered list of assets into two halves,
    allocates capital between the halves in proportion to the inverse
    of their cluster variances, and recurses until single assets remain.

    Args:
        cov: Covariance matrix indexed by asset name.
        sort_ix: Quasi-diagonalized asset order (list of asset names).

    Returns:
        Series of portfolio weights summing to 1.0.
    """
    weights = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        # Bisection: split every cluster into halves.
        clusters = [
            cluster[j:k]
            for cluster in clusters
            for j, k in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
            if len(cluster) > 1
        ]
        for i in range(0, len(clusters), 2):
            first = clusters[i]
            second = clusters[i + 1]
            var_first = get_cluster_variance(cov, first)
            var_second = get_cluster_variance(cov, second)
            denom = var_first + var_second
            if denom <= 0:  # Both clusters degenerate -> split evenly
                alpha = 0.5
            else:
                alpha = 1.0 - var_first / denom
            weights[first] *= alpha
            weights[second] *= 1.0 - alpha
    return weights


def get_hrp_weights(returns: pd.DataFrame, linkage_method: str = "single") -> pd.Series:
    """Compute full HRP portfolio weights from asset returns.

    Args:
        returns: DataFrame of daily asset returns (rows = time).
        linkage_method: Linkage method for hierarchical clustering.

    Returns:
        Series of HRP weights indexed by asset name, summing to 1.0.
    """
    corr = get_corr(returns)
    cov = get_cov(returns)
    distance = get_distance_matrix(corr)
    link = cluster_correlation(distance, method=linkage_method)
    sort_ix = [returns.columns[i] for i in get_quasi_diag(link)]
    weights = get_rec_bipart(cov, sort_ix)
    return weights.reindex(returns.columns)


def plot_dendrogram(
    returns: pd.DataFrame,
    linkage_method: str = "single",
    ax=None,
    **kwargs,
):
    """Plot the hierarchical clustering dendrogram for the assets.

    Args:
        returns: DataFrame of daily asset returns.
        linkage_method: Linkage method for hierarchical clustering.
        ax: Optional matplotlib axes to draw on (e.g., in Streamlit).
        **kwargs: Extra keyword args passed to ``scipy.dendrogram``.

    Returns:
        The matplotlib Axes containing the dendrogram.
    """
    import matplotlib.pyplot as plt

    corr = get_corr(returns)
    distance = get_distance_matrix(corr)
    link = cluster_correlation(distance, method=linkage_method)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    dendrogram(
        link,
        labels=list(returns.columns),
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=10,
        **kwargs,
    )
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_ylabel("Correlation Distance")
    ax.set_xlabel("Asset")
    return ax

