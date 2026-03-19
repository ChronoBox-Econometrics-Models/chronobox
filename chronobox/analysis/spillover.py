"""
Diebold-Yilmaz Spillover Index based on Generalized FEVD.

Measures connectedness in VAR systems using forecast error variance decomposition.
Supports total, directional (FROM/TO), net, and pairwise spillover indices,
as well as rolling window analysis for time-varying connectedness.

References
----------
Diebold, F. X., & Yilmaz, K. (2009). Measuring financial asset return and volatility
    spillovers, with application to global equity markets. Economic Journal, 119, 158-171.

Diebold, F. X., & Yilmaz, K. (2012). Better to give than to receive: Predictive
    directional measurement of volatility spillovers. International Journal of
    Forecasting, 28(1), 57-66.

Pesaran, M. H., & Shin, Y. (1998). Generalized impulse response analysis in
    linear multivariate models. Economics Letters, 58(1), 17-29.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class SpilloverResult:
    """Result from Diebold-Yilmaz spillover analysis.

    Attributes
    ----------
    fevd_table : ndarray, shape (K, K)
        Normalized generalized FEVD table (tilde_theta_ij).
        Row i, column j = contribution of shock j to FEV of variable i.
        Each row sums to 1.
    total_spillover : float
        Total spillover index S(H). Range [0, 100].
    directional_from : ndarray, shape (K,)
        Directional FROM spillover: how much variable i receives from others.
    directional_to : ndarray, shape (K,)
        Directional TO spillover: how much variable i transmits to others.
    net_spillover : ndarray, shape (K,)
        Net spillover: TO - FROM. Positive = net transmitter.
    pairwise_spillover : ndarray, shape (K, K)
        Pairwise net spillover. S_ij = -S_ji.
    horizon : int
        FEVD horizon used.
    var_lags : int
        VAR lag order used.
    """

    fevd_table: np.ndarray
    total_spillover: float
    directional_from: np.ndarray
    directional_to: np.ndarray
    net_spillover: np.ndarray
    pairwise_spillover: np.ndarray
    horizon: int
    var_lags: int

    def summary(self) -> str:
        """Return text summary of spillover results."""
        K = len(self.directional_from)
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("Diebold-Yilmaz Spillover Index")
        lines.append(f"Horizon: {self.horizon}, VAR lags: {self.var_lags}")
        lines.append("=" * 60)
        lines.append(f"Total Spillover: {self.total_spillover:.2f}%")
        lines.append("-" * 60)

        # FEVD table
        header = "       " + "".join(f"  Var{j:>2}" for j in range(K)) + "  | FROM"
        lines.append(header)
        lines.append("-" * len(header))
        for i in range(K):
            row = f"Var{i:>2}: "
            for j in range(K):
                row += f"  {self.fevd_table[i, j]:5.2f}"
            row += f"  | {self.directional_from[i]:5.2f}"
            lines.append(row)

        lines.append("-" * len(header))
        to_row = "  TO:  "
        for j in range(K):
            to_row += f"  {self.directional_to[j]:5.2f}"
        lines.append(to_row)

        net_row = " NET:  "
        for j in range(K):
            net_row += f"  {self.net_spillover[j]:5.2f}"
        lines.append(net_row)

        lines.append("=" * 60)
        return "\n".join(lines)

    def plot_table(self) -> None:
        """Plot the spillover table as a heatmap (requires matplotlib)."""
        import matplotlib.pyplot as plt

        K = self.fevd_table.shape[0]
        _fig, ax = plt.subplots(figsize=(8, 6))  # type: ignore[reportUnknownMemberType]
        im = ax.imshow(self.fevd_table, cmap="YlOrRd", aspect="auto")  # type: ignore[reportUnknownMemberType]
        ax.set_xticks(range(K))  # type: ignore[reportUnknownMemberType]
        ax.set_yticks(range(K))  # type: ignore[reportUnknownMemberType]
        ax.set_xticklabels([f"Var {j}" for j in range(K)])  # type: ignore[reportUnknownMemberType]
        ax.set_yticklabels([f"Var {i}" for i in range(K)])  # type: ignore[reportUnknownMemberType]
        ax.set_title(f"Spillover Table (Total: {self.total_spillover:.1f}%)")  # type: ignore[reportUnknownMemberType]
        plt.colorbar(im, ax=ax)  # type: ignore[reportUnknownMemberType]
        plt.tight_layout()  # type: ignore[reportUnknownMemberType]
        plt.show()  # type: ignore[reportUnknownMemberType]


@dataclass
class RollingSpilloverResult:
    """Result from rolling window spillover analysis.

    Attributes
    ----------
    total_spillover : ndarray, shape (n_windows,)
        Time-varying total spillover index.
    directional_from : ndarray, shape (n_windows, K)
        Time-varying directional FROM spillover.
    directional_to : ndarray, shape (n_windows, K)
        Time-varying directional TO spillover.
    net_spillover : ndarray, shape (n_windows, K)
        Time-varying net spillover.
    window_size : int
        Rolling window size.
    dates : ndarray or None
        Date indices for the windows (end of each window).
    """

    total_spillover: np.ndarray
    directional_from: np.ndarray
    directional_to: np.ndarray
    net_spillover: np.ndarray
    window_size: int
    dates: np.ndarray | None = None

    def plot_total(self) -> None:
        """Plot time-varying total spillover."""
        import matplotlib.pyplot as plt

        _fig, ax = plt.subplots(figsize=(12, 5))  # type: ignore[reportUnknownMemberType]
        x = self.dates if self.dates is not None else np.arange(len(self.total_spillover))
        ax.plot(x, self.total_spillover, linewidth=1.5)  # type: ignore[reportUnknownMemberType]
        ax.set_title("Total Spillover Index (Rolling)")  # type: ignore[reportUnknownMemberType]
        ax.set_ylabel("Spillover (%)")  # type: ignore[reportUnknownMemberType]
        ax.grid(True, alpha=0.3)  # type: ignore[reportUnknownMemberType]
        plt.tight_layout()  # type: ignore[reportUnknownMemberType]
        plt.show()  # type: ignore[reportUnknownMemberType]


def _fit_var_ols(data: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit VAR(p) via OLS. Returns (coefficients, residuals, sigma).

    Parameters
    ----------
    data : ndarray, shape (T, K)
        Multivariate time series.
    lags : int
        Number of lags.

    Returns
    -------
    A_list : ndarray, shape (lags, K, K)
        VAR coefficient matrices A_1, ..., A_p.
    residuals : ndarray, shape (T-lags, K)
        OLS residuals.
    sigma : ndarray, shape (K, K)
        Residual covariance matrix.
    """
    T, K = data.shape
    n = T - lags

    # Dependent: Y[lags:]
    Y = data[lags:]  # (n, K)

    # Regressors: constant + lags
    X = np.ones((n, 1 + lags * K))
    for lag in range(1, lags + 1):
        X[:, 1 + (lag - 1) * K : 1 + lag * K] = data[lags - lag : T - lag]

    # OLS: B = (X'X)^{-1} X'Y
    B = np.linalg.lstsq(X, Y, rcond=None)[0]  # (1+lags*K, K)

    residuals = Y - X @ B
    sigma = (residuals.T @ residuals) / n

    # Extract A matrices
    A_list = np.zeros((lags, K, K))
    for lag in range(lags):
        A_list[lag] = B[1 + lag * K : 1 + (lag + 1) * K].T

    return A_list, residuals, sigma


def _compute_vma_coefficients(
    A_list: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Compute VMA coefficients Phi_0, Phi_1, ..., Phi_{H-1} from VAR coefficients.

    Parameters
    ----------
    A_list : ndarray, shape (p, K, K)
        VAR coefficient matrices.
    horizon : int
        Number of VMA coefficients to compute.

    Returns
    -------
    Phi : ndarray, shape (horizon, K, K)
        VMA coefficient matrices.
    """
    p, K, _ = A_list.shape
    Phi = np.zeros((horizon, K, K))
    Phi[0] = np.eye(K)

    for h in range(1, horizon):
        for j in range(1, min(h, p) + 1):
            Phi[h] += A_list[j - 1] @ Phi[h - j]

    return Phi


def _generalized_fevd(
    Phi: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Compute Generalized FEVD (Pesaran-Shin) and normalize.

    Parameters
    ----------
    Phi : ndarray, shape (H, K, K)
        VMA coefficient matrices.
    sigma : ndarray, shape (K, K)
        Residual covariance matrix.

    Returns
    -------
    table : ndarray, shape (K, K)
        Normalized GFEVD table. Row i, col j = contribution of shock j to FEV of i.
        Each row sums to 1.
    """
    H, K, _ = Phi.shape
    theta = np.zeros((K, K))

    sigma_diag = np.diag(sigma)

    for i in range(K):
        e_i = np.zeros(K)
        e_i[i] = 1.0

        # Denominator: sum_{h=0}^{H-1} e_i' * Phi_h * Sigma * Phi_h' * e_i
        denom = 0.0
        for h in range(H):
            val: float = e_i @ Phi[h] @ sigma @ Phi[h].T @ e_i
            denom += val

        for j in range(K):
            e_j = np.zeros(K)
            e_j[j] = 1.0

            # Numerator: sigma_jj^{-1} * sum_{h=0}^{H-1} (e_i' * Phi_h * Sigma * e_j)^2
            numer = 0.0
            for h in range(H):
                val_n: float = e_i @ Phi[h] @ sigma @ e_j
                numer += val_n**2

            numer /= sigma_diag[j]
            theta[i, j] = numer / denom if denom > 0 else 0.0

    # Normalize rows to sum to 1
    row_sums = theta.sum(axis=1, keepdims=True)
    table: np.ndarray = theta / row_sums

    return table


def _compute_spillover_measures(
    table: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all spillover measures from the normalized GFEVD table.

    Returns (total, from, to, net, pairwise).
    """
    K = table.shape[0]

    # Directional FROM: sum of off-diagonal in row i, divided by K, times 100
    diag_vals = np.diag(table)
    from_spillover = ((table.sum(axis=1) - diag_vals) / K) * 100

    # Directional TO: sum of off-diagonal in column j, divided by K, times 100
    to_spillover = ((table.sum(axis=0) - diag_vals) / K) * 100

    # Total: sum of all off-diagonal, divided by K, times 100
    total = float((table.sum() - np.trace(table)) / K * 100)

    # Net: TO - FROM
    net = to_spillover - from_spillover

    # Pairwise: (theta_ji - theta_ij) / K * 100
    pairwise = (table.T - table) / K * 100

    return total, from_spillover, to_spillover, net, pairwise


class SpilloverIndex:
    """Diebold-Yilmaz Spillover Index based on Generalized FEVD.

    Parameters
    ----------
    lags : int
        VAR lag order. Default is 2.
    horizon : int
        FEVD forecast horizon. Default is 10.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(500, 4)
    >>> sp = SpilloverIndex(lags=2, horizon=10)
    >>> result = sp.fit(data)
    >>> print(f"Total spillover: {result.total_spillover:.1f}%")
    >>> print(result.summary())
    """

    def __init__(self, lags: int = 2, horizon: int = 10) -> None:
        if lags < 1:
            raise ValueError(f"lags must be >= 1, got {lags}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.lags = lags
        self.horizon = horizon

    def fit(self, data: ArrayLike) -> SpilloverResult:
        """Compute spillover index for the full sample.

        Parameters
        ----------
        data : array_like, shape (T, K)
            Multivariate time series with K variables.

        Returns
        -------
        SpilloverResult
            Full spillover analysis results.
        """
        data_arr = np.asarray(data, dtype=np.float64)
        if data_arr.ndim != 2:
            raise ValueError(f"data must be 2-D, got shape {data_arr.shape}")

        T: int = int(data_arr.shape[0])
        K: int = int(data_arr.shape[1])
        if self.lags + self.horizon >= T:
            raise ValueError(
                f"Not enough observations: T={T}, need > lags+horizon={self.lags + self.horizon}"
            )
        if K < 2:
            raise ValueError(f"Need at least 2 variables, got K={K}")

        # Fit VAR
        A_list, _residuals, sigma = _fit_var_ols(data_arr, self.lags)

        # VMA coefficients
        Phi = _compute_vma_coefficients(A_list, self.horizon)

        # Generalized FEVD
        table = _generalized_fevd(Phi, sigma)

        # Spillover measures
        total, from_sp, to_sp, net_sp, pairwise = _compute_spillover_measures(table)

        return SpilloverResult(
            fevd_table=table,
            total_spillover=total,
            directional_from=from_sp,
            directional_to=to_sp,
            net_spillover=net_sp,
            pairwise_spillover=pairwise,
            horizon=self.horizon,
            var_lags=self.lags,
        )

    def rolling(
        self,
        data: ArrayLike,
        window: int = 200,
    ) -> RollingSpilloverResult:
        """Compute rolling window spillover analysis.

        Parameters
        ----------
        data : array_like, shape (T, K)
            Multivariate time series.
        window : int
            Rolling window size. Default is 200.

        Returns
        -------
        RollingSpilloverResult
            Time-varying spillover results.
        """
        data_arr = np.asarray(data, dtype=np.float64)
        if data_arr.ndim != 2:
            raise ValueError(f"data must be 2-D, got shape {data_arr.shape}")

        T: int = int(data_arr.shape[0])
        K: int = int(data_arr.shape[1])
        if window > T:
            raise ValueError(f"window={window} > T={T}")
        if window <= self.lags + self.horizon + K:
            raise ValueError(
                f"window={window} too small for lags={self.lags}, "
                f"horizon={self.horizon}, K={K}"
            )

        n_windows = T - window + 1
        total_arr = np.zeros(n_windows)
        from_arr = np.zeros((n_windows, K))
        to_arr = np.zeros((n_windows, K))
        net_arr = np.zeros((n_windows, K))

        for w in range(n_windows):
            sub_data = data_arr[w : w + window]
            try:
                A_list, _, sigma = _fit_var_ols(sub_data, self.lags)
                Phi = _compute_vma_coefficients(A_list, self.horizon)
                table = _generalized_fevd(Phi, sigma)
                total, from_sp, to_sp, net_sp, _ = _compute_spillover_measures(table)

                total_arr[w] = total
                from_arr[w] = from_sp
                to_arr[w] = to_sp
                net_arr[w] = net_sp
            except (np.linalg.LinAlgError, ValueError):
                total_arr[w] = np.nan
                from_arr[w] = np.nan
                to_arr[w] = np.nan
                net_arr[w] = np.nan

        dates: np.ndarray = np.arange(window - 1, T)

        return RollingSpilloverResult(
            total_spillover=total_arr,
            directional_from=from_arr,
            directional_to=to_arr,
            net_spillover=net_arr,
            window_size=window,
            dates=dates,
        )
