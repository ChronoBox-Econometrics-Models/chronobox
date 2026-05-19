"""
Data generators for advanced model examples (FAVAR, TVP-VAR, GVAR).

Provides functions to generate synthetic factor models, time-varying
parameter VARs, and multi-country panel data with fixed seeds.
"""

import numpy as np
import pandas as pd


def generate_factor_model(
    n: int = 200,
    n_series: int = 10,
    n_factors: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a factor model for FAVAR estimation.

    Generates observable series X driven by latent factors F:
        X_t = Lambda * F_t + e_t
    where factors follow a VAR(1):
        F_t = Phi * F_{t-1} + u_t

    Parameters
    ----------
    n : int
        Number of observations.
    n_series : int
        Number of observable series (informational variables).
    n_factors : int
        Number of latent factors.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (X, F, Lambda)
        X : np.ndarray of shape (n, n_series) - observable series
        F : np.ndarray of shape (n, n_factors) - latent factors
        Lambda : np.ndarray of shape (n_series, n_factors) - factor loadings
    """
    rng = np.random.default_rng(seed)

    # Factor loadings: each series loads on factors with different weights
    Lambda = rng.normal(0, 1, size=(n_series, n_factors))

    # Factor dynamics: VAR(1) with stable eigenvalues
    # Diagonal with moderate persistence + small off-diagonal
    Phi = np.diag(rng.uniform(0.5, 0.8, n_factors))
    off_diag = rng.normal(0, 0.05, size=(n_factors, n_factors))
    np.fill_diagonal(off_diag, 0)
    Phi += off_diag

    # Scale to ensure stationarity
    max_eig = np.max(np.abs(np.linalg.eigvals(Phi)))
    if max_eig >= 0.95:
        Phi *= 0.9 / max_eig

    # Generate factors as VAR(1)
    burn = 200
    total = n + burn
    factor_innovations = rng.normal(0, 0.5, size=(total, n_factors))
    F_full = np.zeros((total, n_factors))

    for t in range(1, total):
        F_full[t] = Phi @ F_full[t - 1] + factor_innovations[t]

    F = F_full[burn:]

    # Generate observables
    idiosyncratic = rng.normal(0, 0.3, size=(n, n_series))
    X = F @ Lambda.T + idiosyncratic

    return X, F, Lambda


def generate_tvp_var(
    n: int = 200,
    k: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a VAR with time-varying parameters.

    Generates a bivariate (or k-variate) VAR(1) where coefficients
    evolve as random walks:
        y_t = A_t * y_{t-1} + u_t
        vec(A_t) = vec(A_{t-1}) + eta_t

    Parameters
    ----------
    n : int
        Number of observations.
    k : int
        Number of variables.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (Y, A_path)
        Y : np.ndarray of shape (n, k) - generated series
        A_path : np.ndarray of shape (n, k, k) - time-varying coefficients
    """
    rng = np.random.default_rng(seed)

    burn = 200
    total = n + burn

    # Initial coefficient matrix (moderate persistence)
    A_init = np.array([[0.5, 0.1], [0.2, 0.3]]) if k == 2 else (
        np.diag(rng.uniform(0.3, 0.6, k))
        + rng.normal(0, 0.05, size=(k, k)) * (1 - np.eye(k))
    )

    # Innovation variances
    sigma_y = np.eye(k) * 0.5  # observation equation
    sigma_a = np.eye(k * k) * 0.001  # state equation (small random walk)

    # Generate time-varying coefficients
    A_path_full = np.zeros((total, k, k))
    A_path_full[0] = A_init

    for t in range(1, total):
        a_vec = A_path_full[t - 1].flatten()
        eta = rng.multivariate_normal(np.zeros(k * k), sigma_a)
        a_new = a_vec + eta
        A_new = a_new.reshape(k, k)

        # Enforce stationarity: shrink if eigenvalues too large
        max_eig = np.max(np.abs(np.linalg.eigvals(A_new)))
        if max_eig >= 0.95:
            A_new *= 0.9 / max_eig

        A_path_full[t] = A_new

    # Generate observations
    Y_full = np.zeros((total, k))
    for t in range(1, total):
        u = rng.multivariate_normal(np.zeros(k), sigma_y)
        Y_full[t] = A_path_full[t] @ Y_full[t - 1] + u

    Y = Y_full[burn:]
    A_path = A_path_full[burn:]

    return Y, A_path


def generate_gvar_data(
    n: int = 80,
    n_countries: int = 5,
    k: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-country panel data for GVAR estimation.

    Generates a panel dataset with n_countries, each with k variables
    that are interconnected through a trade-weight matrix.

    Variables: gdp, inflation, interest_rate, unemployment

    Parameters
    ----------
    n : int
        Number of time periods (quarterly observations).
    n_countries : int
        Number of countries.
    k : int
        Number of variables per country.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel DataFrame with columns: date, country, gdp, inflation,
        interest_rate, unemployment.
    """
    rng = np.random.default_rng(seed)

    country_names = ["US", "UK", "DE", "JP", "BR"][:n_countries]
    var_names = ["gdp", "inflation", "interest_rate", "unemployment"][:k]

    # Trade weight matrix (row-stochastic, zero diagonal)
    W = rng.uniform(0.1, 1.0, size=(n_countries, n_countries))
    np.fill_diagonal(W, 0)
    W = W / W.sum(axis=1, keepdims=True)

    # Country-specific AR coefficients (domestic dynamics)
    A_domestic = []
    for i in range(n_countries):
        A = np.diag(rng.uniform(0.4, 0.7, k))
        # Small cross-variable effects
        A += rng.normal(0, 0.03, size=(k, k)) * (1 - np.eye(k))
        A_domestic.append(A)

    # Spillover strength
    spillover = 0.05

    # Base levels for each country (different means)
    base_levels = {
        "US": [2.5, 2.0, 3.0, 5.0],
        "UK": [2.0, 2.5, 2.5, 5.5],
        "DE": [1.5, 1.8, 1.5, 4.5],
        "JP": [1.0, 0.5, 0.5, 3.5],
        "BR": [3.0, 5.0, 8.0, 8.0],
    }

    burn = 200
    total = n + burn

    # Generate data: Y[t, i, :] = data for country i at time t
    Y = np.zeros((total, n_countries, k))

    for t in range(1, total):
        for i in range(n_countries):
            # Domestic dynamics
            domestic = A_domestic[i] @ Y[t - 1, i, :]

            # Foreign (weighted average of other countries)
            foreign = np.zeros(k)
            for j in range(n_countries):
                if j != i:
                    foreign += W[i, j] * Y[t - 1, j, :]

            # Innovation
            u = rng.normal(0, 0.3, k)

            Y[t, i, :] = domestic + spillover * foreign + u

    Y = Y[burn:]

    # Add base levels and construct DataFrame
    dates = pd.date_range(start="2004-01-01", periods=n, freq="QS")

    rows = []
    for t in range(n):
        for i, country in enumerate(country_names):
            base = base_levels.get(country, [2.0] * k)[:k]
            row = {"date": dates[t], "country": country}
            for v, var_name in enumerate(var_names):
                row[var_name] = round(base[v] + Y[t, i, v], 4)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df
