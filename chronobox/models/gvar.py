"""
Global Vector Autoregression (GVAR).

Models interdependencies between multiple countries/regions
using bilateral trade weights to construct foreign variables.

References
----------
- Pesaran, M.H., Schuermann, T. & Weiner, S.M. (2004). JBES, 22(2), 129-162.
- Dees, S., di Mauro, F., Pesaran, M.H. & Smith, L.V. (2007). Journal of Applied
  Econometrics, 22(1), 1-38.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la


@dataclass
class GVARResults:
    """Results from GVAR estimation.

    Attributes
    ----------
    global_coefs : list of ndarray
        List of F_1, ..., F_p companion matrices of the global system.
        Each has shape (k_total, k_total).
    global_sigma : ndarray of shape (k_total, k_total)
        Global residual covariance matrix.
    global_intercept : ndarray of shape (k_total,)
        Global intercept.
    country_results : dict[str, dict]
        Per-country VARX estimation results.
    country_names : list[str]
        Names of countries.
    country_dims : dict[str, int]
        Number of variables per country.
    trade_weights : ndarray of shape (N, N)
        Trade weight matrix.
    k_total : int
        Total number of variables across all countries.
    n_countries : int
        Number of countries.
    n_obs : int
        Number of observations.
    domestic_lags : int
        Number of domestic lags.
    foreign_lags : int
        Number of foreign lags.
    is_stable : bool
        Whether the global system is stable.
    eigenvalues : ndarray
        Eigenvalues of the companion matrix.
    """

    global_coefs: list[NDArray[np.floating[Any]]]
    global_sigma: NDArray[np.floating[Any]]
    global_intercept: NDArray[np.floating[Any]]
    country_results: dict[str, dict[str, Any]]
    country_names: list[str]
    country_dims: dict[str, int]
    trade_weights: NDArray[np.floating[Any]]
    k_total: int
    n_countries: int
    n_obs: int
    domestic_lags: int
    foreign_lags: int
    is_stable: bool
    eigenvalues: NDArray[np.complexfloating[Any, Any]]

    def girf(
        self,
        shock_country: str | int,
        shock_var: int,
        periods: int = 40,
        shock_size: float | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Compute Generalized Impulse Response Function.

        Parameters
        ----------
        shock_country : str or int
            Country to shock (name or index).
        shock_var : int
            Variable index within the country to shock.
        periods : int
            Number of IRF periods.
        shock_size : float or None
            Size of shock. If None, uses 1 standard deviation.

        Returns
        -------
        ndarray of shape (periods+1, k_total)
            GIRF for all variables in the global system.
        """
        k = self.k_total
        n_steps = periods + 1

        # Resolve country index
        if isinstance(shock_country, str):
            country_idx = self.country_names.index(shock_country)
        else:
            country_idx = shock_country

        # Global variable index
        global_var_idx = 0
        for i, name in enumerate(self.country_names):
            if i == country_idx:
                break
            global_var_idx += self.country_dims[name]
        global_var_idx += shock_var

        # Shock size
        sigma_jj = self.global_sigma[global_var_idx, global_var_idx]
        delta = np.sqrt(sigma_jj) if shock_size is None else shock_size

        # MA coefficients
        p = self.domestic_lags
        phi = np.zeros((n_steps, k, k))
        phi[0] = np.eye(k)
        for h in range(1, n_steps):
            for s in range(1, min(h, p) + 1):
                if s - 1 < len(self.global_coefs):
                    phi[h] += phi[h - s] @ self.global_coefs[s - 1]

        # GIRF: phi_h * Sigma * e_j / sqrt(sigma_jj)
        e_j = np.zeros(k)
        e_j[global_var_idx] = 1.0

        result = np.zeros((n_steps, k))
        for h in range(n_steps):
            result[h] = (
                phi[h] @ self.global_sigma @ e_j * delta / np.sqrt(sigma_jj)
            )

        return result

    def irf_country(
        self,
        shock_country: str | int,
        shock_var: int,
        response_country: str | int,
        periods: int = 40,
    ) -> NDArray[np.floating[Any]]:
        """Extract GIRF for a specific response country.

        Parameters
        ----------
        shock_country : str or int
            Country originating the shock.
        shock_var : int
            Variable index within shock country.
        response_country : str or int
            Country whose response to extract.
        periods : int
            Number of periods.

        Returns
        -------
        ndarray of shape (periods+1, k_response)
        """
        girf_full = self.girf(shock_country, shock_var, periods)

        if isinstance(response_country, str):
            resp_idx = self.country_names.index(response_country)
        else:
            resp_idx = response_country

        # Find start/end indices for response country
        start = 0
        for i, name in enumerate(self.country_names):
            if i == resp_idx:
                break
            start += self.country_dims[name]
        resp_name = self.country_names[resp_idx]
        end = start + self.country_dims[resp_name]

        return girf_full[:, start:end]


class GVAR:
    """Global Vector Autoregression.

    Parameters
    ----------
    trade_weights : ndarray of shape (N, N)
        Bilateral trade weight matrix. trade_weights[i, j] is the weight
        of country j in constructing foreign variables for country i.
        Diagonal should be 0. Each row should sum to 1.
    domestic_lags : int
        Number of domestic lags (default=1).
    foreign_lags : int
        Number of foreign lags (default=1).
    """

    def __init__(
        self,
        trade_weights: NDArray[np.floating[Any]],
        domestic_lags: int = 1,
        foreign_lags: int = 1,
    ) -> None:
        self.trade_weights = np.asarray(trade_weights, dtype=float)
        self.domestic_lags = domestic_lags
        self.foreign_lags = foreign_lags
        self.n_countries = self.trade_weights.shape[0]

        # Validate trade weights
        if self.trade_weights.shape[0] != self.trade_weights.shape[1]:
            msg = "trade_weights must be square."
            raise ValueError(msg)

        # Normalize rows to sum to 1 (excluding diagonal)
        for i in range(self.n_countries):
            self.trade_weights[i, i] = 0.0
            row_sum = self.trade_weights[i].sum()
            if row_sum > 0:
                self.trade_weights[i] /= row_sum

    def fit(
        self,
        data_dict: dict[str, NDArray[np.floating[Any]]],
    ) -> GVARResults:
        """Fit GVAR model.

        Parameters
        ----------
        data_dict : dict[str, ndarray]
            Dictionary mapping country names to data arrays.
            Each array has shape (T, k_i) where k_i is the number of
            variables for country i. All must have the same T.

        Returns
        -------
        GVARResults
        """
        country_names = list(data_dict.keys())
        n = len(country_names)

        if self.n_countries != n:
            msg = f"Expected {self.n_countries} countries, got {n}."
            raise ValueError(msg)

        # Validate data
        t_values = [np.asarray(v).shape[0] for v in data_dict.values()]
        if len(set(t_values)) > 1:
            msg = "All countries must have the same number of observations."
            raise ValueError(msg)
        t_obs = t_values[0]

        # Country dimensions
        country_dims: dict[str, int] = {}
        for name in country_names:
            arr = np.asarray(data_dict[name], dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            data_dict[name] = arr
            country_dims[name] = arr.shape[1]

        k_total = sum(country_dims.values())

        # Stack global data vector y_t
        y_global = np.zeros((t_obs, k_total))
        country_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name in country_names:
            k_i = country_dims[name]
            y_global[:, offset : offset + k_i] = data_dict[name]
            country_slices[name] = (offset, offset + k_i)
            offset += k_i

        p = self.domestic_lags
        q = self.foreign_lags
        max_lag = max(p, q)
        t_eff = t_obs - max_lag

        # For each country, construct foreign variables and estimate VARX
        country_results: dict[str, dict[str, Any]] = {}
        link_matrices: list[NDArray[np.floating[Any]]] = []

        for i, name_i in enumerate(country_names):
            k_i = country_dims[name_i]
            start_i, end_i = country_slices[name_i]

            # Construct foreign variables: y*_{it} = sum(w_{ij} * y_{jt})
            y_star = np.zeros((t_obs, k_i))
            for j, name_j in enumerate(country_names):
                if i == j:
                    continue
                w_ij = self.trade_weights[i, j]
                if w_ij == 0:
                    continue
                k_j = country_dims[name_j]
                start_j, _end_j = country_slices[name_j]

                # Map foreign vars to same dimension as domestic
                k_min = min(k_i, k_j)
                y_star[:, :k_min] += (
                    w_ij * y_global[:, start_j : start_j + k_min]
                )

            y_i = data_dict[name_i]  # (t_obs, k_i)

            # VARX(p, q): y_{it} = const + Phi * y_{i,t-1:t-p}
            #              + Lambda * y*_{i,t:t-q} + u_it
            n_dom_regressors = k_i * p
            n_for_regressors = k_i * (q + 1)  # contemporaneous + q lags
            n_regressors = n_dom_regressors + n_for_regressors + 1

            y_dep = y_i[max_lag:]  # (t_eff, k_i)
            z_mat = np.zeros((t_eff, n_regressors))

            for t in range(t_eff):
                idx = 0
                # Domestic lags
                for s in range(1, p + 1):
                    z_mat[t, idx : idx + k_i] = y_i[max_lag + t - s]
                    idx += k_i
                # Foreign: contemporaneous + lags
                for s in range(0, q + 1):
                    z_mat[t, idx : idx + k_i] = y_star[max_lag + t - s]
                    idx += k_i
                # Intercept
                z_mat[t, -1] = 1.0

            # OLS
            try:
                b_i = la.solve(z_mat.T @ z_mat, z_mat.T @ y_dep)
            except la.LinAlgError:
                b_i = la.lstsq(z_mat, y_dep)[0]

            resid_i = y_dep - z_mat @ b_i
            sigma_i = (resid_i.T @ resid_i) / t_eff

            # Extract coefficient matrices
            idx = 0
            phi_list = []
            for _s in range(p):
                phi_s = b_i[idx : idx + k_i].T  # (k_i, k_i)
                phi_list.append(phi_s)
                idx += k_i

            lambda_list = []
            for _s in range(q + 1):
                lambda_s = b_i[idx : idx + k_i].T  # (k_i, k_i)
                lambda_list.append(lambda_s)
                idx += k_i

            intercept_i = b_i[-1]  # (k_i,)

            country_results[name_i] = {
                "Phi": phi_list,
                "Lambda": lambda_list,
                "Sigma": sigma_i,
                "intercept": intercept_i,
                "resid": resid_i,
            }

            # Build link matrix w_i: maps y_global to [y_i; y*_i]
            w_link = np.zeros((2 * k_i, k_total))
            # Domestic part
            w_link[:k_i, start_i:end_i] = np.eye(k_i)
            # Foreign part
            for j, name_j in enumerate(country_names):
                if i == j:
                    continue
                w_ij = self.trade_weights[i, j]
                if w_ij == 0:
                    continue
                k_j = country_dims[name_j]
                start_j, _ = country_slices[name_j]
                k_min = min(k_i, k_j)
                w_link[k_i : k_i + k_min, start_j : start_j + k_min] += (
                    w_ij * np.eye(k_min)
                )

            link_matrices.append(w_link)

        # Build global system
        # g_0 * y_t = c + g_1 * y_{t-1} + ... + u_t
        g_0 = np.zeros((k_total, k_total))
        g_lags: list[NDArray[np.floating[Any]]] = [
            np.zeros((k_total, k_total)) for _ in range(p)
        ]
        c_global = np.zeros(k_total)
        sigma_global = np.zeros((k_total, k_total))

        offset = 0
        for i, name_i in enumerate(country_names):
            k_i = country_dims[name_i]
            w_link = link_matrices[i]
            cr = country_results[name_i]

            # a_{i0} * w_i * y_t = ... (contemporaneous)
            a_i0 = np.zeros((k_i, 2 * k_i))
            a_i0[:, :k_i] = np.eye(k_i)
            a_i0[:, k_i:] = -cr["Lambda"][0]

            g_0[offset : offset + k_i, :] = a_i0 @ w_link

            # Lag matrices
            for s in range(p):
                a_is = np.zeros((k_i, 2 * k_i))
                a_is[:, :k_i] = cr["Phi"][s]
                if s < q:
                    a_is[:, k_i:] = cr["Lambda"][s + 1]
                g_lags[s][offset : offset + k_i, :] = a_is @ w_link

            c_global[offset : offset + k_i] = cr["intercept"]
            sigma_global[offset : offset + k_i, offset : offset + k_i] = cr[
                "Sigma"
            ]

            offset += k_i

        # Solve: y_t = g_0^{-1} * c + g_0^{-1} * g_1 * y_{t-1} + ...
        try:
            g_0_inv = la.inv(g_0)
        except la.LinAlgError:
            g_0_inv = la.inv(g_0 + np.eye(k_total) * 1e-8)

        global_intercept = g_0_inv @ c_global
        global_coefs = [g_0_inv @ g_s for g_s in g_lags]
        global_sigma = g_0_inv @ sigma_global @ g_0_inv.T
        global_sigma = (global_sigma + global_sigma.T) / 2.0

        # Check stability: eigenvalues of companion matrix
        if p == 1:
            eigenvalues = la.eigvals(global_coefs[0])
        else:
            # Build companion matrix
            companion = np.zeros((k_total * p, k_total * p))
            for s in range(p):
                companion[:k_total, s * k_total : (s + 1) * k_total] = (
                    global_coefs[s]
                )
            if p > 1:
                companion[k_total:, : k_total * (p - 1)] = np.eye(
                    k_total * (p - 1)
                )
            eigenvalues = la.eigvals(companion)

        is_stable = bool(np.all(np.abs(eigenvalues) < 1.0))

        return GVARResults(
            global_coefs=global_coefs,
            global_sigma=global_sigma,
            global_intercept=global_intercept,
            country_results=country_results,
            country_names=country_names,
            country_dims=country_dims,
            trade_weights=self.trade_weights,
            k_total=k_total,
            n_countries=n,
            n_obs=t_eff,
            domestic_lags=p,
            foreign_lags=q,
            is_stable=is_stable,
            eigenvalues=eigenvalues,
        )
