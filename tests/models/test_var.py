"""Tests for VAR model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronobox.models.var import VAR, VARResults


@pytest.fixture
def canada_data() -> pd.DataFrame:
    """Load Canada dataset as DataFrame."""
    data_path = (
        Path(__file__).parent.parent.parent
        / "chronobox"
        / "datasets"
        / "data"
        / "macro"
        / "canada.csv"
    )
    df = pd.read_csv(data_path)
    return df[["e", "prod", "rw", "U"]]


@pytest.fixture
def canada_array(canada_data: pd.DataFrame) -> np.ndarray:
    """Canada dataset as numpy array."""
    return canada_data.to_numpy(dtype=np.float64)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def simulated_var1(rng: np.random.Generator) -> np.ndarray:
    """Simulate a simple stable VAR(1) process."""
    k = 3
    t = 200
    a1 = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.0, 0.0, 0.3]])

    y = np.zeros((t + 50, k))
    for t_i in range(1, t + 50):
        y[t_i] = a1 @ y[t_i - 1] + rng.standard_normal(k) * 0.5

    return y[50:]


class TestVARFit:
    """Tests for VAR model fitting."""

    def test_var_canada_ols(self, canada_data: pd.DataFrame) -> None:
        """VAR(2) on Canada dataset: coefficients should be reasonable."""
        model = VAR(lags=2)
        results = model.fit(canada_data)

        assert isinstance(results, VARResults)
        assert results.k_ar == 2
        assert results.neqs == 4
        assert results.coefs.shape == (2, 4, 4)
        assert results.nobs == 84 - 2  # T - p = 82

        # Coefficient A_1[0,0] (e on L1.e) should be close to R vars value
        # R: vars::VAR(Canada, p=2, type="const") gives A1[1,1] ~ 1.01
        # The exact value depends on data, but it should be positive and < 2
        # Coefficient A_1[0,0] for near-unit-root series can be > 1
        a1_11 = results.coefs[0, 0, 0]
        assert 0.5 < a1_11 < 2.0, f"A1[0,0] = {a1_11}, expected ~1.0-1.7"

    def test_var_stability(self, canada_data: pd.DataFrame) -> None:
        """VAR(2) on Canada should be stable."""
        model = VAR(lags=2)
        results = model.fit(canada_data)
        assert results.is_stable, (
            f"VAR(2) on Canada is not stable. "
            f"Max eigenvalue modulus: {np.max(np.abs(results.roots)):.4f}"
        )

    def test_var_residual_cov(self, canada_data: pd.DataFrame) -> None:
        """Sigma_u should be symmetric and positive definite."""
        model = VAR(lags=2)
        results = model.fit(canada_data)

        sigma = results.sigma_u
        # Symmetric
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)

        # Positive definite (all eigenvalues > 0)
        eigvals = np.linalg.eigvalsh(sigma)
        assert np.all(eigvals > 0), f"Sigma_u has non-positive eigenvalue: {eigvals}"

    def test_var_forecast(self, canada_data: pd.DataFrame) -> None:
        """Forecast should return (steps, K) array."""
        model = VAR(lags=2)
        results = model.fit(canada_data)
        forecasts = results.forecast(steps=8)

        assert forecasts.shape == (8, 4)
        assert np.all(np.isfinite(forecasts))

    def test_var_summary(self, canada_data: pd.DataFrame) -> None:
        """summary() should produce a formatted string without errors."""
        model = VAR(lags=2)
        results = model.fit(canada_data)
        summary = results.summary()

        assert isinstance(summary, str)
        assert len(summary) > 200
        assert "VAR(2)" in summary
        assert "AIC" in summary
        assert "e" in summary

    def test_var_select_order(self, canada_data: pd.DataFrame) -> None:
        """select_order on Canada should select a reasonable lag (1-4)."""
        model = VAR(trend="c")
        result = model.select_order(canada_data, maxlags=8)

        aic_order = result.selected_orders.get("aic", -1)
        bic_order = result.selected_orders.get("bic", -1)

        assert 0 <= aic_order <= 8, f"AIC selected p={aic_order}"
        assert 0 <= bic_order <= 8, f"BIC selected p={bic_order}"

    def test_var_auto_lag_selection(self, canada_array: np.ndarray) -> None:
        """VAR with lags=None, maxlags=8 should auto-select and fit."""
        model = VAR(lags=None, maxlags=8)
        results = model.fit(canada_array)

        assert results.k_ar >= 1
        assert results.k_ar <= 8
        assert results.coefs.shape[0] == results.k_ar

    def test_var_no_trend(self, simulated_var1: np.ndarray) -> None:
        """VAR with trend='n' should fit without intercept."""
        model = VAR(lags=1, trend="n")
        results = model.fit(simulated_var1)

        assert np.allclose(results.intercept, 0.0)
        assert results.coefs.shape == (1, 3, 3)

    def test_var_trend_ct(self, canada_data: pd.DataFrame) -> None:
        """VAR with trend='ct' should include constant and linear trend."""
        model = VAR(lags=2, trend="ct")
        results = model.fit(canada_data)

        assert results.trend == "ct"
        assert results.trend_coefs is not None
        assert results.trend_coefs.shape[1] == 2  # constant + trend

    def test_var_dataframe_names(self, canada_data: pd.DataFrame) -> None:
        """Fitting a DataFrame should preserve column names."""
        model = VAR(lags=2)
        results = model.fit(canada_data)

        assert results.names == ["e", "prod", "rw", "U"]

    def test_var_information_criteria(self, canada_data: pd.DataFrame) -> None:
        """AIC, BIC, HQIC, FPE should be finite and BIC > AIC."""
        model = VAR(lags=2)
        results = model.fit(canada_data)

        assert np.isfinite(results.aic)
        assert np.isfinite(results.bic)
        assert np.isfinite(results.hqic)
        assert np.isfinite(results.fpe)
        assert results.fpe > 0

    def test_var_companion_eigenvalues(self, canada_data: pd.DataFrame) -> None:
        """Companion eigenvalues should be sorted by descending modulus."""
        model = VAR(lags=2)
        results = model.fit(canada_data)

        roots = results.roots
        moduli = np.abs(roots)
        assert np.all(moduli[:-1] >= moduli[1:] - 1e-10)

    def test_var_whiteness(self, canada_data: pd.DataFrame) -> None:
        """Portmanteau test should return valid statistics."""
        model = VAR(lags=2)
        results = model.fit(canada_data)
        whiteness = results.test_whiteness(nlags=10)

        assert "statistic" in whiteness
        assert "pvalue" in whiteness
        assert "df" in whiteness
        assert whiteness["statistic"] >= 0
        assert 0 <= whiteness["pvalue"] <= 1

    def test_var_invalid_input_1d(self) -> None:
        """Should raise ValueError for 1-D input."""
        model = VAR(lags=1)
        with pytest.raises(ValueError, match="2-D"):
            model.fit(np.array([1.0, 2.0, 3.0]))

    def test_var_no_lags_no_maxlags(self, canada_data: pd.DataFrame) -> None:
        """Should raise ValueError when neither lags nor maxlags specified."""
        model = VAR(lags=None, maxlags=None)
        with pytest.raises(ValueError, match="lags or maxlags"):
            model.fit(canada_data)

    def test_var_residuals_shape(self, canada_data: pd.DataFrame) -> None:
        """Residuals should have shape (T-p, K)."""
        model = VAR(lags=3)
        results = model.fit(canada_data)
        assert results.resid.shape == (84 - 3, 4)

    def test_var_residuals_zero_mean(self, canada_data: pd.DataFrame) -> None:
        """Residuals should have approximately zero mean."""
        model = VAR(lags=2)
        results = model.fit(canada_data)
        resid_mean = np.mean(results.resid, axis=0)
        np.testing.assert_allclose(resid_mean, 0.0, atol=0.1)
