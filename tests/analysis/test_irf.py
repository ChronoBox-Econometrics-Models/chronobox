"""Tests for Impulse Response Functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronobox.analysis.irf import IRF
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
def var_results(canada_data: pd.DataFrame) -> VARResults:
    """Fit VAR(2) on Canada data."""
    model = VAR(lags=2)
    return model.fit(canada_data)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def stable_var_results(rng: np.random.Generator) -> VARResults:
    """Fit VAR on a stable simulated process."""
    k = 3
    t = 300
    a1 = np.array([[0.3, 0.05, 0.0], [0.0, 0.25, 0.05], [0.0, 0.0, 0.2]])

    y = np.zeros((t + 50, k))
    for t_i in range(1, t + 50):
        y[t_i] = a1 @ y[t_i - 1] + rng.standard_normal(k) * 0.5

    model = VAR(lags=1)
    return model.fit(y[50:])


@pytest.fixture
def monetary_var_results(rng: np.random.Generator) -> VARResults:
    """Simulate a simple monetary VAR: [output, inflation, interest_rate].

    A shock to interest rate (variable 2) should reduce output (variable 0)
    with some delay (negative IRF).
    """
    k = 3
    t = 500
    # Simple DGP where higher interest rate reduces output
    a1 = np.array([
        [0.6, 0.05, -0.15],  # output: positive persistence, negative from rate
        [0.05, 0.5, 0.02],  # inflation: some persistence
        [0.1, 0.1, 0.6],  # rate: responds to output and inflation
    ])

    sigma = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 0.5, 0.05],
        [0.1, 0.05, 0.3],
    ])
    chol = np.linalg.cholesky(sigma)

    y = np.zeros((t + 100, k))
    for t_i in range(1, t + 100):
        y[t_i] = a1 @ y[t_i - 1] + chol @ rng.standard_normal(k)

    model = VAR(lags=1)
    return model.fit(y[100:], names=["output", "inflation", "interest_rate"])


class TestIRFCholesky:
    """Tests for orthogonalized (Cholesky) IRF."""

    def test_irf_cholesky_shape(self, var_results: VARResults) -> None:
        """IRF should have shape (periods+1, K, K)."""
        irf = IRF(var_results, periods=20, method="cholesky", runs=0)
        assert irf.irfs.shape == (21, 4, 4)

    def test_irf_initial_identity(self, var_results: VARResults) -> None:
        """IRF at h=0 should equal Cholesky factor P of Sigma_u.

        Theta_0 = Phi_0 @ P = I @ P = P
        """
        irf = IRF(var_results, periods=10, method="cholesky", runs=0)
        p_chol = np.linalg.cholesky(var_results.sigma_u)
        np.testing.assert_allclose(irf.irfs[0], p_chol, atol=1e-10)

    def test_irf_decays(self, stable_var_results: VARResults) -> None:
        """IRF of a stable system should decay toward zero."""
        irf = IRF(stable_var_results, periods=50, method="cholesky", runs=0)

        # At horizon 50, all IRF values should be close to 0
        max_abs_h50 = np.max(np.abs(irf.irfs[-1]))
        assert max_abs_h50 < 0.01, (
            f"IRF at h=50 has max abs value {max_abs_h50}, expected < 0.01"
        )

    def test_irf_monetary_shock(self, monetary_var_results: VARResults) -> None:
        """A positive interest rate shock should reduce output (negative sign).

        In the ordering [output, inflation, interest_rate],
        a Cholesky shock to variable 2 (interest_rate) should have a
        negative cumulative effect on variable 0 (output).
        """
        irf = IRF(monetary_var_results, periods=20, method="cholesky", runs=0)

        # Cumulative response of output to interest rate shock
        # Sum the IRF from h=0 to h=20
        cum_response = np.sum(irf.irfs[:, 0, 2])
        assert cum_response < 0, (
            f"Cumulative output response to rate shock = {cum_response:.4f}, "
            f"expected negative"
        )

    def test_irf_cum_irfs(self, var_results: VARResults) -> None:
        """Cumulative IRFs should equal cumsum of point IRFs."""
        irf = IRF(var_results, periods=10, method="cholesky", runs=0)
        expected_cum = np.cumsum(irf.irfs, axis=0)
        np.testing.assert_allclose(irf.cum_irfs, expected_cum, atol=1e-12)


class TestIRFGeneralized:
    """Tests for generalized (Pesaran-Shin) IRF."""

    def test_girf_shape(self, var_results: VARResults) -> None:
        """GIRF should have shape (periods+1, K, K)."""
        irf = IRF(var_results, periods=15, method="generalized", runs=0)
        assert irf.irfs.shape == (16, 4, 4)

    def test_girf_invariant_to_ordering(self, var_results: VARResults) -> None:
        """GIRF should not depend on variable ordering.

        If we permute the variables, the GIRF values should be
        correspondingly permuted (same values, different positions).
        """
        irf_orig = IRF(var_results, periods=10, method="generalized", runs=0)

        # Create a permuted version of the data
        perm = [2, 0, 3, 1]  # arbitrary permutation
        from chronobox.models.var import VAR

        endog_perm = var_results.endog[:, perm]
        names_perm = [var_results.names[i] for i in perm]
        model_perm = VAR(lags=var_results.k_ar, trend=var_results.trend)
        results_perm = model_perm.fit(endog_perm, names=names_perm)
        irf_perm = IRF(results_perm, periods=10, method="generalized", runs=0)

        # Compare: IRF[h, i, j] in original == IRF[h, perm[i], perm[j]] in permuted
        for h in range(11):
            for i in range(4):
                for j in range(4):
                    np.testing.assert_allclose(
                        irf_orig.irfs[h, i, j],
                        irf_perm.irfs[
                            h,
                            perm.index(i) if i in perm else i,
                            perm.index(j) if j in perm else j,
                        ],
                        atol=0.05,
                        err_msg=f"GIRF not invariant at h={h}, i={i}, j={j}",
                    )

    def test_girf_initial_diagonal(self, var_results: VARResults) -> None:
        """GIRF at h=0: diagonal elements should equal sqrt(sigma_jj)."""
        irf = IRF(var_results, periods=5, method="generalized", runs=0)
        sigma = var_results.sigma_u

        for j in range(4):
            expected = sigma[j, j] / np.sqrt(sigma[j, j])  # = sqrt(sigma_jj)
            np.testing.assert_allclose(
                irf.irfs[0, j, j],
                expected,
                atol=1e-10,
                err_msg=f"GIRF[0, {j}, {j}] != sqrt(sigma_{j}{j})",
            )


class TestIRFBootstrap:
    """Tests for bootstrap confidence bands."""

    def test_bootstrap_bands_shape(self, var_results: VARResults) -> None:
        """Bootstrap bands should have same shape as point IRFs."""
        irf = IRF(var_results, periods=10, method="cholesky", runs=50, seed=42)

        assert irf.lower is not None
        assert irf.upper is not None
        assert irf.lower.shape == (11, 4, 4)
        assert irf.upper.shape == (11, 4, 4)

    def test_bootstrap_bands_contain_point(
        self, var_results: VARResults
    ) -> None:
        """Bootstrap bands should generally contain the point estimate.

        Due to sampling variability, we check that most (>80%) of the
        IRF points lie within the confidence bands.
        """
        irf = IRF(var_results, periods=10, method="cholesky", runs=100, seed=42)

        assert irf.lower is not None
        assert irf.upper is not None

        within = (irf.irfs >= irf.lower) & (irf.irfs <= irf.upper)
        pct_within = np.mean(within)

        assert pct_within > 0.5, (
            f"Only {pct_within:.1%} of IRF points within bootstrap bands, "
            f"expected > 50%"
        )

    def test_bootstrap_upper_gt_lower(self, var_results: VARResults) -> None:
        """Upper band should be >= lower band everywhere."""
        irf = IRF(var_results, periods=10, method="cholesky", runs=50, seed=42)

        assert irf.lower is not None
        assert irf.upper is not None
        assert np.all(irf.upper >= irf.lower - 1e-10)

    def test_no_bootstrap(self, var_results: VARResults) -> None:
        """With runs=0, lower and upper should be None."""
        irf = IRF(var_results, periods=10, method="cholesky", runs=0)
        assert irf.lower is None
        assert irf.upper is None


class TestIRFOutput:
    """Tests for IRF output methods."""

    def test_to_dataframe(self, var_results: VARResults) -> None:
        """to_dataframe should return a long-form DataFrame."""
        irf = IRF(var_results, periods=5, method="cholesky", runs=0)
        df = irf.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "horizon" in df.columns
        assert "impulse" in df.columns
        assert "response" in df.columns
        assert "irf" in df.columns
        # 6 horizons * 4 impulses * 4 responses = 96 rows
        assert len(df) == 6 * 4 * 4

    def test_plot_no_error(self, var_results: VARResults) -> None:
        """plot() should run without error."""
        import matplotlib

        matplotlib.use("Agg")

        irf = IRF(var_results, periods=10, method="cholesky", runs=0)
        fig = irf.plot()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_single_pair(self, var_results: VARResults) -> None:
        """plot() with specific impulse/response should work."""
        import matplotlib

        matplotlib.use("Agg")

        irf = IRF(var_results, periods=10, method="cholesky", runs=0)
        fig = irf.plot(impulse="e", response="prod")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_cum_no_error(self, var_results: VARResults) -> None:
        """plot_cum() should run without error."""
        import matplotlib

        matplotlib.use("Agg")

        irf = IRF(var_results, periods=10, method="cholesky", runs=0)
        fig = irf.plot_cum()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_invalid_method(self, var_results: VARResults) -> None:
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="method"):
            IRF(var_results, method="invalid", runs=0)
