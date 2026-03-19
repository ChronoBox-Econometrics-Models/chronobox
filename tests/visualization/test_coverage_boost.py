"""Additional visualization tests for coverage improvement.

Tests edge cases and branches not covered by existing tests.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class TestIRFPlotEdgeCases:
    """Cover uncovered branches in irf_plot.py."""

    def test_irf_with_response_filter(self) -> None:
        from chronobox.visualization.irf_plot import plot_irf

        irf = np.random.randn(3, 3, 20)
        fig = plot_irf(
            irf_array=irf,
            var_names=["GDP", "INF", "IR"],
            response="GDP",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_irf_with_impulse_filter(self) -> None:
        from chronobox.visualization.irf_plot import plot_irf

        irf = np.random.randn(3, 3, 20)
        fig = plot_irf(
            irf_array=irf,
            var_names=["GDP", "INF", "IR"],
            impulse="INF",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_irf_cumulative_with_array(self) -> None:
        from chronobox.visualization.irf_plot import plot_irf

        irf = np.random.randn(2, 2, 10)
        fig = plot_irf(irf_array=irf, cumulative=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_irf_with_int_impulse_response(self) -> None:
        from chronobox.visualization.irf_plot import plot_irf

        irf = np.random.randn(3, 3, 10)
        fig = plot_irf(irf_array=irf, impulse=0, response=1)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestForecastPlotEdgeCases:
    """Cover uncovered branches in forecast_plot.py."""

    def test_forecast_with_se_none(self) -> None:
        from chronobox.visualization.forecast_plot import plot_forecast

        history = np.cumsum(np.random.randn(100))
        forecast = np.array([history[-1] + i * 0.1 for i in range(12)])
        fig = plot_forecast(
            history=history,
            forecast_mean=forecast,
            forecast_se=None,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_forecast_default_labels(self) -> None:
        from chronobox.visualization.forecast_plot import plot_forecast

        history = np.cumsum(np.random.randn(50))
        forecast = np.array([history[-1] + i for i in range(5)])
        fig = plot_forecast(
            history=history,
            forecast_mean=forecast,
            forecast_se=np.ones(5),
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_forecast_custom_ci_levels(self) -> None:
        from chronobox.visualization.forecast_plot import plot_forecast

        history = np.cumsum(np.random.randn(50))
        forecast = np.array([history[-1] + i for i in range(8)])
        fig = plot_forecast(
            history=history,
            forecast_mean=forecast,
            forecast_se=np.ones(8) * 2,
            ci_levels=[0.5, 0.9],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_forecast_with_existing_ax(self) -> None:
        from chronobox.visualization.forecast_plot import plot_forecast

        fig0, ax0 = plt.subplots()
        history = np.cumsum(np.random.randn(50))
        forecast = np.array([history[-1] + i for i in range(5)])
        fig = plot_forecast(
            history=history,
            forecast_mean=forecast,
            forecast_se=np.ones(5),
            ax=ax0,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestCoefPlotEdgeCases:
    """Cover uncovered branches in coef_plot.py."""

    def test_coef_plot_2d_with_ols(self) -> None:
        from chronobox.visualization.coef_plot import plot_tvp_coefs

        T = 50
        coefs = np.random.randn(T, 2)
        fig = plot_tvp_coefs(
            coefs=coefs,
            ols_coefs=np.array([0.5, -0.3]),
            coef_names=["x1", "x2"],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_coef_plot_1d_array(self) -> None:
        from chronobox.visualization.coef_plot import plot_tvp_coefs

        T = 50
        coefs = np.random.randn(T)
        fig = plot_tvp_coefs(coefs=coefs)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_coef_plot_with_confidence(self) -> None:
        from chronobox.visualization.coef_plot import plot_tvp_coefs

        T = 50
        coefs = np.random.randn(T, 3)
        lower = coefs - np.abs(np.random.randn(T, 3))
        upper = coefs + np.abs(np.random.randn(T, 3))
        fig = plot_tvp_coefs(
            coefs=coefs,
            lower=lower,
            upper=upper,
            coef_names=["a", "b", "c"],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestSpilloverPlotEdgeCases:
    """Cover uncovered branches in spillover_plot.py."""

    def test_heatmap_no_annotate(self) -> None:
        from chronobox.visualization.spillover_plot import plot_heatmap

        matrix = np.random.rand(3, 3) * 100
        fig = plot_heatmap(
            spillover_table=matrix,
            var_names=["A", "B", "C"],
            annotate=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_heatmap_custom_cmap(self) -> None:
        from chronobox.visualization.spillover_plot import plot_heatmap

        matrix = np.random.rand(3, 3) * 100
        fig = plot_heatmap(
            spillover_table=matrix,
            var_names=["A", "B", "C"],
            cmap="viridis",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_network_with_negative_values(self) -> None:
        from chronobox.visualization.spillover_plot import plot_network

        matrix = np.array([
            [50.0, -10.0, 5.0],
            [8.0, 60.0, -3.0],
            [-5.0, 7.0, 40.0],
        ])
        fig = plot_network(
            spillover_table=matrix,
            var_names=["A", "B", "C"],
            threshold=2.0,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_network_zero_spillover(self) -> None:
        from chronobox.visualization.spillover_plot import plot_network

        matrix = np.zeros((3, 3))
        np.fill_diagonal(matrix, 100.0)
        fig = plot_network(
            spillover_table=matrix,
            var_names=["A", "B", "C"],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_rolling_with_values(self) -> None:
        from chronobox.visualization.spillover_plot import plot_rolling

        values = np.random.rand(50) * 100
        fig = plot_rolling(rolling_total=values)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_rolling_with_dates(self) -> None:
        import pandas as pd

        from chronobox.visualization.spillover_plot import plot_rolling

        values = np.random.rand(50) * 100
        dates = pd.date_range("2020-01-01", periods=50, freq="ME")
        fig = plot_rolling(rolling_total=values, dates=dates)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestDecompositionPlotEdgeCases:
    """Cover uncovered branches in decomposition_plot.py."""

    def test_decomposition_with_components_dict(self) -> None:
        from chronobox.visualization.decomposition_plot import plot_decomposition

        T = 100
        fig = plot_decomposition(
            components={
                "observed": np.random.randn(T),
                "trend": np.random.randn(T),
                "seasonal": np.random.randn(T),
                "resid": np.random.randn(T),
            },
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_decomposition_with_which_param(self) -> None:
        from chronobox.visualization.decomposition_plot import plot_decomposition

        T = 100
        fig = plot_decomposition(
            components={
                "observed": np.random.randn(T),
                "trend": np.random.randn(T),
                "seasonal": np.random.randn(T),
                "resid": np.random.randn(T),
            },
            which=["observed", "trend"],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_decomposition_with_dates(self) -> None:
        import pandas as pd

        from chronobox.visualization.decomposition_plot import plot_decomposition

        T = 100
        fig = plot_decomposition(
            components={
                "observed": np.random.randn(T),
                "trend": np.random.randn(T),
                "seasonal": np.random.randn(T),
                "resid": np.random.randn(T),
            },
            dates=pd.date_range("2020-01-01", periods=T, freq="ME"),
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestTestPlotEdgeCases:
    """Cover uncovered branches in test_plot.py."""

    def test_cusum_with_raw_arrays(self) -> None:
        from chronobox.visualization.test_plot import plot_cusum

        T = 100
        fig = plot_cusum(
            cusum_path=np.cumsum(np.random.randn(T)),
            upper_band=np.linspace(1, 3, T),
            lower_band=-np.linspace(1, 3, T),
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_bai_perron_with_raw_arrays(self) -> None:
        from chronobox.visualization.test_plot import plot_bai_perron

        T = 200
        fig = plot_bai_perron(
            y=np.random.randn(T),
            break_dates=[50, 120],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_zivot_andrews_with_raw_arrays(self) -> None:
        from chronobox.visualization.test_plot import plot_zivot_andrews

        T = 100
        fig = plot_zivot_andrews(
            y=np.random.randn(T),
            break_index=50,
            za_stat=-4.5,
            critical_values={"1%": -5.34, "5%": -4.80, "10%": -4.58},
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_recursive_coefs_with_raw_arrays(self) -> None:
        from chronobox.visualization.test_plot import plot_recursive_coefs

        T = 80
        K = 3
        coefs = np.random.randn(T, K)
        coef_se = np.abs(np.random.randn(T, K)) * 0.5
        fig = plot_recursive_coefs(
            coefs=coefs,
            coef_se=coef_se,
            coef_names=["x1", "x2", "x3"],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestExportEdgeCases:
    """Cover uncovered branches in export.py."""

    def test_save_figure_png(self) -> None:
        from chronobox.visualization.export import save_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.png")
            save_figure(fig, path)
            assert os.path.exists(path)

        plt.close(fig)

    def test_save_figure_svg(self) -> None:
        from chronobox.visualization.export import save_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.svg")
            save_figure(fig, path, fmt="svg")
            assert os.path.exists(path)

        plt.close(fig)

    def test_save_figure_pdf(self) -> None:
        from chronobox.visualization.export import save_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pdf")
            save_figure(fig, path, fmt="pdf")
            assert os.path.exists(path)

        plt.close(fig)


class TestTsPlotEdgeCases:
    """Cover uncovered branches in ts_plot.py."""

    def test_plot_series_with_annotations_dict(self) -> None:
        from chronobox.visualization.ts_plot import plot_series

        data = np.random.randn(100)
        fig = plot_series(
            data,
            title="Test",
            annotations=[
                {"x": 10, "y": 0.5, "text": "event1"},
                {"x": 50, "y": -0.5, "text": "event2"},
            ],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_series_with_recessions(self) -> None:
        from chronobox.visualization.ts_plot import plot_series

        data = np.random.randn(100)
        fig = plot_series(
            data,
            recessions=[(10, 20), (60, 75)],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_series_multiple_with_secondary_y(self) -> None:
        import pandas as pd

        from chronobox.visualization.ts_plot import plot_series

        data1 = pd.Series(np.random.randn(100), name="A")
        data2 = pd.Series(np.random.randn(100) * 100, name="B")
        fig = plot_series(
            [data1, data2],
            labels=["A", "B"],
            secondary_y=["B"],
            secondary_ylabel="Scale B",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestFEVDPlotEdgeCases:
    """Cover uncovered branches in fevd_plot.py."""

    def test_fevd_bar_plot(self) -> None:
        from chronobox.visualization.fevd_plot import plot_fevd

        # (K, K, periods) shape: K=3 vars, 10 periods
        fevd = np.zeros((3, 3, 10))
        for t in range(10):
            for k in range(3):
                vals = np.random.dirichlet([1, 1, 1])
                fevd[k, :, t] = vals

        fig = plot_fevd(
            fevd_array=fevd,
            var_names=["A", "B", "C"],
            plot_type="bar",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_fevd_single_variable_filter(self) -> None:
        from chronobox.visualization.fevd_plot import plot_fevd

        fevd = np.zeros((3, 3, 10))
        for t in range(10):
            for k in range(3):
                vals = np.random.dirichlet([1, 1, 1])
                fevd[k, :, t] = vals

        fig = plot_fevd(
            fevd_array=fevd,
            var_names=["A", "B", "C"],
            variable="A",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_fevd_int_variable_filter(self) -> None:
        from chronobox.visualization.fevd_plot import plot_fevd

        fevd = np.zeros((3, 3, 10))
        for t in range(10):
            for k in range(3):
                vals = np.random.dirichlet([1, 1, 1])
                fevd[k, :, t] = vals

        fig = plot_fevd(
            fevd_array=fevd,
            var_names=["A", "B", "C"],
            variable=1,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestHDPlotEdgeCases:
    """Cover uncovered branches in hd_plot.py."""

    def test_hd_with_variable_filter(self) -> None:
        from chronobox.visualization.hd_plot import plot_hd

        T, K = 50, 3
        hd = np.random.randn(T, K)
        observed = np.sum(hd, axis=1) + np.random.randn(T) * 0.1
        fig = plot_hd(
            hd_array=hd,
            observed=observed,
            shock_names=["Shock1", "Shock2", "Shock3"],
            variable=0,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_hd_with_dates(self) -> None:
        import pandas as pd

        from chronobox.visualization.hd_plot import plot_hd

        T, K = 50, 2
        hd = np.random.randn(T, K)
        observed = np.sum(hd, axis=1)
        fig = plot_hd(
            hd_array=hd,
            observed=observed,
            shock_names=["S1", "S2"],
            dates=pd.date_range("2020-01-01", periods=T, freq="ME"),
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
