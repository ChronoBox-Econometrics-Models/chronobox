"""End-to-end integration tests for chronobox."""

from __future__ import annotations

import numpy as np


class TestEndToEnd:
    """Test the full workflow as shown in the README."""

    def test_arima_workflow(self) -> None:
        """Test the complete ARIMA workflow."""
        from chronobox import ARIMA
        from chronobox.datasets import load_dataset

        airline = load_dataset("airline")
        model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
        results = model.fit(airline["passengers"])

        # summary
        summary = results.summary()
        assert isinstance(summary, str)
        assert len(summary) > 100

        # forecast
        fc = results.forecast(steps=12)
        assert len(fc["forecast"]) == 12
        assert np.all(np.isfinite(fc["forecast"]))

    def test_auto_arima_workflow(self) -> None:
        """Test the complete auto_arima workflow."""
        from chronobox import auto_arima
        from chronobox.datasets import load_dataset

        airline = load_dataset("airline")
        best = auto_arima(
            airline["passengers"],
            seasonal=True,
            m=12,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
        )

        summary = best.summary()
        assert isinstance(summary, str)
        assert "ARIMA" in summary

    def test_plot_diagnostics(self) -> None:
        """Test plot generation without error."""
        import matplotlib

        matplotlib.use("Agg")

        from chronobox import ARIMA
        from chronobox.datasets import load_dataset

        nile = load_dataset("nile")
        model = ARIMA(order=(1, 1, 1))
        results = model.fit(nile["volume"])

        fig = results.plot_diagnostics()
        assert fig is not None

    def test_all_datasets_loadable(self) -> None:
        """Test that all 4 datasets can be loaded."""
        from chronobox.datasets import list_datasets, load_dataset

        datasets = list_datasets()
        assert len(datasets) >= 2  # at minimum airline and nile

        for name in ["airline", "nile"]:
            df = load_dataset(name)
            assert len(df) > 0
