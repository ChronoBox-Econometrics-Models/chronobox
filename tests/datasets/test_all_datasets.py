"""Tests for all datasets in the catalog."""

from __future__ import annotations

import pandas as pd
import pytest

from chronobox.datasets import DATASET_CATALOG, list_datasets, load_dataset


class TestDatasetCatalog:
    """Tests for dataset catalog."""

    def test_catalog_has_at_least_25_datasets(self) -> None:
        assert len(DATASET_CATALOG) >= 25

    def test_list_datasets_returns_dataframe(self) -> None:
        df = list_datasets()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 25

    def test_list_datasets_filter_by_category(self) -> None:
        classic = list_datasets(category="classic")
        assert len(classic) >= 5
        assert all(classic["category"] == "classic")


class TestSimulatedDatasets:
    """Tests for simulated datasets."""

    @pytest.mark.parametrize(
        "name",
        ["arma11", "var2", "cointegrated", "structural_break", "long_memory"],
    )
    def test_simulated_dataset_loads(self, name: str) -> None:
        data = load_dataset(name)
        assert data is not None
        assert len(data) > 0

    def test_arma11_is_series(self) -> None:
        data = load_dataset("arma11")
        assert isinstance(data, pd.Series)

    def test_var2_is_dataframe(self) -> None:
        data = load_dataset("var2")
        assert isinstance(data, pd.DataFrame)
        assert data.shape[1] == 2

    def test_cointegrated_has_two_columns(self) -> None:
        data = load_dataset("cointegrated")
        assert isinstance(data, pd.DataFrame)
        assert "y" in data.columns
        assert "x" in data.columns


class TestClassicDatasets:
    """Tests for classic datasets (require CSV files)."""

    @pytest.mark.parametrize(
        "name", ["airline", "nile", "sunspot", "lynx", "co2", "uspop"]
    )
    def test_classic_dataset_loads(self, name: str) -> None:
        try:
            data = load_dataset(name)
            assert data is not None
            assert len(data) > 0
        except FileNotFoundError:
            pytest.skip(f"CSV file for {name} not bundled")


class TestUnknownDataset:
    """Tests for error handling."""

    def test_unknown_dataset_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            load_dataset("nonexistent_dataset_xyz")
