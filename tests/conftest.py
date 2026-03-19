"""Shared test fixtures for chronobox."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronobox.datasets import load_dataset


@pytest.fixture
def airline_data() -> pd.DataFrame:
    """Load the airline dataset."""
    return load_dataset("airline")


@pytest.fixture
def airline_passengers(airline_data: pd.DataFrame) -> np.ndarray:
    """Airline passengers as numpy array."""
    if isinstance(airline_data, pd.Series):
        return airline_data.to_numpy(dtype=np.float64)
    return airline_data["passengers"].to_numpy(dtype=np.float64)


@pytest.fixture
def nile_data() -> pd.DataFrame:
    """Load the Nile dataset."""
    return load_dataset("nile")


@pytest.fixture
def nile_volume(nile_data: pd.DataFrame) -> np.ndarray:
    """Nile volume as numpy array."""
    if isinstance(nile_data, pd.Series):
        return nile_data.to_numpy(dtype=np.float64)
    return nile_data["volume"].to_numpy(dtype=np.float64)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)
