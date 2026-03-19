"""Tests for X-13ARIMA-SEATS wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.decomposition.x13_wrapper import X13Wrapper, _find_x13_executable


class TestX13Wrapper:
    """Tests for X-13 wrapper."""

    def test_wrapper_initialization(self) -> None:
        """X13Wrapper should initialize without error."""
        wrapper = X13Wrapper()
        # is_available depends on whether x13as is installed
        assert isinstance(wrapper.is_available, bool)

    def test_wrapper_invalid_path(self) -> None:
        """Invalid path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            X13Wrapper(x13_path="/nonexistent/path/x13as")

    def test_is_available_property(self) -> None:
        """is_available should return bool."""
        wrapper = X13Wrapper()
        assert isinstance(wrapper.is_available, bool)

    def test_seasonal_adjust_unavailable(self) -> None:
        """seasonal_adjust should raise if x13 not available."""
        wrapper = X13Wrapper()
        if not wrapper.is_available:
            y = np.ones(48)
            with pytest.raises(RuntimeError, match="not available"):
                wrapper.seasonal_adjust(y, period=12)

    @pytest.mark.skipif(
        _find_x13_executable() is None,
        reason="X-13ARIMA-SEATS not installed",
    )
    def test_seasonal_adjust_basic(self) -> None:
        """Basic seasonal adjustment should work if x13 is installed."""
        rng = np.random.default_rng(42)
        t = np.arange(120)
        y = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 120)
        wrapper = X13Wrapper()
        result = wrapper.seasonal_adjust(y, period=12)
        assert result.period == 12
        assert len(result.trend) == len(y)
