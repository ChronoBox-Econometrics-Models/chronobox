"""Automatic model selection (Auto-ARIMA, Auto-ETS, etc.)."""

from chronobox.selection.auto_arima import auto_arima
from chronobox.selection.auto_ets import auto_ets

__all__ = ["auto_arima", "auto_ets"]
