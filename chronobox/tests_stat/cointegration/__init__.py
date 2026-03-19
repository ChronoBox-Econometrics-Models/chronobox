"""
Cointegration tests for time series.

Tests
-----
engle_granger_test : Engle-Granger two-step cointegration test
gregory_hansen_test : Gregory-Hansen cointegration with structural break
bounds_test : Pesaran-Shin-Smith ARDL bounds test
phillips_ouliaris_test : Phillips-Ouliaris residual-based cointegration test
"""

from chronobox.tests_stat.cointegration.bounds_test import bounds_test
from chronobox.tests_stat.cointegration.engle_granger import engle_granger_test
from chronobox.tests_stat.cointegration.gregory_hansen import gregory_hansen_test
from chronobox.tests_stat.cointegration.phillips_ouliaris import phillips_ouliaris_test

__all__ = [
    "bounds_test",
    "engle_granger_test",
    "gregory_hansen_test",
    "phillips_ouliaris_test",
]
