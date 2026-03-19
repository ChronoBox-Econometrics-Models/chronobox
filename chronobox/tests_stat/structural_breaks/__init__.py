"""
Structural break tests for time series.

Tests
-----
chow_test : Chow test for known break point
bai_perron_test : Bai-Perron multiple structural changes
cusum_test : CUSUM test for parameter stability
cusumsq_test : CUSUM of squares test
qlr_test : Quandt Likelihood Ratio (sup-Wald) test
"""

from chronobox.tests_stat.structural_breaks.bai_perron import bai_perron_test
from chronobox.tests_stat.structural_breaks.chow import chow_test
from chronobox.tests_stat.structural_breaks.cusum import cusum_test, cusumsq_test
from chronobox.tests_stat.structural_breaks.qlr import qlr_test

__all__ = [
    "bai_perron_test",
    "chow_test",
    "cusum_test",
    "cusumsq_test",
    "qlr_test",
]
