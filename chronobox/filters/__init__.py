"""
chronobox.filters - Economic time series filters.

Implements classical trend-cycle decomposition filters:
- HP (Hodrick-Prescott)
- BK (Baxter-King band-pass)
- CF (Christiano-Fitzgerald)
- Hamilton regression filter
- BN (Beveridge-Nelson decomposition)
"""

from chronobox.filters.bk import bk_filter
from chronobox.filters.bn import bn_decomposition, bn_decomposition_detailed
from chronobox.filters.cf import cf_filter
from chronobox.filters.hamilton import hamilton_filter, hamilton_filter_detailed
from chronobox.filters.hp import hp_filter

__all__ = [
    "bk_filter",
    "bn_decomposition",
    "bn_decomposition_detailed",
    "cf_filter",
    "hamilton_filter",
    "hamilton_filter_detailed",
    "hp_filter",
]
