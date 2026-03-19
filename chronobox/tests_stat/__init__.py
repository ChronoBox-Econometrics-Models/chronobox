"""
Statistical tests for time series analysis.

Provides unit root tests, cointegration tests, structural break tests,
and specification/diagnostic tests. All tests return a standardized
TestResult dataclass.

Submodules
----------
unit_root : ADF, PP, KPSS, ERS/DF-GLS, Zivot-Andrews, Lee-Strazicich, HEGY
cointegration : Engle-Granger, Gregory-Hansen, Bounds/PSS, Phillips-Ouliaris
structural_breaks : Chow, Bai-Perron, CUSUM/CUSUMSQ, QLR
specification : Ljung-Box, Breusch-Godfrey, Durbin-Watson, BDS, ARCH-LM, White, Jarque-Bera, RESET
critical_values : MacKinnon, Osterwald-Lenum, PSS bounds
"""

from chronobox.tests_stat.base import TestResult

# Cointegration tests
from chronobox.tests_stat.cointegration import (
    bounds_test,
    engle_granger_test,
    gregory_hansen_test,
    phillips_ouliaris_test,
)

# Specification tests
from chronobox.tests_stat.specification import (
    arch_lm_test,
    bds_test,
    breusch_godfrey_test,
    durbin_watson_test,
    jarque_bera_test,
    ljung_box_test,
    reset_test,
    white_test,
)

# Structural break tests
from chronobox.tests_stat.structural_breaks import (
    bai_perron_test,
    chow_test,
    cusum_test,
    cusumsq_test,
    qlr_test,
)

# Unit root tests
from chronobox.tests_stat.unit_root import (
    adf_test,
    ers_test,
    hegy_test,
    kpss_test,
    lee_strazicich_test,
    pp_test,
    zivot_andrews_test,
)

__all__ = [
    # Base
    "TestResult",
    # Unit root
    "adf_test",
    "arch_lm_test",
    "bai_perron_test",
    "bds_test",
    "bounds_test",
    "breusch_godfrey_test",
    # Structural breaks
    "chow_test",
    "cusum_test",
    "cusumsq_test",
    "durbin_watson_test",
    # Cointegration
    "engle_granger_test",
    "ers_test",
    "gregory_hansen_test",
    "hegy_test",
    "jarque_bera_test",
    "kpss_test",
    "lee_strazicich_test",
    # Specification
    "ljung_box_test",
    "phillips_ouliaris_test",
    "pp_test",
    "qlr_test",
    "reset_test",
    "white_test",
    "zivot_andrews_test",
]
