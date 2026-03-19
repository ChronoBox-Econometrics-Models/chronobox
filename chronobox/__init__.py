"""chronobox - Time series analysis library with ARIMA and automatic model selection."""

from chronobox.__version__ import __version__
from chronobox.analysis.counterfactual import Counterfactual
from chronobox.analysis.hd import HistoricalDecomposition
from chronobox.decomposition import STL, ClassicalDecomposition
from chronobox.models.arfima import ARFIMA
from chronobox.models.arima import ARIMA
from chronobox.models.bvar import BayesianVAR
from chronobox.models.ets import ETS
from chronobox.models.favar import FAVAR
from chronobox.models.gvar import GVAR
from chronobox.models.holtwinters import HoltWinters
from chronobox.models.svar import SVAR
from chronobox.models.theta import ThetaMethod
from chronobox.models.tvpvar import TVPVAR
from chronobox.models.var import VAR
from chronobox.models.vecm import VECM
from chronobox.selection.auto_arima import auto_arima
from chronobox.selection.auto_ets import auto_ets

__all__ = [
    "ARFIMA",
    "ARIMA",
    "ETS",
    "FAVAR",
    "GVAR",
    "STL",
    "SVAR",
    "TVPVAR",
    "VAR",
    "VECM",
    "BayesianVAR",
    "ClassicalDecomposition",
    "Counterfactual",
    "HistoricalDecomposition",
    "HoltWinters",
    "ThetaMethod",
    "__version__",
    "auto_arima",
    "auto_ets",
]
