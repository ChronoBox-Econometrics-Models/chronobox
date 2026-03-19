"""Auto-ARIMA: Automatic ARIMA order selection via stepwise algorithm."""


from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chronobox._logging import get_logger
from chronobox.core.results import TimeSeriesResults
from chronobox.core.transforms import difference, seasonal_difference
from chronobox.models.arima import ARIMA
from chronobox.utils.validation import validate_endog

logger = get_logger("auto_arima")


def auto_arima(
    y: NDArray[np.float64] | list[float],
    seasonal: bool = True,
    m: int = 1,
    d: int | None = None,
    D: int | None = None,
    max_p: int = 5,
    max_q: int = 5,
    max_P: int = 2,
    max_Q: int = 2,
    max_d: int = 2,
    max_D: int = 1,
    information_criterion: str = "aicc",
    stepwise: bool = True,
    trace: bool = False,
) -> TimeSeriesResults:
    """Automatic ARIMA model selection.

    Implements the Hyndman-Khandakar (2008) stepwise algorithm for
    automatic selection of ARIMA order.

    Parameters
    ----------
    y : array-like
        Time series data.
    seasonal : bool
        Whether to consider seasonal models.
    m : int
        Seasonal period (e.g., 12 for monthly, 4 for quarterly).
        Set to 1 for non-seasonal.
    d : int or None
        Order of first-differencing. If None, determined via ADF test.
    D : int or None
        Order of seasonal differencing. If None, determined automatically.
    max_p : int
        Maximum AR order.
    max_q : int
        Maximum MA order.
    max_P : int
        Maximum seasonal AR order.
    max_Q : int
        Maximum seasonal MA order.
    max_d : int
        Maximum differencing order.
    max_D : int
        Maximum seasonal differencing order.
    information_criterion : str
        Criterion for model selection: 'aic', 'aicc', 'bic'.
    stepwise : bool
        Use stepwise algorithm (True) or exhaustive search (False).
    trace : bool
        Print each model tested and its IC.

    Returns
    -------
    TimeSeriesResults
        Results from the best model.
    """
    y_arr = validate_endog(y)

    # Determine d
    if d is None:
        d = _determine_d(y_arr, max_d)
    if trace:
        print(f"  Determined d = {d}")

    # Determine D
    if seasonal and m > 1:
        if D is None:
            D = _determine_D(y_arr, m, max_D)
        if trace:
            print(f"  Determined D = {D}")
    else:
        D = 0
        m = 1

    s = m if seasonal and m > 1 else 0

    if stepwise:
        return _stepwise_search(
            y_arr, d, D, s, max_p, max_q, max_P, max_Q,
            information_criterion, trace,
        )
    else:
        return _exhaustive_search(
            y_arr, d, D, s, max_p, max_q, max_P, max_Q,
            information_criterion, trace,
        )


def _determine_d(y: NDArray[np.float64], max_d: int) -> int:
    """Determine d via ADF test."""
    try:
        from statsmodels.tsa.stattools import adfuller

        use_statsmodels = True
    except ImportError:
        use_statsmodels = False

    current = y.copy()
    for d_val in range(max_d + 1):
        if d_val > 0:
            current = difference(current, 1)

        if use_statsmodels:
            try:
                result = adfuller(current, autolag="AIC")
                p_value = result[1]
                if p_value < 0.05:
                    return d_val
            except Exception:
                pass
        else:
            # Simple variance-based heuristic
            if d_val > 0:
                var_orig = np.var(y)
                var_diff = np.var(current)
                if var_diff < var_orig * 0.5:
                    return d_val

    return min(1, max_d)


def _determine_D(y: NDArray[np.float64], m: int, max_D: int) -> int:
    """Determine seasonal differencing order D."""
    if max_D == 0 or m <= 1:
        return 0

    var_orig = np.var(y)
    y_sdiff = seasonal_difference(y, m)
    var_sdiff = np.var(y_sdiff)

    if var_sdiff < var_orig:
        return 1
    return 0


def _get_ic(results: TimeSeriesResults, criterion: str) -> float:
    """Get the specified information criterion value."""
    ic_map = {"aic": results.aic, "aicc": results.aicc, "bic": results.bic}
    return ic_map.get(criterion, results.aicc)


def _try_fit(
    y: NDArray[np.float64],
    p: int, d: int, q: int,
    P: int, D: int, Q: int, s: int,
    trace: bool,
    criterion: str,
) -> tuple[TimeSeriesResults | None, float]:
    """Try to fit an ARIMA model, return results and IC or None and inf."""
    try:
        seasonal_order = (P, D, Q, s) if s > 0 else (0, 0, 0, 0)
        model = ARIMA(
            order=(p, d, q),
            seasonal_order=seasonal_order,
        )
        results = model.fit(y, method="css-mle")
        ic = _get_ic(results, criterion)

        if trace:
            name = f"ARIMA({p},{d},{q})"
            if s > 0:
                name += f"({P},{D},{Q})[{s}]"
            print(f"  {name:35s} : {criterion.upper()}={ic:.4f}")

        return results, ic
    except Exception as e:
        if trace:
            name = f"ARIMA({p},{d},{q})"
            if s > 0:
                name += f"({P},{D},{Q})[{s}]"
            print(f"  {name:35s} : FAILED ({e})")
        return None, np.inf


def _stepwise_search(
    y: NDArray[np.float64],
    d: int, D: int, s: int,
    max_p: int, max_q: int, max_P: int, max_Q: int,
    criterion: str, trace: bool,
) -> TimeSeriesResults:
    """Stepwise model selection (Hyndman-Khandakar 2008)."""
    if trace:
        print("Stepwise model selection:")

    # Step 1: Fit initial reference models
    initial_models: list[tuple[int, int, int, int, int, int]] = []

    # ARIMA(0, d, 0)
    initial_models.append((0, 0, 0, 0, D, 0))

    if s > 0:
        # ARIMA(2, d, 2)(1, D, 1)[s]
        initial_models.append((min(2, max_p), min(2, max_q),
                               min(1, max_P), min(1, max_Q), D, 0))
        # ARIMA(1, d, 0)(1, D, 0)[s]
        initial_models.append((min(1, max_p), 0, min(1, max_P), 0, D, 0))
        # ARIMA(0, d, 1)(0, D, 1)[s]
        initial_models.append((0, min(1, max_q), 0, min(1, max_Q), D, 0))
    else:
        # ARIMA(2, d, 2)
        initial_models.append((min(2, max_p), min(2, max_q), 0, 0, D, 0))
        # ARIMA(1, d, 0)
        initial_models.append((min(1, max_p), 0, 0, 0, D, 0))
        # ARIMA(0, d, 1)
        initial_models.append((0, min(1, max_q), 0, 0, D, 0))

    best_results: TimeSeriesResults | None = None
    best_ic = np.inf
    best_order = (0, 0, 0, 0)  # (p, q, P, Q)

    tested: set[tuple[int, int, int, int]] = set()

    for p, q, P, Q, _, _ in initial_models:
        key = (p, q, P, Q)
        if key in tested:
            continue
        tested.add(key)

        results, ic = _try_fit(y, p, d, q, P, D, Q, s, trace, criterion)
        if results is not None and ic < best_ic:
            best_ic = ic
            best_results = results
            best_order = (p, q, P, Q)

    # Step 2: Stepwise exploration
    improved = True
    while improved:
        improved = False
        p_cur, q_cur, P_cur, Q_cur = best_order

        # Generate neighbors
        neighbors: list[tuple[int, int, int, int]] = []

        # Vary p +/- 1
        for dp in [-1, 1]:
            new_p = p_cur + dp
            if 0 <= new_p <= max_p:
                neighbors.append((new_p, q_cur, P_cur, Q_cur))

        # Vary q +/- 1
        for dq in [-1, 1]:
            new_q = q_cur + dq
            if 0 <= new_q <= max_q:
                neighbors.append((p_cur, new_q, P_cur, Q_cur))

        # Vary P +/- 1
        if s > 0:
            for dP in [-1, 1]:
                new_P = P_cur + dP
                if 0 <= new_P <= max_P:
                    neighbors.append((p_cur, q_cur, new_P, Q_cur))

        # Vary Q +/- 1
        if s > 0:
            for dQ in [-1, 1]:
                new_Q = Q_cur + dQ
                if 0 <= new_Q <= max_Q:
                    neighbors.append((p_cur, q_cur, P_cur, new_Q))

        # Vary p and q simultaneously
        for dp, dq in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_p = p_cur + dp
            new_q = q_cur + dq
            if 0 <= new_p <= max_p and 0 <= new_q <= max_q:
                neighbors.append((new_p, new_q, P_cur, Q_cur))

        for p, q, P, Q in neighbors:
            key = (p, q, P, Q)
            if key in tested:
                continue
            tested.add(key)

            results, ic = _try_fit(y, p, d, q, P, D, Q, s, trace, criterion)
            if results is not None and ic < best_ic:
                best_ic = ic
                best_results = results
                best_order = (p, q, P, Q)
                improved = True

    if best_results is None:
        msg = "auto_arima failed to fit any model"
        raise RuntimeError(msg)

    if trace:
        p, q, P, Q = best_order
        name = f"ARIMA({p},{d},{q})"
        if s > 0:
            name += f"({P},{D},{Q})[{s}]"
        print(f"\n  Best model: {name}  {criterion.upper()}={best_ic:.4f}")

    return best_results


def _exhaustive_search(
    y: NDArray[np.float64],
    d: int, D: int, s: int,
    max_p: int, max_q: int, max_P: int, max_Q: int,
    criterion: str, trace: bool,
) -> TimeSeriesResults:
    """Exhaustive search over all order combinations."""
    if trace:
        print("Exhaustive model selection:")

    best_results: TimeSeriesResults | None = None
    best_ic = np.inf

    P_range = range(max_P + 1) if s > 0 else [0]
    Q_range = range(max_Q + 1) if s > 0 else [0]

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in P_range:
                for Q in Q_range:
                    results, ic = _try_fit(
                        y, p, d, q, P, D, Q, s, trace, criterion
                    )
                    if results is not None and ic < best_ic:
                        best_ic = ic
                        best_results = results

    if best_results is None:
        msg = "auto_arima failed to fit any model"
        raise RuntimeError(msg)

    return best_results
