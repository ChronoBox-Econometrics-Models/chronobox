"""Automatic ETS model selection.

Selects the best ETS model from all valid combinations by minimizing AICc.

References
----------
- Hyndman, R.J., Koehler, A.B., Snyder, R.D. & Grose, S. (2002). A state space
  framework for automatic forecasting using exponential smoothing methods.
  International Journal of Forecasting, 18(3), 439-454.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chronobox.models.ets import ETS, ETSResults
from chronobox.utils.validation import validate_endog


@dataclass
class AutoETSResult:
    """Result of automatic ETS model selection.

    Attributes
    ----------
    best_model : ETSResults
        The best-fitting ETS model results.
    best_spec : tuple[str, str, str]
        (error, trend, seasonal) specification of the best model.
    all_results : list[tuple[tuple[str, str, str], float]]
        All fitted models with their AICc values, sorted by AICc.
    n_models_tried : int
        Number of models successfully fitted.
    n_models_failed : int
        Number of models that failed to fit.
    """

    best_model: ETSResults
    best_spec: tuple[str, str, str]
    all_results: list[tuple[tuple[str, str, str], float]]
    n_models_tried: int
    n_models_failed: int

    def summary(self) -> str:
        """Return a formatted summary of the auto-ETS selection."""
        e, t, s = self.best_spec
        lines = [
            "=" * 70,
            f"{'Auto-ETS Model Selection Results':^70}",
            "=" * 70,
            f"Best Model:         ETS({e},{t},{s})",
            f"AICc:               {self.best_model.aicc:.4f}",
            f"Models Tried:       {self.n_models_tried}",
            f"Models Failed:      {self.n_models_failed}",
            "-" * 70,
            f"{'Top 5 Models':^70}",
            "-" * 70,
            f"{'Rank':<6} {'Model':<20} {'AICc':>12}",
            "-" * 70,
        ]
        for i, (spec, aicc) in enumerate(self.all_results[:5]):
            e_i, t_i, s_i = spec
            model_str = f"ETS({e_i},{t_i},{s_i})"
            lines.append(f"{i + 1:<6} {model_str:<20} {aicc:>12.4f}")
        lines.extend([
            "-" * 70,
            "",
            self.best_model.summary(),
        ])
        return "\n".join(lines)

    def forecast(self, steps: int = 1) -> NDArray[np.float64]:
        """Forecast using the best model.

        Parameters
        ----------
        steps : int
            Number of steps ahead.

        Returns
        -------
        NDArray[np.float64]
            Forecasted values.
        """
        return self.best_model.forecast(steps=steps)


def auto_ets(
    endog: NDArray[np.float64] | list[float],
    seasonal_period: int | None = None,
    error_types: list[str] | None = None,
    trend_types: list[str] | None = None,
    seasonal_types: list[str] | None = None,
    allow_multiplicative_trend: bool = True,
    damped: bool | None = None,
    information_criterion: str = "aicc",
    restrict: bool = True,
    verbose: bool = False,
) -> AutoETSResult:
    """Automatically select the best ETS model.

    Fits all valid ETS model combinations and selects the best one by
    information criterion (AICc by default).

    Parameters
    ----------
    endog : array-like
        Time series data.
    seasonal_period : int or None
        Seasonal period. If None or 1, only non-seasonal models are tried.
    error_types : list of str or None
        Error types to try. Default: ['A', 'M'] (if data positive) or ['A'].
    trend_types : list of str or None
        Trend types to try. Default: ['N', 'A', 'Ad', 'M', 'Md'] or subset.
    seasonal_types : list of str or None
        Seasonal types to try. Default depends on seasonal_period.
    allow_multiplicative_trend : bool
        Whether to include multiplicative trend models. Default True.
    damped : bool or None
        If True, only try damped trends. If False, only undamped.
        If None, try both.
    information_criterion : str
        'aic', 'bic', or 'aicc'. Default 'aicc'.
    restrict : bool
        If True, restrict models to common stable combinations.
    verbose : bool
        Print progress information.

    Returns
    -------
    AutoETSResult
        Result containing the best model and comparison information.

    Examples
    --------
    >>> import numpy as np
    >>> from chronobox.selection.auto_ets import auto_ets
    >>> from chronobox.datasets import load_dataset
    >>> airline = load_dataset('airline')
    >>> y = airline['passengers'].values.astype(float)
    >>> result = auto_ets(y, seasonal_period=12)
    >>> print(result.summary())
    """
    y = validate_endog(endog)
    all_positive = bool(np.all(y > 0))

    # Determine seasonal period
    m = seasonal_period if seasonal_period is not None else 1
    has_seasonal = m >= 2

    # Determine candidate components
    if error_types is None:
        error_types = ["A", "M"] if all_positive else ["A"]

    if trend_types is None:
        base_trends = (
            ["N", "A", "M"] if allow_multiplicative_trend and all_positive
            else ["N", "A"]
        )

        # Add damped versions
        trend_types = []
        for t in base_trends:
            if damped is None:
                trend_types.append(t)
                if t in ("A", "M"):
                    trend_types.append(t + "d")
            elif damped:
                if t in ("A", "M"):
                    trend_types.append(t + "d")
                else:
                    trend_types.append(t)
            else:
                trend_types.append(t)

    if seasonal_types is None:
        if has_seasonal:
            seasonal_types = ["N", "A", "M"] if all_positive else ["N", "A"]
        else:
            seasonal_types = ["N"]

    # Build candidate models
    candidates: list[tuple[str, str, str]] = []
    for e in error_types:
        for t in trend_types:
            for s in seasonal_types:
                # Skip invalid combinations
                if e == "M" and not all_positive:
                    continue
                if s == "M" and not all_positive:
                    continue
                if t in ("M", "Md") and not all_positive:
                    continue
                if s != "N" and not has_seasonal:
                    continue

                # Restrict numerically unstable combinations
                if restrict and e == "M" and t in ("M", "Md") and s == "M":
                    continue

                candidates.append((e, t, s))

    if not candidates:
        msg = "No valid ETS model candidates found for the given data"
        raise ValueError(msg)

    # Fit all candidates
    results: list[tuple[tuple[str, str, str], float, ETSResults]] = []
    n_failed = 0

    for spec in candidates:
        e, t, s = spec
        try:
            sp = m if s != "N" else None
            model = ETS(error=e, trend=t, seasonal=s, seasonal_period=sp)
            res = model.fit(y)

            # Get information criterion
            if information_criterion == "aicc":
                ic = res.aicc
            elif information_criterion == "aic":
                ic = res.aic
            elif information_criterion == "bic":
                ic = res.bic
            else:
                ic = res.aicc

            if np.isfinite(ic):
                results.append((spec, ic, res))
                if verbose:
                    print(f"  ETS({e},{t},{s}): {information_criterion}={ic:.4f}")
            else:
                n_failed += 1
                if verbose:
                    print(f"  ETS({e},{t},{s}): non-finite {information_criterion}")

        except Exception as ex:
            n_failed += 1
            if verbose:
                print(f"  ETS({e},{t},{s}): FAILED - {ex}")

    if not results:
        msg = "All ETS model candidates failed to fit"
        raise RuntimeError(msg)

    # Sort by information criterion
    results.sort(key=lambda x: x[1])

    best_spec, best_ic, best_model = results[0]

    all_ic = [(spec, ic) for spec, ic, _ in results]

    if verbose:
        e, t, s = best_spec
        print(
            f"\nBest model: ETS({e},{t},{s}) "
            f"with {information_criterion}={best_ic:.4f}"
        )

    return AutoETSResult(
        best_model=best_model,
        best_spec=best_spec,
        all_results=all_ic,
        n_models_tried=len(results),
        n_models_failed=n_failed,
    )
