"""VAR report transformer.

Extracts coefficients by equation, stability analysis, Granger causality,
IRF/FEVD plots, and forecast from VAR results.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chronobox.reports.transformers.base_transformer import BaseTransformer


class VARTransformer(BaseTransformer):
    """Transform VAR/VECM results into report context.

    Sections:
    - Model info (K variables, p lags, nobs)
    - Coefficients by equation
    - Stability (companion eigenvalues)
    - Granger causality matrix
    - IRF plots
    - FEVD plots
    - Forecast
    """

    def transform(self, results: Any, **kwargs: Any) -> dict[str, Any]:
        """Transform VAR results.

        Parameters
        ----------
        results : VARResults
            VAR model results.

        Returns
        -------
        dict
            Template context.
        """
        context: dict[str, Any] = {
            "title": "VAR Model Report",
            "model_info": self._extract_model_info(results),
            "sections": [],
            "tables": {},
            "plots": {},
        }

        # Coefficients by equation
        coefs = self._extract_coefficients(results)
        if coefs:
            context["tables"]["coefficients"] = coefs
            for eq_name, eq_table in coefs.items():
                context["sections"].append(
                    self._build_section(
                        f"Equation: {eq_name}",
                        table=eq_table,
                        collapsible=True,
                    )
                )

        # Stability
        stability = self._extract_stability(results)
        if stability:
            context["stability"] = stability
            context["sections"].append(
                self._build_section(
                    "Stability Analysis",
                    content=stability.get("summary", ""),
                    table=stability.get("eigenvalues_table"),
                )
            )

        # Granger causality
        granger = self._extract_granger(results)
        if granger:
            context["tables"]["granger"] = granger
            context["sections"].append(
                self._build_section("Granger Causality", table=granger)
            )

        # IRF info
        irf_info = self._extract_irf_info(results)
        if irf_info:
            context["irf"] = irf_info
            context["sections"].append(
                self._build_section(
                    "Impulse Response Functions", content="IRF plots below."
                )
            )

        # FEVD info
        fevd_info = self._extract_fevd_info(results)
        if fevd_info:
            context["fevd"] = fevd_info
            context["sections"].append(
                self._build_section(
                    "Forecast Error Variance Decomposition",
                    content="FEVD plots below.",
                )
            )

        return context

    def _extract_model_info(self, results: Any) -> dict[str, Any]:
        """Extract model information."""
        info: dict[str, Any] = {"model_type": "VAR"}

        for attr, key in [
            ("k_ar", "lags"),
            ("neqs", "n_equations"),
            ("nobs", "nobs"),
            ("k_trend", "trend_terms"),
        ]:
            val = self._safe_getattr(results, attr)
            if val is not None:
                info[key] = val

        var_names = self._safe_getattr(results, "names")
        if var_names is None:
            var_names = self._safe_getattr(results, "var_names")
        if var_names is not None:
            info["variables"] = list(var_names)

        return info

    def _extract_coefficients(
        self, results: Any
    ) -> dict[str, list[dict[str, Any]]]:
        """Extract coefficients by equation."""
        coefs: dict[str, list[dict[str, Any]]] = {}

        params = self._safe_getattr(results, "params")
        if params is None:
            params = self._safe_getattr(results, "coefs")
        if params is None:
            return coefs

        var_names = self._safe_getattr(
            results, "names"
        ) or self._safe_getattr(results, "var_names")
        if var_names is None:
            return coefs

        params = np.asarray(params)
        bse = self._safe_getattr(results, "bse")
        tvalues = self._safe_getattr(results, "tvalues")
        pvalues = self._safe_getattr(results, "pvalues")

        # Build coefficient tables per equation
        for i, name in enumerate(var_names):
            if params.ndim == 2 and i < params.shape[1]:
                eq_params = params[:, i]
            elif params.ndim == 3:
                eq_params = params[:, :, i].ravel()
            else:
                continue

            n_params = len(eq_params)
            param_names = [
                f"L{j + 1}.{vn}"
                for j in range(n_params // len(var_names) + 1)
                for vn in var_names
            ][:n_params]

            eq_bse = None
            eq_tval = None
            eq_pval = None
            if bse is not None:
                bse_arr = np.asarray(bse)
                if bse_arr.ndim == 2 and i < bse_arr.shape[1]:
                    eq_bse = bse_arr[:, i]
            if tvalues is not None:
                tv_arr = np.asarray(tvalues)
                if tv_arr.ndim == 2 and i < tv_arr.shape[1]:
                    eq_tval = tv_arr[:, i]
            if pvalues is not None:
                pv_arr = np.asarray(pvalues)
                if pv_arr.ndim == 2 and i < pv_arr.shape[1]:
                    eq_pval = pv_arr[:, i]

            coefs[str(name)] = self._format_params_table(
                names=param_names,
                values=eq_params,
                std_errors=eq_bse,
                t_stats=eq_tval,
                p_values=eq_pval,
            )

        return coefs

    def _extract_stability(self, results: Any) -> dict[str, Any] | None:
        """Extract stability analysis (companion eigenvalues)."""
        eigenvalues = self._safe_getattr(results, "eigenvalues")
        if eigenvalues is None:
            eigenvalues = self._safe_getattr(results, "roots")
        if eigenvalues is None:
            return None

        eigenvalues = np.asarray(eigenvalues)
        moduli = np.abs(eigenvalues)
        is_stable = bool(np.all(moduli < 1.0))

        table = []
        for i, ev in enumerate(eigenvalues):
            table.append({
                "index": i + 1,
                "real": float(np.real(ev)),
                "imag": float(np.imag(ev)),
                "modulus": float(np.abs(ev)),
            })

        summary = (
            f"System is {'stable' if is_stable else 'UNSTABLE'}. "
            f"Max eigenvalue modulus: {float(np.max(moduli)):.4f}"
        )

        return {
            "is_stable": is_stable,
            "max_modulus": float(np.max(moduli)),
            "eigenvalues_table": table,
            "summary": summary,
        }

    def _extract_granger(self, results: Any) -> list[dict[str, Any]] | None:
        """Extract Granger causality matrix."""
        granger = self._safe_getattr(results, "granger_causality")
        if granger is None:
            return None

        if isinstance(granger, dict):
            table = []
            for key, val in granger.items():
                if isinstance(val, dict):
                    table.append({
                        "test": str(key),
                        "statistic": val.get("statistic"),
                        "p_value": val.get("p_value"),
                    })
            return table if table else None

        return None

    def _extract_irf_info(self, results: Any) -> dict[str, Any] | None:
        """Extract IRF availability info."""
        irf = self._safe_getattr(results, "irf")
        if irf is None:
            irf = self._safe_getattr(results, "irfs")
        if irf is not None:
            return {"available": True}
        return None

    def _extract_fevd_info(self, results: Any) -> dict[str, Any] | None:
        """Extract FEVD availability info."""
        fevd = self._safe_getattr(results, "fevd")
        if fevd is None:
            fevd = self._safe_getattr(results, "variance_decomp")
        if fevd is not None:
            return {"available": True}
        return None
