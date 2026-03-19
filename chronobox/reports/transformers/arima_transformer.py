"""ARIMA/SARIMA report transformer.

Extracts parameter table, information criteria, diagnostic tests,
residual plot, forecast, and polynomial roots from ARIMA results.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chronobox.reports.transformers.base_transformer import BaseTransformer


class ARIMATransformer(BaseTransformer):
    """Transform ARIMA/SARIMA results into report context.

    Extracted sections:
    - Model summary (type, order, nobs)
    - Parameter table (name, value, SE, t-stat, p-value, significance)
    - Information criteria (AIC, BIC, HQIC, AICc, log-likelihood)
    - Diagnostics (Ljung-Box, Jarque-Bera, ARCH-LM)
    - Residual diagnostics plot
    - Forecast table and plot
    - AR/MA polynomial roots
    """

    def transform(self, results: Any, **kwargs: Any) -> dict[str, Any]:
        """Transform ARIMA results to template context.

        Parameters
        ----------
        results : ARIMAResults or similar
            ARIMA model results.
        **kwargs
            Additional arguments.

        Returns
        -------
        dict[str, Any]
            Template context.
        """
        context: dict[str, Any] = {
            "title": "ARIMA Model Report",
            "model_info": self._extract_model_info(results),
            "sections": [],
            "tables": {},
            "plots": {},
        }

        # Parameter table
        params_table = self._extract_params(results)
        if params_table:
            context["tables"]["parameters"] = params_table
            context["sections"].append(
                self._build_section("Parameter Estimates", table=params_table)
            )

        # Information criteria
        ic = self._extract_ic(results)
        if ic:
            context["information_criteria"] = ic
            context["sections"].append(
                self._build_section(
                    "Information Criteria",
                    content=self._ic_to_html(ic),
                )
            )

        # Diagnostics
        diagnostics = self._extract_diagnostics(results)
        if diagnostics:
            context["diagnostics"] = diagnostics
            context["sections"].append(
                self._build_section(
                    "Diagnostic Tests",
                    table=diagnostics,
                )
            )

        # Roots
        roots = self._extract_roots(results)
        if roots:
            context["roots"] = roots
            context["sections"].append(
                self._build_section(
                    "Polynomial Roots",
                    table=roots,
                    collapsible=True,
                )
            )

        return context

    def _extract_model_info(self, results: Any) -> dict[str, Any]:
        """Extract model information.

        Parameters
        ----------
        results : Any
            ARIMA results.

        Returns
        -------
        dict
            Model information.
        """
        info: dict[str, Any] = {"model_type": "ARIMA"}

        for attr, key in [
            ("order", "order"),
            ("seasonal_order", "seasonal_order"),
            ("nobs", "nobs"),
            ("n_obs", "nobs"),
        ]:
            val = self._safe_getattr(results, attr)
            if val is not None:
                info[key] = val

        # Determine if SARIMA
        seasonal = self._safe_getattr(results, "seasonal_order")
        if seasonal is not None and any(s > 0 for s in seasonal[:3]):
            info["model_type"] = "SARIMA"

        return info

    def _extract_params(self, results: Any) -> list[dict[str, Any]]:
        """Extract parameter estimates.

        Parameters
        ----------
        results : Any
            Model results.

        Returns
        -------
        list[dict]
            Parameter table rows.
        """
        params = self._safe_getattr(results, "params")
        if params is None:
            params = self._safe_getattr(results, "parameters")
        if params is None:
            return []

        params = np.asarray(params, dtype=np.float64)
        n = len(params)

        # Get parameter names
        names = self._safe_getattr(results, "param_names")
        if names is None:
            names = self._safe_getattr(results, "parameter_names")
        if names is None:
            names = [f"param_{i}" for i in range(n)]

        # Standard errors, t-stats, p-values
        bse = self._safe_getattr(results, "bse")
        if bse is None:
            bse = self._safe_getattr(results, "std_errors")

        tvalues = self._safe_getattr(results, "tvalues")
        if tvalues is None:
            tvalues = self._safe_getattr(results, "t_stats")

        pvalues = self._safe_getattr(results, "pvalues")
        if pvalues is None:
            pvalues = self._safe_getattr(results, "p_values")

        return self._format_params_table(
            names=list(names),
            values=params,
            std_errors=np.asarray(bse) if bse is not None else None,
            t_stats=np.asarray(tvalues) if tvalues is not None else None,
            p_values=np.asarray(pvalues) if pvalues is not None else None,
        )

    def _extract_ic(self, results: Any) -> dict[str, float | None]:
        """Extract information criteria.

        Parameters
        ----------
        results : Any
            Model results.

        Returns
        -------
        dict
            Information criteria.
        """
        return self._format_ic_table(
            aic=self._safe_getattr(results, "aic"),
            bic=self._safe_getattr(results, "bic"),
            hqic=self._safe_getattr(results, "hqic"),
            aicc=self._safe_getattr(results, "aicc"),
            loglike=self._safe_getattr(
                results, "llf", self._safe_getattr(results, "loglike")
            ),
        )

    def _extract_diagnostics(self, results: Any) -> list[dict[str, Any]]:
        """Extract diagnostic test results.

        Parameters
        ----------
        results : Any
            Model results.

        Returns
        -------
        list[dict]
            Diagnostic test table rows.
        """
        diags: list[dict[str, Any]] = []

        # Ljung-Box
        lb = self._safe_getattr(results, "ljung_box")
        if lb is not None:
            if isinstance(lb, dict):
                diags.append({
                    "test": "Ljung-Box",
                    "statistic": lb.get("statistic"),
                    "p_value": lb.get("p_value"),
                    "conclusion": "No autocorrelation"
                    if lb.get("p_value", 0) > 0.05
                    else "Autocorrelation detected",
                })
            elif hasattr(lb, "statistic"):
                diags.append({
                    "test": "Ljung-Box",
                    "statistic": float(lb.statistic),
                    "p_value": float(lb.pvalue) if hasattr(lb, "pvalue") else None,
                })

        # Jarque-Bera
        jb = self._safe_getattr(results, "jarque_bera")
        if jb is not None:
            if isinstance(jb, dict):
                diags.append({
                    "test": "Jarque-Bera",
                    "statistic": jb.get("statistic"),
                    "p_value": jb.get("p_value"),
                    "conclusion": "Normal"
                    if jb.get("p_value", 0) > 0.05
                    else "Non-normal",
                })
            elif isinstance(jb, (tuple, list)):
                diags.append({
                    "test": "Jarque-Bera",
                    "statistic": float(jb[0]),
                    "p_value": float(jb[1]) if len(jb) > 1 else None,
                })

        # ARCH-LM
        arch = self._safe_getattr(results, "arch_lm")
        if arch is not None and isinstance(arch, dict):
            diags.append({
                "test": "ARCH-LM",
                "statistic": arch.get("statistic"),
                "p_value": arch.get("p_value"),
                "conclusion": "No ARCH"
                if arch.get("p_value", 0) > 0.05
                else "ARCH effects",
            })

        return diags

    def _extract_roots(self, results: Any) -> list[dict[str, Any]]:
        """Extract AR/MA polynomial roots.

        Parameters
        ----------
        results : Any
            Model results.

        Returns
        -------
        list[dict]
            Roots table.
        """
        roots: list[dict[str, Any]] = []

        ar_roots = self._safe_getattr(results, "arroots")
        if ar_roots is None:
            ar_roots = self._safe_getattr(results, "ar_roots")
        if ar_roots is not None:
            ar_roots = np.asarray(ar_roots)
            for i, root in enumerate(ar_roots):
                roots.append({
                    "type": "AR",
                    "index": i + 1,
                    "real": float(np.real(root)),
                    "imag": float(np.imag(root)),
                    "modulus": float(np.abs(root)),
                    "inside_unit_circle": float(np.abs(root)) > 1.0,
                })

        ma_roots = self._safe_getattr(results, "maroots")
        if ma_roots is None:
            ma_roots = self._safe_getattr(results, "ma_roots")
        if ma_roots is not None:
            ma_roots = np.asarray(ma_roots)
            for i, root in enumerate(ma_roots):
                roots.append({
                    "type": "MA",
                    "index": i + 1,
                    "real": float(np.real(root)),
                    "imag": float(np.imag(root)),
                    "modulus": float(np.abs(root)),
                    "inside_unit_circle": float(np.abs(root)) > 1.0,
                })

        return roots

    @staticmethod
    def _ic_to_html(ic: dict[str, float | None]) -> str:
        """Convert IC dict to simple HTML table.

        Parameters
        ----------
        ic : dict
            Information criteria.

        Returns
        -------
        str
            HTML table.
        """
        rows = ""
        for name, val in ic.items():
            if val is not None:
                rows += f"<tr><td>{name}</td><td class='num'>{val:.4f}</td></tr>\n"
        return (
            f"<table class='cb-table'><thead><tr><th>Criterion</th>"
            f"<th>Value</th></tr></thead><tbody>{rows}</tbody></table>"
        )
