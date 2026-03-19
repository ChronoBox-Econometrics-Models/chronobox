"""X-13ARIMA-SEATS wrapper.

Provides a Python interface to the X-13ARIMA-SEATS seasonal adjustment program
from the US Census Bureau.

Note: Requires the x13as executable to be installed and accessible.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
from numpy.typing import NDArray

from chronobox.decomposition.stl import DecompositionResult
from chronobox.utils.validation import validate_endog


def _find_x13_executable() -> str | None:
    """Try to find the X-13ARIMA-SEATS executable."""
    # Common names
    names = ["x13as", "x13ashtml", "x13as_ascii"]

    for name in names:
        path = shutil.which(name)
        if path is not None:
            return path

    # Check common locations
    common_paths = [
        "/usr/local/bin/x13as",
        "/usr/bin/x13as",
        os.path.expanduser("~/x13as/x13as"),
    ]
    for p in common_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    return None


class X13Wrapper:
    """Wrapper for X-13ARIMA-SEATS seasonal adjustment.

    Requires the X-13ARIMA-SEATS executable to be installed.
    Download from: https://www.census.gov/data/software/x13as.html

    Parameters
    ----------
    x13_path : str or None
        Path to the X-13 executable. If None, tries to auto-detect.

    Examples
    --------
    >>> from chronobox.decomposition.x13_wrapper import X13Wrapper
    >>> x13 = X13Wrapper()
    >>> # result = x13.seasonal_adjust(y, period=12)  # requires x13as
    """

    def __init__(self, x13_path: str | None = None) -> None:
        if x13_path is not None:
            if not os.path.isfile(x13_path):
                msg = f"X-13 executable not found at: {x13_path}"
                raise FileNotFoundError(msg)
            self.x13_path = x13_path
        else:
            detected = _find_x13_executable()
            if detected is None:
                self.x13_path = None
                warnings.warn(
                    "X-13ARIMA-SEATS executable not found. "
                    "Install from https://www.census.gov/data/software/x13as.html "
                    "or provide the path via x13_path parameter.",
                    stacklevel=2,
                )
            else:
                self.x13_path = detected

    @property
    def is_available(self) -> bool:
        """Check if X-13 executable is available."""
        return self.x13_path is not None

    def seasonal_adjust(
        self,
        endog: NDArray[np.float64] | list[float],
        period: int = 12,
        start_year: int = 2000,
        start_month: int = 1,
        transform: str = "auto",
    ) -> DecompositionResult:
        """Perform seasonal adjustment using X-13ARIMA-SEATS.

        Parameters
        ----------
        endog : array-like
            Time series data.
        period : int
            Seasonal period (e.g., 12 for monthly, 4 for quarterly).
        start_year : int
            Starting year of the series.
        start_month : int
            Starting month/quarter.
        transform : str
            Transformation: 'auto', 'log', or 'none'.

        Returns
        -------
        DecompositionResult
            Seasonal adjustment results.

        Raises
        ------
        RuntimeError
            If X-13 is not available or execution fails.
        """
        if not self.is_available:
            msg = (
                "X-13ARIMA-SEATS executable not available. "
                "Install it or provide the path via x13_path parameter."
            )
            raise RuntimeError(msg)

        y = validate_endog(endog)
        n = len(y)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "series")

            # Write data file
            data_file = base + ".dat"
            with open(data_file, "w") as f:
                for val in y:
                    f.write(f"{val:.6f}\n")

            # Write spec file
            spec_file = base + ".spc"
            period_str = str(period)

            transform_spec = ""
            if transform == "log":
                transform_spec = "transform { function = log }"
            elif transform == "auto":
                transform_spec = "transform { function = auto }"

            spec_content = f"""series {{
    file = "{data_file}"
    format = "free"
    start = {start_year}.{start_month:02d}
    period = {period_str}
}}
{transform_spec}
x11 {{
    save = (d10 d11 d12 d13)
}}
"""
            with open(spec_file, "w") as f:
                f.write(spec_content)

            # Run X-13
            try:
                result = subprocess.run(
                    [self.x13_path, base],  # type: ignore[list-item]
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                msg = "X-13 execution timed out"
                raise RuntimeError(msg) from e
            except Exception as e:
                msg = f"X-13 execution failed: {e}"
                raise RuntimeError(msg) from e

            if result.returncode != 0:
                msg = f"X-13 returned error code {result.returncode}: {result.stderr}"
                raise RuntimeError(msg)

            # Read results
            try:
                trend = self._read_output(base + ".d12", n)
                seasonal = self._read_output(base + ".d10", n)
                self._read_output(base + ".d11", n)  # seasonally adjusted
                remainder = self._read_output(base + ".d13", n)
            except FileNotFoundError:
                # Try alternative output files
                trend = np.full(n, np.nan)
                seasonal = np.full(n, np.nan)
                remainder = np.full(n, np.nan)
                warnings.warn(
                    "Could not read all X-13 output files. "
                    "Results may be incomplete.",
                    stacklevel=2,
                )

        return DecompositionResult(
            observed=y.copy(),
            trend=trend,
            seasonal=seasonal,
            remainder=remainder,
            weights=None,
            period=period,
            model="x13",
        )

    def _read_output(self, filepath: str, expected_length: int) -> NDArray[np.float64]:
        """Read an X-13 output file."""
        if not os.path.exists(filepath):
            msg = f"X-13 output file not found: {filepath}"
            raise FileNotFoundError(msg)

        values: list[float] = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("-"):
                    continue
                parts = line.split()
                for p in parts:
                    try:
                        values.append(float(p))
                    except ValueError:
                        continue

        arr = np.array(values, dtype=np.float64)
        if len(arr) != expected_length:
            if len(arr) > expected_length:
                arr = arr[:expected_length]
            else:
                arr = np.pad(
                    arr,
                    (0, expected_length - len(arr)),
                    constant_values=np.nan,
                )
        return arr
