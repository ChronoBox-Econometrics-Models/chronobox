---
title: "Contributing Guide"
description: "How to contribute to ChronoBox — setup, code standards, templates, and PR process."
---

# Contributing to ChronoBox

Thank you for your interest in contributing to ChronoBox! Whether you are reporting a bug, proposing a feature, improving documentation, or submitting code, your help is welcome and appreciated.

## Types of Contributions

| Type | Where | Description |
|------|-------|-------------|
| Bug reports | [GitHub Issues](https://github.com/NodesEcon/chronobox/issues) | Reproducible problem with expected vs. actual behavior |
| Feature requests | [GitHub Issues](https://github.com/NodesEcon/chronobox/issues) | Proposals with `[Feature]` label |
| Code (PR) | [Pull Requests](https://github.com/NodesEcon/chronobox/pulls) | New models, tests, bug fixes |
| Documentation | `docs/` directory | Tutorials, API docs, examples |
| Test additions | `tests/` directory | Unit, integration, and validation tests |

---

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/NodesEcon/chronobox.git
cd chronobox
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows
```

### 3. Install in Development Mode

```bash
pip install -e ".[dev,all]"
```

### 4. Install Pre-Commit Hooks

```bash
pre-commit install
```

### 5. Verify Setup

```bash
pytest tests/ -v --timeout=60
```

### Development Dependencies

| Tool | Version | Purpose |
|------|---------|---------|
| pytest | >= 8.0 | Testing (with xdist, timeout) |
| pytest-cov | >= 5.0 | Coverage measurement |
| ruff | >= 0.15.0 | Linting and formatting |
| pyright | >= 1.1.400 | Static type checking |
| interrogate | >= 1.7.0 | Docstring coverage |

---

## Code Standards

### Style

- **Formatter**: ruff format (line length 88)
- **Import sorting**: ruff (isort-compatible)
- **Type hints**: Required for all public API functions and methods
- **Docstrings**: NumPy-style for all public classes and methods
- **Python version**: 3.10+ compatibility

```bash
# Format and lint
ruff format chronobox/
ruff check chronobox/ --fix

# Type check
pyright chronobox/

# Run all hooks manually
pre-commit run --all-files
```

### Branch Naming

Use descriptive branch names:

- `feature/add-garch-model` — New features
- `fix/arima-forecast-ci` — Bug fixes
- `docs/update-var-tutorial` — Documentation changes
- `test/add-vecm-tests` — Test additions

### Commit Messages

```text
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `style`, `chore`

**Example**:

```text
feat(var): Add bootstrapped IRF confidence intervals

Implements Efron (1979) bootstrap for impulse response
confidence bands with bias-corrected percentile method.

Closes #45
```

---

## Reporting Bugs

File issues on [GitHub](https://github.com/NodesEcon/chronobox/issues) with:

1. A clear title describing the problem
2. **Minimal reproducible example** (MRE)
3. Expected vs. actual behavior
4. ChronoBox version: `pip show chronobox`
5. Python version: `python --version`

### Bug Report Template

```markdown
**Describe the bug**
A clear description of the bug.

**To Reproduce**
```python
import chronobox as cb

# Minimal code to reproduce the issue
data = cb.datasets.load_airline()
model = cb.ARIMA(data, order=(1, 1, 1))
result = model.fit()
# Error occurs here
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened (include traceback if applicable).

**Environment**
- ChronoBox version:
- Python version:
- OS:
- NumPy/SciPy versions:
```

---

## Suggesting Features

Open a [GitHub Issue](https://github.com/NodesEcon/chronobox/issues) with the `[Feature]` label. Include:

1. **Use case**: What problem does it solve?
2. **Description**: What should the feature do?
3. **References**: Academic papers, existing implementations (R, Stata)
4. **API sketch**: How would users call it?

### Feature Request Template

```markdown
**Feature description**
A clear description of the proposed feature.

**Use case**
Why is this feature important for time series analysis?

**Proposed API**
```python
# How the feature would be used
model = cb.GARCH(data, p=1, q=1)
result = model.fit()
result.conditional_volatility()
```

**References**
- Author (Year). "Title". *Journal*.
- R package: `rugarch`
- Stata command: `arch`
```

---

## Git Workflow

### Fork → Branch → PR

```
main (upstream)
  │
  ├── fork (your GitHub copy)
  │     │
  │     ├── feature/my-feature (your branch)
  │     │     ├── commit 1
  │     │     ├── commit 2
  │     │     └── commit 3
  │     │
  │     └── PR → upstream/main
  │
  └── merge
```

### Step-by-Step

1. **Fork** the repository on GitHub

2. **Clone** your fork:
    ```bash
    git clone https://github.com/YOUR-USERNAME/chronobox.git
    cd chronobox
    git remote add upstream https://github.com/NodesEcon/chronobox.git
    ```

3. **Create a feature branch**:
    ```bash
    git checkout -b feature/my-new-feature
    ```

4. **Make changes**: code, tests, documentation.

5. **Run checks locally**:
    ```bash
    pytest tests/ -v
    pre-commit run --all-files
    ```

6. **Commit with a clear message**:
    ```bash
    git commit -m "feat(ets): Add damped trend component

    Implements Gardner & McKenzie (1985) damped trend
    for ETS models with phi parameter.

    Closes #42"
    ```

7. **Push and open a PR**:
    ```bash
    git push origin feature/my-new-feature
    ```

8. **Fill out the PR template and address review comments.**

---

## Adding a New Model

Place your code in the appropriate module. Every model must:

1. Inherit from the appropriate base class
2. Implement `fit()` returning a results object
3. Implement `forecast()` for prediction
4. Have comprehensive tests
5. Be exported from the package `__init__.py`
6. Have a documentation page

### Model Template

```python
"""My model module.

Implements the Author (Year) model for time series analysis.
"""

import numpy as np
import pandas as pd

from chronobox.models.base import TimeSeriesModel


class MyModel(TimeSeriesModel):
    """Short description of the model.

    Longer description explaining the model, its assumptions,
    and when to use it.

    Parameters
    ----------
    endog : array_like
        Endogenous variable (time series).
    order : tuple
        Model order specification.
    trend : str, optional
        Trend component ('n', 'c', 't', 'ct').

    References
    ----------
    .. [1] Author, A. (Year). Title. *Journal*, vol(issue), pages.
    """

    def __init__(self, endog, order, trend="c", **kwargs):
        super().__init__(endog)
        self.order = order
        self.trend = trend

    def fit(self, method="mle", **kwargs):
        """Estimate the model.

        Parameters
        ----------
        method : str, default='mle'
            Estimation method.

        Returns
        -------
        ModelResult
            Fitted model results with params, residuals,
            information criteria, summary().
        """
        # 1. Estimate parameters
        # 2. Compute residuals and fitted values
        # 3. Return results object
        ...

    def forecast(self, steps=1, alpha=0.05):
        """Generate out-of-sample forecasts.

        Parameters
        ----------
        steps : int, default=1
            Number of forecast steps.
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns
        -------
        ForecastResult
            Point forecasts with confidence intervals.
        """
        ...
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/models/arima/ -v
pytest tests/models/var/ -v

# Specific test
pytest tests/models/var/test_irf.py::test_bootstrap_ci -v

# With coverage
pytest tests/ --cov=chronobox --cov-report=html --cov-branch

# In parallel
pytest tests/ -n auto
```

### Writing Tests

```python
import pytest
import numpy as np
from chronobox.models.arima import ARIMA
from chronobox.datasets import load_airline


class TestMyModel:
    """Tests for MyModel."""

    @pytest.fixture
    def ts_data(self):
        """Load standard test dataset."""
        return load_airline()

    def test_basic_estimation(self, ts_data):
        """Test that estimation runs and returns results."""
        model = ARIMA(ts_data, order=(1, 1, 1))
        result = model.fit()
        assert result.params is not None
        assert result.aic is not None

    def test_forecast_length(self, ts_data):
        """Test forecast returns correct number of steps."""
        model = ARIMA(ts_data, order=(1, 1, 1))
        result = model.fit()
        fc = result.forecast(steps=12)
        assert len(fc.mean) == 12

    def test_residuals_white_noise(self, ts_data):
        """Test that residuals are approximately white noise."""
        model = ARIMA(ts_data, order=(1, 1, 1))
        result = model.fit()
        # Ljung-Box test on residuals
        assert result.ljung_box(lags=10).pvalue > 0.05
```

### Validation Against R or Stata

For models with R or Stata equivalents, add validation tests:

```python
def test_against_r_reference(self):
    """Validate against R forecast package."""
    # R reference values (from validated R script)
    r_aic = 1012.45
    r_coefficients = {"ar1": 0.8423, "ma1": -0.5031}

    model = ARIMA(data, order=(1, 1, 1))
    result = model.fit()

    np.testing.assert_allclose(
        result.aic, r_aic, rtol=1e-2,
        err_msg="AIC doesn't match R"
    )
    for param, r_val in r_coefficients.items():
        np.testing.assert_allclose(
            result.params[param], r_val, rtol=1e-3,
            err_msg=f"{param} doesn't match R"
        )
```

---

## Building Documentation

ChronoBox uses MkDocs with Material theme:

```bash
# Local preview with auto-reload
mkdocs serve

# Build static site
mkdocs build
```

Documentation source lives in `docs/`. API reference uses `mkdocstrings` for automatic generation from docstrings.

---

## Code Review Checklist

### PR Checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] New code has tests
- [ ] Public API has docstrings (NumPy style)
- [ ] Type annotations on public methods
- [ ] Documentation updated (if applicable)
- [ ] Exports added to `__init__.py` (if applicable)
- [ ] No breaking changes to existing API

### PR Template

```markdown
## Summary

Brief description of what this PR does and why.

## Changes

- List of specific changes

## Testing

- How was this tested?
- Any new tests added?

## Checklist

- [ ] Tests pass
- [ ] Docs updated
- [ ] Changelog entry added (if user-facing)
```

---

## Recognition

Contributors are recognized in:

- The [Changelog](changelog.md)
- Release notes
- The AUTHORS file

Significant contributions may result in co-authorship on methodological papers.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Questions?

- **General questions**: [GitHub Discussions](https://github.com/NodesEcon/chronobox/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/NodesEcon/chronobox/issues)
- **Feature requests**: [GitHub Issues](https://github.com/NodesEcon/chronobox/issues) with `[Feature]` label

## See Also

- [Code of Conduct](code-of-conduct.md) — Community standards
- [Changelog](changelog.md) — Version history
- [Roadmap](roadmap.md) — Planned features
- [API Reference](../api/index.md) — Full API documentation
