---
title: Installation
description: Install ChronoBox and its dependencies in any Python environment
---

# Installation

## Quick Install

```bash
pip install chronobox
```

That's it. ChronoBox and all core dependencies --- including [KalmanBox](https://github.com/NodesEcon/kalmanbox) --- will be installed automatically.

## Requirements

**Python**: >= 3.11 (3.11, 3.12 supported)

**Core dependencies** (installed automatically):

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| NumPy | >= 1.24 | Array operations and linear algebra |
| SciPy | >= 1.10 | Statistical functions and optimization |
| pandas | >= 2.0 | Time series data handling |
| KalmanBox | latest | State-space models, Kalman filter, and MLE |
| matplotlib | >= 3.7 | Plotting and visualization |

**Optional dependencies**:

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| Numba | >= 0.57 | JIT compilation for 5x+ speedup |

!!! info "About KalmanBox"
    [KalmanBox](https://github.com/NodesEcon/kalmanbox) is installed automatically as a
    core dependency. It provides the state-space representation and Kalman filter used
    internally by ETS, ARIMA, and TVP models. You don't need to install or configure it
    separately --- ChronoBox handles everything behind the scenes.

## Installation Options

=== "pip (Recommended)"

    ```bash
    pip install chronobox
    ```

    Upgrade to the latest version:

    ```bash
    pip install --upgrade chronobox
    ```

=== "From Source"

    ```bash
    git clone https://github.com/NodesEcon/chronobox.git
    cd chronobox
    pip install -e .
    ```

=== "Development Mode"

    ```bash
    git clone https://github.com/NodesEcon/chronobox.git
    cd chronobox
    pip install -e ".[dev]"
    ```

### Optional Extras

```bash
# Development tools (testing, linting, type checking)
pip install chronobox[dev]

# Documentation tools (MkDocs Material, mkdocstrings)
pip install chronobox[docs]

# JIT compilation for performance
pip install chronobox[numba]
```

## Virtual Environments

We recommend using a virtual environment to avoid dependency conflicts:

=== "venv"

    ```bash
    python -m venv chronobox_env
    source chronobox_env/bin/activate   # Linux/macOS
    chronobox_env\Scripts\activate      # Windows
    pip install chronobox
    ```

=== "conda"

    ```bash
    conda create -n chronobox_env python=3.11
    conda activate chronobox_env
    pip install chronobox
    ```

## Verification

Verify that ChronoBox is installed correctly:

```python
import chronobox
print(f"ChronoBox version: {chronobox.__version__}")
```

Expected output:

```text
ChronoBox version: 0.x.x
```

Run a quick smoke test with a built-in dataset:

```python
from chronobox import ARIMA
from chronobox.datasets import load_dataset

# Load the classic airline passengers dataset
data = load_dataset("airline")
print(f"Loaded {len(data)} observations")

# Fit a simple model
model = ARIMA(order=(0,1,1), seasonal_order=(0,1,1,12))
result = model.fit(data["passengers"])
print(f"AIC: {result.aic:.2f}")
```

!!! tip "Verification checklist"
    If the above runs without errors, ChronoBox and all dependencies
    (including KalmanBox) are correctly installed and working.

## Installation for Development

If you want to contribute to ChronoBox or run the test suite:

```bash
# Clone the repository
git clone https://github.com/NodesEcon/chronobox.git
cd chronobox

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Verify the test suite runs
pytest
```

The `[dev]` extra installs: pytest, pytest-cov, ruff, pyright, hypothesis, bandit, pre-commit, mkdocs-material, and mkdocstrings.

## Troubleshooting

### `ModuleNotFoundError: No module named 'chronobox'`

Ensure ChronoBox is installed in the active Python environment:

```bash
pip list | grep chronobox
```

If not found, install it. If using Jupyter, make sure the notebook kernel matches the environment where ChronoBox is installed.

### `ModuleNotFoundError: No module named 'kalmanbox'`

KalmanBox should be installed automatically as a dependency. If missing, install it manually:

```bash
pip install kalmanbox
```

Then reinstall ChronoBox to ensure all dependencies are in sync:

```bash
pip install --force-reinstall chronobox
```

### Dependency Conflicts

Create a fresh virtual environment:

```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install chronobox
```

### Windows: NumPy / SciPy Build Errors

Install the Microsoft Visual C++ Redistributable from
[Microsoft's download page](https://aka.ms/vs/17/release/vc_redist.x64.exe),
then retry the installation.

### macOS Apple Silicon (M1/M2/M3)

Use conda with native ARM64 support:

```bash
conda create -n chronobox_env python=3.11
conda activate chronobox_env
pip install chronobox
```

### Performance: Numba Not Available

ChronoBox works without Numba, but large datasets benefit from JIT compilation:

```bash
pip install chronobox[numba]
```

!!! warning "Numba requires a compatible NumPy version"
    If Numba installation fails, ensure your NumPy version is compatible.
    Check the [Numba release notes](https://numba.readthedocs.io/en/stable/release-notes.html)
    for supported NumPy versions.

### System Information for Bug Reports

```python
import sys, platform
import chronobox

print(f"Python:     {sys.version}")
print(f"Platform:   {platform.platform()}")
print(f"ChronoBox:  {chronobox.__version__}")
```

Include this output when [reporting issues](https://github.com/NodesEcon/chronobox/issues).

## Next Steps

- **[Quick Start](quickstart.md)** --- Fit your first ARIMA model in 5 minutes
- **[Core Concepts](core-concepts.md)** --- Understand time series fundamentals
- **[Choosing a Model](choosing-model.md)** --- Find the right model for your data
