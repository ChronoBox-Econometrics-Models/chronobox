# Installation

## From PyPI

```bash
pip install chronobox
```

## From source

```bash
git clone https://github.com/nodesecon/chronobox.git
cd chronobox
pip install -e ".[dev]"
```

## Dependencies

### Required
- Python >= 3.11
- NumPy >= 1.24
- SciPy >= 1.10
- pandas >= 2.0
- kalmanbox (Kalman filter and MLE)
- matplotlib >= 3.7

### Optional
- numba >= 0.57 (JIT compilation for 5x+ speedup)
- statsmodels >= 0.14 (additional statistical tests)

### Development
- pytest >= 7.0
- ruff >= 0.4
- pyright >= 1.1
- mkdocs-material >= 9.0

## Verify installation

```python
import chronobox
print(chronobox.__version__)
```
