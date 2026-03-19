# Performance & Numba

## Overview

ChronoBox uses NumPy for vectorized operations and optionally Numba
for JIT compilation of performance-critical loops.

## Numba Acceleration

Install numba for automatic speedup:

```bash
pip install numba
```

When numba is installed, critical functions are automatically JIT-compiled
on first call. Subsequent calls run at native speed.

## Performance Tips

### 1. Use vectorized operations

```python
# Fast: vectorized NumPy
import numpy as np
diff = np.diff(data)

# Slow: Python loop
diff = [data[i] - data[i-1] for i in range(1, len(data))]
```

### 2. Pre-allocate arrays

```python
# Fast
result = np.empty(n)
for i in range(n):
    result[i] = compute(i)

# Slow
result = []
for i in range(n):
    result.append(compute(i))
```

### 3. Use appropriate data types

```python
# 64-bit float (default, accurate)
data = np.array(values, dtype=np.float64)

# 32-bit float (faster, less precise)
data = np.array(values, dtype=np.float32)
```

## Benchmarks

See the [Benchmarks](../benchmarks.md) page for detailed performance
comparisons.

## Memory Usage

For large datasets:

- Use `pd.Series` with appropriate dtypes
- Avoid unnecessary copies
- Process data in chunks when possible

## Kalman Filter Performance

The Kalman filter (via `kalmanbox`) is the main computational bottleneck
for ARIMA estimation. Performance scales as:

- ARIMA: $O(T \cdot r^3)$ where $r$ is the state dimension
- VAR: $O(T \cdot K^3)$ where $K$ is the number of variables
