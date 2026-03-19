# SVAR - Structural VAR

## Overview

Structural VAR models impose identifying restrictions on the contemporaneous
relationships between variables, allowing for structural interpretation of
shocks and impulse responses.

## Basic Usage

```python
from chronobox import SVAR
from chronobox.datasets import load_dataset

data = load_dataset('us_macro_quarterly')
model = SVAR(data)
results = model.fit(maxlags=4, identification='cholesky')
```

## Identification Schemes

### Cholesky Decomposition

The most common identification strategy. Assumes a recursive causal ordering:

```python
results = model.fit(maxlags=4, identification='cholesky')
```

!!! note
    Variable ordering matters. The first variable is not contemporaneously
    affected by any other variable.

### Short-run Restrictions

Impose zero restrictions on the contemporaneous impact matrix:

```python
import numpy as np

# A matrix: contemporaneous effects
A = np.array([
    [1, 0, 0],
    ['nan', 1, 0],
    ['nan', 'nan', 1],
])

results = model.fit(maxlags=4, identification='short_run', A=A)
```

### Long-run Restrictions (Blanchard-Quah)

```python
results = model.fit(maxlags=4, identification='long_run')
```

## Structural IRF

```python
sirf = results.irf(periods=40, structural=True)

from chronobox.visualization import plot_irf
plot_irf(sirf)
```

## Historical Decomposition

```python
from chronobox.analysis import HistoricalDecomposition

hd = HistoricalDecomposition(results)
hd_result = hd.compute()

from chronobox.visualization import plot_hd
plot_hd(hd_result)
```

## Counterfactual Analysis

```python
from chronobox.analysis import Counterfactual

cf = Counterfactual(results)
cf_result = cf.compute(shock_idx=0, zero_after=40)
```
