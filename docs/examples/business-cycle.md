# Example: Business Cycle Analysis

## Overview

Extract business cycle components from GDP using multiple filters
and compare the results.

## Load Data

```python
from chronobox.datasets import load_dataset
import numpy as np

gdp = load_dataset('us_gdp')
log_gdp = np.log(gdp)
```

## Step 1: HP Filter

```python
from chronobox.filters import hp_filter

trend_hp, cycle_hp = hp_filter(log_gdp, lamb=1600)
```

## Step 2: BK Filter

```python
from chronobox.filters import bk_filter

cycle_bk = bk_filter(log_gdp, low=6, high=32, K=12)
```

## Step 3: CF Filter

```python
from chronobox.filters import cf_filter

cycle_cf, trend_cf = cf_filter(log_gdp, low=6, high=32)
```

## Step 4: Hamilton Filter

```python
from chronobox.filters import hamilton_filter

trend_ham, cycle_ham = hamilton_filter(log_gdp, h=8, p=4)
```

## Step 5: Compare Filters

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

axes[0].plot(cycle_hp)
axes[0].set_title("HP Filter (lambda=1600)")
axes[0].axhline(0, color='black', linewidth=0.5)

axes[1].plot(cycle_bk)
axes[1].set_title("Baxter-King Filter")
axes[1].axhline(0, color='black', linewidth=0.5)

axes[2].plot(cycle_cf)
axes[2].set_title("Christiano-Fitzgerald Filter")
axes[2].axhline(0, color='black', linewidth=0.5)

axes[3].plot(cycle_ham)
axes[3].set_title("Hamilton Filter (h=8, p=4)")
axes[3].axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('business_cycles.png', dpi=150)
plt.show()
```

## Step 6: Correlation Matrix

```python
import pandas as pd

cycles = pd.DataFrame({
    'HP': cycle_hp,
    'BK': cycle_bk,
    'CF': cycle_cf,
    'Hamilton': cycle_ham,
}).dropna()

print("Correlation between cycle estimates:")
print(cycles.corr().round(3))
```

## Step 7: STL Decomposition

```python
from chronobox.decomposition import STL
from chronobox.visualization import plot_decomposition

stl = STL(gdp, period=4)
result = stl.fit()
plot_decomposition(result)
```

## Key Findings

- HP filter produces smoother cycles but with endpoint bias
- BK filter is symmetric but loses observations
- CF filter preserves sample size with minimal bias
- Hamilton filter avoids spurious cyclicality entirely
