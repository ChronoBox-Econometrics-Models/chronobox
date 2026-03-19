# Diagnostic Tests

## Overview

Diagnostic tests evaluate the adequacy of fitted models by checking
residual properties: serial correlation, heteroskedasticity, normality,
and functional form.

## Serial Correlation Tests

### Ljung-Box Test

```python
from chronobox.tests_stat import ljung_box_test

result = ljung_box_test(residuals, lags=10)
print(f"Q-stat: {result.statistic}")
print(f"p-value: {result.p_value}")
```

### Breusch-Godfrey LM Test

```python
from chronobox.tests_stat import breusch_godfrey_test

result = breusch_godfrey_test(residuals, nlags=4)
print(result)
```

### Durbin-Watson Test

```python
from chronobox.tests_stat import durbin_watson_test

result = durbin_watson_test(residuals)
print(f"DW statistic: {result.statistic}")
# Close to 2 = no autocorrelation
```

## Heteroskedasticity Tests

### ARCH-LM Test

```python
from chronobox.tests_stat import arch_lm_test

result = arch_lm_test(residuals, nlags=5)
print(f"p-value: {result.p_value}")
```

### White Test

```python
from chronobox.tests_stat import white_test

result = white_test(residuals, exog)
print(result)
```

## Normality Tests

### Jarque-Bera Test

```python
from chronobox.tests_stat import jarque_bera_test

result = jarque_bera_test(residuals)
print(f"JB statistic: {result.statistic}")
print(f"p-value: {result.p_value}")
print(f"Skewness: {result.skewness}")
print(f"Kurtosis: {result.kurtosis}")
```

## Specification Tests

### RESET Test

Ramsey RESET tests for omitted nonlinearities:

```python
from chronobox.tests_stat import reset_test

result = reset_test(residuals, fitted_values, power=3)
print(result)
```

### BDS Test

Tests for independence (iid) in residuals:

```python
from chronobox.tests_stat import bds_test

result = bds_test(residuals, max_dim=5)
print(result)
```

## Structural Break Tests

### CUSUM Test

```python
from chronobox.tests_stat import cusum_test

result = cusum_test(residuals)

from chronobox.visualization import plot_cusum
plot_cusum(result)
```

### Chow Test

```python
from chronobox.tests_stat import chow_test

result = chow_test(y, X, break_point=50)
print(f"F-stat: {result.statistic}")
print(f"p-value: {result.p_value}")
```

### Bai-Perron Test

```python
from chronobox.tests_stat import bai_perron_test

result = bai_perron_test(y, X, max_breaks=5)
print(f"Breaks found: {result.n_breaks}")
print(f"Break dates: {result.break_dates}")
```
