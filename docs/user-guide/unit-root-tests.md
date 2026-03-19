# Unit Root Tests

## Overview

Unit root tests determine whether a time series is stationary or contains
a unit root (non-stationary). ChronoBox provides several tests with
different null hypotheses.

## Available Tests

| Test | Null Hypothesis | Import |
|------|----------------|--------|
| ADF | Unit root (non-stationary) | `adf_test` |
| PP | Unit root (non-stationary) | `pp_test` |
| KPSS | Stationary | `kpss_test` |
| ERS/DF-GLS | Unit root (non-stationary) | `ers_test` |
| Zivot-Andrews | Unit root (with structural break) | `zivot_andrews_test` |
| Lee-Strazicich | Unit root (with breaks) | `lee_strazicich_test` |
| HEGY | Seasonal unit root | `hegy_test` |

## ADF Test

```python
from chronobox.tests_stat import adf_test

result = adf_test(data, maxlag=None, regression='c')
print(result)
print(f"Statistic: {result.statistic}")
print(f"p-value: {result.p_value}")
print(f"Lags used: {result.lags}")
```

## PP Test

```python
from chronobox.tests_stat import pp_test

result = pp_test(data, regression='c')
print(result)
```

## KPSS Test

!!! warning
    The KPSS test has the **opposite** null hypothesis (stationarity).
    Reject = non-stationary.

```python
from chronobox.tests_stat import kpss_test

result = kpss_test(data, regression='c')
print(result)
```

## ERS/DF-GLS Test

More powerful than ADF, especially for small samples:

```python
from chronobox.tests_stat import ers_test

result = ers_test(data)
print(result)
```

## Zivot-Andrews Test

Tests for unit root allowing for one structural break:

```python
from chronobox.tests_stat import zivot_andrews_test

result = zivot_andrews_test(data, model='intercept')
print(f"Break date: {result.break_date}")
```

## Confirmatory Strategy

Best practice is to use multiple tests:

```python
from chronobox.tests_stat import adf_test, kpss_test

adf = adf_test(data)
kpss = kpss_test(data)

if adf.p_value > 0.05 and kpss.p_value < 0.05:
    print("Non-stationary (both tests agree)")
elif adf.p_value < 0.05 and kpss.p_value > 0.05:
    print("Stationary (both tests agree)")
else:
    print("Conflicting results - further analysis needed")
```
