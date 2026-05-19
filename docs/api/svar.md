---
title: "SVAR & Advanced VAR API"
description: "API reference for SVAR, BayesianVAR, FAVAR, TVPVAR, GVAR, and HistoricalDecomposition"
---

# SVAR & Advanced VAR API Reference

!!! info "Module"
    **Import**: `from chronobox import SVAR, BayesianVAR, FAVAR, TVPVAR, GVAR`
    **Source**: `chronobox/models/svar.py`, `chronobox/models/bvar.py`, `chronobox/models/favar.py`, `chronobox/models/tvpvar.py`, `chronobox/models/gvar.py`

## Overview

| Class | Description | Use Case |
|-------|-------------|----------|
| `SVAR` | Structural VAR with multiple identification schemes | Structural shock identification |
| `BayesianVAR` | Bayesian VAR with configurable priors | Small samples, shrinkage, model comparison |
| `FAVAR` | Factor-Augmented VAR | Large panels + policy variables |
| `TVPVAR` | Time-Varying Parameter VAR | Structural change, evolving dynamics |
| `GVAR` | Global VAR | Multi-country interdependencies |
| `HistoricalDecomposition` | Historical decomposition of structural shocks | Shock attribution over time |

---

## SVAR

Structural Vector Autoregression with four identification schemes:

- **Cholesky**: Recursive ordering via $\Sigma_u = PP'$
- **AB-model**: $A\mathbf{u}_t = B\boldsymbol{\varepsilon}_t$ (Amisano & Giannini, 1997)
- **Blanchard-Quah**: Long-run restrictions $C(1) = \Theta(1) P$ lower triangular
- **Sign restrictions**: Monte Carlo draws with sign constraints on IRFs (Uhlig, 2005)

$$
A_0 \mathbf{y}_t = A_1^* \mathbf{y}_{t-1} + \cdots + A_p^* \mathbf{y}_{t-p} + B \boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim N(\mathbf{0}, I_K)
$$

```python
SVAR(
    var_results: VARResults,
    method: str = "cholesky",
    a_mat: ndarray | None = None,
    b_mat: ndarray | None = None,
    sign_restrictions: dict | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `var_results` | `VARResults` | *required* | Fitted reduced-form VAR (must have `coefs`, `sigma_u`, `resid`) |
| `method` | `str` | `"cholesky"` | `'cholesky'`, `'ab'`, `'long_run'`, or `'sign'` |
| `a_mat` | `ndarray \| None` | `None` | A matrix for AB-model. `NaN` = free parameter, `float` = restricted |
| `b_mat` | `ndarray \| None` | `None` | B matrix for AB-model. `NaN` = free parameter, `float` = restricted |
| `sign_restrictions` | `dict \| None` | `None` | Sign restrictions: `{(shock, var, horizons): '+'/'-'}` |

### `.fit()` Method

```python
SVAR.fit(
    n_draws: int = 1000,
    max_draws: int = 50000,
) -> SVARResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_draws` | `int` | `1000` | Number of accepted draws (sign restrictions only) |
| `max_draws` | `int` | `50000` | Maximum total draws (sign restrictions only) |

**Returns**: `SVARResults`

### SVARResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `method` | `str` | Identification method used |
| `A0_inv` | `ndarray(K, K)` | Structural impact matrix $A_0^{-1} B$ |
| `A0` | `ndarray(K, K) \| None` | $A_0$ matrix |
| `B` | `ndarray(K, K) \| None` | $B$ matrix |
| `structural_shocks` | `ndarray(T, K)` | Estimated structural shocks $\boldsymbol{\varepsilon}_t$ |
| `sigma_u` | `ndarray(K, K)` | Reduced-form residual covariance |
| `coefs` | `ndarray(p, K, K)` | VAR coefficient matrices |
| `accepted_draws` | `list[ndarray] \| None` | Accepted impact matrices (sign restrictions only) |

| Method | Returns | Description |
|--------|---------|-------------|
| `irf(periods=20)` | `ndarray(H+1, K, K)` | Structural IRF. For sign restrictions, returns median across draws |
| `irf_with_bands(periods=20, alpha=0.16)` | `(median, lower, upper)` | IRF with confidence bands (68% by default) |
| `fevd(periods=20)` | `ndarray(H+1, K, K)` | Structural FEVD |

### Example

```python
import numpy as np
from chronobox import VAR, SVAR

# Fit reduced-form VAR
data = np.random.randn(200, 3)
var = VAR(lags=2)
var_results = var.fit(data, names=["GDP", "INF", "RATE"])

# --- Cholesky identification ---
svar = SVAR(var_results, method="cholesky")
results = svar.fit()
irf = results.irf(periods=20)
print(f"Impact of shock 0 on var 1: {irf[0, 1, 0]:.4f}")

# --- AB-model ---
K = 3
A = np.eye(K)
A[1, 0] = np.nan  # free parameter
A[2, 0] = np.nan
A[2, 1] = np.nan
B = np.diag([np.nan, np.nan, np.nan])

svar_ab = SVAR(var_results, method="ab", a_mat=A, b_mat=B)
results_ab = svar_ab.fit()

# --- Blanchard-Quah long-run restrictions ---
svar_lr = SVAR(var_results, method="long_run")
results_lr = svar_lr.fit()

# --- Sign restrictions ---
sign_restr = {
    (0, 0, range(0, 5)): "+",  # shock 0 -> var 0 positive for h=0..4
    (0, 1, range(0, 5)): "-",  # shock 0 -> var 1 negative for h=0..4
}
svar_sign = SVAR(var_results, method="sign", sign_restrictions=sign_restr)
results_sign = svar_sign.fit(n_draws=500)
median, lower, upper = results_sign.irf_with_bands(periods=20)
```

::: chronobox.models.svar.SVAR
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

::: chronobox.models.svar.SVARResults
    options:
      show_root_heading: true
      show_source: false
      members:
        - irf
        - irf_with_bands
        - fevd

---

## BayesianVAR

Bayesian Vector Autoregression with four prior specifications:

| Prior | Description | Best For |
|-------|-------------|----------|
| `'minnesota'` | Litterman (1986) shrinkage toward random walk | General macro forecasting |
| `'normal_wishart'` | Conjugate prior with analytical posterior | Analytical tractability |
| `'sims_zha'` | Sims & Zha (1998) via dummy observations | Policy analysis |
| `'flat'` | Diffuse prior (OLS-equivalent) | Large samples, baseline |

$$
\mathbf{y}_t = \mathbf{c} + A_1 \mathbf{y}_{t-1} + \cdots + A_p \mathbf{y}_{t-p} + \mathbf{u}_t, \quad \mathbf{u}_t \sim N(\mathbf{0}, \Sigma)
$$

with prior $p(A, \Sigma)$ depending on the chosen specification.

```python
BayesianVAR(
    lags: int = 1,
    prior: str = "minnesota",
    **prior_kwargs,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lags` | `int` | `1` | Number of lags $p$ |
| `prior` | `str` | `"minnesota"` | `'minnesota'`, `'normal_wishart'`, `'sims_zha'`, `'flat'` |
| `**prior_kwargs` | | | Prior hyperparameters (see below) |

### Prior Hyperparameters

=== "Minnesota"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `lambda_1` | `float` | `0.1` | Overall tightness |
    | `lambda_2` | `float` | `0.5` | Cross-variable tightness |
    | `lambda_3` | `float` | `1.0` | Lag decay |
    | `delta` | `float` | `1.0` | Prior mean for first own lag (1 = random walk) |

=== "Normal-Wishart"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `V_0` | `ndarray` | | Prior covariance for coefficients |
    | `S_0` | `ndarray` | | Prior scale matrix for $\Sigma$ |
    | `v_0` | `float` | | Prior degrees of freedom |

=== "Sims-Zha"

    Constructed via dummy observations. See Sims & Zha (1998).

### `.fit()` Method

```python
BayesianVAR.fit(
    endog: ndarray,
    n_draws: int = 5000,
    n_burn: int = 1000,
) -> BVARResults
```

**Returns**: `BVARResults`

### BVARResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefs_mean` | `ndarray(p, K, K)` | Posterior mean of VAR coefficients |
| `coefs_draws` | `ndarray(n_draws, p, K, K)` | Posterior draws of coefficients |
| `sigma_mean` | `ndarray(K, K)` | Posterior mean of $\Sigma$ |
| `sigma_draws` | `ndarray(n_draws, K, K)` | Posterior draws of $\Sigma$ |
| `intercept_mean` | `ndarray(K,)` | Posterior mean of intercept |
| `log_marginal_likelihood` | `float` | Log marginal likelihood (model comparison) |

| Method | Returns | Description |
|--------|---------|-------------|
| `forecast(steps=12, n_draws=None)` | `dict` | Bayesian predictive distribution with `mean`, `median`, credible intervals |
| `irf(periods=20, method='cholesky')` | `ndarray(H+1, K, K)` | IRF from posterior mean |
| `irf_draws_compute(periods=20)` | `ndarray(n_draws, H+1, K, K)` | IRF for each posterior draw |

### Example

```python
import numpy as np
from chronobox import BayesianVAR

data = np.random.randn(200, 3)

# Minnesota prior (default)
bvar = BayesianVAR(lags=2, prior="minnesota", lambda_1=0.1, lambda_2=0.5)
results = bvar.fit(data, n_draws=5000, n_burn=1000)

# Bayesian forecasts with credible intervals
fc = results.forecast(steps=12)
print(f"Mean forecast shape: {fc['mean'].shape}")      # (12, 3)
print(f"95% interval width: {(fc['upper_95'] - fc['lower_95']).mean():.4f}")

# IRF with posterior uncertainty
irf_draws = results.irf_draws_compute(periods=20)
irf_median = np.median(irf_draws, axis=0)
irf_lower = np.percentile(irf_draws, 16, axis=0)
irf_upper = np.percentile(irf_draws, 84, axis=0)

# Model comparison via marginal likelihood
bvar1 = BayesianVAR(lags=1, prior="minnesota").fit(data)
bvar2 = BayesianVAR(lags=2, prior="minnesota").fit(data)
print(f"Log ML (p=1): {bvar1.log_marginal_likelihood:.2f}")
print(f"Log ML (p=2): {bvar2.log_marginal_likelihood:.2f}")
```

::: chronobox.models.bvar.BayesianVAR
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

::: chronobox.models.bvar.BVARResults
    options:
      show_root_heading: true
      show_source: false
      members:
        - forecast
        - irf
        - irf_draws_compute

---

## FAVAR

Factor-Augmented Vector Autoregression (Bernanke, Boivin & Eliasz, 2005).

Two-step estimation:

1. **Step 1**: Extract $K$ latent factors from a large panel $X_t$ via PCA
2. **Step 2**: Estimate VAR on $[F_t; Y_t]$ (factors + policy variables)

$$
X_t = \Lambda^f F_t + \Lambda^y Y_t + e_t
$$

$$
\begin{bmatrix} F_t \\ Y_t \end{bmatrix} = \Phi(L) \begin{bmatrix} F_{t-1} \\ Y_{t-1} \end{bmatrix} + \mathbf{u}_t
$$

```python
FAVAR(
    n_factors: int = 3,
    lags: int = 2,
    method: str = "two_step",
    remove_y_from_factors: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_factors` | `int` | `3` | Number of latent factors to extract |
| `lags` | `int` | `2` | VAR lag order |
| `method` | `str` | `"two_step"` | `'two_step'` (PCA + OLS) or `'bayesian'` (Kalman filter via kalmanbox) |
| `remove_y_from_factors` | `bool` | `True` | Remove $Y$ component from extracted factors |

### `.fit()` Method

```python
FAVAR.fit(
    panel: ndarray,   # (T, N) large panel of informational variables
    policy: ndarray,  # (T, M) policy/observable variables
) -> FAVARResults
```

### FAVARResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `factors` | `ndarray(T, K)` | Estimated latent factors |
| `loadings` | `ndarray(N, K)` | Factor loadings $\Lambda^f$ |
| `loadings_y` | `ndarray(N, M)` | Policy variable loadings $\Lambda^y$ |
| `var_coefs` | `ndarray(p, K+M, K+M)` | VAR coefficients for $[F; Y]$ |
| `var_sigma` | `ndarray(K+M, K+M)` | VAR residual covariance |
| `explained_variance_ratio` | `ndarray(K,)` | Variance explained by each factor |
| `total_explained_variance` | `float` | Total variance explained |

| Method | Returns | Description |
|--------|---------|-------------|
| `irf(periods=20, identification='cholesky')` | `ndarray(H+1, K+M, K+M)` | IRF in the $[F; Y]$ space |
| `irf_panel(periods=20, identification='cholesky')` | `ndarray(H+1, N, K+M)` | IRF for all $N$ panel variables |

### Example

```python
import numpy as np
from chronobox import FAVAR

# Large panel (50 macro variables) + 1 policy variable
panel = np.random.randn(200, 50)
policy = np.random.randn(200, 1)

# Two-step FAVAR with 3 factors
favar = FAVAR(n_factors=3, lags=2, method="two_step")
results = favar.fit(panel, policy)

print(f"Factors shape: {results.factors.shape}")           # (200, 3)
print(f"Explained variance: {results.total_explained_variance:.2%}")

# IRF in factor space
irf_fy = results.irf(periods=20)
print(f"IRF shape (factor space): {irf_fy.shape}")         # (21, 4, 4)

# IRF for all panel variables
irf_panel = results.irf_panel(periods=20)
print(f"IRF shape (panel): {irf_panel.shape}")             # (21, 50, 4)
```

::: chronobox.models.favar.FAVAR
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

::: chronobox.models.favar.FAVARResults
    options:
      show_root_heading: true
      show_source: false
      members:
        - irf
        - irf_panel

---

## TVPVAR

Time-Varying Parameter VAR estimated via Kalman filter (Primiceri, 2005).

Coefficients evolve as a random walk:

$$
\mathbf{y}_t = X_t \boldsymbol{\beta}_t + \boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim N(\mathbf{0}, H)
$$

$$
\boldsymbol{\beta}_t = \boldsymbol{\beta}_{t-1} + \mathbf{u}_t, \quad \mathbf{u}_t \sim N(\mathbf{0}, Q)
$$

```python
TVPVAR(
    lags: int = 1,
    stochastic_volatility: bool = False,
    Q_scale: float = 0.01,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lags` | `int` | `1` | Number of lags |
| `stochastic_volatility` | `bool` | `False` | Reserved for future extension |
| `Q_scale` | `float` | `0.01` | Scale for state evolution covariance $Q = Q_\text{scale}^2 \cdot I$. Smaller = slower coefficient change |

### `.fit()` Method

```python
TVPVAR.fit(
    endog: ndarray,              # (T, K) endogenous variables
    Q: ndarray | None = None,    # (d, d) custom state covariance
) -> TVPVARResults
```

### TVPVARResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `beta_filtered` | `ndarray(T, d)` | Filtered time-varying coefficients |
| `beta_smoothed` | `ndarray(T, d) \| None` | Smoothed coefficients (RTS smoother) |
| `coefs_time` | `ndarray(T, p, K, K)` | Time-varying coefficient matrices $A_{1t}, \ldots, A_{pt}$ |
| `intercept_time` | `ndarray(T, K)` | Time-varying intercept |
| `sigma` | `ndarray(K, K)` | Observation noise covariance (constant) |
| `Q` | `ndarray(d, d)` | State evolution covariance |
| `log_likelihood` | `float` | Log-likelihood from Kalman filter |

| Method | Returns | Description |
|--------|---------|-------------|
| `time_varying_irf(t, periods=20, identification='cholesky')` | `ndarray(H+1, K, K)` | Structural IRF at time point $t$ |
| `coefficient_path(lag, i, j)` | `ndarray(T,)` | Time path of coefficient $A_{\text{lag}}[i, j]$ |

### Example

```python
import numpy as np
from chronobox import TVPVAR

data = np.random.randn(300, 2)

# Fit TVP-VAR(1)
tvpvar = TVPVAR(lags=1, Q_scale=0.01)
results = tvpvar.fit(data)

# IRF at different time points
irf_early = results.time_varying_irf(t=50, periods=20)
irf_late = results.time_varying_irf(t=250, periods=20)

# Compare impulse responses over time
print(f"Early impact [0,1]: {irf_early[0, 0, 1]:.4f}")
print(f"Late impact [0,1]:  {irf_late[0, 0, 1]:.4f}")

# Track coefficient evolution
path = results.coefficient_path(lag=0, i=0, j=1)
print(f"Coefficient a_01 range: [{path.min():.4f}, {path.max():.4f}]")
```

::: chronobox.models.tvpvar.TVPVAR
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

::: chronobox.models.tvpvar.TVPVARResults
    options:
      show_root_heading: true
      show_source: false
      members:
        - time_varying_irf
        - coefficient_path

---

## GVAR

Global Vector Autoregression (Pesaran, Schuermann & Weiner, 2004).

Models interdependencies between $N$ countries using trade-weighted foreign variables:

$$
\mathbf{y}_{it} = c_i + \Phi_i \mathbf{y}_{i,t-1} + \Lambda_{i0} \mathbf{y}^*_{it} + \Lambda_{i1} \mathbf{y}^*_{i,t-1} + \mathbf{u}_{it}
$$

where $\mathbf{y}^*_{it} = \sum_{j \neq i} w_{ij} \mathbf{y}_{jt}$ are trade-weighted foreign variables.

```python
GVAR(
    trade_weights: ndarray,
    domestic_lags: int = 1,
    foreign_lags: int = 1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trade_weights` | `ndarray(N, N)` | *required* | Bilateral trade weight matrix. Diagonal = 0, rows sum to 1 |
| `domestic_lags` | `int` | `1` | Number of domestic lags $p$ |
| `foreign_lags` | `int` | `1` | Number of foreign lags $q$ |

### `.fit()` Method

```python
GVAR.fit(
    data_dict: dict[str, ndarray],  # {"country": ndarray(T, k_i)}
) -> GVARResults
```

### GVARResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `global_coefs` | `list[ndarray(k_total, k_total)]` | Global system coefficient matrices $F_1, \ldots, F_p$ |
| `global_sigma` | `ndarray(k_total, k_total)` | Global residual covariance |
| `country_results` | `dict[str, dict]` | Per-country VARX estimation results |
| `country_names` | `list[str]` | Country names |
| `country_dims` | `dict[str, int]` | Number of variables per country |
| `trade_weights` | `ndarray(N, N)` | Trade weight matrix |
| `is_stable` | `bool` | Whether the global system is stable |
| `eigenvalues` | `ndarray` | Companion matrix eigenvalues |

| Method | Returns | Description |
|--------|---------|-------------|
| `girf(shock_country, shock_var, periods=40, shock_size=None)` | `ndarray(H+1, k_total)` | Generalized IRF for entire global system |
| `irf_country(shock_country, shock_var, response_country, periods=40)` | `ndarray(H+1, k_resp)` | GIRF for a specific response country |

### Example

```python
import numpy as np
from chronobox import GVAR

# 3 countries, 2 variables each
np.random.seed(42)
data = {
    "US": np.random.randn(200, 2),
    "EU": np.random.randn(200, 2),
    "JP": np.random.randn(200, 2),
}

# Trade weight matrix
W = np.array([
    [0.0, 0.6, 0.4],  # US imports: 60% EU, 40% JP
    [0.5, 0.0, 0.5],  # EU imports: 50% US, 50% JP
    [0.7, 0.3, 0.0],  # JP imports: 70% US, 30% EU
])

# Fit GVAR
gvar = GVAR(trade_weights=W, domestic_lags=1, foreign_lags=1)
results = gvar.fit(data)
print(f"Stable: {results.is_stable}")

# GIRF: shock to US variable 0
girf = results.girf(shock_country="US", shock_var=0, periods=40)
print(f"GIRF shape: {girf.shape}")  # (41, 6)

# Response of EU to US shock
eu_response = results.irf_country(
    shock_country="US", shock_var=0,
    response_country="EU", periods=40,
)
print(f"EU response shape: {eu_response.shape}")  # (41, 2)
```

::: chronobox.models.gvar.GVAR
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

::: chronobox.models.gvar.GVARResults
    options:
      show_root_heading: true
      show_source: false
      members:
        - girf
        - irf_country

---

## HistoricalDecomposition

Decomposes observed time series movements into contributions from each structural shock.

$$
\mathbf{y}_t = \mathbf{y}_t^{\text{base}} + \sum_{k=1}^{K} \mathbf{y}_t^{(k)}
$$

where $\mathbf{y}_t^{(k)} = \sum_{s=0}^{t-1} \Theta_s \mathbf{e}_k \varepsilon_{k,t-s}$ is the contribution of shock $k$.

```python
HistoricalDecomposition(
    svar_results: SVARResults,
    variable_names: list[str] | None = None,
    shock_names: list[str] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `svar_results` | `SVARResults` | *required* | Fitted SVAR results |
| `variable_names` | `list[str] \| None` | `None` | Names for variables |
| `shock_names` | `list[str] \| None` | `None` | Names for structural shocks |

### HistoricalDecompositionResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `decomposition` | `ndarray(T, K, K)` | `HD[t, k, i]` = contribution of shock $k$ to variable $i$ at time $t$ |
| `base` | `ndarray(T, K)` | Deterministic (base) forecast component |
| `observed` | `ndarray(T, K)` | Observed values |
| `structural_shocks` | `ndarray(T, K)` | Structural shocks $\boldsymbol{\varepsilon}_t$ |

| Method | Returns | Description |
|--------|---------|-------------|
| `verify_decomposition(atol=1e-8)` | `bool` | Verify that base + sum(HD) = observed |

### Example

```python
from chronobox import VAR, SVAR
from chronobox.analysis.hd import HistoricalDecomposition

# Fit VAR -> SVAR
var_results = VAR(lags=2).fit(data)
svar_results = SVAR(var_results, method="cholesky").fit()

# Historical decomposition
hd = HistoricalDecomposition(svar_results)
result = hd.fit()

# Verify decomposition
assert result.verify_decomposition()

# Contribution of shock 0 to variable 1 over time
contrib = result.decomposition[:, 0, 1]
```

::: chronobox.analysis.hd.HistoricalDecomposition
    options:
      show_root_heading: false
      show_source: true
      members:
        - fit

::: chronobox.analysis.hd.HistoricalDecompositionResult
    options:
      show_root_heading: true
      show_source: false
      members:
        - verify_decomposition

---

## See Also

- [VAR & VECM API](var.md) -- Reduced-form VAR and VECM
- [Filters API](filters.md) -- Economic filters (HP, BK, CF, Hamilton, BN)
- [Spillover API](spillover.md) -- Diebold-Yilmaz spillover index
- [SVAR Theory](../theory/svar-theory.md) -- Identification schemes and structural analysis
- [BVAR Theory](../theory/bvar-theory.md) -- Bayesian VAR priors and estimation
- [SVAR User Guide](../user-guide/svar/index.md) -- Step-by-step SVAR guide
