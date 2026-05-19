---
title: TVP-VAR
description: VAR com parametros variantes no tempo --- estimacao via kalmanbox com volatilidade estocastica opcional.
---

# TVP-VAR (Time-Varying Parameter VAR)

!!! info "Quick Reference"
    - **Classe**: `chronobox.TVPVAR`
    - **Import**: `from chronobox import TVPVAR`
    - **R equivalente**: `bvarsv::bvar.sv.tvp()`, `tvpvar` custom
    - **Estimacao**: Filtro de Kalman via kalmanbox + Gibbs sampling

---

## Overview

O TVP-VAR (Time-Varying Parameter VAR) permite que os coeficientes do VAR
**mudem ao longo do tempo**, capturando mudancas estruturais na economia.
Introduzido por Cogley & Sargent (2001, 2005) e Primiceri (2005), o modelo
e amplamente usado para estudar:

- Mudancas na transmissao de politica monetaria (Great Moderation)
- Evolucao da curva de Phillips
- Variacoes na persistencia da inflacao
- Instabilidade nos mecanismos de transmissao

O chronobox implementa o TVP-VAR usando o **kalmanbox** como backend para o
filtro de Kalman, garantindo estabilidade numerica e eficiencia computacional.

### Quando usar

- Suspeita de mudancas estruturais nos coeficientes ao longo do tempo
- Analise de politica monetaria em periodos de transicao (ex.: 1970s → 2000s)
- Quando testes de quebra estrutural (Chow, Bai-Perron) indicam instabilidade
- Comparacao de IRFs em diferentes periodos historicos

---

## Formulacao Matematica

### State-Space Representation

O TVP-VAR e um modelo state-space onde os coeficientes sao os estados:

**Equacao de observacao**:

$$
\mathbf{y}_t = \mathbf{X}_t \boldsymbol{\beta}_t + \mathbf{u}_t, \qquad \mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_t)
$$

onde $\mathbf{X}_t = \mathbf{I}_K \otimes (1, \mathbf{y}'_{t-1}, \ldots, \mathbf{y}'_{t-p})$
e $\boldsymbol{\beta}_t = \text{vec}(\mathbf{c}_t, \mathbf{A}_{1,t}, \ldots, \mathbf{A}_{p,t})$.

**Equacao de transicao (random walk)**:

$$
\boldsymbol{\beta}_t = \boldsymbol{\beta}_{t-1} + \mathbf{v}_t, \qquad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})
$$

A especificacao random walk implica que as mudancas nos parametros sao
**permanentes e graduais**. A matriz $\mathbf{Q}$ controla a velocidade de
variacao --- se $\mathbf{Q} = \mathbf{0}$, o modelo reduz-se a um VAR padrao.

### Stochastic Volatility

Primiceri (2005) adiciona volatilidade estocastica permitindo que $\boldsymbol{\Sigma}_t$
tambem mude no tempo. A decomposicao:

$$
\boldsymbol{\Sigma}_t = \mathbf{L}_t^{-1} \mathbf{H}_t (\mathbf{L}_t^{-1})'
$$

onde $\mathbf{L}_t$ e triangular inferior (relacoes contemporaneas variantes)
e $\mathbf{H}_t = \text{diag}(h_{1t}, \ldots, h_{Kt})$ com:

$$
\ln h_{it} = \ln h_{i,t-1} + \sigma_i \eta_{it}, \qquad \eta_{it} \sim \mathcal{N}(0, 1)
$$

### Filtro de Kalman (via kalmanbox)

O filtro de Kalman estima os coeficientes variantes no tempo:

**Previsao**:

$$
\boldsymbol{\beta}_{t|t-1} = \boldsymbol{\beta}_{t-1|t-1}
$$

$$
\mathbf{P}_{t|t-1} = \mathbf{P}_{t-1|t-1} + \mathbf{Q}
$$

**Atualizacao**:

$$
\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{X}_t' (\mathbf{X}_t \mathbf{P}_{t|t-1} \mathbf{X}_t' + \boldsymbol{\Sigma}_t)^{-1}
$$

$$
\boldsymbol{\beta}_{t|t} = \boldsymbol{\beta}_{t|t-1} + \mathbf{K}_t (\mathbf{y}_t - \mathbf{X}_t \boldsymbol{\beta}_{t|t-1})
$$

$$
\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{X}_t) \mathbf{P}_{t|t-1}
$$

O kalmanbox implementa o filtro com otimizacoes numericas (square-root filter,
Joseph form) para estabilidade.

### Estimacao via Gibbs Sampling

O algoritmo MCMC alterna entre:

1. $\boldsymbol{\beta}_{1:T} \mid \mathbf{Q}, \boldsymbol{\Sigma}_{1:T}, \mathbf{Y}$
   --- Carter-Kohn smoother (forward-filter, backward-sample)
2. $\mathbf{Q} \mid \boldsymbol{\beta}_{1:T}$ --- Inverse-Wishart
3. $\boldsymbol{\Sigma}_{1:T} \mid \boldsymbol{\beta}_{1:T}, \mathbf{Y}$
   --- univariate stochastic volatility (se habilitado)

---

## Quick Example

```python
from chronobox import TVPVAR
from chronobox.datasets import load_macro

data = load_macro()

# TVP-VAR com volatilidade constante
model = TVPVAR(
    lags=2,
    stochastic_volatility=False,
    n_draws=5000,
    n_burn=2000,
    seed=42,
)
results = model.fit(data)

# Coeficientes variantes no tempo
betas = results.time_varying_coefs
print(betas["gdp.L1->gdp"])  # Serie temporal do coeficiente

# IRF em dois periodos diferentes
irf_early = results.irf(steps=20, date="1985-01")
irf_late  = results.irf(steps=20, date="2015-01")
```

---

## Guia Detalhado

### Construtor

```python
TVPVAR(
    lags=2,                        # Numero de defasagens
    trend='c',                     # Componente deterministico
    stochastic_volatility=False,   # Volatilidade estocastica
    Q_prior_scale=1e-4,            # Escala da prior sobre Q
    n_draws=5000,                  # Total de draws MCMC
    n_burn=2000,                   # Burn-in
    n_thin=1,                      # Thinning
    seed=None,                     # Semente
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `lags` | `int` | `2` | Numero de defasagens |
| `trend` | `str` | `'c'` | `'n'`, `'c'`, `'t'`, `'ct'` |
| `stochastic_volatility` | `bool` | `False` | Habilitar volatilidade variante |
| `Q_prior_scale` | `float` | `1e-4` | Escala da prior Inverse-Wishart sobre $\mathbf{Q}$ |
| `n_draws` | `int` | `5000` | Total de draws do Gibbs sampler |
| `n_burn` | `int` | `2000` | Draws descartados |
| `n_thin` | `int` | `1` | Thinning |
| `seed` | `int \| None` | `None` | Reprodutibilidade |

### TVP-VAR sem Stochastic Volatility

```python
model = TVPVAR(
    lags=2,
    stochastic_volatility=False,
    n_draws=10000,
    n_burn=5000,
)
results = model.fit(data)

# Media posterior dos coeficientes em cada t
print(results.time_varying_coefs.keys())
# dict_keys(['const->gdp', 'gdp.L1->gdp', 'infl.L1->gdp', ...])
```

### TVP-VAR com Stochastic Volatility

```python
model = TVPVAR(
    lags=2,
    stochastic_volatility=True,
    n_draws=20000,   # SV precisa de mais draws
    n_burn=10000,
    seed=42,
)
results = model.fit(data)

# Volatilidade variante no tempo
print(results.time_varying_volatility)  # DataFrame (T x K)
```

### IRFs Variantes no Tempo

```python
# IRF em uma data especifica
irf_1980 = results.irf(steps=20, date="1980-01")
irf_2000 = results.irf(steps=20, date="2000-01")
irf_2020 = results.irf(steps=20, date="2020-01")

# Comparar respostas de GDP a choque de juros
for label, irf in [("1980", irf_1980), ("2000", irf_2000), ("2020", irf_2020)]:
    val = irf.value("interest_rate", "gdp", step=8)
    print(f"IRF({label}) rate->GDP h=8: {val:.4f}")
```

!!! tip "Mudanca estrutural"
    Se as IRFs sao qualitativamente diferentes entre periodos, ha evidencia
    de mudanca estrutural na transmissao de politica monetaria. Isso e tipico
    da Great Moderation (1984+) e do periodo pos-2008.

### Integracao com kalmanbox

```python
# O TVP-VAR usa kalmanbox internamente
# Voce pode acessar o modelo state-space subjacente
ss_model = results.state_space_model

# Filtered states (coeficientes filtrados)
filtered = results.filtered_states

# Smoothed states (coeficientes suavizados)
smoothed = results.smoothed_states
```

---

## Interpretacao

### Evolucao dos Coeficientes

```python
# Plotar evolucao do coeficiente de juros sobre GDP
import matplotlib.pyplot as plt

coef = results.time_varying_coefs["rate.L1->gdp"]
plt.plot(coef.index, coef["median"], label="Mediana posterior")
plt.fill_between(
    coef.index, coef["ci_16"], coef["ci_84"],
    alpha=0.3, label="68% CI"
)
plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
plt.title("Efeito da taxa de juros sobre GDP ao longo do tempo")
plt.legend()
plt.show()
```

### Volatilidade Variante

```python
# Se stochastic_volatility=True
vol = results.time_varying_volatility
print(vol.describe())
```

```text
          vol_gdp    vol_infl   vol_rate
mean      0.0084     0.0051     0.0041
std       0.0032     0.0028     0.0019
min       0.0031     0.0014     0.0012
max       0.0213     0.0187     0.0098
```

Picos de volatilidade correspondem a periodos de crise (ex.: 2008-09).

---

## Diagnosticos

### 1. Convergencia MCMC

```python
diag = results.mcmc_diagnostics()
print(f"ESS minimo: {diag['ess_min']:.0f}")
print(f"Maior R-hat: {diag['rhat_max']:.4f}")
# R-hat < 1.1 indica convergencia
```

### 2. Necessidade de Parametros Variantes

```python
# Comparar com VAR de coeficientes fixos
from chronobox import VAR

var_results = VAR(lags=2).fit(data)

print(f"TVP-VAR log-ML: {results.marginal_likelihood():.2f}")
print(f"VAR fixo BIC:    {var_results.bic:.2f}")
```

### 3. Sensibilidade a Prior de Q

```python
for q_scale in [1e-5, 1e-4, 1e-3, 1e-2]:
    model = TVPVAR(lags=2, Q_prior_scale=q_scale, n_draws=5000, n_burn=2000)
    res = model.fit(data)
    val = res.irf(steps=20, date="2000-01").value("interest_rate", "gdp", step=8)
    print(f"Q_scale={q_scale:.0e}: IRF(8) = {val:.4f}")
```

!!! warning "Sensibilidade a Q"
    `Q_prior_scale` muito grande permite variacao excessiva (overfitting temporal).
    Muito pequeno forca coeficientes constantes. Valores tipicos: $10^{-5}$ a $10^{-3}$.

### Checklist de Diagnostico

| Verificacao | Metodo | Esperado |
|---|---|---|
| Convergencia MCMC | ESS, R-hat, trace plots | ESS > 500, R-hat < 1.1 |
| Necessidade de TVP | Comparar com VAR fixo | TVP-VAR com melhor log-ML |
| Sensibilidade a Q | Variar `Q_prior_scale` | Resultados qualitativamente robustos |
| Stochastic volatility | Comparar com/sem SV | SV melhora se volatilidade varia |
| Estabilidade local | Eigenvalues por periodo | Estavel na maioria dos periodos |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import TVPVAR

    model = TVPVAR(
        lags=2,
        stochastic_volatility=True,
        n_draws=20000,
        n_burn=10000,
    )
    results = model.fit(data)

    irf = results.irf(steps=20, date="2000-01")
    coefs = results.time_varying_coefs
    ```

=== "bvarsv (R)"

    ```r
    library(bvarsv)

    # TVP-VAR com stochastic volatility
    fit <- bvar.sv.tvp(
      Y = as.matrix(data),
      p = 2,
      nrep = 20000,
      nburn = 10000
    )

    # Coeficientes variantes no tempo
    # fit$Beta.postmean  (array T x K x (Kp+1))

    # IRF em periodo especifico
    # Implementacao manual necessaria
    ```

**Mapeamento de parametros**:

| chronobox | bvarsv (R) | Descricao |
|---|---|---|
| `lags=2` | `p=2` | Numero de defasagens |
| `stochastic_volatility=True` | Default em `bvar.sv.tvp()` | Volatilidade variante |
| `n_draws=20000` | `nrep=20000` | Total de draws |
| `n_burn=10000` | `nburn=10000` | Burn-in |
| `results.time_varying_coefs` | `fit$Beta.postmean` | Coeficientes variantes |
| `results.irf(date=...)` | Implementacao manual | IRF em data especifica |

---

## Referencias

- Primiceri, G. E. (2005). Time Varying Structural Vector Autoregressions and
  Monetary Policy. *Review of Economic Studies*, 72(3), 821--852.
- Cogley, T. & Sargent, T. J. (2005). Drifts and Volatilities: Monetary Policies
  and Outcomes in the Post WWII US. *Review of Economic Dynamics*, 8(2), 262--302.
- Cogley, T. & Sargent, T. J. (2001). Evolving Post-World War II US Inflation
  Dynamics. *NBER Macroeconomics Annual*, 16, 331--373.
- Carter, C. K. & Kohn, R. (1994). On Gibbs Sampling for State Space Models.
  *Biometrika*, 81(3), 541--553.
