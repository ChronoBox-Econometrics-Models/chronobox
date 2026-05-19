---
title: BVAR
description: VAR Bayesiano --- Minnesota prior, Normal-Wishart, SSVS e estimacao por Gibbs sampling.
---

# BVAR (Bayesian VAR)

!!! info "Quick Reference"
    - **Classe**: `chronobox.BayesianVAR`
    - **Import**: `from chronobox import BayesianVAR`
    - **R equivalente**: `BVAR::bvar()`, `bvartools::bvar()`
    - **Estimacao**: Gibbs sampling (posterior via MCMC)

---

## Overview

O BVAR (Bayesian Vector Autoregression) combina a verossimilhanca do VAR com
**distribuicoes a priori** sobre os coeficientes. Isso permite:

- **Regularizacao**: evita sobreajuste em sistemas com muitas variaveis e lags
- **Shrinkage**: encolhe coeficientes na direcao de um modelo parcimonioso
- **Quantificacao de incerteza**: posterior completa dos parametros
- **Previsao superior**: priors bem calibradas melhoram previsoes fora da amostra

O BVAR tornou-se o modelo padrao para previsao macroeconomica em bancos centrais
apos os trabalhos de Doan, Litterman & Sims (1984) e Banbura, Giannone &
Reichlin (2010).

### Quando usar

- Sistemas com muitas variaveis ($K > 5$) onde o VAR classico sofre com a "maldição da dimensionalidade"
- Previsao macroeconomica onde regularizacao melhora performance
- Quando ha informacao a priori economica sobre os parametros
- Quando se deseja quantificacao Bayesiana completa de incerteza

---

## Formulacao Matematica

### Framework Bayesiano

O modelo VAR(p) com $K$ variaveis:

$$
\mathbf{y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{u}_t, \qquad \mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})
$$

Vetorizando: $\mathbf{y}_t = \mathbf{B} \mathbf{z}_t + \mathbf{u}_t$, onde
$\boldsymbol{\beta} = \text{vec}(\mathbf{B})$.

A inferencia Bayesiana combina:

$$
\underbrace{p(\boldsymbol{\beta}, \boldsymbol{\Sigma} \mid \mathbf{Y})}_{\text{posterior}} \propto \underbrace{p(\mathbf{Y} \mid \boldsymbol{\beta}, \boldsymbol{\Sigma})}_{\text{verossimilhanca}} \times \underbrace{p(\boldsymbol{\beta}, \boldsymbol{\Sigma})}_{\text{prior}}
$$

### Minnesota Prior

A Minnesota prior (Litterman, 1986) centra os coeficientes em um **random walk**
--- cada variavel depende apenas de seu proprio primeiro lag:

$$
E[A_{i,jj}^{(1)}] = 1, \qquad E[A_{i,jk}^{(l)}] = 0 \quad \text{para } j \neq k \text{ ou } l > 1
$$

A variancia dos coeficientes segue uma estrutura hierarquica:

$$
\text{Var}(A_{i,jk}^{(l)}) =
\begin{cases}
\dfrac{\lambda_1^2}{l^2} & \text{se } j = k \quad \text{(proprio lag)} \\[10pt]
\dfrac{\lambda_1^2 \lambda_2^2}{l^2} \cdot \dfrac{\sigma_j^2}{\sigma_k^2} & \text{se } j \neq k \quad \text{(cross-variable)}
\end{cases}
$$

onde:

- $\lambda_1$ (**tightness**): controla o shrinkage geral --- valores menores = mais shrinkage
- $\lambda_2$ (**cross-variable shrinkage**): penaliza coeficientes de outras variaveis ($0 < \lambda_2 \leq 1$)
- $l$: ordem da defasagem --- lags mais distantes recebem mais shrinkage
- $\sigma_j^2 / \sigma_k^2$: ajuste de escala entre variaveis

!!! tip "Calibracao de hiperparametros"
    Valores tipicos: $\lambda_1 \in [0.1, 0.3]$ (moderado shrinkage),
    $\lambda_2 = 0.5$ (cross-variable mais apertado). Para muitas variaveis,
    use $\lambda_1 = 0.1$ para shrinkage forte.

### Normal-Wishart Prior

A prior conjugada Natural Normal-Wishart:

$$
\boldsymbol{\beta} \mid \boldsymbol{\Sigma} \sim \mathcal{N}(\boldsymbol{\beta}_0, \boldsymbol{\Sigma} \otimes \mathbf{V}_0)
$$

$$
\boldsymbol{\Sigma} \sim \mathcal{IW}(\mathbf{S}_0, \nu_0)
$$

onde $\mathcal{IW}$ e a distribuicao Inverse-Wishart. A posterior e analitica:

$$
\boldsymbol{\beta} \mid \boldsymbol{\Sigma}, \mathbf{Y} \sim \mathcal{N}(\bar{\boldsymbol{\beta}}, \boldsymbol{\Sigma} \otimes \bar{\mathbf{V}})
$$

$$
\boldsymbol{\Sigma} \mid \mathbf{Y} \sim \mathcal{IW}(\bar{\mathbf{S}}, \bar{\nu})
$$

com:

$$
\bar{\mathbf{V}} = (\mathbf{V}_0^{-1} + \mathbf{Z}\mathbf{Z}')^{-1}
$$

$$
\bar{\boldsymbol{\beta}} = \text{vec}\bigl(\bar{\mathbf{V}}(\mathbf{V}_0^{-1}\mathbf{B}_0 + \mathbf{Z}\mathbf{Y}')\bigr)
$$

### SSVS (Stochastic Search Variable Selection)

O SSVS (George, Sun & Ni, 2008) e uma prior de **spike-and-slab** que seleciona
variaveis automaticamente. Para cada coeficiente $\beta_j$:

$$
\beta_j \mid \gamma_j \sim (1 - \gamma_j) \cdot \mathcal{N}(0, \tau_0^2) + \gamma_j \cdot \mathcal{N}(0, \tau_1^2)
$$

onde $\gamma_j \in \{0, 1\}$ e um indicador:

- $\gamma_j = 0$: prior concentrada em zero (spike) --- variavel excluida
- $\gamma_j = 1$: prior difusa (slab) --- variavel incluida

O Gibbs sampler alterna entre:

1. Amostrar $\boldsymbol{\beta} \mid \boldsymbol{\gamma}, \boldsymbol{\Sigma}, \mathbf{Y}$
2. Amostrar $\boldsymbol{\gamma} \mid \boldsymbol{\beta}, \boldsymbol{\Sigma}, \mathbf{Y}$
3. Amostrar $\boldsymbol{\Sigma} \mid \boldsymbol{\beta}, \mathbf{Y}$

### Gibbs Sampling

Para priors nao conjugadas, o Gibbs sampler gera amostras da posterior:

1. Inicializar $\boldsymbol{\beta}^{(0)}, \boldsymbol{\Sigma}^{(0)}$
2. Para $s = 1, \ldots, S$:
    - Amostrar $\boldsymbol{\beta}^{(s)} \sim p(\boldsymbol{\beta} \mid \boldsymbol{\Sigma}^{(s-1)}, \mathbf{Y})$
    - Amostrar $\boldsymbol{\Sigma}^{(s)} \sim p(\boldsymbol{\Sigma} \mid \boldsymbol{\beta}^{(s)}, \mathbf{Y})$
3. Descartar burn-in e usar as amostras restantes para inferencia

---

## Quick Example

```python
from chronobox import BayesianVAR
from chronobox.datasets import load_macro

data = load_macro()

# BVAR com Minnesota prior
model = BayesianVAR(
    lags=4,
    prior="minnesota",
    lambda1=0.2,      # tightness geral
    lambda2=0.5,      # cross-variable shrinkage
    n_draws=5000,
    n_burn=2000,
    seed=42,
)
results = model.fit(data)

# Resumo posterior
print(results.summary())

# Previsao com bandas de credibilidade
fc = results.forecast(steps=12, ci=0.90)
print(fc["forecast"])
print(fc["ci_lower"])
print(fc["ci_upper"])
```

---

## Guia Detalhado

### Construtor

```python
BayesianVAR(
    lags=1,                # Numero de defasagens
    prior="minnesota",     # Tipo de prior
    lambda1=0.2,           # Overall tightness
    lambda2=0.5,           # Cross-variable shrinkage
    lambda3=1.0,           # Lag decay (expoente)
    lambda4=1e5,           # Constante/exogenas (prior difusa)
    trend='c',             # Componente deterministico
    n_draws=5000,          # Numero total de draws do Gibbs
    n_burn=2000,           # Burn-in
    n_thin=1,              # Thinning
    seed=None              # Semente
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `lags` | `int` | `1` | Numero de defasagens do VAR |
| `prior` | `str` | `'minnesota'` | `'minnesota'`, `'normal_wishart'`, `'ssvs'`, `'flat'` |
| `lambda1` | `float` | `0.2` | Overall tightness ($\lambda_1$) |
| `lambda2` | `float` | `0.5` | Cross-variable shrinkage ($\lambda_2$) |
| `lambda3` | `float` | `1.0` | Lag decay exponent |
| `lambda4` | `float` | `1e5` | Prior variance para deterministicos |
| `trend` | `str` | `'c'` | `'n'`, `'c'`, `'t'`, `'ct'` |
| `n_draws` | `int` | `5000` | Numero total de draws MCMC |
| `n_burn` | `int` | `2000` | Draws descartados (burn-in) |
| `n_thin` | `int` | `1` | Manter 1 a cada `n_thin` draws |
| `seed` | `int \| None` | `None` | Semente para reprodutibilidade |

### Minnesota Prior

```python
model = BayesianVAR(
    lags=4,
    prior="minnesota",
    lambda1=0.1,    # Shrinkage forte
    lambda2=0.5,    # Cross-variable mais apertado
    n_draws=10000,
    n_burn=5000,
)
results = model.fit(data)

# Posterior dos coeficientes
print(results.coef_posterior_mean)
print(results.coef_posterior_std)
```

### Normal-Wishart Prior

```python
model = BayesianVAR(
    lags=4,
    prior="normal_wishart",
    lambda1=0.2,
    n_draws=10000,
    n_burn=5000,
)
results = model.fit(data)
```

### SSVS

```python
model = BayesianVAR(
    lags=4,
    prior="ssvs",
    n_draws=20000,   # SSVS precisa de mais draws
    n_burn=10000,
    seed=42,
)
results = model.fit(data)

# Probabilidade de inclusao de cada coeficiente
print(results.inclusion_probs)
```

### Convergencia MCMC

```python
# Trace plots e diagnosticos
print(results.mcmc_diagnostics())

# Effective sample size
print(f"ESS minimo: {results.ess_min:.0f}")

# Geweke test
print(results.geweke_test())
```

!!! warning "Convergencia"
    Sempre verifique a convergencia do Gibbs sampler antes de interpretar
    resultados. Se o ESS (Effective Sample Size) for muito baixo, aumente
    `n_draws` e/ou `n_thin`.

---

## Interpretacao

### Posterior dos Coeficientes

```python
# Media posterior
print(results.summary())
```

```text
                  BayesianVAR(4) Posterior Summary
========================================================================
Prior:            Minnesota (lambda1=0.2, lambda2=0.5)
Draws:            5000 (burn-in: 2000, effective: 3000)
ESS min:          1847
========================================================================

Equation: gdp
------------------------------------------------------------------------
              post_mean  post_std   ci_5%     ci_95%    prob>0
------------------------------------------------------------------------
const          0.0128    0.0052    0.0043     0.0213    0.993
gdp.L1         0.3102    0.0856    0.1694     0.4510    1.000
gdp.L2        -0.0876    0.0812   -0.2213     0.0461    0.141
infl.L1       -0.0412    0.0287   -0.0884     0.0060    0.076
...
```

A coluna `prob>0` indica a probabilidade posterior de o coeficiente ser positivo
--- analoga ao p-value (mas com interpretacao Bayesiana).

### Previsao com Incerteza

```python
fc = results.forecast(steps=12, ci=0.90)

# Mediana e bandas
print(fc["forecast"])      # media posterior
print(fc["ci_lower"])      # percentil 5
print(fc["ci_upper"])      # percentil 95
```

!!! tip "Vantagem Bayesiana"
    As bandas de previsao do BVAR incorporam **incerteza dos parametros** e
    **incerteza dos erros futuros**, ao contrario do VAR frequentista que
    condiciona nos parametros estimados.

---

## Diagnosticos

### 1. Convergencia MCMC

```python
diag = results.mcmc_diagnostics()
print(f"ESS minimo: {diag['ess_min']:.0f}")
print(f"Geweke p-value minimo: {diag['geweke_pvalue_min']:.4f}")
```

### 2. Marginal Likelihood

```python
# Para comparacao de modelos (priors, lags)
ml = results.marginal_likelihood()
print(f"Log marginal likelihood: {ml:.2f}")
```

### 3. Previsao Fora da Amostra

```python
# Leave-last-h-out forecast evaluation
from chronobox.evaluation import forecast_evaluation

eval_result = forecast_evaluation(
    model=BayesianVAR(lags=4, prior="minnesota", lambda1=0.2),
    data=data,
    h=8,
    window="expanding",
)
print(eval_result.rmse)
```

### Checklist de Diagnostico

| Verificacao | Metodo | Esperado |
|---|---|---|
| Convergencia MCMC | Trace plots, ESS, Geweke | ESS > 1000, Geweke p > 0.05 |
| Estabilidade | Eigenvalues por draw | Maioria dos draws estavel |
| Prior sensitivity | Variar $\lambda_1$, $\lambda_2$ | Resultados robustos |
| Forecast accuracy | RMSE fora da amostra | Melhor que VAR classico |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import BayesianVAR

    model = BayesianVAR(
        lags=4,
        prior="minnesota",
        lambda1=0.2,
        lambda2=0.5,
        n_draws=10000,
        n_burn=5000,
    )
    results = model.fit(data)

    fc = results.forecast(steps=12, ci=0.90)
    irf = results.irf(steps=40)
    ```

=== "BVAR (R)"

    ```r
    library(BVAR)

    # Minnesota prior
    mn <- bv_minnesota(
      lambda = bv_lambda(mode = 0.2, sd = 0.4),
      alpha  = bv_alpha(mode = 2)
    )

    # Estimar BVAR(4)
    fit <- bvar(y, lags = 4, priors = mn,
                n_draw = 10000, n_burn = 5000)
    summary(fit)

    # Previsao
    predict(fit, horizon = 12, conf_bands = 0.90)

    # IRF
    irf(fit, n.ahead = 40)
    ```

**Mapeamento de parametros**:

| chronobox | BVAR (R) | Descricao |
|---|---|---|
| `prior="minnesota"` | `bv_minnesota()` | Minnesota prior |
| `lambda1=0.2` | `bv_lambda(mode=0.2)` | Overall tightness |
| `lambda2=0.5` | `bv_alpha(mode=2)` | Cross-variable (escala diferente) |
| `n_draws=10000` | `n_draw=10000` | Total de draws |
| `n_burn=5000` | `n_burn=5000` | Burn-in |
| `prior="ssvs"` | `bv_ssvs()` | SSVS prior |

---

## Referencias

- Litterman, R. B. (1986). Forecasting with Bayesian Vector Autoregressions ---
  Five Years of Experience. *Journal of Business & Economic Statistics*, 4(1), 25--38.
- Doan, T., Litterman, R. & Sims, C. (1984). Forecasting and Conditional Projection
  Using Realistic Prior Distributions. *Econometric Reviews*, 3(1), 1--100.
- Banbura, M., Giannone, D. & Reichlin, L. (2010). Large Bayesian Vector Auto Regressions.
  *Journal of Applied Econometrics*, 25(1), 71--92.
- George, E. I., Sun, D. & Ni, S. (2008). Bayesian Stochastic Search for VAR Model
  Restrictions. *Journal of Econometrics*, 142(1), 553--580.
- Koop, G. & Korobilis, D. (2010). Bayesian Multivariate Time Series Methods for
  Empirical Macroeconomics. *Foundations and Trends in Econometrics*, 3(4), 267--358.
