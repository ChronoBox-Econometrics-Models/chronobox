---
title: VECM
description: Modelo de Correcao de Erros Vetorial --- cointegracao, Johansen procedure, relacoes de longo prazo.
---

# VECM (Vector Error Correction Model)

!!! info "Quick Reference"
    - **Classe**: `chronobox.VECM`
    - **Import**: `from chronobox import VECM`
    - **R equivalente**: `urca::cajorls()`, `urca::ca.jo()`, `vars::vec2var()`
    - **Estimacao**: Johansen MLE

---

## Overview

O VECM (Vector Error Correction Model) e a representacao adequada para um
sistema de series temporais **I(1) cointegradas**. Enquanto o VAR em diferencas
perde a informacao de longo prazo, o VECM preserva as **relacoes de equilibrio**
entre as variaveis, modelando simultaneamente:

- **Dinamica de curto prazo**: ajustes temporarios (via $\boldsymbol{\Gamma}_i$)
- **Equilibrio de longo prazo**: relacao estavel entre niveis (via $\boldsymbol{\Pi}$)

### Quando usar

- Series I(1) que compartilham uma relacao de equilibrio de longo prazo
- Quando o teste de Johansen indica rank de cointegracao $r \geq 1$
- Analise de velocidade de ajuste ao equilibrio
- Distinguir efeitos de curto e longo prazo

---

## Formulacao Matematica

### Equacao do Modelo

O VECM e derivado de um VAR(p) em niveis, reescrito em termos de diferencas:

$$
\Delta \mathbf{y}_t = \boldsymbol{\Pi} \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \Delta \mathbf{y}_{t-i} + \mathbf{u}_t
$$

onde:

- $\Delta \mathbf{y}_t = \mathbf{y}_t - \mathbf{y}_{t-1}$ e o vetor de diferencas
- $\boldsymbol{\Pi} = \sum_{j=1}^{p} \mathbf{A}_j - \mathbf{I}_K$ e a matriz de impacto de longo prazo
- $\boldsymbol{\Gamma}_i = -\sum_{j=i+1}^{p} \mathbf{A}_j$ sao matrizes de ajuste de curto prazo
- $\mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_u)$

### Decomposicao de Granger

O ponto central do VECM e a decomposicao da matriz $\boldsymbol{\Pi}$:

$$
\boldsymbol{\Pi} = \boldsymbol{\alpha}\boldsymbol{\beta}'
$$

onde:

- $\boldsymbol{\beta}$ e a matriz $K \times r$ de **vetores de cointegracao** (relacoes de longo prazo)
- $\boldsymbol{\alpha}$ e a matriz $K \times r$ de **loading coefficients** (velocidade de ajuste)
- $r = \text{rank}(\boldsymbol{\Pi})$ e o **rank de cointegracao** ($0 < r < K$)

### Interpretacao de $\boldsymbol{\alpha}$ e $\boldsymbol{\beta}$

**Vetores de cointegracao** ($\boldsymbol{\beta}$):

O termo $\boldsymbol{\beta}'\mathbf{y}_{t-1}$ define as relacoes de equilibrio.
Para $r = 1$ com $K = 2$:

$$
\beta_1 y_{1,t-1} + \beta_2 y_{2,t-1} = \text{ECT}_{t-1}
$$

O ECT (Error Correction Term) mede o **desvio do equilibrio** no periodo anterior.

**Loading coefficients** ($\boldsymbol{\alpha}$):

Cada $\alpha_{ij}$ mede a velocidade com que a variavel $i$ se ajusta ao
desvio da relacao de cointegracao $j$:

- $\alpha_{ij} < 0$: a variavel corrige na direcao do equilibrio (esperado)
- $|\alpha_{ij}|$ grande: ajuste rapido
- $\alpha_{ij} \approx 0$: a variavel $i$ e **fracamente exogena** em relacao a relacao $j$

### Rank de Cointegracao

O rank $r$ de $\boldsymbol{\Pi}$ determina o numero de relacoes de equilibrio:

| $r$ | Interpretacao | Modelo Adequado |
|---|---|---|
| $r = 0$ | Sem cointegracao | VAR em diferencas |
| $0 < r < K$ | $r$ relacoes de cointegracao | VECM com rank $r$ |
| $r = K$ | Todas estacionarias | VAR em niveis |

### Componentes Deterministicos

O VECM permite diferentes especificacoes de termos deterministicos:

$$
\Delta \mathbf{y}_t = \boldsymbol{\alpha}(\boldsymbol{\beta}'\mathbf{y}_{t-1} + \boldsymbol{\mu}_0) + \boldsymbol{\delta}_0 + \boldsymbol{\Gamma}_i \Delta \mathbf{y}_{t-i} + \mathbf{u}_t
$$

| Caso | Constante restrita | Constante irrestrita | Tendencia |
|---|---|---|---|
| 1 | Nao | Nao | Nao |
| 2 | Sim ($\boldsymbol{\mu}_0$) | Nao | Nao |
| 3 | Sim | Sim ($\boldsymbol{\delta}_0$) | Nao |
| 4 | Sim | Sim | Restrita |
| 5 | Sim | Sim | Irrestrita |

---

## Quick Example

```python
from chronobox import VECM
from chronobox.tests_stat import johansen_test
from chronobox.datasets import load_macro

# Carregar dados
data = load_macro()  # Series I(1)

# 1. Testar cointegracao via Johansen
joh = johansen_test(data, det_order=0, k_ar_diff=2)
print(joh.summary())

# 2. Estimar VECM com rank r=1
model = VECM(lags=2, coint_rank=1)
results = model.fit(data)

# 3. Resumo
print(results.summary())

# 4. Vetores de cointegracao e loadings
print("Beta (cointegracao):")
print(results.beta)
print("Alpha (loading):")
print(results.alpha)

# 5. Previsao
fc = results.forecast(steps=12)
```

---

## Guia Detalhado

### Procedimento de Johansen

O procedimento de Johansen testa sequencialmente o rank de cointegracao usando
duas estatisticas:

**Estatistica Traco** --- testa $H_0: r \leq r_0$ contra $H_1: r > r_0$:

$$
\lambda_{\text{trace}}(r_0) = -T \sum_{i=r_0+1}^{K} \ln(1 - \hat{\lambda}_i)
$$

**Estatistica Maximo Autovalor** --- testa $H_0: r = r_0$ contra $H_1: r = r_0 + 1$:

$$
\lambda_{\max}(r_0) = -T \ln(1 - \hat{\lambda}_{r_0+1})
$$

onde $\hat{\lambda}_1 > \hat{\lambda}_2 > \cdots > \hat{\lambda}_K$ sao os
eigenvalues ordenados da equacao de autovalores generalizada.

```python
from chronobox.tests_stat import johansen_test

# Teste de Johansen
joh = johansen_test(data, det_order=0, k_ar_diff=2)
print(joh.summary())
```

```text
Johansen Cointegration Test
===================================================
det_order = 0, k_ar_diff = 2

Trace Test:
  H0: r <= 0   stat = 42.31   cv_5% = 29.80   reject ✓
  H0: r <= 1   stat = 14.22   cv_5 = 15.49   fail to reject
  H0: r <= 2   stat =  3.41   cv_5% =  3.84   fail to reject

Max Eigenvalue Test:
  H0: r = 0    stat = 28.09   cv_5% = 21.13   reject ✓
  H0: r = 1    stat = 10.81   cv_5% = 14.26   fail to reject
  H0: r = 2    stat =  3.41   cv_5% =  3.84   fail to reject

Conclusion: 1 cointegrating relation(s) at 5% level
```

!!! warning "Escolha de det_order"
    O parametro `det_order` afeta os valores criticos. Use `det_order=0`
    (constante restrita ao espaco de cointegracao) como ponto de partida.
    Use `det_order=1` se as series apresentam tendencia linear nos niveis.

### Construtor

```python
VECM(
    lags=1,            # Numero de lags em diferencas (p-1 do VAR em niveis)
    coint_rank=1,      # Rank de cointegracao
    det_order=0,       # Termos deterministicos
    exog=None          # Variaveis exogenas
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `lags` | `int` | `1` | Defasagens em $\Delta \mathbf{y}$ (= lags do VAR em niveis $- 1$) |
| `coint_rank` | `int` | `1` | Rank de cointegracao $r$ (de Johansen) |
| `det_order` | `int` | `0` | Termos deterministicos: `0` (constante restrita), `1` (tendencia restrita) |
| `exog` | `array-like \| None` | `None` | Variaveis exogenas |

### Resultados

| Atributo | Tipo | Descricao |
|---|---|---|
| `alpha` | `np.ndarray` | Matriz de loading $\boldsymbol{\alpha}$ ($K \times r$) |
| `beta` | `np.ndarray` | Vetores de cointegracao $\boldsymbol{\beta}$ ($K \times r$) |
| `gamma` | `list[np.ndarray]` | Matrizes de curto prazo $\boldsymbol{\Gamma}_i$ |
| `sigma_u` | `np.ndarray` | Covariancia dos residuos $\hat{\boldsymbol{\Sigma}}_u$ |
| `residuals` | `np.ndarray` | Residuos do modelo |
| `ect` | `np.ndarray` | Error Correction Terms $\boldsymbol{\beta}'\mathbf{y}_{t-1}$ |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |

### Interpretando a Relacao de Longo Prazo

```python
# Vetores de cointegracao (normalizados)
print("Relacao de longo prazo:")
print(results.beta)
```

```text
Relacao de longo prazo:
         coint_eq1
gdp        1.0000
infl      -0.4523
rate       0.2187
```

Interpretacao: no equilibrio de longo prazo,

$$
\text{gdp}_t - 0.4523 \cdot \text{infl}_t + 0.2187 \cdot \text{rate}_t \approx \text{constante}
$$

```python
# Loadings (velocidade de ajuste)
print("Alpha (velocidade de ajuste):")
print(results.alpha)
```

```text
Alpha (velocidade de ajuste):
         coint_eq1
gdp       -0.1234    ← GDP corrige 12.3% do desvio por periodo
infl       0.0567    ← Inflacao se afasta (sinal positivo)
rate      -0.0891    ← Juros corrigem 8.9% do desvio por periodo
```

!!! tip "Exogeneidade fraca"
    Se $\alpha_{i} \approx 0$ e nao significante, a variavel $i$ e
    **fracamente exogena** em relacao a relacao de cointegracao --- ela
    "dirige" a relacao mas nao e afetada por desvios do equilibrio.

---

## Diagnosticos

### 1. Verificar Residuos do VECM

```python
from chronobox.tests_stat import portmanteau_test, multivariate_normality_test

# Autocorrelacao residual
pt = portmanteau_test(results.residuals, lags=12)
print(f"Portmanteau p-value: {pt.pvalue:.4f}")

# Normalidade
norm = multivariate_normality_test(results.residuals)
print(f"JB multivariado p-value: {norm.pvalue:.4f}")
```

### 2. Estabilidade do VECM

```python
# Eigenvalues do companion form
# VECM deve ter exatamente K - r eigenvalues unitarios
eigenvalues = results.eigenvalues
print("Eigenvalues (modulo):")
for ev in eigenvalues:
    mod = abs(ev)
    marker = " ← unitario" if abs(mod - 1.0) < 0.01 else ""
    print(f"  |{ev:.4f}| = {mod:.4f}{marker}")
```

### 3. Verificar ECT

```python
import matplotlib.pyplot as plt

# Plotar o Error Correction Term
plt.figure(figsize=(10, 4))
plt.plot(results.ect)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Error Correction Term")
plt.ylabel("ECT")
plt.show()
```

O ECT deve oscilar ao redor de zero, indicando reversao ao equilibrio.

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import VECM
    from chronobox.tests_stat import johansen_test

    # Teste de Johansen
    joh = johansen_test(data, det_order=0, k_ar_diff=2)

    # Estimar VECM
    model = VECM(lags=2, coint_rank=1)
    results = model.fit(data)

    print(results.beta)   # Vetores de cointegracao
    print(results.alpha)  # Loading coefficients
    ```

=== "urca (R)"

    ```r
    library(urca)

    # Teste de Johansen
    joh <- ca.jo(data, type = "trace", ecdet = "const", K = 3)
    summary(joh)

    # Estimar VECM
    vecm <- cajorls(joh, r = 1)
    vecm$beta   # Vetores de cointegracao
    vecm$rlm    # Modelo completo
    ```

=== "vars (R)"

    ```r
    library(vars)
    library(urca)

    # Johansen
    joh <- ca.jo(data, type = "trace", ecdet = "const", K = 3)

    # Converter VECM para VAR (para IRF, FEVD)
    var_from_vecm <- vec2var(joh, r = 1)

    # Agora pode usar irf(), fevd(), predict()
    irf(var_from_vecm, n.ahead = 20)
    ```

**Mapeamento de parametros**:

| chronobox | urca (R) | Descricao |
|---|---|---|
| `johansen_test(data, det_order=0, k_ar_diff=2)` | `ca.jo(data, ecdet="const", K=3)` | Teste de Johansen |
| `VECM(lags=2, coint_rank=1)` | `cajorls(joh, r=1)` | Estimacao VECM |
| `results.beta` | `vecm$beta` | Vetores de cointegracao |
| `results.alpha` | Extrair de `vecm$rlm` | Loading coefficients |
| `det_order=0` | `ecdet="const"` | Constante restrita |
| `det_order=1` | `ecdet="trend"` | Tendencia restrita |

!!! tip "Nota sobre K vs lags"
    Em `urca::ca.jo()`, o parametro `K` refere-se ao numero de lags do VAR
    **em niveis**. No chronobox, `lags` refere-se ao numero de lags em
    **diferencas**, ou seja, `lags = K - 1`.

---

## Referencias

- Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration
  Vectors in Gaussian Vector Autoregressive Models. *Econometrica*, 59(6),
  1551--1580.
- Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector
  Autoregressive Models*. Oxford University Press.
- Engle, R. F. & Granger, C. W. J. (1987). Co-Integration and Error
  Correction: Representation, Estimation, and Testing. *Econometrica*, 55(2),
  251--276.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
