---
title: VAR(p)
description: Modelo Vetorial Autoregressivo --- estimacao OLS, selecao de lags e analise de estabilidade.
---

# VAR(p)

!!! info "Quick Reference"
    - **Classe**: `chronobox.VAR`
    - **Import**: `from chronobox import VAR`
    - **R equivalente**: `vars::VAR(y, p, type)`
    - **Estimacao**: OLS equacao por equacao

---

## Overview

O modelo VAR (Vector Autoregression) e a generalizacao multivariada do AR(p).
Em vez de modelar uma unica serie, o VAR modela **$K$ series simultaneamente**,
permitindo que cada variavel dependa de seus proprios valores passados e dos
valores passados de todas as outras variaveis do sistema.

Introduzido por Sims (1980), o VAR tornou-se o modelo padrao para analise
macroeconomica empirica por sua abordagem **ateoretica** --- nao exige
restricoes estruturais a priori.

### Quando usar

- Analise de interdependencias entre multiplas series temporais
- Previsao conjunta de variaveis macroeconomicas
- Base para IRF, FEVD e testes de causalidade de Granger
- Series estacionarias (ou diferenciadas para atingir estacionariedade)

---

## Formulacao Matematica

### Equacao do Modelo

Um VAR(p) com $K$ variaveis e definido por:

$$
\mathbf{y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{y}_{t-1} + \mathbf{A}_2 \mathbf{y}_{t-2} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{u}_t
$$

onde:

- $\mathbf{y}_t = (y_{1t}, y_{2t}, \ldots, y_{Kt})'$ e o vetor $K \times 1$ de variaveis endogenas
- $\mathbf{c}$ e o vetor $K \times 1$ de interceptos
- $\mathbf{A}_i$ sao matrizes $K \times K$ de coeficientes para a defasagem $i$
- $\mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_u)$ e o vetor de inovacoes

### Forma Compacta

Empilhando todas as observacoes:

$$
\mathbf{Y} = \mathbf{B}\mathbf{Z} + \mathbf{U}
$$

onde:

- $\mathbf{Y} = (\mathbf{y}_1, \ldots, \mathbf{y}_T)$ tem dimensao $K \times T$
- $\mathbf{B} = (\mathbf{c}, \mathbf{A}_1, \ldots, \mathbf{A}_p)$ tem dimensao $K \times (Kp + 1)$
- $\mathbf{Z}_t = (1, \mathbf{y}'_{t-1}, \ldots, \mathbf{y}'_{t-p})'$ e o vetor de regressores

### Exemplo: VAR(1) Bivariado

Para $K = 2$ variaveis (por exemplo, PIB e inflacao):

$$
\begin{pmatrix} y_{1t} \\ y_{2t} \end{pmatrix}
=
\begin{pmatrix} c_1 \\ c_2 \end{pmatrix}
+
\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}
\begin{pmatrix} y_{1,t-1} \\ y_{2,t-1} \end{pmatrix}
+
\begin{pmatrix} u_{1t} \\ u_{2t} \end{pmatrix}
$$

- $a_{12}$: efeito da inflacao passada sobre o PIB atual
- $a_{21}$: efeito do PIB passado sobre a inflacao atual

### Estimacao OLS

O estimador OLS equacao por equacao e:

$$
\hat{\mathbf{B}} = \mathbf{Y}\mathbf{Z}'(\mathbf{Z}\mathbf{Z}')^{-1}
$$

Como todas as equacoes possuem os mesmos regressores (mesmas defasagens de todas
as variaveis), o OLS equacao por equacao e **identico ao GLS** e e **BLUE**.

A estimativa da matriz de covariancia dos residuos e:

$$
\hat{\boldsymbol{\Sigma}}_u = \frac{1}{T - Kp - 1} \sum_{t=1}^{T} \hat{\mathbf{u}}_t \hat{\mathbf{u}}_t'
$$

### Condicao de Estabilidade

O VAR(p) e estavel (estacionario) se e somente se todos os eigenvalues da
**companion matrix** estao dentro do circulo unitario:

$$
\mathbf{F} =
\begin{pmatrix}
\mathbf{A}_1 & \mathbf{A}_2 & \cdots & \mathbf{A}_{p-1} & \mathbf{A}_p \\
\mathbf{I}_K & \mathbf{0} & \cdots & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{I}_K & \cdots & \mathbf{0} & \mathbf{0} \\
\vdots & & \ddots & & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & \mathbf{I}_K & \mathbf{0}
\end{pmatrix}
$$

Estabilidade requer: $|\lambda_i(\mathbf{F})| < 1$ para todo $i$.

### Selecao de Lags

A ordem $p$ e selecionada minimizando criterios de informacao:

$$
\text{AIC}(p) = \ln|\hat{\boldsymbol{\Sigma}}_u(p)| + \frac{2pK^2}{T}
$$

$$
\text{BIC}(p) = \ln|\hat{\boldsymbol{\Sigma}}_u(p)| + \frac{pK^2 \ln T}{T}
$$

$$
\text{HQC}(p) = \ln|\hat{\boldsymbol{\Sigma}}_u(p)| + \frac{2pK^2 \ln(\ln T)}{T}
$$

$$
\text{FPE}(p) = \left(\frac{T + Kp + 1}{T - Kp - 1}\right)^K |\hat{\boldsymbol{\Sigma}}_u(p)|
$$

!!! tip "Regra pratica"
    BIC tende a selecionar modelos mais parcimoniosos (menos lags), enquanto
    AIC tende a selecionar mais lags. Para previsao, AIC costuma performar
    melhor; para identificacao do verdadeiro modelo, BIC e consistente.

---

## Quick Example

```python
from chronobox import VAR
from chronobox.datasets import load_macro

import pandas as pd

# Carregar dados macroeconomicos
data = load_macro()  # GDP, inflacao, taxa de juros

# Ajustar VAR(2)
model = VAR(lags=2)
results = model.fit(data)

# Resumo
print(results.summary())

# Selecao automatica de lags
model_auto = VAR(lags='auto', max_lags=8, ic='aic')
results_auto = model_auto.fit(data)
print(f"Lags selecionados: {results_auto.k_ar}")

# Previsao 8 passos a frente
fc = results.forecast(steps=8)
print(fc["forecast"])
```

---

## Guia Detalhado

### Construtor

```python
VAR(
    lags=1,          # Numero de defasagens (int ou 'auto')
    trend='c',       # Componente deterministico
    max_lags=12,     # Maximo de lags para selecao automatica
    ic='aic',        # Criterio de informacao para selecao
    exog=None        # Variaveis exogenas
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `lags` | `int \| str` | `1` | Ordem do VAR ou `'auto'` para selecao automatica |
| `trend` | `str` | `'c'` | `'n'` (nenhum), `'c'` (constante), `'t'` (tendencia), `'ct'` (ambos) |
| `max_lags` | `int` | `12` | Maximo de lags quando `lags='auto'` |
| `ic` | `str` | `'aic'` | Criterio: `'aic'`, `'bic'`, `'hqc'`, `'fpe'` |
| `exog` | `array-like \| None` | `None` | Variaveis exogenas adicionais |

**Opcoes de `trend`**:

| Valor | Significado | Equacao |
|---|---|---|
| `'n'` | Sem constante | $\mathbf{y}_t = \mathbf{A}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{u}_t$ |
| `'c'` | Constante | $\mathbf{y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{u}_t$ |
| `'t'` | Tendencia linear | $\mathbf{y}_t = \boldsymbol{\delta} t + \mathbf{A}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{u}_t$ |
| `'ct'` | Constante + tendencia | $\mathbf{y}_t = \mathbf{c} + \boldsymbol{\delta} t + \mathbf{A}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{u}_t$ |

### Metodo `fit()`

```python
results = model.fit(
    endog,           # DataFrame ou array (T x K)
    method='ols'     # Metodo de estimacao
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `endog` | `pd.DataFrame \| np.ndarray` | --- | Dados multivariados ($T \times K$) |
| `method` | `str` | `'ols'` | Metodo de estimacao: `'ols'` |

### Selecao Automatica de Lags

```python
model = VAR(lags='auto', max_lags=8, ic='aic')
results = model.fit(data)

# Tabela comparativa de criterios
print(results.lag_selection)
```

```text
  lags        AIC        BIC        HQC        FPE
     1    -12.341    -12.012    -12.204    4.38e-06
     2    -12.567    -12.045    -12.356    3.48e-06  ← AIC
     3    -12.523    -11.808    -12.231    3.62e-06
     4    -12.498    -11.590    -12.125    3.73e-06
```

### Verificacao de Estabilidade

```python
# Eigenvalues da companion matrix
eigenvalues = results.eigenvalues
print("Eigenvalues (modulo):")
for ev in eigenvalues:
    print(f"  |{ev:.4f}| = {abs(ev):.4f}")

# Teste rapido
print(f"VAR estavel: {results.is_stable}")
```

!!! warning "VAR instavel"
    Se algum eigenvalue tem modulo $\geq 1$, o VAR e instavel e as previsoes
    serao explosivas. Reduza o numero de lags ou verifique se as series
    precisam ser diferenciadas.

### Variaveis Exogenas

```python
import numpy as np

# Dummy para crise financeira de 2008
crisis = np.where((data.index >= '2008-09') & (data.index <= '2009-06'), 1, 0)

model = VAR(lags=2, exog=crisis)
results = model.fit(data)
```

---

## Interpretacao

### Lendo o `summary()`

```python
print(results.summary())
```

```text
                         VAR(2) Results
==========================================================================
No. of Equations:        3         No. Observations:          120
Nobs per equation:     118         Log Likelihood:         452.31
Estimation method:     OLS         AIC:                   -12.567
                                   BIC:                   -12.045
                                   HQC:                   -12.356
==========================================================================

Equation: gdp
--------------------------------------------------------------------------
              coef     std err       t     P>|t|    [0.025     0.975]
--------------------------------------------------------------------------
const        0.0132     0.0054    2.444    0.016     0.003     0.024
gdp.L1       0.3241     0.0923    3.512    0.001     0.142     0.507
gdp.L2      -0.1023     0.0901   -1.135    0.259    -0.281     0.076
infl.L1     -0.0534     0.0312   -1.712    0.090    -0.115     0.008
infl.L2      0.0215     0.0298    0.721    0.472    -0.037     0.080
rate.L1      0.0102     0.0189    0.540    0.590    -0.027     0.048
rate.L2     -0.0078     0.0185   -0.422    0.674    -0.044     0.029

Equation: infl
--------------------------------------------------------------------------
...
```

**Como interpretar os coeficientes**:

| Coeficiente | Significado |
|---|---|
| `gdp.L1` na equacao `gdp` | Persistencia do PIB (efeito proprio) |
| `infl.L1` na equacao `gdp` | Efeito da inflacao passada sobre o PIB |
| `rate.L1` na equacao `infl` | Efeito dos juros passados sobre a inflacao |

!!! tip "Coeficientes vs causalidade"
    Coeficientes individuais em VARs sao dificeis de interpretar isoladamente
    devido a multicolinearidade entre lags. Use **IRF** e **FEVD** para
    interpretar a dinamica do sistema, e **Granger causality** para testar
    precedencia temporal.

### Numero de Parametros

Um VAR(p) com $K$ variaveis e constante estima:

$$
\text{Parametros por equacao} = Kp + 1
$$

$$
\text{Total de parametros} = K(Kp + 1)
$$

Para $K=3$ e $p=4$: $3 \times (3 \times 4 + 1) = 39$ parametros. O numero
cresce rapidamente, o que pode causar sobreajuste em amostras pequenas.

---

## Diagnosticos

### 1. Estabilidade

```python
print(f"VAR estavel: {results.is_stable}")
print(f"Maior eigenvalue (modulo): {max(abs(results.eigenvalues)):.4f}")
```

### 2. Autocorrelacao dos Residuos

```python
from chronobox.tests_stat import portmanteau_test

# Teste de Portmanteau multivariado
pt = portmanteau_test(results.residuals, lags=12, adjusted=True)
print(f"Portmanteau p-value: {pt.pvalue:.4f}")
# p > 0.05 → residuos nao autocorrelacionados
```

### 3. Normalidade Multivariada

```python
from chronobox.tests_stat import multivariate_normality_test

norm = multivariate_normality_test(results.residuals)
print(f"Jarque-Bera multivariado p-value: {norm.pvalue:.4f}")
```

### 4. Heterocedasticidade

```python
from chronobox.tests_stat import arch_test_multivariate

arch = arch_test_multivariate(results.residuals, lags=5)
print(f"ARCH multivariado p-value: {arch.pvalue:.4f}")
```

### Checklist de Diagnostico

| Teste | $H_0$ | Resultado Desejado |
|---|---|---|
| Estabilidade | --- | Todos $\|\lambda_i\| < 1$ |
| Portmanteau | Sem autocorrelacao | $p > 0.05$ |
| JB multivariado | Normalidade | $p > 0.05$ |
| ARCH multivariado | Sem heterocedasticidade | $p > 0.05$ |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import VAR

    model = VAR(lags=2, trend='c')
    results = model.fit(data)

    print(results.summary())
    fc = results.forecast(steps=12)
    irf = results.irf(steps=20)
    ```

=== "vars (R)"

    ```r
    library(vars)

    # Selecao de lags
    VARselect(y, lag.max = 8, type = "const")

    # Estimar VAR(2)
    fit <- VAR(y, p = 2, type = "const")
    summary(fit)

    # Previsao
    predict(fit, n.ahead = 12)

    # IRF
    irf(fit, impulse = "gdp", response = "infl", n.ahead = 20)
    ```

**Mapeamento de parametros**:

| chronobox | vars (R) | Descricao |
|---|---|---|
| `lags=2` | `p=2` | Numero de defasagens |
| `trend='c'` | `type="const"` | Constante |
| `trend='ct'` | `type="both"` | Constante + tendencia |
| `trend='t'` | `type="trend"` | Apenas tendencia |
| `trend='n'` | `type="none"` | Sem deterministicos |
| `lags='auto'` | `VARselect()` | Selecao automatica |
| `results.forecast(steps=h)` | `predict(fit, n.ahead=h)` | Previsao |
| `results.irf(steps=h)` | `irf(fit, n.ahead=h)` | Impulso-resposta |

---

## Referencias

- Sims, C. A. (1980). Macroeconomics and Reality. *Econometrica*, 48(1), 1--48.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Kilian, L. & Lutkepohl, H. (2017). *Structural Vector Autoregressive Analysis*.
  Cambridge University Press.
