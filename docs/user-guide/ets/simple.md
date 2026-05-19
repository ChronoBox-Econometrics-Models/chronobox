---
title: Simple Exponential Smoothing (SES)
description: Suavizacao Exponencial Simples --- ETS(A,N,N) e ETS(M,N,N) para series sem tendencia nem sazonalidade.
---

# Simple Exponential Smoothing (SES)

!!! info "Quick Reference"
    - **Classe**: `chronobox.ETS`
    - **Import**: `from chronobox import ETS`
    - **Modelos**: ETS(A,N,N) e ETS(M,N,N)
    - **R equivalente**: `forecast::ets(y, model="ANN")` / `fable::ETS(y ~ error("A") + trend("N") + season("N"))`
    - **Parametro**: $\alpha \in (0, 1)$

---

## Overview

A **Suavizacao Exponencial Simples** (SES) e o modelo mais basico da familia ETS.
Ele e adequado para series temporais sem tendencia nem sazonalidade --- ou seja,
series que flutuam em torno de um nivel constante (possivelmente variavel no tempo).

A ideia central e que a previsao e uma **media ponderada exponencialmente** das
observacoes passadas, com pesos que decaem geometricamente:

$$
\hat{y}_{T+1|T} = \alpha y_T + \alpha(1-\alpha) y_{T-1} + \alpha(1-\alpha)^2 y_{T-2} + \cdots
$$

O parametro $\alpha$ controla a **taxa de decaimento**:

- $\alpha \to 1$: previsao segue de perto a observacao mais recente (reativo)
- $\alpha \to 0$: previsao e quase uma media de longo prazo (suave)

### Quando usar

- Serie sem tendencia clara nem sazonalidade
- Previsoes de curto prazo para dados ruidosos
- Como benchmark simples para comparar com modelos mais complexos

---

## Formulacao Matematica

### Forma de Recorrencia

A forma classica da SES e a equacao de atualizacao do nivel:

$$
\hat{y}_{t+1|t} = \alpha y_t + (1 - \alpha)\hat{y}_{t|t-1}
$$

Equivalentemente, a **equacao de correcao de erro**:

$$
\hat{y}_{t+1|t} = \hat{y}_{t|t-1} + \alpha(y_t - \hat{y}_{t|t-1}) = \hat{y}_{t|t-1} + \alpha\, e_t
$$

onde $e_t = y_t - \hat{y}_{t|t-1}$ e o erro de previsao um passo a frente.

### State-Space: ETS(A,N,N)

O modelo com **erro aditivo** possui a seguinte representacao em espaco de estados:

**Equacao de observacao**:

$$
y_t = l_{t-1} + \varepsilon_t
$$

**Equacao de transicao (nivel)**:

$$
l_t = l_{t-1} + \alpha\,\varepsilon_t
$$

onde $\varepsilon_t \sim \text{NID}(0, \sigma^2)$.

A previsao $h$ passos a frente e **flat** (constante):

$$
\hat{y}_{T+h|T} = l_T, \quad h = 1, 2, \ldots
$$

com intervalo de previsao:

$$
\hat{y}_{T+h|T} \pm z_{\alpha/2}\,\sigma\sqrt{1 + (h-1)\alpha^2}
$$

### State-Space: ETS(M,N,N)

O modelo com **erro multiplicativo**:

**Equacao de observacao**:

$$
y_t = l_{t-1}(1 + \varepsilon_t)
$$

**Equacao de transicao (nivel)**:

$$
l_t = l_{t-1}(1 + \alpha\,\varepsilon_t)
$$

onde $\varepsilon_t \sim \text{NID}(0, \sigma^2)$.

!!! note "Quando usar erro multiplicativo?"
    O ETS(M,N,N) e preferivel quando a variancia dos erros e **proporcional
    ao nivel** da serie. Se a serie assume apenas valores positivos e a
    variabilidade cresce com o nivel, erro multiplicativo e mais adequado.

### Parametros

| Parametro | Dominio | Descricao |
|---|---|---|
| $\alpha$ | $(0, 1)$ | Taxa de suavizacao do nivel |
| $l_0$ | $\mathbb{R}$ | Estado inicial do nivel |

---

## Quick Example

```python
from chronobox import ETS
from chronobox.datasets import load_oil

# Carregar producao anual de petroleo
y = load_oil()

# Ajustar SES --- ETS(A,N,N)
model = ETS(error='A', trend='N', seasonal='N')
results = model.fit(y)

# Resumo
print(results.summary())

# Previsao 5 anos a frente
fc = results.forecast(steps=5, alpha=0.05)
print(fc["forecast"])
```

---

## Guia Detalhado

### Construtor

```python
ETS(
    error='A',       # Tipo de erro: 'A' (aditivo) ou 'M' (multiplicativo)
    trend='N',       # Tipo de tendencia: 'N' (nenhuma)
    seasonal='N',    # Tipo de sazonalidade: 'N' (nenhuma)
    damped=False     # Tendencia amortecida (nao aplicavel ao SES)
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `error` | `str` | `'A'` | Tipo de erro: `'A'` ou `'M'` |
| `trend` | `str` | `'N'` | Tipo de tendencia |
| `seasonal` | `str` | `'N'` | Tipo de sazonalidade |
| `damped` | `bool` | `False` | Se a tendencia e amortecida |

### Metodo `fit()`

```python
results = model.fit(
    endog,                # Serie temporal (array-like)
    optimized=True,       # Estimar parametros por MLE
    start_params=None,    # Valores iniciais para o otimizador
    maxiter=500           # Iteracoes maximas
)
```

### Metodo `forecast()`

```python
fc = results.forecast(
    steps=12,    # Horizonte de previsao
    alpha=0.05   # Nivel de significancia (IC 95%)
)
```

Retorna um dicionario:

| Chave | Tipo | Descricao |
|---|---|---|
| `'forecast'` | `np.ndarray` | Previsoes pontuais (constantes para SES) |
| `'lower'` | `np.ndarray` | Limite inferior do IC |
| `'upper'` | `np.ndarray` | Limite superior do IC |

!!! tip "Previsoes flat"
    A SES produz previsoes **constantes** (flat) para todos os horizontes.
    Isso e esperado --- sem tendencia nem sazonalidade, a melhor previsao
    e simplesmente o nivel atual estimado $l_T$.

### Comparando ETS(A,N,N) vs ETS(M,N,N)

=== "Erro Aditivo --- ETS(A,N,N)"

    ```python
    model_a = ETS(error='A', trend='N', seasonal='N')
    results_a = model_a.fit(y)
    print(f"AIC: {results_a.aic:.2f}")
    print(f"Alpha: {results_a.params['smoothing_level']:.4f}")
    ```

    Indicado quando a variancia dos erros e **constante** ao longo da serie.

=== "Erro Multiplicativo --- ETS(M,N,N)"

    ```python
    model_m = ETS(error='M', trend='N', seasonal='N')
    results_m = model_m.fit(y)
    print(f"AIC: {results_m.aic:.2f}")
    print(f"Alpha: {results_m.params['smoothing_level']:.4f}")
    ```

    Indicado quando a variancia dos erros e **proporcional ao nivel**.

### Efeito do Parametro $\alpha$

```python
import numpy as np

for alpha_val in [0.1, 0.5, 0.9]:
    model = ETS(error='A', trend='N', seasonal='N')
    results = model.fit(y, start_params={'smoothing_level': alpha_val})
    print(f"Alpha={alpha_val:.1f} → AIC={results.aic:.2f}")
```

!!! warning "Alpha proximo dos limites"
    - $\alpha \approx 1$: o modelo e essencialmente um **random walk**,
      muito reativo e com previsoes instaveis
    - $\alpha \approx 0$: o modelo ignora dados recentes, previsao e
      quase a media historica

---

## Interpretacao

### Lendo o `summary()`

```python
print(results.summary())
```

```text
                       ETS(A,N,N) Results
==========================================================================
Dep. Variable:                y       No. Observations:            50
Method:                     MLE       Log Likelihood:         -183.95
Date:                2026-04-09       AIC:                     371.90
                                      BIC:                     377.64
                                      AICc:                    372.36
==========================================================================
         Smoothing Parameters
------------------------------------------
alpha              0.8340    (0.06, 0.99)
==========================================================================
         Initial States
------------------------------------------
l0               111.2100
==========================================================================
sigma2            99.3621
==========================================================================
```

**Como interpretar**:

| Campo | Significado |
|---|---|
| `alpha` | Parametro de suavizacao. Proximo de 1 = serie muito reativa |
| `l0` | Estado inicial estimado do nivel |
| `sigma2` | Variancia estimada dos erros $\sigma^2$ |
| `AIC/BIC` | Criterios de informacao (menor = melhor) |

### Criterios de Informacao

$$
\text{AIC} = -2\ln\hat{L} + 2k, \qquad
\text{BIC} = -2\ln\hat{L} + k\ln n
$$

$$
\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}
$$

onde $k$ e o numero de parametros estimados (para SES: $\alpha$, $l_0$, $\sigma^2$, logo $k=3$).

---

## Diagnosticos

Apos ajustar o modelo, valide a qualidade dos residuos:

### 1. Autocorrelacao Residual

```python
from chronobox.tests_stat import ljung_box_test

lb = ljung_box_test(results.residuals, lags=10)
print(f"Ljung-Box p-value: {lb.pvalue:.4f}")
# p > 0.05 → residuos nao autocorrelacionados ✓
```

!!! warning "Residuos autocorrelacionados"
    Se o teste rejeitar $H_0$, o SES esta mal especificado. Considere
    adicionar componentes de **tendencia** (Holt) ou **sazonalidade**
    (Holt-Winters), ou use Auto-ETS para selecao automatica.

### 2. Normalidade dos Residuos

```python
from chronobox.tests_stat import jarque_bera_test

jb = jarque_bera_test(results.residuals)
print(f"Jarque-Bera p-value: {jb.pvalue:.4f}")
# p > 0.05 → residuos normais ✓
```

### 3. Heterocedasticidade

```python
from chronobox.tests_stat import arch_test

arch = arch_test(results.residuals, lags=5)
print(f"ARCH p-value: {arch.pvalue:.4f}")
# p > 0.05 → variancia constante ✓
```

!!! tip "Erro aditivo vs multiplicativo"
    Se o teste ARCH rejeitar $H_0$ no ETS(A,N,N), experimente o ETS(M,N,N).
    O erro multiplicativo captura naturalmente a heterocedasticidade
    proporcional ao nivel.

### Checklist de Diagnostico

| Teste | $H_0$ | Resultado Desejado |
|---|---|---|
| Ljung-Box | Sem autocorrelacao | $p > 0.05$ |
| Jarque-Bera | Normalidade | $p > 0.05$ |
| ARCH-LM | Sem heterocedasticidade | $p > 0.05$ |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import ETS

    model = ETS(error='A', trend='N', seasonal='N')
    results = model.fit(y)

    print(results.summary())
    fc = results.forecast(steps=5)
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    fit <- ets(y, model = "ANN")

    summary(fit)
    fc <- forecast(fit, h = 5)
    ```

=== "fable (R)"

    ```r
    library(fable)

    fit <- y_tsibble |>
      model(ETS(y ~ error("A") + trend("N") + season("N")))

    report(fit)
    fc <- forecast(fit, h = 5)
    ```

**Mapeamento de parametros**:

| chronobox | forecast (R) | fable (R) |
|---|---|---|
| `error='A'` | `model="ANN"` | `error("A")` |
| `error='M'` | `model="MNN"` | `error("M")` |
| `results.params['smoothing_level']` | `fit$par["alpha"]` | `tidy(fit)` |
| `results.aic` | `fit$aic` | `glance(fit)$AIC` |

---

## Referencias

- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008).
  *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
  Chapters 2--3.
- Hyndman, R. J. & Athanasopoulos, G. (2021).
  *Forecasting: Principles and Practice*. 3rd ed. OTexts. Chapter 8.1.
- Gardner, E. S. (1985). Exponential smoothing: The state of the art.
  *Journal of Forecasting*, 4(1), 1--28.
