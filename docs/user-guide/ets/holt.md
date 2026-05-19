---
title: Holt Linear Trend
description: Metodo de Holt com tendencia linear e damped --- ETS(A,A,N), ETS(A,Ad,N), ETS(M,A,N), ETS(M,Ad,N).
---

# Holt Linear Trend

!!! info "Quick Reference"
    - **Classe**: `chronobox.ETS`
    - **Import**: `from chronobox import ETS`
    - **Modelos**: ETS(A,A,N), ETS(A,A$_d$,N), ETS(M,A,N), ETS(M,A$_d$,N)
    - **R equivalente**: `forecast::ets(y, model="AAN")` / `forecast::holt(y)`
    - **Parametros**: $\alpha, \beta^*, \phi$

---

## Overview

O metodo de **Holt** (1957) estende a SES para series com **tendencia linear**.
Alem de suavizar o **nivel**, ele tambem suaviza a **tendencia** (slope), permitindo
que as previsoes capturem movimentos ascendentes ou descendentes.

A versao **damped** (amortecida), proposta por Gardner & McKenzie (1985), introduz
um parametro $\phi \in (0, 1)$ que gradualmente atenua a tendencia, evitando
previsoes que extrapolam indefinidamente em linha reta.

### Quando usar

- Serie com tendencia clara (crescente ou decrescente) mas sem sazonalidade
- **Holt Linear**: quando a tendencia deve ser projetada indefinidamente
- **Holt Damped**: quando a tendencia deve se atenuar ao longo do horizonte (mais conservador)

!!! tip "Damped e quase sempre melhor"
    Na pratica, o metodo damped tende a produzir previsoes mais acuradas
    que o Holt linear puro, especialmente para horizontes mais longos.
    Makridakis & Hibon (2000) mostraram que o damped trend e consistentemente
    um dos melhores metodos de previsao em competicoes.

---

## Formulacao Matematica

### Equacoes de Recorrencia (Holt Linear)

O metodo de Holt possui duas equacoes de suavizacao:

**Nivel**:

$$
l_t = \alpha y_t + (1 - \alpha)(l_{t-1} + b_{t-1})
$$

**Tendencia**:

$$
b_t = \beta^*(l_t - l_{t-1}) + (1 - \beta^*) b_{t-1}
$$

**Previsao** $h$ passos a frente:

$$
\hat{y}_{T+h|T} = l_T + h\, b_T
$$

### Equacoes de Recorrencia (Holt Damped)

Com amortecimento, a tendencia decai geometricamente:

**Nivel**:

$$
l_t = \alpha y_t + (1 - \alpha)(l_{t-1} + \phi\, b_{t-1})
$$

**Tendencia**:

$$
b_t = \beta^*(l_t - l_{t-1}) + (1 - \beta^*)\phi\, b_{t-1}
$$

**Previsao** $h$ passos a frente:

$$
\hat{y}_{T+h|T} = l_T + (\phi + \phi^2 + \cdots + \phi^h)\, b_T = l_T + \phi\frac{1 - \phi^h}{1 - \phi}\, b_T
$$

Quando $h \to \infty$, a previsao converge para $l_T + \frac{\phi}{1 - \phi}\, b_T$.

### State-Space: ETS(A,A,N)

**Equacao de observacao**:

$$
y_t = l_{t-1} + b_{t-1} + \varepsilon_t
$$

**Equacoes de transicao**:

$$
l_t = l_{t-1} + b_{t-1} + \alpha\,\varepsilon_t
$$

$$
b_t = b_{t-1} + \beta\,\varepsilon_t
$$

onde $\beta = \alpha\beta^*$ e $\varepsilon_t \sim \text{NID}(0, \sigma^2)$.

### State-Space: ETS(A,A$_d$,N)

**Equacao de observacao**:

$$
y_t = l_{t-1} + \phi\, b_{t-1} + \varepsilon_t
$$

**Equacoes de transicao**:

$$
l_t = l_{t-1} + \phi\, b_{t-1} + \alpha\,\varepsilon_t
$$

$$
b_t = \phi\, b_{t-1} + \beta\,\varepsilon_t
$$

### State-Space: ETS(M,A,N)

**Equacao de observacao**:

$$
y_t = (l_{t-1} + b_{t-1})(1 + \varepsilon_t)
$$

**Equacoes de transicao**:

$$
l_t = (l_{t-1} + b_{t-1})(1 + \alpha\,\varepsilon_t)
$$

$$
b_t = b_{t-1} + \beta(l_{t-1} + b_{t-1})\varepsilon_t
$$

### State-Space: ETS(M,A$_d$,N)

**Equacao de observacao**:

$$
y_t = (l_{t-1} + \phi\, b_{t-1})(1 + \varepsilon_t)
$$

**Equacoes de transicao**:

$$
l_t = (l_{t-1} + \phi\, b_{t-1})(1 + \alpha\,\varepsilon_t)
$$

$$
b_t = \phi\, b_{t-1} + \beta(l_{t-1} + \phi\, b_{t-1})\varepsilon_t
$$

### Parametros

| Parametro | Dominio | Descricao |
|---|---|---|
| $\alpha$ | $(0, 1)$ | Taxa de suavizacao do nivel |
| $\beta^*$ | $(0, 1)$ | Taxa de suavizacao da tendencia |
| $\phi$ | $(0, 1)$ | Parametro de amortecimento (damping). $\phi = 1$: sem damping |
| $l_0$ | $\mathbb{R}$ | Estado inicial do nivel |
| $b_0$ | $\mathbb{R}$ | Estado inicial da tendencia |

!!! note "Relacao entre $\beta$ e $\beta^*$"
    Na representacao state-space, usa-se $\beta = \alpha\beta^*$, onde
    $\beta^*$ e o parametro da forma de recorrencia classica. Isso garante
    que $\beta < \alpha$ como condicao de estabilidade.

---

## Quick Example

```python
from chronobox import ETS
from chronobox.datasets import load_air_passengers

# Carregar dados (usar tendencia, ignorar sazonalidade por enquanto)
y = load_air_passengers()

# Holt Damped --- ETS(A,Ad,N)
model = ETS(error='A', trend='A', seasonal='N', damped=True)
results = model.fit(y)

print(results.summary())

# Previsao 24 passos a frente
fc = results.forecast(steps=24, alpha=0.05)
print(fc["forecast"])
```

---

## Guia Detalhado

### Construtor

```python
ETS(
    error='A',       # 'A' ou 'M'
    trend='A',       # 'A' para tendencia aditiva
    seasonal='N',    # 'N' para sem sazonalidade
    damped=True      # True para tendencia amortecida
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `error` | `str` | `'A'` | Tipo de erro: `'A'` ou `'M'` |
| `trend` | `str` | `'A'` | `'A'` para Holt, `'N'` para SES |
| `seasonal` | `str` | `'N'` | `'N'` sem sazonalidade |
| `damped` | `bool` | `False` | `True` para Holt Damped |

### Escolhendo entre Linear e Damped

=== "Holt Linear --- ETS(A,A,N)"

    ```python
    model = ETS(error='A', trend='A', seasonal='N', damped=False)
    results = model.fit(y)
    print(f"AIC: {results.aic:.2f}")
    ```

    Previsoes crescem/decrescem **indefinidamente** em linha reta.
    Use quando ha evidencia de tendencia que persiste no longo prazo.

=== "Holt Damped --- ETS(A,Ad,N)"

    ```python
    model = ETS(error='A', trend='A', seasonal='N', damped=True)
    results = model.fit(y)
    print(f"AIC: {results.aic:.2f}")
    print(f"Phi: {results.params['damping_factor']:.4f}")
    ```

    Previsoes convergem para um **patamar** no longo prazo.
    Mais conservador e geralmente mais acurado para horizontes longos.

=== "Comparacao via AIC"

    ```python
    model_lin = ETS(error='A', trend='A', seasonal='N', damped=False)
    model_dmp = ETS(error='A', trend='A', seasonal='N', damped=True)

    res_lin = model_lin.fit(y)
    res_dmp = model_dmp.fit(y)

    print(f"Holt Linear AIC: {res_lin.aic:.2f}")
    print(f"Holt Damped AIC: {res_dmp.aic:.2f}")
    print(f"Melhor: {'Damped' if res_dmp.aic < res_lin.aic else 'Linear'}")
    ```

### O parametro $\phi$

```python
# Verificar o valor estimado de phi
print(f"Phi estimado: {results.params['damping_factor']:.4f}")
```

| Valor de $\phi$ | Comportamento |
|---|---|
| $\phi \approx 1.0$ | Quase sem damping (tendencia persistente) |
| $\phi \approx 0.9$ | Damping moderado (padrao em muitas implementacoes) |
| $\phi \approx 0.8$ | Damping forte (tendencia se dissipa rapidamente) |
| $\phi \to 0$ | Converge para SES (sem tendencia) |

!!! warning "Intervalo de $\phi$"
    O parametro $\phi$ e tipicamente restrito a $(0.8, 0.98)$ na estimacao
    para evitar solucoes degeneradas. Valores fora desse intervalo geralmente
    indicam que o modelo nao e adequado.

---

## Interpretacao

### Lendo o `summary()`

```python
print(results.summary())
```

```text
                      ETS(A,Ad,N) Results
==========================================================================
Dep. Variable:                y       No. Observations:           144
Method:                     MLE       Log Likelihood:         -686.42
Date:                2026-04-09       AIC:                    1382.84
                                      BIC:                    1397.64
                                      AICc:                   1383.54
==========================================================================
         Smoothing Parameters
------------------------------------------
alpha              0.9241    (0.06, 0.99)
beta*              0.0312    (0.001, 0.30)
phi                0.9800    (0.80, 0.98)
==========================================================================
         Initial States
------------------------------------------
l0               117.4200
b0                 2.1340
==========================================================================
sigma2           132.9800
==========================================================================
```

**Como interpretar**:

| Campo | Significado |
|---|---|
| `alpha` | Suavizacao do nivel. Proximo de 1 = nivel reage rapido |
| `beta*` | Suavizacao da tendencia. Proximo de 0 = tendencia estavel |
| `phi` | Amortecimento. Proximo de 1 = tendencia persiste |
| `l0` | Nivel inicial estimado |
| `b0` | Tendencia inicial estimada (unidades/periodo) |
| `sigma2` | Variancia estimada dos erros |

### Equivalencia ARIMA

| ETS | ARIMA equivalente |
|---|---|
| ETS(A,A,N) | ARIMA(0,2,2) |
| ETS(A,A$_d$,N) | ARIMA(1,1,2) |

---

## Diagnosticos

### Checklist

```python
from chronobox.tests_stat import ljung_box_test, jarque_bera_test, arch_test

residuals = results.residuals

# 1. Autocorrelacao
lb = ljung_box_test(residuals, lags=10)
print(f"Ljung-Box p-value: {lb.pvalue:.4f}")

# 2. Normalidade
jb = jarque_bera_test(residuals)
print(f"Jarque-Bera p-value: {jb.pvalue:.4f}")

# 3. Heterocedasticidade
arch = arch_test(residuals, lags=5)
print(f"ARCH p-value: {arch.pvalue:.4f}")
```

| Teste | $H_0$ | Resultado Desejado |
|---|---|---|
| Ljung-Box | Sem autocorrelacao | $p > 0.05$ |
| Jarque-Bera | Normalidade | $p > 0.05$ |
| ARCH-LM | Sem heterocedasticidade | $p > 0.05$ |

!!! warning "Residuos com padrao sazonal"
    Se os residuos mostrarem autocorrelacao nos lags sazonais (ex: lag 12
    para dados mensais), o modelo precisa de um componente sazonal.
    Use Holt-Winters em vez de Holt.

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import ETS

    # Holt Damped
    model = ETS(error='A', trend='A', seasonal='N', damped=True)
    results = model.fit(y)

    print(results.summary())
    fc = results.forecast(steps=12)
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    # Holt Damped
    fit <- ets(y, model = "AAN", damped = TRUE)

    # Ou usando a funcao holt() diretamente
    fit <- holt(y, damped = TRUE, h = 12)

    summary(fit)
    ```

=== "fable (R)"

    ```r
    library(fable)

    fit <- y_tsibble |>
      model(ETS(y ~ error("A") + trend("Ad") + season("N")))

    report(fit)
    fc <- forecast(fit, h = 12)
    ```

**Mapeamento de parametros**:

| chronobox | forecast (R) | fable (R) |
|---|---|---|
| `error='A', trend='A'` | `model="AAN"` | `error("A") + trend("A")` |
| `damped=True` | `damped=TRUE` | `trend("Ad")` |
| `results.params['smoothing_trend']` | `fit$par["beta"]` | `tidy(fit)` |
| `results.params['damping_factor']` | `fit$par["phi"]` | `tidy(fit)` |

---

## Referencias

- Holt, C. C. (1957). Forecasting seasonals and trends by exponentially
  weighted moving averages. ONR Memorandum No. 52. (Reprinted 2004,
  *International Journal of Forecasting*, 20(1), 5--10.)
- Gardner, E. S. & McKenzie, E. (1985). Forecasting trends in time series.
  *Management Science*, 31(10), 1237--1246.
- Makridakis, S. & Hibon, M. (2000). The M3-Competition: results, conclusions
  and implications. *International Journal of Forecasting*, 16(4), 451--476.
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008).
  *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
