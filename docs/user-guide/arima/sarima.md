---
title: SARIMA
description: ARIMA Sazonal --- modelagem de series temporais com padroes periodicos.
---

# SARIMA --- ARIMA Sazonal

!!! info "Quick Reference"
    - **Classe**: `chronobox.ARIMA` (com `seasonal_order`)
    - **Import**: `from chronobox import ARIMA`
    - **R equivalente**: `forecast::Arima(y, order=c(p,d,q), seasonal=c(P,D,Q))`
    - **Notacao**: ARIMA(p,d,q)(P,D,Q)[s]

---

## Overview

O modelo SARIMA (Seasonal ARIMA) estende o ARIMA para capturar padroes sazonais
periodicos. E o modelo padrao para series com sazonalidade --- dados mensais com
ciclo anual, dados trimestrais, dados horarios com ciclo diario, etc.

### Quando usar

- Serie temporal com padroes sazonais regulares
- A sazonalidade tem periodo conhecido (ex.: $s=12$ para mensal, $s=4$ para trimestral)
- Os padroes sazonais sao aproximadamente estaveis ao longo do tempo
- Previsao com horizonte de pelo menos um ciclo sazonal

---

## Formulacao Matematica

### Equacao Completa

$$
\phi(B)\,\Phi(B^s)\,(1-B)^d\,(1-B^s)^D\, y_t = c + \theta(B)\,\Theta(B^s)\,\varepsilon_t
$$

onde $\varepsilon_t \sim \text{WN}(0, \sigma^2)$.

### Componentes

| Componente | Formula | Descricao |
|---|---|---|
| AR nao sazonal | $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ | Dependencia de lags recentes |
| AR sazonal | $\Phi(B^s) = 1 - \Phi_1 B^s - \cdots - \Phi_P B^{Ps}$ | Dependencia de lags sazonais |
| Diferenciacao | $(1-B)^d$ | Diferenciacao regular |
| Dif. sazonal | $(1-B^s)^D$ | Diferenciacao sazonal |
| MA nao sazonal | $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ | Choques recentes |
| MA sazonal | $\Theta(B^s) = 1 + \Theta_1 B^s + \cdots + \Theta_Q B^{Qs}$ | Choques sazonais |

### Multiplicacao de Polinomios

No SARIMA, os polinomios nao sazonais e sazonais sao **multiplicados**. Por exemplo,
para um ARIMA(1,1,1)(1,1,1)[12]:

**Parte AR**:

$$
\phi(B)\,\Phi(B^{12}) = (1 - \phi_1 B)(1 - \Phi_1 B^{12})
= 1 - \phi_1 B - \Phi_1 B^{12} + \phi_1 \Phi_1 B^{13}
$$

Isso gera dependencia nos lags 1, 12 e 13.

**Parte MA**:

$$
\theta(B)\,\Theta(B^{12}) = (1 + \theta_1 B)(1 + \Theta_1 B^{12})
= 1 + \theta_1 B + \Theta_1 B^{12} + \theta_1 \Theta_1 B^{13}
$$

**Diferenciacao**:

$$
(1-B)(1-B^{12})\,y_t = y_t - y_{t-1} - y_{t-12} + y_{t-13}
$$

### Exemplo: Airline Model

O classico modelo airline de Box-Jenkins e um ARIMA(0,1,1)(0,1,1)[12]:

$$
(1-B)(1-B^{12})\,y_t = (1 + \theta_1 B)(1 + \Theta_1 B^{12})\,\varepsilon_t
$$

---

## Quick Example

```python
from chronobox import ARIMA
from chronobox.datasets import load_airline

# Carregar dados mensais (s=12)
y = load_airline()

# Ajustar SARIMA(1,1,1)(1,1,1)[12]
model = ARIMA(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
results = model.fit(y)

print(results.summary())

# Previsao de 24 meses
fc = results.forecast(steps=24, alpha=0.05)
```

---

## Guia Detalhado

### Construtor

```python
ARIMA(
    order=(p, d, q),
    seasonal_order=(P, D, Q, s),
    trend=None
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `order` | `tuple[int, int, int]` | `(1, 0, 0)` | Ordem nao sazonal (p, d, q) |
| `seasonal_order` | `tuple[int, int, int, int]` | `(0, 0, 0, 0)` | Ordem sazonal (P, D, Q, s) |
| `trend` | `str \| None` | `None` | Tendencia: `'n'`, `'c'`, `'t'`, `'ct'` |

**Significado dos parametros sazonais**:

| Parametro | Descricao | Valores tipicos |
|---|---|---|
| `P` | Ordem AR sazonal | 0, 1, 2 |
| `D` | Diferenciacao sazonal | 0, 1 |
| `Q` | Ordem MA sazonal | 0, 1, 2 |
| `s` | Periodo sazonal | 4 (trim.), 12 (mensal), 52 (semanal) |

### Periodos Sazonais Comuns

| Frequencia dos dados | Periodo $s$ | Exemplo |
|---|---|---|
| Trimestral | 4 | PIB trimestral |
| Mensal | 12 | Producao industrial |
| Semanal | 52 | Vendas semanais |
| Diario (semana util) | 5 | Retornos de acoes |
| Diario (semana completa) | 7 | Trafego web |
| Horario | 24 | Consumo de energia |

### Escolhendo (P, D, Q)

1. **Diferenciacao sazonal $D$**: aplique uma diferenciacao sazonal e teste
   estacionariedade. Raramente $D > 1$.

2. **Ordem AR sazonal $P$**: observe a PACF nos lags sazonais ($s, 2s, 3s, \ldots$).
   Corte apos lag $Ps$ sugere AR($P$) sazonal.

3. **Ordem MA sazonal $Q$**: observe a ACF nos lags sazonais.
   Corte apos lag $Qs$ sugere MA($Q$) sazonal.

```python
from chronobox import ARIMA
from chronobox.datasets import load_airline

y = load_airline()

# Airline model classico: (0,1,1)(0,1,1)[12]
model = ARIMA(
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12)
)
results = model.fit(y)
print(results.summary())
```

!!! tip "Parcimonia"
    Modelos SARIMA com muitos parametros tendem a sobreajustar. Na pratica,
    $P \leq 2$ e $Q \leq 2$ sao suficientes. Se precisar de ordens maiores,
    considere modelos alternativos (ETS, STL + ARIMA).

---

## Interpretacao

### Coeficientes Sazonais

```text
                    ARIMA(1,1,1)(1,1,1)[12] Results
==========================================================================
              coef     std err       t     P>|t|    [0.025     0.975]
--------------------------------------------------------------------------
ar.L1        0.1245     0.0921    1.352    0.176    -0.056     0.305
ma.L1       -0.4412     0.0812   -5.434    0.000    -0.600    -0.282
ar.S.L12     0.0631     0.1044    0.605    0.546    -0.142     0.268
ma.S.L12    -0.5732     0.0889   -6.448    0.000    -0.748    -0.399
sigma2     119.8200     9.3400   12.829    0.000   101.514   138.126
==========================================================================
```

| Coeficiente | Significado |
|---|---|
| `ar.L1` ($\phi_1$) | Persistencia de curto prazo |
| `ma.L1` ($\theta_1$) | Resposta a choques recentes |
| `ar.S.L12` ($\Phi_1$) | Persistencia sazonal (ano-a-ano) |
| `ma.S.L12` ($\Theta_1$) | Resposta a choques sazonais |

### Nome do Modelo

O chronobox exibe o nome completo do modelo:

```python
print(results.model_name)
# "ARIMA(1,1,1)(1,1,1)[12]"
```

---

## Diagnosticos

Os diagnosticos do SARIMA sao os mesmos do ARIMA, com atencao especial aos
**lags sazonais**.

### Autocorrelacao nos Lags Sazonais

```python
from chronobox.tests_stat import ljung_box_test

# Testar ate 2 ciclos sazonais
lb = ljung_box_test(results.residuals, lags=24)
print(f"Ljung-Box (24 lags) p-value: {lb.pvalue:.4f}")
```

!!! warning "Picos sazonais na ACF residual"
    Se a ACF dos residuos ainda mostra picos nos lags $s, 2s, 3s$, o componente
    sazonal esta mal especificado. Aumente $P$ ou $Q$.

### Checklist

| Verificacao | Como testar | Resultado esperado |
|---|---|---|
| Residuos sem autocorrelacao | Ljung-Box (lags $\geq 2s$) | $p > 0.05$ |
| Sem padrao sazonal residual | ACF nos lags $s, 2s, 3s$ | Dentro das bandas |
| Normalidade | Jarque-Bera | $p > 0.05$ |
| Homocedasticidade | ARCH-LM | $p > 0.05$ |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import ARIMA

    model = ARIMA(
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 12)
    )
    results = model.fit(y)
    fc = results.forecast(steps=24)
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    fit <- Arima(y,
                 order = c(1, 1, 1),
                 seasonal = c(0, 1, 1))
    fc <- forecast(fit, h = 24)
    ```

=== "stats (R)"

    ```r
    fit <- arima(y,
                 order = c(1, 1, 1),
                 seasonal = list(order = c(0, 1, 1),
                                 period = 12))
    predict(fit, n.ahead = 24)
    ```

**Mapeamento de parametros**:

| chronobox | forecast (R) | stats (R) |
|---|---|---|
| `seasonal_order=(P,D,Q,s)` | `seasonal=c(P,D,Q)` | `seasonal=list(order=c(P,D,Q), period=s)` |
| `order=(p,d,q)` | `order=c(p,d,q)` | `order=c(p,d,q)` |

!!! note "Periodo sazonal no R"
    No `forecast::Arima`, o periodo e inferido do objeto `ts()`. No
    `stats::arima`, deve ser passado explicitamente em `seasonal$period`.
    No chronobox, o periodo e sempre o quarto elemento de `seasonal_order`.

---

## Referencias

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
  *Time Series Analysis: Forecasting and Control*. 5th ed. Wiley.
- Hyndman, R. J. & Athanasopoulos, G. (2021).
  *Forecasting: Principles and Practice*. 3rd ed. OTexts. Cap. 9.
- Box, G. E. P. & Jenkins, G. M. (1970). *Time Series Analysis:
  Forecasting and Control*. Holden-Day. (Airline model original)
