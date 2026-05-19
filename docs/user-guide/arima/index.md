---
title: Familia ARIMA
description: Modelos Autoregressivos Integrados de Medias Moveis --- ARIMA, SARIMA, ARFIMA e Auto-ARIMA.
---

# Familia ARIMA

A familia ARIMA (AutoRegressive Integrated Moving Average) e a base da modelagem
univariada de series temporais. Combinando componentes autoregressivos, diferenciacao
e medias moveis, esses modelos capturam uma ampla variedade de padroes temporais.

---

## Equacao Geral

O modelo ARIMA(p,d,q) e definido por:

$$
\phi(B)(1-B)^d\, y_t = c + \theta(B)\,\varepsilon_t
$$

onde:

- $B$ e o **operador de defasagem** (backshift): $B\, y_t = y_{t-1}$
- $\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$ e o **polinomio AR**
- $\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$ e o **polinomio MA**
- $(1-B)^d$ e o **operador de diferenciacao** de ordem $d$
- $c$ e uma constante (intercepto)
- $\varepsilon_t \sim \text{WN}(0, \sigma^2)$ e ruido branco

---

## Modelos da Familia

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **ARIMA(p,d,q)**

    ---

    Modelo base com componentes AR, diferenciacao inteira e MA.

    [:octicons-arrow-right-24: ARIMA](arima.md)

-   :material-calendar-sync:{ .lg .middle } **SARIMA**

    ---

    Extensao sazonal: ARIMA(p,d,q)(P,D,Q)[s] para dados com periodicidade.

    [:octicons-arrow-right-24: SARIMA](sarima.md)

-   :material-memory:{ .lg .middle } **ARFIMA**

    ---

    Integracao fracionaria $(1-B)^d$ com $d \in (-0.5, 0.5)$ para memoria longa.

    [:octicons-arrow-right-24: ARFIMA](arfima.md)

-   :material-auto-fix:{ .lg .middle } **Auto-ARIMA**

    ---

    Selecao automatica de (p,d,q) via criterios de informacao.

    [:octicons-arrow-right-24: Auto-ARIMA](auto-arima.md)

</div>

---

## Quando usar ARIMA?

| Cenario | Modelo Recomendado |
|---|---|
| Serie sem sazonalidade | [ARIMA(p,d,q)](arima.md) |
| Serie com sazonalidade periodica | [SARIMA](sarima.md) |
| Decaimento lento da ACF, memoria longa | [ARFIMA](arfima.md) |
| Selecao automatica de ordem | [Auto-ARIMA](auto-arima.md) |
| Nao sabe qual modelo usar | Comece pelo [Auto-ARIMA](auto-arima.md) |

!!! info "Pre-requisito: Estacionariedade"
    Modelos ARIMA assumem que a serie diferenciada $w_t = (1-B)^d y_t$ e
    estacionaria. Use testes de raiz unitaria (ADF, KPSS, PP) para determinar
    a ordem de diferenciacao $d$ adequada. Veja
    [Diagnosticos: Raiz Unitaria](../../diagnostics/unit-root.md).

---

## Fluxo de Trabalho Tipico

```python
from chronobox import ARIMA, auto_arima
from chronobox.tests_stat import adf_test, ljung_box_test

# 1. Testar estacionariedade
result = adf_test(y)
print(f"ADF p-value: {result.pvalue:.4f}")

# 2. Ajustar modelo (manual ou automatico)
model = ARIMA(order=(1, 1, 1))
results = model.fit(y)

# --- ou ---
results = auto_arima(y, seasonal=True, m=12)

# 3. Diagnosticos
print(results.summary())
lb = ljung_box_test(results.residuals, lags=10)

# 4. Previsao
fc = results.forecast(steps=12, alpha=0.05)
```

---

## Referencias

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
  *Time Series Analysis: Forecasting and Control*. 5th ed. Wiley.
- Hyndman, R. J. & Athanasopoulos, G. (2021).
  *Forecasting: Principles and Practice*. 3rd ed. OTexts.
