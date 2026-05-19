---
title: ARIMA(p,d,q)
description: Modelo Autoregressivo Integrado de Medias Moveis --- estimacao, previsao e diagnosticos.
---

# ARIMA(p,d,q)

!!! info "Quick Reference"
    - **Classe**: `chronobox.ARIMA`
    - **Import**: `from chronobox import ARIMA`
    - **R equivalente**: `forecast::Arima(y, order=c(p,d,q))`
    - **Metodos de estimacao**: MLE (Kalman), CSS, CSS-MLE

---

## Overview

O modelo ARIMA (AutoRegressive Integrated Moving Average) combina tres componentes
para modelar series temporais:

- **AR(p)** --- Autoregressivo: o valor atual depende de $p$ valores passados
- **I(d)** --- Integrado: a serie e diferenciada $d$ vezes para atingir estacionariedade
- **MA(q)** --- Media Movel: o valor atual depende de $q$ erros passados

E o modelo mais utilizado em econometria de series temporais para previsao de curto
e medio prazo de series nao sazonais.

### Quando usar

- Serie temporal univariada sem sazonalidade forte
- Dados que se tornam estacionarios apos diferenciacao
- Previsoes de curto a medio prazo
- Quando voce precisa de intervalos de confianca para as previsoes

---

## Formulacao Matematica

### Equacao do Modelo

$$
\phi(B)(1-B)^d\, y_t = c + \theta(B)\,\varepsilon_t
$$

onde $\varepsilon_t \sim \text{WN}(0, \sigma^2)$.

### Operador Backshift

O operador $B$ (backshift ou lag) desloca a serie no tempo:

$$
B^k\, y_t = y_{t-k}
$$

O operador de diferenciacao e:

$$
(1-B)^d\, y_t = \Delta^d y_t
$$

Para $d=1$: $(1-B)y_t = y_t - y_{t-1}$. Para $d=2$: $(1-B)^2 y_t = y_t - 2y_{t-1} + y_{t-2}$.

### Polinomios AR e MA

**Polinomio Autoregressivo**:

$$
\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p
$$

**Polinomio de Media Movel**:

$$
\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q
$$

### Forma Expandida

Um ARIMA(1,1,1) expandido fica:

$$
y_t - y_{t-1} = c + \phi_1(y_{t-1} - y_{t-2}) + \varepsilon_t + \theta_1 \varepsilon_{t-1}
$$

### Condicoes de Estacionariedade e Invertibilidade

- **Estacionariedade**: as raizes de $\phi(B) = 0$ devem estar fora do circulo unitario
- **Invertibilidade**: as raizes de $\theta(B) = 0$ devem estar fora do circulo unitario

---

## Quick Example

```python
from chronobox import ARIMA
from chronobox.datasets import load_airline

# Carregar dados
y = load_airline()

# Ajustar ARIMA(1,1,1)
model = ARIMA(order=(1, 1, 1))
results = model.fit(y)

# Resumo dos resultados
print(results.summary())

# Previsao 12 passos a frente
fc = results.forecast(steps=12, alpha=0.05)
print(fc["forecast"])
```

---

## Guia Detalhado

### Construtor

```python
ARIMA(
    order=(p, d, q),          # Ordem do modelo (AR, diferenciacao, MA)
    seasonal_order=(0,0,0,0), # Ordem sazonal (ver SARIMA)
    trend=None                # Componente deterministico
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `order` | `tuple[int, int, int]` | `(1, 0, 0)` | Ordens (p, d, q) do modelo |
| `seasonal_order` | `tuple[int, int, int, int]` | `(0, 0, 0, 0)` | Ordem sazonal (P, D, Q, s) |
| `trend` | `str \| None` | `None` | Tendencia: `'n'`, `'c'`, `'t'`, `'ct'` |

**Opcoes de `trend`**:

| Valor | Significado | Equacao |
|---|---|---|
| `'n'` | Sem constante | $\phi(B)\Delta^d y_t = \theta(B)\varepsilon_t$ |
| `'c'` | Constante (drift se $d>0$) | $\phi(B)\Delta^d y_t = c + \theta(B)\varepsilon_t$ |
| `'t'` | Tendencia linear | $\phi(B)\Delta^d y_t = \beta t + \theta(B)\varepsilon_t$ |
| `'ct'` | Constante + tendencia | $\phi(B)\Delta^d y_t = c + \beta t + \theta(B)\varepsilon_t$ |

### Metodo `fit()`

```python
results = model.fit(
    endog,            # Serie temporal (array-like)
    method='css-mle', # Metodo de estimacao
    maxiter=500       # Iteracoes maximas do otimizador
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `endog` | `array-like` | --- | Serie temporal observada |
| `method` | `str` | `'css-mle'` | Metodo de estimacao |
| `maxiter` | `int` | `500` | Numero maximo de iteracoes |

**Metodos de estimacao**:

=== "CSS-MLE (default)"

    Inicia com CSS (Conditional Sum of Squares) e refina com MLE.
    Combina velocidade do CSS com precisao do MLE.

    ```python
    results = model.fit(y, method='css-mle')
    ```

=== "MLE"

    Maxima Verossimilhanca exata via **filtro de Kalman** (kalmanbox).
    Usa inicializacao difusa exata para estados nao estacionarios.

    ```python
    results = model.fit(y, method='mle')
    ```

=== "CSS"

    Conditional Sum of Squares. Mais rapido, mas perde as primeiras
    $\max(p, q)$ observacoes. Adequado para series longas.

    ```python
    results = model.fit(y, method='css')
    ```

### Metodo `forecast()`

```python
fc = results.forecast(
    steps=12,    # Horizonte de previsao
    alpha=0.05   # Nivel de significancia (IC 95%)
)
```

Retorna um dicionario com:

| Chave | Tipo | Descricao |
|---|---|---|
| `'forecast'` | `np.ndarray` | Previsoes pontuais |
| `'lower'` | `np.ndarray` | Limite inferior do IC |
| `'upper'` | `np.ndarray` | Limite superior do IC |

### Escolhendo (p, d, q)

1. **Ordem de diferenciacao $d$**: use testes de raiz unitaria (ADF, KPSS)
2. **Ordem AR $p$**: observe a PACF --- corta apos lag $p$
3. **Ordem MA $q$**: observe a ACF --- corta apos lag $q$

```python
from chronobox.tests_stat import adf_test, kpss_test

# Determinar d
adf = adf_test(y)
print(f"ADF p-value: {adf.pvalue:.4f}")  # p < 0.05 → estacionaria

# Se nao estacionaria, diferenciar e testar novamente
if adf.pvalue > 0.05:
    adf_diff = adf_test(y.diff().dropna())
    print(f"ADF (1a diff) p-value: {adf_diff.pvalue:.4f}")
```

!!! tip "Regra pratica"
    Na maioria dos casos economicos, $d \leq 2$, $p \leq 5$ e $q \leq 5$.
    Se voce precisa de ordens maiores, considere modelos alternativos.

---

## Interpretacao

### Lendo o `summary()`

```python
print(results.summary())
```

```text
                          ARIMA(1,1,1) Results
==========================================================================
Dep. Variable:                y       No. Observations:           144
Method:                   css-mle     Log Likelihood:         -504.92
Date:                  2026-04-09     AIC:                    1015.84
                                      BIC:                    1024.72
                                      AICc:                   1016.14
==========================================================================
              coef     std err       t     P>|t|    [0.025     0.975]
--------------------------------------------------------------------------
ar.L1        0.2341     0.0892    2.625    0.009     0.059     0.409
ma.L1       -0.5823     0.0734   -7.934    0.000    -0.726    -0.438
sigma2     132.4500    10.2100   12.973    0.000   112.438   152.462
==========================================================================
Ljung-Box (Q):                8.42     Prob(Q):                 0.491
Jarque-Bera (JB):             3.21     Prob(JB):                0.201
==========================================================================
```

**Como interpretar**:

| Campo | Significado |
|---|---|
| `ar.L1` | Coeficiente AR($\phi_1$). Positivo = persistencia |
| `ma.L1` | Coeficiente MA($\theta_1$). Negativo = correcao de erros passados |
| `sigma2` | Variancia do ruido $\sigma^2$ |
| `AIC/BIC` | Criterios de informacao (menor = melhor) |
| `Ljung-Box` | Teste de autocorrelacao residual. $p > 0.05$ = residuos OK |
| `Jarque-Bera` | Teste de normalidade residual. $p > 0.05$ = residuos normais |

### Criterios de Informacao

$$
\text{AIC} = -2\ln\hat{L} + 2k, \qquad
\text{BIC} = -2\ln\hat{L} + k\ln n
$$

$$
\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}
$$

onde $k$ e o numero de parametros e $n$ o numero de observacoes.

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
    Se o teste Ljung-Box rejeitar $H_0$ ($p < 0.05$), o modelo esta
    mal especificado. Aumente $p$ ou $q$ e reestime.

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
    from chronobox import ARIMA

    model = ARIMA(order=(1, 1, 1), trend='c')
    results = model.fit(y, method='mle')

    print(results.summary())
    fc = results.forecast(steps=12)
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    fit <- Arima(y, order = c(1, 1, 1),
                 include.drift = TRUE,
                 method = "ML")

    summary(fit)
    fc <- forecast(fit, h = 12)
    ```

=== "stats (R)"

    ```r
    fit <- arima(y, order = c(1, 1, 1),
                 method = "ML")

    summary(fit)
    predict(fit, n.ahead = 12)
    ```

**Mapeamento de parametros**:

| chronobox | forecast (R) | stats (R) |
|---|---|---|
| `order=(p,d,q)` | `order=c(p,d,q)` | `order=c(p,d,q)` |
| `trend='c'` | `include.drift=TRUE` | --- |
| `trend='n'` | `include.drift=FALSE` | --- |
| `method='mle'` | `method="ML"` | `method="ML"` |
| `method='css'` | `method="CSS"` | `method="CSS"` |
| `results.aic` | `fit$aic` | `fit$aic` |

---

## Referencias

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
  *Time Series Analysis: Forecasting and Control*. 5th ed. Wiley.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Hyndman, R. J. & Athanasopoulos, G. (2021).
  *Forecasting: Principles and Practice*. 3rd ed. OTexts.
- Ljung, G. M. & Box, G. E. P. (1978). On a Measure of Lack of Fit in
  Time Series Models. *Biometrika*, 65(2), 297--303.
