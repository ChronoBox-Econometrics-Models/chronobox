---
title: ARFIMA
description: ARIMA Fracionariamente Integrado --- modelagem de series com memoria longa.
---

# ARFIMA --- Integracao Fracionaria

!!! info "Quick Reference"
    - **Classe**: `chronobox.ARFIMA`
    - **Import**: `from chronobox import ARFIMA`
    - **R equivalente**: `forecast::arfima(y)` ou `fracdiff::fracdiff(y)`
    - **Caso de uso**: Series com memoria longa (decaimento hiperbolico da ACF)

---

## Overview

O modelo ARFIMA (AutoRegressive Fractionally Integrated Moving Average) generaliza
o ARIMA ao permitir uma **ordem de diferenciacao fracionaria** $d \in (-0.5, 0.5)$.
Enquanto o ARIMA padrao restringe $d$ a inteiros (0 ou 1), o ARFIMA captura
comportamentos intermediarios --- series que nao sao estacionarias ($d=0$) nem
precisam de diferenciacao completa ($d=1$).

Esse comportamento intermediario e chamado **memoria longa**: a funcao de
autocorrelacao decai hiperbolicamente (lentamente) em vez de exponencialmente.

### Quando usar

- A ACF decai muito lentamente, mas a serie parece estacionaria
- Diferenciacao inteira ($d=1$) parece "excessiva" (sobre-diferenciacao)
- Series financeiras de volatilidade, inflacao, taxas de juros de longo prazo
- Dados hidrologicos, geofisicos ou de trafego de rede
- Quando $0 < d < 0.5$: **estacionaria com memoria longa**
- Quando $-0.5 < d < 0$: **antipersistencia** (mean-reverting rapido)

---

## Formulacao Matematica

### Equacao do Modelo

$$
\phi(B)\,(1-B)^d\, y_t = \theta(B)\,\varepsilon_t, \qquad d \in (-0.5,\, 0.5)
$$

### Operador de Diferenciacao Fracionaria

A generalizacao de $(1-B)^d$ para $d$ nao inteiro usa a expansao binomial:

$$
(1-B)^d = \sum_{k=0}^{\infty} \binom{d}{k} (-B)^k
= \sum_{k=0}^{\infty} \pi_k\, B^k
$$

onde os coeficientes sao calculados recursivamente:

$$
\pi_0 = 1, \qquad \pi_k = \pi_{k-1} \cdot \frac{k - 1 - d}{k}, \quad k \geq 1
$$

### Propriedades da Memoria Longa

Para $0 < d < 0.5$, a ACF satisfaz:

$$
\rho(h) \sim C \cdot h^{2d-1} \quad \text{quando } h \to \infty
$$

| Parametro $d$ | Comportamento |
|---|---|
| $d = 0$ | Memoria curta (ARMA padrao) |
| $0 < d < 0.5$ | Memoria longa, estacionario |
| $d = 0.5$ | Fronteira da nao estacionariedade |
| $-0.5 < d < 0$ | Antipersistente (mean-reverting rapido) |

### Espectro

No dominio da frequencia, a memoria longa se manifesta como um polo na frequencia zero:

$$
f(\omega) \sim C \cdot |\omega|^{-2d} \quad \text{quando } \omega \to 0
$$

---

## Quick Example

```python
from chronobox import ARFIMA

# Ajustar ARFIMA(1, d, 1) com estimacao do parametro d
model = ARFIMA(order=(1, 0.0, 1))
results = model.fit(y, method='css', estimate_d=True)

print(f"d estimado: {results.d:.4f}")
print(results.summary())
```

---

## Guia Detalhado

### Construtor

```python
ARFIMA(
    order=(p, d, q)  # (AR order, fracional d, MA order)
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `order` | `tuple[int, float, int]` | `(0, 0.0, 0)` | Ordens (p, d, q) com $d$ fracionario |

### Estimacao do Parametro $d$

O chronobox oferece dois estimadores semiparametricos para $d$:

=== "GPH (Geweke & Porter-Hudak)"

    Regressao log-periodograma nas frequencias baixas.

    ```python
    model = ARFIMA(order=(1, 0.0, 1))
    d_hat = model.estimate_d(y, method='gph')
    print(f"d (GPH): {d_hat:.4f}")
    ```

    - Referencia: Geweke & Porter-Hudak (1983)
    - Parametro de bandwidth: $m = n^{\alpha}$ com $\alpha = 0.5$ (default)

=== "Local Whittle"

    Estimador de maxima verossimilhanca local no dominio da frequencia.

    ```python
    model = ARFIMA(order=(1, 0.0, 1))
    d_hat = model.estimate_d(y, method='whittle')
    print(f"d (Whittle): {d_hat:.4f}")
    ```

    - Referencia: Robinson (1995)
    - Parametro de bandwidth: $m = n^{\alpha}$ com $\alpha = 0.65$ (default)
    - Geralmente mais eficiente que GPH

### Estimacao Conjunta

Para estimar $d$ junto com os parametros AR e MA:

```python
model = ARFIMA(order=(1, 0.0, 1))
results = model.fit(y, method='css', estimate_d=True)

print(f"d = {results.d:.4f}")
print(f"AR(1) = {results.ar_params[0]:.4f}")
print(f"MA(1) = {results.ma_params[0]:.4f}")
```

### Metodo `fit()`

```python
results = model.fit(
    endog,              # Serie temporal
    method='css',       # Metodo de estimacao
    estimate_d=False    # Estimar d conjuntamente?
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `endog` | `array-like` | --- | Serie temporal observada |
| `method` | `str` | `'css'` | Metodo de estimacao |
| `estimate_d` | `bool` | `False` | Se `True`, estima $d$ junto com AR/MA |

### Funcoes Utilitarias

```python
from chronobox.models.arfima import (
    fractional_diff_coefficients,
    fractional_diff,
    estimate_d_gph,
    estimate_d_local_whittle,
    simulate_arfima
)

# Coeficientes da expansao (1-B)^d
coeffs = fractional_diff_coefficients(d=0.3, n=100)

# Aplicar diferenciacao fracionaria
y_frac = fractional_diff(y, d=0.3)

# Simular ARFIMA
y_sim = simulate_arfima(n=500, d=0.4, ar=[0.5], ma=[0.3])
```

---

## Interpretacao

### Resultados do ARFIMA

```python
print(results.summary())
```

```text
                         ARFIMA(1, 0.35, 1) Results
==========================================================================
Dep. Variable:                y       No. Observations:            500
Method:                     css       Log Likelihood:          -712.45
                                      AIC:                    1432.90
                                      BIC:                    1449.68
==========================================================================
              coef     std err       t     P>|t|
--------------------------------------------------------------------------
d            0.3500     0.0420    8.333    0.000
ar.L1        0.2140     0.0650    3.292    0.001
ma.L1        0.1530     0.0710    2.155    0.031
sigma2       1.0240     0.0648   15.802    0.000
==========================================================================
```

| Parametro | Interpretacao |
|---|---|
| `d = 0.35` | Memoria longa significativa. ACF decai como $h^{-0.30}$ |
| `ar.L1` | Componente autoregressivo de curto prazo |
| `ma.L1` | Componente de media movel de curto prazo |

### Interpretando o parametro $d$

| Valor de $d$ | Interpretacao pratica |
|---|---|
| $d \approx 0$ | Sem memoria longa --- use ARMA |
| $d \approx 0.1\text{--}0.2$ | Memoria longa fraca |
| $d \approx 0.3\text{--}0.4$ | Memoria longa moderada |
| $d \approx 0.5$ | Fronteira --- serie quase nao estacionaria |
| $d < 0$ | Antipersistente --- reversao a media mais rapida que ARMA |

!!! tip "Regra pratica"
    Se $d > 0.5$, a serie nao e estacionaria. Aplique diferenciacao inteira
    ($\Delta y_t$) e reestime o ARFIMA na serie diferenciada.

---

## Diagnosticos

### 1. Verificar Memoria Longa

Antes de ajustar um ARFIMA, confirme que a serie tem memoria longa:

```python
from chronobox.models.arfima import estimate_d_gph, estimate_d_local_whittle

d_gph = estimate_d_gph(y)
d_whittle = estimate_d_local_whittle(y)

print(f"d (GPH):     {d_gph:.4f}")
print(f"d (Whittle): {d_whittle:.4f}")
```

!!! note "ARFIMA vs ARIMA com d=1"
    Se $d$ estimado esta proximo de 0, use ARMA. Se esta proximo de 0.5 ou acima,
    considere diferenciar a serie e usar ARIMA. O ARFIMA e mais adequado quando
    $d$ esta claramente entre 0 e 0.5.

### 2. Residuos

```python
from chronobox.tests_stat import ljung_box_test

lb = ljung_box_test(results.resid, lags=20)
print(f"Ljung-Box p-value: {lb.pvalue:.4f}")
```

### 3. Comparar Estimadores

Uma boa pratica e comparar GPH e Whittle. Se divergem muito, os resultados
podem ser sensiveis a escolha de bandwidth.

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import ARFIMA

    model = ARFIMA(order=(1, 0.0, 1))
    results = model.fit(y, estimate_d=True)
    print(f"d = {results.d:.4f}")
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    fit <- arfima(y)
    summary(fit)
    # d estimado automaticamente
    ```

=== "fracdiff (R)"

    ```r
    library(fracdiff)

    fit <- fracdiff(y, nar = 1, nma = 1)
    summary(fit)
    # fit$d contém o d estimado
    ```

**Mapeamento de parametros**:

| chronobox | forecast (R) | fracdiff (R) |
|---|---|---|
| `ARFIMA(order=(p, d, q))` | `arfima(y)` | `fracdiff(y, nar=p, nma=q)` |
| `estimate_d=True` | automatico | automatico |
| `method='gph'` | --- | `fdGPH(y)` |
| `method='whittle'` | --- | `fdSperio(y)` |
| `results.d` | `fit$d` | `fit$d` |

---

## Referencias

- Granger, C. W. J. & Joyeux, R. (1980). An Introduction to Long-Memory
  Time Series Models and Fractional Differencing. *Journal of Time Series
  Analysis*, 1(1), 15--29.
- Hosking, J. R. M. (1981). Fractional Differencing. *Biometrika*, 68(1),
  165--176.
- Geweke, J. & Porter-Hudak, S. (1983). The Estimation and Application of
  Long Memory Time Series Models. *Journal of Time Series Analysis*, 4(4),
  221--238.
- Robinson, P. M. (1995). Gaussian Semiparametric Estimation of Long Range
  Dependence. *Annals of Statistics*, 23(5), 1630--1661.
- Beran, J. (1994). *Statistics for Long-Memory Processes*. Chapman & Hall.
