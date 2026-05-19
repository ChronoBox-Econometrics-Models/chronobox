---
title: Beveridge-Nelson Decomposition
description: Decomposicao de Beveridge-Nelson para separacao de componentes permanente e transitorio.
---

# Beveridge-Nelson Decomposition

!!! info "Quick Reference"
    - **Funcao**: `chronobox.filters.bn_decomposition`
    - **Import**: `from chronobox.filters import bn_decomposition`
    - **R equivalente**: `BeveridgeNelson::BeveridgeNelson(y, ar.order=2)`
    - **Retorno**: `(trend, cycle)` --- arrays NumPy

---

## Overview

A decomposicao de Beveridge e Nelson (1981) separa uma serie integrada $I(1)$ em
um componente **permanente** (tendencia = random walk com drift) e um componente
**transitorio** (ciclo = estacionario). Diferente dos filtros mecanicos como HP
e BK, a decomposicao BN e baseada em um modelo ARIMA para as primeiras diferencas
da serie, o que lhe confere fundamentacao estatistica.

### Quando usar

- Decomposicao permanente-transitoria com fundamento estatistico
- Series $I(1)$ onde a tendencia e estocastica (random walk)
- Quando voce quer que a tendencia seja um random walk (nao uma curva suave)
- Analise de choques permanentes vs transitorios

!!! warning "Hipotese fundamental"
    A decomposicao BN assume que a serie e $I(1)$ --- integrada de ordem 1.
    Aplique testes de raiz unitaria (ADF, KPSS) antes de usar. Se a serie for
    $I(0)$, a decomposicao nao faz sentido; se for $I(2)$, aplique uma
    diferenca antes.

---

## Formulacao Matematica

### Representacao de Wold

Seja $\Delta y_t = y_t - y_{t-1}$ a primeira diferenca de uma serie $I(1)$.
Se $\Delta y_t$ e estacionaria, admite a representacao de Wold:

$$
\Delta y_t = \mu + \psi(B)\,\varepsilon_t = \mu + \sum_{j=0}^{\infty} \psi_j\, \varepsilon_{t-j}
$$

onde $\psi_0 = 1$, $\mu = E[\Delta y_t]$ (drift) e $\varepsilon_t \sim WN(0, \sigma^2)$.

### Decomposicao

Beveridge e Nelson (1981) mostram que $y_t$ pode ser decomposta em:

$$
y_t = \tau_t + c_t
$$

onde:

**Componente permanente (tendencia)**:

$$
\tau_t = \tau_{t-1} + \mu + \psi(1)\,\varepsilon_t
$$

O termo $\psi(1) = \sum_{j=0}^{\infty} \psi_j$ e o **multiplicador de longo prazo**
(long-run multiplier). A tendencia e um random walk com drift $\mu$ e inovacoes
escaladas por $\psi(1)$.

**Componente transitorio (ciclo)**:

$$
c_t = -\sum_{j=0}^{\infty} \psi_j^* \,\varepsilon_{t-j}
$$

onde $\psi_j^* = \sum_{k=j+1}^{\infty} \psi_k$. O ciclo e estacionario e
representa desvios temporarios do nivel permanente.

### Implementacao via ARMA

Na pratica, os coeficientes $\psi_j$ sao obtidos a partir de um modelo
ARMA$(p, q)$ para $\Delta y_t$:

$$
\phi(B)\,\Delta y_t = \mu + \theta(B)\,\varepsilon_t
$$

Os coeficientes MA($\infty$) sao calculados recursivamente:

$$
\psi_j = \phi_1 \psi_{j-1} + \phi_2 \psi_{j-2} + \cdots + \phi_p \psi_{j-p}, \quad j \geq 1
$$

com $\psi_0 = 1$ (para o caso $q = 0$; com MA, os primeiros $q$ termos incluem
os coeficientes $\theta_j$).

O multiplicador de longo prazo e:

$$
\psi(1) = \frac{\theta(1)}{\phi(1)} = \frac{1 + \theta_1 + \cdots + \theta_q}{1 - \phi_1 - \cdots - \phi_p}
$$

!!! tip "Regra pratica"
    Se $|\psi(1)| > 1$, choques permanentes sao amplificados na tendencia.
    Se $|\psi(1)| < 1$, choques permanentes sao amortecidos. Se $\psi(1) \approx 1$,
    a serie se comporta aproximadamente como um random walk puro.

---

## Quick Example

```python
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset
from chronobox.filters import bn_decomposition

# PIB trimestral dos EUA
gdp = load_dataset("us_gdp")
y = gdp.values

# Beveridge-Nelson decomposition com AR(2) para Delta y
trend, cycle = bn_decomposition(y, p=2, q=0)

# Visualizar
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(y, label="PIB observado", color="gray", alpha=0.6)
axes[0].plot(trend, label="Tendencia BN", color="red", linewidth=2)
axes[0].set_title("Beveridge-Nelson: Componente Permanente")
axes[0].legend()

axes[1].plot(cycle, color="steelblue", linewidth=1.5)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].fill_between(range(len(cycle)), cycle, alpha=0.3)
axes[1].set_title("Beveridge-Nelson: Componente Transitorio (Ciclo)")

plt.tight_layout()
plt.show()
```

---

## Guia Detalhado

### Assinatura da Funcao

```python
bn_decomposition(
    y,              # Serie temporal I(1) (array-like, 1-D)
    p=2,            # Ordem AR para ARMA de Delta y
    q=0,            # Ordem MA para ARMA de Delta y
    n_ma_terms=200  # Termos para truncar MA(infinito)
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `y` | `array-like` | --- | Serie temporal $I(1)$, 1-D |
| `p` | `int` | `2` | Ordem AR do modelo ARMA para $\Delta y_t$ |
| `q` | `int` | `0` | Ordem MA do modelo ARMA para $\Delta y_t$ |
| `n_ma_terms` | `int` | `200` | Numero de termos para truncar a representacao MA($\infty$) |

**Retorno**: tupla `(trend, cycle)` de `np.ndarray` com shape `(T,)`.

### Versao Detalhada

Para acessar os coeficientes e diagnosticos, use `bn_decomposition_detailed`:

```python
from chronobox.filters import bn_decomposition_detailed

result = bn_decomposition_detailed(y, p=2, q=0)

print(f"Multiplicador de longo prazo psi(1): {result.psi_one:.4f}")
print(f"Coeficientes AR: {result.ar_coeffs}")
print(f"Drift: {result.drift:.4f}")
```

| Atributo | Tipo | Descricao |
|---|---|---|
| `trend` | `np.ndarray` | Componente permanente (tendencia) |
| `cycle` | `np.ndarray` | Componente transitorio (ciclo) |
| `psi_one` | `float` | Multiplicador de longo prazo $\psi(1)$ |
| `ar_coeffs` | `np.ndarray` | Coeficientes AR $(\phi_1, \ldots, \phi_p)$ |
| `ma_coeffs` | `np.ndarray` | Coeficientes MA $(\theta_1, \ldots, \theta_q)$ |
| `drift` | `float` | Drift estimado $\mu$ |
| `residuals` | `np.ndarray` | Residuos ARMA $\varepsilon_t$ |

### Escolhendo $p$ e $q$

Use criterios de informacao (AIC, BIC) para selecionar a ordem ARMA de $\Delta y_t$:

```python
import numpy as np
from chronobox import ARIMA

y = ...  # sua serie I(1)
dy = np.diff(y)  # primeiras diferencas

# Testar diferentes ordens
best_aic = np.inf
best_order = (1, 0)

for p in range(1, 6):
    for q in [0]:  # q=0 (AR puro) e mais robusto
        model = ARIMA(order=(p, 0, q), trend='c')
        results = model.fit(dy)
        if results.aic < best_aic:
            best_aic = results.aic
            best_order = (p, q)

print(f"Melhor ordem: AR({best_order[0]}), MA({best_order[1]})")
print(f"AIC: {best_aic:.2f}")

# Usar a melhor ordem na decomposicao
from chronobox.filters import bn_decomposition
trend, cycle = bn_decomposition(y, p=best_order[0], q=best_order[1])
```

### Comparacao com HP Filter

```python
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset
from chronobox.filters import bn_decomposition, hp_filter

gdp = load_dataset("us_gdp")
y = gdp.values

# Decomposicoes
bn_trend, bn_cycle = bn_decomposition(y, p=2)
hp_trend, hp_cycle = hp_filter(y, frequency="quarterly")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(y, color="gray", alpha=0.4, label="PIB")
axes[0].plot(hp_trend, label="Tendencia HP", color="red", linewidth=2)
axes[0].plot(bn_trend, label="Tendencia BN", color="steelblue", linewidth=2)
axes[0].set_title("Tendencias: HP (suave) vs BN (random walk)")
axes[0].legend()

axes[1].plot(hp_cycle, label="Ciclo HP", color="red", alpha=0.7)
axes[1].plot(bn_cycle, label="Ciclo BN", color="steelblue", alpha=0.7)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_title("Componentes Ciclicos")
axes[1].legend()

plt.tight_layout()
plt.show()
```

!!! info "Diferenca conceitual"
    A tendencia HP e **suave** (penaliza mudancas na taxa de crescimento),
    enquanto a tendencia BN e um **random walk** (pode mudar de direcao
    abruptamente). A escolha depende da sua visao sobre a natureza da
    tendencia: deterministica (suave) ou estocastica (random walk).

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox.filters import bn_decomposition

    trend, cycle = bn_decomposition(y, p=2, q=0)
    ```

=== "BeveridgeNelson (R)"

    ```r
    library(BeveridgeNelson)

    bn <- BeveridgeNelson(y, ar.order = 2)
    trend <- bn$trend
    cycle <- bn$cycle
    ```

=== "Implementacao manual (R)"

    ```r
    bn_decomp <- function(y, p = 2) {
      dy <- diff(y)
      fit <- ar(dy, order.max = p, method = "ols")

      # Coeficientes MA(infinito) via recursao
      n_terms <- 200
      psi <- numeric(n_terms)
      psi[1] <- 1
      for (j in 2:n_terms) {
        for (k in 1:min(j-1, p)) {
          psi[j] <- psi[j] + fit$ar[k] * psi[j - k]
        }
      }

      psi_one <- sum(psi)
      mu <- mean(dy)

      # Tendencia: random walk com drift
      eps <- fit$resid
      eps[is.na(eps)] <- 0
      trend <- cumsum(c(y[1], mu + psi_one * eps[-1]))

      # Ciclo
      cycle <- y - trend

      list(trend = trend, cycle = cycle, psi_one = psi_one)
    }
    ```

**Mapeamento de parametros**:

| chronobox | BeveridgeNelson (R) | Descricao |
|---|---|---|
| `p=2` | `ar.order=2` | Ordem AR |
| `q=0` | --- | Ordem MA (R usa apenas AR) |
| `n_ma_terms=200` | --- | Truncamento MA($\infty$) |
| `trend` (retorno) | `bn$trend` | Componente permanente |
| `cycle` (retorno) | `bn$cycle` | Componente transitorio |

---

## Referencias

- Beveridge, S. & Nelson, C. R. (1981). A New Approach to Decomposition of
  Economic Time Series into Permanent and Transitory Components with Particular
  Attention to Measurement of the 'Business Cycle'. *Journal of Monetary Economics*,
  7(2), 151--174.
- Morley, J. C. (2002). A State-Space Approach to Calculating the
  Beveridge-Nelson Decomposition. *Economics Letters*, 75(1), 123--127.
- Kamber, G., Morley, J. & Wong, B. (2018). Intuitive and Reliable Estimates
  of the Output Gap from a Beveridge-Nelson Filter. *Review of Economics and
  Statistics*, 100(3), 550--566.
- Nelson, C. R. & Plosser, C. I. (1982). Trends and Random Walks in
  Macroeconomic Time Series. *Journal of Monetary Economics*, 10(2), 139--162.
