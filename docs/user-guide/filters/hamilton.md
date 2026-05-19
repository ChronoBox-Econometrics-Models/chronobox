---
title: Hamilton Filter
description: Filtro de Hamilton (2018) --- alternativa moderna ao HP filter baseada em regressao.
---

# Hamilton Filter (2018)

!!! info "Quick Reference"
    - **Funcao**: `chronobox.filters.hamilton_filter`
    - **Import**: `from chronobox.filters import hamilton_filter`
    - **R equivalente**: `neverhpfilter::yth_filter(y, h=8, p=4)`
    - **Retorno**: `(trend, cycle)` --- arrays NumPy

---

## Overview

O filtro de Hamilton (2018) e uma alternativa ao [Hodrick-Prescott](hp.md)
proposta no artigo "Why You Should Never Use the Hodrick-Prescott Filter".
Em vez de resolver um problema de otimizacao com penalizacao de suavidade,
Hamilton propoe uma **regressao simples** de $y_{t+h}$ em $y_t, y_{t-1}, \ldots, y_{t-p+1}$.
O residuo dessa regressao e interpretado como o componente ciclico.

### Quando usar

- Como alternativa robusta ao HP filter
- Quando voce precisa evitar ciclos espurios (spurious cycles)
- Quando as estimativas nas extremidades sao importantes (sem end-point problem)
- Quando voce quer um filtro com fundamentacao estatistica clara

### Vantagens sobre o HP Filter

| Problema do HP | Solucao do Hamilton |
|---|---|
| Ciclos espurios em random walks | Regressao nao gera ciclos espurios |
| End-point problem | Estimativas consistentes em toda a amostra |
| Sem fundamentacao estatistica | Baseado em regressao OLS padrao |
| Dependencia de $\lambda$ arbitrario | Parametros $h$ e $p$ com interpretacao clara |

---

## Formulacao Matematica

### Regressao de Hamilton

O filtro e baseado na regressao OLS:

$$
y_{t+h} = \beta_0 + \beta_1 y_t + \beta_2 y_{t-1} + \cdots + \beta_p y_{t-p+1} + v_{t+h}
$$

onde:

- $h$ e o **horizonte de previsao** (default: $h = 8$ para dados trimestrais = 2 anos)
- $p$ e o **numero de lags** na regressao (default: $p = 4$)
- $v_{t+h}$ e o residuo

### Componentes

O **componente de tendencia** (fitted value) e:

$$
\hat{\tau}_{t+h} = \hat{\beta}_0 + \hat{\beta}_1 y_t + \hat{\beta}_2 y_{t-1} + \cdots + \hat{\beta}_p y_{t-p+1}
$$

O **componente ciclico** (residuo) e:

$$
\hat{c}_{t+h} = y_{t+h} - \hat{\tau}_{t+h} = v_{t+h}
$$

### Intuicao

A ideia e simples: se voce pode prever $y_{t+h}$ a partir dos valores passados
$y_t, \ldots, y_{t-p+1}$, entao a parte previsivel e a "tendencia" e a parte
imprevisivel (residuo) e o "ciclo". O horizonte $h$ define o que conta como
"longo prazo" --- com $h = 8$ trimestres (2 anos), o filtro separa movimentos
previsiveis em 2 anos (tendencia) de flutuacoes de curto prazo (ciclo).

### Escolha de $h$ e $p$

Hamilton (2018) recomenda:

| Frequencia | $h$ | $p$ | Justificativa |
|---|---|---|---|
| Trimestral | 8 | 4 | 2 anos a frente, 1 ano de lags |
| Mensal | 24 | 12 | 2 anos a frente, 1 ano de lags |
| Anual | 2 | 2 | 2 anos a frente, 2 anos de lags |

!!! tip "Regra pratica"
    Use $h$ = numero de observacoes em 2 anos e $p$ = numero de observacoes
    em 1 ano. Essa configuracao captura bem ciclos de negocios tipicos.

### Observacoes Perdidas

As primeiras $h + p - 1$ observacoes recebem `NaN`, pois nao e possivel
construir a regressao para elas. Com os defaults ($h=8$, $p=4$), perdem-se
11 observacoes no inicio.

---

## Quick Example

```python
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset
from chronobox.filters import hamilton_filter, hp_filter

# PIB trimestral dos EUA
gdp = load_dataset("us_gdp")
y = gdp.values

# Hamilton filter (h=8, p=4)
ham_trend, ham_cycle = hamilton_filter(y, h=8, p=4)

# HP filter para comparacao
hp_trend, hp_cycle = hp_filter(y, frequency="quarterly")

# Comparar ciclos
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(hp_cycle, label="HP ($\\lambda = 1600$)", color="red", alpha=0.8)
axes[0].plot(ham_cycle, label="Hamilton ($h=8, p=4$)", color="steelblue", alpha=0.8)
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_title("Componente Ciclico --- Hamilton vs HP")
axes[0].legend()

axes[1].plot(y, color="gray", alpha=0.5, label="PIB observado")
axes[1].plot(hp_trend, label="Tendencia HP", color="red", linewidth=2)
axes[1].plot(ham_trend, label="Tendencia Hamilton", color="steelblue", linewidth=2)
axes[1].set_title("Componentes de Tendencia")
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## Guia Detalhado

### Assinatura da Funcao

```python
hamilton_filter(
    y,        # Serie temporal (array-like, 1-D)
    h=8,      # Horizonte de previsao
    p=4       # Numero de lags na regressao
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `y` | `array-like` | --- | Serie temporal 1-D |
| `h` | `int` | `8` | Horizonte de previsao. $h = 8$ para dados trimestrais (2 anos) |
| `p` | `int` | `4` | Numero de lags na regressao. $p = 4$ para dados trimestrais (1 ano) |

**Retorno**: tupla `(trend, cycle)` de `np.ndarray` com shape `(T,)`.
As primeiras $h + p - 1$ observacoes sao `NaN`.

### Versao Detalhada

Para acessar os coeficientes da regressao, use `hamilton_filter_detailed`:

```python
from chronobox.filters import hamilton_filter_detailed

result = hamilton_filter_detailed(y, h=8, p=4)

print(f"Coeficientes: {result.coefficients}")
print(f"Horizonte: {result.h}")
print(f"Lags: {result.p}")
print(f"Tendencia: {result.trend}")
print(f"Ciclo: {result.cycle}")
```

| Atributo | Tipo | Descricao |
|---|---|---|
| `trend` | `np.ndarray` | Componente de tendencia (fitted values) |
| `cycle` | `np.ndarray` | Componente ciclico (residuos) |
| `coefficients` | `np.ndarray` | Coeficientes OLS $[\beta_0, \beta_1, \ldots, \beta_p]$ |
| `h` | `int` | Horizonte de previsao utilizado |
| `p` | `int` | Numero de lags utilizado |

### Configuracoes por Frequencia

=== "Dados trimestrais"

    ```python
    # h=8 (2 anos), p=4 (1 ano de lags)
    trend, cycle = hamilton_filter(y, h=8, p=4)
    ```

=== "Dados mensais"

    ```python
    # h=24 (2 anos), p=12 (1 ano de lags)
    trend, cycle = hamilton_filter(y, h=24, p=12)
    ```

=== "Dados anuais"

    ```python
    # h=2 (2 anos), p=2 (2 anos de lags)
    trend, cycle = hamilton_filter(y, h=2, p=2)
    ```

### Comparacao com HP: End-Point Problem

O HP filter e notoriamente instavel nas extremidades da serie. Novas observacoes
alteram significativamente as estimativas do ciclo para os periodos recentes.
O Hamilton filter nao sofre desse problema:

```python
import numpy as np
import matplotlib.pyplot as plt
from chronobox.filters import hamilton_filter, hp_filter

# Simular end-point problem: filtrar com e sem as ultimas 4 obs.
from chronobox.datasets import load_dataset

gdp = load_dataset("us_gdp")
y = gdp.values

# Amostra completa vs amostra truncada
hp_t_full, hp_c_full = hp_filter(y, frequency="quarterly")
hp_t_trunc, hp_c_trunc = hp_filter(y[:-4], frequency="quarterly")

ham_t_full, ham_c_full = hamilton_filter(y, h=8, p=4)
ham_t_trunc, ham_c_trunc = hamilton_filter(y[:-4], h=8, p=4)

# Comparar os ultimos 20 periodos da amostra truncada
n = len(y) - 4

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(n-20, n), hp_c_full[n-20:n], label="Amostra completa", linewidth=2)
axes[0].plot(range(n-20, n), hp_c_trunc[-20:], label="Sem 4 ultimas obs.",
             linewidth=2, linestyle="--")
axes[0].set_title("HP Filter --- End-Point Problem")
axes[0].legend()

axes[1].plot(range(n-20, n), ham_c_full[n-20:n], label="Amostra completa", linewidth=2)
axes[1].plot(range(n-20, n), ham_c_trunc[-20:], label="Sem 4 ultimas obs.",
             linewidth=2, linestyle="--")
axes[1].set_title("Hamilton Filter --- Sem End-Point Problem")
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox.filters import hamilton_filter

    trend, cycle = hamilton_filter(y, h=8, p=4)
    ```

=== "neverhpfilter (R)"

    ```r
    library(neverhpfilter)

    ham <- yth_filter(y, h = 8, p = 4)
    trend <- ham$trend
    cycle <- ham$cycle
    ```

=== "Implementacao manual (R)"

    ```r
    # O Hamilton filter e simplesmente uma regressao OLS
    hamilton_filter <- function(y, h = 8, p = 4) {
      T <- length(y)
      n <- T - h - p + 1

      # Construir matriz X (lags) e vetor Y (h-passos a frente)
      Y <- y[(h + p):T]
      X <- matrix(1, nrow = n, ncol = p + 1)
      for (j in 1:p) {
        X[, j + 1] <- y[(p - j + 1):(T - h - j + 1)]
      }

      # OLS
      beta <- solve(t(X) %*% X) %*% t(X) %*% Y
      trend <- X %*% beta
      cycle <- Y - trend

      list(trend = trend, cycle = cycle, coefficients = beta)
    }
    ```

**Mapeamento de parametros**:

| chronobox | neverhpfilter (R) | Descricao |
|---|---|---|
| `h=8` | `h=8` | Horizonte de previsao |
| `p=4` | `p=4` | Numero de lags |
| `trend` (retorno) | `ham$trend` | Componente de tendencia |
| `cycle` (retorno) | `ham$cycle` | Componente ciclico |

---

## Referencias

- Hamilton, J. D. (2018). Why You Should Never Use the Hodrick-Prescott Filter.
  *Review of Economics and Statistics*, 100(5), 831--843.
- Hodrick, R. J. & Prescott, E. C. (1997). Postwar U.S. Business Cycles:
  An Empirical Investigation. *Journal of Money, Credit and Banking*, 29(1), 1--16.
- Quast, J. & Wolters, M. H. (2022). Reliable Real-Time Output Gap Estimates
  Based on a Modified Hamilton Filter. *Journal of Business & Economic Statistics*,
  40(1), 152--168.
