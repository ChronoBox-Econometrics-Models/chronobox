---
title: Baxter-King Filter
description: Filtro band-pass de Baxter-King para extracao de ciclos de negocios.
---

# Baxter-King Band-Pass Filter

!!! info "Quick Reference"
    - **Funcao**: `chronobox.filters.bk_filter`
    - **Import**: `from chronobox.filters import bk_filter`
    - **R equivalente**: `mFilter::bkfilter(y, pl=6, pu=32, nfix=12)`
    - **Retorno**: `cycle` --- array NumPy (T - 2K observacoes)

---

## Overview

O filtro de Baxter e King (1999) e uma aproximacao finita do filtro band-pass ideal.
Ele extrai o componente ciclico da serie preservando apenas flutuacoes dentro de
uma banda de frequencias especificada --- tipicamente ciclos com periodicidade
entre 6 e 32 trimestres (1.5 a 8 anos), conforme a definicao classica de ciclos
de negocios de Burns e Mitchell (1946).

### Quando usar

- Extrair ciclos de negocios de series macroeconomicas
- Quando voce quer isolar uma banda de frequencia especifica
- Analise que exige filtro simetrico (sem defasagem de fase)
- Quando a perda de $2K$ observacoes nas extremidades e aceitavel

!!! warning "Perda de observacoes"
    O filtro BK perde $K$ observacoes em cada extremo da serie. Com o default
    `trunc=12` e dados trimestrais, voce perde 24 observacoes (6 anos). Para
    series curtas, considere o [Christiano-Fitzgerald](cf.md) que nao perde
    observacoes.

---

## Formulacao Matematica

### Filtro Band-Pass Ideal

O filtro band-pass ideal no dominio da frequencia tem funcao de transferencia:

$$
G(\omega) =
\begin{cases}
1 & \text{se } \omega_L \leq |\omega| \leq \omega_H \\
0 & \text{caso contrario}
\end{cases}
$$

onde $\omega_L = 2\pi / p_H$ e $\omega_H = 2\pi / p_L$ sao as frequencias
de corte correspondentes aos periodos $p_L$ (minimo) e $p_H$ (maximo).

No dominio do tempo, o filtro ideal requer infinitos pesos:

$$
c_t = \sum_{j=-\infty}^{\infty} a_j\, y_{t-j}
$$

onde os pesos ideais sao:

$$
a_j = \frac{\sin(j\omega_H) - \sin(j\omega_L)}{\pi j}, \quad j \neq 0
$$

$$
a_0 = \frac{\omega_H - \omega_L}{\pi}
$$

### Aproximacao de Baxter-King

Baxter e King (1999) truncam os pesos em $K$ leads/lags e ajustam para que
somem zero (garantindo que a media da serie filtrada seja zero):

$$
\hat{c}_t = \sum_{j=-K}^{K} \tilde{a}_j\, y_{t-j}
$$

onde os pesos ajustados sao:

$$
\tilde{a}_j = a_j - \frac{1}{2K+1} \sum_{k=-K}^{K} a_k
$$

A restricao $\sum_{j=-K}^{K} \tilde{a}_j = 0$ garante que o filtro elimine
tendencias lineares (o filtro passa apenas ciclos, nao niveis ou tendencias).

### Propriedades Espectrais

O filtro BK aproxima o filtro ideal com qualidade crescente em $K$.
A funcao gain do filtro truncado e:

$$
|\hat{G}(\omega)| = \left| \tilde{a}_0 + 2\sum_{j=1}^{K} \tilde{a}_j \cos(j\omega) \right|
$$

Quanto maior $K$, mais proxima a funcao gain fica do filtro ideal (retangular),
mas mais observacoes sao perdidas.

---

## Quick Example

```python
import numpy as np
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset
from chronobox.filters import bk_filter

# PIB trimestral dos EUA
gdp = load_dataset("us_gdp")
y = gdp.values
T = len(y)

# Extrair ciclo de negocios (6-32 trimestres)
K = 12
cycle = bk_filter(y, low=6, high=32, trunc=K)

# Visualizar (cycle tem T - 2K observacoes)
fig, ax = plt.subplots(figsize=(12, 5))
x_cycle = np.arange(K, T - K)
ax.plot(x_cycle, cycle, color="steelblue", linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.5)
ax.fill_between(x_cycle, cycle, alpha=0.3)
ax.set_title("Ciclo de Negocios --- Filtro Baxter-King (6--32 trimestres)")
ax.set_xlabel("Observacao")
plt.tight_layout()
plt.show()
```

---

## Guia Detalhado

### Assinatura da Funcao

```python
bk_filter(
    y,              # Serie temporal (array-like, 1-D)
    low=6,          # Periodo minimo do ciclo
    high=32,        # Periodo maximo do ciclo
    trunc=12        # Ordem de truncamento K
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `y` | `array-like` | --- | Serie temporal 1-D |
| `low` | `int` | `6` | Periodo minimo do ciclo (em observacoes). Default: 6 trimestres = 1.5 anos |
| `high` | `int` | `32` | Periodo maximo do ciclo (em observacoes). Default: 32 trimestres = 8 anos |
| `trunc` | `int` | `12` | Ordem de truncamento $K$. Perde $K$ observacoes em cada extremo |

**Retorno**: `np.ndarray` com shape `(T - 2*trunc,)` --- o componente ciclico.

### Escolhendo os Parametros

Os valores default (`low=6, high=32, trunc=12`) sao adequados para **dados trimestrais**
e capturam ciclos de negocios classicos (1.5 a 8 anos).

Para outras frequencias:

=== "Dados trimestrais"

    ```python
    # Ciclos de negocios: 1.5 a 8 anos
    cycle = bk_filter(y, low=6, high=32, trunc=12)
    ```

=== "Dados mensais"

    ```python
    # Ciclos de negocios: 1.5 a 8 anos
    cycle = bk_filter(y, low=18, high=96, trunc=36)
    ```

=== "Dados anuais"

    ```python
    # Ciclos de negocios: 2 a 8 anos
    cycle = bk_filter(y, low=2, high=8, trunc=3)
    ```

!!! tip "Regra pratica"
    Baxter e King recomendam $K \geq 12$ para dados trimestrais. Valores
    menores de $K$ produzem uma aproximacao mais pobre do filtro ideal,
    mas preservam mais observacoes. O trade-off depende do tamanho da amostra.

### Efeito do Truncamento $K$

```python
import numpy as np
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset
from chronobox.filters import bk_filter

gdp = load_dataset("us_gdp")
y = gdp.values
T = len(y)

fig, ax = plt.subplots(figsize=(12, 5))

for K, cor in [(6, "green"), (12, "steelblue"), (24, "red")]:
    cycle = bk_filter(y, low=6, high=32, trunc=K)
    x = np.arange(K, T - K)
    ax.plot(x, cycle, label=f"K = {K}", color=cor, alpha=0.8)

ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Efeito do truncamento $K$ no filtro BK")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox.filters import bk_filter

    cycle = bk_filter(y, low=6, high=32, trunc=12)
    ```

=== "mFilter (R)"

    ```r
    library(mFilter)

    bk <- bkfilter(y, pl = 6, pu = 32, nfix = 12)
    cycle <- bk$cycle
    ```

**Mapeamento de parametros**:

| chronobox | mFilter (R) | Descricao |
|---|---|---|
| `low=6` | `pl=6` | Periodo minimo |
| `high=32` | `pu=32` | Periodo maximo |
| `trunc=12` | `nfix=12` | Ordem de truncamento $K$ |
| retorno `cycle` | `bk$cycle` | Componente ciclico |

---

## Referencias

- Baxter, M. & King, R. G. (1999). Measuring Business Cycles: Approximate
  Band-Pass Filters for Economic Time Series. *Review of Economics and Statistics*,
  81(4), 575--593.
- Burns, A. F. & Mitchell, W. C. (1946). *Measuring Business Cycles*. NBER.
- Stock, J. H. & Watson, M. W. (1999). Business Cycle Fluctuations in US
  Macroeconomic Time Series. In *Handbook of Macroeconomics*, Vol. 1A.
  Elsevier. 3--64.
