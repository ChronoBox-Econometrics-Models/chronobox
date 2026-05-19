---
title: Christiano-Fitzgerald Filter
description: Filtro band-pass assimetrico otimo de Christiano-Fitzgerald para ciclos de negocios.
---

# Christiano-Fitzgerald Band-Pass Filter

!!! info "Quick Reference"
    - **Funcao**: `chronobox.filters.cf_filter`
    - **Import**: `from chronobox.filters import cf_filter`
    - **R equivalente**: `mFilter::cffilter(y, pl=6, pu=32, drift=FALSE)`
    - **Retorno**: `cycle` --- array NumPy (mesmo tamanho da serie original)

---

## Overview

O filtro de Christiano e Fitzgerald (2003) e um filtro band-pass assimetrico
otimo que resolve a principal limitacao do [Baxter-King](bk.md): a perda de
observacoes nas extremidades da serie. O CF filter usa **todas as observacoes**,
aplicando pesos assimetricos nas bordas para extrair o componente ciclico.

### Quando usar

- Extrair ciclos de negocios sem perder observacoes
- Series curtas onde a perda de $2K$ observacoes do BK e inaceitavel
- Quando as estimativas nas extremidades sao importantes (analise conjuntural)
- Analise comparativa com o Baxter-King

### CF vs BK

| Caracteristica | Baxter-King | Christiano-Fitzgerald |
|---|---|---|
| Observacoes perdidas | $2K$ | 0 |
| Pesos do filtro | Simetricos (fixos) | Assimetricos (variam por obs.) |
| Qualidade no centro | Excelente | Excelente |
| Qualidade nas bordas | N/A (descartadas) | Boa (assimetrico otimo) |
| Simplicidade | Mais simples | Mais complexo |

---

## Formulacao Matematica

### Filtro Assimetrico Otimo

O CF filter busca a melhor aproximacao linear do filtro band-pass ideal
$c_t^* = \sum_{j=-\infty}^{\infty} a_j y_{t-j}$, mas restrita as observacoes
disponiveis:

$$
\hat{c}_t = \sum_{j=-(t-1)}^{T-t} \hat{a}_{t,j}\, y_{t-j}
$$

Os pesos $\hat{a}_{t,j}$ sao escolhidos para minimizar o erro quadratico medio:

$$
\min_{\hat{a}_{t,j}} \; E\left[(c_t^* - \hat{c}_t)^2\right]
$$

sujeito a $\sum_j \hat{a}_{t,j} = 0$ (o filtro remove a media).

### Pesos Ideais

Os pesos do filtro ideal band-pass sao:

$$
a_j = \frac{\sin(j\omega_H) - \sin(j\omega_L)}{\pi j}, \quad j \neq 0
$$

$$
a_0 = \frac{\omega_H - \omega_L}{\pi}
$$

onde $\omega_L = 2\pi / p_H$ e $\omega_H = 2\pi / p_L$.

### Hipotese sobre o DGP

Christiano e Fitzgerald derivam os pesos otimos assumindo que os dados seguem
um random walk: $y_t = y_{t-1} + \varepsilon_t$. Sob essa hipotese, os pesos
assimetricos nas bordas sao obtidos analiticamente e minimizam o MSE de
aproximacao ao filtro ideal.

Para observacoes interiores (longe das bordas), os pesos CF convergem para
os pesos BK simetricos. A diferenca esta apenas nas primeiras e ultimas
observacoes da serie.

### Opcao de Drift

Se `drift=True`, a funcao remove a tendencia linear antes de filtrar:

$$
\tilde{y}_t = y_t - \hat{\alpha} - \hat{\beta}\, t
$$

Isso e util quando a serie tem tendencia deterministica forte que poderia
distorcer o componente ciclico extraido.

---

## Quick Example

```python
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset
from chronobox.filters import cf_filter, bk_filter

# PIB trimestral dos EUA
gdp = load_dataset("us_gdp")
y = gdp.values
T = len(y)

# CF filter --- sem perda de observacoes
cf_cycle = cf_filter(y, low=6, high=32)

# BK filter para comparacao
K = 12
bk_cycle = bk_filter(y, low=6, high=32, trunc=K)

# Visualizar
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(cf_cycle, label="Christiano-Fitzgerald", color="steelblue", linewidth=1.5)
ax.plot(range(K, T - K), bk_cycle, label="Baxter-King", color="red",
        linewidth=1.5, linestyle="--")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Comparacao CF vs BK (6--32 trimestres)")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Guia Detalhado

### Assinatura da Funcao

```python
cf_filter(
    y,              # Serie temporal (array-like, 1-D)
    low=6,          # Periodo minimo do ciclo
    high=32,        # Periodo maximo do ciclo
    drift=False     # Remover drift linear
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `y` | `array-like` | --- | Serie temporal 1-D com pelo menos 4 observacoes |
| `low` | `int` | `6` | Periodo minimo do ciclo (em observacoes) |
| `high` | `int` | `32` | Periodo maximo do ciclo (em observacoes) |
| `drift` | `bool` | `False` | Se `True`, remove tendencia linear antes de filtrar |

**Retorno**: `np.ndarray` com shape `(T,)` --- o componente ciclico, mesmo tamanho da serie original.

### Com e Sem Drift

=== "Sem drift (default)"

    ```python
    # Adequado para series sem tendencia linear forte
    cycle = cf_filter(y, low=6, high=32, drift=False)
    ```

=== "Com drift"

    ```python
    # Remove tendencia linear antes de filtrar
    # Adequado para series com tendencia deterministica
    cycle = cf_filter(y, low=6, high=32, drift=True)
    ```

### Parametros por Frequencia

=== "Dados trimestrais"

    ```python
    # Ciclos de 1.5 a 8 anos
    cycle = cf_filter(y, low=6, high=32)
    ```

=== "Dados mensais"

    ```python
    # Ciclos de 1.5 a 8 anos
    cycle = cf_filter(y, low=18, high=96)
    ```

=== "Dados anuais"

    ```python
    # Ciclos de 2 a 8 anos
    cycle = cf_filter(y, low=2, high=8)
    ```

### Comparacao Detalhada com BK

```python
import numpy as np
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset
from chronobox.filters import cf_filter, bk_filter

gdp = load_dataset("us_gdp")
y = gdp.values
T = len(y)
K = 12

cf_cycle = cf_filter(y, low=6, high=32)
bk_cycle = bk_filter(y, low=6, high=32, trunc=K)

# Correlacao na regiao de sobreposicao
cf_overlap = cf_cycle[K:T-K]
corr = np.corrcoef(cf_overlap, bk_cycle)[0, 1]
print(f"Correlacao CF vs BK (regiao de sobreposicao): {corr:.4f}")

# Visualizar diferencas nas extremidades
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Inicio da serie
axes[0].plot(cf_cycle[:30], label="CF", color="steelblue")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].axvline(K, color="gray", linestyle="--", alpha=0.5, label=f"Inicio BK (obs {K})")
axes[0].set_title("Inicio da serie")
axes[0].legend()

# Final da serie
axes[1].plot(range(T-30, T), cf_cycle[-30:], label="CF", color="steelblue")
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].axvline(T-K, color="gray", linestyle="--", alpha=0.5, label=f"Fim BK (obs {T-K})")
axes[1].set_title("Final da serie")
axes[1].legend()

plt.suptitle("CF preserva observacoes nas extremidades")
plt.tight_layout()
plt.show()
```

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox.filters import cf_filter

    cycle = cf_filter(y, low=6, high=32, drift=False)
    ```

=== "mFilter (R)"

    ```r
    library(mFilter)

    cf <- cffilter(y, pl = 6, pu = 32, drift = FALSE)
    cycle <- cf$cycle
    ```

**Mapeamento de parametros**:

| chronobox | mFilter (R) | Descricao |
|---|---|---|
| `low=6` | `pl=6` | Periodo minimo |
| `high=32` | `pu=32` | Periodo maximo |
| `drift=False` | `drift=FALSE` | Remocao de tendencia linear |
| retorno `cycle` | `cf$cycle` | Componente ciclico |

---

## Referencias

- Christiano, L. J. & Fitzgerald, T. J. (2003). The Band Pass Filter.
  *International Economic Review*, 44(2), 435--465.
- Baxter, M. & King, R. G. (1999). Measuring Business Cycles: Approximate
  Band-Pass Filters for Economic Time Series. *Review of Economics and Statistics*,
  81(4), 575--593.
- Murray, C. J. (2003). Cyclical Properties of Baxter-King Filtered Time Series.
  *Review of Economics and Statistics*, 85(2), 472--476.
