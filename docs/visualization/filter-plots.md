---
title: Filter Plots
description: Graficos de filtros economicos --- tendencia, ciclo, comparacao HP vs Hamilton vs BK e shaded recession areas.
---

# Filter Plots

!!! info "Quick Reference"
    - **Funcao**: `plot_filter()`
    - **Import**: `from chronobox.visualization import plot_filter`
    - **Input**: Objeto de resultado de filtro ou arrays NumPy
    - **Output**: `matplotlib.figure.Figure` com tendencia e ciclo

---

## Overview

Os graficos de filtros economicos visualizam a **decomposicao** de uma serie
temporal em componentes de **tendencia** e **ciclo**. Sao fundamentais para
analise de ciclos de negocios e gap de produto.

O modulo oferece:

- **`plot_filter()`** --- tendencia + ciclo de um filtro individual
- **Comparacao de filtros** --- HP, Hamilton e BK lado a lado
- **Shaded recession areas** --- periodos de recessao sombreados
- **Componentes sobrepostos** --- serie original, tendencia e ciclo juntos

### Quando usar

- Visualizar output gap (hiato do produto)
- Comparar resultados de diferentes filtros
- Analisar ciclos de negocios com recessoes marcadas
- Apresentar decomposicao tendencia-ciclo em relatorios

---

## Quick Start

```python
from chronobox.filters import hp_filter
from chronobox.visualization import plot_filter, set_theme

set_theme("professional")

# Aplicar filtro HP
result = hp_filter(gdp, lamb=1600)

# Plotar tendencia e ciclo
fig = plot_filter(result)
fig.savefig("hp_filter.png", dpi=150, bbox_inches="tight")
```

---

## Tendencia + Ciclo

O grafico padrao mostra dois paineis verticais: tendencia sobreposta a
serie original (painel superior) e componente ciclico (painel inferior).

```python
fig = plot_filter(
    result,
    show_trend=True,
    show_cycle=True,
    title="Filtro HP - PIB Real",
)
```

### Painel superior (Tendencia)

- **Linha cinza**: serie original
- **Linha azul**: tendencia estimada

### Painel inferior (Ciclo)

- **Linha azul**: componente ciclico ($y_t - \tau_t$)
- **Linha tracejada**: zero (referencia)
- **Area sombreada**: periodos de ciclo negativo (abaixo de zero)

---

## Apenas Tendencia

```python
fig = plot_filter(
    result,
    show_trend=True,
    show_cycle=False,
    title="Tendencia do PIB Real (HP Filter)",
)
```

## Apenas Ciclo

```python
fig = plot_filter(
    result,
    show_trend=False,
    show_cycle=True,
    title="Componente Ciclico do PIB Real",
)
```

---

## Comparacao de Filtros

Uma das visualizacoes mais uteis e a comparacao lado a lado de diferentes
filtros aplicados a mesma serie:

```python
from chronobox.filters import hp_filter, hamilton_filter, bk_filter
from chronobox.visualization import plot_filter

# Aplicar filtros
hp_result = hp_filter(gdp, lamb=1600)
hamilton_result = hamilton_filter(gdp, h=8, p=4)
bk_result = bk_filter(gdp, low=6, high=32, K=12)

# Comparar ciclos
fig = plot_filter(
    [hp_result, hamilton_result, bk_result],
    labels=["HP (λ=1600)", "Hamilton (h=8)", "BK (6-32)"],
    compare="cycle",
    title="Comparacao de Filtros - Ciclo do PIB",
    figsize=(14, 6),
)
```

### Comparar tendencias

```python
fig = plot_filter(
    [hp_result, hamilton_result, bk_result],
    labels=["HP", "Hamilton", "BK"],
    compare="trend",
    title="Comparacao de Tendencias",
)
```

### Grid individual

```python
# Cada filtro em seu proprio painel
fig = plot_filter(
    [hp_result, hamilton_result, bk_result],
    labels=["HP", "Hamilton", "BK"],
    compare="grid",
    figsize=(14, 10),
    title="Comparacao de Filtros - Grid",
)
```

!!! note "Diferencas entre filtros"
    - **HP**: suavidade controlada por $\lambda$; sofre end-point bias
    - **Hamilton**: robusto a end-point bias; requer mais dados
    - **BK**: filtro de banda ideal; perde observacoes nas pontas

---

## Shaded Recession Areas

Adicione periodos de recessao como areas sombreadas para contextualizar
o componente ciclico:

```python
# Datas de recessao (lista de tuplas inicio/fim)
recessions = [
    ("2001-03-01", "2001-11-01"),
    ("2008-12-01", "2009-06-01"),
    ("2014-04-01", "2016-12-01"),
    ("2020-03-01", "2020-06-01"),
]

fig = plot_filter(
    result,
    recession_dates=recessions,
    recession_color="#d62728",
    recession_alpha=0.15,
    title="Ciclo do PIB com Recessoes",
)
```

### Recessoes CODACE (Brasil)

```python
from chronobox.datasets import load_recessions_brazil

recessions_br = load_recessions_brazil()

fig = plot_filter(
    result,
    recession_dates=recessions_br,
    title="Output Gap com Recessoes CODACE",
)
```

### Recessoes NBER (EUA)

```python
from chronobox.datasets import load_recessions_usa

recessions_us = load_recessions_usa()

fig = plot_filter(
    result,
    recession_dates=recessions_us,
    title="US Output Gap with NBER Recessions",
)
```

---

## Componentes Sobrepostos

Para visualizar todos os componentes em uma unica figura:

```python
fig = plot_filter(
    result,
    overlay=True,
    title="PIB Real - Componentes Sobrepostos",
)
```

Isso gera um unico eixo com:

- **Serie original**: linha cinza
- **Tendencia**: linha azul grossa
- **Ciclo**: linha vermelha (escala no eixo secundario)

---

## Exemplos com GDP

### Filtro HP com diferentes lambdas

```python
from chronobox.filters import hp_filter
from chronobox.visualization import plot_filter

# lambda=100 (anual), lambda=1600 (trimestral), lambda=129600 (mensal)
hp_100 = hp_filter(gdp_annual, lamb=100)
hp_1600 = hp_filter(gdp_quarterly, lamb=1600)
hp_129600 = hp_filter(gdp_monthly, lamb=129600)

# Comparar efeito do lambda
results = [
    hp_filter(gdp, lamb=100),
    hp_filter(gdp, lamb=1600),
    hp_filter(gdp, lamb=14400),
]

fig = plot_filter(
    results,
    labels=["λ=100", "λ=1,600", "λ=14,400"],
    compare="cycle",
    title="Efeito do Lambda no Filtro HP",
)
```

### Hamilton Filter

```python
from chronobox.filters import hamilton_filter
from chronobox.visualization import plot_filter

result = hamilton_filter(gdp, h=8, p=4)

fig = plot_filter(
    result,
    show_trend=True,
    show_cycle=True,
    recession_dates=recessions,
    title="Hamilton Filter - PIB Real (h=8, p=4)",
)
```

### Baxter-King

```python
from chronobox.filters import bk_filter
from chronobox.visualization import plot_filter

# Ciclos de 6 a 32 trimestres (padrao para ciclos de negocios)
result = bk_filter(gdp, low=6, high=32, K=12)

fig = plot_filter(
    result,
    show_trend=True,
    show_cycle=True,
    title="Baxter-King Band-Pass Filter",
)
```

!!! warning "Perda de observacoes no BK"
    O filtro Baxter-King perde $K$ observacoes em cada ponta da serie.
    Com `K=12`, as primeiras e ultimas 12 observacoes nao terao valores
    filtrados. O grafico automaticamente ajusta o eixo X.

---

## Uso com Arrays NumPy

```python
import numpy as np
import pandas as pd
from chronobox.visualization import plot_filter

# Serie e componentes
index = pd.date_range("2000-01-01", periods=100, freq="QS")
original = np.cumsum(np.random.randn(100)) + 100
trend = np.linspace(95, 105, 100)
cycle = original - trend

fig = plot_filter(
    trend=trend,
    cycle=cycle,
    original=original,
    index=index,
    title="Filtro a Partir de Arrays",
)
```

---

## Parametros de `plot_filter`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `filter_result` | objeto/list | `None` | Resultado(s) de filtro |
| `trend` | ndarray | `None` | Componente de tendencia (array direto) |
| `cycle` | ndarray | `None` | Componente ciclico (array direto) |
| `original` | ndarray | `None` | Serie original (array direto) |
| `index` | DatetimeIndex | `None` | Indice temporal |
| `show_trend` | bool | `True` | Exibir painel de tendencia |
| `show_cycle` | bool | `True` | Exibir painel de ciclo |
| `overlay` | bool | `False` | Sobrepor componentes em eixo unico |
| `compare` | str | `None` | Modo de comparacao: `"cycle"`, `"trend"`, `"grid"` |
| `labels` | list[str] | `None` | Rotulos para comparacao de filtros |
| `recession_dates` | list[tuple] | `None` | Periodos de recessao `[(inicio, fim), ...]` |
| `recession_color` | str | `"gray"` | Cor das areas de recessao |
| `recession_alpha` | float | `0.2` | Transparencia das areas de recessao |
| `title` | str | `None` | Titulo geral |
| `figsize` | tuple | auto | Tamanho da figura |

---

## Dicas de Visualizacao

!!! tip "Comparacao e essencial"
    Nunca confie em um unico filtro. Sempre compare HP, Hamilton e BK para
    verificar se as conclusoes sobre o ciclo sao robustas.

!!! tip "Recessoes dao contexto"
    Marcar recessoes no grafico de ciclo permite validar se o filtro
    captura corretamente os periodos de contracao economica.

!!! tip "Escala do ciclo"
    O componente ciclico geralmente e expresso em **desvio percentual**
    da tendencia. Para converter: `cycle_pct = (cycle / trend) * 100`.
