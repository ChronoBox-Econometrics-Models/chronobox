---
title: Spillover Plots
description: Graficos de spillover --- heatmap da tabela de conectividade, network graph, spillover dinamico e direcional.
---

# Spillover Plots

!!! info "Quick Reference"
    - **Funcoes**: `plot_spillover_table()`, `plot_spillover_network()`, `plot_spillover_dynamic()`
    - **Import**: `from chronobox.visualization import plot_spillover_table, plot_spillover_network, plot_spillover_dynamic`
    - **Input**: Tabela de spillover ou resultado de spillover dinamico
    - **Output**: `matplotlib.figure.Figure`

---

## Overview

Os graficos de spillover visualizam a **conectividade** e a **transmissao de
choques** entre variaveis em um sistema. Baseados no framework de
Diebold-Yilmaz (2009, 2012), eles permitem analisar:

- **Quem transmite e quem recebe** choques no sistema
- **Intensidade das conexoes** entre variaveis
- **Evolucao temporal** da conectividade
- **Estrutura de rede** do sistema

O modulo oferece tres tipos principais de visualizacao:

| Funcao | Tipo | Uso |
|---|---|---|
| `plot_spillover_table()` | Heatmap | Magnitudes de transmissao pairwise |
| `plot_spillover_network()` | Network graph | Estrutura de conexoes como grafo |
| `plot_spillover_dynamic()` | Time series | Evolucao do Total Spillover Index |

---

## Quick Start

```python
from chronobox import VAR
from chronobox.spillover import SpilloverTable
from chronobox.visualization import (
    plot_spillover_table,
    plot_spillover_network,
    set_theme,
)

set_theme("professional")

# Estimar VAR e calcular spillover
model = VAR(lags=4)
results = model.fit(data)
table = SpilloverTable(results, horizon=10)

# Heatmap
fig = plot_spillover_table(table)
fig.savefig("spillover_heatmap.png", dpi=150, bbox_inches="tight")
```

---

## Heatmap da Tabela de Spillover

O heatmap exibe a **tabela de decomposicao da variancia generalizada** como
uma matriz de cores, onde cada celula $(i,j)$ mostra a contribuicao da
variavel $j$ para a variancia do erro de previsao da variavel $i$:

$$
s_{ij}(H) = \frac{\tilde{d}_{ij}^H}{\sum_{j=1}^{K} \tilde{d}_{ij}^H} \times 100
$$

```python
fig = plot_spillover_table(
    table,
    annot=True,
    fmt=".1f",
    cmap="RdYlBu_r",
    title="Spillover Table - Mercados Financeiros",
    figsize=(10, 8),
)
```

### Elementos do heatmap

- **Diagonal**: variancia propria (own variance share)
- **Fora da diagonal**: spillover pairwise
- **Linha "TO"**: total transmitido por cada variavel
- **Coluna "FROM"**: total recebido por cada variavel
- **Canto inferior direito**: Total Spillover Index (TSI)

### Opcoes de colormap

```python
# Sequencial (magnitudes)
fig = plot_spillover_table(table, cmap="YlOrRd")

# Divergente (centrado em zero)
fig = plot_spillover_table(table, cmap="RdBu_r")

# Divergente (teal)
fig = plot_spillover_table(table, cmap="BrBG")
```

### Formatar valores

```python
# Sem decimais
fig = plot_spillover_table(table, annot=True, fmt=".0f")

# Duas decimais
fig = plot_spillover_table(table, annot=True, fmt=".2f")

# Sem anotacao (apenas cores)
fig = plot_spillover_table(table, annot=False)
```

---

## Network Graph

O grafico de rede visualiza o sistema de spillover como um **grafo dirigido**,
onde nos representam variaveis e arestas representam fluxos de spillover:

```python
fig = plot_spillover_network(
    table,
    threshold=5.0,
    layout="spring",
    node_size="to_spillover",
    title="Rede de Spillover",
    figsize=(10, 10),
)
```

### Elementos do network

- **Nos**: variaveis do sistema
- **Tamanho do no**: proporcional ao spillover transmitido (`"to_spillover"`)
  ou recebido (`"from_spillover"`)
- **Cor do no**: net spillover (azul = receptor liquido, vermelho = transmissor liquido)
- **Arestas**: fluxos de spillover (espessura proporcional a magnitude)
- **Setas**: direcao do fluxo

### Threshold

O `threshold` filtra arestas fracas para melhorar a legibilidade:

```python
# Apenas spillovers acima de 10%
fig = plot_spillover_network(table, threshold=10.0)

# Todas as conexoes (pode ficar poluido)
fig = plot_spillover_network(table, threshold=0.0)

# Apenas conexoes fortes
fig = plot_spillover_network(table, threshold=15.0)
```

### Layout

```python
# Spring layout (force-directed, padrao)
fig = plot_spillover_network(table, layout="spring")

# Circular
fig = plot_spillover_network(table, layout="circular")

# Kamada-Kawai (baseado em distancia)
fig = plot_spillover_network(table, layout="kamada_kawai")

# Shell (concentrico)
fig = plot_spillover_network(table, layout="shell")
```

### Tamanho e cor dos nos

```python
# Tamanho pelo spillover transmitido
fig = plot_spillover_network(table, node_size="to_spillover")

# Tamanho pelo spillover recebido
fig = plot_spillover_network(table, node_size="from_spillover")

# Tamanho fixo
fig = plot_spillover_network(table, node_size=800)
```

---

## Spillover Dinamico

O grafico de spillover dinamico mostra a evolucao temporal do **Total
Spillover Index (TSI)** calculado em janelas moveis:

$$
S(H) = \frac{\sum_{i \neq j} \tilde{d}_{ij}^H}{\sum_{i,j} \tilde{d}_{ij}^H} \times 100
$$

```python
from chronobox.spillover import DynamicSpillover
from chronobox.visualization import plot_spillover_dynamic

# Calcular spillover dinamico
dynamic = DynamicSpillover(data, lags=4, horizon=10, window=200)

# Plotar evolucao do TSI
fig = plot_spillover_dynamic(
    dynamic,
    title="Total Spillover Index - Mercados Financeiros",
    figsize=(14, 6),
)
```

### Elementos do grafico

- **Linha azul**: Total Spillover Index ao longo do tempo
- **Area sombreada** (opcional): intervalo de confianca
- **Linha tracejada**: media do TSI
- **Recessoes** (opcional): areas sombreadas verticais

### Com recessoes

```python
recessions = [
    ("2008-09-01", "2009-06-01"),
    ("2020-03-01", "2020-06-01"),
]

fig = plot_spillover_dynamic(
    dynamic,
    recession_dates=recessions,
    title="TSI com Crises Financeiras",
)
```

### Anotacoes de eventos

```python
fig = plot_spillover_dynamic(dynamic, title="Total Spillover Index")

ax = fig.axes[0]
ax.axvline(x="2008-09-15", color="red", linestyle="--", alpha=0.7)
ax.annotate(
    "Lehman Brothers",
    xy=("2008-09-15", 85),
    xytext=("2009-06-01", 90),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=10,
    color="red",
)
```

---

## Directional Spillover Plots

Visualize os spillovers **direcionais** (FROM e TO) por variavel ao longo
do tempo:

### FROM others (recebido)

```python
fig = plot_spillover_dynamic(
    dynamic,
    direction="from",
    title="Directional Spillover FROM Others",
    figsize=(14, 6),
)
```

### TO others (transmitido)

```python
fig = plot_spillover_dynamic(
    dynamic,
    direction="to",
    title="Directional Spillover TO Others",
    figsize=(14, 6),
)
```

### NET spillover

```python
fig = plot_spillover_dynamic(
    dynamic,
    direction="net",
    title="Net Directional Spillover",
    figsize=(14, 6),
)
```

O net spillover mostra se cada variavel e **transmissora liquida** (positivo)
ou **receptora liquida** (negativo) de choques ao longo do tempo.

---

## Exemplos com Dados Financeiros

### Indices de mercado

```python
import pandas as pd
from chronobox import VAR
from chronobox.spillover import SpilloverTable, DynamicSpillover
from chronobox.visualization import (
    plot_spillover_table,
    plot_spillover_network,
    plot_spillover_dynamic,
    set_theme,
)

set_theme("professional")

# Retornos de indices
returns = pd.DataFrame({
    "SP500": sp500_returns,
    "FTSE": ftse_returns,
    "DAX": dax_returns,
    "Nikkei": nikkei_returns,
    "Ibovespa": ibov_returns,
})

# Spillover estatico
model = VAR(lags=4)
results = model.fit(returns)
table = SpilloverTable(results, horizon=10)

# Heatmap
fig1 = plot_spillover_table(
    table,
    annot=True,
    cmap="YlOrRd",
    title="Spillover - Indices Globais",
)

# Network
fig2 = plot_spillover_network(
    table,
    threshold=5.0,
    layout="spring",
    node_size="to_spillover",
    title="Rede de Conectividade",
)

# Dinamico
dynamic = DynamicSpillover(returns, lags=4, horizon=10, window=200)

fig3 = plot_spillover_dynamic(
    dynamic,
    title="Evolucao da Conectividade Global",
)
```

### Mercado de cambio

```python
fx_returns = pd.DataFrame({
    "USD/BRL": usdbrl_returns,
    "EUR/USD": eurusd_returns,
    "GBP/USD": gbpusd_returns,
    "USD/JPY": usdjpy_returns,
})

model = VAR(lags=2)
results = model.fit(fx_returns)
table = SpilloverTable(results, horizon=10)

fig = plot_spillover_network(
    table,
    threshold=3.0,
    layout="circular",
    title="Rede de Spillover - Mercado de Cambio",
)
```

---

## Parametros

### `plot_spillover_table`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `table` | SpilloverTable | obrigatorio | Tabela de spillover |
| `annot` | bool | `True` | Exibir valores nas celulas |
| `fmt` | str | `".1f"` | Formato dos numeros |
| `cmap` | str | `"RdYlBu_r"` | Colormap do matplotlib |
| `title` | str | `None` | Titulo do grafico |
| `figsize` | tuple | auto | Tamanho da figura |

### `plot_spillover_network`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `table` | SpilloverTable | obrigatorio | Tabela de spillover |
| `threshold` | float | `5.0` | Spillover minimo para exibir aresta |
| `layout` | str | `"spring"` | Algoritmo de layout do grafo |
| `node_size` | str/int | `"to_spillover"` | Criterio de tamanho dos nos |
| `node_cmap` | str | `"RdYlBu_r"` | Colormap dos nos |
| `edge_cmap` | str | `"Greys"` | Colormap das arestas |
| `title` | str | `None` | Titulo do grafico |
| `figsize` | tuple | `(10, 10)` | Tamanho da figura |

### `plot_spillover_dynamic`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `dynamic_result` | DynamicSpillover | obrigatorio | Resultado do spillover dinamico |
| `direction` | str | `"total"` | `"total"`, `"from"`, `"to"`, `"net"` |
| `recession_dates` | list[tuple] | `None` | Periodos de recessao |
| `show_mean` | bool | `True` | Exibir linha de media |
| `title` | str | `None` | Titulo do grafico |
| `figsize` | tuple | `(14, 6)` | Tamanho da figura |

---

## Dicas de Visualizacao

!!! tip "Threshold no network"
    Comece com `threshold=5.0` e ajuste. Valores muito baixos geram grafos
    poluidos; valores muito altos mostram apenas as conexoes dominantes.

!!! tip "Janela do spillover dinamico"
    A escolha da janela (`window`) afeta a suavidade do TSI. Janelas maiores
    (200-250) produzem series mais suaves; janelas menores (100-150) capturam
    mudancas mais rapidas, mas com mais ruido.

!!! warning "Interpretacao causal"
    Spillover mede **conectividade estatistica**, nao causalidade. Um alto
    spillover de A para B indica que choques em A ajudam a prever a variancia
    de B, nao que A causa B em sentido economico.
