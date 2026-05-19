---
title: FEVD Plots
description: Graficos de decomposicao da variancia do erro de previsao --- stacked area e stacked bar.
---

# FEVD Plots

!!! info "Quick Reference"
    - **Funcao**: `plot_fevd()`
    - **Import**: `from chronobox.visualization import plot_fevd`
    - **Tipos**: Stacked area chart, stacked bar chart
    - **Input**: Objeto de resultado FEVD ou array NumPy (H, K, K)

---

## Overview

O grafico de FEVD (Forecast Error Variance Decomposition) mostra a **proporcao
da variancia do erro de previsao** de cada variavel que e atribuivel a choques
em cada uma das variaveis do sistema.

Enquanto a IRF mostra *como* um choque se propaga, a FEVD mostra a
**importancia relativa** de cada fonte de choque. E uma ferramenta
complementar essencial para interpretar modelos VAR.

### Quando usar

- Quantificar a importancia relativa de cada choque para cada variavel
- Identificar variaveis "exogenas" (maior parte da variancia propria) vs "endogenas"
- Complementar a analise de IRF
- Comunicar resultados de dependencia entre variaveis

---

## Quick Start

```python
from chronobox import VAR
from chronobox.visualization import plot_fevd, set_theme

set_theme("professional")

# Estimar VAR
model = VAR(lags=4)
results = model.fit(data)

# Calcular FEVD
fevd = results.fevd(steps=20)

# Plot
fig = plot_fevd(fevd)
fig.savefig("fevd.png", dpi=150, bbox_inches="tight")
```

---

## Stacked Area Chart (padrao)

O stacked area chart mostra a evolucao das proporcoes ao longo do horizonte
de previsao. Cada area colorida representa a contribuicao de um choque:

```python
from chronobox.visualization import plot_fevd

fig = plot_fevd(
    fevd,
    plot_type="area",
    title="FEVD - Stacked Area",
)
```

A soma das areas em cada horizonte e sempre igual a 1 (100%),
representando a decomposicao completa da variancia.

---

## Stacked Bar Chart

O stacked bar chart mostra a mesma informacao em barras empilhadas,
facilitando a leitura de valores exatos em cada horizonte:

```python
fig = plot_fevd(
    fevd,
    plot_type="bar",
    title="FEVD - Stacked Bar",
)
```

!!! tip "Area vs Bar"
    - **Area**: Melhor para visualizar tendencias e transicoes suaves
    - **Bar**: Melhor para leitura precisa de proporcoes em horizontes especificos

---

## FEVD para Variavel Individual

Para focar na decomposicao de uma variavel especifica, use o parametro `variable`:

```python
# FEVD apenas para o PIB
fig = plot_fevd(
    fevd,
    variable="gdp",
    title="FEVD: PIB",
)
```

Usando indice numerico:

```python
# FEVD da primeira variavel
fig = plot_fevd(
    fevd,
    variable=0,
    title="FEVD: Primeira Variavel",
)
```

---

## Interpretacao Visual

### Variavel "exogena"

Se a maior parte da variancia de uma variavel e explicada **por seu proprio
choque** (a cor propria domina o grafico), a variavel e relativamente exogena
no sistema:

```python
# Se a area/barra do proprio choque domina em todos os horizontes,
# a variavel e dirigida principalmente por seus proprios choques
fig = plot_fevd(fevd, variable="gdp")
# -> Se "gdp" explica >80% da variancia de "gdp", PIB e exogeno
```

### Variavel "endogena"

Se choques de **outras variaveis** explicam uma parcela significativa da
variancia, a variavel e endogena e responde a choques do sistema:

```python
# Se choques externos explicam parcela grande da variancia,
# a variavel responde fortemente ao sistema
fig = plot_fevd(fevd, variable="exchange_rate")
# -> Se "interest_rate" explica 40% da variancia do cambio, ha forte
#    transmissao de politica monetaria para o cambio
```

### Evolucao temporal

A FEVD tipicamente muda com o horizonte:

- **Curto prazo** (h=1): a maior parte da variancia e propria (choque proprio)
- **Longo prazo** (h→∞): choques de outras variaveis ganham importancia

```python
fig = plot_fevd(
    fevd,
    plot_type="area",
    title="FEVD: Evolucao Temporal da Importancia dos Choques",
)
```

---

## Uso com Arrays NumPy

Para flexibilidade total, fornaeca arrays diretamente:

```python
import numpy as np
from chronobox.visualization import plot_fevd

# FEVD array: shape (H, K, K)
# fevd_array[h, i, j] = proporcao da variancia de var i explicada por choque j
# no horizonte h
K = 3
H = 20

# Exemplo sintetico: variancia propria dominante decaindo
fevd_array = np.zeros((H, K, K))
for h in range(H):
    for i in range(K):
        own = max(0.5, 1.0 - 0.02 * h)
        others = (1.0 - own) / (K - 1)
        for j in range(K):
            fevd_array[h, i, j] = own if i == j else others

fig = plot_fevd(
    fevd_array=fevd_array,
    var_names=["PIB", "Inflacao", "Selic"],
    plot_type="area",
    title="FEVD Sintetica",
)
```

---

## Parametros de `plot_fevd`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `fevd_results` | objeto | `None` | Resultado FEVD com arrays e nomes |
| `fevd_array` | ndarray (H,K,K) | `None` | Array FEVD direto |
| `var_names` | list[str] | `None` | Nomes das variaveis/choques |
| `variable` | str ou int | `None` | Plotar apenas para esta variavel de resposta |
| `plot_type` | str | `"area"` | `"area"` para stacked area, `"bar"` para stacked bar |
| `title` | str | `None` | Titulo geral |
| `figsize` | tuple | auto | Tamanho da figura |

---

## Exemplo Completo com VAR

```python
import pandas as pd
from chronobox import VAR
from chronobox.visualization import plot_irf, plot_fevd, set_theme

set_theme("professional")

# Dados macroeconomicos
data = pd.DataFrame({
    "gdp": gdp_growth,
    "inflation": cpi_change,
    "interest_rate": selic,
})

# Estimar VAR(4)
model = VAR(lags=4)
results = model.fit(data)

# IRF e FEVD
irf = results.irf(steps=24)
fevd = results.fevd(steps=24)

# Plotar lado a lado
import matplotlib.pyplot as plt

fig_irf = plot_irf(irf, title="IRF")
fig_irf.savefig("irf.png", dpi=150, bbox_inches="tight")

fig_fevd = plot_fevd(fevd, plot_type="area", title="FEVD")
fig_fevd.savefig("fevd.png", dpi=150, bbox_inches="tight")
```

---

## Comparacao entre Temas

=== "Professional"

    ```python
    set_theme("professional")
    fig = plot_fevd(fevd, plot_type="area")
    # Cores claras, grid discreto
    ```

=== "Academic"

    ```python
    set_theme("academic")
    fig = plot_fevd(fevd, plot_type="bar")
    # Tons de cinza, ideal para impressao P&B
    ```

=== "Presentation"

    ```python
    set_theme("presentation")
    fig = plot_fevd(fevd, plot_type="area")
    # Cores vibrantes, fontes grandes
    ```

=== "BCB"

    ```python
    set_theme("bcb")
    fig = plot_fevd(fevd, plot_type="area")
    # Verde/azul institucional
    ```

---

## FEVD Ortogonal vs Generalizada

!!! warning "FEVD generalizada: linhas nao somam 1"
    Na FEVD generalizada (Pesaran-Shin), as proporcoes **nao somam 1** por
    construcao, pois os choques nao sao ortogonais. O chronobox normaliza
    automaticamente para exibicao, mas e importante entender essa distincao.

```python
# FEVD ortogonal (Cholesky) - depende da ordenacao
fevd_orth = results.fevd(steps=20, method="cholesky")

# FEVD generalizada - invariante a ordenacao
fevd_gen = results.fevd(steps=20, method="generalized")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Comparar visualmente
plot_fevd(fevd_orth, variable="gdp", title="Ortogonal")
plot_fevd(fevd_gen, variable="gdp", title="Generalizada")
```
