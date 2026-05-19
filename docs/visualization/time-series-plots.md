---
title: Time Series Plots
description: Graficos de series temporais, decomposicao e sazonalidade com plot_series e plot_decomposition.
---

# Time Series Plots

!!! info "Quick Reference"
    - **Funcao**: `plot_series()`, `plot_decomposition()`
    - **Import**: `from chronobox.visualization import plot_series, plot_decomposition`
    - **Retorno**: `matplotlib.figure.Figure`
    - **Temas**: Todos os 4 temas suportados

---

## Overview

Os graficos de series temporais sao a base de qualquer analise. O chronobox
oferece duas funcoes principais:

- **`plot_series()`** --- Plotar uma ou mais series temporais, com suporte a
  anotacoes, barras de recessao e eixo Y secundario
- **`plot_decomposition()`** --- Plotar componentes de uma decomposicao
  (tendencia, sazonalidade, ciclo, residuo) em paineis verticais

---

## `plot_series` --- Serie Basica

### Plot de uma serie

```python
import pandas as pd
from chronobox.visualization import plot_series, set_theme

set_theme("professional")

# Carregar dados
gdp = pd.read_csv("gdp.csv", index_col=0, parse_dates=True)["gdp"]

# Plot basico
fig = plot_series(gdp, title="PIB Real - Brasil")
fig.savefig("pib.png", dpi=150, bbox_inches="tight")
```

### Multiplas series

```python
import pandas as pd
from chronobox.visualization import plot_series

# DataFrame com multiplas colunas
data = pd.DataFrame({
    "PIB": gdp,
    "Consumo": consumo,
    "Investimento": investimento,
})

fig = plot_series(
    data,
    title="Componentes do PIB",
    ylabel="R$ bilhoes (2010)",
)
```

Tambem aceita uma lista de Series:

```python
fig = plot_series(
    [gdp, consumo, investimento],
    labels=["PIB", "Consumo", "Investimento"],
    title="Componentes do PIB",
)
```

### Eixo Y secundario

Quando as series possuem escalas muito diferentes, use o eixo Y secundario:

```python
fig = plot_series(
    data[["PIB", "Selic"]],
    secondary_y=["Selic"],
    ylabel="R$ bilhoes",
    secondary_ylabel="% a.a.",
    title="PIB e Taxa Selic",
)
```

### Barras de recessao

Adicione sombreamento para periodos de recessao:

```python
from datetime import datetime

recessions = [
    (datetime(2008, 10, 1), datetime(2009, 3, 1)),
    (datetime(2014, 4, 1), datetime(2016, 12, 1)),
    (datetime(2020, 3, 1), datetime(2020, 6, 1)),
]

fig = plot_series(
    gdp,
    title="PIB Real com Recessoes",
    recessions=recessions,
    recession_color="#cccccc",
    recession_alpha=0.3,
)
```

### Anotacoes

```python
annotations = [
    {
        "x": datetime(2008, 9, 15),
        "y": 150.0,
        "text": "Lehman Brothers",
        "arrowprops": {"arrowstyle": "->", "color": "red"},
    },
    {
        "x": datetime(2020, 3, 11),
        "y": 140.0,
        "text": "Pandemia",
        "fontsize": 10,
        "color": "darkred",
    },
]

fig = plot_series(
    gdp,
    title="PIB Real - Eventos",
    annotations=annotations,
    recessions=recessions,
)
```

---

## Parametros de `plot_series`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `data` | Series, DataFrame, ndarray, list | --- | Dados. DataFrame plota cada coluna; lista plota cada elemento |
| `labels` | list[str] | `None` | Rotulos para cada serie |
| `title` | str | `None` | Titulo do grafico |
| `xlabel` | str | `None` | Rotulo do eixo X |
| `ylabel` | str | `None` | Rotulo do eixo Y |
| `secondary_y` | list[str] | `None` | Series a plotar no eixo Y secundario |
| `secondary_ylabel` | str | `None` | Rotulo do eixo Y secundario |
| `annotations` | list[dict] | `None` | Anotacoes com chaves `x`, `y`, `text` |
| `recessions` | list[tuple] | `None` | Periodos de recessao `(inicio, fim)` |
| `recession_color` | str | `"#cccccc"` | Cor do sombreamento de recessao |
| `recession_alpha` | float | `0.3` | Transparencia do sombreamento |
| `figsize` | tuple | `(12, 5)` | Tamanho da figura (largura, altura) em polegadas |
| `legend` | bool | `True` | Exibir legenda |
| `grid` | bool | `True` | Exibir grid |
| `ax` | Axes | `None` | Axes existente para plotar. Se `None`, cria nova figura |
| `**kwargs` | --- | --- | Argumentos extras passados para `plt.plot()` |

---

## `plot_decomposition` --- Componentes

Plota os componentes de uma decomposicao (STL, ETS, X-13, filtros) em paineis
verticais alinhados.

### Decomposicao basica

```python
from chronobox import STL
from chronobox.visualization import plot_decomposition

# Decomposicao STL
model = STL(period=12)
results = model.fit(monthly_data)

# Plot de todos os componentes
fig = plot_decomposition(results, title="Decomposicao STL")
```

### Selecionar componentes

```python
# Plotar apenas tendencia e sazonalidade
fig = plot_decomposition(
    results,
    which=["trend", "seasonal"],
    title="Tendencia e Sazonalidade",
)
```

### Componentes manuais

Voce pode fornecer componentes como dicionario, sem precisar de um objeto de resultado:

```python
import numpy as np

fig = plot_decomposition(
    components={
        "observed": y,
        "trend": trend_estimado,
        "seasonal": sazonal_estimado,
        "remainder": residuo,
    },
    title="Decomposicao Manual",
    dates=y.index,
)
```

---

## Parametros de `plot_decomposition`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `results` | objeto | `None` | Resultado de decomposicao com atributos de componentes |
| `components` | dict[str, ndarray] | `None` | Dicionario de componentes manual |
| `which` | list[str] | `None` | Componentes a plotar. Se `None`, plota todos |
| `title` | str | `None` | Titulo geral |
| `figsize` | tuple | auto | Tamanho da figura |
| `share_x` | bool | `True` | Compartilhar eixo X entre paineis |
| `dates` | array | `None` | Indice de datas para o eixo X |

!!! tip "Componentes reconhecidos"
    O chronobox reconhece automaticamente os seguintes nomes de atributos:

    - **Observed**: `observed`, `endog`, `y`, `data`
    - **Trend**: `trend`, `trend_component`, `level`
    - **Seasonal**: `seasonal`, `seasonal_component`
    - **Cycle**: `cycle`, `cyclical`, `cycle_component`
    - **Remainder**: `resid`, `residual`, `remainder`, `irregular`

---

## Sazonalidade e Subseries

### Plot sazonal

Para visualizar padroes sazonais, plote as series por periodo:

```python
import matplotlib.pyplot as plt
import numpy as np

# Dados mensais
monthly = gdp.resample("M").mean()

fig, ax = plt.subplots(figsize=(10, 5))
for year in range(2015, 2024):
    mask = monthly.index.year == year
    subset = monthly[mask]
    ax.plot(subset.index.month, subset.values, alpha=0.5, label=str(year))

ax.set_xlabel("Mes")
ax.set_ylabel("PIB")
ax.set_title("Sazonalidade por Ano")
ax.legend(ncol=3, fontsize=8)
ax.set_xticks(range(1, 13))
fig.tight_layout()
```

### Subseries sazonais (seasonal subseries)

```python
fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharey=True)
meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
         "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

for i, ax in enumerate(axes.flat):
    mask = monthly.index.month == (i + 1)
    subset = monthly[mask]
    ax.plot(subset.index.year, subset.values, marker="o", markersize=3)
    ax.axhline(subset.mean(), color="red", linestyle="--", linewidth=0.8)
    ax.set_title(meses[i], fontsize=10)
    ax.tick_params(labelsize=8)

fig.suptitle("Subseries Sazonais", fontsize=14, fontweight="bold")
fig.tight_layout()
```

---

## Exemplos por Tipo de Dado

=== "Dados Mensais"

    ```python
    fig = plot_series(
        monthly_gdp,
        title="PIB Mensal - Brasil",
        ylabel="Indice (2010=100)",
    )
    ```

=== "Dados Trimestrais"

    ```python
    fig = plot_series(
        quarterly_gdp,
        title="PIB Trimestral",
        ylabel="R$ bilhoes",
    )
    ```

=== "Dados Diarios"

    ```python
    fig = plot_series(
        daily_returns,
        title="Retornos Diarios - IBOVESPA",
        ylabel="Retorno (%)",
        figsize=(14, 4),
    )
    ```

=== "NumPy Array"

    ```python
    import numpy as np

    y = np.random.randn(200).cumsum()
    fig = plot_series(y, title="Random Walk", labels=["Simulacao"])
    ```

---

## Composicao com Subplots

Para criar paineis customizados, use o parametro `ax`:

```python
import matplotlib.pyplot as plt
from chronobox.visualization import plot_series

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_series(gdp, ax=axes[0, 0], title="PIB")
plot_series(inflation, ax=axes[0, 1], title="Inflacao")
plot_series(unemployment, ax=axes[1, 0], title="Desemprego")
plot_series(selic, ax=axes[1, 1], title="Selic")

fig.suptitle("Painel Macroeconomico", fontsize=16, fontweight="bold")
fig.tight_layout()
fig.savefig("painel_macro.png", dpi=200, bbox_inches="tight")
```

---

## Comparacao entre Temas

=== "Professional"

    ```python
    from chronobox.visualization import set_theme, plot_series
    set_theme("professional")
    fig = plot_series(data, title="PIB Real")
    # Azul/cinza, limpo, ideal para relatorios
    ```

=== "Academic"

    ```python
    set_theme("academic")
    fig = plot_series(data, title="PIB Real")
    # Preto/branco, serif, DPI 300
    ```

=== "Presentation"

    ```python
    set_theme("presentation")
    fig = plot_series(data, title="PIB Real")
    # Cores vibrantes, fontes grandes
    ```

=== "BCB"

    ```python
    set_theme("bcb")
    fig = plot_series(data, title="PIB Real")
    # Verde/azul institucional
    ```
