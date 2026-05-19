---
title: Visualization
description: Graficos para series temporais, diagnosticos, IRF, FEVD, spillover e mais --- 4 temas integrados.
---

# Visualization

O modulo `chronobox.visualization` fornece **11 tipos de graficos** e **4 temas**
para visualizar resultados de modelos, diagnosticos e analises de series temporais.

---

## Quick Start

```python
from chronobox.visualization import plot_series, set_theme

# Selecionar tema
set_theme("professional")

# Plotar serie temporal
fig = plot_series(data, title="PIB Real")
fig.savefig("pib.png", dpi=150, bbox_inches="tight")
```

---

## Graficos Disponiveis

| Funcao | Descricao | Pagina |
|---|---|---|
| `plot_series` | Series temporais com anotacoes e eixo secundario | [Time Series Plots](time-series-plots.md) |
| `plot_decomposition` | Paineis verticais (tendencia, sazonal, ciclo, residuo) | [Time Series Plots](time-series-plots.md) |
| `plot_acf` / `plot_pacf` | Autocorrelacao e autocorrelacao parcial | [ACF & PACF](acf-pacf.md) |
| `plot_irf` | Funcoes impulso-resposta (grid KxK) | [IRF Plots](irf-plots.md) |
| `plot_fevd` | Decomposicao da variancia (stacked area/bar) | [FEVD Plots](fevd-plots.md) |
| `plot_forecast` | Fan chart com bandas de confianca degradee | [Forecast Plots](forecast-plots.md) |
| `plot_diagnostics` | Painel 2x2 de diagnosticos residuais | [Diagnostic Plots](diagnostic-plots.md) |
| `plot_hd` | Decomposicao historica | [Filter Plots](filter-plots.md) |
| `plot_network` | Grafo de spillover | [Spillover Plots](spillover-plots.md) |
| `plot_heatmap` | Heatmap de spillover | [Spillover Plots](spillover-plots.md) |
| `plot_rolling` | Spillover em janela movel | [Spillover Plots](spillover-plots.md) |

---

## API de Acesso

Todas as funcoes estao disponveis via import direto:

```python
from chronobox.visualization import (
    plot_series,
    plot_decomposition,
    plot_diagnostics,
    plot_forecast,
    plot_irf,
    plot_fevd,
    plot_hd,
    plot_network,
    plot_heatmap,
    plot_rolling,
    plot_tvp_coefs,
)
```

Ou via metodos dos objetos de resultado:

```python
# Apos estimar um modelo
results = model.fit(data)

# Metodos de visualizacao integrados
fig = results.plot_diagnostics()
fig = results.irf(20).plot()
fig = results.fevd(20).plot()
```

---

## Temas

O chronobox inclui 4 temas pre-configurados que controlam cores, fontes,
espessuras e estilo dos graficos.

| Tema | Descricao | Uso recomendado |
|---|---|---|
| `professional` | Azul/cinza, Helvetica, limpo | Relatorios e dashboards |
| `academic` | Preto/branco, serif, minimalista | Papers e teses (DPI 300) |
| `presentation` | Cores vibrantes, fontes grandes | Slides e apresentacoes |
| `bcb` | Verde/azul institucional | Estilo Banco Central do Brasil |

```python
from chronobox.visualization import set_theme, get_theme, list_themes

# Listar temas disponiveis
print(list_themes())
# ['academic', 'bcb', 'presentation', 'professional']

# Ativar tema
set_theme("academic")

# Consultar tema ativo
theme = get_theme()
print(theme.name)        # 'academic'
print(theme.figure_dpi)  # 300
```

=== "Professional"

    ```python
    set_theme("professional")
    # Cores: azul, cinza, teal
    # Fonte: sans-serif, 11pt
    # DPI: 150
    ```

=== "Academic"

    ```python
    set_theme("academic")
    # Cores: preto, cinza
    # Fonte: serif, 10pt
    # DPI: 300 (print-ready)
    ```

=== "Presentation"

    ```python
    set_theme("presentation")
    # Cores vibrantes, alto contraste
    # Fonte: sans-serif, 14pt
    # Linhas grossas (2.5pt)
    ```

=== "BCB"

    ```python
    set_theme("bcb")
    # Verde/azul institucional
    # Estilo Banco Central do Brasil
    # DPI: 150
    ```

### Tema Customizado

```python
from chronobox.visualization.themes import ThemeConfig, register_theme

meu_tema = ThemeConfig(
    name="meu_tema",
    colors=["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
    background="#ffffff",
    font_family="sans-serif",
    font_size=12,
    title_size=16,
    label_size=13,
    tick_size=11,
    line_width=2.0,
    grid_alpha=0.3,
    grid_style="--",
    figure_dpi=200,
)

register_theme("meu_tema", meu_tema)
set_theme("meu_tema")
```

---

## Export

Todos os graficos retornam um objeto `matplotlib.figure.Figure`, permitindo
export em qualquer formato suportado pelo matplotlib:

```python
fig = plot_series(data, title="PIB")

# PNG (raster)
fig.savefig("grafico.png", dpi=300, bbox_inches="tight")

# SVG (vetorial)
fig.savefig("grafico.svg", bbox_inches="tight")

# PDF (vetorial, print-ready)
fig.savefig("grafico.pdf", bbox_inches="tight")

# Fundo transparente
fig.savefig("grafico.png", dpi=300, transparent=True, bbox_inches="tight")
```

!!! tip "DPI para publicacao"
    Use `dpi=300` para artigos e relatorios impressos. O tema `academic` ja
    configura DPI=300 por padrao.

---

## Integracao com Matplotlib

Como todas as funcoes retornam objetos `Figure` do matplotlib, voce pode
customizar qualquer aspecto apos a criacao:

```python
fig = plot_series(data)

# Acessar axes para customizacao
ax = fig.axes[0]
ax.set_xlim("2010-01-01", "2023-12-31")
ax.axvline(x="2020-03-01", color="red", linestyle="--", label="COVID-19")
ax.legend()

fig.savefig("customizado.png", dpi=150, bbox_inches="tight")
```

Tambem e possivel compor graficos em subplots customizados:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Usar parametro ax= para plotar em axes existentes
plot_series(gdp, ax=axes[0], title="PIB")
plot_series(inflation, ax=axes[1], title="Inflacao")

fig.tight_layout()
fig.savefig("painel.png", dpi=150)
```
