---
title: Themes & Customization
description: Temas built-in, customizacao global e por plot, paletas de cores, integracao matplotlib e export de figuras.
---

# Themes & Customization

!!! info "Quick Reference"
    - **Funcoes**: `set_theme()`, `get_theme()`, `list_themes()`, `register_theme()`
    - **Import**: `from chronobox.visualization import set_theme, get_theme, list_themes`
    - **Temas built-in**: `professional`, `academic`, `presentation`, `bcb`

---

## Overview

O sistema de temas do chronobox permite controlar a aparencia de **todos os
graficos** de forma centralizada. Ao definir um tema, todas as funcoes de
visualizacao (`plot_series`, `plot_irf`, `plot_forecast`, etc.) adotam
automaticamente cores, fontes, espessuras e resolucao do tema ativo.

### Fluxo de uso

```python
from chronobox.visualization import set_theme, plot_series, plot_irf

# 1. Definir tema (uma vez)
set_theme("professional")

# 2. Todos os graficos seguem o tema
fig1 = plot_series(data)           # usa tema professional
fig2 = plot_irf(irf)              # usa tema professional
fig3 = plot_forecast(forecast)    # usa tema professional
```

---

## Temas Built-in

O chronobox inclui **4 temas** pre-configurados, cada um otimizado para
um contexto especifico:

### Professional

Tema padrao, otimizado para **relatorios e dashboards**. Cores neutras,
fontes sans-serif e resolucao media.

```python
set_theme("professional")
```

| Propriedade | Valor |
|---|---|
| Paleta | Azul, cinza, teal |
| Fonte | Sans-serif (Roboto/Helvetica) |
| Tamanho fonte | 11pt |
| Titulo | 14pt, bold |
| Linhas | 1.8pt |
| Grid | Tracejado, alpha=0.3 |
| DPI | 150 |
| Fundo | Branco |

=== "Exemplo"

    ```python
    from chronobox.visualization import set_theme, plot_series

    set_theme("professional")
    fig = plot_series(gdp, title="PIB Real")
    fig.savefig("gdp_pro.png", dpi=150, bbox_inches="tight")
    ```

---

### Academic

Otimizado para **papers e teses**. Preto e branco, fontes serif e alta
resolucao para impressao.

```python
set_theme("academic")
```

| Propriedade | Valor |
|---|---|
| Paleta | Preto, cinza escuro, cinza claro |
| Fonte | Serif (Times/Computer Modern) |
| Tamanho fonte | 10pt |
| Titulo | 12pt, bold |
| Linhas | 1.5pt |
| Grid | Pontilhado, alpha=0.2 |
| DPI | 300 (print-ready) |
| Fundo | Branco |

=== "Exemplo"

    ```python
    set_theme("academic")
    fig = plot_irf(irf, title="Impulse Response Functions")
    fig.savefig("irf_paper.pdf", bbox_inches="tight")
    ```

!!! tip "Para publicacao"
    O tema `academic` ja configura DPI=300 e fontes serif. Exporte em PDF
    para graficos vetoriais de alta qualidade.

---

### Presentation

Otimizado para **slides e apresentacoes**. Cores vibrantes, fontes grandes
e alto contraste.

```python
set_theme("presentation")
```

| Propriedade | Valor |
|---|---|
| Paleta | Cores vibrantes, alto contraste |
| Fonte | Sans-serif, 14pt |
| Titulo | 18pt, bold |
| Linhas | 2.5pt |
| Grid | Tracejado, alpha=0.2 |
| DPI | 150 |
| Fundo | Branco |

=== "Exemplo"

    ```python
    set_theme("presentation")
    fig = plot_forecast(forecast, figsize=(16, 8), title="Projecao Macro")
    fig.savefig("forecast_slides.png", dpi=150, bbox_inches="tight")
    ```

---

### BCB

Estilo institucional do **Banco Central do Brasil**. Cores verde/azul,
visual limpo para relatorios oficiais.

```python
set_theme("bcb")
```

| Propriedade | Valor |
|---|---|
| Paleta | Verde BCB, azul institucional |
| Fonte | Sans-serif (Helvetica) |
| Tamanho fonte | 11pt |
| Titulo | 14pt, bold |
| Linhas | 2.0pt |
| Grid | Tracejado, alpha=0.25 |
| DPI | 150 |
| Fundo | Branco |

=== "Exemplo"

    ```python
    set_theme("bcb")
    fig = plot_series(ipca, title="IPCA - Variacao Mensal (%)")
    fig.savefig("ipca_bcb.png", dpi=150, bbox_inches="tight")
    ```

---

## Comparacao Visual dos Temas

```python
from chronobox.visualization import set_theme, plot_series
import matplotlib.pyplot as plt

themes = ["professional", "academic", "presentation", "bcb"]
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for ax, theme in zip(axes.flat, themes):
    set_theme(theme)
    plot_series(gdp, ax=ax, title=f"Tema: {theme}")

fig.tight_layout()
fig.savefig("theme_comparison.png", dpi=150, bbox_inches="tight")
```

---

## Customizacao Global

### `set_theme()`

Define o tema ativo para todas as funcoes de visualizacao subsequentes:

```python
from chronobox.visualization import set_theme

set_theme("academic")
# Todos os graficos usam tema academic a partir daqui
```

### `get_theme()`

Consulta o tema ativo:

```python
from chronobox.visualization import get_theme

theme = get_theme()
print(theme.name)        # 'academic'
print(theme.figure_dpi)  # 300
print(theme.colors)      # ['#000000', '#404040', '#808080', ...]
print(theme.font_family) # 'serif'
```

### `list_themes()`

Lista todos os temas disponiveis (built-in + customizados):

```python
from chronobox.visualization import list_themes

print(list_themes())
# ['academic', 'bcb', 'presentation', 'professional']
```

### Reset para padrao

```python
set_theme("professional")  # volta ao tema padrao
```

---

## Customizacao por Plot

Qualquer funcao de visualizacao aceita parametros de estilo que **sobrescrevem**
o tema ativo para aquele grafico especifico:

```python
set_theme("professional")

# Este grafico usa cores customizadas, mas o resto do tema permanece
fig = plot_series(
    data,
    colors=["#e74c3c", "#3498db"],
    line_width=2.5,
    title="Customizacao Pontual",
)
```

### Parametros de estilo comuns

Estes parametros estao disponiveis em todas as funcoes `plot_*`:

| Parametro | Tipo | Descricao |
|---|---|---|
| `colors` | list[str] | Paleta de cores (hex ou nome) |
| `line_width` | float | Espessura das linhas |
| `font_size` | int | Tamanho base da fonte |
| `title_size` | int | Tamanho do titulo |
| `grid_alpha` | float | Transparencia do grid |
| `figsize` | tuple | Tamanho da figura `(largura, altura)` |

---

## Tema Customizado

### Criar e registrar

Use `ThemeConfig` para criar um tema completo e `register_theme()` para
registra-lo no sistema:

```python
from chronobox.visualization.themes import ThemeConfig, register_theme
from chronobox.visualization import set_theme

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

### Propriedades de `ThemeConfig`

| Propriedade | Tipo | Default | Descricao |
|---|---|---|---|
| `name` | str | obrigatorio | Nome do tema |
| `colors` | list[str] | `None` | Paleta principal de cores |
| `background` | str | `"#ffffff"` | Cor de fundo |
| `font_family` | str | `"sans-serif"` | Familia de fontes |
| `font_size` | int | `11` | Tamanho base da fonte |
| `title_size` | int | `14` | Tamanho do titulo |
| `label_size` | int | `12` | Tamanho dos rotulos de eixo |
| `tick_size` | int | `10` | Tamanho dos ticks |
| `line_width` | float | `1.8` | Espessura das linhas |
| `grid_alpha` | float | `0.3` | Transparencia do grid |
| `grid_style` | str | `"--"` | Estilo do grid (`"-"`, `"--"`, `":"`, `"-."`) |
| `figure_dpi` | int | `150` | Resolucao da figura |

### Herdar e modificar

Crie um tema baseado em outro, modificando apenas o necessario:

```python
from chronobox.visualization import get_theme
from chronobox.visualization.themes import ThemeConfig, register_theme

# Partir do tema professional
base = get_theme()

# Modificar cores e DPI
custom = ThemeConfig(
    name="professional_hires",
    colors=base.colors,
    background=base.background,
    font_family=base.font_family,
    font_size=base.font_size,
    title_size=base.title_size,
    label_size=base.label_size,
    tick_size=base.tick_size,
    line_width=base.line_width,
    grid_alpha=base.grid_alpha,
    grid_style=base.grid_style,
    figure_dpi=300,  # alta resolucao
)

register_theme("professional_hires", custom)
```

---

## Integracao com Matplotlib Styles

O chronobox e compativel com os **styles** do matplotlib. Voce pode combinar
temas do chronobox com styles do matplotlib:

### Usar style do matplotlib

```python
import matplotlib.pyplot as plt

# Aplicar style do matplotlib
plt.style.use("seaborn-v0_8-whitegrid")

# O tema do chronobox complementa o style
set_theme("professional")
fig = plot_series(data, title="Com Seaborn Style")
```

### Styles disponiveis

```python
import matplotlib.pyplot as plt
print(plt.style.available)
# ['Solarize_Light2', 'bmh', 'classic', 'dark_background',
#  'fast', 'fivethirtyeight', 'ggplot', 'grayscale', ...]
```

### Context manager

Use `plt.style.context` para aplicar um style temporariamente:

```python
import matplotlib.pyplot as plt

with plt.style.context("ggplot"):
    fig = plot_series(data, title="Estilo ggplot temporario")
```

!!! note "Prioridade"
    O tema do chronobox tem prioridade sobre o style do matplotlib.
    Propriedades definidas pelo tema (cores, fontes) sobrescrevem as do style.

---

## Paletas de Cores

O chronobox organiza paletas de cores em tres categorias:

### Sequencial

Para dados ordenados (ex: intensidade, magnitude):

```python
from chronobox.visualization.themes import get_palette

# Paletas sequenciais disponiveis
sequential = get_palette("blues", n=5)
# ['#deebf7', '#9ecae1', '#4292c6', '#2171b5', '#084594']

sequential = get_palette("greens", n=5)
sequential = get_palette("reds", n=5)
sequential = get_palette("oranges", n=5)
```

### Divergente

Para dados com ponto central (ex: spillover, correlacao):

```python
divergent = get_palette("rdbu", n=7)
# Vermelho -> Branco -> Azul

divergent = get_palette("brbg", n=7)
# Marrom -> Branco -> Verde

divergent = get_palette("piyg", n=7)
# Rosa -> Branco -> Verde
```

### Qualitativa

Para categorias distintas (ex: variaveis, modelos):

```python
qualitative = get_palette("set1", n=5)
qualitative = get_palette("set2", n=8)
qualitative = get_palette("tab10", n=10)
qualitative = get_palette("dark2", n=8)
```

### Usar paleta em graficos

```python
from chronobox.visualization.themes import get_palette
from chronobox.visualization import plot_series

# Paleta qualitativa para multiplas series
colors = get_palette("set2", n=4)

fig = plot_series(
    data[["gdp", "inflation", "interest_rate", "exchange_rate"]],
    colors=colors,
    title="Variaveis Macroeconomicas",
)
```

---

## Export de Figuras

Todos os graficos retornam `matplotlib.figure.Figure`, permitindo export
em qualquer formato:

### Formatos disponiveis

```python
fig = plot_series(data, title="PIB Real")

# PNG (raster)
fig.savefig("output.png", dpi=300, bbox_inches="tight")

# SVG (vetorial, editavel)
fig.savefig("output.svg", bbox_inches="tight")

# PDF (vetorial, print-ready)
fig.savefig("output.pdf", bbox_inches="tight")

# EPS (vetorial, LaTeX)
fig.savefig("output.eps", bbox_inches="tight")

# TIFF (raster, alta qualidade)
fig.savefig("output.tiff", dpi=300, bbox_inches="tight")
```

### Helper `save_figure()`

O chronobox oferece uma funcao helper que simplifica o export com
configuracoes otimizadas:

```python
from chronobox.visualization import save_figure

fig = plot_series(data, title="PIB Real")

# Export com configuracoes otimizadas
save_figure(fig, "output.pdf", dpi=300)
save_figure(fig, "output.png", dpi=150, transparent=False)
save_figure(fig, "output.svg")
```

### Parametros de `save_figure`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `fig` | Figure | obrigatorio | Figura do matplotlib |
| `path` | str | obrigatorio | Caminho do arquivo de saida |
| `dpi` | int | tema ativo | Resolucao (apenas raster) |
| `transparent` | bool | `False` | Fundo transparente |
| `bbox_inches` | str | `"tight"` | Ajuste das margens |
| `pad_inches` | float | `0.1` | Padding das margens |
| `facecolor` | str | `None` | Cor de fundo (override) |

### DPI por contexto

| Contexto | DPI Recomendado | Formato |
|---|---|---|
| Web / dashboard | 72-150 | PNG |
| Relatorio | 150-200 | PNG ou PDF |
| Artigo / paper | 300 | PDF ou EPS |
| Poster | 300-600 | PDF ou TIFF |
| Slide / apresentacao | 150 | PNG |

---

## Exemplos de Cada Tema

### Workflow completo com tema Professional

```python
from chronobox import VAR
from chronobox.visualization import (
    set_theme, plot_series, plot_forecast,
    plot_irf, plot_diagnostics, save_figure,
)

set_theme("professional")

# Dados
model = VAR(lags=4)
results = model.fit(data)

# Serie temporal
fig1 = plot_series(data, title="Dados Observados")
save_figure(fig1, "series_pro.png")

# Previsao
forecast = results.forecast(steps=12)
fig2 = plot_forecast(forecast, title="Projecao")
save_figure(fig2, "forecast_pro.png")

# IRF
irf = results.irf(steps=20)
fig3 = plot_irf(irf, title="Impulse Response")
save_figure(fig3, "irf_pro.png")

# Diagnosticos
fig4 = plot_diagnostics(results, title="Diagnosticos")
save_figure(fig4, "diag_pro.png")
```

### Workflow com tema Academic

```python
set_theme("academic")

fig = plot_irf(irf, title="Impulse Response Functions")
save_figure(fig, "figure_1.pdf", dpi=300)
# Preto/branco, serif, DPI 300, vetorial
```

### Workflow com tema BCB

```python
set_theme("bcb")

fig = plot_forecast(
    forecast,
    alpha_levels=[0.30, 0.60, 0.90],
    title="Projecao do PIB Real",
)
save_figure(fig, "projecao_bcb.png", dpi=150)
```

---

## Dicas

!!! tip "Consistencia visual"
    Defina o tema **uma vez** no inicio do script ou notebook. Todos os
    graficos subsequentes serao consistentes visualmente.

!!! tip "Tema por contexto"
    Use `professional` para trabalho diario, `academic` ao preparar figuras
    para papers, `presentation` para slides e `bcb` para relatorios
    institucionais.

!!! warning "Fechar figuras"
    Ao gerar muitos graficos em sequencia, use `plt.close(fig)` apos salvar
    cada figura para liberar memoria:

    ```python
    import matplotlib.pyplot as plt

    for var in variables:
        fig = plot_series(data[var], title=var)
        save_figure(fig, f"{var}.png")
        plt.close(fig)
    ```
