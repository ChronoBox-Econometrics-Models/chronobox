---
title: Diagnostic Plots
description: Graficos de diagnostico --- painel 4-em-1 de residuos, CUSUM com bandas, unit circle de eigenvalues e mais.
---

# Diagnostic Plots

!!! info "Quick Reference"
    - **Funcoes**: `plot_diagnostics()`, `plot_cusum()`, `plot_unit_circle()`
    - **Import**: `from chronobox.visualization import plot_diagnostics, plot_cusum, plot_unit_circle`
    - **Input**: Objeto de resultado de modelo estimado
    - **Output**: `matplotlib.figure.Figure`

---

## Overview

Os graficos de diagnostico permitem avaliar visualmente a **qualidade do
ajuste** e a **adequacao das hipoteses** de um modelo estimado. Sao
complementares aos testes estatisticos formais (Ljung-Box, Jarque-Bera, etc.)
e devem ser consultados em toda analise.

O modulo oferece tres funcoes principais:

| Funcao | Descricao | Uso |
|---|---|---|
| `plot_diagnostics()` | Painel 2x2 de diagnosticos residuais | Verificar hipoteses do modelo |
| `plot_cusum()` | CUSUM com bandas de significancia | Detectar instabilidade parametrica |
| `plot_unit_circle()` | Circulo unitario com eigenvalues | Verificar estabilidade do VAR |

---

## Painel 4-em-1: `plot_diagnostics()`

O painel de diagnosticos exibe quatro graficos em uma grade 2x2:

```python
from chronobox import ARIMA
from chronobox.visualization import plot_diagnostics, set_theme

set_theme("professional")

model = ARIMA(order=(2, 1, 1))
results = model.fit(data)

fig = plot_diagnostics(results)
fig.savefig("diagnostics.png", dpi=150, bbox_inches="tight")
```

### Os quatro paineis

#### 1. Residuos vs. Tempo (superior esquerdo)

Exibe os residuos padronizados ao longo do tempo. Permite identificar:

- **Heterocedasticidade**: variancia nao constante ao longo do tempo
- **Outliers**: residuos acima de $\pm 2\sigma$
- **Padroes estruturais**: tendencia ou sazonalidade nos residuos

```
Ideal: residuos dispersos aleatoriamente em torno de zero,
       sem padrao aparente e variancia constante.
```

#### 2. Histograma dos Residuos (superior direito)

Histograma com curva normal sobreposta e estatisticas descritivas.
Permite avaliar:

- **Normalidade**: formato de sino simetrico
- **Assimetria**: desvio a esquerda ou direita
- **Curtose**: caudas mais pesadas que a normal

$$
\text{Jarque-Bera} = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)
$$

#### 3. QQ-Plot (inferior esquerdo)

Compara os quantis empiricos dos residuos com os quantis teoricos de uma
distribuicao normal. Permite identificar:

- **Caudas pesadas**: pontos se afastam da reta nas extremidades
- **Assimetria**: curvatura sistematica
- **Normalidade**: pontos alinhados na reta de 45 graus

```
Ideal: todos os pontos sobre a reta de referencia.
```

#### 4. ACF dos Residuos (inferior direito)

Funcao de autocorrelacao dos residuos com bandas de confianca a 95%.
Permite identificar:

- **Autocorrelacao residual**: barras fora das bandas indicam correlacao
  serial nao capturada pelo modelo
- **Sazonalidade residual**: picos regulares nos lags sazonais

```
Ideal: todas as barras dentro das bandas de confianca
       (exceto lag 0 que e sempre 1).
```

---

## Opcoes do Painel

### Modelo univariado

```python
fig = plot_diagnostics(
    results,
    lags=40,
    title="Diagnosticos - ARIMA(2,1,1)",
    figsize=(12, 10),
)
```

### Modelo multivariado (por variavel)

Para modelos VAR/VECM, o painel e gerado para cada variavel separadamente:

```python
from chronobox import VAR
from chronobox.visualization import plot_diagnostics

model = VAR(lags=4)
results = model.fit(data)

# Diagnostico de uma variavel especifica
fig = plot_diagnostics(
    results,
    variable="gdp",
    title="Diagnosticos - Residuos do PIB",
)
```

### Todas as variaveis

```python
# Gera um painel para cada variavel
figs = plot_diagnostics(
    results,
    variable="all",
    figsize=(12, 10),
)

# figs e uma lista de figuras
for var_name, fig in figs.items():
    fig.savefig(f"diag_{var_name}.png", dpi=150, bbox_inches="tight")
```

### Estilo customizado

```python
fig = plot_diagnostics(
    results,
    style={
        "residual_color": "#2c3e50",
        "hist_color": "#3498db",
        "hist_bins": 30,
        "qq_color": "#e74c3c",
        "acf_color": "#2980b9",
    },
    figsize=(14, 12),
)
```

---

## CUSUM: `plot_cusum()`

O teste CUSUM (Cumulative Sum) detecta **instabilidade parametrica** ao
longo do tempo. O grafico mostra a soma cumulativa dos residuos recursivos
com bandas de significancia:

$$
W_t = \sum_{s=k+1}^{t} \frac{w_s}{\hat{\sigma}}, \quad t = k+1, \ldots, T
$$

onde $w_s$ sao os residuos recursivos e $\hat{\sigma}$ e o desvio padrao
estimado.

```python
from chronobox.visualization import plot_cusum

fig = plot_cusum(
    results,
    significance=0.05,
    title="CUSUM Test - Estabilidade Parametrica",
)
```

### Elementos do grafico

- **Linha azul**: CUSUM acumulado
- **Linhas vermelhas tracejadas**: bandas de significancia a 5%
- **Interpretacao**: se a linha azul cruza as bandas, rejeita-se a hipotese
  de estabilidade parametrica

### CUSUM de quadrados

```python
fig = plot_cusum(
    results,
    squared=True,
    significance=0.05,
    title="CUSUM of Squares",
)
```

O CUSUM de quadrados e sensivel a mudancas na **variancia** dos residuos,
enquanto o CUSUM padrao detecta mudancas na **media**.

### Com marcacao de quebra

```python
fig = plot_cusum(results, significance=0.05)

# Marcar data de possivel quebra estrutural
ax = fig.axes[0]
ax.axvline(x="2008-09-15", color="gray", linestyle=":", alpha=0.7)
ax.text("2008-10-01", ax.get_ylim()[1] * 0.9, "Lehman", fontsize=9)
```

---

## Unit Circle: `plot_unit_circle()`

O grafico do circulo unitario mostra os **eigenvalues** (autovalores) da
matriz companion de um modelo VAR. Se todos os eigenvalues estao **dentro**
do circulo unitario, o sistema e **estavel** (estacionario):

$$
|\lambda_i| < 1, \quad \forall\, i = 1, \ldots, Kp
$$

```python
from chronobox import VAR
from chronobox.visualization import plot_unit_circle

model = VAR(lags=4)
results = model.fit(data)

fig = plot_unit_circle(
    results,
    title="VAR Stability - Eigenvalues",
)
```

### Elementos do grafico

- **Circulo preto**: circulo unitario ($|\lambda| = 1$)
- **Pontos azuis**: eigenvalues da companion matrix
- **Pontos vermelhos** (se houver): eigenvalues fora do circulo (instavel)
- **Tabela lateral**: modulos dos eigenvalues

### Interpretacao

```
Estavel:    Todos os pontos DENTRO do circulo
Instavel:   Um ou mais pontos SOBRE ou FORA do circulo
Unit root:  Ponto exatamente sobre o circulo (usar VECM)
```

!!! warning "Eigenvalues na fronteira"
    Se um eigenvalue tem modulo muito proximo de 1 (ex: 0.98), o sistema e
    tecnicamente estavel mas altamente persistente. Considere testar
    cointegração e usar VECM em vez de VAR em niveis.

### Opcoes

```python
fig = plot_unit_circle(
    results,
    show_table=True,     # Tabela com modulos
    show_labels=True,    # Rotulos nos eigenvalues
    title="Estabilidade do VAR(4)",
    figsize=(8, 8),
)
```

---

## Exemplos com Diferentes Modelos

### ARIMA

```python
from chronobox import ARIMA
from chronobox.visualization import plot_diagnostics, plot_cusum

model = ARIMA(order=(1, 1, 1))
results = model.fit(gdp)

# Painel de diagnosticos
fig1 = plot_diagnostics(results, title="ARIMA(1,1,1) - PIB")

# CUSUM
fig2 = plot_cusum(results, title="CUSUM - ARIMA(1,1,1)")
```

### ETS

```python
from chronobox import ETS
from chronobox.visualization import plot_diagnostics

model = ETS(error="add", trend="add", seasonal="mul", seasonal_periods=12)
results = model.fit(retail_sales)

fig = plot_diagnostics(
    results,
    lags=36,
    title="ETS(A,A,M) - Vendas no Varejo",
)
```

### VAR

```python
from chronobox import VAR
from chronobox.visualization import (
    plot_diagnostics,
    plot_cusum,
    plot_unit_circle,
)

model = VAR(lags=4)
results = model.fit(data[["gdp", "inflation", "interest_rate"]])

# Diagnosticos por variavel
fig1 = plot_diagnostics(results, variable="gdp")
fig2 = plot_diagnostics(results, variable="inflation")

# Estabilidade
fig3 = plot_unit_circle(results, title="VAR(4) Stability")

# CUSUM por equacao
fig4 = plot_cusum(results, variable="gdp")
```

### SVAR

```python
from chronobox import SVAR
from chronobox.visualization import plot_diagnostics, plot_unit_circle

model = SVAR(lags=4, svar_type="B", B=B_matrix)
results = model.fit(data)

# Herda estabilidade do VAR subjacente
fig = plot_unit_circle(results, title="SVAR Stability")
```

### VECM

```python
from chronobox import VECM
from chronobox.visualization import plot_diagnostics

model = VECM(lags=3, coint_rank=1)
results = model.fit(data)

fig = plot_diagnostics(
    results,
    variable="gdp",
    title="VECM - Diagnosticos dos Residuos",
)
```

---

## Parametros

### `plot_diagnostics`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `results` | objeto | obrigatorio | Resultado de modelo estimado |
| `variable` | str | `None` | Variavel especifica (multivariado) ou `"all"` |
| `lags` | int | `20` | Numero de lags na ACF |
| `style` | dict | `None` | Dict com cores e opcoes de estilo |
| `title` | str | `None` | Titulo geral |
| `figsize` | tuple | `(12, 10)` | Tamanho da figura |

### `plot_cusum`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `results` | objeto | obrigatorio | Resultado de modelo estimado |
| `variable` | str | `None` | Variavel especifica (multivariado) |
| `squared` | bool | `False` | CUSUM de quadrados |
| `significance` | float | `0.05` | Nivel de significancia das bandas |
| `title` | str | `None` | Titulo do grafico |
| `figsize` | tuple | `(12, 5)` | Tamanho da figura |

### `plot_unit_circle`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `results` | objeto | obrigatorio | Resultado de modelo VAR/SVAR/VECM |
| `show_table` | bool | `True` | Exibir tabela com modulos |
| `show_labels` | bool | `False` | Rotulos nos pontos de eigenvalues |
| `title` | str | `None` | Titulo do grafico |
| `figsize` | tuple | `(8, 8)` | Tamanho da figura |

---

## Dicas de Visualizacao

!!! tip "Checklist de diagnosticos"
    Apos estimar qualquer modelo, verifique sempre:

    1. **Residuos vs. tempo**: sem padrao ou heterocedasticidade
    2. **Histograma**: formato aproximadamente normal
    3. **QQ-plot**: pontos na reta de 45 graus
    4. **ACF**: sem autocorrelacao significativa

!!! tip "CUSUM para mudanca de regime"
    O CUSUM e especialmente util para detectar mudancas de regime que testes
    pontuais (como Chow) podem nao capturar. Use-o como complemento.

!!! warning "Normalidade nem sempre e critica"
    Para amostras grandes ($T > 100$), desvios moderados da normalidade nao
    invalidam a inferencia. O CLT garante consistencia dos estimadores OLS.
    Foque mais na autocorrelacao e heterocedasticidade.
