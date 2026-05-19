---
title: Forecast Plots
description: Graficos de previsao com fan charts, bandas de confianca degradee, historico + forecast e fitted values.
---

# Forecast Plots

!!! info "Quick Reference"
    - **Funcao**: `plot_forecast()`
    - **Import**: `from chronobox.visualization import plot_forecast`
    - **Input**: Objeto de resultado de previsao ou arrays NumPy
    - **Output**: `matplotlib.figure.Figure` com historico e fan chart

---

## Overview

O grafico de previsao (forecast plot) e a visualizacao mais comum em series
temporais. Ele combina o **historico observado** com a **previsao futura**,
exibindo bandas de confianca em formato **fan chart** --- faixas degradee que
representam diferentes niveis de incerteza.

O `plot_forecast()` gera automaticamente:

- Linha solida para dados historicos
- Linha tracejada para previsao pontual
- Bandas sombreadas com transparencia crescente para intervalos de confianca
- Transicao suave entre historico e previsao

### Quando usar

- Apresentar previsoes de modelos ARIMA, ETS, VAR, BVAR
- Comparar previsao pontual com intervalos de incerteza
- Avaliar visualmente fitted values vs. dados observados
- Relatorios e dashboards de projecoes economicas

---

## Quick Start

```python
from chronobox import ARIMA
from chronobox.visualization import plot_forecast, set_theme

set_theme("professional")

# Estimar modelo
model = ARIMA(order=(1, 1, 1))
results = model.fit(data)

# Previsao
forecast = results.forecast(steps=24)

# Fan chart
fig = plot_forecast(forecast)
fig.savefig("forecast.png", dpi=150, bbox_inches="tight")
```

---

## Fan Chart com Multiplos Intervalos

O fan chart exibe multiplos niveis de confianca simultaneamente, criando um
efeito visual de "leque" que comunica a incerteza de forma intuitiva:

$$
\hat{y}_{T+h} \pm z_{\alpha/2} \cdot \hat{\sigma}_h, \quad h = 1, \ldots, H
$$

onde $\hat{\sigma}_h$ cresce com o horizonte $h$, refletindo o aumento
da incerteza ao longo do tempo.

```python
fig = plot_forecast(
    forecast,
    alpha_levels=[0.50, 0.80, 0.95],
    title="PIB Real - Previsao com Fan Chart",
)
```

Cada nivel de confianca e renderizado com transparencia decrescente:

| Nivel | Cobertura | Transparencia |
|---|---|---|
| 50% | Faixa interna | Alta (mais escuro) |
| 80% | Faixa intermediaria | Media |
| 95% | Faixa externa | Baixa (mais claro) |

!!! tip "Escolha de niveis"
    Para relatorios do Banco Central, os niveis 30%, 60% e 90% sao comuns.
    Para publicacoes academicas, 68% (1$\sigma$) e 95% (2$\sigma$) sao padrao.

---

## Historico + Previsao

A visualizacao padrao combina dados historicos e previsao na mesma figura.
O parametro `show_history` controla quantos periodos do historico sao exibidos:

```python
# Mostrar ultimos 60 periodos do historico
fig = plot_forecast(
    forecast,
    show_history=True,
    history_periods=60,
    title="Inflacao - Historico e Projecao",
)
```

### Controlar janela de historico

```python
# Historico completo
fig = plot_forecast(forecast, show_history=True, history_periods=None)

# Ultimos 2 anos (24 meses)
fig = plot_forecast(forecast, show_history=True, history_periods=24)

# Sem historico (apenas previsao)
fig = plot_forecast(forecast, show_history=False)
```

---

## In-Sample Fitted Values

Para avaliar o ajuste do modelo, use `show_fitted=True` para sobrepor
os valores ajustados (in-sample) aos dados observados:

```python
fig = plot_forecast(
    forecast,
    show_fitted=True,
    title="Ajuste In-Sample e Previsao",
)
```

Isso gera tres elementos visuais:

- **Linha preta**: dados observados
- **Linha azul tracejada**: fitted values (in-sample)
- **Linha azul + bandas**: previsao out-of-sample

!!! note "Diagnostico visual"
    Se os fitted values divergem sistematicamente dos dados observados,
    o modelo pode estar mal especificado. Verifique residuos e considere
    reestimar com ordens diferentes.

---

## Previsao Univariada

### ARIMA

```python
from chronobox import ARIMA
from chronobox.visualization import plot_forecast, set_theme

set_theme("professional")

model = ARIMA(order=(2, 1, 1))
results = model.fit(gdp)

forecast = results.forecast(steps=12)

fig = plot_forecast(
    forecast,
    alpha_levels=[0.50, 0.80, 0.95],
    show_history=True,
    history_periods=48,
    colors={"history": "#333333", "forecast": "#1f77b4"},
    title="PIB Real - ARIMA(2,1,1)",
    ylabel="Indice (2010=100)",
)
```

### ETS

```python
from chronobox import ETS
from chronobox.visualization import plot_forecast

model = ETS(error="add", trend="add", seasonal="mul", seasonal_periods=12)
results = model.fit(retail_sales)

forecast = results.forecast(steps=24)

fig = plot_forecast(
    forecast,
    alpha_levels=[0.80, 0.95],
    show_fitted=True,
    title="Vendas no Varejo - ETS(A,A,M)",
)
```

### SARIMA

```python
from chronobox import ARIMA
from chronobox.visualization import plot_forecast

model = ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(airline_passengers)

forecast = results.forecast(steps=36)

fig = plot_forecast(
    forecast,
    alpha_levels=[0.50, 0.80, 0.95],
    show_history=True,
    history_periods=72,
    title="Passageiros Aereos - SARIMA(1,1,1)(1,1,1)[12]",
)
```

---

## Previsao Multivariada

Para modelos VAR/BVAR, o `plot_forecast()` gera um painel com uma previsao
por variavel:

```python
from chronobox import VAR
from chronobox.visualization import plot_forecast

model = VAR(lags=4)
results = model.fit(data[["gdp", "inflation", "interest_rate"]])

forecast = results.forecast(steps=12)

fig = plot_forecast(
    forecast,
    alpha_levels=[0.80, 0.95],
    show_history=True,
    history_periods=40,
    figsize=(14, 10),
    title="Projecao Macroeconomica - VAR(4)",
)
```

### Selecionar variaveis

```python
# Plotar apenas PIB e Inflacao
fig = plot_forecast(
    forecast,
    variables=["gdp", "inflation"],
    figsize=(12, 8),
    title="Projecao PIB e Inflacao",
)
```

### Layout customizado

```python
# Forcar grid 2x2
fig = plot_forecast(
    forecast,
    ncols=2,
    figsize=(14, 8),
    title="Projecao Macro",
)
```

---

## Uso com Arrays NumPy

Para flexibilidade total, forneça arrays diretamente:

```python
import numpy as np
import pandas as pd
from chronobox.visualization import plot_forecast

# Dados observados
history = pd.Series(
    np.cumsum(np.random.randn(100)) + 100,
    index=pd.date_range("2015-01-01", periods=100, freq="MS"),
    name="PIB",
)

# Previsao pontual
forecast_mean = np.array([100.5, 101.2, 101.8, 102.3, 102.7])

# Bandas de confianca
forecast_lower = {
    0.95: forecast_mean - 3.0,
    0.80: forecast_mean - 1.5,
}
forecast_upper = {
    0.95: forecast_mean + 3.0,
    0.80: forecast_mean + 1.5,
}

forecast_index = pd.date_range("2023-05-01", periods=5, freq="MS")

fig = plot_forecast(
    forecast_mean=forecast_mean,
    forecast_lower=forecast_lower,
    forecast_upper=forecast_upper,
    forecast_index=forecast_index,
    history=history,
    title="Previsao a Partir de Arrays",
)
```

---

## Parametros de `plot_forecast`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `forecast_result` | objeto | `None` | Resultado de previsao do modelo |
| `forecast_mean` | ndarray | `None` | Array com previsao pontual |
| `forecast_lower` | dict | `None` | Dict `{nivel: array}` bandas inferiores |
| `forecast_upper` | dict | `None` | Dict `{nivel: array}` bandas superiores |
| `forecast_index` | DatetimeIndex | `None` | Indice temporal da previsao |
| `history` | Series/DataFrame | `None` | Dados historicos (se array direto) |
| `alpha_levels` | list[float] | `[0.80, 0.95]` | Niveis de confianca do fan chart |
| `show_history` | bool | `True` | Exibir historico na figura |
| `history_periods` | int | `None` | Periodos do historico a exibir (`None` = todos) |
| `show_fitted` | bool | `False` | Exibir fitted values (in-sample) |
| `variables` | list[str] | `None` | Variaveis a exibir (multivariado) |
| `colors` | dict | `None` | Dict com cores `{"history", "forecast", "fitted"}` |
| `title` | str | `None` | Titulo geral |
| `ylabel` | str | `None` | Rotulo do eixo Y |
| `figsize` | tuple | auto | Tamanho da figura |
| `ncols` | int | `1` | Colunas no grid (multivariado) |

---

## Opcoes de Customizacao

### Cores e estilo

```python
fig = plot_forecast(
    forecast,
    colors={
        "history": "#2c3e50",
        "forecast": "#e74c3c",
        "fitted": "#3498db",
        "ci_fill": "#e74c3c",
    },
    alpha_levels=[0.50, 0.80, 0.95],
    title="Forecast Customizado",
)
```

### Anotacoes

```python
fig = plot_forecast(forecast, title="Projecao IPCA")

# Adicionar anotacoes pos-criacao
ax = fig.axes[0]
ax.axhline(y=4.5, color="red", linestyle=":", label="Meta de inflacao")
ax.axhspan(3.0, 6.0, alpha=0.05, color="red", label="Banda da meta")
ax.legend()
```

### Eixo secundario

```python
fig = plot_forecast(forecast, title="Taxa de Cambio")

ax = fig.axes[0]
ax2 = ax.twinx()
ax2.bar(
    interest_rate.index, interest_rate.values,
    alpha=0.2, color="gray", label="Selic"
)
ax2.set_ylabel("Selic (%)")
```

---

## Temas e Export

=== "Professional"

    ```python
    set_theme("professional")
    fig = plot_forecast(forecast, title="Projecao Macro")
    fig.savefig("forecast_pro.png", dpi=150, bbox_inches="tight")
    ```

=== "Academic"

    ```python
    set_theme("academic")
    fig = plot_forecast(forecast, title="Projecao Macro")
    fig.savefig("forecast_paper.pdf", bbox_inches="tight")
    # Preto/branco, serif, DPI 300
    ```

=== "Presentation"

    ```python
    set_theme("presentation")
    fig = plot_forecast(
        forecast,
        figsize=(16, 8),
        title="Projecao Macro",
    )
    fig.savefig("forecast_slides.png", dpi=150, bbox_inches="tight")
    ```

=== "BCB"

    ```python
    set_theme("bcb")
    fig = plot_forecast(
        forecast,
        alpha_levels=[0.30, 0.60, 0.90],
        title="Projecao do PIB",
    )
    fig.savefig("forecast_bcb.png", dpi=150, bbox_inches="tight")
    ```

---

## Dicas de Visualizacao

!!! tip "Fan chart para relatorios"
    Use 3 niveis de confianca (ex: 50%, 80%, 95%) para comunicar incerteza
    de forma gradual. Isso e mais informativo do que uma unica banda.

!!! tip "Historico contextual"
    Mostre historico suficiente para dar contexto (2-5 anos), mas nao tanto
    que a previsao fique comprimida. Ajuste `history_periods` conforme
    o horizonte de previsao.

!!! warning "Intervalos em modelos nao-lineares"
    Para modelos com sazonalidade multiplicativa ou transformacoes log,
    as bandas de confianca podem ser assimetricas. O `plot_forecast()`
    respeita a assimetria automaticamente quando disponivel no resultado.
