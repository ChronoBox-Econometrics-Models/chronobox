---
title: Previsao Multivariada
description: Previsao h-passos a frente com intervalos de confianca e fan charts para VAR e VECM.
---

# Previsao Multivariada

!!! info "Quick Reference"
    - **Metodo**: `results.forecast(steps, alpha, method)`
    - **Import**: `from chronobox import VAR` (ou `VECM`)
    - **R equivalente**: `predict(fit, n.ahead)` do `vars`
    - **IC**: Analitico, Bootstrap

---

## Overview

A previsao em modelos VAR e VECM gera **trajetorias conjuntas** para todas as
variaveis do sistema, capturando as interdependencias entre elas. Diferente
de previsoes univariadas independentes, a previsao multivariada leva em conta
como choques em uma variavel afetam as previsoes de todas as outras.

### Quando usar

- Prever multiplas variaveis simultaneamente com consistencia interna
- Gerar cenarios macroeconomicos coerentes
- Construir intervalos de confianca que refletem incerteza conjunta
- Avaliar a incerteza de previsao por fonte de choque

---

## Formulacao Matematica

### Previsao h-passos a frente

Dado um VAR(p) estimado, a previsao condicional para $h$ passos a frente e:

$$
\hat{\mathbf{y}}_{T+h|T} = \hat{\mathbf{c}} + \hat{\mathbf{A}}_1 \hat{\mathbf{y}}_{T+h-1|T} + \cdots + \hat{\mathbf{A}}_p \hat{\mathbf{y}}_{T+h-p|T}
$$

onde $\hat{\mathbf{y}}_{T+j|T} = \mathbf{y}_{T+j}$ para $j \leq 0$ (valores
observados) e $\hat{\mathbf{y}}_{T+j|T}$ para $j > 0$ (previsoes anteriores).

### Erro de Previsao

O erro de previsao $h$-passos a frente e:

$$
\mathbf{e}_{T+h|T} = \mathbf{y}_{T+h} - \hat{\mathbf{y}}_{T+h|T} = \sum_{s=0}^{h-1} \boldsymbol{\Phi}_s \mathbf{u}_{T+h-s}
$$

### Matriz de Covariancia do Erro

$$
\boldsymbol{\Sigma}(h) = \text{Cov}(\mathbf{e}_{T+h|T}) = \sum_{s=0}^{h-1} \boldsymbol{\Phi}_s \boldsymbol{\Sigma}_u \boldsymbol{\Phi}_s'
$$

A incerteza cresce com $h$ --- quanto mais longe a previsao, maior a variancia.

### Intervalos de Confianca Analiticos

Assumindo normalidade, o intervalo de confianca $(1-\alpha)$ para a variavel $i$ e:

$$
\hat{y}_{i,T+h|T} \pm z_{\alpha/2} \sqrt{[\boldsymbol{\Sigma}(h)]_{ii}}
$$

onde $z_{\alpha/2}$ e o quantil da distribuicao normal padrao.

### Intervalos de Confianca Bootstrap

Os IC bootstrap nao assumem normalidade e incorporam incerteza de estimacao:

1. Re-amostrar os residuos $\hat{\mathbf{u}}_t$ com reposicao
2. Reconstruir as series usando os residuos bootstrap
3. Re-estimar o VAR e gerar previsoes
4. Repetir $B$ vezes
5. Extrair percentis $\alpha/2$ e $1-\alpha/2$

---

## Quick Example

```python
from chronobox import VAR
from chronobox.datasets import load_macro

# Ajustar VAR
data = load_macro()
model = VAR(lags=2)
results = model.fit(data)

# Previsao 12 passos a frente
fc = results.forecast(steps=12, alpha=0.05)

# Acessar previsoes
print(fc["forecast"])  # DataFrame com previsoes pontuais
print(fc["lower"])     # Limites inferiores do IC 95%
print(fc["upper"])     # Limites superiores do IC 95%

# Plotar
results.plot_forecast(steps=12, alpha=0.05)
```

---

## Guia Detalhado

### Parametros de Previsao

```python
fc = results.forecast(
    steps=12,           # Horizonte de previsao
    alpha=0.05,         # Nivel de significancia (IC 95%)
    method='analytic',  # 'analytic' ou 'bootstrap'
    n_boot=1000,        # Replicacoes bootstrap
    exog_future=None    # Valores futuros de variaveis exogenas
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `steps` | `int` | --- | Horizonte de previsao |
| `alpha` | `float` | `0.05` | Nivel de significancia ($1-\alpha$ IC) |
| `method` | `str` | `'analytic'` | `'analytic'` ou `'bootstrap'` |
| `n_boot` | `int` | `1000` | Replicacoes bootstrap (se `method='bootstrap'`) |
| `exog_future` | `array-like \| None` | `None` | Exogenas futuras (se modelo tem exogenas) |

### Resultado da Previsao

| Chave | Tipo | Descricao |
|---|---|---|
| `'forecast'` | `pd.DataFrame` | Previsoes pontuais ($h \times K$) |
| `'lower'` | `pd.DataFrame` | Limites inferiores do IC |
| `'upper'` | `pd.DataFrame` | Limites superiores do IC |

### Previsao Analitica vs Bootstrap

=== "Analitico"

    Rapido, assume normalidade dos erros. Nao incorpora incerteza de estimacao
    dos parametros.

    ```python
    fc = results.forecast(steps=12, alpha=0.05, method='analytic')
    ```

=== "Bootstrap"

    Mais lento, mas nao assume normalidade e incorpora incerteza de estimacao.
    Recomendado para amostras pequenas.

    ```python
    fc = results.forecast(
        steps=12,
        alpha=0.05,
        method='bootstrap',
        n_boot=2000
    )
    ```

!!! tip "Quando usar bootstrap"
    Use bootstrap quando:

    - A amostra e pequena ($T < 100$)
    - Os residuos nao sao normais (caudas pesadas, assimetria)
    - Voce quer incorporar incerteza de estimacao dos parametros

### Exemplo Completo: Previsao Macroeconomica

```python
from chronobox import VAR
from chronobox.datasets import load_macro
import pandas as pd

# Dados: GDP, inflacao, taxa de juros (trimestrais)
data = load_macro()

# Dividir em treino e teste
train = data.iloc[:-8]
test = data.iloc[-8:]

# Ajustar VAR
model = VAR(lags='auto', max_lags=8, ic='aic')
results = model.fit(train)
print(f"Lags selecionados: {results.k_ar}")

# Previsao 8 passos a frente
fc = results.forecast(steps=8, alpha=0.05)

# Avaliar
from chronobox.metrics import rmse, mae

for col in data.columns:
    r = rmse(test[col], fc["forecast"][col])
    m = mae(test[col], fc["forecast"][col])
    print(f"{col}: RMSE={r:.4f}, MAE={m:.4f}")
```

### Previsao com VECM

```python
from chronobox import VECM

# VECM gera previsoes em niveis (nao diferencas)
model = VECM(lags=2, coint_rank=1)
results = model.fit(data)

fc = results.forecast(steps=12, alpha=0.05)

# Previsoes sao em niveis
print(fc["forecast"])
```

!!! tip "VECM vs VAR em diferencas"
    O VECM gera previsoes diretamente em **niveis**, enquanto um VAR em
    diferencas gera previsoes de $\Delta y$. Para obter previsoes em niveis
    a partir de um VAR em diferencas, voce precisa acumular as diferencas
    previstas --- o VECM faz isso automaticamente.

### Plotando Previsoes

```python
# Plot com historico e previsao
results.plot_forecast(steps=12, alpha=0.05)

# Plot para uma variavel especifica
results.plot_forecast(steps=12, alpha=0.05, variable='gdp')
```

### Fan Charts

Fan charts mostram multiplos niveis de confianca, dando uma visualizacao
mais completa da incerteza:

```python
# Fan chart com multiplos niveis de confianca
results.plot_forecast(
    steps=12,
    alpha=[0.01, 0.05, 0.10, 0.20],  # IC 99%, 95%, 90%, 80%
    variable='gdp'
)
```

### Previsao com Variaveis Exogenas

```python
import numpy as np

# Modelo com exogenas
crisis_dummy = np.where(
    (data.index >= '2008-09') & (data.index <= '2009-06'), 1, 0
)
model = VAR(lags=2, exog=crisis_dummy)
results = model.fit(data)

# Previsao: precisamos fornecer valores futuros das exogenas
exog_future = np.zeros(12)  # Sem crise no futuro
fc = results.forecast(steps=12, alpha=0.05, exog_future=exog_future)
```

---

## Diagnosticos

### Avaliacao de Previsao

```python
from chronobox.metrics import rmse, mae, mape

# Previsao rolling-window
window = 80
horizons = [1, 4, 8]
errors = {h: {col: [] for col in data.columns} for h in horizons}

for t in range(window, len(data) - max(horizons)):
    train = data.iloc[:t]
    model = VAR(lags=2)
    res = model.fit(train)
    fc = res.forecast(steps=max(horizons))

    for h in horizons:
        for col in data.columns:
            error = data[col].iloc[t + h - 1] - fc["forecast"][col].iloc[h - 1]
            errors[h][col].append(error)

# RMSE por horizonte
for h in horizons:
    print(f"\nHorizonte h={h}:")
    for col in data.columns:
        r = np.sqrt(np.mean(np.array(errors[h][col])**2))
        print(f"  {col}: RMSE = {r:.4f}")
```

### Comparacao com Modelos Univariados

```python
from chronobox import ARIMA

# Benchmark: ARIMA univariado para cada variavel
for col in data.columns:
    # VAR
    fc_var = results.forecast(steps=8)

    # ARIMA
    arima = ARIMA(order=(1, 1, 1))
    arima_res = arima.fit(train[col])
    fc_arima = arima_res.forecast(steps=8)

    rmse_var = rmse(test[col], fc_var["forecast"][col])
    rmse_arima = rmse(test[col], fc_arima["forecast"])

    gain = (rmse_arima - rmse_var) / rmse_arima * 100
    print(f"{col}: VAR={rmse_var:.4f}, ARIMA={rmse_arima:.4f}, ganho={gain:.1f}%")
```

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import VAR

    model = VAR(lags=2)
    results = model.fit(data)

    # Previsao com IC
    fc = results.forecast(steps=12, alpha=0.05)
    print(fc["forecast"])

    # Plot
    results.plot_forecast(steps=12)
    ```

=== "vars (R)"

    ```r
    library(vars)

    fit <- VAR(y, p = 2, type = "const")

    # Previsao
    pred <- predict(fit, n.ahead = 12, ci = 0.95)

    # Acessar previsoes
    pred$fcst$gdp     # Previsao + IC para GDP
    pred$fcst$infl    # Previsao + IC para inflacao

    # Plot
    plot(pred)

    # Fan chart
    fanchart(pred)
    ```

**Mapeamento de parametros**:

| chronobox | vars (R) | Descricao |
|---|---|---|
| `results.forecast(steps=12)` | `predict(fit, n.ahead=12)` | Previsao |
| `alpha=0.05` | `ci=0.95` | Nivel de confianca |
| `fc["forecast"]` | `pred$fcst$var` | Previsoes pontuais |
| `fc["lower"]` / `fc["upper"]` | Colunas lower/upper | IC |
| `results.plot_forecast()` | `plot(pred)` | Plot |
| --- | `fanchart(pred)` | Fan chart |

---

## Referencias

- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
  Capitulo 2.2 (Previsao) e 3.5 (Bootstrap).
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
  Capitulo 11.4.
- Kilian, L. & Lutkepohl, H. (2017). *Structural Vector Autoregressive Analysis*.
  Cambridge University Press.
