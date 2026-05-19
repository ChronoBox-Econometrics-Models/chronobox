---
title: IRF Plots
description: Graficos de funcoes impulso-resposta --- grid KxK, IRF individual, acumulada e com intervalos de confianca.
---

# IRF Plots

!!! info "Quick Reference"
    - **Funcao**: `plot_irf()`
    - **Import**: `from chronobox.visualization import plot_irf`
    - **Input**: Objeto de resultado IRF ou arrays NumPy
    - **Output**: `matplotlib.figure.Figure` com grid KxK

---

## Overview

O grafico de Impulse Response Functions (IRF) mostra como cada variavel do
sistema reage a um choque (impulso) em cada uma das outras variaveis. E a
principal ferramenta para **interpretar modelos VAR e SVAR**.

O `plot_irf()` gera automaticamente um grid de $K \times K$ subplots, onde
cada celula $(i, j)$ mostra a **resposta da variavel $i$** a um **choque na
variavel $j$** ao longo do horizonte de previsao.

### Quando usar

- Visualizar transmissao de choques entre variaveis
- Comparar velocidade e magnitude de respostas
- Apresentar resultados de modelos VAR/SVAR
- Verificar convergencia do sistema (respostas devem convergir a zero)

---

## Quick Start

```python
from chronobox import VAR
from chronobox.visualization import plot_irf, set_theme

set_theme("professional")

# Estimar VAR
model = VAR(lags=4)
results = model.fit(data)

# Calcular IRF
irf = results.irf(steps=20)

# Grid completo KxK
fig = plot_irf(irf)
fig.savefig("irf_grid.png", dpi=150, bbox_inches="tight")
```

---

## Grid Completo (KxK)

O grid completo mostra todas as combinacoes impulso-resposta:

```python
from chronobox.visualization import plot_irf

fig = plot_irf(
    irf,
    title="Impulse Response Functions",
    figsize=(16, 12),
)
```

Cada subplot mostra:

- **Linha azul**: resposta pontual
- **Area sombreada**: intervalo de confianca (bootstrap)
- **Linha tracejada**: zero (referencia)
- **Titulo da coluna**: variavel de impulso (choque)
- **Titulo da linha**: variavel de resposta

---

## IRF Individual

Para focar em uma relacao especifica, use os parametros `impulse` e `response`:

```python
# Resposta do PIB a um choque na Selic
fig = plot_irf(
    irf,
    impulse="selic",
    response="gdp",
    title="Resposta do PIB a choque na Selic",
)
```

### Fixar apenas o impulso

```python
# Todas as respostas a um choque na Selic
fig = plot_irf(
    irf,
    impulse="selic",
    title="Choque na Selic: Todas as Respostas",
)
```

### Fixar apenas a resposta

```python
# Todas as fontes de choque sobre o PIB
fig = plot_irf(
    irf,
    response="gdp",
    title="PIB: Todas as Fontes de Choque",
)
```

---

## IRF Acumulada

A IRF acumulada mostra o efeito **total acumulado** ate cada horizonte.
E especialmente util quando a variavel esta em diferencas (ex: crescimento do PIB)
e se deseja ver o efeito sobre o **nivel**:

$$
\text{CIRF}(h) = \sum_{s=0}^{h} \boldsymbol{\Psi}_s
$$

```python
fig = plot_irf(
    irf,
    cumulative=True,
    title="Cumulative Impulse Response Functions",
)
```

!!! tip "Quando usar IRF acumulada"
    - Variaveis em primeira diferenca: a IRF acumulada recupera o efeito sobre o nivel
    - Analise de multiplicadores: efeito total acumulado de um choque
    - Se a variavel ja esta em nivel: a IRF regular e suficiente

---

## Intervalos de Confianca

Os intervalos de confianca sao gerados via **bootstrap** e aparecem como
bandas sombreadas ao redor da IRF pontual:

```python
# IRF com bootstrap (calculado durante irf())
irf = results.irf(steps=20, method="cholesky")

fig = plot_irf(
    irf,
    sigs=0.95,  # 95% de confianca
    title="IRF com IC 95%",
)
```

O parametro `sigs` controla o nivel de confianca (default: 0.95).

!!! note "Interpretacao das bandas"
    Se a banda de confianca **inclui o zero** em todo o horizonte, o efeito
    nao e estatisticamente significativo ao nivel escolhido. Se a banda
    **exclui o zero** em algum intervalo, o efeito e significativo naquele
    horizonte.

---

## Uso com Arrays NumPy

Para flexibilidade total, voce pode fornecer arrays diretamente:

```python
import numpy as np
from chronobox.visualization import plot_irf

# IRF array: shape (H, K, K)
# irf_array[h, i, j] = resposta de var i a choque em var j no horizonte h
irf_array = np.random.randn(20, 3, 3) * 0.1
irf_array = np.cumsum(irf_array, axis=0) * 0.3  # decaimento

# Bandas de confianca (opcional)
irf_lower = irf_array - 0.5
irf_upper = irf_array + 0.5

fig = plot_irf(
    irf_array=irf_array,
    irf_lower=irf_lower,
    irf_upper=irf_upper,
    var_names=["PIB", "Inflacao", "Selic"],
    periods=15,
    title="IRF a Partir de Arrays",
)
```

---

## Parametros de `plot_irf`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `irf_results` | objeto | `None` | Resultado IRF com arrays e nomes de variaveis |
| `irf_array` | ndarray (H,K,K) | `None` | Array IRF direto |
| `irf_lower` | ndarray (H,K,K) | `None` | Banda inferior de confianca |
| `irf_upper` | ndarray (H,K,K) | `None` | Banda superior de confianca |
| `var_names` | list[str] | `None` | Nomes das variaveis |
| `impulse` | str ou int | `None` | Filtrar por variavel de impulso |
| `response` | str ou int | `None` | Filtrar por variavel de resposta |
| `cumulative` | bool | `False` | IRF acumulada |
| `sigs` | float | `0.95` | Nivel de confianca |
| `periods` | int | `None` | Numero de horizontes a plotar |
| `title` | str | `None` | Titulo geral |
| `figsize` | tuple | auto | Tamanho da figura |

---

## Exemplos com SVAR

O SVAR usa restricoes de identificacao para obter choques estruturais.
A IRF estrutural e especialmente informativa:

### Cholesky (Recursivo)

```python
from chronobox import VAR
from chronobox.visualization import plot_irf

model = VAR(lags=4)
results = model.fit(data[["gdp", "inflation", "interest_rate"]])

# IRF com identificacao de Cholesky
irf = results.irf(steps=24, method="cholesky")

fig = plot_irf(
    irf,
    title="IRF Ortogonal (Cholesky)",
    figsize=(14, 10),
)
```

!!! warning "Ordenacao importa"
    Na identificacao de Cholesky, a **ordenacao das variaveis** afeta os
    resultados. A variavel mais "exogena" deve vir primeiro.

### SVAR com Restricoes

```python
from chronobox import SVAR
import numpy as np

# Restricoes de curto prazo (matriz B)
B = np.array([
    [np.nan, 0,      0     ],
    [np.nan, np.nan, 0     ],
    [np.nan, np.nan, np.nan],
])

model = SVAR(lags=4, svar_type="B", B=B)
results = model.fit(data)

irf = results.irf(steps=24)

fig = plot_irf(
    irf,
    title="IRF Estrutural (SVAR)",
    figsize=(14, 10),
)
```

---

## Dicas de Visualizacao

### Ajustar horizontes

```python
# Menos horizontes para efeitos de curto prazo
fig = plot_irf(irf, periods=10, title="Curto Prazo (10 periodos)")

# Mais horizontes para convergencia
fig = plot_irf(irf, periods=40, title="Longo Prazo (40 periodos)")
```

### Salvar em alta resolucao

```python
fig = plot_irf(irf, figsize=(16, 12))
fig.savefig("irf_hires.pdf", bbox_inches="tight")  # vetorial
fig.savefig("irf_hires.png", dpi=300, bbox_inches="tight")  # raster
```

### Tema academico para papers

```python
from chronobox.visualization import set_theme

set_theme("academic")
fig = plot_irf(irf, title="Impulse Response Functions")
fig.savefig("irf_paper.pdf", bbox_inches="tight")
# Preto/branco, serif, DPI 300
```
