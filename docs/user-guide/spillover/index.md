---
title: Spillover Analysis
description: Framework Diebold-Yilmaz para medir transmissao de choques e conectividade entre variaveis.
---

# Spillover Analysis

!!! info "Quick Reference"
    - **Classe**: `chronobox.analysis.SpilloverIndex`
    - **Import**: `from chronobox.analysis import SpilloverIndex`
    - **R equivalente**: `frequencyConnectedness::spilloverDY12()`
    - **Base teorica**: Diebold & Yilmaz (2009, 2012)

---

## Overview

Spillover analysis mede a **transmissao de choques** entre variaveis em um
sistema multivariado. A pergunta central e: quando ocorre um choque inesperado
em uma variavel, quanto desse choque se propaga para as demais?

O framework de Diebold & Yilmaz (2009, 2012) utiliza a **Forecast Error
Variance Decomposition (FEVD) generalizada** de Pesaran & Shin (1998) para
construir uma tabela de conectividade e derivar indices de spillover em
multiplas dimensoes.

### Aplicacoes

- Medir **contaminacao entre mercados financeiros** (acoes, cambio, commodities)
- Identificar **transmissao de volatilidade** em periodos de crise
- Analisar **conectividade macroeconomica** entre paises ou setores
- Monitorar **risco sistemico** via evolucao temporal da conectividade

---

## Framework Diebold-Yilmaz

### Ideia Central

A partir de um VAR(p) estimado para $K$ variaveis, calcula-se a FEVD
generalizada que mede a contribuicao de cada variavel $j$ na variancia do
erro de previsao da variavel $i$:

$$
\tilde{\theta}_{ij}^H = \frac{\sigma_{jj}^{-1} \sum_{h=0}^{H-1} (\mathbf{e}_i' \boldsymbol{\Phi}_h \boldsymbol{\Sigma} \mathbf{e}_j)^2}{\sum_{h=0}^{H-1} \mathbf{e}_i' \boldsymbol{\Phi}_h \boldsymbol{\Sigma} \boldsymbol{\Phi}_h' \mathbf{e}_i}
$$

onde $\boldsymbol{\Phi}_h$ sao os coeficientes VMA, $\boldsymbol{\Sigma}$ e a
matriz de covariancia dos residuos, e $H$ e o horizonte de previsao.

### Normalizacao

Como a FEVD generalizada nao soma 1 nas linhas (ao contrario da Cholesky),
normaliza-se:

$$
\tilde{d}_{ij}^H = \frac{\tilde{\theta}_{ij}^H}{\sum_{j=1}^{K} \tilde{\theta}_{ij}^H}
$$

A tabela normalizada $[\tilde{d}_{ij}^H]$ e a **spillover table**.

### Indices de Spillover

A partir da tabela, derivam-se quatro indices:

| Indice | Formula | Interpretacao |
|---|---|---|
| **Total Spillover** | $S^H = \frac{\sum_{i \neq j} \tilde{d}_{ij}^H}{K} \times 100$ | Conectividade media do sistema |
| **Directional FROM** | $S_{i \leftarrow}^H = \frac{\sum_{j \neq i} \tilde{d}_{ij}^H}{K} \times 100$ | Quanto a variavel $i$ **recebe** das demais |
| **Directional TO** | $S_{\rightarrow j}^H = \frac{\sum_{i \neq j} \tilde{d}_{ij}^H}{K} \times 100$ | Quanto a variavel $j$ **transmite** para as demais |
| **Net Spillover** | $S_j^H = S_{\rightarrow j}^H - S_{j \leftarrow}^H$ | Positivo = transmissor liquido |

!!! tip "Vantagem da FEVD generalizada"
    Ao contrario da decomposicao de Cholesky, a FEVD generalizada de
    Pesaran-Shin **nao depende da ordenacao das variaveis**, o que e essencial
    para a construcao de indices de spillover robustos.

---

## Quick Example

```python
import numpy as np
from chronobox.analysis import SpilloverIndex

# Dados simulados: 4 mercados financeiros
np.random.seed(42)
T, K = 500, 4
data = np.random.randn(T, K)

# Spillover estatico
sp = SpilloverIndex(lags=2, horizon=10)
result = sp.fit(data)

print(f"Total Spillover: {result.total_spillover:.2f}%")
print(result.summary())

# Spillover dinamico (rolling window)
rolling = sp.rolling(data, window=200)
rolling.plot_total()
```

---

## Paginas da Secao

| Pagina | Conteudo |
|---|---|
| [Static Spillover](static.md) | Spillover table, indices estaticos, interpretacao |
| [Dynamic Spillover](dynamic.md) | Rolling window, evolucao temporal, network plots |

---

## Quando Usar

- Avaliar **grau de interconexao** entre mercados ou variaveis
- Identificar **canais de transmissao** de choques
- Monitorar **evolucao da conectividade** ao longo do tempo
- Comparar **periodos de crise vs. tranquilidade**
- Detectar **transmissores e receptores liquidos** de choques

!!! warning "Pre-requisitos"
    1. As variaveis devem ser **estacionarias** (ou transformadas para tal)
    2. O sistema deve ter ao menos **2 variaveis** ($K \geq 2$)
    3. A amostra deve ser suficientemente grande para estimar o VAR subjacente
    4. Para rolling spillover, o tamanho da janela deve exceder $p + H + K$

---

## Referencias

- Diebold, F. X. & Yilmaz, K. (2009). Measuring Financial Asset Return and
  Volatility Spillovers, with Application to Global Equity Markets. *Economic
  Journal*, 119, 158--171.
- Diebold, F. X. & Yilmaz, K. (2012). Better to Give than to Receive:
  Predictive Directional Measurement of Volatility Spillovers. *International
  Journal of Forecasting*, 28(1), 57--66.
- Pesaran, M. H. & Shin, Y. (1998). Generalized Impulse Response Analysis in
  Linear Multivariate Models. *Economics Letters*, 58(1), 17--29.
