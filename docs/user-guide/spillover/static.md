---
title: Static Spillover
description: Spillover table e indices estaticos Diebold-Yilmaz para amostra completa.
---

# Static Spillover

!!! info "Quick Reference"
    - **Classe**: `chronobox.analysis.SpilloverIndex`
    - **Metodo**: `SpilloverIndex.fit(data)`
    - **Resultado**: `SpilloverResult`
    - **Referencia**: Diebold & Yilmaz (2009, 2012)

---

## Overview

O spillover **estatico** calcula os indices de conectividade para a **amostra
completa**, produzindo uma unica spillover table que resume a estrutura de
transmissao de choques entre todas as variaveis do sistema.

O procedimento segue tres etapas:

1. Estimar um VAR(p) para o sistema de $K$ variaveis
2. Calcular a FEVD generalizada (Pesaran-Shin) com horizonte $H$
3. Normalizar a tabela e computar os indices de spillover

---

## Formulacao Matematica

### Spillover Table

A spillover table e a matriz $K \times K$ normalizada da FEVD generalizada:

$$
S_{ij} = \tilde{d}_{ij}^H = \frac{\tilde{\theta}_{ij}^H}{\sum_{k=1}^{K} \tilde{\theta}_{ik}^H} \times 100
$$

onde $\tilde{\theta}_{ij}^H$ e o elemento $(i,j)$ da FEVD generalizada nao
normalizada. Cada linha soma 100%.

A entrada $S_{ij}$ representa a **proporcao da variancia do erro de previsao**
da variavel $i$ que e atribuida a choques na variavel $j$.

### Total Spillover Index (TSI)

O indice total mede a conectividade media do sistema:

$$
\text{TSI}(H) = \frac{\sum_{i=1}^{K} \sum_{\substack{j=1 \\ j \neq i}}^{K} \tilde{d}_{ij}^H}{K} \times 100
$$

Valores altos indicam alta interdependencia; valores baixos indicam
variaveis relativamente isoladas.

### Directional Spillover FROM Others

Quanto a variavel $i$ **recebe** de choques nas demais variaveis:

$$
S_{i \leftarrow \bullet}^H = \frac{\sum_{\substack{j=1 \\ j \neq i}}^{K} \tilde{d}_{ij}^H}{K} \times 100
$$

### Directional Spillover TO Others

Quanto a variavel $j$ **transmite** para as demais:

$$
S_{\bullet \leftarrow j}^H = \frac{\sum_{\substack{i=1 \\ i \neq j}}^{K} \tilde{d}_{ij}^H}{K} \times 100
$$

### Net Spillover

O spillover liquido de cada variavel:

$$
S_j^{\text{net}} = S_{\bullet \leftarrow j}^H - S_{j \leftarrow \bullet}^H
$$

- $S_j^{\text{net}} > 0$: variavel $j$ e **transmissora liquida** de choques
- $S_j^{\text{net}} < 0$: variavel $j$ e **receptora liquida** de choques

### Pairwise Net Spillover

O spillover liquido entre pares especificos:

$$
S_{ij}^{\text{net}} = \frac{\tilde{d}_{ji}^H - \tilde{d}_{ij}^H}{K} \times 100
$$

Note que $S_{ij}^{\text{net}} = -S_{ji}^{\text{net}}$ (anti-simetria).

---

## Quick Example

```python
import numpy as np
import pandas as pd
from chronobox.analysis import SpilloverIndex

# Simular retornos de 4 mercados com dependencia cruzada
np.random.seed(42)
T = 500
e = np.random.randn(T, 4)

# Introduzir dependencia: mercado 0 influencia os demais
data = np.zeros((T, 4))
data[0] = e[0]
for t in range(1, T):
    data[t, 0] = 0.3 * data[t-1, 0] + e[t, 0]
    data[t, 1] = 0.2 * data[t-1, 1] + 0.4 * data[t-1, 0] + e[t, 1]
    data[t, 2] = 0.1 * data[t-1, 2] + 0.3 * data[t-1, 0] + e[t, 2]
    data[t, 3] = 0.2 * data[t-1, 3] + 0.1 * data[t-1, 1] + e[t, 3]

# Calcular spillover estatico
sp = SpilloverIndex(lags=2, horizon=10)
result = sp.fit(data)

# Exibir resultados
print(result.summary())
```

---

## Guia Detalhado

### Construtor

```python
SpilloverIndex(
    lags=2,        # Ordem do VAR subjacente
    horizon=10     # Horizonte da FEVD
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `lags` | `int` | `2` | Ordem do VAR. Deve ser $\geq 1$ |
| `horizon` | `int` | `10` | Horizonte da FEVD generalizada. Deve ser $\geq 1$ |

### Metodo `fit()`

```python
result = sp.fit(data)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `data` | `array_like` | --- | Dados multivariados, shape $(T, K)$ com $K \geq 2$ |

**Retorna**: `SpilloverResult`

### Atributos do `SpilloverResult`

| Atributo | Tipo | Descricao |
|---|---|---|
| `fevd_table` | `ndarray (K, K)` | Tabela GFEVD normalizada (linhas somam 1) |
| `total_spillover` | `float` | Total Spillover Index (0--100) |
| `directional_from` | `ndarray (K,)` | Spillover FROM others por variavel |
| `directional_to` | `ndarray (K,)` | Spillover TO others por variavel |
| `net_spillover` | `ndarray (K,)` | Net spillover (TO - FROM) por variavel |
| `pairwise_spillover` | `ndarray (K, K)` | Pairwise net spillover |
| `horizon` | `int` | Horizonte usado |
| `var_lags` | `int` | Lags do VAR usado |

### Metodos do `SpilloverResult`

| Metodo | Descricao |
|---|---|
| `summary()` | Retorna texto formatado com a tabela completa |
| `plot_table()` | Plota a spillover table como heatmap |

---

## Interpretando a Spillover Table

### Exemplo de Tabela

Considere 3 variaveis (SP500, FTSE, Nikkei):

|  | SP500 | FTSE | Nikkei | **FROM** |
|---|---|---|---|---|
| **SP500** | 65.2 | 20.1 | 14.7 | **11.6** |
| **FTSE** | 25.3 | 55.4 | 19.3 | **14.9** |
| **Nikkei** | 18.6 | 16.2 | 65.2 | **11.6** |
| **TO** | **14.6** | **12.1** | **11.3** |  |
| **NET** | **+3.0** | **-2.8** | **-0.3** |  |

**Leitura**:

- **Diagonal**: contribuicao propria (own variance share). SP500 explica 65.2%
  de sua propria variancia
- **Fora da diagonal**: contribuicao cruzada. FTSE recebe 25.3% de SP500
- **FROM**: media do que cada variavel recebe das demais
- **TO**: media do que cada variavel transmite para as demais
- **NET**: SP500 e transmissor liquido (+3.0), FTSE e receptor (-2.8)
- **Total Spillover**: media dos FROM (ou TO) = $(11.6 + 14.9 + 11.6)/3 \approx 12.7\%$

!!! tip "Interpretacao economica"
    Neste exemplo, SP500 e o principal transmissor de choques, o que e
    consistente com a lideranca do mercado americano nos mercados globais.
    FTSE e o principal receptor, refletindo maior exposicao a choques externos.

### Heatmap da Tabela

```python
# Visualizar como heatmap
result.plot_table()
```

### Acessando Componentes Individuais

```python
# Total spillover
print(f"Total Spillover: {result.total_spillover:.2f}%")

# Spillover direcional
for i in range(len(result.directional_from)):
    print(f"Var {i}: FROM={result.directional_from[i]:.2f}%, "
          f"TO={result.directional_to[i]:.2f}%, "
          f"NET={result.net_spillover[i]:.2f}%")

# Pairwise: quanto Var 0 transmite para Var 1 (liquido)
print(f"Pairwise (0->1): {result.pairwise_spillover[0, 1]:.2f}%")
```

---

## Exemplo Completo: Mercados Financeiros

```python
import numpy as np
import pandas as pd
from chronobox.analysis import SpilloverIndex

# Simular retornos diarios de 4 indices
np.random.seed(123)
T = 1000
K = 4
labels = ['SP500', 'FTSE', 'Nikkei', 'DAX']

# Gerar dados com estrutura de dependencia
e = np.random.randn(T, K)
data = np.zeros((T, K))
data[0] = e[0]
for t in range(1, T):
    data[t, 0] = 0.05 * data[t-1, 0] + e[t, 0]
    data[t, 1] = 0.03 * data[t-1, 1] + 0.25 * data[t-1, 0] + e[t, 1]
    data[t, 2] = 0.04 * data[t-1, 2] + 0.15 * data[t-1, 0] + 0.10 * data[t-1, 1] + e[t, 2]
    data[t, 3] = 0.02 * data[t-1, 3] + 0.20 * data[t-1, 0] + 0.12 * data[t-1, 1] + e[t, 3]

df = pd.DataFrame(data, columns=labels)

# --- Spillover estatico ---
sp = SpilloverIndex(lags=2, horizon=10)
result = sp.fit(df.values)

# Resumo completo
print(result.summary())

# Identificar transmissores e receptores
for i, label in enumerate(labels):
    role = "Transmissor" if result.net_spillover[i] > 0 else "Receptor"
    print(f"{label}: NET = {result.net_spillover[i]:+.2f}% ({role})")

# Heatmap
result.plot_table()
```

---

## Escolha de Parametros

### Lags do VAR

| Criterio | Recomendacao |
|---|---|
| Dados diarios | `lags=2` a `lags=5` |
| Dados mensais | `lags=1` a `lags=4` |
| Dados trimestrais | `lags=1` a `lags=2` |

!!! tip "Selecao de lags"
    Use criterios de informacao (AIC, BIC) para selecionar a ordem do VAR
    antes de calcular o spillover. Uma boa pratica e estimar o VAR separadamente
    com `chronobox.VAR(lags='auto')` e usar o numero de lags selecionado.

### Horizonte da FEVD

O horizonte $H$ controla ate que ponto no futuro os choques sao rastreados:

- $H = 10$: padrao na literatura, captura efeitos de curto/medio prazo
- $H$ pequeno ($\leq 5$): foco em transmissao imediata
- $H$ grande ($\geq 20$): captura efeitos de longo prazo, mas pode
  introduzir ruido

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox.analysis import SpilloverIndex

    sp = SpilloverIndex(lags=2, horizon=10)
    result = sp.fit(data)

    print(result.summary())
    print(f"Total: {result.total_spillover:.2f}%")
    result.plot_table()
    ```

=== "frequencyConnectedness (R)"

    ```r
    library(frequencyConnectedness)
    library(vars)

    # Estimar VAR
    fit <- VAR(data, p = 2)

    # Spillover Diebold-Yilmaz (2012)
    sp <- spilloverDY12(fit, n.ahead = 10)
    print(sp)

    # Ou Diebold-Yilmaz (2009) com Cholesky
    sp09 <- spilloverDY09(fit, n.ahead = 10)
    ```

**Mapeamento**:

| chronobox | frequencyConnectedness (R) | Descricao |
|---|---|---|
| `SpilloverIndex(lags=2, horizon=10)` | `VAR(data, p=2)` + `spilloverDY12(fit, n.ahead=10)` | Setup + calculo |
| `result.total_spillover` | `overall(sp)` | Total Spillover Index |
| `result.directional_from` | `from(sp)` | Directional FROM |
| `result.directional_to` | `to(sp)` | Directional TO |
| `result.net_spillover` | `net(sp)` | Net spillover |
| `result.plot_table()` | `plot(sp)` | Heatmap |

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
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
  Springer.
