---
title: FEVD
description: Decomposicao da variancia do erro de previsao --- ortogonal e generalizada.
---

# Forecast Error Variance Decomposition (FEVD)

!!! info "Quick Reference"
    - **Metodo**: `results.fevd(steps, method)`
    - **Import**: `from chronobox import VAR` (ou `VECM`)
    - **R equivalente**: `vars::fevd()`
    - **Tipos**: Ortogonal (Cholesky), Generalizada

---

## Overview

A FEVD (Forecast Error Variance Decomposition) responde a pergunta:
**qual fracao da variancia do erro de previsao de cada variavel e atribuivel a
choques em cada uma das variaveis do sistema?**

Enquanto a IRF mostra *como* um choque se propaga, a FEVD mostra a
**importancia relativa** de cada fonte de choque para explicar a incerteza
de previsao de cada variavel.

### Quando usar

- Quantificar a importancia relativa de cada choque para cada variavel
- Identificar variaveis "dirigidas externamente" vs "auto-determinadas"
- Complementar a analise de IRF com uma medida de importancia
- Entender a estrutura de interdependencia do sistema

---

## Formulacao Matematica

### FEVD Ortogonal

A partir da representacao VMA com choques ortogonais ($\boldsymbol{\Psi}_h = \boldsymbol{\Phi}_h \mathbf{P}$),
o erro de previsao $h$-passos a frente e:

$$
\mathbf{y}_{T+h} - \hat{\mathbf{y}}_{T+h|T} = \sum_{s=0}^{h-1} \boldsymbol{\Psi}_s \boldsymbol{\varepsilon}_{T+h-s}
$$

A variancia do erro de previsao da variavel $i$ no horizonte $h$ e:

$$
\text{MSE}_i(h) = \sum_{s=0}^{h-1} \sum_{j=1}^{K} (\boldsymbol{\Psi}_s)_{ij}^2
$$

A contribuicao do choque $j$ para a variancia da variavel $i$ e:

$$
\theta_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\boldsymbol{\Psi}_s)_{ij}^2}{\text{MSE}_i(h)}
$$

Por construcao, $\sum_{j=1}^{K} \theta_{ij}(h) = 1$ para todo $i$ e $h$.

### FEVD Generalizada

A FEVD generalizada (baseada na GIRF de Pesaran-Shin) nao depende da ordenacao:

$$
\tilde{\theta}_{ij}(h) = \frac{\sigma_{jj}^{-1} \sum_{s=0}^{h-1} (\mathbf{e}_i' \boldsymbol{\Phi}_s \boldsymbol{\Sigma}_u \mathbf{e}_j)^2}{\sum_{s=0}^{h-1} \mathbf{e}_i' \boldsymbol{\Phi}_s \boldsymbol{\Sigma}_u \boldsymbol{\Phi}_s' \mathbf{e}_i}
$$

!!! warning "FEVD generalizada: linhas nao somam 1"
    Na FEVD generalizada, $\sum_{j} \tilde{\theta}_{ij}(h) \neq 1$ em geral
    (os choques nao sao ortogonais). Para interpretar como proporcoes,
    normalize cada linha:

    $$\theta_{ij}^*(h) = \frac{\tilde{\theta}_{ij}(h)}{\sum_{j=1}^{K} \tilde{\theta}_{ij}(h)}$$

### Propriedades

| Propriedade | FEVD Ortogonal | FEVD Generalizada |
|---|---|---|
| Soma das linhas | $= 1$ (exato) | $\neq 1$ (normalizar) |
| Depende da ordenacao | Sim | Nao |
| Choques ortogonais | Sim | Nao |
| Base teorica | Cholesky | Pesaran-Shin (1998) |

---

## Quick Example

```python
from chronobox import VAR
from chronobox.datasets import load_macro

# Ajustar VAR
data = load_macro()
model = VAR(lags=2)
results = model.fit(data)

# FEVD ortogonal, 20 periodos
fevd = results.fevd(steps=20, method='cholesky')

# FEVD generalizada
gfevd = results.fevd(steps=20, method='generalized')

# Resumo
print(fevd.summary())

# Plot
fevd.plot()
```

---

## Guia Detalhado

### Parametros da FEVD

```python
fevd = results.fevd(
    steps=20,            # Horizonte
    method='cholesky'    # 'cholesky' ou 'generalized'
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `steps` | `int` | `20` | Horizonte da decomposicao |
| `method` | `str` | `'cholesky'` | `'cholesky'` ou `'generalized'` |

### Interpretando a FEVD

```python
fevd = results.fevd(steps=20, method='cholesky')
print(fevd.summary())
```

```text
FEVD for gdp (h = 1, 5, 10, 20):
  h     gdp     infl     rate
  1   100.00     0.00     0.00
  5    82.34    12.41     5.25
 10    74.12    17.23     8.65
 20    71.89    18.94     9.17

FEVD for infl (h = 1, 5, 10, 20):
  h     gdp     infl     rate
  1     3.21    96.79     0.00
  5    11.45    78.32    10.23
 10    15.67    69.44    14.89
 20    16.23    68.12    15.65

FEVD for rate (h = 1, 5, 10, 20):
  h     gdp     infl     rate
  1     1.12     5.43    93.45
  5     8.34    14.23    77.43
 10    12.56    18.89    68.55
 20    13.21    19.78    67.01
```

**Como interpretar**:

- No horizonte $h=1$, 100% da variancia do GDP e explicada por choques proprios
  (por construcao, se GDP e a primeira variavel na ordenacao Cholesky)
- No horizonte $h=20$, choques na inflacao explicam ~19% da variancia do GDP
- A taxa de juros e majoritariamente auto-determinada (67% no $h=20$)
- Choques no GDP explicam ~16% da variancia da inflacao no longo prazo

### Acessando Valores

```python
# FEVD como array (steps x K x K)
# fevd.decomp[h, i, j] = contribuicao do choque j na variavel i no horizonte h
print(fevd.decomp[10, 0, 2])  # Contribuicao de 'rate' em 'gdp' no h=10

# Como DataFrame
df = fevd.to_dataframe('gdp')
print(df)
```

### Exemplo: Dominancia de Choques

```python
from chronobox import VAR
from chronobox.datasets import load_macro

data = load_macro()
model = VAR(lags=4)
results = model.fit(data[["gdp", "infl", "rate"]])

# FEVD de longo prazo
fevd = results.fevd(steps=40, method='cholesky')

# Identificar o choque dominante para cada variavel
for var in data.columns:
    df = fevd.to_dataframe(var)
    last = df.iloc[-1]
    dominant = last.idxmax()
    print(f"{var}: choque dominante = {dominant} ({last[dominant]:.1f}%)")
```

### Plotando a FEVD

```python
# Plot stacked bars para todas as variaveis
fevd.plot()

# Plot para uma variavel especifica
fevd.plot(variable='gdp')

# Plot stacked area
fevd.plot(kind='area')
```

### Comparando Ortogonal vs Generalizada

```python
# FEVD ortogonal (depende da ordenacao)
fevd_orth = results.fevd(steps=20, method='cholesky')

# FEVD generalizada (invariante a ordenacao)
fevd_gen = results.fevd(steps=20, method='generalized')

# Comparar contribuicoes no horizonte h=20
print("Ortogonal:")
print(fevd_orth.to_dataframe('gdp').iloc[-1])

print("\nGeneralizada (normalizada):")
print(fevd_gen.to_dataframe('gdp').iloc[-1])
```

!!! tip "Spillover analysis"
    A FEVD generalizada e a base da analise de spillover de
    Diebold & Yilmaz (2012). Os elementos fora da diagonal de
    $\boldsymbol{\Theta}^*(h)$ medem spillovers entre variaveis.

---

## Diagnosticos

### Estabilidade da FEVD

A FEVD deve estabilizar (convergir) a medida que $h$ aumenta:

```python
# Verificar convergencia
for var in data.columns:
    df = fevd.to_dataframe(var)
    diff = df.iloc[-1] - df.iloc[-5]
    print(f"{var}: variacao ultimos 5 periodos = {diff.abs().sum():.4f}")
    # Valores proximos de zero indicam convergencia
```

### Sensibilidade a Ordenacao

```python
import pandas as pd

# Testar diferentes ordenacoes
orderings = [
    ["gdp", "infl", "rate"],
    ["rate", "infl", "gdp"],
    ["infl", "gdp", "rate"],
]

results_list = []
for ordering in orderings:
    res = model.fit(data[ordering])
    f = res.fevd(steps=20, method='cholesky')
    results_list.append(f.to_dataframe('gdp').iloc[-1])

comparison = pd.DataFrame(results_list, index=[str(o) for o in orderings])
print(comparison)
```

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import VAR

    model = VAR(lags=2)
    results = model.fit(data)

    fevd = results.fevd(steps=20, method='cholesky')
    fevd.plot()
    print(fevd.summary())
    ```

=== "vars (R)"

    ```r
    library(vars)

    fit <- VAR(y, p = 2, type = "const")

    # FEVD
    fv <- fevd(fit, n.ahead = 20)
    summary(fv)
    plot(fv)

    # Acessar valores
    fv$gdp   # FEVD do GDP (data.frame)
    ```

**Mapeamento de parametros**:

| chronobox | vars (R) | Descricao |
|---|---|---|
| `results.fevd(steps=20)` | `fevd(fit, n.ahead=20)` | FEVD ortogonal |
| `fevd.plot()` | `plot(fv)` | Plot stacked bars |
| `fevd.to_dataframe('gdp')` | `fv$gdp` | Valores para uma variavel |

---

## Referencias

- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
  Capitulo 2.3.3.
- Pesaran, M. H. & Shin, Y. (1998). Generalized Impulse Response Analysis in
  Linear Multivariate Models. *Economics Letters*, 58(1), 17--29.
- Diebold, F. X. & Yilmaz, K. (2012). Better to Give than to Receive:
  Predictive Directional Measurement of Volatility Spillovers.
  *International Journal of Forecasting*, 28(1), 57--66.
- Kilian, L. & Lutkepohl, H. (2017). *Structural Vector Autoregressive Analysis*.
  Cambridge University Press.
