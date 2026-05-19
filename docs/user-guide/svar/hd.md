---
title: Historical Decomposition
description: Atribuicao de movimentos historicos a choques estruturais individuais e analise contrafactual.
---

# Historical Decomposition

!!! info "Quick Reference"
    - **Metodo**: `svar_results.historical_decomposition()`
    - **Import**: `from chronobox import SVAR`
    - **R equivalente**: `svars::hd()`, `vars::historical_decomposition()`
    - **Requisito**: modelo SVAR identificado

---

## Overview

A Historical Decomposition (HD) decompoe os **movimentos observados** de cada
variavel em contribuicoes de cada **choque estrutural** identificado. Enquanto
a IRF mostra o efeito tipico de um choque, a HD mostra **quanto de cada variacao
historica** se deve a cada tipo de choque.

Aplicacoes classicas:

- Quanto da recessao de 2008 se deve a choques financeiros vs. choques de demanda?
- Quanto da inflacao alta de 2022 se deve a choques de oferta vs. politica monetaria?
- Qual choque foi o principal motor do ciclo economico em um determinado periodo?

A HD tambem permite **analise contrafactual**: o que teria acontecido com o
PIB se nao houvesse choques monetarios?

### Quando usar

- Atribuir movimentos historicos a choques economicos especificos
- Explicar episodios economicos (recessoes, crises, booms)
- Analise contrafactual (cenarios "what if")
- Comunicacao de resultados de politica economica
- Complemento a IRFs e FEVDs

---

## Formulacao Matematica

### Representacao MA Estrutural

O VAR estrutural identificado pode ser escrito na forma de medias moveis (MA):

$$
\mathbf{y}_t = \boldsymbol{\mu}_t + \sum_{s=0}^{t-1} \boldsymbol{\Psi}_s \mathbf{e}_{t-s}
$$

onde:

- $\boldsymbol{\mu}_t$ e o componente deterministico (constante, tendencia, condicoes iniciais)
- $\boldsymbol{\Psi}_s = \mathbf{J} \mathbf{F}^s \mathbf{J}' \mathbf{A}_0^{-1} \mathbf{B}_0$ sao as matrizes de impulso-resposta estrutural
- $\mathbf{e}_{t-s}$ sao os choques estruturais no periodo $t-s$
- $\mathbf{F}$ e a companion matrix

### Decomposicao por Choque

A contribuicao do choque $j$ para a variavel $i$ no periodo $t$ e:

$$
\text{HD}_{ij}(t) = \sum_{s=0}^{t-1} \psi_{ij,s} \cdot e_{j,t-s}
$$

onde $\psi_{ij,s}$ e o elemento $(i,j)$ de $\boldsymbol{\Psi}_s$.

A decomposicao completa satisfaz:

$$
y_{it} - \mu_{it} = \sum_{j=1}^{K} \text{HD}_{ij}(t)
$$

ou seja, a soma das contribuicoes de todos os choques reconstroi perfeitamente
o desvio da variavel em relacao ao seu componente deterministico.

### Analise Contrafactual

O valor contrafactual de $y_{it}$ **excluindo** o choque $j$ e:

$$
y_{it}^{(\text{no } j)} = y_{it} - \text{HD}_{ij}(t) = \mu_{it} + \sum_{k \neq j} \text{HD}_{ik}(t)
$$

Generalizando para excluir multiplos choques $\mathcal{J}$:

$$
y_{it}^{(\text{no } \mathcal{J})} = y_{it} - \sum_{j \in \mathcal{J}} \text{HD}_{ij}(t)
$$

---

## Quick Example

```python
from chronobox import VAR, SVAR
from chronobox.datasets import load_macro

# Estimar SVAR
data = load_macro()
var_results = VAR(lags=4).fit(data)
svar_results = SVAR(var_results, identification="cholesky").identify()

# Historical Decomposition
hd = svar_results.historical_decomposition()

# Contribuicao de cada choque para o GDP
print(hd.table("gdp"))

# Counterfactual: GDP sem choques de juros
cf = svar_results.counterfactual(
    variable="gdp",
    exclude_shocks=["interest_rate"],
)
print(cf)
```

---

## Guia Detalhado

### Metodo `historical_decomposition()`

```python
hd = svar_results.historical_decomposition(
    start=None,          # Data inicial (None = inicio da amostra)
    end=None,            # Data final (None = fim da amostra)
    cumulative=False,    # Contribuicoes acumuladas
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `start` | `str \| None` | `None` | Data inicial para a decomposicao |
| `end` | `str \| None` | `None` | Data final |
| `cumulative` | `bool` | `False` | Se `True`, acumula contribuicoes ao longo do tempo |

### Acessando Resultados

```python
hd = svar_results.historical_decomposition()

# Contribuicao de cada choque para cada variavel
print(hd.contributions)
# Dict[str, DataFrame]: {variavel: DataFrame com colunas = choques}

# Contribuicao dos choques para GDP
gdp_hd = hd.contributions["gdp"]
print(gdp_hd.head())
```

```text
            e_gdp    e_infl   e_rate   deterministic
2000-01     0.0023  -0.0012   0.0005   0.0132
2000-02     0.0045   0.0008  -0.0031   0.0128
2000-03    -0.0067   0.0034  -0.0018   0.0135
...
```

### Analise de um Episodio Especifico

```python
# Recessao de 2008-2009
hd_crisis = svar_results.historical_decomposition(
    start="2007-01",
    end="2010-12",
)

# Contribuicao de cada choque para a queda do GDP
gdp_crisis = hd_crisis.contributions["gdp"]
print(gdp_crisis.loc["2008-09":"2009-06"].mean())
```

```text
e_gdp          -0.0234
e_inflation     0.0012
e_interest_rate 0.0089
deterministic   0.0131
dtype: float64
```

O choque de GDP proprio (demanda) foi o principal responsavel pela queda,
parcialmente compensado pelo relaxamento monetario (juros positivo = expansionista).

### Analise Contrafactual

```python
# O que teria acontecido sem choques monetarios?
cf = svar_results.counterfactual(
    variable="gdp",
    exclude_shocks=["interest_rate"],
)

print(cf.head())
```

```text
            observed    counterfactual   difference
2000-01     0.0148      0.0143           0.0005
2000-02     0.0150      0.0181          -0.0031
...
```

### Contrafactual com Multiplos Choques

```python
# GDP sem choques de inflacao nem de juros
cf = svar_results.counterfactual(
    variable="gdp",
    exclude_shocks=["inflation", "interest_rate"],
)
```

### Contribuicoes Acumuladas

```python
# Decomposicao acumulada (util para series em nivel)
hd_cum = svar_results.historical_decomposition(cumulative=True)
print(hd_cum.contributions["gdp"].tail())
```

---

## Interpretacao

### Lendo o Grafico de Barras Empilhadas

A visualizacao padrao da HD e um grafico de barras empilhadas (stacked bar chart):

```python
# Barras empilhadas: contribuicao de cada choque para GDP
hd.plot("gdp")
```

- **Barras positivas**: choques que empurraram o GDP para cima
- **Barras negativas**: choques que empurraram o GDP para baixo
- **Linha preta**: valor observado (desvio do deterministico)
- A soma das barras reproduz a linha preta em cada periodo

### Dominancia de Choques

```python
# Qual choque dominou cada periodo?
dominant = hd.dominant_shock("gdp")
print(dominant.value_counts())
```

```text
e_gdp              68
e_interest_rate     32
e_inflation         20
Name: dominant_shock, dtype: int64
```

### Periodo de Crise vs Normal

```python
# Comparar composicao em crise vs. periodos normais
crisis = gdp_hd.loc["2008-01":"2009-12"]
normal = gdp_hd.loc["2004-01":"2007-12"]

print("=== Crise 2008-09 ===")
print(crisis.mean())
print("\n=== Normal 2004-07 ===")
print(normal.mean())
```

!!! tip "Interpretacao economica"
    A HD nao identifica "causas" no sentido profundo --- ela decompoe
    movimentos em contribuicoes de choques **conforme o modelo identificado**.
    A qualidade da decomposicao depende da qualidade da identificacao
    estrutural. Sempre relate a HD junto com o esquema de identificacao usado.

---

## Diagnosticos

### 1. Verificar Reconstrucao Exata

```python
# A soma das contribuicoes deve reconstruir a serie
import numpy as np

gdp_observed = data["gdp"] - hd.deterministic["gdp"]
gdp_reconstructed = hd.contributions["gdp"].sum(axis=1)

max_error = np.max(np.abs(gdp_observed - gdp_reconstructed))
print(f"Erro maximo de reconstrucao: {max_error:.2e}")
# Deve ser < 1e-10 (erro numerico)
```

### 2. Sensibilidade a Identificacao

```python
# Comparar HD com diferentes esquemas de identificacao
import itertools

for order in itertools.permutations(data.columns):
    d = data[list(order)]
    var_res = VAR(lags=4).fit(d)
    svar_res = SVAR(var_res, identification="cholesky").identify()
    hd_res = svar_res.historical_decomposition()
    # Comparar contribuicoes para GDP na crise
    crisis_contrib = hd_res.contributions["gdp"].loc["2008-01":"2009-12"].mean()
    print(f"Ordem {order}: {crisis_contrib.to_dict()}")
```

### 3. Bandas de Confianca via Bootstrap

```python
hd_boot = svar_results.historical_decomposition(
    ci_method="bootstrap",
    n_bootstrap=1000,
    ci=0.90,
)

# Contribuicao com incerteza
print(hd_boot.contributions_ci["gdp"]["e_interest_rate"])
```

### Checklist de Diagnostico

| Verificacao | Metodo | Esperado |
|---|---|---|
| Reconstrucao exata | Soma das contribuicoes | Erro < $10^{-10}$ |
| SVAR identificado | `svar_results.identification_check()` | Rank condition satisfeita |
| Robustez a identificacao | Variar esquema | Resultados qualitativamente similares |
| Bandas de confianca | Bootstrap | Contribuicoes significativas |
| VAR subjacente estavel | `var_results.is_stable` | `True` |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import VAR, SVAR

    var_results = VAR(lags=4).fit(data)
    svar_results = SVAR(var_results, identification="cholesky").identify()

    hd = svar_results.historical_decomposition()
    cf = svar_results.counterfactual(
        variable="gdp",
        exclude_shocks=["interest_rate"],
    )
    ```

=== "svars (R)"

    ```r
    library(vars)
    library(svars)

    # Estimar SVAR
    fit <- VAR(y, p = 4, type = "const")
    svar <- id.chol(fit)

    # Historical Decomposition
    hd_result <- hd(svar, series = 1)  # serie 1 = GDP
    plot(hd_result)

    # Counterfactual (manual)
    # Reconstruir serie excluindo contribuicao de choque j
    ```

**Mapeamento de parametros**:

| chronobox | svars (R) | Descricao |
|---|---|---|
| `svar_results.historical_decomposition()` | `hd(svar, series=1)` | HD |
| `hd.contributions["gdp"]` | `hd_result$hidec` | Contribuicoes |
| `hd.plot("gdp")` | `plot(hd_result)` | Grafico de barras empilhadas |
| `svar_results.counterfactual(...)` | Implementacao manual | Contrafactual |
| `hd.deterministic["gdp"]` | `hd_result$hidec[,"Base"]` | Componente deterministico |

---

## Referencias

- Burbidge, J. & Harrison, A. (1985). A Historical Decomposition of the Great
  Depression to Determine the Role of Money. *Journal of Monetary Economics*,
  16(1), 45--54.
- Kilian, L. & Lutkepohl, H. (2017). *Structural Vector Autoregressive Analysis*.
  Cambridge University Press. Capitulo 4.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
  Springer. Capitulo 9.
