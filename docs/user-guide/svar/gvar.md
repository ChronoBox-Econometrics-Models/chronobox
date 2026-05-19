---
title: GVAR
description: Global VAR --- modelagem de transmissao de choques entre paises com matrizes de pesos.
---

# GVAR (Global VAR)

!!! info "Quick Reference"
    - **Classe**: `chronobox.GVAR`
    - **Import**: `from chronobox import GVAR`
    - **R equivalente**: `GVAR::gvar()` (pacote GVAR)
    - **Estimacao**: VARX pais por pais + solucao global

---

## Overview

O GVAR (Global Vector Autoregression) de Pesaran, Schuermann & Weiner (2004)
modela a **transmissao internacional de choques** entre economias interligadas.
Em vez de estimar um unico VAR enorme com todas as variaveis de todos os paises,
o GVAR:

1. Estima um **VARX** (VAR com variaveis exogenas) para cada pais
2. Usa **variaveis foreign** (medias ponderadas das variaveis dos outros paises) como exogenas
3. Combina todos os modelos individuais em uma **solucao global**

Isso permite modelar dezenas de paises sem a explosao de parametros de um VAR global direto.

### Quando usar

- Analise de transmissao de choques entre paises (spillovers)
- Efeitos de politica monetaria de um pais sobre outros
- Impacto de crises financeiras globais
- Previsao macroeconomica multi-pais
- Analise de transmissao via comercio ou fluxos financeiros

---

## Formulacao Matematica

### VARX por Pais

Para cada pais $i = 0, 1, \ldots, N$, o modelo individual e um VARX($p_i$, $q_i$):

$$
\mathbf{y}_{it} = \mathbf{c}_i + \mathbf{A}_{i1} \mathbf{y}_{i,t-1} + \cdots + \mathbf{A}_{ip_i} \mathbf{y}_{i,t-p_i} + \boldsymbol{\Lambda}_{i0} \mathbf{y}_{it}^* + \cdots + \boldsymbol{\Lambda}_{iq_i} \mathbf{y}_{i,t-q_i}^* + \mathbf{u}_{it}
$$

onde:

- $\mathbf{y}_{it}$: vetor $k_i \times 1$ de variaveis **domesticas** do pais $i$
- $\mathbf{y}_{it}^*$: vetor $k_i^* \times 1$ de variaveis **foreign** do pais $i$
- $\mathbf{A}_{ij}$: coeficientes das variaveis domesticas defasadas
- $\boldsymbol{\Lambda}_{ij}$: coeficientes das variaveis foreign

### Variaveis Foreign (Star Variables)

As variaveis foreign sao medias ponderadas das variaveis dos outros paises:

$$
\mathbf{y}_{it}^* = \sum_{j=0}^{N} w_{ij} \mathbf{y}_{jt}, \qquad w_{ii} = 0, \quad \sum_{j \neq i} w_{ij} = 1
$$

Os pesos $w_{ij}$ refletem a importancia economica do pais $j$ para o pais $i$,
tipicamente baseados em:

- **Comercio bilateral**: $w_{ij} = \frac{\text{comercio}_{ij}}{\sum_{k \neq i} \text{comercio}_{ik}}$
- **Fluxos financeiros**: exposicao bancaria, investimento direto
- **Distancia geografica**: pesos inversamente proporcionais

### Matriz de Pesos (Link Matrix)

A matriz de pesos $\mathbf{W}$ ($N+1 \times N+1$) e uma **link matrix** com:

$$
\mathbf{W} =
\begin{pmatrix}
0 & w_{01} & w_{02} & \cdots & w_{0N} \\
w_{10} & 0 & w_{12} & \cdots & w_{1N} \\
\vdots & & \ddots & & \vdots \\
w_{N0} & w_{N1} & \cdots & w_{N,N-1} & 0
\end{pmatrix}
$$

Diagonal zero (pais nao e seu proprio foreign), linhas somam 1.

### Solucao Global

Combinando todos os modelos VARX individuais:

1. Definir o vetor global $\mathbf{x}_t = (\mathbf{y}_{0t}', \mathbf{y}_{1t}', \ldots, \mathbf{y}_{Nt}')'$
2. Usar a link matrix para expressar $\mathbf{y}_{it}^*$ em termos de $\mathbf{x}_t$:
   $\mathbf{y}_{it}^* = \mathbf{W}_i \mathbf{x}_t$
3. Empilhar todos os modelos:

$$
\mathbf{G}_0 \mathbf{x}_t = \mathbf{a}_0 + \mathbf{G}_1 \mathbf{x}_{t-1} + \cdots + \mathbf{G}_p \mathbf{x}_{t-p} + \mathbf{u}_t
$$

4. Resolver para a forma reduzida:

$$
\mathbf{x}_t = \mathbf{G}_0^{-1}\mathbf{a}_0 + \mathbf{G}_0^{-1}\mathbf{G}_1 \mathbf{x}_{t-1} + \cdots + \mathbf{G}_0^{-1}\mathbf{u}_t
$$

### GIRF (Generalized Impulse Response Function)

O GVAR usa GIRFs (Pesaran & Shin, 1998) que nao dependem da ordenacao das
variaveis:

$$
\text{GIRF}(\mathbf{x}_{t+h} \mid e_{jt} = \delta_j) = E[\mathbf{x}_{t+h} \mid e_{jt} = \delta_j, \boldsymbol{\Omega}_{t-1}] - E[\mathbf{x}_{t+h} \mid \boldsymbol{\Omega}_{t-1}]
$$

Para um choque de um desvio-padrao ($\delta_j = \sigma_{jj}^{1/2}$):

$$
\text{GIRF}(h) = \frac{\mathbf{F}^h \boldsymbol{\Sigma}_u \mathbf{e}_j}{\sqrt{\sigma_{jj}}}
$$

---

## Quick Example

```python
from chronobox import GVAR
from chronobox.datasets import load_gvar_data

# Dados de multiplos paises
data, weights = load_gvar_data()
# data: dict de DataFrames {country: DataFrame}
# weights: DataFrame (N+1 x N+1) de pesos comerciais

# Configurar GVAR
model = GVAR(
    lags_domestic=2,
    lags_foreign=1,
    weights=weights,
)
results = model.fit(data)

# GIRF: choque nos juros dos EUA
girf = results.girf(
    shock_country="US",
    shock_variable="interest_rate",
    steps=40,
)

# Efeito sobre o GDP do Brasil
print(girf.table("BR", "gdp"))

# Efeito sobre todos os paises
print(girf.summary(variable="gdp", step=8))
```

---

## Guia Detalhado

### Construtor

```python
GVAR(
    lags_domestic=2,       # p_i: lags das variaveis domesticas
    lags_foreign=1,        # q_i: lags das variaveis foreign
    weights=None,          # Matriz de pesos (DataFrame ou array)
    trend='c',             # Componente deterministico
    global_variables=None, # Variaveis globais (ex.: preco do petroleo)
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `lags_domestic` | `int \| dict` | `2` | Lags domesticos (int uniforme ou dict por pais) |
| `lags_foreign` | `int \| dict` | `1` | Lags foreign (int uniforme ou dict por pais) |
| `weights` | `pd.DataFrame \| np.ndarray` | `None` | Matriz de pesos $\mathbf{W}$ |
| `trend` | `str` | `'c'` | `'n'`, `'c'`, `'t'`, `'ct'` |
| `global_variables` | `pd.DataFrame \| None` | `None` | Variaveis globais exogenas |

### Preparacao dos Dados

```python
import pandas as pd

# Dados por pais: dict de DataFrames com as mesmas colunas
data = {
    "US": pd.DataFrame({
        "gdp": us_gdp, "inflation": us_infl, "interest_rate": us_rate
    }),
    "EU": pd.DataFrame({
        "gdp": eu_gdp, "inflation": eu_infl, "interest_rate": eu_rate
    }),
    "BR": pd.DataFrame({
        "gdp": br_gdp, "inflation": br_infl, "interest_rate": br_rate
    }),
    # ... mais paises
}

# Matriz de pesos comerciais (linhas somam 1, diagonal = 0)
weights = pd.DataFrame(
    [[0.0, 0.3, 0.2, ...],  # US
     [0.4, 0.0, 0.1, ...],  # EU
     [0.3, 0.2, 0.0, ...],  # BR
     ...],
    index=["US", "EU", "BR", ...],
    columns=["US", "EU", "BR", ...],
)
```

### Variaveis Globais

```python
import pandas as pd

# Variaveis exogenas globais (afetam todos os paises)
global_vars = pd.DataFrame({
    "oil_price": oil_series,
    "vix": vix_series,
})

model = GVAR(
    lags_domestic=2,
    lags_foreign=1,
    weights=weights,
    global_variables=global_vars,
)
results = model.fit(data)
```

### Lags Especificos por Pais

```python
# Diferentes ordens para diferentes paises
model = GVAR(
    lags_domestic={"US": 4, "EU": 2, "BR": 2},
    lags_foreign={"US": 2, "EU": 1, "BR": 1},
    weights=weights,
)
results = model.fit(data)
```

### GIRF (Generalized IRF)

```python
# Choque no GDP dos EUA
girf = results.girf(
    shock_country="US",
    shock_variable="gdp",
    steps=40,
)

# Resposta do GDP de cada pais
for country in data.keys():
    val = girf.value(country, "gdp", step=8)
    print(f"  {country} GDP response at h=8: {val:.4f}")
```

### Previsao Multi-Pais

```python
fc = results.forecast(steps=8)

# Previsao por pais
for country in data.keys():
    print(f"\n{country}:")
    print(fc[country]["forecast"])
```

---

## Interpretacao

### Resultados por Pais

```python
# Resumo do VARX individual de cada pais
for country in data.keys():
    print(f"\n=== {country} ===")
    print(results.country_results[country].summary())
```

### Weak Exogeneity Test

Um pressuposto chave do GVAR e que as variaveis foreign sao **fracamente
exogenas** para os parametros de longo prazo:

```python
# Teste de exogeneidade fraca por pais
for country in data.keys():
    we_test = results.weak_exogeneity_test(country)
    print(f"{country}: p-value = {we_test.pvalue:.4f}")
    # p > 0.05 → nao rejeita exogeneidade fraca (desejado)
```

### Mapa de Transmissao de Choques

```python
# Impacto de choque de juros dos EUA sobre GDP de todos os paises em h=8
girf = results.girf(shock_country="US", shock_variable="interest_rate", steps=40)

impacts = {}
for country in data.keys():
    impacts[country] = girf.value(country, "gdp", step=8)

# Ordenar por magnitude
for country, val in sorted(impacts.items(), key=lambda x: x[1]):
    print(f"  {country}: {val:+.4f}")
```

---

## Diagnosticos

### 1. Estabilidade Global

```python
# Eigenvalues da companion matrix global
print(f"GVAR estavel: {results.is_stable}")
print(f"Maior eigenvalue (modulo): {max(abs(results.eigenvalues)):.4f}")
```

### 2. Weak Exogeneity

```python
we_results = results.weak_exogeneity_test_all()
print(we_results)
# Todas as rejeicoes indicam possivel problema
```

### 3. Residuos por Pais

```python
from chronobox.tests_stat import portmanteau_test

for country in data.keys():
    resid = results.country_results[country].residuals
    pt = portmanteau_test(resid, lags=8)
    print(f"{country}: Portmanteau p = {pt.pvalue:.4f}")
```

### Checklist de Diagnostico

| Verificacao | Metodo | Esperado |
|---|---|---|
| Estabilidade global | Eigenvalues do sistema | Todos $\|\lambda\| < 1$ |
| Weak exogeneity | F-test por pais | $p > 0.05$ para maioria |
| Residuos | Portmanteau por pais | Sem autocorrelacao |
| Pesos | Robustez a pesos alternativos | Resultados qualitativamente similares |
| Cointegracao | Trace test por pais | Ranks de cointegracao adequados |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import GVAR

    model = GVAR(
        lags_domestic=2,
        lags_foreign=1,
        weights=weights,
    )
    results = model.fit(data)

    girf = results.girf(
        shock_country="US",
        shock_variable="interest_rate",
        steps=40,
    )
    ```

=== "GVAR (R)"

    ```r
    library(GVAR)

    # Configurar modelo
    gvar_setup <- gvar(
      data = country_data,
      weights = W,
      p.domestic = 2,
      q.foreign = 1
    )

    # Estimar
    fit <- estimate(gvar_setup)

    # GIRF
    girf <- irf(fit,
                 shock = "US.interest_rate",
                 n.ahead = 40,
                 type = "GIRF")
    plot(girf)
    ```

**Mapeamento de parametros**:

| chronobox | GVAR (R) | Descricao |
|---|---|---|
| `lags_domestic=2` | `p.domestic=2` | Lags domesticos |
| `lags_foreign=1` | `q.foreign=1` | Lags foreign |
| `weights=W` | `weights=W` | Matriz de pesos |
| `results.girf(...)` | `irf(fit, type="GIRF")` | GIRF |
| `global_variables=...` | `global=...` | Variaveis globais |

---

## Referencias

- Pesaran, M. H., Schuermann, T. & Weiner, S. M. (2004). Modeling Regional
  Interdependencies Using a Global Error-Correcting Macroeconometric Model.
  *Journal of Business & Economic Statistics*, 22(2), 129--162.
- Dees, S., di Mauro, F., Pesaran, M. H. & Smith, L. V. (2007). Exploring the
  International Linkages of the Euro Area: A Global VAR Analysis.
  *Journal of Applied Econometrics*, 22(1), 1--38.
- Pesaran, M. H. & Shin, Y. (1998). Generalized Impulse Response Analysis in
  Linear Multivariate Models. *Economics Letters*, 58(1), 17--29.
- Chudik, A. & Pesaran, M. H. (2016). Theory and Practice of GVAR Modelling.
  *Journal of Economic Surveys*, 30(1), 165--197.
