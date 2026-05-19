---
title: FAVAR
description: Factor-Augmented VAR --- reducao de dimensionalidade com fatores latentes para politica monetaria.
---

# FAVAR (Factor-Augmented VAR)

!!! info "Quick Reference"
    - **Classe**: `chronobox.FAVAR`
    - **Import**: `from chronobox import FAVAR`
    - **R equivalente**: Nao ha pacote padrao; implementacao custom com `vars` + `psych::principal()`
    - **Estimacao**: PCA + VAR (two-step) ou Kalman filter (one-step)

---

## Overview

O FAVAR (Factor-Augmented VAR) de Bernanke, Boivin & Eliasz (2005) resolve
um problema fundamental dos VARs: a necessidade de trabalhar com poucas
variaveis. VARs padrao tipicamente incluem 3--7 variaveis, mas bancos centrais
monitoram centenas de indicadores.

O FAVAR extrai **fatores latentes** de um grande painel de series temporais
e estima um VAR sobre esses fatores junto com variaveis de interesse (ex.: taxa
de juros). Isso permite:

- Usar informacao de centenas de series sem explodir o numero de parametros
- Capturar movimentos comuns (ciclo economico, condicoes financeiras)
- Gerar IRFs para qualquer variavel do painel original
- Mitigar omitted variable bias

### Quando usar

- Analise de politica monetaria com muitos indicadores macroeconomicos
- Quando poucas variaveis nao capturam adequadamente o estado da economia
- Previsao usando informacao de grandes paineis de dados
- Quando ha suspeita de omitted variable bias em VARs pequenos

---

## Formulacao Matematica

### Modelo de Fatores

Seja $\mathbf{X}_t$ um vetor $N \times 1$ de variaveis informacionais (painel
grande, $N \gg K$). Assume-se que $\mathbf{X}_t$ e dirigido por poucos fatores
latentes:

$$
\mathbf{X}_t = \boldsymbol{\Lambda} \mathbf{F}_t + \mathbf{e}_t
$$

onde:

- $\mathbf{F}_t$ e o vetor $r \times 1$ de fatores latentes ($r \ll N$)
- $\boldsymbol{\Lambda}$ e a matriz $N \times r$ de factor loadings
- $\mathbf{e}_t$ sao erros idiossincraticos (fracamente correlacionados)

### VAR sobre Fatores

O FAVAR estima um VAR sobre os fatores extraidos mais as variaveis observadas
de interesse $\mathbf{Y}_t$ (ex.: taxa de juros):

$$
\begin{pmatrix} \mathbf{F}_t \\ \mathbf{Y}_t \end{pmatrix}
= \mathbf{\Phi}(L)
\begin{pmatrix} \mathbf{F}_{t-1} \\ \mathbf{Y}_{t-1} \end{pmatrix}
+ \mathbf{v}_t
$$

onde $\mathbf{\Phi}(L)$ e o polinomio matricial de defasagens.

### Extracao de Fatores: PCA

No metodo two-step:

1. Estimar $\hat{\mathbf{F}}_t$ por componentes principais de $\mathbf{X}_t$
2. Estimar o VAR sobre $(\hat{\mathbf{F}}_t', \mathbf{Y}_t')'$

Os primeiros $r$ componentes principais sao os autovetor da matriz de covariancia:

$$
\hat{\mathbf{F}} = \mathbf{X} \hat{\mathbf{V}}_r
$$

onde $\hat{\mathbf{V}}_r$ contem os $r$ autovetores associados aos maiores
autovalores de $\frac{1}{T}\mathbf{X}'\mathbf{X}$.

### Extracao de Fatores: Kalman Filter

No metodo one-step (via kalmanbox), fatores e coeficientes do VAR sao estimados
conjuntamente em um state-space model:

**Equacao de observacao**:

$$
\mathbf{X}_t = \boldsymbol{\Lambda} \mathbf{F}_t + \mathbf{e}_t
$$

**Equacao de transicao**:

$$
\begin{pmatrix} \mathbf{F}_t \\ \mathbf{Y}_t \end{pmatrix}
= \mathbf{\Phi}
\begin{pmatrix} \mathbf{F}_{t-1} \\ \mathbf{Y}_{t-1} \end{pmatrix}
+ \mathbf{v}_t
$$

!!! tip "PCA vs Kalman"
    O metodo PCA (two-step) e rapido e simples, mas nao estima incerteza dos
    fatores. O metodo Kalman (one-step) e mais eficiente estatisticamente,
    mas computacionalmente mais intenso.

### Numero de Fatores

O numero de fatores $r$ pode ser selecionado pelo criterio de Bai & Ng (2002):

$$
IC(r) = \ln\left(\frac{1}{NT} \sum_{i=1}^{N}\sum_{t=1}^{T} \hat{e}_{it}^2\right) + r \cdot g(N, T)
$$

onde $g(N, T)$ e uma funcao de penalizacao (ex.: $\frac{N+T}{NT}\ln\left(\frac{NT}{N+T}\right)$).

### IRF no FAVAR

As IRFs para as variaveis do painel original sao recuperadas via loadings:

$$
\text{IRF}_{X_i}(h) = \boldsymbol{\lambda}_i' \cdot \text{IRF}_{F}(h)
$$

onde $\boldsymbol{\lambda}_i'$ e a $i$-esima linha de $\boldsymbol{\Lambda}$.
Isso permite calcular IRFs para **todas as $N$ variaveis** do painel.

---

## Quick Example

```python
from chronobox import FAVAR
from chronobox.datasets import load_macro, load_macro_panel

# Variaveis de interesse (observadas no VAR)
Y = load_macro()[["interest_rate"]]

# Painel grande de informacionais (N >> K)
X = load_macro_panel()  # ~100 series macroeconomicas

# FAVAR com 3 fatores, PCA
model = FAVAR(
    n_factors=3,
    lags=4,
    method="pca",
)
results = model.fit(Y=Y, X=X)

# IRF: choque de juros sobre GDP (variavel do painel)
irf = results.irf(steps=40, identification="cholesky")
print(irf.table("interest_rate", "gdp"))

# FEVD
fevd = results.fevd(steps=40)
print(fevd.table("gdp"))
```

---

## Guia Detalhado

### Construtor

```python
FAVAR(
    n_factors=3,         # Numero de fatores latentes
    lags=4,              # Numero de defasagens do VAR
    method="pca",        # Metodo de extracao de fatores
    trend='c',           # Componente deterministico
    n_factors_method="bai_ng",  # Criterio para selecao do numero de fatores
    standardize=True,    # Padronizar X antes de extrair fatores
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `n_factors` | `int \| str` | `3` | Numero de fatores ou `'auto'` (Bai-Ng) |
| `lags` | `int` | `4` | Numero de defasagens do VAR |
| `method` | `str` | `'pca'` | `'pca'` (two-step) ou `'kalman'` (one-step via kalmanbox) |
| `trend` | `str` | `'c'` | `'n'`, `'c'`, `'t'`, `'ct'` |
| `n_factors_method` | `str` | `'bai_ng'` | Criterio para $r$ automatico |
| `standardize` | `bool` | `True` | Padronizar $\mathbf{X}$ (media 0, variancia 1) |

### Selecao do Numero de Fatores

```python
# Selecao automatica via Bai-Ng
model = FAVAR(n_factors="auto", lags=4)
results = model.fit(Y=Y, X=X)
print(f"Fatores selecionados: {results.n_factors}")

# Scree plot manual
import numpy as np

cov = np.cov(X.values, rowvar=False)
eigenvalues = np.linalg.eigvalsh(cov)[::-1]
explained = eigenvalues / eigenvalues.sum()
cumulative = np.cumsum(explained)
print("Variancia explicada acumulada:", cumulative[:10])
```

### Metodo PCA (Two-Step)

```python
model = FAVAR(n_factors=3, lags=4, method="pca")
results = model.fit(Y=Y, X=X)

# Fatores extraidos
print(results.factors)           # DataFrame (T x r)
print(results.loadings)          # DataFrame (N x r)
print(results.explained_variance)  # Variancia explicada por fator
```

### Metodo Kalman (One-Step)

```python
model = FAVAR(n_factors=3, lags=4, method="kalman")
results = model.fit(Y=Y, X=X)

# Fatores suavizados (Kalman smoother)
print(results.factors)
print(results.factor_uncertainty)  # Std dos fatores
```

### IRF para Variaveis do Painel

```python
# IRF de choque de juros sobre qualquer variavel do painel
irf = results.irf(steps=40, identification="cholesky")

# Variavel especifica do painel
print(irf.table("interest_rate", "industrial_production"))
print(irf.table("interest_rate", "unemployment"))
print(irf.table("interest_rate", "sp500"))
```

---

## Interpretacao

### Fatores Latentes

```python
print(results.factors.head())
```

```text
            Factor_1   Factor_2   Factor_3
2000-01     0.8231    -0.4512     0.1023
2000-02     0.7896    -0.3987     0.0845
2000-03     0.9102    -0.5234     0.1456
...
```

Os fatores nao tem interpretacao direta, mas podem ser correlacionados com
variaveis economicas para entender o que capturam:

```python
import numpy as np

# Correlacao dos fatores com variaveis selecionadas
corr = np.corrcoef(results.factors.T, X[["gdp", "unemployment", "sp500"]].T)
print(corr[:3, 3:])
```

Tipicamente: Fator 1 ≈ atividade economica, Fator 2 ≈ precos/inflacao,
Fator 3 ≈ condicoes financeiras.

### IRF Estrutural no FAVAR

A principal vantagem: IRFs para **centenas** de variaveis a partir de um
unico modelo:

```python
# Top 10 variaveis mais afetadas por choque de juros (horizonte 8)
impacts = {}
for var in X.columns:
    impacts[var] = abs(irf.value("interest_rate", var, step=8))

top10 = sorted(impacts.items(), key=lambda x: x[1], reverse=True)[:10]
for var, val in top10:
    print(f"  {var:30s}: {val:.4f}")
```

---

## Diagnosticos

### 1. Adequacao dos Fatores

```python
# Variancia explicada
print(results.explained_variance)
# Idealmente > 50% com os fatores selecionados
```

### 2. Estabilidade do VAR

```python
print(f"VAR estavel: {results.is_stable}")
```

### 3. Sensibilidade ao Numero de Fatores

```python
for r in range(1, 8):
    model = FAVAR(n_factors=r, lags=4, method="pca")
    res = model.fit(Y=Y, X=X)
    irf_val = res.irf(steps=20, identification="cholesky").value(
        "interest_rate", "gdp", step=8
    )
    print(f"r={r}: IRF(8) GDP = {irf_val:.4f}")
```

### Checklist de Diagnostico

| Verificacao | Metodo | Esperado |
|---|---|---|
| Variancia explicada | `explained_variance` | > 50% com $r$ fatores |
| Numero de fatores | Bai-Ng, scree plot | Estavel entre criterios |
| Estabilidade VAR | Eigenvalues | Todos dentro do circulo unitario |
| Sensibilidade a $r$ | Variar $r$ | IRFs qualitativamente similares |
| Residuos | Portmanteau test | Sem autocorrelacao |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import FAVAR

    model = FAVAR(n_factors=3, lags=4, method="pca")
    results = model.fit(Y=Y, X=X)

    irf = results.irf(steps=40, identification="cholesky")
    fevd = results.fevd(steps=40)
    ```

=== "R (custom)"

    ```r
    library(vars)

    # 1. Extrair fatores via PCA
    pca <- prcomp(X, scale. = TRUE)
    F_hat <- pca$x[, 1:3]  # 3 fatores

    # 2. Combinar fatores + variavel de interesse
    Z <- cbind(F_hat, Y)

    # 3. Estimar VAR
    fit <- VAR(Z, p = 4, type = "const")

    # 4. SVAR com Cholesky
    svar <- id.chol(fit)
    irf(svar, n.ahead = 40)

    # 5. Recuperar IRFs para variaveis do painel
    # IRF_Xi = loadings[i,] %*% IRF_F
    loadings <- pca$rotation[, 1:3]
    ```

!!! note "R: sem pacote padrao"
    Nao ha um pacote R padrao para FAVAR. A implementacao tipica usa
    `prcomp()` para PCA + `vars::VAR()` para o VAR, combinando manualmente.
    Em Python, o `chronobox.FAVAR` integra ambos os passos.

---

## Referencias

- Bernanke, B. S., Boivin, J. & Eliasz, P. (2005). Measuring the Effects of
  Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach.
  *Quarterly Journal of Economics*, 120(1), 387--422.
- Stock, J. H. & Watson, M. W. (2002). Forecasting Using Principal Components
  from a Large Number of Predictors. *Journal of the American Statistical
  Association*, 97(460), 1167--1179.
- Bai, J. & Ng, S. (2002). Determining the Number of Factors in Approximate
  Factor Models. *Econometrica*, 70(1), 191--221.
