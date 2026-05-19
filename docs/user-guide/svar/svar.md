---
title: SVAR
description: VAR Estrutural --- identificacao de choques via restricoes de curto prazo, longo prazo e de sinal.
---

# SVAR (Structural VAR)

!!! info "Quick Reference"
    - **Classe**: `chronobox.SVAR`
    - **Import**: `from chronobox import SVAR`
    - **R equivalente**: `vars::SVAR()`, `svars::id.chol()`, `svars::id.st()`
    - **Identificacao**: Cholesky, restricoes customizadas, longo prazo, sinal

---

## Overview

O SVAR (Structural Vector Autoregression) estende o VAR reduzido impondo
**restricoes de identificacao** que permitem recuperar choques estruturais com
interpretacao economica. Enquanto o VAR reduzido estima correlacoes, o SVAR
estima **relacoes causais contemporaneas**.

A motivacao central e o **problema de identificacao**: os residuos do VAR
reduzido ($\mathbf{u}_t$) sao combinacoes lineares dos choques estruturais
($\mathbf{e}_t$). Sem restricoes adicionais, nao e possivel separar os choques.

### Quando usar

- Analise de choques de politica monetaria, fiscal, de oferta, de demanda
- Impulse Response Functions (IRF) com interpretacao causal
- Variance Decomposition (FEVD) atribuindo variancia a choques especificos
- Historical Decomposition para explicar episodios historicos
- Qualquer analise que exija identificacao de choques estruturais

---

## Formulacao Matematica

### Do VAR Reduzido ao SVAR

O VAR reduzido estima:

$$
\mathbf{y}_t = \mathbf{A}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{u}_t
\qquad \mathbf{u}_t \sim (\mathbf{0}, \boldsymbol{\Sigma}_u)
$$

O modelo estrutural (forma AB) e:

$$
\mathbf{A}_0 \mathbf{y}_t = \mathbf{A}_0^{*} \mathbf{y}_{t-1} + \cdots + \mathbf{A}_p^{*} \mathbf{y}_{t-p} + \mathbf{B}_0 \mathbf{e}_t
$$

onde $\mathbf{e}_t \sim (\mathbf{0}, \mathbf{I}_K)$ sao os **choques estruturais** ortogonais.

A relacao entre as formas e:

$$
\mathbf{u}_t = \mathbf{A}_0^{-1} \mathbf{B}_0 \mathbf{e}_t
$$

$$
\boldsymbol{\Sigma}_u = \mathbf{A}_0^{-1} \mathbf{B}_0 \mathbf{B}_0' (\mathbf{A}_0^{-1})'
$$

### Condicao de Identificacao

A matriz $\boldsymbol{\Sigma}_u$ fornece $K(K+1)/2$ equacoes (por simetria).
A matriz $\mathbf{A}_0^{-1}\mathbf{B}_0$ tem $K^2$ elementos livres. Para
identificacao exata, precisamos de pelo menos:

$$
K^2 - \frac{K(K+1)}{2} = \frac{K(K-1)}{2} \text{ restricoes}
$$

Para $K=3$: 3 restricoes. Para $K=4$: 6 restricoes.

### Restricoes de Curto Prazo (Cholesky)

A decomposicao de Cholesky impoe uma estrutura **triangular inferior** a
$\mathbf{A}_0^{-1}\mathbf{B}_0$:

$$
\mathbf{u}_t =
\begin{pmatrix}
b_{11} & 0 & 0 \\
b_{21} & b_{22} & 0 \\
b_{31} & b_{32} & b_{33}
\end{pmatrix}
\mathbf{e}_t
$$

Isso implica uma **ordenacao causal recursiva**: a primeira variavel nao e
afetada contemporaneamente pelas demais; a segunda so e afetada pela primeira;
e assim por diante.

!!! warning "Sensibilidade a ordenacao"
    Os resultados de Cholesky dependem da **ordem das variaveis**. A variavel
    mais "lenta" (ex.: PIB) deve vir primeiro, e a mais "rapida" (ex.: taxa de
    juros) por ultimo. Sempre teste a robustez reordenando as variaveis.

### Restricoes Customizadas de Curto Prazo

Restricoes mais gerais definem zeros (ou valores fixos) em posicoes especificas
de $\mathbf{A}_0$ e $\mathbf{B}_0$:

$$
\mathbf{A}_0 =
\begin{pmatrix}
1 & 0 & a_{13} \\
a_{21} & 1 & 0 \\
a_{31} & a_{32} & 1
\end{pmatrix}, \qquad
\mathbf{B}_0 =
\begin{pmatrix}
b_{11} & 0 & 0 \\
0 & b_{22} & 0 \\
0 & 0 & b_{33}
\end{pmatrix}
$$

A **condicao de ordem** exige pelo menos $K(K-1)/2$ restricoes. A **condicao
de rank** exige que essas restricoes sejam independentes.

### Restricoes de Longo Prazo (Blanchard-Quah)

Impoe que certos choques nao tenham efeito permanente sobre certas variaveis.
O multiplicador de longo prazo e:

$$
\boldsymbol{\Xi} = (\mathbf{I}_K - \mathbf{A}_1 - \cdots - \mathbf{A}_p)^{-1}
$$

O efeito acumulado dos choques estruturais e $\boldsymbol{\Xi} \mathbf{A}_0^{-1} \mathbf{B}_0$.
Restricoes de longo prazo impoem zeros nesta matriz:

$$
\boldsymbol{\Xi} \mathbf{A}_0^{-1} \mathbf{B}_0 =
\begin{pmatrix}
* & 0 & 0 \\
* & * & 0 \\
* & * & *
\end{pmatrix}
$$

!!! tip "Exemplo classico"
    Blanchard & Quah (1989): choques de demanda nao tem efeito permanente sobre
    o produto --- apenas choques de oferta afetam o produto no longo prazo.

### Sign Restrictions

Em vez de impor zeros exatos, restricoes de sinal exigem que as IRFs tenham
sinais especificos em determinados horizontes:

$$
\text{IRF}_{ij}(h) \geq 0 \quad \text{ou} \quad \text{IRF}_{ij}(h) \leq 0
$$

O algoritmo gera rotacoes aleatorias da decomposicao de Cholesky e mantem
apenas as que satisfazem as restricoes de sinal.

---

## Quick Example

```python
from chronobox import VAR, SVAR
from chronobox.datasets import load_macro

# Dados: GDP, inflacao, taxa de juros
data = load_macro()

# Estimar VAR reduzido
var_model = VAR(lags=4)
var_results = var_model.fit(data)

# SVAR com Cholesky (ordenacao: GDP → inflacao → juros)
svar = SVAR(var_results, identification="cholesky")
svar_results = svar.identify()

# IRF estrutural: choque de juros sobre GDP
sirf = svar_results.irf(steps=40)
print(sirf.table("interest_rate", "gdp"))

# FEVD estrutural
sfevd = svar_results.fevd(steps=40)
print(sfevd.table("gdp"))
```

---

## Guia Detalhado

### Construtor

```python
SVAR(
    var_results,              # Resultado de um VAR estimado
    identification="cholesky",  # Metodo de identificacao
    A_matrix=None,            # Restricoes sobre A0 (forma AB)
    B_matrix=None,            # Restricoes sobre B0 (forma AB)
    lr_matrix=None,           # Restricoes de longo prazo
    sign_restrictions=None,   # Dict de restricoes de sinal
    n_rotations=1000,         # Numero de rotacoes (sign restrictions)
    seed=None                 # Semente para reprodutibilidade
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `var_results` | `VARResults` | --- | Resultado de `VAR.fit()` |
| `identification` | `str` | `'cholesky'` | `'cholesky'`, `'short_run'`, `'long_run'`, `'sign'` |
| `A_matrix` | `np.ndarray \| None` | `None` | Restricoes sobre $\mathbf{A}_0$ (`np.nan` = livre) |
| `B_matrix` | `np.ndarray \| None` | `None` | Restricoes sobre $\mathbf{B}_0$ (`np.nan` = livre) |
| `lr_matrix` | `np.ndarray \| None` | `None` | Restricoes de longo prazo (`np.nan` = livre, `0` = zero) |
| `sign_restrictions` | `dict \| None` | `None` | Restricoes de sinal para as IRFs |
| `n_rotations` | `int` | `1000` | Numero de rotacoes aleatorias (sign restrictions) |
| `seed` | `int \| None` | `None` | Semente para reprodutibilidade |

### Identificacao por Cholesky

```python
from chronobox import VAR, SVAR

# Ordenacao importa: GDP (lenta) → Inflacao → Juros (rapida)
data = data[["gdp", "inflation", "interest_rate"]]

var_results = VAR(lags=4).fit(data)
svar = SVAR(var_results, identification="cholesky")
svar_results = svar.identify()

# Matriz de impacto estimada (triangular inferior)
print(svar_results.B0)
```

### Restricoes Customizadas (Forma AB)

```python
import numpy as np

# K=3 variaveis: GDP, inflacao, juros
# np.nan = parametro livre, valor numerico = restricao

A = np.array([
    [1,      0,      0     ],
    [np.nan, 1,      0     ],
    [np.nan, np.nan, 1     ],
])

B = np.array([
    [np.nan, 0,      0     ],
    [0,      np.nan, 0     ],
    [0,      0,      np.nan],
])

svar = SVAR(var_results, identification="short_run", A_matrix=A, B_matrix=B)
svar_results = svar.identify()
```

### Restricoes de Longo Prazo (Blanchard-Quah)

```python
# Restricao: choque de demanda nao afeta GDP no longo prazo
# Linha 1 = GDP, Coluna 2 = choque de demanda
lr = np.array([
    [np.nan, 0,      0     ],
    [np.nan, np.nan, 0     ],
    [np.nan, np.nan, np.nan],
])

svar = SVAR(var_results, identification="long_run", lr_matrix=lr)
svar_results = svar.identify()
```

### Sign Restrictions

```python
# Choque de politica monetaria (choque 3):
# - Juros sobem (positivo)
# - GDP cai (negativo)
# - Inflacao cai (negativo)
# Horizonte: primeiros 4 periodos

sign_restr = {
    "shock_3": {
        "interest_rate": ("+", 0, 4),  # positivo de h=0 a h=4
        "gdp":           ("-", 0, 4),  # negativo de h=0 a h=4
        "inflation":     ("-", 0, 4),  # negativo de h=0 a h=4
    }
}

svar = SVAR(
    var_results,
    identification="sign",
    sign_restrictions=sign_restr,
    n_rotations=5000,
    seed=42,
)
svar_results = svar.identify()

# Mediana e bandas das IRFs aceitas
print(f"Rotacoes aceitas: {svar_results.n_accepted}/{svar_results.n_rotations}")
```

---

## Interpretacao

### IRF Estrutural vs Reduzida

| Tipo | Comando | Interpretacao |
|---|---|---|
| IRF reduzida | `var_results.irf()` | Resposta a um choque de 1 desvio-padrao em $\mathbf{u}_t$ |
| IRF estrutural | `svar_results.irf()` | Resposta a um choque de 1 d.p. em $\mathbf{e}_t$ (choque **identificado**) |

A IRF estrutural e a unica que permite afirmacoes do tipo "um choque de politica
monetaria de 25 bps reduz o PIB em X% apos Y trimestres".

### Lendo a Matriz de Impacto

```python
print(svar_results.B0)
```

```text
              e_gdp   e_infl   e_rate
gdp          0.0084   0.0000   0.0000
inflation    0.0032   0.0051   0.0000
interest_rate 0.0011  0.0023   0.0041
```

- Diagonal: magnitude dos choques proprios
- Abaixo da diagonal: efeitos contemporaneos cruzados
- Zeros: restricoes impostas

### FEVD Estrutural

```python
sfevd = svar_results.fevd(steps=40)
print(sfevd.table("gdp"))
```

```text
Variance decomposition of gdp:
  step   e_gdp   e_infl  e_rate
     1  100.00     0.00    0.00
     4   82.31    12.45    5.24
     8   71.02    18.67   10.31
    20   64.53    21.12   14.35
    40   63.18    21.89   14.93
```

!!! tip "Interpretacao da FEVD"
    No horizonte de 40 periodos, ~63% da variancia do erro de previsao do GDP
    se deve a choques proprios, ~22% a choques de inflacao, e ~15% a choques de
    juros. Esses valores so tem sentido com **identificacao estrutural**.

---

## Diagnosticos

### 1. Verificar Condicoes de Identificacao

```python
# Condicao de ordem e rank
print(svar_results.identification_check())
```

```text
Restrictions:      3
Required (order):  3
Rank condition:    SATISFIED
Identification:    EXACT
```

### 2. Robustez a Ordenacao (Cholesky)

```python
import itertools

orderings = list(itertools.permutations(data.columns))
for order in orderings:
    d = data[list(order)]
    res = VAR(lags=4).fit(d)
    svar_res = SVAR(res, identification="cholesky").identify()
    irf_val = svar_res.irf(steps=20).value("interest_rate", "gdp", step=8)
    print(f"Ordem {order}: IRF(8) = {irf_val:.4f}")
```

### 3. Bootstrap de Intervalos de Confianca

```python
# IRF com bandas de confianca via bootstrap
sirf = svar_results.irf(steps=40, ci_method="bootstrap", n_bootstrap=1000, ci=0.95)
print(sirf.table("interest_rate", "gdp"))
```

### Checklist de Diagnostico

| Verificacao | Metodo | Esperado |
|---|---|---|
| VAR subjacente estavel | `var_results.is_stable` | `True` |
| Condicao de ordem | $\geq K(K-1)/2$ restricoes | Satisfeita |
| Condicao de rank | Restricoes independentes | Satisfeita |
| Robustez a ordenacao | Testar permutacoes | Resultados qualitativamente similares |
| Bandas de confianca | Bootstrap | IRFs significativas nos horizontes de interesse |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import VAR, SVAR

    var_results = VAR(lags=4).fit(data)

    # Cholesky
    svar = SVAR(var_results, identification="cholesky")
    svar_results = svar.identify()

    # IRF estrutural
    sirf = svar_results.irf(steps=40)

    # Long-run
    svar_lr = SVAR(var_results, identification="long_run", lr_matrix=lr)
    svar_lr_results = svar_lr.identify()
    ```

=== "vars / svars (R)"

    ```r
    library(vars)
    library(svars)

    # Estimar VAR
    fit <- VAR(y, p = 4, type = "const")

    # Cholesky
    svar_chol <- id.chol(fit)
    irf_chol <- irf(svar_chol, n.ahead = 40)

    # Restricoes de curto prazo (forma AB)
    amat <- diag(3)
    amat[2,1] <- NA; amat[3,1] <- NA; amat[3,2] <- NA
    bmat <- diag(3)
    diag(bmat) <- NA
    svar_ab <- SVAR(fit, Amat = amat, Bmat = bmat)

    # Blanchard-Quah (long-run)
    svar_bq <- BQ(fit)

    # Sign restrictions
    svar_sign <- id.sign(fit, n.ahead = 4)
    ```

**Mapeamento de parametros**:

| chronobox | vars / svars (R) | Descricao |
|---|---|---|
| `identification="cholesky"` | `id.chol(fit)` | Decomposicao de Cholesky |
| `identification="short_run"` | `SVAR(fit, Amat, Bmat)` | Restricoes de curto prazo |
| `identification="long_run"` | `BQ(fit)` | Blanchard-Quah |
| `identification="sign"` | `id.sign(fit)` | Restricoes de sinal |
| `svar_results.irf(steps=h)` | `irf(svar, n.ahead=h)` | IRF estrutural |
| `svar_results.fevd(steps=h)` | `fevd(svar, n.ahead=h)` | FEVD estrutural |

---

## Referencias

- Sims, C. A. (1980). Macroeconomics and Reality. *Econometrica*, 48(1), 1--48.
- Blanchard, O. J. & Quah, D. (1989). The Dynamic Effects of Aggregate Demand and
  Supply Disturbances. *American Economic Review*, 79(4), 655--673.
- Uhlig, H. (2005). What Are the Effects of Monetary Policy on Output? Results from
  an Agnostic Identification Procedure. *Journal of Monetary Economics*, 52(2), 381--419.
- Kilian, L. & Lutkepohl, H. (2017). *Structural Vector Autoregressive Analysis*.
  Cambridge University Press.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
