---
title: SVAR & Advanced Models
description: Modelos VAR estruturais e avancados --- SVAR, BVAR, FAVAR, TVP-VAR, GVAR e Historical Decomposition.
---

# SVAR & Modelos Avancados

Modelos VAR reduzidos capturam correlacoes dinamicas entre variaveis, mas nao
permitem interpretar **choques estruturais** --- nao e possivel distinguir um
choque de politica monetaria de um choque de oferta sem restricoes adicionais.
Os modelos desta secao estendem o VAR basico para superar essa limitacao.

---

## Por que modelos estruturais?

O VAR reduzido estima a forma:

$$
\mathbf{y}_t = \mathbf{A}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{u}_t
$$

Os residuos $\mathbf{u}_t$ sao correlacionados contemporaneamente --- sao uma
**mistura** de choques estruturais. Para recuperar choques com interpretacao
economica, precisamos impor restricoes de identificacao.

---

## O problema de identificacao

Um sistema VAR com $K$ variaveis possui $K(K-1)/2$ elementos livres na matriz
de impacto contemporaneo. Para identificar os choques estruturais, precisamos de
pelo menos $K(K-1)/2$ restricoes. As estrategias incluem:

| Estrategia | Abordagem | Modelo |
|---|---|---|
| Restricoes de curto prazo | Zeros na matriz $\mathbf{A}_0$ ou $\mathbf{B}_0$ | [SVAR](svar.md) |
| Restricoes de longo prazo | Zeros no multiplicador de longo prazo | [SVAR](svar.md) |
| Restricoes de sinal | Sinais nas IRFs | [SVAR](svar.md) |
| Priors Bayesianas | Regularizacao via distribuicao a priori | [BVAR](bvar.md) |
| Fatores latentes | Reducao de dimensionalidade | [FAVAR](favar.md) |
| Parametros variantes | Coeficientes que mudam no tempo | [TVP-VAR](tvp-var.md) |
| Modelagem global | VARs interligados por pesos | [GVAR](gvar.md) |

---

## Modelos Disponiveis

<div class="grid cards" markdown>

-   :material-matrix:{ .lg .middle } **SVAR**

    ---

    VAR estrutural com restricoes de curto prazo, longo prazo e de sinal.

    [:octicons-arrow-right-24: SVAR](svar.md)

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **BVAR**

    ---

    VAR Bayesiano com Minnesota prior, Normal-Wishart e SSVS.

    [:octicons-arrow-right-24: BVAR](bvar.md)

-   :material-blur:{ .lg .middle } **FAVAR**

    ---

    Factor-Augmented VAR para sistemas com muitas variaveis.

    [:octicons-arrow-right-24: FAVAR](favar.md)

-   :material-chart-timeline-variant:{ .lg .middle } **TVP-VAR**

    ---

    VAR com parametros variantes no tempo via kalmanbox.

    [:octicons-arrow-right-24: TVP-VAR](tvp-var.md)

-   :material-earth:{ .lg .middle } **GVAR**

    ---

    Global VAR para modelagem de transmissao internacional de choques.

    [:octicons-arrow-right-24: GVAR](gvar.md)

-   :material-chart-waterfall:{ .lg .middle } **Historical Decomposition**

    ---

    Atribuicao de movimentos historicos a choques estruturais individuais.

    [:octicons-arrow-right-24: Historical Decomposition](hd.md)

</div>

---

## Fluxo de Trabalho Tipico

```python
from chronobox import VAR, SVAR
from chronobox.datasets import load_macro

# 1. Estimar VAR reduzido
data = load_macro()
var_model = VAR(lags=2)
var_results = var_model.fit(data)

# 2. Identificar choques estruturais (Cholesky)
svar_model = SVAR(var_results, identification="cholesky")
svar_results = svar_model.identify()

# 3. IRF estrutural
sirf = svar_results.irf(steps=40)

# 4. Historical Decomposition
hd = svar_results.historical_decomposition()

# 5. Counterfactual: economia sem choques monetarios
cf = svar_results.counterfactual(exclude_shocks=["interest_rate"])
```

---

## Referencias

- Kilian, L. & Lutkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Sims, C. A. (1980). Macroeconomics and Reality. *Econometrica*, 48(1), 1--48.
