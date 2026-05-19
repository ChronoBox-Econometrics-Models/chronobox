---
title: ARDL & Error Correction
description: Modelos ARDL (Autoregressive Distributed Lag) e ECM (Error Correction Model) --- bounds test, cointegracao com ordens mistas e multiplicadores dinamicos.
---

# ARDL & Error Correction

O ARDL (Autoregressive Distributed Lag) e uma classe de modelos que permite
analisar relacoes de **curto e longo prazo** entre variaveis, mesmo quando elas
possuem ordens de integracao diferentes. Combinado com o bounds test de Pesaran,
Shin & Smith (2001), o ARDL oferece uma abordagem flexivel para testar
cointegracao sem as restricoes do metodo de Johansen.

---

## Por que ARDL?

Na pratica economica, e comum trabalhar com variaveis que possuem ordens de
integracao mistas --- algumas sao I(0) e outras sao I(1). O metodo de Johansen
exige que **todas** as variaveis sejam I(1), o que limita sua aplicabilidade.

O ARDL resolve esse problema:

| Caracteristica | Johansen | ARDL |
|---|---|---|
| Ordens de integracao | Todas I(1) | Mistura de I(0) e I(1) |
| Tamanho amostral | Requer amostras grandes | Funciona bem em amostras pequenas |
| Numero de variaveis | Multiplas equacoes (sistema) | Equacao unica (mais eficiente) |
| Teste de cointegracao | Trace e max-eigenvalue | Bounds test (F-statistic) |
| Exogeneidade | Todas endogenas | Permite regressores exogenos |

!!! warning "Restricao importante"
    O ARDL **nao** e valido quando alguma variavel e I(2) ou de ordem superior.
    Antes de estimar, verifique a ordem de integracao com testes ADF/KPSS.

---

## ARDL e ECM: a conexao

O modelo ARDL e o ECM (Error Correction Model) sao duas faces da mesma moeda.
Todo ARDL pode ser **reparametrizado** em forma ECM, separando a dinamica em:

- **Relacao de longo prazo**: equilibrio entre as variaveis em niveis
- **Dinamica de curto prazo**: ajustamento via diferencas e velocidade de correcao

```
ARDL(p, q) em niveis  ──reparametrizacao──►  ECM com correcao de erros
         │                                            │
    y_t, x_t em niveis                      Δy_t, Δx_t + termo de erro
         │                                            │
    bounds test para                         α = velocidade de ajuste
    testar cointegracao                      β = relacao de longo prazo
```

---

## Modelos da Secao

<div class="grid cards" markdown>

-   :material-chart-timeline:{ .lg .middle } **ARDL**

    ---

    Modelo autoregressivo com defasagens distribuidas: estimacao, selecao
    de lags e bounds test de Pesaran et al. (2001).

    [:octicons-arrow-right-24: ARDL](ardl.md)

-   :material-swap-horizontal:{ .lg .middle } **Error Correction Model (ECM)**

    ---

    Reparametrizacao do ARDL em forma de correcao de erros: velocidade de
    ajuste, multiplicadores de curto e longo prazo.

    [:octicons-arrow-right-24: ECM](ecm.md)

</div>

---

## Fluxo de Trabalho Tipico

```python
from chronobox import ARDL
from chronobox.tests_stat import adf_test, kpss_test

import pandas as pd

# 1. Carregar dados
data = pd.DataFrame({
    "consumption": consumption_series,
    "income": income_series,
    "wealth": wealth_series,
})

# 2. Verificar ordem de integracao (deve ser I(0) ou I(1), nunca I(2))
for col in data.columns:
    adf = adf_test(data[col])
    print(f"{col}: ADF p-value = {adf.pvalue:.4f}")

# 3. Estimar ARDL com selecao automatica de lags
model = ARDL(lags=4, exog_lags=4, trend='c', ic='aic')
results = model.fit(y=data["consumption"], exog=data[["income", "wealth"]])
print(results.summary())

# 4. Bounds test para cointegracao
bounds = results.bounds_test()
print(bounds.summary())

# 5. Se cointegrado, extrair ECM
ecm = results.to_ecm()
print(ecm.summary())

# 6. Multiplicadores dinamicos
dm = ecm.dynamic_multipliers(steps=20)
```

---

## Referencias

- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). Bounds testing approaches to the
  analysis of level relationships. *Journal of Applied Econometrics*, 16(3), 289--326.
- Pesaran, M. H. & Shin, Y. (1999). An autoregressive distributed-lag modelling approach
  to cointegration analysis. In S. Strom (Ed.), *Econometrics and Economic Theory in the
  20th Century: The Ragnar Frisch Centennial Symposium*. Cambridge University Press.
- Kripfganz, S. & Schneider, D. C. (2023). ardl: Estimating autoregressive distributed
  lag and equilibrium correction models. *The Stata Journal*, 23(4), 983--1019.
