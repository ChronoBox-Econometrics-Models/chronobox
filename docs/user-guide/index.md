---
title: User Guide
description: Guias completos para todos os modelos e ferramentas do chronobox.
---

# User Guide

Bem-vindo ao User Guide do **chronobox**. Aqui voce encontra documentacao detalhada
de cada modelo, com formulacao matematica, exemplos de codigo, interpretacao de
resultados e diagnosticos.

---

## Modelos Univariados

Modelos para series temporais individuais --- previsao, decomposicao e analise de
componentes.

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **ARIMA**

    ---

    Familia completa: ARIMA(p,d,q), SARIMA, ARFIMA e Auto-ARIMA.

    [:octicons-arrow-right-24: ARIMA](arima/index.md)

-   :material-sine-wave:{ .lg .middle } **ETS**

    ---

    Suavizacao Exponencial: modelos aditivos, multiplicativos e amortecidos.

    [:octicons-arrow-right-24: ETS](ets/index.md)

-   :material-chart-bar:{ .lg .middle } **Decomposicao**

    ---

    STL e X-13 ARIMA-SEATS para decomposicao sazonal.

    [:octicons-arrow-right-24: Decomposicao](decomposition/index.md)

</div>

---

## Modelos Multivariados

Modelos para sistemas de multiplas series temporais --- transmissao de choques,
causalidade e cointegra cao.

<div class="grid cards" markdown>

-   :material-vector-polyline:{ .lg .middle } **VAR & VECM**

    ---

    Vetores Autoregressivos, Modelos de Correcao de Erros e causalidade de Granger.

    [:octicons-arrow-right-24: VAR & VECM](var/index.md)

-   :material-graph:{ .lg .middle } **SVAR & Avancados**

    ---

    VAR Estrutural, Bayesiano (BVAR), FAVAR, TVP-VAR e GVAR.

    [:octicons-arrow-right-24: SVAR & Avancados](svar/index.md)

</div>

---

## Ferramentas Analiticas

Filtros, testes e ferramentas complementares para analise de series temporais.

<div class="grid cards" markdown>

-   :material-filter:{ .lg .middle } **Filtros Economicos**

    ---

    Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald, Hamilton e Beveridge-Nelson.

    [:octicons-arrow-right-24: Filtros](filters/index.md)

-   :material-swap-horizontal:{ .lg .middle } **ARDL & ECM**

    ---

    Modelos Autoregressivos de Defasagens Distribuidas e Correcao de Erros.

    [:octicons-arrow-right-24: ARDL](ardl/index.md)

-   :material-connection:{ .lg .middle } **Spillover**

    ---

    Analise de transbordamento e conectividade entre series.

    [:octicons-arrow-right-24: Spillover](spillover/index.md)

-   :material-flask:{ .lg .middle } **Experiment**

    ---

    Workflow para comparacao sistematica de modelos.

    [:octicons-arrow-right-24: Experiment](experiment/index.md)

</div>

---

## Como usar este guia

Cada pagina de modelo segue uma estrutura padronizada:

| Secao | Conteudo |
|---|---|
| **Overview** | O que e o modelo e quando usa-lo |
| **Formulacao Matematica** | Equacoes e notacao formal |
| **Quick Example** | Codigo minimo funcional |
| **Guia Detalhado** | Parametros, opcoes e configuracao |
| **Interpretacao** | Como ler e interpretar os resultados |
| **Diagnosticos** | Testes de validacao do modelo |
| **Equivalentes R** | Comparacao com pacotes R |
| **Referencias** | Papers e livros de referencia |

!!! tip "Primeira vez?"
    Se voce esta comecando, siga a ordem recomendada no [Getting Started](../getting-started/index.md)
    antes de mergulhar nos guias detalhados.
