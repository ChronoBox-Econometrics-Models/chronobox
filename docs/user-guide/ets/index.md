---
title: ETS (Exponential Smoothing)
description: Familia de modelos de Suavizacao Exponencial com espaco de estados --- taxonomia, estimacao e previsao.
---

# ETS (Exponential Smoothing)

!!! info "Quick Reference"
    - **Modulo**: `chronobox.ets`
    - **Import**: `from chronobox import ETS, AutoETS, Theta`
    - **R equivalente**: `forecast::ets()` / `fable::ETS()`
    - **Modelos**: 30 combinacoes Error x Trend x Season

---

## Overview

A familia **ETS** (Error, Trend, Seasonality) fornece um framework unificado para
modelos de suavizacao exponencial usando representacao em **espaco de estados**
(state-space). Cada modelo e classificado por tres componentes:

| Componente | Opcoes | Significado |
|---|---|---|
| **Erro (E)** | A, M | Aditivo ou Multiplicativo |
| **Tendencia (T)** | N, A, A$_d$, M, M$_d$ | Nenhuma, Aditiva, Aditiva Damped, Multiplicativa, Multiplicativa Damped |
| **Sazonalidade (S)** | N, A, M | Nenhuma, Aditiva, Multiplicativa |

A combinacao desses componentes gera **30 modelos** distintos, cada um com equacoes
de observacao e transicao de estado proprias.

### Relacao com nomes classicos

Os modelos ETS englobam os metodos classicos de suavizacao exponencial:

| Modelo Classico | Notacao ETS | Parametros |
|---|---|---|
| Simple Exponential Smoothing (SES) | ETS(A,N,N) | $\alpha$ |
| Holt Linear Trend | ETS(A,A,N) | $\alpha, \beta^*$ |
| Holt Damped Trend | ETS(A,A$_d$,N) | $\alpha, \beta^*, \phi$ |
| Holt-Winters Aditivo | ETS(A,A,A) | $\alpha, \beta^*, \gamma$ |
| Holt-Winters Multiplicativo | ETS(A,A,M) | $\alpha, \beta^*, \gamma$ |
| Holt-Winters Damped Aditivo | ETS(A,A$_d$,A) | $\alpha, \beta^*, \gamma, \phi$ |
| Holt-Winters Damped Multiplicativo | ETS(A,A$_d$,M) | $\alpha, \beta^*, \gamma, \phi$ |

---

## Taxonomia ETS: Os 30 Modelos

A tabela completa cruza Erro x Tendencia x Sazonalidade:

### Erro Aditivo (A)

| Tendencia \ Sazonalidade | **N** (Nenhuma) | **A** (Aditiva) | **M** (Multiplicativa) |
|---|---|---|---|
| **N** (Nenhuma) | ETS(A,N,N) | ETS(A,N,A) | ETS(A,N,M) |
| **A** (Aditiva) | ETS(A,A,N) | ETS(A,A,A) | ETS(A,A,M) |
| **A$_d$** (Aditiva Damped) | ETS(A,A$_d$,N) | ETS(A,A$_d$,A) | ETS(A,A$_d$,M) |
| **M** (Multiplicativa) | ETS(A,M,N) | ETS(A,M,A) | ETS(A,M,M) |
| **M$_d$** (Multiplicativa Damped) | ETS(A,M$_d$,N) | ETS(A,M$_d$,A) | ETS(A,M$_d$,M) |

### Erro Multiplicativo (M)

| Tendencia \ Sazonalidade | **N** (Nenhuma) | **A** (Aditiva) | **M** (Multiplicativa) |
|---|---|---|---|
| **N** (Nenhuma) | ETS(M,N,N) | ETS(M,N,A) | ETS(M,N,M) |
| **A** (Aditiva) | ETS(M,A,N) | ETS(M,A,A) | ETS(M,A,M) |
| **A$_d$** (Aditiva Damped) | ETS(M,A$_d$,N) | ETS(M,A$_d$,A) | ETS(M,A$_d$,M) |
| **M** (Multiplicativa) | ETS(M,M,N) | ETS(M,M,A) | ETS(M,M,M) |
| **M$_d$** (Multiplicativa Damped) | ETS(M,M$_d$,N) | ETS(M,M$_d$,A) | ETS(M,M$_d$,M) |

!!! warning "Modelos com tendencia multiplicativa"
    Os modelos com tendencia multiplicativa (M, M$_d$) podem ser **instáveis**
    e gerar previsoes explosivas. Muitas implementacoes restringem ou excluem
    essas combinacoes por padrao. Use com cautela.

---

## Representacao em Espaco de Estados

Todos os modelos ETS possuem uma representacao state-space da forma:

$$
y_t = h(\mathbf{x}_{t-1}) + k(\mathbf{x}_{t-1})\,\varepsilon_t
$$

$$
\mathbf{x}_t = f(\mathbf{x}_{t-1}) + g(\mathbf{x}_{t-1})\,\varepsilon_t
$$

onde:

- $y_t$ e a observacao no tempo $t$
- $\mathbf{x}_t$ e o vetor de estados (nivel, tendencia, sazonalidade)
- $\varepsilon_t$ e o erro (inovacao)
- $h(\cdot)$ e a equacao de **observacao**
- $f(\cdot)$ e a equacao de **transicao**

Para modelos com **erro aditivo**, $k(\mathbf{x}_{t-1}) = 1$.
Para modelos com **erro multiplicativo**, $k(\mathbf{x}_{t-1}) = \mu_t$ (media condicional).

### Relacao com ARIMA

Muitos modelos ETS possuem equivalentes ARIMA:

| ETS | ARIMA equivalente |
|---|---|
| ETS(A,N,N) | ARIMA(0,1,1) |
| ETS(A,A,N) | ARIMA(0,2,2) |
| ETS(A,A$_d$,N) | ARIMA(1,1,2) |
| ETS(A,N,A) | ARIMA(0,1,$m$+1)(0,1,0)$_m$ |
| ETS(A,A,A) | ARIMA(0,1,$m$+1)(0,1,0)$_m$ |

!!! note "Equivalencia limitada"
    A equivalencia ETS-ARIMA so vale para modelos com **erro aditivo** e
    **sem tendencia multiplicativa**. Modelos com erro multiplicativo nao
    possuem equivalente ARIMA direto.

---

## Estimacao

A estimacao dos modelos ETS e feita por **maxima verossimilhanca** (MLE),
maximizando a log-verossimilhanca condicional aos estados iniciais.

O vetor de parametros inclui:

- Parametros de suavizacao: $\alpha$, $\beta^*$, $\gamma$, $\phi$
- Estados iniciais: $l_0$, $b_0$, $s_0, s_{-1}, \ldots, s_{-m+1}$

A otimizacao e feita numericamente, e os estados iniciais podem ser estimados
conjuntamente ou inicializados por heuristicas.

---

## Paginas desta Secao

| Pagina | Descricao |
|---|---|
| [Simple Exponential Smoothing](simple.md) | SES --- ETS(A,N,N) e ETS(M,N,N) |
| [Holt Linear Trend](holt.md) | Holt com e sem damping --- ETS(·,A,N), ETS(·,A$_d$,N) |
| [Holt-Winters Seasonal](holt-winters.md) | Metodo sazonal completo --- ETS(·,·,A) e ETS(·,·,M) |
| [ETS Taxonomy](ets-taxonomy.md) | Todos os 30 modelos com equacoes state-space |
| [Auto-ETS](auto-ets.md) | Selecao automatica entre os 30 modelos |
| [Theta Method](theta.md) | Metodo Theta e sua relacao com ETS |

---

## Referencias

- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008).
  *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
- Hyndman, R. J. & Athanasopoulos, G. (2021).
  *Forecasting: Principles and Practice*. 3rd ed. OTexts.
- Gardner, E. S. (2006). Exponential smoothing: The state of the art --- Part II.
  *International Journal of Forecasting*, 22(4), 637--666.
