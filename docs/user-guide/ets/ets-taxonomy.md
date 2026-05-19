---
title: ETS Taxonomy
description: Taxonomia completa dos 30 modelos ETS com equacoes state-space, condicoes de estabilidade e guia de selecao.
---

# ETS Taxonomy --- Os 30 Modelos

!!! info "Quick Reference"
    - **Classe**: `chronobox.ETS`
    - **Import**: `from chronobox import ETS`
    - **Modelos**: 30 combinacoes = 2 (Erro) x 5 (Tendencia) x 3 (Sazonalidade)
    - **R equivalente**: `forecast::ets(y, model="ZZZ")` (selecao automatica)

---

## Overview

A taxonomia ETS classifica todos os modelos de suavizacao exponencial em uma
grade tridimensional:

- **Erro (E)**: Aditivo (A) ou Multiplicativo (M) --- 2 opcoes
- **Tendencia (T)**: Nenhuma (N), Aditiva (A), Aditiva Damped (A$_d$), Multiplicativa (M), Multiplicativa Damped (M$_d$) --- 5 opcoes
- **Sazonalidade (S)**: Nenhuma (N), Aditiva (A), Multiplicativa (M) --- 3 opcoes

Total: $2 \times 5 \times 3 = 30$ modelos.

Cada modelo possui uma forma **state-space** com equacao de observacao e
equacoes de transicao, permitindo estimacao por maxima verossimilhanca e
calculo analitico de intervalos de previsao.

---

## Tabela Completa dos 30 Modelos

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

---

## Formulacao Matematica

### Notacao Geral

Todos os modelos ETS seguem a forma state-space:

$$
y_t = w(\mathbf{x}_{t-1}) + r(\mathbf{x}_{t-1})\,\varepsilon_t
$$

$$
\mathbf{x}_t = f(\mathbf{x}_{t-1}) + g(\mathbf{x}_{t-1})\,\varepsilon_t
$$

onde:

- $\mathbf{x}_t = (l_t, b_t, s_t, s_{t-1}, \ldots, s_{t-m+1})'$ e o vetor de estados
- $w(\cdot)$ e a funcao de observacao
- $r(\cdot)$ e o fator de escala do erro
- $f(\cdot)$ e a funcao de transicao
- $g(\cdot)$ e o vetor de ganho

Para modelos com **erro aditivo**: $r(\mathbf{x}_{t-1}) = 1$

Para modelos com **erro multiplicativo**: $r(\mathbf{x}_{t-1}) = \mu_t$ (media condicional)

### Definicoes auxiliares

Para simplificar as equacoes, definimos:

$$
\mu_t = w(\mathbf{x}_{t-1})
$$

que e a media condicional (previsao um passo a frente).

---

## Equacoes State-Space por Modelo

### Modelos sem Sazonalidade (S = N)

#### ETS(A,N,N) --- SES Aditivo

$$
y_t = l_{t-1} + \varepsilon_t
$$

$$
l_t = l_{t-1} + \alpha\,\varepsilon_t
$$

#### ETS(M,N,N) --- SES Multiplicativo

$$
y_t = l_{t-1}(1 + \varepsilon_t)
$$

$$
l_t = l_{t-1}(1 + \alpha\,\varepsilon_t)
$$

#### ETS(A,A,N) --- Holt Aditivo

$$
y_t = l_{t-1} + b_{t-1} + \varepsilon_t
$$

$$
l_t = l_{t-1} + b_{t-1} + \alpha\,\varepsilon_t
$$

$$
b_t = b_{t-1} + \beta\,\varepsilon_t
$$

#### ETS(A,A$_d$,N) --- Holt Damped Aditivo

$$
y_t = l_{t-1} + \phi\, b_{t-1} + \varepsilon_t
$$

$$
l_t = l_{t-1} + \phi\, b_{t-1} + \alpha\,\varepsilon_t
$$

$$
b_t = \phi\, b_{t-1} + \beta\,\varepsilon_t
$$

#### ETS(M,A,N) --- Holt Multiplicativo

$$
y_t = (l_{t-1} + b_{t-1})(1 + \varepsilon_t)
$$

$$
l_t = (l_{t-1} + b_{t-1})(1 + \alpha\,\varepsilon_t)
$$

$$
b_t = b_{t-1} + \beta(l_{t-1} + b_{t-1})\varepsilon_t
$$

#### ETS(M,A$_d$,N) --- Holt Damped Multiplicativo

$$
y_t = (l_{t-1} + \phi\, b_{t-1})(1 + \varepsilon_t)
$$

$$
l_t = (l_{t-1} + \phi\, b_{t-1})(1 + \alpha\,\varepsilon_t)
$$

$$
b_t = \phi\, b_{t-1} + \beta(l_{t-1} + \phi\, b_{t-1})\varepsilon_t
$$

#### ETS(A,M,N) --- Tendencia Multiplicativa, Erro Aditivo

$$
y_t = l_{t-1}\, b_{t-1} + \varepsilon_t
$$

$$
l_t = l_{t-1}\, b_{t-1} + \alpha\,\varepsilon_t
$$

$$
b_t = b_{t-1} + \beta\,\varepsilon_t / l_{t-1}
$$

#### ETS(A,M$_d$,N) --- Tendencia Multiplicativa Damped, Erro Aditivo

$$
y_t = l_{t-1}\, b_{t-1}^\phi + \varepsilon_t
$$

$$
l_t = l_{t-1}\, b_{t-1}^\phi + \alpha\,\varepsilon_t
$$

$$
b_t = b_{t-1}^\phi + \beta\,\varepsilon_t / l_{t-1}
$$

#### ETS(M,M,N) --- Tendencia Multiplicativa, Erro Multiplicativo

$$
y_t = l_{t-1}\, b_{t-1}(1 + \varepsilon_t)
$$

$$
l_t = l_{t-1}\, b_{t-1}(1 + \alpha\,\varepsilon_t)
$$

$$
b_t = b_{t-1}(1 + \beta\,\varepsilon_t)
$$

#### ETS(M,M$_d$,N) --- Tendencia Multiplicativa Damped, Erro Multiplicativo

$$
y_t = l_{t-1}\, b_{t-1}^\phi(1 + \varepsilon_t)
$$

$$
l_t = l_{t-1}\, b_{t-1}^\phi(1 + \alpha\,\varepsilon_t)
$$

$$
b_t = b_{t-1}^\phi(1 + \beta\,\varepsilon_t)
$$

---

### Modelos com Sazonalidade Aditiva (S = A)

#### ETS(A,N,A) --- SES com Sazonalidade Aditiva

$$
y_t = l_{t-1} + s_{t-m} + \varepsilon_t
$$

$$
l_t = l_{t-1} + \alpha\,\varepsilon_t
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t
$$

#### ETS(A,A,A) --- Holt-Winters Aditivo

$$
y_t = l_{t-1} + b_{t-1} + s_{t-m} + \varepsilon_t
$$

$$
l_t = l_{t-1} + b_{t-1} + \alpha\,\varepsilon_t
$$

$$
b_t = b_{t-1} + \beta\,\varepsilon_t
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t
$$

#### ETS(A,A$_d$,A) --- Holt-Winters Aditivo Damped

$$
y_t = l_{t-1} + \phi\, b_{t-1} + s_{t-m} + \varepsilon_t
$$

$$
l_t = l_{t-1} + \phi\, b_{t-1} + \alpha\,\varepsilon_t
$$

$$
b_t = \phi\, b_{t-1} + \beta\,\varepsilon_t
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t
$$

#### ETS(A,M,A) --- Tendencia Multiplicativa com Sazonalidade Aditiva

$$
y_t = l_{t-1}\, b_{t-1} + s_{t-m} + \varepsilon_t
$$

$$
l_t = l_{t-1}\, b_{t-1} + \alpha\,\varepsilon_t
$$

$$
b_t = b_{t-1} + \beta\,\varepsilon_t / l_{t-1}
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t
$$

#### ETS(A,M$_d$,A) --- Tendencia Multiplicativa Damped com Sazonalidade Aditiva

$$
y_t = l_{t-1}\, b_{t-1}^\phi + s_{t-m} + \varepsilon_t
$$

$$
l_t = l_{t-1}\, b_{t-1}^\phi + \alpha\,\varepsilon_t
$$

$$
b_t = b_{t-1}^\phi + \beta\,\varepsilon_t / l_{t-1}
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t
$$

#### ETS(M,N,A) --- Erro Multiplicativo com Sazonalidade Aditiva

$$
y_t = (l_{t-1} + s_{t-m})(1 + \varepsilon_t)
$$

$$
l_t = l_{t-1} + \alpha(l_{t-1} + s_{t-m})\varepsilon_t
$$

$$
s_t = s_{t-m} + \gamma(l_{t-1} + s_{t-m})\varepsilon_t
$$

#### ETS(M,A,A) --- Holt com Sazonalidade Aditiva, Erro Multiplicativo

$$
y_t = (l_{t-1} + b_{t-1} + s_{t-m})(1 + \varepsilon_t)
$$

$$
l_t = l_{t-1} + b_{t-1} + \alpha(l_{t-1} + b_{t-1} + s_{t-m})\varepsilon_t
$$

$$
b_t = b_{t-1} + \beta(l_{t-1} + b_{t-1} + s_{t-m})\varepsilon_t
$$

$$
s_t = s_{t-m} + \gamma(l_{t-1} + b_{t-1} + s_{t-m})\varepsilon_t
$$

#### ETS(M,A$_d$,A) --- Holt Damped com Sazonalidade Aditiva, Erro Multiplicativo

$$
y_t = (l_{t-1} + \phi\, b_{t-1} + s_{t-m})(1 + \varepsilon_t)
$$

$$
l_t = l_{t-1} + \phi\, b_{t-1} + \alpha(l_{t-1} + \phi\, b_{t-1} + s_{t-m})\varepsilon_t
$$

$$
b_t = \phi\, b_{t-1} + \beta(l_{t-1} + \phi\, b_{t-1} + s_{t-m})\varepsilon_t
$$

$$
s_t = s_{t-m} + \gamma(l_{t-1} + \phi\, b_{t-1} + s_{t-m})\varepsilon_t
$$

#### ETS(M,M,A)

$$
y_t = (l_{t-1}\, b_{t-1} + s_{t-m})(1 + \varepsilon_t)
$$

$$
l_t = l_{t-1}\, b_{t-1} + \alpha(l_{t-1}\, b_{t-1} + s_{t-m})\varepsilon_t
$$

$$
b_t = b_{t-1} + \beta(l_{t-1}\, b_{t-1} + s_{t-m})\varepsilon_t / l_{t-1}
$$

$$
s_t = s_{t-m} + \gamma(l_{t-1}\, b_{t-1} + s_{t-m})\varepsilon_t
$$

#### ETS(M,M$_d$,A)

$$
y_t = (l_{t-1}\, b_{t-1}^\phi + s_{t-m})(1 + \varepsilon_t)
$$

$$
l_t = l_{t-1}\, b_{t-1}^\phi + \alpha(l_{t-1}\, b_{t-1}^\phi + s_{t-m})\varepsilon_t
$$

$$
b_t = b_{t-1}^\phi + \beta(l_{t-1}\, b_{t-1}^\phi + s_{t-m})\varepsilon_t / l_{t-1}
$$

$$
s_t = s_{t-m} + \gamma(l_{t-1}\, b_{t-1}^\phi + s_{t-m})\varepsilon_t
$$

---

### Modelos com Sazonalidade Multiplicativa (S = M)

#### ETS(A,N,M) --- SES com Sazonalidade Multiplicativa

$$
y_t = l_{t-1}\, s_{t-m} + \varepsilon_t
$$

$$
l_t = l_{t-1} + \alpha\,\varepsilon_t / s_{t-m}
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t / l_{t-1}
$$

#### ETS(A,A,M) --- Holt-Winters Multiplicativo

$$
y_t = (l_{t-1} + b_{t-1})\, s_{t-m} + \varepsilon_t
$$

$$
l_t = l_{t-1} + b_{t-1} + \alpha\,\varepsilon_t / s_{t-m}
$$

$$
b_t = b_{t-1} + \beta\,\varepsilon_t / s_{t-m}
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t / (l_{t-1} + b_{t-1})
$$

#### ETS(A,A$_d$,M) --- Holt-Winters Multiplicativo Damped

$$
y_t = (l_{t-1} + \phi\, b_{t-1})\, s_{t-m} + \varepsilon_t
$$

$$
l_t = l_{t-1} + \phi\, b_{t-1} + \alpha\,\varepsilon_t / s_{t-m}
$$

$$
b_t = \phi\, b_{t-1} + \beta\,\varepsilon_t / s_{t-m}
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t / (l_{t-1} + \phi\, b_{t-1})
$$

#### ETS(A,M,M)

$$
y_t = l_{t-1}\, b_{t-1}\, s_{t-m} + \varepsilon_t
$$

$$
l_t = l_{t-1}\, b_{t-1} + \alpha\,\varepsilon_t / s_{t-m}
$$

$$
b_t = b_{t-1} + \beta\,\varepsilon_t / (l_{t-1}\, s_{t-m})
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t / (l_{t-1}\, b_{t-1})
$$

#### ETS(A,M$_d$,M)

$$
y_t = l_{t-1}\, b_{t-1}^\phi\, s_{t-m} + \varepsilon_t
$$

$$
l_t = l_{t-1}\, b_{t-1}^\phi + \alpha\,\varepsilon_t / s_{t-m}
$$

$$
b_t = b_{t-1}^\phi + \beta\,\varepsilon_t / (l_{t-1}\, s_{t-m})
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t / (l_{t-1}\, b_{t-1}^\phi)
$$

#### ETS(M,N,M)

$$
y_t = l_{t-1}\, s_{t-m}(1 + \varepsilon_t)
$$

$$
l_t = l_{t-1}(1 + \alpha\,\varepsilon_t)
$$

$$
s_t = s_{t-m}(1 + \gamma\,\varepsilon_t)
$$

#### ETS(M,A,M)

$$
y_t = (l_{t-1} + b_{t-1})\, s_{t-m}(1 + \varepsilon_t)
$$

$$
l_t = (l_{t-1} + b_{t-1})(1 + \alpha\,\varepsilon_t)
$$

$$
b_t = b_{t-1} + \beta(l_{t-1} + b_{t-1})\varepsilon_t
$$

$$
s_t = s_{t-m}(1 + \gamma\,\varepsilon_t)
$$

#### ETS(M,A$_d$,M)

$$
y_t = (l_{t-1} + \phi\, b_{t-1})\, s_{t-m}(1 + \varepsilon_t)
$$

$$
l_t = (l_{t-1} + \phi\, b_{t-1})(1 + \alpha\,\varepsilon_t)
$$

$$
b_t = \phi\, b_{t-1} + \beta(l_{t-1} + \phi\, b_{t-1})\varepsilon_t
$$

$$
s_t = s_{t-m}(1 + \gamma\,\varepsilon_t)
$$

#### ETS(M,M,M)

$$
y_t = l_{t-1}\, b_{t-1}\, s_{t-m}(1 + \varepsilon_t)
$$

$$
l_t = l_{t-1}\, b_{t-1}(1 + \alpha\,\varepsilon_t)
$$

$$
b_t = b_{t-1}(1 + \beta\,\varepsilon_t)
$$

$$
s_t = s_{t-m}(1 + \gamma\,\varepsilon_t)
$$

#### ETS(M,M$_d$,M)

$$
y_t = l_{t-1}\, b_{t-1}^\phi\, s_{t-m}(1 + \varepsilon_t)
$$

$$
l_t = l_{t-1}\, b_{t-1}^\phi(1 + \alpha\,\varepsilon_t)
$$

$$
b_t = b_{t-1}^\phi(1 + \beta\,\varepsilon_t)
$$

$$
s_t = s_{t-m}(1 + \gamma\,\varepsilon_t)
$$

---

## Condicoes de Estabilidade

As condicoes de estabilidade garantem que os estados nao divergem. Elas sao
expressas como restricoes sobre os parametros de suavizacao.

### Modelos com Tendencia Aditiva (Erro Aditivo)

Para ETS(A,A,N), os parametros devem satisfazer:

$$
0 < \alpha < 1, \quad 0 < \beta < \alpha, \quad 0 < \beta^* < 1
$$

Para ETS(A,A$_d$,N), adicionalmente:

$$
0 < \phi < 1
$$

### Modelos com Sazonalidade

Para modelos com sazonalidade aditiva:

$$
0 < \gamma < 1 - \alpha
$$

Para modelos com sazonalidade multiplicativa (e erro aditivo):

$$
0 < \gamma^* < 1 - \alpha
$$

### Classificacao de Estabilidade

| Modelo | Estabilidade |
|---|---|
| ETS(A,N,N), ETS(M,N,N) | Sempre estavel |
| ETS(A,A,N), ETS(A,A$_d$,N) | Estavel com restricoes padrao |
| ETS(A,A,A), ETS(A,A$_d$,A) | Estavel com restricoes padrao |
| ETS(A,A,M), ETS(A,A$_d$,M) | Estavel com restricoes padrao |
| ETS(A,M,N), ETS(A,M$_d$,N) | Potencialmente instavel |
| ETS(M,M,N), ETS(M,M$_d$,N) | Potencialmente instavel |
| ETS(A,M,M), ETS(M,M,M) | Potencialmente instavel |

!!! warning "Modelos instáveis"
    Modelos com **tendencia multiplicativa** podem gerar previsoes que
    divergem exponencialmente. Muitas implementacoes (incluindo `forecast::ets()`
    do R) restringem esses modelos por padrao. Use o parametro `restrict=True`
    no Auto-ETS para excluir modelos instáveis.

---

## Contagem de Parametros

| Componentes | Parametros de suavizacao | Estados iniciais | Total |
|---|---|---|---|
| ETS(·,N,N) | $\alpha$ (1) | $l_0$ (1) | $2 + \sigma^2 = 3$ |
| ETS(·,A,N) | $\alpha, \beta^*$ (2) | $l_0, b_0$ (2) | $4 + \sigma^2 = 5$ |
| ETS(·,A$_d$,N) | $\alpha, \beta^*, \phi$ (3) | $l_0, b_0$ (2) | $5 + \sigma^2 = 6$ |
| ETS(·,N,A/M) | $\alpha, \gamma$ (2) | $l_0, s_0 \ldots s_{-m+1}$ ($1+m$) | $3+m + \sigma^2$ |
| ETS(·,A,A/M) | $\alpha, \beta^*, \gamma$ (3) | $l_0, b_0, s_0 \ldots s_{-m+1}$ ($2+m$) | $5+m + \sigma^2$ |
| ETS(·,A$_d$,A/M) | $\alpha, \beta^*, \gamma, \phi$ (4) | $l_0, b_0, s_0 \ldots s_{-m+1}$ ($2+m$) | $6+m + \sigma^2$ |

---

## Quick Example

```python
from chronobox import ETS
from chronobox.datasets import load_airline

y = load_airline()

# Ajustar qualquer modelo da taxonomia especificando os componentes
model = ETS(error='M', trend='A', seasonal='M',
            seasonal_periods=12, damped=True)
results = model.fit(y)

print(results.summary())
print(f"Modelo: ETS(M,Ad,M)")
print(f"AIC: {results.aic:.2f}")
print(f"Parametros: {results.params}")
```

---

## Guia de Selecao

### Arvore de Decisao

```text
Serie temporal
├── Sem tendencia, sem sazonalidade
│   └── ETS(·,N,N) → SES
├── Com tendencia, sem sazonalidade
│   ├── Tendencia persistente → ETS(·,A,N) → Holt
│   └── Tendencia se atenuando → ETS(·,Ad,N) → Holt Damped
└── Com tendencia e sazonalidade
    ├── Amplitude sazonal constante → ETS(·,·,A) → HW Aditivo
    └── Amplitude sazonal crescente → ETS(·,·,M) → HW Multiplicativo
```

### Criterio para Tipo de Erro

| Criterio | Erro Aditivo (A) | Erro Multiplicativo (M) |
|---|---|---|
| Variancia do erro | Constante | Proporcional ao nivel |
| Dados negativos | Permite | Requer dados positivos |
| Log-verossimilhanca | Gaussiana | Condicional |
| Intervalos de previsao | Simetricos | Assimetricos |

!!! tip "Na pratica"
    Use **Auto-ETS** para selecao automatica entre os 30 modelos via
    criterios de informacao. A selecao manual e util para impor restricoes
    teoricas (ex: forcar sazonalidade multiplicativa em dados de vendas).

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import ETS

    # Especificar qualquer modelo da taxonomia
    model = ETS(error='A', trend='A', seasonal='M',
                seasonal_periods=12, damped=True)
    results = model.fit(y)
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    # Modelo especifico: AAM damped
    fit <- ets(y, model = "AAM", damped = TRUE)

    # Selecao automatica
    fit <- ets(y, model = "ZZZ")
    ```

=== "fable (R)"

    ```r
    library(fable)

    # Modelo especifico
    fit <- y_tsibble |>
      model(ETS(y ~ error("A") + trend("Ad") + season("M")))

    # Selecao automatica
    fit <- y_tsibble |>
      model(ETS(y))
    ```

**Mapeamento de notacao**:

| chronobox | forecast (R) | Descricao |
|---|---|---|
| `error='A'` | `"A"` (1a letra) | Erro aditivo |
| `error='M'` | `"M"` (1a letra) | Erro multiplicativo |
| `trend='A'` | `"A"` (2a letra) | Tendencia aditiva |
| `trend='N'` | `"N"` (2a letra) | Sem tendencia |
| `seasonal='M'` | `"M"` (3a letra) | Sazonalidade multiplicativa |
| `damped=True` | `damped=TRUE` | Tendencia amortecida |
| --- | `"Z"` | Selecao automatica |

---

## Referencias

- Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002).
  A state space framework for automatic forecasting using exponential
  smoothing methods. *International Journal of Forecasting*, 18(3), 439--454.
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008).
  *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
- Pegels, C. C. (1969). Exponential forecasting: Some new variations.
  *Management Science*, 15(5), 311--315.
- Taylor, J. W. (2003). Exponential smoothing with a damped multiplicative
  trend. *International Journal of Forecasting*, 19(4), 715--725.
