---
title: Theta Method
description: Metodo Theta de Assimakopoulos & Nikolopoulos (2000) e sua relacao com ETS.
---

# Theta Method

!!! info "Quick Reference"
    - **Classe**: `chronobox.Theta`
    - **Import**: `from chronobox import Theta`
    - **R equivalente**: `forecast::thetaf(y)` / `fable::THETA(y)`
    - **Equivalencia**: ETS(A,A,N) com $\alpha$ especifico + drift

---

## Overview

O **metodo Theta** foi proposto por Assimakopoulos & Nikolopoulos (2000) e
ganhou destaque ao vencer a competicao M3 de previsao, superando metodos
mais complexos como ARIMA e modelos neurais.

A ideia central e decompor a serie em **linhas Theta** --- transformacoes
que amplificam ou atenuam a curvatura da serie --- e combinar previsoes
dessas linhas para obter a previsao final.

Hyndman & Billah (2003) demonstraram que o metodo Theta padrao (com $\theta = 0$
e $\theta = 2$) e equivalente a um **SES com drift** --- essencialmente um
caso especial de ETS(A,A,N) com restricoes nos parametros.

### Quando usar

- Previsao automatica de curto prazo sem sazonalidade
- Como benchmark robusto (vencedor da M3)
- Quando se deseja um metodo simples mas competitivo

---

## Formulacao Matematica

### Linhas Theta

Dada uma serie $y_t$, a **linha Theta** com parametro $\theta$ e definida pela
relacao com a segunda diferenca:

$$
\Delta^2 z_t(\theta) = \theta \cdot \Delta^2 y_t
$$

onde $\Delta^2 y_t = y_t - 2y_{t-1} + y_{t-2}$.

A solucao geral e:

$$
z_t(\theta) = \theta\, y_t + (1 - \theta)\, \hat{\ell}_t
$$

onde $\hat{\ell}_t$ e a reta de regressao linear ajustada a $y_t$.

**Interpretacao geometrica**:

- $\theta = 0$: $z_t(0) = \hat{\ell}_t$ (tendencia linear pura)
- $\theta = 1$: $z_t(1) = y_t$ (serie original)
- $\theta = 2$: $z_t(2) = 2y_t - \hat{\ell}_t$ (curvatura amplificada)
- $\theta > 1$: amplifica a curvatura
- $0 < \theta < 1$: suaviza a curvatura

### Metodo Theta Padrao

O metodo padrao usa duas linhas:

1. **$z_t(0)$**: a reta de tendencia linear (extrapolada para previsao)
2. **$z_t(2)$**: serie com curvatura amplificada (prevista via SES)

A previsao final e a **media** das previsoes das duas linhas:

$$
\hat{y}_{T+h|T} = \frac{1}{2}\left[\hat{z}_{T+h}(0) + \hat{z}_{T+h}(2)\right]
$$

onde:

- $\hat{z}_{T+h}(0) = a + b(T + h)$ (extrapolacao linear)
- $\hat{z}_{T+h}(2) = l_T$ (previsao SES, flat)

### Equivalencia com ETS(A,A,N)

Hyndman & Billah (2003) mostraram que o metodo Theta padrao e equivalente a:

$$
\hat{y}_{T+h|T} = l_T + h \cdot \frac{b}{2}
$$

onde $l_T$ e o nivel da SES aplicada a serie original e $b$ e o coeficiente
angular da regressao linear.

Isso corresponde a um **ETS(A,A,N)** com:

- $\alpha$ estimado via SES
- $b_T = b/2$ (metade do drift linear)
- $\beta^* = 0$ (tendencia fixa, nao atualizada)

### State-Space

A representacao state-space do metodo Theta:

**Equacao de observacao**:

$$
y_t = l_{t-1} + b + \varepsilon_t
$$

**Equacao de transicao (nivel)**:

$$
l_t = l_{t-1} + b + \alpha\,\varepsilon_t
$$

onde $b$ e o drift (fixo, estimado por regressao linear) e $\varepsilon_t \sim \text{NID}(0, \sigma^2)$.

**Previsao**:

$$
\hat{y}_{T+h|T} = l_T + h \cdot b
$$

com intervalos de previsao:

$$
\hat{y}_{T+h|T} \pm z_{\alpha/2}\,\sigma\sqrt{1 + (h-1)\alpha^2}
$$

### Parametros

| Parametro | Dominio | Descricao |
|---|---|---|
| $\alpha$ | $(0, 1)$ | Suavizacao do nivel (estimado via SES) |
| $b$ | $\mathbb{R}$ | Drift (metade do slope da regressao linear) |
| $l_0$ | $\mathbb{R}$ | Estado inicial do nivel |

---

## Quick Example

```python
from chronobox import Theta
from chronobox.datasets import load_oil

# Carregar dados anuais
y = load_oil()

# Ajustar metodo Theta
model = Theta()
results = model.fit(y)

# Resumo
print(results.summary())

# Previsao 5 anos a frente
fc = results.forecast(steps=5, alpha=0.05)
print(fc["forecast"])
```

---

## Guia Detalhado

### Construtor

```python
Theta(
    theta=2,              # Parametro Theta (padrao = 2)
    deseasonalize=True,   # Remover sazonalidade antes de aplicar
    seasonal_periods=None  # Periodo sazonal (se aplicavel)
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `theta` | `float` | `2` | Valor de $\theta$ para a segunda linha |
| `deseasonalize` | `bool` | `True` | Remover sazonalidade via decomposicao STL |
| `seasonal_periods` | `int \| None` | `None` | Periodo sazonal. `None` = detectar |

### Aplicando a Series Sazonais

Quando a serie possui sazonalidade, o metodo Theta primeiro **dessazonaliza**
a serie, aplica o metodo e depois **re-sazonaliza** as previsoes:

```python
from chronobox import Theta
from chronobox.datasets import load_airline

y = load_airline()

# Theta com dessazonalizacao automatica
model = Theta(deseasonalize=True, seasonal_periods=12)
results = model.fit(y)

fc = results.forecast(steps=24, alpha=0.05)
print(fc["forecast"])  # Previsoes ja re-sazonalizadas
```

### Variando o Parametro $\theta$

=== "Theta padrao (θ = 2)"

    ```python
    model = Theta(theta=2)
    results = model.fit(y)
    print(f"Drift: {results.params['drift']:.4f}")
    ```

    Amplifica a curvatura da serie. Comportamento padrao.

=== "Theta conservador (θ = 1)"

    ```python
    model = Theta(theta=1)
    results = model.fit(y)
    print(f"Drift: {results.params['drift']:.4f}")
    ```

    Linha Theta e a propria serie. Equivale a SES pura.

=== "Theta agressivo (θ = 3)"

    ```python
    model = Theta(theta=3)
    results = model.fit(y)
    print(f"Drift: {results.params['drift']:.4f}")
    ```

    Amplifica mais a curvatura. Pode ser instavel.

!!! note "Theta otimizado"
    A versao padrao usa $\theta = 2$, que foi o valor vencedor na M3.
    Algumas implementacoes modernas permitem **otimizar** $\theta$ via
    criterio de informacao, mas a versao fixa ja e muito competitiva.

---

## Interpretacao

### Lendo o `summary()`

```python
print(results.summary())
```

```text
                       Theta Method Results
==========================================================================
Dep. Variable:                y       No. Observations:            50
Method:                   Theta       Log Likelihood:         -183.95
Date:                2026-04-09       AIC:                     371.90
                                      BIC:                     377.64
==========================================================================
Theta Parameter:              2
Drift:                   0.8321
Alpha (SES):             0.8340
==========================================================================
         Initial States
------------------------------------------
l0               111.2100
==========================================================================
sigma2            99.3621
==========================================================================
```

**Como interpretar**:

| Campo | Significado |
|---|---|
| `Theta` | Parametro $\theta$ usado (padrao = 2) |
| `Drift` | Componente de tendencia $b$ (metade do slope linear) |
| `Alpha (SES)` | Parametro de suavizacao estimado via SES |
| `l0` | Nivel inicial |
| `sigma2` | Variancia dos erros |

### Comparando com SES e Holt

```python
from chronobox import ETS, Theta

# SES
ses = ETS(error='A', trend='N', seasonal='N')
res_ses = ses.fit(y)

# Holt
holt = ETS(error='A', trend='A', seasonal='N')
res_holt = holt.fit(y)

# Theta
theta = Theta()
res_theta = theta.fit(y)

print(f"SES   AIC: {res_ses.aic:.2f}")
print(f"Holt  AIC: {res_holt.aic:.2f}")
print(f"Theta AIC: {res_theta.aic:.2f}")
```

---

## Diagnosticos

```python
from chronobox.tests_stat import ljung_box_test, jarque_bera_test, arch_test

residuals = results.residuals

lb = ljung_box_test(residuals, lags=10)
print(f"Ljung-Box p-value: {lb.pvalue:.4f}")

jb = jarque_bera_test(residuals)
print(f"Jarque-Bera p-value: {jb.pvalue:.4f}")

arch = arch_test(residuals, lags=5)
print(f"ARCH p-value: {arch.pvalue:.4f}")
```

| Teste | $H_0$ | Resultado Desejado |
|---|---|---|
| Ljung-Box | Sem autocorrelacao | $p > 0.05$ |
| Jarque-Bera | Normalidade | $p > 0.05$ |
| ARCH-LM | Sem heterocedasticidade | $p > 0.05$ |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import Theta

    model = Theta()
    results = model.fit(y)

    print(results.summary())
    fc = results.forecast(steps=12)
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    # Metodo Theta
    fc <- thetaf(y, h = 12)

    summary(fc)
    ```

=== "fable (R)"

    ```r
    library(fable)

    fit <- y_tsibble |>
      model(THETA(y))

    report(fit)
    fc <- forecast(fit, h = 12)
    ```

**Mapeamento de parametros**:

| chronobox | forecast (R) | fable (R) |
|---|---|---|
| `Theta()` | `thetaf(y)` | `THETA(y)` |
| `deseasonalize=True` | Padrao | Padrao |
| `results.params['drift']` | `fc$model$drift` | `tidy(fit)` |
| `results.params['alpha']` | `fc$model$alpha` | `tidy(fit)` |

---

## Referencias

- Assimakopoulos, V. & Nikolopoulos, K. (2000). The theta model: a
  decomposition approach to forecasting. *International Journal of
  Forecasting*, 16(4), 521--530.
- Hyndman, R. J. & Billah, B. (2003). Unmasking the Theta method.
  *International Journal of Forecasting*, 19(2), 287--290.
- Makridakis, S. & Hibon, M. (2000). The M3-Competition: results, conclusions
  and implications. *International Journal of Forecasting*, 16(4), 451--476.
- Hyndman, R. J. & Athanasopoulos, G. (2021).
  *Forecasting: Principles and Practice*. 3rd ed. OTexts. Section 8.8.
