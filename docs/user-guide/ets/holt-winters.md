---
title: Holt-Winters Seasonal
description: Metodo de Holt-Winters para series com tendencia e sazonalidade --- aditivo e multiplicativo.
---

# Holt-Winters Seasonal

!!! info "Quick Reference"
    - **Classe**: `chronobox.ETS`
    - **Import**: `from chronobox import ETS`
    - **Modelos**: ETS(A,A,A), ETS(A,A$_d$,A), ETS(A,A,M), ETS(A,A$_d$,M), ETS(M,A,M), ETS(M,A$_d$,M)
    - **R equivalente**: `forecast::ets(y, model="AAA")` / `forecast::hw(y)`
    - **Parametros**: $\alpha, \beta^*, \gamma, \phi$

---

## Overview

O metodo de **Holt-Winters** (1960) estende o metodo de Holt adicionando um
componente de **sazonalidade**. Existem duas variantes fundamentais:

- **Aditivo**: sazonalidade constante em magnitude (ex: +50 unidades no verao)
- **Multiplicativo**: sazonalidade proporcional ao nivel (ex: +20% no verao)

Combinando com a opcao de damped trend, os principais modelos Holt-Winters sao:

| Modelo | Notacao ETS | Sazonalidade | Damping |
|---|---|---|---|
| Holt-Winters Aditivo | ETS(A,A,A) | Aditiva | Nao |
| Holt-Winters Aditivo Damped | ETS(A,A$_d$,A) | Aditiva | Sim |
| Holt-Winters Multiplicativo | ETS(A,A,M) | Multiplicativa | Nao |
| Holt-Winters Multiplicativo Damped | ETS(A,A$_d$,M) | Multiplicativa | Sim |
| HW Multiplicativo (erro M) | ETS(M,A,M) | Multiplicativa | Nao |
| HW Multiplicativo Damped (erro M) | ETS(M,A$_d$,M) | Multiplicativa | Sim |

### Quando usar

- Serie com **tendencia** e **sazonalidade** clara
- **Aditivo**: amplitude sazonal constante ao longo do tempo
- **Multiplicativo**: amplitude sazonal cresce/decresce com o nivel

!!! tip "Regra pratica"
    Se a amplitude sazonal nos graficos parece **constante** ao longo do
    tempo, use sazonalidade aditiva. Se a amplitude **cresce** com o nivel
    da serie, use multiplicativa. Na duvida, compare via AIC ou use Auto-ETS.

---

## Formulacao Matematica

### Holt-Winters Aditivo

**Equacoes de recorrencia**:

$$
l_t = \alpha(y_t - s_{t-m}) + (1 - \alpha)(l_{t-1} + b_{t-1})
$$

$$
b_t = \beta^*(l_t - l_{t-1}) + (1 - \beta^*) b_{t-1}
$$

$$
s_t = \gamma(y_t - l_{t-1} - b_{t-1}) + (1 - \gamma) s_{t-m}
$$

**Previsao** $h$ passos a frente:

$$
\hat{y}_{T+h|T} = l_T + h\, b_T + s_{T+h-m(k+1)}
$$

onde $k = \lfloor (h-1)/m \rfloor$ e $m$ e o periodo sazonal.

### Holt-Winters Multiplicativo

**Equacoes de recorrencia**:

$$
l_t = \alpha\frac{y_t}{s_{t-m}} + (1 - \alpha)(l_{t-1} + b_{t-1})
$$

$$
b_t = \beta^*(l_t - l_{t-1}) + (1 - \beta^*) b_{t-1}
$$

$$
s_t = \gamma\frac{y_t}{l_{t-1} + b_{t-1}} + (1 - \gamma) s_{t-m}
$$

**Previsao** $h$ passos a frente:

$$
\hat{y}_{T+h|T} = (l_T + h\, b_T)\, s_{T+h-m(k+1)}
$$

### Versoes Damped

Para ambas as variantes, a versao damped substitui $b_{t-1}$ por $\phi b_{t-1}$
e a previsao usa $(\phi + \phi^2 + \cdots + \phi^h)b_T$ em vez de $h\, b_T$:

$$
\hat{y}_{T+h|T} = \left(l_T + \phi\frac{1 - \phi^h}{1 - \phi}\, b_T\right) \cdot s_{T+h-m(k+1)}
\quad \text{(multiplicativo damped)}
$$

### State-Space: ETS(A,A,A)

**Equacao de observacao**:

$$
y_t = l_{t-1} + b_{t-1} + s_{t-m} + \varepsilon_t
$$

**Equacoes de transicao**:

$$
l_t = l_{t-1} + b_{t-1} + \alpha\,\varepsilon_t
$$

$$
b_t = b_{t-1} + \beta\,\varepsilon_t
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t
$$

onde $\beta = \alpha\beta^*$ e $\varepsilon_t \sim \text{NID}(0, \sigma^2)$.

### State-Space: ETS(A,A$_d$,M)

**Equacao de observacao**:

$$
y_t = (l_{t-1} + \phi\, b_{t-1})\, s_{t-m} + \varepsilon_t
$$

**Equacoes de transicao**:

$$
l_t = l_{t-1} + \phi\, b_{t-1} + \alpha\,\varepsilon_t / s_{t-m}
$$

$$
b_t = \phi\, b_{t-1} + \beta\,\varepsilon_t / s_{t-m}
$$

$$
s_t = s_{t-m} + \gamma\,\varepsilon_t / (l_{t-1} + \phi\, b_{t-1})
$$

### State-Space: ETS(M,A,M)

**Equacao de observacao**:

$$
y_t = (l_{t-1} + b_{t-1})\, s_{t-m}\,(1 + \varepsilon_t)
$$

**Equacoes de transicao**:

$$
l_t = (l_{t-1} + b_{t-1})(1 + \alpha\,\varepsilon_t)
$$

$$
b_t = b_{t-1} + \beta(l_{t-1} + b_{t-1})\varepsilon_t
$$

$$
s_t = s_{t-m}(1 + \gamma\,\varepsilon_t)
$$

### Parametros

| Parametro | Dominio | Descricao |
|---|---|---|
| $\alpha$ | $(0, 1)$ | Taxa de suavizacao do nivel |
| $\beta^*$ | $(0, 1)$ | Taxa de suavizacao da tendencia |
| $\gamma$ | $(0, 1-\alpha)$ | Taxa de suavizacao sazonal |
| $\phi$ | $(0, 1)$ | Amortecimento (apenas versoes damped) |
| $l_0$ | $\mathbb{R}$ | Nivel inicial |
| $b_0$ | $\mathbb{R}$ | Tendencia inicial |
| $s_0, \ldots, s_{-m+1}$ | $\mathbb{R}$ | Estados sazonais iniciais |

!!! note "Restricao sobre $\gamma$"
    O parametro $\gamma$ e restrito a $(0, 1-\alpha)$ para garantir
    estabilidade. Isso implica que a suavizacao sazonal nunca pode
    ser mais agressiva que a suavizacao do nivel.

---

## Quick Example

```python
from chronobox import ETS
from chronobox.datasets import load_airline

# Carregar dados de passageiros (serie classica com tendencia + sazonalidade)
y = load_airline()

# Holt-Winters Multiplicativo Damped --- ETS(A,Ad,M)
model = ETS(error='A', trend='A', seasonal='M', seasonal_periods=12, damped=True)
results = model.fit(y)

print(results.summary())

# Previsao 24 meses a frente
fc = results.forecast(steps=24, alpha=0.05)
print(fc["forecast"])
```

---

## Guia Detalhado

### Construtor

```python
ETS(
    error='A',              # 'A' ou 'M'
    trend='A',              # 'A' para tendencia aditiva
    seasonal='M',           # 'A' (aditiva) ou 'M' (multiplicativa)
    seasonal_periods=12,    # Periodo sazonal (m)
    damped=True             # Tendencia amortecida
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `error` | `str` | `'A'` | Tipo de erro |
| `trend` | `str` | `'A'` | Tipo de tendencia |
| `seasonal` | `str` | `'N'` | `'A'` (aditiva) ou `'M'` (multiplicativa) |
| `seasonal_periods` | `int` | `None` | Periodo sazonal ($m$). Ex: 12 mensal, 4 trimestral |
| `damped` | `bool` | `False` | Tendencia amortecida |

### Periodos Sazonais Comuns

| Frequencia | Periodo ($m$) | Exemplo |
|---|---|---|
| Mensal | 12 | Dados mensais com ciclo anual |
| Trimestral | 4 | PIB trimestral |
| Semanal | 52 | Dados semanais com ciclo anual |
| Diario | 7 | Dados diarios com ciclo semanal |

### Aditivo vs Multiplicativo

=== "Sazonalidade Aditiva --- ETS(A,A,A)"

    ```python
    model = ETS(error='A', trend='A', seasonal='A',
                seasonal_periods=12, damped=False)
    results = model.fit(y)
    print(f"AIC: {results.aic:.2f}")
    ```

    A sazonalidade e **somada** ao nivel: $\hat{y} = l + hb + s$.
    Amplitude sazonal e constante em unidades absolutas.

=== "Sazonalidade Multiplicativa --- ETS(A,A,M)"

    ```python
    model = ETS(error='A', trend='A', seasonal='M',
                seasonal_periods=12, damped=False)
    results = model.fit(y)
    print(f"AIC: {results.aic:.2f}")
    ```

    A sazonalidade **multiplica** o nivel: $\hat{y} = (l + hb) \times s$.
    Amplitude sazonal cresce proporcionalmente ao nivel.

=== "Comparacao via AIC"

    ```python
    models = {
        'ETS(A,A,A)': ETS(error='A', trend='A', seasonal='A',
                          seasonal_periods=12),
        'ETS(A,Ad,A)': ETS(error='A', trend='A', seasonal='A',
                           seasonal_periods=12, damped=True),
        'ETS(A,A,M)': ETS(error='A', trend='A', seasonal='M',
                          seasonal_periods=12),
        'ETS(A,Ad,M)': ETS(error='A', trend='A', seasonal='M',
                           seasonal_periods=12, damped=True),
    }

    for name, m in models.items():
        res = m.fit(y)
        print(f"{name}: AIC={res.aic:.2f}")
    ```

### Inicializacao dos Estados Sazonais

Os estados sazonais iniciais $s_0, s_{-1}, \ldots, s_{-m+1}$ sao estimados
por padrao usando decomposicao classica:

1. Calcula-se a media movel centrada (trend-cycle)
2. Os fatores sazonais sao obtidos dividindo (multiplicativo) ou subtraindo (aditivo) a serie pela media movel
3. Os fatores sao normalizados para que somem zero (aditivo) ou $m$ (multiplicativo)

```python
# Estados sazonais estimados
seasonal_states = results.params['initial_seasonal']
print(f"Estados sazonais: {seasonal_states}")
print(f"Soma: {sum(seasonal_states):.4f}")  # ~0 (aditivo) ou ~m (multiplicativo)
```

!!! note "Normalizacao sazonal"
    Para sazonalidade **aditiva**, os $m$ estados sazonais somam zero:
    $\sum_{i=0}^{m-1} s_i = 0$. Para **multiplicativa**, somam $m$:
    $\sum_{i=0}^{m-1} s_i = m$.

---

## Interpretacao

### Lendo o `summary()`

```python
print(results.summary())
```

```text
                     ETS(A,Ad,M) Results
==========================================================================
Dep. Variable:                y       No. Observations:           144
Method:                     MLE       Log Likelihood:         -504.31
Date:                2026-04-09       AIC:                    1040.62
                                      BIC:                    1088.41
                                      AICc:                   1046.84
==========================================================================
         Smoothing Parameters
------------------------------------------
alpha              0.5712    (0.06, 0.99)
beta*              0.0001    (0.001, 0.30)
gamma              0.0001    (0.001, 0.99)
phi                0.9800    (0.80, 0.98)
==========================================================================
         Initial States
------------------------------------------
l0               117.4200
b0                 0.8321
s0      [0.91, 0.88, 0.96, 1.00, 1.04, 1.17,
         1.21, 1.19, 1.07, 0.96, 0.87, 0.84]
==========================================================================
sigma2           132.4500
==========================================================================
```

**Como interpretar os estados sazonais (multiplicativo)**:

| Mes | Fator Sazonal | Interpretacao |
|---|---|---|
| Janeiro | 0.91 | 9% abaixo da media |
| Julho | 1.21 | 21% acima da media |
| Dezembro | 0.84 | 16% abaixo da media |

### Decomposicao dos Componentes

```python
# Acessar componentes estimados
level = results.states['level']
trend = results.states['trend']
seasonal = results.states['seasonal']

# Componente ajustado (fitted values)
fitted = results.fittedvalues
```

---

## Diagnosticos

```python
from chronobox.tests_stat import ljung_box_test, jarque_bera_test, arch_test

residuals = results.residuals

# Testar nos lags sazonais tambem
lb = ljung_box_test(residuals, lags=2 * 12)
print(f"Ljung-Box (lag 24) p-value: {lb.pvalue:.4f}")

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

!!! warning "Lags sazonais nos residuos"
    Para dados sazonais, verifique a ACF dos residuos nos lags multiplos
    de $m$ (12, 24, 36...). Se houver autocorrelacao sazonal residual,
    considere trocar entre sazonalidade aditiva e multiplicativa, ou
    ajustar a ordem do modelo.

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import ETS

    model = ETS(error='A', trend='A', seasonal='M',
                seasonal_periods=12, damped=True)
    results = model.fit(y)

    print(results.summary())
    fc = results.forecast(steps=24)
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    # Holt-Winters Multiplicativo Damped
    fit <- ets(y, model = "AAM", damped = TRUE)

    # Ou usando hw() diretamente
    fit <- hw(y, seasonal = "multiplicative",
              damped = TRUE, h = 24)

    summary(fit)
    ```

=== "fable (R)"

    ```r
    library(fable)

    fit <- y_tsibble |>
      model(ETS(y ~ error("A") + trend("Ad") + season("M")))

    report(fit)
    fc <- forecast(fit, h = 24)
    ```

**Mapeamento de parametros**:

| chronobox | forecast (R) | fable (R) |
|---|---|---|
| `seasonal='A'` | `model="AAA"` | `season("A")` |
| `seasonal='M'` | `model="AAM"` | `season("M")` |
| `seasonal_periods=12` | Detecta automaticamente | Detecta automaticamente |
| `results.params['smoothing_seasonal']` | `fit$par["gamma"]` | `tidy(fit)` |

---

## Referencias

- Winters, P. R. (1960). Forecasting sales by exponentially weighted moving
  averages. *Management Science*, 6(3), 324--342.
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008).
  *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
  Chapters 4--5.
- Hyndman, R. J. & Athanasopoulos, G. (2021).
  *Forecasting: Principles and Practice*. 3rd ed. OTexts. Chapter 8.3--8.5.
- Chatfield, C. (1978). The Holt-Winters forecasting procedure.
  *Applied Statistics*, 27(3), 264--279.
