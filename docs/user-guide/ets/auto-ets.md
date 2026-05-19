---
title: Auto-ETS
description: Selecao automatica entre os 30 modelos ETS via criterios de informacao.
---

# Auto-ETS

!!! info "Quick Reference"
    - **Classe**: `chronobox.AutoETS`
    - **Import**: `from chronobox import AutoETS`
    - **R equivalente**: `forecast::ets(y)` / `fable::ETS(y)`
    - **Criterios**: AIC, BIC, AICc

---

## Overview

O **Auto-ETS** seleciona automaticamente o melhor modelo entre os 30 modelos
da taxonomia ETS, combinando:

1. **Enumeracao** de todas as combinacoes viáveis de Erro, Tendencia e Sazonalidade
2. **Estimacao** de cada modelo por maxima verossimilhanca
3. **Selecao** do modelo com o menor criterio de informacao (AICc por padrao)

Isso elimina a necessidade de escolha manual dos componentes, embora o usuario
possa restringir o espaco de busca conforme conhecimento previo.

### Quando usar

- Quando voce nao tem certeza de qual modelo ETS e mais adequado
- Para automacao de pipelines de previsao
- Como benchmark para comparar com modelos escolhidos manualmente
- Quando precisa ajustar muitas series de uma vez

---

## Formulacao Matematica

### Criterios de Informacao

A selecao entre modelos e baseada nos criterios de informacao:

**AIC** (Akaike Information Criterion):

$$
\text{AIC} = -2\ln\hat{L} + 2k
$$

**BIC** (Bayesian Information Criterion):

$$
\text{BIC} = -2\ln\hat{L} + k\ln n
$$

**AICc** (AIC corrigido para amostras pequenas):

$$
\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}
$$

onde:

- $\hat{L}$ e a verossimilhanca maximizada
- $k$ e o numero de parametros estimados
- $n$ e o numero de observacoes

!!! note "Log-verossimilhanca entre modelos com erros diferentes"
    Os modelos com erro **aditivo** e **multiplicativo** possuem
    log-verossimilhancas em escalas diferentes. Para comparacao justa,
    a log-verossimilhanca dos modelos multiplicativos e ajustada
    por um fator de correcao:
    
    $$
    \ln L^* = \ln L - n\ln\left(\sum_{t=1}^n |y_t| / n\right)
    $$
    
    Isso permite comparar AIC/BIC entre todos os 30 modelos diretamente.

### Algoritmo de Selecao

O algoritmo Auto-ETS segue estas etapas:

1. **Determinar componentes candidatos**:
    - Erro: {A, M} (M apenas se $y_t > 0$ para todo $t$)
    - Tendencia: {N, A, A$_d$} (M, M$_d$ se `allow_multiplicative_trend=True`)
    - Sazonalidade: {N} se nao sazonal; {A, M} se sazonal (M apenas se $y_t > 0$)

2. **Restringir combinacoes instáveis** (se `restrict=True`):
    - Excluir modelos com tendencia multiplicativa e sazonalidade aditiva
    - Excluir combinacoes conhecidamente problematicas

3. **Estimar cada modelo** por MLE

4. **Selecionar** o modelo com menor criterio de informacao

---

## Quick Example

```python
from chronobox import AutoETS
from chronobox.datasets import load_airline

y = load_airline()

# Selecao automatica entre todos os modelos ETS
auto = AutoETS(seasonal_periods=12, ic='aicc')
results = auto.fit(y)

# Ver modelo selecionado
print(f"Modelo selecionado: {results.model_name}")
print(results.summary())

# Previsao
fc = results.forecast(steps=24, alpha=0.05)
print(fc["forecast"])
```

---

## Guia Detalhado

### Construtor

```python
AutoETS(
    seasonal_periods=None,              # Periodo sazonal (m)
    error=None,                         # Restringir tipo de erro
    trend=None,                         # Restringir tipo de tendencia
    seasonal=None,                      # Restringir tipo de sazonalidade
    damped=None,                        # Restringir damping
    ic='aicc',                          # Criterio de informacao
    restrict=True,                      # Restringir modelos instáveis
    allow_multiplicative_trend=False,   # Permitir tendencia multiplicativa
    maxiter=500                         # Iteracoes maximas por modelo
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `seasonal_periods` | `int \| None` | `None` | Periodo sazonal. `None` = detectar automaticamente |
| `error` | `str \| None` | `None` | Fixar erro: `'A'` ou `'M'`. `None` = testar ambos |
| `trend` | `str \| None` | `None` | Fixar tendencia: `'N'`, `'A'`. `None` = testar todas |
| `seasonal` | `str \| None` | `None` | Fixar sazonalidade: `'N'`, `'A'`, `'M'`. `None` = testar todas |
| `damped` | `bool \| None` | `None` | Fixar damping. `None` = testar ambos |
| `ic` | `str` | `'aicc'` | Criterio: `'aic'`, `'bic'`, `'aicc'` |
| `restrict` | `bool` | `True` | Excluir modelos potencialmente instáveis |
| `allow_multiplicative_trend` | `bool` | `False` | Incluir tendencia multiplicativa |
| `maxiter` | `int` | `500` | Iteracoes maximas do otimizador |

### Restringindo o Espaco de Busca

=== "Selecao totalmente automatica"

    ```python
    auto = AutoETS(seasonal_periods=12)
    results = auto.fit(y)
    print(f"Modelo: {results.model_name}")
    ```

    Testa todas as combinacoes viáveis (tipicamente 14--18 modelos).

=== "Fixar tipo de erro"

    ```python
    auto = AutoETS(seasonal_periods=12, error='A')
    results = auto.fit(y)
    print(f"Modelo: {results.model_name}")
    ```

    Restringe a modelos com erro aditivo (15 modelos).

=== "Fixar sazonalidade multiplicativa"

    ```python
    auto = AutoETS(seasonal_periods=12, seasonal='M')
    results = auto.fit(y)
    print(f"Modelo: {results.model_name}")
    ```

    Util quando voce sabe que a sazonalidade e multiplicativa.

=== "Forcar damped trend"

    ```python
    auto = AutoETS(seasonal_periods=12, damped=True)
    results = auto.fit(y)
    print(f"Modelo: {results.model_name}")
    ```

    Todos os modelos com tendencia usarao damping.

### Comparando Criterios

```python
results_aic = AutoETS(seasonal_periods=12, ic='aic').fit(y)
results_bic = AutoETS(seasonal_periods=12, ic='bic').fit(y)
results_aicc = AutoETS(seasonal_periods=12, ic='aicc').fit(y)

print(f"AIC  → {results_aic.model_name}")
print(f"BIC  → {results_bic.model_name}")
print(f"AICc → {results_aicc.model_name}")
```

!!! tip "Qual criterio usar?"
    - **AICc** (padrao): melhor para amostras pequenas a moderadas.
      Corrige o vies do AIC e converge para o AIC com $n$ grande.
    - **BIC**: penaliza mais parametros, seleciona modelos mais simples.
      Preferido se o objetivo e interpretacao.
    - **AIC**: classico, mas pode selecionar modelos complexos demais
      em amostras pequenas.

### Acessando Resultados da Selecao

```python
# Modelo selecionado
print(f"Modelo: {results.model_name}")
print(f"AICc:   {results.aicc:.2f}")

# Parametros estimados
print(f"Alpha:  {results.params['smoothing_level']:.4f}")

# Ranking de todos os modelos testados
ranking = results.model_comparison
for row in ranking.head(5):
    print(f"  {row['model']:15s}  AICc={row['aicc']:.2f}")
```

### Permitindo Tendencia Multiplicativa

```python
auto = AutoETS(
    seasonal_periods=12,
    allow_multiplicative_trend=True,
    restrict=False
)
results = auto.fit(y)
```

!!! warning "Cuidado com modelos irrestritos"
    Desabilitar `restrict` e habilitar `allow_multiplicative_trend`
    permite testar todos os 30 modelos, incluindo combinacoes
    potencialmente instáveis. As previsoes podem divergir se um modelo
    instavel for selecionado. Use apenas com dados que voce conhece bem.

---

## Interpretacao

### Lendo o `summary()`

```python
print(results.summary())
```

```text
                    AutoETS Results
==========================================================================
Selected Model:        ETS(M,Ad,M)
Dep. Variable:                y       No. Observations:           144
Method:                     MLE       Log Likelihood:         -504.31
Date:                2026-04-09       AIC:                    1040.62
                                      BIC:                    1088.41
                                      AICc:                   1046.84
==========================================================================
Models Evaluated:             14
Selection Criterion:        AICc
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
sigma2           0.0012
==========================================================================

         Model Comparison (top 5)
------------------------------------------
ETS(M,Ad,M)           AICc=1046.84  ****
ETS(A,Ad,M)           AICc=1047.12
ETS(M,A,M)            AICc=1049.31
ETS(A,A,M)            AICc=1050.55
ETS(A,Ad,A)           AICc=1063.21
==========================================================================
```

### Interpretando a Selecao

| Campo | Significado |
|---|---|
| `Selected Model` | Modelo com menor criterio de informacao |
| `Models Evaluated` | Numero de modelos testados |
| `Selection Criterion` | Criterio usado (AIC, BIC ou AICc) |
| `Model Comparison` | Ranking dos melhores modelos |

!!! tip "Modelos proximos"
    Se dois modelos tem AICc muito proximo (diferenca < 2), eles sao
    essencialmente equivalentes em termos de qualidade de ajuste.
    Nesse caso, prefira o modelo mais simples (menos parametros) ou
    aquele com interpretacao mais natural para o problema.

---

## Diagnosticos

Os diagnosticos sao os mesmos de qualquer modelo ETS:

```python
from chronobox.tests_stat import ljung_box_test, jarque_bera_test, arch_test

residuals = results.residuals

lb = ljung_box_test(residuals, lags=2 * 12)
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

!!! warning "Auto-ETS nao garante residuos limpos"
    O Auto-ETS seleciona o melhor modelo da familia ETS, mas isso
    nao garante que os residuos passarao em todos os testes. Se os
    diagnosticos falharem, considere modelos fora da familia ETS
    (ARIMA, TBATS, etc.).

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import AutoETS

    auto = AutoETS(seasonal_periods=12, ic='aicc')
    results = auto.fit(y)

    print(results.model_name)
    print(results.summary())
    fc = results.forecast(steps=24)
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    # Selecao automatica (padrao do ets())
    fit <- ets(y)

    # Equivalente explicito
    fit <- ets(y, model = "ZZZ", ic = "aicc",
               restrict = TRUE,
               allow.multiplicative.trend = FALSE)

    summary(fit)
    fc <- forecast(fit, h = 24)
    ```

=== "fable (R)"

    ```r
    library(fable)

    # Selecao automatica
    fit <- y_tsibble |>
      model(ETS(y))

    report(fit)
    fc <- forecast(fit, h = 24)
    ```

**Mapeamento de parametros**:

| chronobox | forecast (R) | fable (R) |
|---|---|---|
| `AutoETS()` | `ets(y)` | `ETS(y)` |
| `ic='aicc'` | `ic="aicc"` | Padrao |
| `restrict=True` | `restrict=TRUE` | Padrao |
| `allow_multiplicative_trend=False` | `allow.multiplicative.trend=FALSE` | Padrao |
| `error='A'` | `model="AZZ"` | `error("A")` |
| `seasonal='M'` | `model="ZZM"` | `season("M")` |
| `results.model_name` | `fit$method` | `report(fit)` |
| `results.model_comparison` | --- | `glance(fit)` |

!!! note "Diferenca na notacao R"
    No pacote `forecast`, a letra `"Z"` no string do modelo indica
    selecao automatica daquele componente. Ex: `"ZZZ"` = selecao
    total; `"AZM"` = erro aditivo, tendencia automatica, sazonalidade
    multiplicativa.

---

## Referencias

- Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002).
  A state space framework for automatic forecasting using exponential
  smoothing methods. *International Journal of Forecasting*, 18(3), 439--454.
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008).
  *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
  Chapter 10.
- Burnham, K. P. & Anderson, D. R. (2002).
  *Model Selection and Multimodel Inference*. 2nd ed. Springer.
- Hyndman, R. J. & Athanasopoulos, G. (2021).
  *Forecasting: Principles and Practice*. 3rd ed. OTexts. Chapter 8.6.
