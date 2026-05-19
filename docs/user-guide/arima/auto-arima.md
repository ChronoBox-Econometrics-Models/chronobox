---
title: Auto-ARIMA
description: Selecao automatica da ordem ARIMA via criterios de informacao e busca stepwise.
---

# Auto-ARIMA

!!! info "Quick Reference"
    - **Funcao**: `chronobox.auto_arima`
    - **Import**: `from chronobox import auto_arima`
    - **R equivalente**: `forecast::auto.arima(y)`
    - **Algoritmo**: Hyndman-Khandakar (2008), stepwise ou grid search

---

## Overview

A funcao `auto_arima` automatiza a selecao do melhor modelo ARIMA(p,d,q)(P,D,Q)[s]
para uma serie temporal. Ela determina automaticamente:

1. A ordem de diferenciacao $d$ (via teste ADF)
2. A ordem de diferenciacao sazonal $D$ (via comparacao de variancia)
3. As ordens AR e MA ($p, q, P, Q$) otimas (via criterio de informacao)

### Quando usar

- Voce nao sabe quais ordens (p,d,q) escolher
- Precisa de um baseline rapido para comparacao
- Esta analisando muitas series e nao pode selecionar manualmente cada modelo
- Prototipagem rapida antes de refinamento manual

!!! warning "Auto-ARIMA nao substitui analise"
    A selecao automatica e uma ferramenta, nao um substituto para diagnosticos.
    Sempre verifique os residuos do modelo selecionado e avalie se o resultado
    faz sentido para o problema.

---

## Formulacao Matematica

### Criterios de Informacao

O Auto-ARIMA minimiza um dos seguintes criterios:

**AIC** (Akaike Information Criterion):

$$
\text{AIC} = -2\ln\hat{L} + 2k
$$

**BIC** (Bayesian Information Criterion):

$$
\text{BIC} = -2\ln\hat{L} + k\ln n
$$

**AICc** (AIC corrigido, default):

$$
\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}
$$

onde $\hat{L}$ e a verossimilhanca maximizada, $k$ o numero de parametros e
$n$ o numero de observacoes.

!!! tip "Qual criterio usar?"
    - **AICc** (default): melhor para amostras pequenas e moderadas. Converge
      para AIC quando $n \to \infty$.
    - **BIC**: penaliza mais parametros; tende a selecionar modelos mais
      parcimoniosos. Preferido quando o objetivo e identificar o modelo "verdadeiro".
    - **AIC**: sem correcao de amostra finita. Use quando $n$ e muito grande.

### Algoritmo Stepwise (Hyndman-Khandakar)

O algoritmo stepwise segue estes passos:

1. **Determinar $d$**: aplica o teste ADF iterativamente ate a serie diferenciada
   ser estacionaria ($p < 0.05$), respeitando `max_d`.

2. **Determinar $D$** (se sazonal): compara a variancia da serie original com a
   serie sazonalmente diferenciada, respeitando `max_D`.

3. **Modelos iniciais**: avalia um conjunto de modelos de referencia:
    - ARIMA(0,d,0), ARIMA(1,d,0), ARIMA(0,d,1), ARIMA(2,d,2)
    - Versoes sazonais se `seasonal=True`

4. **Busca local**: a partir do melhor modelo inicial, explora vizinhos ($\pm 1$
   em $p$, $q$, $P$, $Q$) e aceita se o IC melhora.

5. **Parada**: quando nenhum vizinho melhora o IC.

### Grid Search (alternativa)

Quando `stepwise=False`, o algoritmo testa **todas** as combinacoes de
$(p, q) \in [0, \text{max\_p}] \times [0, \text{max\_q}]$ e (opcionalmente)
$(P, Q) \in [0, \text{max\_P}] \times [0, \text{max\_Q}]$.

!!! note "Performance"
    O stepwise e $O(k)$ com $k$ pequeno (tipicamente 20--50 modelos avaliados).
    O grid search e $O(\text{max\_p} \times \text{max\_q} \times \text{max\_P}
    \times \text{max\_Q})$, que pode ser lento para ordens altas.

---

## Quick Example

```python
from chronobox import auto_arima
from chronobox.datasets import load_airline

y = load_airline()

# Selecao automatica com sazonalidade mensal
results = auto_arima(y, seasonal=True, m=12)

print(results.model_name)   # e.g., "ARIMA(1,1,1)(0,1,1)[12]"
print(results.summary())

# Previsao
fc = results.forecast(steps=24)
```

---

## Guia Detalhado

### Assinatura Completa

```python
auto_arima(
    y,                              # Serie temporal
    seasonal=True,                  # Incluir componente sazonal?
    m=1,                            # Periodo sazonal
    d=None,                         # Ordem de diferenciacao (auto se None)
    D=None,                         # Ordem de dif. sazonal (auto se None)
    max_p=5,                        # Maximo p
    max_q=5,                        # Maximo q
    max_P=2,                        # Maximo P
    max_Q=2,                        # Maximo Q
    max_d=2,                        # Maximo d
    max_D=1,                        # Maximo D
    information_criterion='aicc',   # Criterio de selecao
    stepwise=True,                  # Busca stepwise ou grid
    trace=False                     # Imprimir modelos testados?
)
```

### Tabela de Parametros

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `y` | `array-like` | --- | Serie temporal observada |
| `seasonal` | `bool` | `True` | Incluir termos sazonais (P, D, Q) |
| `m` | `int` | `1` | Periodo sazonal. `1` = sem sazonalidade |
| `d` | `int \| None` | `None` | Diferenciacao. `None` = determinado via ADF |
| `D` | `int \| None` | `None` | Dif. sazonal. `None` = determinado automaticamente |
| `max_p` | `int` | `5` | Ordem AR maxima |
| `max_q` | `int` | `5` | Ordem MA maxima |
| `max_P` | `int` | `2` | Ordem AR sazonal maxima |
| `max_Q` | `int` | `2` | Ordem MA sazonal maxima |
| `max_d` | `int` | `2` | Diferenciacao maxima |
| `max_D` | `int` | `1` | Dif. sazonal maxima |
| `information_criterion` | `str` | `'aicc'` | `'aic'`, `'aicc'`, ou `'bic'` |
| `stepwise` | `bool` | `True` | Busca stepwise (rapida) ou grid (exaustiva) |
| `trace` | `bool` | `False` | Imprimir cada modelo avaliado |

### Retorno

A funcao retorna um objeto `TimeSeriesResults` --- o mesmo tipo retornado por
`ARIMA.fit()`. Voce pode acessar todos os atributos e metodos usuais.

### Exemplos de Uso

#### Dados sem sazonalidade

```python
from chronobox import auto_arima

# PIB trimestral (sem sazonalidade apos ajuste)
results = auto_arima(y, seasonal=False)
print(results.model_name)  # e.g., "ARIMA(1,1,0)"
```

#### Dados mensais com sazonalidade

```python
from chronobox import auto_arima
from chronobox.datasets import load_airline

y = load_airline()
results = auto_arima(y, seasonal=True, m=12)
print(results.model_name)  # e.g., "ARIMA(0,1,1)(0,1,1)[12]"
```

#### Grid search com BIC

```python
results = auto_arima(
    y,
    seasonal=True,
    m=12,
    stepwise=False,           # Grid search exaustivo
    information_criterion='bic',
    max_p=3,
    max_q=3,
    trace=True                # Ver todos os modelos testados
)
```

#### Fixando a ordem de diferenciacao

```python
# Voce ja sabe que d=1 e D=1
results = auto_arima(y, seasonal=True, m=12, d=1, D=1)
```

### Modo `trace`

Com `trace=True`, cada modelo testado e impresso:

```text
 ARIMA(0,1,0)(0,1,0)[12]  : AICc=1010.23
 ARIMA(1,1,0)(1,1,0)[12]  : AICc= 998.41
 ARIMA(0,1,1)(0,1,1)[12]  : AICc= 990.87
 ARIMA(1,1,1)(0,1,1)[12]  : AICc= 991.52
 ARIMA(0,1,1)(1,1,1)[12]  : AICc= 992.10

 Best model: ARIMA(0,1,1)(0,1,1)[12]  : AICc=990.87
```

---

## Interpretacao

### Resultado da Selecao

```python
results = auto_arima(y, seasonal=True, m=12, trace=True)

# Modelo selecionado
print(results.model_name)

# Criterio de informacao do modelo selecionado
print(f"AICc: {results.aicc:.2f}")
print(f"AIC:  {results.aic:.2f}")
print(f"BIC:  {results.bic:.2f}")

# Parametros estimados
print(results.summary())
```

### Validando a Selecao

O Auto-ARIMA seleciona o melhor modelo *dentro do espaco de busca*. Sempre valide:

1. **Os residuos sao ruido branco?**

    ```python
    from chronobox.tests_stat import ljung_box_test

    lb = ljung_box_test(results.residuals, lags=24)
    print(f"Ljung-Box p-value: {lb.pvalue:.4f}")
    ```

2. **O modelo faz sentido economico?** Um ARIMA(5,2,4) pode ter o menor AICc
   mas provavelmente esta sobreajustado.

3. **Compare com alternativas manuais** se o resultado parecer estranho.

!!! tip "Boa pratica"
    Use Auto-ARIMA como ponto de partida. Se o modelo selecionado tiver ordens
    altas ou residuos problematicos, ajuste manualmente a partir da sugestao.

---

## Diagnosticos

### Fluxo Completo

```python
from chronobox import auto_arima
from chronobox.datasets import load_airline
from chronobox.tests_stat import ljung_box_test, jarque_bera_test

y = load_airline()

# 1. Selecao automatica
results = auto_arima(y, seasonal=True, m=12)
print(results.summary())

# 2. Diagnostico dos residuos
lb = ljung_box_test(results.residuals, lags=24)
print(f"Ljung-Box p-value: {lb.pvalue:.4f}")

jb = jarque_bera_test(results.residuals)
print(f"Jarque-Bera p-value: {jb.pvalue:.4f}")

# 3. Previsao
fc = results.forecast(steps=24, alpha=0.05)
```

### Comparando Criterios

```python
# Comparar modelos selecionados por diferentes criterios
for ic in ['aic', 'aicc', 'bic']:
    res = auto_arima(y, seasonal=True, m=12, information_criterion=ic)
    print(f"{ic.upper():5s} -> {res.model_name:30s} (valor: {getattr(res, ic):.2f})")
```

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import auto_arima

    results = auto_arima(
        y,
        seasonal=True,
        m=12,
        stepwise=True,
        information_criterion='aicc',
        trace=True
    )
    print(results.summary())
    fc = results.forecast(steps=24)
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    fit <- auto.arima(y,
                      seasonal = TRUE,
                      stepwise = TRUE,
                      ic = "aicc",
                      trace = TRUE)
    summary(fit)
    fc <- forecast(fit, h = 24)
    ```

**Mapeamento de parametros**:

| chronobox | forecast (R) | Descricao |
|---|---|---|
| `seasonal=True` | `seasonal=TRUE` | Incluir sazonalidade |
| `m=12` | `ts(y, frequency=12)` | Periodo sazonal |
| `stepwise=True` | `stepwise=TRUE` | Algoritmo stepwise |
| `information_criterion='aicc'` | `ic="aicc"` | Criterio de selecao |
| `max_p=5` | `max.p=5` | Ordem AR maxima |
| `max_q=5` | `max.q=5` | Ordem MA maxima |
| `max_P=2` | `max.P=2` | AR sazonal maximo |
| `max_Q=2` | `max.Q=2` | MA sazonal maximo |
| `max_d=2` | `max.d=2` | Diferenciacao maxima |
| `max_D=1` | `max.D=1` | Dif. sazonal maxima |
| `d=None` | `d=NA` | Auto-deteccao |
| `trace=True` | `trace=TRUE` | Verbose |

!!! note "Algoritmo identico"
    O chronobox implementa o algoritmo Hyndman-Khandakar (2008), o mesmo
    usado pelo `forecast::auto.arima()` do R. Os resultados devem ser
    muito proximos para a mesma serie e configuracao.

---

## Referencias

- Hyndman, R. J. & Khandakar, Y. (2008). Automatic Time Series Forecasting:
  The forecast Package for R. *Journal of Statistical Software*, 27(3), 1--22.
- Hyndman, R. J. & Athanasopoulos, G. (2021).
  *Forecasting: Principles and Practice*. 3rd ed. OTexts.
- Akaike, H. (1974). A New Look at the Statistical Model Identification.
  *IEEE Transactions on Automatic Control*, 19(6), 716--723.
- Schwarz, G. (1978). Estimating the Dimension of a Model.
  *Annals of Statistics*, 6(2), 461--464.
- Hurvich, C. M. & Tsai, C.-L. (1989). Regression and Time Series Model
  Selection in Small Samples. *Biometrika*, 76(2), 297--307.
