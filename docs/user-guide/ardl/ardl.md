---
title: ARDL
description: Modelo Autoregressive Distributed Lag --- estimacao, selecao de lags, bounds test de Pesaran et al. (2001) e analise de cointegracao.
---

# ARDL

!!! info "Quick Reference"
    - **Classe**: `chronobox.ARDL`
    - **Import**: `from chronobox import ARDL`
    - **R equivalente**: `ARDL::ardl()`, `dynamac::dynardl()`
    - **Estimacao**: OLS (equacao unica)

---

## Overview

O modelo ARDL (Autoregressive Distributed Lag) generaliza o modelo autoregressivo
ao incluir **defasagens da variavel dependente** e **defasagens de variaveis
explicativas**. E particularmente util para:

- Modelar relacoes dinamicas entre variaveis com **ordens de integracao mistas** (I(0) e I(1))
- Testar cointegracao via **bounds test** sem exigir pre-teste de raiz unitaria
- Estimar relacoes de **curto e longo prazo** em uma unica equacao
- Trabalhar com **amostras pequenas**, onde o metodo de Johansen perde poder

### Quando usar

- Variaveis com mistura de I(0) e I(1) --- o Johansen nao se aplica
- Amostra pequena (T < 80) onde testes de cointegracao multivariados perdem poder
- Interesse em relacoes de longo prazo e velocidade de ajustamento
- Modelo de equacao unica com uma variavel dependente clara

!!! tip "ARDL vs Johansen"
    Use Johansen quando **todas** as variaveis sao I(1) e voce quer modelar
    multiplas relacoes de cointegracao. Use ARDL quando ha mistura de ordens
    de integracao ou quando o tamanho amostral e limitado.

---

## Formulacao Matematica

### Equacao Geral

Um modelo ARDL($p, q_1, q_2, \ldots, q_k$) e definido por:

$$
y_t = c + \delta t + \sum_{i=1}^{p} \phi_i \, y_{t-i} + \sum_{j=1}^{k} \sum_{\ell=0}^{q_j} \beta_{j,\ell} \, x_{j,t-\ell} + \epsilon_t
$$

onde:

- $y_t$ e a variavel dependente
- $x_{j,t}$ sao as $k$ variaveis explicativas
- $p$ e o numero de defasagens da variavel dependente
- $q_j$ e o numero de defasagens da $j$-esima variavel explicativa
- $\phi_i$ sao os coeficientes autoregressivos
- $\beta_{j,\ell}$ sao os coeficientes das defasagens distribuidas
- $c$ e o intercepto e $\delta$ o coeficiente de tendencia (opcionals)
- $\epsilon_t \sim \text{i.i.d.}(0, \sigma^2)$

!!! note "Efeito contemporaneo"
    Note que a soma sobre $\ell$ comeca em $\ell = 0$, o que significa que o
    ARDL inclui o **efeito contemporaneo** de $x_{j,t}$ sobre $y_t$. Isso e
    diferente do VAR, onde apenas defasagens sao incluidas.

### Exemplo: ARDL(2, 1, 1)

Com duas variaveis explicativas e lags especificos:

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \beta_{1,0} x_{1,t} + \beta_{1,1} x_{1,t-1} + \beta_{2,0} x_{2,t} + \beta_{2,1} x_{2,t-1} + \epsilon_t
$$

### Selecao de Lags

As ordens $p, q_1, \ldots, q_k$ sao selecionadas minimizando criterios de informacao:

$$
\text{AIC}(p, q_1, \ldots, q_k) = \ln(\hat{\sigma}^2) + \frac{2m}{T}
$$

$$
\text{BIC}(p, q_1, \ldots, q_k) = \ln(\hat{\sigma}^2) + \frac{m \ln T}{T}
$$

onde $m = p + 1 + \sum_{j=1}^{k}(q_j + 1)$ e o numero total de parametros estimados.

!!! tip "Regra pratica"
    O BIC tende a selecionar modelos mais parcimoniosos. Para o bounds test,
    Pesaran et al. (2001) recomendam usar o AIC para selecao de lags, pois
    subajustar os lags pode invalidar o teste.

### Relacao de Longo Prazo

Se as variaveis sao cointegradas, os **multiplicadores de longo prazo** sao:

$$
\theta_j = \frac{\sum_{\ell=0}^{q_j} \beta_{j,\ell}}{1 - \sum_{i=1}^{p} \phi_i}
$$

Ou seja, no equilibrio de longo prazo, a relacao entre $y$ e $x_j$ e:

$$
y^* = \frac{c}{1 - \sum_{i=1}^{p} \phi_i} + \sum_{j=1}^{k} \theta_j \, x_j^*
$$

---

## Bounds Test (Pesaran, Shin & Smith, 2001)

O bounds test e o principal teste de cointegracao associado ao ARDL. Ele testa
se existe uma relacao de longo prazo entre as variaveis.

### Reparametrizacao Condicional

Para aplicar o bounds test, o ARDL e reparametrizado na forma de correcao de erros
condicional (Conditional ECM):

$$
\Delta y_t = c + \delta t + \underbrace{\pi_y \, y_{t-1} + \sum_{j=1}^{k} \pi_j \, x_{j,t-1}}_{\text{componente de niveis}} + \sum_{i=1}^{p-1} \gamma_i \, \Delta y_{t-i} + \sum_{j=1}^{k} \sum_{\ell=0}^{q_j-1} \delta_{j,\ell} \, \Delta x_{j,t-\ell} + \epsilon_t
$$

onde:

- $\pi_y = -(1 - \sum_{i=1}^{p} \phi_i)$ e o coeficiente de ajustamento
- $\pi_j = \sum_{\ell=0}^{q_j} \beta_{j,\ell}$ sao os coeficientes de longo prazo das explicativas

### Hipoteses do Teste

O bounds test avalia a significancia **conjunta** dos coeficientes das variaveis
em niveis:

$$
H_0: \pi_y = \pi_1 = \pi_2 = \cdots = \pi_k = 0 \quad \text{(sem relacao de longo prazo)}
$$

$$
H_1: \pi_y \neq 0 \cup \pi_1 \neq 0 \cup \cdots \cup \pi_k \neq 0 \quad \text{(existe relacao de longo prazo)}
$$

### F-statistic e as Bandas

A estatistica F resultante e comparada com **duas bandas de valores criticos**:

| Banda | Pressuposicao | Decisao se $F >$ banda |
|---|---|---|
| **Banda inferior** I(0) | Todos os regressores sao I(0) | Rejeita $H_0$ mesmo no melhor caso para $H_0$ |
| **Banda superior** I(1) | Todos os regressores sao I(1) | Rejeita $H_0$ mesmo no pior caso para $H_0$ |

**Regras de decisao**:

| Resultado | Interpretacao |
|---|---|
| $F > \text{banda superior}$ | **Rejeita** $H_0$ --- evidencia de cointegracao |
| $F < \text{banda inferior}$ | **Nao rejeita** $H_0$ --- sem evidencia de cointegracao |
| banda inferior $\leq F \leq$ banda superior | **Inconclusivo** --- necessario determinar ordens de integracao |

!!! warning "Resultado inconclusivo"
    Quando o teste e inconclusivo, voce deve determinar as ordens de integracao
    exatas dos regressores via testes ADF/KPSS e usar os valores criticos
    correspondentes (nao as bandas).

### t-statistic Complementar

Alem do F-test, Pesaran et al. (2001) propoem um t-test para o coeficiente de
ajustamento:

$$
H_0: \pi_y = 0 \quad \text{vs} \quad H_1: \pi_y < 0
$$

O t-test tambem possui bandas I(0) e I(1) e segue a mesma logica de decisao.

---

## Quick Example

```python
from chronobox import ARDL
from chronobox.datasets import load_macro

import pandas as pd

# Carregar dados
data = load_macro()

# Definir variaveis
y = data["consumption"]
X = data[["income", "wealth"]]

# Estimar ARDL com selecao automatica de lags (maximo 4)
model = ARDL(lags=4, exog_lags=4, trend='c', ic='aic')
results = model.fit(y=y, exog=X)

# Resumo do modelo
print(results.summary())
print(f"Lags selecionados: ARDL({results.ar_lags}, {results.dl_lags})")

# Bounds test
bounds = results.bounds_test()
print(bounds.summary())
```

```text
                     ARDL(2, 1, 1) Results
==========================================================================
Dep. Variable:        consumption    No. Observations:          120
Method:                       OLS    R-squared:              0.9834
                                     Adj. R-squared:         0.9825
Trend:                   Constant    AIC:                   -3.4521
                                     BIC:                   -3.2847
==========================================================================

                 coef     std err       t     P>|t|    [0.025     0.975]
--------------------------------------------------------------------------
const          0.2341     0.0876    2.672    0.009     0.061     0.408
consumption.L1 0.6123     0.0834    7.341    0.000     0.447     0.778
consumption.L2-0.1245     0.0712   -1.749    0.083    -0.265     0.017
income         0.2534     0.0645    3.929    0.000     0.126     0.381
income.L1      0.0823     0.0598    1.376    0.171    -0.036     0.201
wealth         0.0412     0.0198    2.081    0.040     0.002     0.081
wealth.L1      0.0267     0.0187    1.428    0.156    -0.010     0.064

==========================================================================
Bounds Test (Pesaran, Shin & Smith, 2001)
==========================================================================
F-statistic:        5.423       k (regressors):           2
t-statistic:       -3.812

Critical Values (5% significance):
                     I(0)        I(1)
  F-statistic:      3.79        4.85
  t-statistic:     -2.86       -3.78

Decision (5%):      Reject H0 --- evidence of cointegration
==========================================================================
```

---

## Guia Detalhado

### Construtor

```python
ARDL(
    lags=1,            # Defasagens da variavel dependente (int ou 'auto')
    exog_lags=1,       # Defasagens das explicativas (int, list ou 'auto')
    trend='c',         # Componente deterministico
    ic='aic',          # Criterio de informacao para selecao
    max_lags=12,       # Maximo de lags para selecao automatica
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `lags` | `int \| str` | `1` | Defasagens de $y_t$, ou `'auto'` para selecao automatica |
| `exog_lags` | `int \| list \| str` | `1` | Defasagens das explicativas: `int` (mesmo para todas), `list` (por variavel), ou `'auto'` |
| `trend` | `str` | `'c'` | `'n'` (nenhum), `'c'` (constante), `'t'` (tendencia), `'ct'` (ambos) |
| `ic` | `str` | `'aic'` | Criterio para selecao: `'aic'`, `'bic'`, `'hqc'` |
| `max_lags` | `int` | `12` | Maximo de lags quando usando selecao automatica |

**Opcoes de `trend`**:

| Valor | Significado | Caso no bounds test |
|---|---|---|
| `'n'` | Sem constante | Caso I (nao restrito) |
| `'c'` | Constante irrestrita | Caso III (padrao de PSS) |
| `'ct'` | Constante + tendencia irrestrita | Caso V |

!!! note "Casos do bounds test"
    Os valores criticos do bounds test dependem da especificacao do componente
    deterministico. Pesaran et al. (2001) definem 5 casos. O caso III
    (constante irrestrita, sem tendencia) e o mais utilizado na pratica.

### Metodo `fit()`

```python
results = model.fit(
    y,                 # Serie dependente
    exog=None,         # Variaveis explicativas
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `y` | `pd.Series \| np.ndarray` | --- | Variavel dependente ($T \times 1$) |
| `exog` | `pd.DataFrame \| np.ndarray \| None` | `None` | Variaveis explicativas ($T \times k$) |

### Selecao Automatica de Lags

```python
# Selecao automatica com AIC
model = ARDL(lags='auto', exog_lags='auto', max_lags=6, ic='aic')
results = model.fit(y=y, exog=X)

# Tabela de selecao
print(results.lag_selection)
```

```text
  ARDL(p, q1, q2)      AIC        BIC
  ARDL(2, 1, 1)      -3.452    -3.285   ← AIC
  ARDL(2, 1, 0)      -3.441    -3.301
  ARDL(1, 1, 1)      -3.438    -3.298
  ARDL(3, 1, 1)      -3.435    -3.241
  ...
```

!!! tip "Lags diferentes por variavel"
    Use uma lista para especificar lags diferentes para cada variavel explicativa:
    ```python
    # 3 lags para income, 1 lag para wealth
    model = ARDL(lags=2, exog_lags=[3, 1])
    ```

### Bounds Test

```python
# Executar bounds test
bounds = results.bounds_test()

# Acessar resultados
print(f"F-statistic: {bounds.f_stat:.3f}")
print(f"t-statistic: {bounds.t_stat:.3f}")
print(f"Decisao (5%): {bounds.decision}")

# Valores criticos em tabela
print(bounds.critical_values)
```

| Atributo | Tipo | Descricao |
|---|---|---|
| `bounds.f_stat` | `float` | F-statistic do teste conjunto |
| `bounds.t_stat` | `float` | t-statistic para $\pi_y$ |
| `bounds.decision` | `str` | `'reject'`, `'inconclusive'`, `'fail_to_reject'` |
| `bounds.critical_values` | `pd.DataFrame` | Bandas I(0) e I(1) para 1%, 5% e 10% |
| `bounds.k` | `int` | Numero de regressores no teste |

### Multiplicadores de Longo Prazo

```python
# Coeficientes de longo prazo
lr = results.long_run_coefficients()
print(lr)
```

```text
Long-Run Coefficients (ARDL → Equilibrium)
===========================================
             coef     std err       t     P>|t|
income      0.6548     0.0923    7.094    0.000
wealth      0.1325     0.0412    3.217    0.002
constant    0.4569     0.1534    2.979    0.004

Long-run equation: consumption* = 0.457 + 0.655 * income* + 0.133 * wealth*
```

### Conversao para ECM

```python
# Reparametrizar em forma ECM
ecm = results.to_ecm()
print(ecm.summary())

# Velocidade de ajustamento
print(f"Alpha (velocidade de ajuste): {ecm.alpha:.4f}")
```

---

## Interpretacao

### Lendo os Resultados

**Coeficientes de curto prazo**: os coeficientes do ARDL em niveis representam
efeitos de curto prazo (impacto e defasados). Nao confunda com efeitos de longo
prazo.

**Multiplicadores de longo prazo** ($\theta_j$): representam o efeito total
de uma mudanca permanente em $x_j$ sobre $y$ apos o sistema convergir para o
equilibrio.

**Bounds test**: a estatistica F testa a existencia de cointegracao. A
vantagem do bounds test e que nao exige conhecer as ordens de integracao
exatas --- as bandas cobrem todos os cenarios possiveis entre I(0) e I(1).

| Resultado | Acao |
|---|---|
| Rejeita $H_0$ | Estimar ECM e interpretar relacoes de longo prazo |
| Inconclusivo | Testar ordens de integracao e usar valores criticos exatos |
| Nao rejeita $H_0$ | Nao ha evidencia de cointegracao; usar modelo em diferencas |

!!! tip "Significancia dos coeficientes"
    Mesmo que o bounds test rejeite $H_0$, verifique se os coeficientes de
    longo prazo sao individualmente significativos. Um teste conjunto significativo
    nao garante que todas as relacoes bilaterais sejam significativas.

---

## Diagnosticos

### 1. Autocorrelacao dos Residuos

```python
from chronobox.tests_stat import breusch_godfrey_test

bg = breusch_godfrey_test(results.residuals, lags=4)
print(f"Breusch-Godfrey p-value: {bg.pvalue:.4f}")
# p > 0.05 → sem autocorrelacao serial
```

### 2. Heterocedasticidade

```python
from chronobox.tests_stat import arch_test

arch = arch_test(results.residuals, lags=4)
print(f"ARCH p-value: {arch.pvalue:.4f}")
```

### 3. Normalidade

```python
from chronobox.tests_stat import jarque_bera_test

jb = jarque_bera_test(results.residuals)
print(f"Jarque-Bera p-value: {jb.pvalue:.4f}")
```

### 4. Estabilidade (CUSUM)

```python
from chronobox.tests_stat import cusum_test

cs = cusum_test(results.residuals)
print(f"CUSUM estavel: {cs.is_stable}")
```

### Checklist de Diagnostico

| Teste | $H_0$ | Resultado Desejado |
|---|---|---|
| Breusch-Godfrey | Sem autocorrelacao serial | $p > 0.05$ |
| ARCH | Sem heterocedasticidade | $p > 0.05$ |
| Jarque-Bera | Normalidade | $p > 0.05$ |
| CUSUM | Parametros estaveis | Dentro das bandas |
| Bounds test | Sem cointegracao | $F > $ banda superior |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import ARDL

    # Estimar ARDL(2, 1, 1)
    model = ARDL(lags=2, exog_lags=[1, 1], trend='c')
    results = model.fit(y=y, exog=X)

    # Bounds test
    bounds = results.bounds_test()
    print(bounds.summary())

    # Coeficientes de longo prazo
    lr = results.long_run_coefficients()

    # Converter para ECM
    ecm = results.to_ecm()
    ```

=== "ARDL (R)"

    ```r
    library(ARDL)

    # Estimar ARDL(2, 1, 1)
    fit <- ardl(consumption ~ income + wealth, data = data,
                order = c(2, 1, 1))
    summary(fit)

    # Bounds test
    bounds_f_test(fit, case = 3)

    # Coeficientes de longo prazo
    multipliers(fit, type = "lr")

    # Reparametrizar como ECM
    ecm_fit <- recm(fit, case = 3)
    summary(ecm_fit)
    ```

=== "dynamac (R)"

    ```r
    library(dynamac)

    # Estimar ARDL e ECM simultaneamente
    fit <- dynardl(consumption ~ income + wealth, data = data,
                   lags = list("consumption" = 2,
                               "income" = 1,
                               "wealth" = 1),
                   ec = TRUE, simulate = TRUE)
    summary(fit)

    # Bounds test (Pesaran, Shin & Smith)
    pssbounds(fit, fstat = 5.423, obs = 120, case = 3, k = 2)

    # Multiplicadores dinamicos
    dynardl.simulation.plot(fit, response = "consumption",
                            bw = TRUE)
    ```

**Mapeamento de parametros**:

| chronobox | ARDL (R) | dynamac (R) | Descricao |
|---|---|---|---|
| `lags=2` | `order=c(2,...)` | `lags=list("y"=2)` | Defasagens de $y_t$ |
| `exog_lags=[1,1]` | `order=c(.,1,1)` | `lags=list("x1"=1)` | Defasagens das explicativas |
| `trend='c'` | `case=3` | --- | Constante irrestrita |
| `results.bounds_test()` | `bounds_f_test()` | `pssbounds()` | Bounds test |
| `results.long_run_coefficients()` | `multipliers(type="lr")` | --- | Multiplicadores LP |
| `results.to_ecm()` | `recm()` | `ec=TRUE` | Converter para ECM |

---

## Referencias

- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). Bounds testing approaches to the
  analysis of level relationships. *Journal of Applied Econometrics*, 16(3), 289--326.
- Pesaran, M. H. & Shin, Y. (1999). An autoregressive distributed-lag modelling approach
  to cointegration analysis. In S. Strom (Ed.), *Econometrics and Economic Theory in the
  20th Century: The Ragnar Frisch Centennial Symposium*. Cambridge University Press.
- Narayan, P. K. (2005). The saving and investment nexus for China: evidence from
  cointegration tests. *Applied Economics*, 37(17), 1979--1990.
- Kripfganz, S. & Schneider, D. C. (2023). ardl: Estimating autoregressive distributed
  lag and equilibrium correction models. *The Stata Journal*, 23(4), 983--1019.
