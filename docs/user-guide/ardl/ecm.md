---
title: Error Correction Model (ECM)
description: ECM derivado do ARDL --- velocidade de ajuste, multiplicadores de curto e longo prazo, e multiplicadores dinamicos.
---

# Error Correction Model (ECM)

!!! info "Quick Reference"
    - **Classe**: `chronobox.ECM` (ou via `ARDL.to_ecm()`)
    - **Import**: `from chronobox import ECM`
    - **R equivalente**: `ARDL::recm()`, `dynamac::dynardl(ec=TRUE)`
    - **Pre-requisito**: bounds test rejeita $H_0$ (evidencia de cointegracao)

---

## Overview

O ECM (Error Correction Model) e a **reparametrizacao** do ARDL em forma de
correcao de erros. Enquanto o ARDL modela as variaveis em niveis, o ECM separa
explicitamente:

- **Dinamica de longo prazo**: a relacao de equilibrio entre as variaveis
- **Dinamica de curto prazo**: como o sistema responde a desvios do equilibrio
- **Velocidade de ajuste** ($\alpha$): quao rapidamente o sistema retorna ao equilibrio

### Quando usar

- Apos o bounds test rejeitar $H_0$ (existe cointegracao)
- Para separar efeitos de curto e longo prazo
- Para estimar a velocidade de ajuste ao equilibrio
- Para calcular multiplicadores dinamicos

!!! warning "Pre-requisito"
    So faz sentido estimar o ECM se houver evidencia de cointegracao (bounds
    test rejeita $H_0$). Sem cointegracao, o termo de correcao de erros nao
    tem interpretacao economica e o modelo em diferencas e mais apropriado.

---

## Formulacao Matematica

### Derivacao a partir do ARDL

Considere um ARDL(1, 1) simples:

$$
y_t = c + \phi_1 y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + \epsilon_t
$$

Subtraindo $y_{t-1}$ de ambos os lados:

$$
y_t - y_{t-1} = c + (\phi_1 - 1) y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + \epsilon_t
$$

Somando e subtraindo $\beta_0 x_{t-1}$:

$$
\Delta y_t = c + (\phi_1 - 1) y_{t-1} + (\beta_0 + \beta_1) x_{t-1} + \beta_0 \Delta x_t + \epsilon_t
$$

Definindo $\alpha = \phi_1 - 1$ e $\theta = \frac{\beta_0 + \beta_1}{1 - \phi_1}$:

$$
\boxed{\Delta y_t = c + \alpha \left( y_{t-1} - \theta \, x_{t-1} \right) + \beta_0 \Delta x_t + \epsilon_t}
$$

O termo $\left( y_{t-1} - \theta \, x_{t-1} \right)$ e o **erro de equilibrio**:
o desvio da relacao de longo prazo.

### Forma Geral

Para um ARDL($p, q_1, \ldots, q_k$), o ECM correspondente e:

$$
\Delta y_t = c + \underbrace{\alpha \left( y_{t-1} - \sum_{j=1}^{k} \theta_j \, x_{j,t-1} \right)}_{\text{correcao de erros}} + \underbrace{\sum_{i=1}^{p-1} \gamma_i \, \Delta y_{t-i} + \sum_{j=1}^{k} \sum_{\ell=0}^{q_j - 1} \delta_{j,\ell} \, \Delta x_{j,t-\ell}}_{\text{dinamica de curto prazo}} + \epsilon_t
$$

onde:

- $\alpha = -(1 - \sum_{i=1}^{p} \phi_i)$ e a **velocidade de ajuste** (deve ser negativa)
- $\theta_j = \frac{\sum_{\ell=0}^{q_j} \beta_{j,\ell}}{1 - \sum_{i=1}^{p} \phi_i}$ sao os **multiplicadores de longo prazo**
- $\gamma_i$ e $\delta_{j,\ell}$ capturam a **dinamica de curto prazo**

### Interpretacao dos Parametros

| Parametro | Simbolo | Interpretacao |
|---|---|---|
| Velocidade de ajuste | $\alpha$ | Fracao do desvio corrigida por periodo. Deve ser $\alpha < 0$ |
| Multiplicador de longo prazo | $\theta_j$ | Efeito total de $x_j$ sobre $y$ no equilibrio |
| Coef. de curto prazo | $\delta_{j,0}$ | Efeito imediato de $\Delta x_j$ sobre $\Delta y$ |
| Persistencia | $\gamma_i$ | Dinamica de ajustamento da propria variavel |

!!! tip "Velocidade de ajuste"
    O parametro $\alpha$ deve ser **negativo e significativo** para que o
    mecanismo de correcao de erros funcione. Por exemplo, $\alpha = -0.25$
    significa que 25% do desvio do equilibrio e corrigido a cada periodo.

    - $\alpha$ proximo de $0$: ajuste lento (ou sem equilibrio)
    - $\alpha$ proximo de $-1$: ajuste rapido (quase completo em 1 periodo)
    - $\alpha > 0$: **explosivo** --- o sistema diverge do equilibrio

---

## Multiplicadores

### Short-Run Multiplier (Impacto)

O efeito imediato de uma mudanca unitaria em $x_j$ sobre $y$:

$$
\text{SR}_j = \delta_{j,0}
$$

### Long-Run Multiplier (Equilibrio)

O efeito total apos o sistema convergir para o novo equilibrio:

$$
\text{LR}_j = \theta_j = \frac{\sum_{\ell=0}^{q_j} \beta_{j,\ell}}{1 - \sum_{i=1}^{p} \phi_i}
$$

### Dynamic Multipliers

Os multiplicadores dinamicos rastreiam o efeito **periodo a periodo** de um choque
unitario permanente em $x_j$ sobre $y$:

$$
m_j(h) = \frac{\partial y_{t+h}}{\partial x_t}, \quad h = 0, 1, 2, \ldots
$$

A sequencia $\{m_j(0), m_j(1), m_j(2), \ldots\}$ converge para $\theta_j$ conforme
$h \to \infty$, mostrando a **trajetoria de ajustamento** do curto para o longo prazo.

!!! note "Interpretacao grafica"
    O grafico dos multiplicadores dinamicos mostra:

    - **Impacto inicial**: $m_j(0) = \delta_{j,0}$ (efeito de curto prazo)
    - **Trajetoria**: como o efeito evolui ao longo do tempo
    - **Convergencia**: o nivel para o qual converge e o multiplicador de longo prazo $\theta_j$
    - **Velocidade**: quao rapido converge depende de $\alpha$

---

## Quick Example

```python
from chronobox import ARDL
from chronobox.datasets import load_macro

import pandas as pd

# Carregar dados
data = load_macro()
y = data["consumption"]
X = data[["income", "wealth"]]

# 1. Estimar ARDL
model = ARDL(lags=2, exog_lags=[1, 1], trend='c')
results = model.fit(y=y, exog=X)

# 2. Verificar cointegracao
bounds = results.bounds_test()
print(f"Bounds test F-stat: {bounds.f_stat:.3f}")
print(f"Decisao: {bounds.decision}")

# 3. Converter para ECM
ecm = results.to_ecm()
print(ecm.summary())

# 4. Multiplicadores
print(f"\nVelocidade de ajuste (alpha): {ecm.alpha:.4f}")
print(f"\nMultiplicadores de curto prazo:")
print(ecm.short_run_coefficients())
print(f"\nMultiplicadores de longo prazo:")
print(ecm.long_run_coefficients())

# 5. Multiplicadores dinamicos
dm = ecm.dynamic_multipliers(steps=30)
print(dm)
```

```text
               ECM (Error Correction Model) Results
==========================================================================
Dep. Variable:       Δconsumption    No. Observations:          118
Method:                       OLS    R-squared:              0.4523
Trend:                   Constant    Adj. R-squared:         0.4278
==========================================================================

                       coef     std err       t     P>|t|
--------------------------------------------------------------------------
const                 0.2341     0.0876    2.672    0.009
EC(t-1)              -0.5122     0.0834   -6.141    0.000
Δconsumption(t-1)     0.1245     0.0712    1.749    0.083
Δincome(t)            0.2534     0.0645    3.929    0.000
Δwealth(t)            0.0412     0.0198    2.081    0.040

==========================================================================
Long-Run Equation: consumption = 0.457 + 0.655*income + 0.133*wealth
Speed of Adjustment: α = -0.5122 (51.2% per period)
==========================================================================

Velocidade de ajuste (alpha): -0.5122

Multiplicadores de curto prazo:
             coef     std err       t     P>|t|
income      0.2534     0.0645    3.929    0.000
wealth      0.0412     0.0198    2.081    0.040

Multiplicadores de longo prazo:
             coef     std err       t     P>|t|
income      0.6548     0.0923    7.094    0.000
wealth      0.1325     0.0412    3.217    0.002
```

---

## Guia Detalhado

### Duas Formas de Estimar

=== "Via ARDL (recomendado)"

    ```python
    from chronobox import ARDL

    # Estimar ARDL primeiro, depois reparametrizar
    model = ARDL(lags=2, exog_lags=[1, 1], trend='c')
    results = model.fit(y=y, exog=X)

    # Converter para ECM
    ecm = results.to_ecm()
    ```

=== "Diretamente"

    ```python
    from chronobox import ECM

    # Estimar ECM diretamente (requer relacao de longo prazo conhecida)
    model = ECM(lags=1, exog_lags=[0, 0], trend='c')
    results = model.fit(y=y, exog=X)
    ```

!!! tip "Qual abordagem usar?"
    A abordagem via ARDL e **preferida** porque permite aplicar o bounds test
    antes de estimar o ECM, garantindo que a reparametrizacao e valida.

### Construtor ECM

```python
ECM(
    lags=1,            # Defasagens de Δy_t
    exog_lags=0,       # Defasagens de Δx_t (int ou list)
    trend='c',         # Componente deterministico
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `lags` | `int` | `1` | Defasagens de $\Delta y_t$ no ECM |
| `exog_lags` | `int \| list` | `0` | Defasagens de $\Delta x_{j,t}$ |
| `trend` | `str` | `'c'` | `'n'`, `'c'`, `'t'`, `'ct'` |

### Atributos do Resultado ECM

| Atributo | Tipo | Descricao |
|---|---|---|
| `ecm.alpha` | `float` | Velocidade de ajuste ($\alpha$) |
| `ecm.long_run_coefficients()` | `pd.DataFrame` | Multiplicadores de longo prazo ($\theta_j$) |
| `ecm.short_run_coefficients()` | `pd.DataFrame` | Coeficientes de curto prazo ($\delta_{j,\ell}$) |
| `ecm.ec_term` | `pd.Series` | Serie do termo de correcao de erros |
| `ecm.residuals` | `pd.Series` | Residuos do ECM |

### Multiplicadores Dinamicos

```python
# Calcular multiplicadores dinamicos
dm = ecm.dynamic_multipliers(steps=30)

# dm e um DataFrame com colunas por variavel
print(dm.head(10))
```

```text
   step    income    wealth
      0    0.2534    0.0412
      1    0.3891    0.0687
      2    0.4623    0.0856
      3    0.5089    0.0965
      5    0.5612    0.1098
     10    0.6234    0.1243
     15    0.6445    0.1301
     20    0.6521    0.1319
     25    0.6543    0.1324
     30    0.6548    0.1325
```

O multiplicador converge para o valor de longo prazo. A velocidade de convergencia
depende de $\alpha$: quanto mais negativo, mais rapida a convergencia.

### Grafico dos Multiplicadores Dinamicos

```python
from chronobox.visualization import plot_dynamic_multipliers

# Grafico com intervalos de confianca (bootstrap)
fig = plot_dynamic_multipliers(
    ecm,
    steps=30,
    ci=0.95,
    variables=["income", "wealth"],
)
fig.show()
```

O grafico mostra:

- Linha solida: multiplicador pontual em cada horizonte
- Area sombreada: intervalo de confianca a 95% (via bootstrap)
- Linha tracejada horizontal: multiplicador de longo prazo ($\theta_j$)
- Convergencia visual do curto para o longo prazo

### Previsao com ECM

```python
# Previsao h passos a frente
fc = ecm.forecast(steps=12, exog_future=X_future)

print(fc["forecast"])
print(fc["confidence_interval"])
```

!!! warning "Previsao com ECM"
    Para previsao fora da amostra, voce precisa fornecer valores futuros das
    variaveis explicativas (`exog_future`). Se as explicativas tambem precisam
    ser previstas, considere usar um sistema VAR/VECM.

---

## Interpretacao

### Lendo o Summary do ECM

**Coeficiente EC(t-1)**: este e o $\alpha$ (velocidade de ajuste). Deve ser:

- **Negativo**: o sistema corrige desvios do equilibrio
- **Significativo**: o mecanismo de correcao e ativo
- **Entre -1 e 0**: ajuste monotono ao equilibrio

| Valor de $\alpha$ | Interpretacao |
|---|---|
| $-0.10$ | Ajuste lento: 10% do desvio corrigido por periodo |
| $-0.50$ | Ajuste moderado: 50% por periodo |
| $-0.90$ | Ajuste rapido: 90% por periodo |
| $-1.00$ | Ajuste completo em 1 periodo |
| $< -1$ | Overshooting: corrige mais que o desvio (oscilatoria) |
| $> 0$ | Explosivo: o sistema diverge do equilibrio |

**Coeficientes $\Delta x_j$**: efeitos de **curto prazo**. Uma mudanca temporaria
em $x_j$ tem efeito apenas transitorio sobre $y$.

**Equacao de longo prazo**: a relacao de equilibrio. Uma mudanca **permanente**
em $x_j$ leva a uma mudanca permanente de $\theta_j$ em $y$.

### Exemplo de Interpretacao Economica

Considere o ECM estimado para a funcao consumo:

$$
\Delta C_t = 0.23 - 0.51(C_{t-1} - 0.65 Y_{t-1} - 0.13 W_{t-1}) + 0.25 \Delta Y_t + 0.04 \Delta W_t + \epsilon_t
$$

Interpretacao:

1. **Longo prazo**: no equilibrio, $C^* = 0.46 + 0.65 Y^* + 0.13 W^*$
    - Uma unidade a mais de renda permanente aumenta o consumo em 0.65
    - Uma unidade a mais de riqueza permanente aumenta o consumo em 0.13

2. **Curto prazo**: um aumento temporario de 1 unidade na renda aumenta o
   consumo em 0.25 no periodo corrente

3. **Velocidade de ajuste**: $\alpha = -0.51$ significa que 51% do desvio do
   equilibrio e corrigido a cada periodo. A meia-vida do ajuste e:

$$
t_{1/2} = \frac{\ln(0.5)}{\ln(1 + \alpha)} = \frac{\ln(0.5)}{\ln(0.49)} \approx 0.97 \text{ periodos}
$$

!!! tip "Meia-vida"
    A meia-vida mede quantos periodos sao necessarios para corrigir metade do
    desvio do equilibrio. Para $\alpha = -0.51$, a meia-vida e menor que 1
    periodo, indicando ajuste muito rapido.

---

## Diagnosticos

### 1. Verificacao do Termo de Correcao de Erros

```python
# O EC term deve ser estacionario (I(0))
from chronobox.tests_stat import adf_test

ec_series = ecm.ec_term
adf = adf_test(ec_series)
print(f"ADF p-value do EC term: {adf.pvalue:.4f}")
# p < 0.05 → EC term estacionario (consistente com cointegracao)
```

### 2. Autocorrelacao dos Residuos

```python
from chronobox.tests_stat import breusch_godfrey_test

bg = breusch_godfrey_test(ecm.residuals, lags=4)
print(f"Breusch-Godfrey p-value: {bg.pvalue:.4f}")
```

### 3. Estabilidade dos Parametros

```python
from chronobox.tests_stat import cusum_test

cs = cusum_test(ecm.residuals)
print(f"CUSUM estavel: {cs.is_stable}")
```

### Checklist de Diagnostico

| Teste | $H_0$ | Resultado Desejado |
|---|---|---|
| EC term estacionario (ADF) | Raiz unitaria | $p < 0.05$ (rejeitar) |
| $\alpha < 0$ e significativo | --- | $t < 0$ e $p < 0.05$ |
| Breusch-Godfrey | Sem autocorrelacao | $p > 0.05$ |
| CUSUM | Parametros estaveis | Dentro das bandas |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import ARDL

    # Estimar ARDL e converter para ECM
    model = ARDL(lags=2, exog_lags=[1, 1], trend='c')
    results = model.fit(y=y, exog=X)

    ecm = results.to_ecm()
    print(ecm.summary())

    # Multiplicadores
    sr = ecm.short_run_coefficients()
    lr = ecm.long_run_coefficients()
    dm = ecm.dynamic_multipliers(steps=30)

    # Grafico
    from chronobox.visualization import plot_dynamic_multipliers
    plot_dynamic_multipliers(ecm, steps=30)
    ```

=== "ARDL (R)"

    ```r
    library(ARDL)

    # Estimar ARDL
    fit <- ardl(consumption ~ income + wealth, data = data,
                order = c(2, 1, 1))

    # Reparametrizar como ECM (Restricted ECM)
    ecm_fit <- recm(fit, case = 3)
    summary(ecm_fit)

    # Multiplicadores de curto e longo prazo
    multipliers(fit, type = "sr")
    multipliers(fit, type = "lr")

    # Multiplicadores dinamicos (interim)
    multipliers(fit, type = 15)  # 15 periodos
    ```

=== "dynamac (R)"

    ```r
    library(dynamac)

    # Estimar ARDL com ECM
    fit <- dynardl(consumption ~ income + wealth, data = data,
                   lags = list("consumption" = 2,
                               "income" = 1,
                               "wealth" = 1),
                   ec = TRUE, simulate = TRUE,
                   shockvar = "income")

    summary(fit)

    # Grafico de multiplicadores dinamicos
    dynardl.simulation.plot(fit, response = "consumption",
                            bw = TRUE)
    ```

**Mapeamento de parametros**:

| chronobox | ARDL (R) | dynamac (R) | Descricao |
|---|---|---|---|
| `results.to_ecm()` | `recm(fit, case=3)` | `ec=TRUE` | ECM reparametrizado |
| `ecm.alpha` | coef de `ect` | coef de `l.1.consumption` | Velocidade de ajuste |
| `ecm.short_run_coefficients()` | `multipliers(type="sr")` | --- | Multiplicadores CP |
| `ecm.long_run_coefficients()` | `multipliers(type="lr")` | --- | Multiplicadores LP |
| `ecm.dynamic_multipliers(steps=h)` | `multipliers(type=h)` | `simulate=TRUE` | Multiplicadores dinamicos |
| `ecm.ec_term` | --- | --- | Serie do EC term |

---

## Referencias

- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). Bounds testing approaches to the
  analysis of level relationships. *Journal of Applied Econometrics*, 16(3), 289--326.
- Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error correction:
  representation, estimation, and testing. *Econometrica*, 55(2), 251--276.
- Hassler, U. & Wolters, J. (2006). Autoregressive distributed lag models and
  cointegration. *Allgemeines Statistisches Archiv*, 90(1), 59--74.
- Jordan, S. & Philips, A. Q. (2018). Cointegration testing and dynamic simulations
  of autoregressive distributed lag models. *The Stata Journal*, 18(4), 902--923.
- Banerjee, A., Dolado, J. J. & Mestre, R. (1998). Error-correction mechanism tests
  for cointegration in a single-equation framework. *Journal of Time Series Analysis*,
  19(3), 267--283.
