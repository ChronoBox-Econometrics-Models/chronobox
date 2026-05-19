---
title: STL Decomposition
description: Seasonal-Trend decomposition using LOESS --- algoritmo robusto e flexivel para decompor series temporais.
---

# STL Decomposition

!!! info "Quick Reference"
    - **Classe**: `chronobox.STL`
    - **Import**: `from chronobox import STL`
    - **Tipo**: Decomposicao aditiva
    - **R equivalente**: `stats::stl()` / `forecast::mstl()`
    - **Parametros-chave**: `period`, `seasonal`, `trend`, `robust`

---

## Overview

A **STL** (Seasonal and Trend decomposition using LOESS) e o metodo de referencia
para decomposicao de series temporais sazonais. Proposto por Cleveland et al. (1990),
o algoritmo utiliza suavizacao local (LOESS) de forma iterativa para separar a serie
em **tendencia**, **sazonalidade** e **residuo**:

$$
y_t = T_t + S_t + R_t
$$

### Vantagens do STL

- **Sazonalidade variavel**: o componente sazonal pode mudar ao longo do tempo
- **Robusto a outliers**: modo robusto com pesos bisquare reduz influencia de pontos atipicos
- **Controle do usuario**: janelas de suavizacao configuraveis para tendencia e sazonalidade
- **Qualquer periodicidade**: funciona com qualquer periodo sazonal (4, 7, 12, 52, ...)

### Quando usar

- Analise exploratoria de series com sazonalidade
- Ajuste sazonal quando flexibilidade e necessaria
- Pre-processamento para remover sazonalidade antes de modelar
- Series com outliers (usar `robust=True`)

---

## Formulacao Matematica

### Algoritmo LOESS

O nucleo do STL e a suavizacao **LOESS** (LOcally Estimated Scatterplot Smoothing).
Para cada ponto $x_0$, o LOESS ajusta uma regressao local ponderada usando os $q$
vizinhos mais proximos:

$$
\hat{f}(x_0) = \hat{\beta}_0, \quad \text{onde} \quad
(\hat{\beta}_0, \hat{\beta}_1) = \arg\min_{\beta} \sum_{i=1}^{q} w_i(x_0)
\left(y_i - \beta_0 - \beta_1(x_i - x_0)\right)^2
$$

Os pesos usam o **kernel tricube**:

$$
w_i(x_0) = W\!\left(\frac{|x_i - x_0|}{d_q(x_0)}\right), \qquad
W(u) = \begin{cases} (1 - u^3)^3, & 0 \le u < 1 \\ 0, & u \ge 1 \end{cases}
$$

onde $d_q(x_0)$ e a distancia ao $q$-esimo vizinho mais proximo.

### Procedimento Iterativo

O STL opera com dois loops aninhados:

**Inner loop** (repete `n_inner` vezes):

1. **Detrending**: $y_t^{(d)} = y_t - T_t$
2. **Cycle-subseries smoothing**: para cada posicao sazonal $s = 1, \ldots, m$,
   aplica LOESS a subserie $\{y_s^{(d)}, y_{s+m}^{(d)}, y_{s+2m}^{(d)}, \ldots\}$
   com janela `seasonal`
3. **Low-pass filter**: aplica $\text{MA}(m) \circ \text{MA}(m) \circ \text{MA}(3)$
   seguido de LOESS ao resultado do passo 2
4. **Seasonal**: $S_t = C_t - L_t$ (cycle-subseries menos low-pass)
5. **Trend smoothing**: aplica LOESS a $y_t - S_t$ com janela `trend`

**Outer loop** (repete `n_outer` vezes, se `robust=True`):

6. **Remainder**: $R_t = y_t - T_t - S_t$
7. **Robustness weights**: pesos bisquare baseados nos residuos

$$
\rho_t = B\!\left(\frac{|R_t|}{6 \cdot \text{median}(|R_t|)}\right), \qquad
B(u) = \begin{cases} (1 - u^2)^2, & 0 \le u < 1 \\ 0, & u \ge 1 \end{cases}
$$

### Parametros do Algoritmo

| Parametro | Simbolo | Default | Papel |
|---|---|---|---|
| `period` | $m$ | --- | Periodo sazonal |
| `seasonal` | $n_s$ | 7 | Janela LOESS para sazonalidade (impar, $\ge 7$) |
| `trend` | $n_t$ | auto | Janela LOESS para tendencia |
| `low_pass` | $n_l$ | $m$ | Janela LOESS para low-pass filter |
| `robust` | --- | `False` | Ativar pesos de robustez |
| `n_inner` | $n_i$ | 2 | Iteracoes do inner loop |
| `n_outer` | $n_o$ | 0 (15 se robusto) | Iteracoes do outer loop |

!!! note "Default do trend"
    O valor default da janela de tendencia e calculado como:

    $$
    n_t = \left\lceil \frac{1.5 \cdot m}{1 - 1.5 / n_s} \right\rceil_{\text{odd}}
    $$

    Isso garante que a tendencia seja suficientemente suave em relacao ao
    periodo sazonal.

---

## Quick Example

```python
from chronobox import STL
from chronobox.datasets import load_dataset

# Carregar passageiros airline (mensal, 144 obs)
y = load_dataset("airline")

# Decomposicao STL
stl = STL(period=12)
result = stl.fit(y.values)

# Resumo
print(result.summary())
```

```text
============================================================
                   Decomposition Results
============================================================
Model:              additive
Period:             12
No. Observations:   144
------------------------------------------------------------
Component              Mean          Std
------------------------------------------------------------
Trend                280.2985      77.3441
Seasonal               0.1873      19.4582
Remainder             -0.4858      10.3217
============================================================
```

---

## Guia Detalhado

### Construtor

```python
STL(
    period,             # Periodo sazonal (obrigatorio)
    seasonal=7,         # Janela LOESS para sazonalidade (impar >= 7)
    trend=None,         # Janela LOESS para tendencia (None = auto)
    low_pass=None,      # Janela LOESS para low-pass (None = period)
    robust=False,       # Ativar robustez contra outliers
    seasonal_deg=1,     # Grau do polinomio LOESS para sazonalidade
    trend_deg=1,        # Grau do polinomio LOESS para tendencia
    low_pass_deg=1,     # Grau do polinomio LOESS para low-pass
    n_inner=2,          # Iteracoes do inner loop
    n_outer=0           # Iteracoes do outer loop (0 = sem robustez)
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `period` | `int` | --- | Periodo sazonal ($\ge 2$) |
| `seasonal` | `int` | `7` | Janela LOESS sazonal (forcado impar, $\ge 7$) |
| `trend` | `int \| None` | `None` | Janela LOESS tendencia (auto se `None`) |
| `low_pass` | `int \| None` | `None` | Janela LOESS low-pass (period se `None`) |
| `robust` | `bool` | `False` | Habilitar iteracoes de robustez |
| `seasonal_deg` | `int` | `1` | Grau LOESS sazonal (0 = media, 1 = linear) |
| `trend_deg` | `int` | `1` | Grau LOESS tendencia |
| `low_pass_deg` | `int` | `1` | Grau LOESS low-pass |
| `n_inner` | `int` | `2` | Numero de iteracoes internas |
| `n_outer` | `int` | `0` | Numero de iteracoes de robustez |

### Metodo `fit()`

```python
result = stl.fit(endog)
```

| Parametro | Tipo | Descricao |
|---|---|---|
| `endog` | `array-like` | Serie temporal ($n \ge 2m$) |

Retorna um `DecompositionResult`:

| Atributo | Tipo | Descricao |
|---|---|---|
| `observed` | `NDArray` | Serie original |
| `trend` | `NDArray` | Componente de tendencia $T_t$ |
| `seasonal` | `NDArray` | Componente sazonal $S_t$ |
| `remainder` | `NDArray` | Residuo $R_t = y_t - T_t - S_t$ |
| `weights` | `NDArray \| None` | Pesos de robustez (se `robust=True`) |
| `period` | `int` | Periodo sazonal |
| `model` | `str` | `'additive'` |

### Efeito da Janela Sazonal

A janela `seasonal` controla quao rapido o componente sazonal pode mudar:

=== "seasonal=7 (default)"

    ```python
    stl = STL(period=12, seasonal=7)
    result = stl.fit(y.values)
    print(f"Std sazonal: {result.seasonal.std():.2f}")
    print(f"Std residuo: {result.remainder.std():.2f}")
    ```

    Sazonalidade mais flexivel --- acompanha mudancas ao longo do tempo.

=== "seasonal=15 (mais suave)"

    ```python
    stl = STL(period=12, seasonal=15)
    result = stl.fit(y.values)
    print(f"Std sazonal: {result.seasonal.std():.2f}")
    print(f"Std residuo: {result.remainder.std():.2f}")
    ```

    Sazonalidade mais estavel --- mais proxima de um padrao fixo.

=== "seasonal=21 (quase fixa)"

    ```python
    stl = STL(period=12, seasonal=21)
    result = stl.fit(y.values)
    print(f"Std sazonal: {result.seasonal.std():.2f}")
    print(f"Std residuo: {result.remainder.std():.2f}")
    ```

    Sazonalidade quase constante --- similar a decomposicao classica.

!!! tip "Regra pratica"
    - Valores menores de `seasonal` permitem mais variacao no padrao sazonal
    - Valores maiores forcam um padrao sazonal mais estavel
    - O minimo e 7 (recomendacao de Cleveland et al.)

### Modo Robusto

O modo robusto reduz a influencia de **outliers** nos componentes estimados:

```python
# STL padrao (sensivel a outliers)
stl_normal = STL(period=12)
result_normal = stl_normal.fit(y_with_outliers)

# STL robusto (resistente a outliers)
stl_robust = STL(period=12, robust=True)
result_robust = stl_robust.fit(y_with_outliers)

# Comparar pesos de robustez
print(f"Obs com peso < 0.5: {(result_robust.weights < 0.5).sum()}")
print(f"Obs com peso = 0:   {(result_robust.weights == 0).sum()}")
```

!!! warning "Custo computacional"
    O modo robusto executa `n_outer=15` iteracoes adicionais por default,
    tornando o ajuste significativamente mais lento. Use apenas quando
    houver suspeita de outliers.

### Plots dos Componentes

```python
import matplotlib.pyplot as plt

stl = STL(period=12)
result = stl.fit(y.values)

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

axes[0].plot(result.observed)
axes[0].set_ylabel("Observado")
axes[0].set_title("Decomposicao STL")

axes[1].plot(result.trend)
axes[1].set_ylabel("Tendencia")

axes[2].plot(result.seasonal)
axes[2].set_ylabel("Sazonal")

axes[3].plot(result.remainder)
axes[3].set_ylabel("Residuo")
axes[3].axhline(y=0, color='grey', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.show()
```

### Serie Dessazonalizada

```python
# Remover sazonalidade
y_sa = result.observed - result.seasonal

# Comparar original vs dessazonalizada
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(result.observed, label="Original", alpha=0.7)
ax.plot(y_sa, label="Dessazonalizada", linewidth=2)
ax.legend()
ax.set_title("Serie Original vs Dessazonalizada (STL)")
plt.show()
```

---

## Interpretacao

### Lendo o `summary()`

```python
print(result.summary())
```

| Campo | Significado |
|---|---|
| `Model` | Tipo de decomposicao (`additive`) |
| `Period` | Periodo sazonal utilizado |
| `No. Observations` | Numero de observacoes |
| `Trend Mean/Std` | Media e desvio-padrao da tendencia |
| `Seasonal Mean/Std` | Media e desvio do componente sazonal (media $\approx 0$ para aditiva) |
| `Remainder Mean/Std` | Media e desvio do residuo (media $\approx 0$) |

### Forca da Tendencia e Sazonalidade

Metricas de Hyndman & Athanasopoulos (2021) para quantificar a importancia
de cada componente:

```python
import numpy as np

# Forca da tendencia
var_remainder = np.var(result.remainder)
var_trend_remainder = np.var(result.trend + result.remainder)
F_trend = max(0, 1 - var_remainder / var_trend_remainder)

# Forca da sazonalidade
var_seasonal_remainder = np.var(result.seasonal + result.remainder)
F_seasonal = max(0, 1 - var_remainder / var_seasonal_remainder)

print(f"Forca da tendencia:    {F_trend:.4f}")
print(f"Forca da sazonalidade: {F_seasonal:.4f}")
```

$$
F_T = \max\!\left(0,\; 1 - \frac{\text{Var}(R_t)}{\text{Var}(T_t + R_t)}\right), \qquad
F_S = \max\!\left(0,\; 1 - \frac{\text{Var}(R_t)}{\text{Var}(S_t + R_t)}\right)
$$

Valores proximos de 1 indicam componente forte; proximos de 0 indicam fraco.

---

## Diagnosticos

### 1. Residuo deve ser Ruido Branco

```python
from chronobox.tests_stat import ljung_box_test

lb = ljung_box_test(result.remainder, lags=2 * result.period)
print(f"Ljung-Box p-value: {lb.pvalue:.4f}")
# p > 0.05 → residuo nao autocorrelacionado
```

!!! warning "Residuo autocorrelacionado"
    Se o teste rejeitar $H_0$, a decomposicao nao capturou toda a estrutura
    da serie. Tente ajustar as janelas `seasonal` e `trend` ou use o
    modo robusto.

### 2. Sazonalidade Residual

Verifique se nao restou sazonalidade no residuo:

```python
# ACF do residuo nos lags sazonais
import numpy as np

r = result.remainder
n = len(r)
for lag in [12, 24]:
    acf_val = np.corrcoef(r[:n-lag], r[lag:])[0, 1]
    print(f"ACF lag {lag}: {acf_val:.4f}")
    # Valores proximos de 0 indicam boa remocao sazonal
```

### 3. Pesos de Robustez (se `robust=True`)

```python
if result.weights is not None:
    low_weight = result.weights < 0.5
    print(f"Observacoes com peso baixo: {low_weight.sum()}")
    print(f"Indices: {np.where(low_weight)[0]}")
```

Observacoes com peso baixo sao candidatas a outliers.

### Checklist de Diagnostico

| Verificacao | Como testar | Resultado esperado |
|---|---|---|
| Residuo sem autocorrelacao | Ljung-Box | $p > 0.05$ |
| Sem sazonalidade residual | ACF nos lags sazonais | $\approx 0$ |
| Media do residuo | `np.mean(result.remainder)` | $\approx 0$ |
| Outliers identificados | Pesos de robustez | Peso $< 0.5$ |

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox import STL
    from chronobox.datasets import load_dataset

    y = load_dataset("airline")

    stl = STL(period=12, seasonal=7, robust=True)
    result = stl.fit(y.values)

    print(result.summary())
    ```

=== "stats (R)"

    ```r
    data(AirPassengers)
    y <- AirPassengers

    fit <- stl(y, s.window = 7, robust = TRUE)

    summary(fit)
    plot(fit)
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    fit <- mstl(AirPassengers, s.window = 7)

    autoplot(fit)
    ```

**Mapeamento de parametros**:

| chronobox | stats::stl (R) | forecast::mstl (R) |
|---|---|---|
| `period=12` | Definido no `ts()` | Definido no `ts()` |
| `seasonal=7` | `s.window=7` | `s.window=7` |
| `trend=None` | `t.window` (auto) | `t.window` (auto) |
| `robust=True` | `robust=TRUE` | `robust=TRUE` |
| `n_inner=2` | `inner=2` | `inner=2` |
| `n_outer=15` | `outer=15` | `outer=15` |
| `seasonal_deg=1` | `s.degree=1` | --- |
| `trend_deg=1` | `t.degree=1` | --- |
| `result.trend` | `fit$time.series[,"trend"]` | `fit[,"Trend"]` |
| `result.seasonal` | `fit$time.series[,"seasonal"]` | `fit[,"Seasonal"]` |
| `result.remainder` | `fit$time.series[,"remainder"]` | `fit[,"Remainder"]` |

!!! note "Diferenca entre `stl()` e `mstl()`"
    A funcao `mstl()` do pacote `forecast` suporta **multiplas sazonalidades**
    (ex: diaria + semanal + anual). O `stl()` basico trata apenas um periodo.
    O `chronobox.STL` atual suporta um unico periodo, similar ao `stl()`.

---

## Referencias

- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990).
  STL: A seasonal-trend decomposition procedure based on loess.
  *Journal of Official Statistics*, 6(1), 3--73.
- Cleveland, W. S. (1979). Robust locally weighted regression and smoothing
  scatterplots. *Journal of the American Statistical Association*, 74(368),
  829--836.
- Hyndman, R. J. & Athanasopoulos, G. (2021).
  *Forecasting: Principles and Practice*. 3rd ed. OTexts. Chapter 3.
- Dokumentov, A. & Hyndman, R. J. (2022). STR: Seasonal-Trend decomposition
  using Regression. *INFORMS Journal on Data Science*, 1(1), 50--62.
