---
title: ACF & PACF Plots
description: Graficos de autocorrelacao e autocorrelacao parcial com bandas de confianca.
---

# ACF & PACF Plots

!!! info "Quick Reference"
    - **Funcao**: `plot_diagnostics()` (painel 2x2 inclui ACF/PACF)
    - **Import**: `from chronobox.visualization import plot_diagnostics`
    - **Bandas**: Bartlett ($\pm 1.96 / \sqrt{n}$)
    - **Uso principal**: Identificacao da ordem de modelos ARIMA

---

## Overview

Os graficos de ACF (Autocorrelation Function) e PACF (Partial Autocorrelation
Function) sao ferramentas fundamentais para:

- **Identificar a ordem** de modelos AR, MA e ARMA
- **Diagnosticar residuos** de modelos estimados
- **Detectar sazonalidade** e padroes de dependencia temporal

### Quando usar

- Antes de estimar: identificar a ordem $(p, d, q)$ de um ARIMA
- Apos estimar: verificar se os residuos sao ruido branco
- Explorar dados: detectar dependencia temporal, sazonalidade

---

## Formulacao Matematica

### ACF --- Funcao de Autocorrelacao

A autocorrelacao no lag $k$ mede a correlacao linear entre $y_t$ e $y_{t-k}$:

$$
\hat{\rho}(k) = \frac{\sum_{t=k+1}^{T} (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{T} (y_t - \bar{y})^2}
$$

**Propriedades**: $\hat{\rho}(0) = 1$, $|\hat{\rho}(k)| \leq 1$.

### PACF --- Funcao de Autocorrelacao Parcial

A PACF no lag $k$ mede a correlacao entre $y_t$ e $y_{t-k}$ **controlando
pelos lags intermediarios** $y_{t-1}, \ldots, y_{t-k+1}$:

$$
\phi_{kk} = \text{Corr}(y_t - \hat{y}_t, \; y_{t-k} - \hat{y}_{t-k})
$$

onde $\hat{y}_t$ e $\hat{y}_{t-k}$ sao projecoes lineares sobre
$y_{t-1}, \ldots, y_{t-k+1}$.

O chronobox calcula a PACF via algoritmo de **Durbin-Levinson**.

### Bandas de Confianca (Bartlett)

Sob a hipotese nula de ruido branco, a ACF amostral e aproximadamente normal:

$$
\hat{\rho}(k) \sim N\left(0, \frac{1}{T}\right) \quad \text{para } T \text{ grande}
$$

As bandas de confianca a 95% sao:

$$
\pm \frac{1.96}{\sqrt{T}}
$$

!!! warning "Bartlett vs. White Noise"
    As bandas $\pm 1.96/\sqrt{T}$ assumem ruido branco. Para processos MA(q),
    a formula de Bartlett generalizada e:

    $$\text{Var}[\hat{\rho}(k)] \approx \frac{1}{T}\left(1 + 2\sum_{j=1}^{q} \hat{\rho}(j)^2\right) \quad \text{para } k > q$$

---

## ACF/PACF via Diagnosticos

O modo mais simples de gerar graficos ACF/PACF e atraves do painel de diagnosticos,
que inclui ACF e PACF lado a lado:

```python
from chronobox.visualization import plot_diagnostics

# Apos estimar um modelo
results = model.fit(data)

# Painel 2x2: residuos, ACF/PACF, QQ-plot, histograma
fig = plot_diagnostics(results, lags=30)
fig.savefig("diagnosticos.png", dpi=150, bbox_inches="tight")
```

### Apenas com array de residuos

```python
import numpy as np
from chronobox.visualization import plot_diagnostics

residuals = np.random.randn(200)  # exemplo
fig = plot_diagnostics(residuals=residuals, lags=20)
```

---

## ACF/PACF Standalone

Para gerar graficos ACF e PACF individuais, use matplotlib diretamente com
as funcoes internas do chronobox:

```python
import numpy as np
import matplotlib.pyplot as plt
from chronobox.visualization.diagnostics_plot import _compute_acf, _compute_pacf

y = np.random.randn(300)
lags = 40

acf_vals = _compute_acf(y, lags)
pacf_vals = _compute_pacf(y, lags)

# Bandas de confianca
n = len(y)
sig = 1.96 / np.sqrt(n)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ACF
axes[0].bar(range(lags + 1), acf_vals, color="#2980b9", alpha=0.7, width=0.6)
axes[0].axhline(y=sig, color="red", linestyle="--", linewidth=0.8)
axes[0].axhline(y=-sig, color="red", linestyle="--", linewidth=0.8)
axes[0].axhline(y=0, color="black", linewidth=0.5)
axes[0].set_title("ACF", fontweight="bold")
axes[0].set_xlabel("Lag")

# PACF
axes[1].bar(range(lags + 1), pacf_vals, color="#e74c3c", alpha=0.7, width=0.6)
axes[1].axhline(y=sig, color="red", linestyle="--", linewidth=0.8)
axes[1].axhline(y=-sig, color="red", linestyle="--", linewidth=0.8)
axes[1].axhline(y=0, color="black", linewidth=0.5)
axes[1].set_title("PACF", fontweight="bold")
axes[1].set_xlabel("Lag")

fig.suptitle("ACF e PACF", fontsize=14, fontweight="bold")
fig.tight_layout()
```

---

## Interpretacao Visual

A chave para identificar modelos ARIMA esta no comportamento conjunto de ACF e PACF:

### AR(p) --- ACF decai, PACF corta

| Lag | ACF | PACF |
|---|---|---|
| 1 | Alto | **Significativo** |
| 2 | Decai | **Significativo** (se AR(2)) |
| 3+ | Decai exponencialmente | Zero (apos lag $p$) |

```python
# Exemplo: AR(2)
import numpy as np

np.random.seed(42)
n = 500
e = np.random.randn(n)
y = np.zeros(n)
for t in range(2, n):
    y[t] = 0.6 * y[t-1] - 0.3 * y[t-2] + e[t]

# ACF: decaimento exponencial/oscilatório
# PACF: corta apos lag 2
```

!!! tip "Regra pratica --- AR(p)"
    Se a PACF **corta abruptamente** apos o lag $p$ (cai dentro das bandas)
    e a ACF **decai gradualmente** (exponencial ou oscilatoria), o processo
    e provavelmente AR($p$).

### MA(q) --- ACF corta, PACF decai

| Lag | ACF | PACF |
|---|---|---|
| 1 | **Significativo** | Alto |
| 2 | **Significativo** (se MA(2)) | Decai |
| 3+ | Zero (apos lag $q$) | Decai exponencialmente |

```python
# Exemplo: MA(1)
np.random.seed(42)
n = 500
e = np.random.randn(n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = e[t] + 0.7 * e[t-1]

# ACF: corta apos lag 1
# PACF: decaimento exponencial
```

!!! tip "Regra pratica --- MA(q)"
    Se a ACF **corta abruptamente** apos o lag $q$ e a PACF **decai
    gradualmente**, o processo e provavelmente MA($q$).

### ARMA(p,q) --- Ambos decaem

Quando **ambos** ACF e PACF decaem gradualmente (sem corte abrupto), o
processo possui componentes AR e MA simultaneamente:

| Processo | ACF | PACF |
|---|---|---|
| AR(p) | Decai | Corta apos lag $p$ |
| MA(q) | Corta apos lag $q$ | Decai |
| ARMA(p,q) | Decai | Decai |
| Ruido Branco | Zero (dentro das bandas) | Zero (dentro das bandas) |

---

## Sazonalidade na ACF

Dados sazonais apresentam picos na ACF nos lags multiplos do periodo sazonal:

```python
# Dados mensais com sazonalidade (periodo = 12)
# ACF mostrara picos nos lags 12, 24, 36, ...

fig = plot_diagnostics(
    residuals=seasonal_data,
    lags=48,  # pelo menos 2 ciclos sazonais
    title="ACF/PACF com Sazonalidade",
)
```

!!! note "Identificacao sazonal"
    Se a ACF apresenta picos significativos nos lags $s, 2s, 3s, \ldots$
    (onde $s$ e o periodo sazonal), considere um modelo SARIMA com
    componente sazonal.

---

## Parametros de `plot_diagnostics`

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `results` | objeto | `None` | Resultado de modelo com atributo `.resid` |
| `residuals` | ndarray | `None` | Array de residuos (se `results` nao fornecido) |
| `lags` | int | `20` | Numero de lags para ACF/PACF |
| `figsize` | tuple | `(12, 10)` | Tamanho da figura |
| `title` | str | `None` | Titulo geral do painel |

---

## Exemplos com Diferentes Processos

=== "Ruido Branco"

    ```python
    import numpy as np
    from chronobox.visualization import plot_diagnostics

    # Ruido branco: todos os lags dentro das bandas
    white_noise = np.random.randn(500)
    fig = plot_diagnostics(residuals=white_noise, lags=30,
                           title="Ruido Branco")
    ```

=== "AR(1)"

    ```python
    # AR(1): PACF corta apos lag 1, ACF decai exponencialmente
    np.random.seed(42)
    n = 500
    e = np.random.randn(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.8 * y[t-1] + e[t]

    fig = plot_diagnostics(residuals=y, lags=30, title="AR(1): phi=0.8")
    ```

=== "MA(2)"

    ```python
    # MA(2): ACF corta apos lag 2, PACF decai
    np.random.seed(42)
    n = 500
    e = np.random.randn(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = e[t] + 0.5 * e[t-1] - 0.3 * e[t-2]

    fig = plot_diagnostics(residuals=y, lags=30, title="MA(2)")
    ```

=== "Random Walk"

    ```python
    # Random walk: ACF decai lentamente (persistencia)
    np.random.seed(42)
    rw = np.random.randn(500).cumsum()

    fig = plot_diagnostics(residuals=rw, lags=40,
                           title="Random Walk (nao estacionario)")
    ```
