---
title: Hodrick-Prescott Filter
description: Filtro HP para separacao tendencia-ciclo --- formulacao, propriedades espectrais, criticas e exemplos.
---

# Hodrick-Prescott Filter

!!! info "Quick Reference"
    - **Funcao**: `chronobox.filters.hp_filter`
    - **Import**: `from chronobox.filters import hp_filter`
    - **R equivalente**: `mFilter::hpfilter(y, freq=1600)`
    - **Retorno**: `(trend, cycle)` --- arrays NumPy

---

## Overview

O filtro de Hodrick e Prescott (1997) e o metodo mais utilizado em macroeconomia
para extrair a tendencia de uma serie temporal. Apesar de amplamente adotado por
bancos centrais e organismos internacionais, o filtro tem recebido criticas
importantes, especialmente de Hamilton (2018).

### Quando usar

- Extrair tendencia suave de series macroeconomicas
- Estimar output gap para analise conjuntural
- Comparacoes rapidas com literatura existente (que usa HP extensivamente)
- Quando voce esta ciente das limitacoes e quer resultados comparaveis

!!! warning "Criticas importantes"
    Hamilton (2018) argumenta que o HP filter nunca deveria ser usado porque:
    (1) produz ciclos espurios em series integradas, (2) sofre do end-point
    problem, e (3) nao tem fundamentacao estatistica formal. Considere o
    [Hamilton filter](hamilton.md) como alternativa.

---

## Formulacao Matematica

### Problema de Otimizacao

O filtro HP encontra a tendencia $\tau_t$ que resolve:

$$
\min_{\{\tau_t\}_{t=1}^T} \left\{
\sum_{t=1}^{T} (y_t - \tau_t)^2 +
\lambda \sum_{t=2}^{T-1} [(\tau_{t+1} - \tau_t) - (\tau_t - \tau_{t-1})]^2
\right\}
$$

O primeiro termo penaliza o **desvio da tendencia em relacao aos dados** (goodness of fit).
O segundo termo penaliza a **variacao na taxa de crescimento da tendencia** (suavidade).
O parametro $\lambda$ controla o trade-off entre ajuste e suavidade.

### Solucao Matricial

Definindo a matriz de diferencas de segunda ordem $K$ de dimensao $(T-2) \times T$:

$$
K = \begin{pmatrix}
1 & -2 & 1 & 0 & \cdots & 0 \\
0 & 1 & -2 & 1 & \cdots & 0 \\
\vdots & & \ddots & \ddots & \ddots & \vdots \\
0 & \cdots & 0 & 1 & -2 & 1
\end{pmatrix}
$$

A solucao analitica e:

$$
\hat{\tau} = (I_T + \lambda K^\top K)^{-1}\, y
$$

e o componente ciclico e:

$$
\hat{c} = y - \hat{\tau} = \lambda K^\top K (I_T + \lambda K^\top K)^{-1}\, y
$$

### Parametro $\lambda$

O parametro $\lambda$ controla a suavidade da tendencia:

- $\lambda \to 0$: tendencia = dados originais (sem filtragem)
- $\lambda \to \infty$: tendencia = reta (tendencia linear)

| Frequencia | $\lambda$ | Referencia |
|---|---|---|
| Anual | 6.25 | Ravn & Uhlig (2002) |
| Trimestral | 1,600 | Hodrick & Prescott (1997) |
| Mensal | 129,600 | Ravn & Uhlig (2002) |

!!! tip "Regra de Ravn-Uhlig"
    A regra de Ravn e Uhlig (2002) para converter $\lambda$ entre frequencias e:
    $\lambda_f = \lambda_q \cdot \left(\frac{f}{4}\right)^4$, onde $f$ e o numero
    de observacoes por ano. Assim, $\lambda_{\text{mensal}} = 1600 \times 3^4 = 129{,}600$.

### Propriedades Espectrais

No dominio da frequencia, o filtro HP atua como um **high-pass filter**. A funcao
de transferencia (gain) e:

$$
|G(\omega)|^2 = \left[\frac{4\lambda(1-\cos\omega)^2}{1 + 4\lambda(1-\cos\omega)^2}\right]^2
$$

Para $\lambda = 1600$ (trimestral), o filtro remove flutuacoes com periodicidade
acima de aproximadamente 40 trimestres (10 anos), preservando ciclos mais curtos.

---

## Quick Example

```python
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset
from chronobox.filters import hp_filter

# PIB trimestral dos EUA
gdp = load_dataset("us_gdp")
y = gdp.values

# Aplicar HP filter (lambda automatico para dados trimestrais)
trend, cycle = hp_filter(y, frequency="quarterly")

# Visualizar
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(y, label="PIB observado", color="steelblue")
axes[0].plot(trend, label="Tendencia HP", color="red", linewidth=2)
axes[0].set_title("PIB dos EUA --- Tendencia HP ($\\lambda = 1600$)")
axes[0].legend()

axes[1].plot(cycle, color="steelblue")
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].fill_between(range(len(cycle)), cycle, alpha=0.3)
axes[1].set_title("Componente Ciclico (Output Gap)")

plt.tight_layout()
plt.show()
```

---

## Guia Detalhado

### Assinatura da Funcao

```python
hp_filter(
    y,                    # Serie temporal (array-like, 1-D)
    lamb=None,            # Parametro lambda (float)
    frequency=None        # Frequencia dos dados
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `y` | `array-like` | --- | Serie temporal 1-D com pelo menos 4 observacoes |
| `lamb` | `float \| None` | `None` | Parametro de suavidade $\lambda$. Se `None`, inferido de `frequency` |
| `frequency` | `str \| None` | `None` | `"annual"`, `"quarterly"` ou `"monthly"`. Default: `"quarterly"` |

**Retorno**: tupla `(trend, cycle)` de `np.ndarray` com shape `(T,)`.

### Exemplos por Frequencia

=== "Trimestral ($\lambda = 1600$)"

    ```python
    trend, cycle = hp_filter(y, frequency="quarterly")
    # equivalente a:
    trend, cycle = hp_filter(y, lamb=1600)
    ```

=== "Mensal ($\lambda = 129600$)"

    ```python
    trend, cycle = hp_filter(y, frequency="monthly")
    # equivalente a:
    trend, cycle = hp_filter(y, lamb=129600)
    ```

=== "Anual ($\lambda = 6.25$)"

    ```python
    trend, cycle = hp_filter(y, frequency="annual")
    # equivalente a:
    trend, cycle = hp_filter(y, lamb=6.25)
    ```

### Efeito de $\lambda$

```python
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset
from chronobox.filters import hp_filter

gdp = load_dataset("us_gdp")
y = gdp.values

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y, label="Dados", color="gray", alpha=0.5)

for lam, cor in [(10, "green"), (1600, "red"), (100000, "purple")]:
    trend, _ = hp_filter(y, lamb=lam)
    ax.plot(trend, label=f"$\\lambda = {lam}$", color=cor, linewidth=2)

ax.set_title("Efeito de $\\lambda$ na suavidade da tendencia")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Criticas de Hamilton (2018)

James Hamilton argumenta em "Why You Should Never Use the Hodrick-Prescott Filter"
que o HP filter tem problemas graves:

1. **Ciclos espurios**: aplicado a series integradas (random walk), o HP filter
   produz padroes ciclicos que nao existem nos dados originais

2. **End-point problem**: as estimativas da tendencia nas extremidades da amostra
   sao muito sensiveis a novas observacoes, exatamente onde a analise conjuntural
   mais importa

3. **Sem fundamentacao estatistica**: nao existe um DGP (data generating process)
   bem definido para o qual o HP filter seja o estimador otimo

4. **Dependencia de $\lambda$**: resultados mudam substancialmente com a escolha
   de $\lambda$, e nao existe metodo objetivo para escolhe-lo

Veja o [Hamilton filter](hamilton.md) para a alternativa proposta.

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox.filters import hp_filter

    trend, cycle = hp_filter(y, lamb=1600)
    ```

=== "mFilter (R)"

    ```r
    library(mFilter)

    hp <- hpfilter(y, freq = 1600)
    trend <- hp$trend
    cycle <- hp$cycle
    ```

=== "stats/neverhpfilter (R)"

    ```r
    # Alternativa: Hamilton filter via neverhpfilter
    library(neverhpfilter)

    ham <- yth_filter(y, h = 8, p = 4)
    ```

**Mapeamento de parametros**:

| chronobox | mFilter (R) | Descricao |
|---|---|---|
| `lamb=1600` | `freq=1600` | Parametro de suavidade $\lambda$ |
| `frequency="quarterly"` | `freq=1600` | Atalho para $\lambda$ padrao |
| `trend` (retorno) | `hp$trend` | Componente de tendencia |
| `cycle` (retorno) | `hp$cycle` | Componente ciclico |

---

## Referencias

- Hodrick, R. J. & Prescott, E. C. (1997). Postwar U.S. Business Cycles:
  An Empirical Investigation. *Journal of Money, Credit and Banking*, 29(1), 1--16.
- Ravn, M. O. & Uhlig, H. (2002). On Adjusting the Hodrick-Prescott Filter
  for the Frequency of Observations. *Review of Economics and Statistics*,
  84(2), 371--376.
- Hamilton, J. D. (2018). Why You Should Never Use the Hodrick-Prescott Filter.
  *Review of Economics and Statistics*, 100(5), 831--843.
- King, R. G. & Rebelo, S. T. (1993). Low Frequency Filtering and Real
  Business Cycles. *Journal of Economic Dynamics and Control*, 17(1--2), 207--231.
