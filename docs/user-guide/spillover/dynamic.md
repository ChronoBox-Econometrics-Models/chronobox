---
title: Dynamic Spillover
description: Rolling window spillover para analise de conectividade variante no tempo.
---

# Dynamic Spillover

!!! info "Quick Reference"
    - **Classe**: `chronobox.analysis.SpilloverIndex`
    - **Metodo**: `SpilloverIndex.rolling(data, window)`
    - **Resultado**: `RollingSpilloverResult`
    - **Referencia**: Diebold & Yilmaz (2012)

---

## Overview

O spillover **dinamico** estende a analise estatica ao calcular indices de
spillover para **janelas moveis** (rolling windows), produzindo series
temporais de conectividade que revelam como a estrutura de transmissao de
choques evolui ao longo do tempo.

Periodos de **alta conectividade** tipicamente coincidem com crises
financeiras, contagio entre mercados ou choques macroeconomicos, enquanto
periodos de **baixa conectividade** indicam mercados mais isolados.

---

## Formulacao Matematica

### Rolling Window

Para uma janela de tamanho $W$, o procedimento em cada ponto $t$ e:

1. Selecionar a sub-amostra $[\mathbf{y}_{t-W+1}, \ldots, \mathbf{y}_t]$
2. Estimar VAR(p) na sub-amostra
3. Calcular GFEVD e indices de spillover
4. Armazenar $S^H(t)$, $S_{i \leftarrow}^H(t)$, $S_{\rightarrow j}^H(t)$, $S_j^{\text{net}}(t)$

O resultado e uma serie temporal de cada indice:

$$
\{S^H(t)\}_{t=W}^{T}, \quad \{S_{i \leftarrow}^H(t)\}_{t=W}^{T}, \quad \{S_{\rightarrow j}^H(t)\}_{t=W}^{T}, \quad \{S_j^{\text{net}}(t)\}_{t=W}^{T}
$$

### Numero de Janelas

Para uma serie de comprimento $T$ e janela $W$:

$$
n_{\text{windows}} = T - W + 1
$$

### Tradeoff Bias-Variancia da Janela

- **Janela grande** ($W \uparrow$): estimativas mais estaveis, mas menos
  responsivas a mudancas estruturais
- **Janela pequena** ($W \downarrow$): captura mudancas rapidas, mas com
  maior variancia nas estimativas

---

## Quick Example

```python
import numpy as np
from chronobox.analysis import SpilloverIndex

# Gerar dados com mudanca estrutural
np.random.seed(42)
T = 800
K = 3
e = np.random.randn(T, K)

data = np.zeros((T, K))
for t in range(1, T):
    if t < 400:
        # Periodo tranquilo: baixa conectividade
        data[t, 0] = 0.3 * data[t-1, 0] + e[t, 0]
        data[t, 1] = 0.2 * data[t-1, 1] + 0.1 * data[t-1, 0] + e[t, 1]
        data[t, 2] = 0.2 * data[t-1, 2] + e[t, 2]
    else:
        # Periodo de crise: alta conectividade
        data[t, 0] = 0.2 * data[t-1, 0] + e[t, 0]
        data[t, 1] = 0.1 * data[t-1, 1] + 0.5 * data[t-1, 0] + e[t, 1]
        data[t, 2] = 0.1 * data[t-1, 2] + 0.4 * data[t-1, 0] + 0.3 * data[t-1, 1] + e[t, 2]

# Rolling spillover
sp = SpilloverIndex(lags=2, horizon=10)
rolling = sp.rolling(data, window=200)

# Plot: observar aumento do spillover na "crise"
rolling.plot_total()
```

---

## Guia Detalhado

### Metodo `rolling()`

```python
rolling_result = sp.rolling(
    data,            # Dados multivariados (T, K)
    window=200       # Tamanho da janela movel
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `data` | `array_like` | --- | Dados multivariados, shape $(T, K)$ |
| `window` | `int` | `200` | Tamanho da janela rolling |

**Restricoes**:

- `window` $\leq T$
- `window` $> p + H + K$ (precisa de observacoes suficientes para estimar o VAR)

### Atributos do `RollingSpilloverResult`

| Atributo | Tipo | Descricao |
|---|---|---|
| `total_spillover` | `ndarray (n_windows,)` | Serie temporal do TSI |
| `directional_from` | `ndarray (n_windows, K)` | Serie temporal do FROM por variavel |
| `directional_to` | `ndarray (n_windows, K)` | Serie temporal do TO por variavel |
| `net_spillover` | `ndarray (n_windows, K)` | Serie temporal do NET por variavel |
| `window_size` | `int` | Tamanho da janela usada |
| `dates` | `ndarray or None` | Indices temporais (fim de cada janela) |

### Metodos do `RollingSpilloverResult`

| Metodo | Descricao |
|---|---|
| `plot_total()` | Plota a evolucao temporal do Total Spillover Index |

---

## Evolucao Temporal do TSI

### Plot do Total Spillover

```python
import matplotlib.pyplot as plt

sp = SpilloverIndex(lags=2, horizon=10)
rolling = sp.rolling(data, window=200)

# Plot automatico
rolling.plot_total()
```

### Plot Customizado

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(rolling.dates, rolling.total_spillover, linewidth=1.2, color='steelblue')
ax.set_title('Total Spillover Index (Rolling Window = 200)')
ax.set_ylabel('Spillover (%)')
ax.set_xlabel('Observacao')
ax.grid(True, alpha=0.3)

# Destacar periodos de alta conectividade
mean_sp = rolling.total_spillover.mean()
ax.axhline(mean_sp, color='red', linestyle='--', alpha=0.7, label=f'Media: {mean_sp:.1f}%')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Spillover Direcional Dinamico

### FROM e TO ao Longo do Tempo

```python
import matplotlib.pyplot as plt

labels = ['Var 0', 'Var 1', 'Var 2']
K = len(labels)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Directional FROM
for i in range(K):
    axes[0].plot(rolling.dates, rolling.directional_from[:, i],
                 label=labels[i], linewidth=1)
axes[0].set_title('Directional FROM Spillover (Rolling)')
axes[0].set_ylabel('FROM (%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Directional TO
for i in range(K):
    axes[1].plot(rolling.dates, rolling.directional_to[:, i],
                 label=labels[i], linewidth=1)
axes[1].set_title('Directional TO Spillover (Rolling)')
axes[1].set_ylabel('TO (%)')
axes[1].set_xlabel('Observacao')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Net Spillover Dinamico

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 5))
for i in range(K):
    ax.plot(rolling.dates, rolling.net_spillover[:, i],
            label=labels[i], linewidth=1)
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
ax.set_title('Net Spillover (Rolling)')
ax.set_ylabel('Net Spillover (%)')
ax.set_xlabel('Observacao')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

!!! tip "Interpretacao do Net dinamico"
    Quando o net spillover de uma variavel cruza de negativo para positivo,
    ela passa de **receptora** a **transmissora** de choques. Essas mudancas
    de regime sao particularmente informativas para entender a dinamica de
    crises.

---

## Identificando Periodos de Alta/Baixa Conectividade

```python
import numpy as np

# Estatisticas descritivas
tsi = rolling.total_spillover
print(f"TSI - Media: {tsi.mean():.2f}%, Std: {tsi.std():.2f}%")
print(f"TSI - Min: {tsi.min():.2f}%, Max: {tsi.max():.2f}%")

# Periodos acima de 1 desvio padrao da media
threshold = tsi.mean() + tsi.std()
high_connectivity = np.where(tsi > threshold)[0]
print(f"\nPeriodos de alta conectividade (TSI > {threshold:.1f}%):")
print(f"  {len(high_connectivity)} janelas de {len(tsi)} total")

# Periodos abaixo de 1 desvio padrao
low_threshold = tsi.mean() - tsi.std()
low_connectivity = np.where(tsi < low_threshold)[0]
print(f"\nPeriodos de baixa conectividade (TSI < {low_threshold:.1f}%):")
print(f"  {len(low_connectivity)} janelas de {len(tsi)} total")
```

---

## Exemplo Completo: Crise Financeira e Conectividade

```python
import numpy as np
import matplotlib.pyplot as plt
from chronobox.analysis import SpilloverIndex

# Simular cenario de crise financeira
np.random.seed(2024)
T = 1200
K = 4
labels = ['EUA', 'Europa', 'Asia', 'EM']

# Estrutura: baixa conectividade -> crise -> recuperacao
e = np.random.randn(T, K)
data = np.zeros((T, K))

for t in range(1, T):
    if t < 400:
        # Pre-crise: baixa conectividade
        A = np.array([[0.3, 0.05, 0.02, 0.01],
                       [0.05, 0.25, 0.03, 0.02],
                       [0.02, 0.03, 0.28, 0.01],
                       [0.01, 0.02, 0.01, 0.30]])
    elif t < 700:
        # Crise: alta conectividade
        A = np.array([[0.20, 0.15, 0.10, 0.08],
                       [0.20, 0.15, 0.12, 0.10],
                       [0.15, 0.12, 0.18, 0.08],
                       [0.12, 0.10, 0.08, 0.20]])
    else:
        # Pos-crise: conectividade moderada
        A = np.array([[0.28, 0.08, 0.04, 0.03],
                       [0.08, 0.22, 0.05, 0.04],
                       [0.04, 0.05, 0.25, 0.03],
                       [0.03, 0.04, 0.03, 0.27]])

    data[t] = A @ data[t-1] + e[t]

# Rolling spillover
sp = SpilloverIndex(lags=2, horizon=10)
rolling = sp.rolling(data, window=200)

# --- Visualizacao completa ---
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# 1. Total Spillover Index
axes[0].plot(rolling.dates, rolling.total_spillover, 'steelblue', linewidth=1.2)
axes[0].axvspan(400, 700, alpha=0.15, color='red', label='Crise')
axes[0].set_title('Total Spillover Index')
axes[0].set_ylabel('TSI (%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Directional TO
for i in range(K):
    axes[1].plot(rolling.dates, rolling.directional_to[:, i],
                 label=labels[i], linewidth=1)
axes[1].axvspan(400, 700, alpha=0.15, color='red')
axes[1].set_title('Directional TO Spillover')
axes[1].set_ylabel('TO (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Net Spillover
for i in range(K):
    axes[2].plot(rolling.dates, rolling.net_spillover[:, i],
                 label=labels[i], linewidth=1)
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].axvspan(400, 700, alpha=0.15, color='red')
axes[2].set_title('Net Spillover')
axes[2].set_ylabel('Net (%)')
axes[2].set_xlabel('Observacao')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Spillover Dinamico: Pre-Crise, Crise e Pos-Crise', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
```

---

## Network Plots

Visualizacoes de rede permitem identificar a **topologia de conectividade**
entre variaveis. No contexto de spillover, os nos representam variaveis e as
arestas representam a intensidade da transmissao de choques.

### Rede de Spillover Estatico

```python
import numpy as np
import matplotlib.pyplot as plt

# Computar spillover estatico para a rede
sp = SpilloverIndex(lags=2, horizon=10)
result = sp.fit(data)

labels = ['EUA', 'Europa', 'Asia', 'EM']
K = len(labels)

# Layout circular
angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
x = np.cos(angles)
y = np.sin(angles)

fig, ax = plt.subplots(figsize=(8, 8))

# Desenhar arestas (pairwise spillover)
for i in range(K):
    for j in range(K):
        if i != j:
            weight = result.fevd_table[i, j]
            if weight > 0.05:  # threshold para visualizacao
                ax.annotate("",
                    xy=(x[i], y[i]), xytext=(x[j], y[j]),
                    arrowprops=dict(
                        arrowstyle="->",
                        linewidth=weight * 10,
                        alpha=0.6,
                        color='steelblue'
                    ))

# Desenhar nos
node_sizes = np.abs(result.net_spillover) * 50 + 200
colors = ['#e74c3c' if ns > 0 else '#3498db' for ns in result.net_spillover]
for i in range(K):
    ax.scatter(x[i], y[i], s=node_sizes[i], c=colors[i],
               zorder=5, edgecolors='black', linewidth=1.5)
    ax.annotate(labels[i], (x[i], y[i]),
                textcoords="offset points", xytext=(0, 15),
                ha='center', fontsize=12, fontweight='bold')

ax.set_title('Spillover Network\n(Vermelho = Transmissor, Azul = Receptor)')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.show()
```

!!! note "Interpretacao da rede"
    - **Tamanho do no**: proporcional ao valor absoluto do net spillover
    - **Cor vermelha**: transmissor liquido (net > 0)
    - **Cor azul**: receptor liquido (net < 0)
    - **Espessura da aresta**: proporcional a intensidade do spillover pairwise

---

## Escolha do Tamanho da Janela

| Frequencia dos dados | Janela recomendada | Justificativa |
|---|---|---|
| Diario | 150--250 | Captura trimestre a semestre de dados |
| Semanal | 52--104 | 1 a 2 anos de dados |
| Mensal | 36--60 | 3 a 5 anos de dados |
| Trimestral | 20--40 | 5 a 10 anos de dados |

!!! warning "Janela muito pequena"
    Se $W \leq p + H + K$, nao ha observacoes suficientes para estimar o
    VAR dentro da janela. O metodo `rolling()` levantara um `ValueError`
    neste caso.

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox.analysis import SpilloverIndex

    sp = SpilloverIndex(lags=2, horizon=10)
    rolling = sp.rolling(data, window=200)

    rolling.plot_total()
    print(f"Media TSI: {rolling.total_spillover.mean():.2f}%")
    ```

=== "frequencyConnectedness (R)"

    ```r
    library(frequencyConnectedness)
    library(vars)

    # Rolling spillover
    sp <- spilloverRollingDY12(
      data,
      n.ahead = 10,
      no.corr = FALSE,
      window = 200
    )

    # Plot total spillover
    plotOverall(sp)

    # Plot directional FROM/TO
    plotFrom(sp)
    plotTo(sp)
    plotNet(sp)
    ```

---

## Referencias

- Diebold, F. X. & Yilmaz, K. (2012). Better to Give than to Receive:
  Predictive Directional Measurement of Volatility Spillovers. *International
  Journal of Forecasting*, 28(1), 57--66.
- Diebold, F. X. & Yilmaz, K. (2014). On the Network Topology of Variance
  Decompositions: Measuring the Connectedness of Financial Firms. *Journal of
  Econometrics*, 182(1), 119--134.
- Pesaran, M. H. & Shin, Y. (1998). Generalized Impulse Response Analysis in
  Linear Multivariate Models. *Economics Letters*, 58(1), 17--29.
- Barunik, J. & Krehlik, T. (2018). Measuring the Frequency Dynamics of
  Financial Connectedness and Systemic Risk. *Journal of Financial
  Econometrics*, 16(2), 271--296.
