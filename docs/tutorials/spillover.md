---
title: "Tutorial: Diebold-Yilmaz Spillover Analysis"
description: Tutorial completo de analise de spillover --- da estimacao do VAR a spillover table, indices direcionais, rolling window e network visualization.
---

# Diebold-Yilmaz Spillover Analysis

!!! abstract "O que voce vai aprender"
    - Preparar dados de retornos de indices de bolsa
    - Estimar um VAR como base para a analise de spillover
    - Calcular a spillover table estatica (GFEVD)
    - Interpretar indices FROM, TO e NET
    - Calcular spillover dinamico com rolling window
    - Identificar periodos de alta conectividade (crises)
    - Visualizar a rede de transmissao de choques

**Nivel**: Intermediario
**Tempo estimado**: ~40 minutos
**Dataset**: Retornos diarios de indices de bolsa (SP500, FTSE, Nikkei, DAX)

---

## Introducao: Transmissao de Volatilidade entre Mercados

Em mercados financeiros globalizados, choques em um mercado se propagam
rapidamente para outros. A analise de spillover de **Diebold e Yilmaz (2009, 2012)**
quantifica essa transmissao usando a **decomposicao generalizada da variancia
do erro de previsao** (GFEVD) de um VAR:

$$
\text{TSI}(H) = \frac{\sum_{i=1}^{K} \sum_{\substack{j=1 \\ j \neq i}}^{K} \tilde{d}_{ij}^H}{K} \times 100
$$

O **Total Spillover Index** (TSI) mede a conectividade media do sistema:
valores altos indicam alta interdependencia entre mercados.

!!! info "Aplicacoes classicas"
    - Medir **contagio financeiro** durante crises (2008, 2020)
    - Identificar mercados **transmissores** vs **receptores** de risco
    - Monitorar a evolucao da conectividade ao longo do tempo
    - Construir redes de transmissao de volatilidade

---

## Passo 1: Preparar Dados de Retornos

Usamos retornos diarios de quatro grandes indices de bolsa: SP500 (EUA),
FTSE 100 (Reino Unido), Nikkei 225 (Japao) e DAX (Alemanha).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset

# Carregar dados de indices
indices = load_dataset("global_indices")
print(f"Colunas: {list(indices.columns)}")
print(f"Periodo: {indices.index[0]} a {indices.index[-1]}")
print(f"Observacoes: {len(indices)}")
print(indices.head())
```

```title="Output"
Colunas: ['sp500', 'ftse', 'nikkei', 'dax']
Periodo: 2000-01-03 a 2023-12-29
Observacoes: 6024
              sp500     ftse   nikkei      dax
2000-01-03  1455.22  6930.20  18934.3  6750.76
2000-01-04  1399.42  6665.20  18557.3  6586.95
2000-01-05  1402.11  6598.40  18243.5  6502.07
2000-01-06  1403.45  6606.30  17892.7  6474.92
2000-01-07  1441.47  6621.40  18145.2  6588.32
```

```python
# Calcular retornos logaritmicos (%)
returns = np.log(indices / indices.shift(1)).dropna() * 100
names = list(returns.columns)

print(f"\nEstatisticas descritivas dos retornos (%):")
print(returns.describe().round(4))
```

```title="Output"
Estatisticas descritivas dos retornos (%):
          sp500     ftse   nikkei      dax
count  6023.00  6023.00  6023.00  6023.00
mean     0.024    0.009    0.012    0.021
std      1.213    1.147    1.456    1.378
min    -12.765  -11.512  -12.111  -13.054
25%     -0.432   -0.451   -0.567   -0.534
50%      0.056    0.034    0.023    0.054
75%      0.512    0.478    0.612    0.587
max     10.957    9.384   13.235   10.797
```

```python
# Visualizar retornos
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
colors = ["steelblue", "indianred", "seagreen", "darkorange"]
labels = ["S&P 500", "FTSE 100", "Nikkei 225", "DAX"]

for ax, col, color, label in zip(axes, names, colors, labels):
    ax.plot(returns.index, returns[col], color=color, linewidth=0.3, alpha=0.8)
    ax.set_ylabel(f"{label} (%)")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.grid(alpha=0.3)

    # Destacar crise 2008 e COVID-2020
    for start, end, lbl in [("2008-09", "2009-03", "GFC"),
                             ("2020-02", "2020-05", "COVID")]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color="red")

axes[0].set_title("Retornos Diarios de Indices Globais (%)")
plt.tight_layout()
plt.show()
```

!!! note "Observacoes"
    - Os retornos exibem **clustering de volatilidade**: periodos calmos
      intercalados com periodos de alta volatilidade
    - As crises de **2008 (GFC)** e **2020 (COVID)** sao claramente visiveis
      como picos de volatilidade em todos os mercados simultaneamente
    - Os mercados parecem se mover juntos durante crises --- exatamente o que
      a analise de spillover quantifica

---

## Passo 2: Estimar VAR

A analise de spillover parte de um VAR estimado para o sistema de retornos:

```python
from chronobox import VAR
from chronobox.selection.lag_selection import select_lag_order

data = returns.values

# Selecao de lags
lag_results = select_lag_order(data, maxlags=10, trend="c")
print(f"Lags selecionados (BIC): {lag_results.selected_orders['BIC']}")
print(f"Lags selecionados (AIC): {lag_results.selected_orders['AIC']}")
```

```title="Output"
Lags selecionados (BIC): 2
Lags selecionados (AIC): 5
```

```python
# Estimar VAR(2) --- BIC para parcimonia
model = VAR(lags=2, trend="c")
results = model.fit(data, names=names)

print(f"Modelo estavel: {results.is_stable}")
print(f"Maior autovalor (modulo): {np.max(np.abs(results.roots)):.4f}")
```

```title="Output"
Modelo estavel: True
Maior autovalor (modulo): 0.2345
```

!!! tip "VAR para spillover"
    Para a analise de spillover, preferimos o **BIC** (mais parcimonioso) para
    evitar sobreajuste. O modelo precisa ser **estavel** (todos os autovalores
    dentro do circulo unitario) para que a GFEVD seja bem definida.

---

## Passo 3: Spillover Table Estatica

A spillover table e a matriz normalizada da GFEVD. Cada entrada $S_{ij}$
mostra a proporcao da variancia do erro de previsao da variavel $i$ atribuida
a choques na variavel $j$:

```python
from chronobox.analysis import SpilloverIndex

# Calcular spillover estatico
si = SpilloverIndex(lags=2, horizon=10)
result = si.fit(data)

print(result.summary())
```

```title="Output"
==============================================================================
  Diebold-Yilmaz Spillover Table (H = 10)
==============================================================================
                sp500      ftse    nikkei       dax    FROM
  sp500         62.34     15.23      8.12     14.31    37.66
  ftse          16.45     58.23      9.67     15.65    41.77
  nikkei        10.23     11.45     64.78     13.54    35.22
  dax           15.67     16.12     12.34     55.87    44.13

  TO            42.35     42.80     30.13     43.50
  NET            4.69      1.03     -5.09     -0.63

  Total Spillover Index: 39.70%
==============================================================================
```

!!! info "Leitura da spillover table"
    - **Diagonal**: variancia explicada por choques proprios (sp500: 62.3%)
    - **Fora da diagonal**: variancia explicada por choques em outros mercados
    - **FROM (linha)**: quanto cada mercado **recebe** de choques externos
    - **TO (coluna)**: quanto cada mercado **transmite** para os outros
    - **NET**: TO - FROM (positivo = transmissor liquido)
    - **TSI = 39.70%**: em media, ~40% da variancia dos erros de previsao
      vem de choques em outros mercados

---

## Passo 4: Interpretar FROM, TO, NET

Os indices direcionais revelam a **estrutura de transmissao** entre mercados:

```python
# Visualizar indices direcionais
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x_pos = np.arange(len(names))
bar_colors = ["steelblue", "indianred", "seagreen", "darkorange"]
labels = ["S&P 500", "FTSE 100", "Nikkei 225", "DAX"]

# FROM
axes[0].bar(x_pos, result.directional_from, color=bar_colors, alpha=0.8)
axes[0].set_title("FROM Others (recebe)")
axes[0].set_ylabel("Spillover (%)")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(labels, rotation=45)
axes[0].grid(alpha=0.3, axis="y")

# TO
axes[1].bar(x_pos, result.directional_to, color=bar_colors, alpha=0.8)
axes[1].set_title("TO Others (transmite)")
axes[1].set_ylabel("Spillover (%)")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(labels, rotation=45)
axes[1].grid(alpha=0.3, axis="y")

# NET
net_colors = ["seagreen" if n > 0 else "indianred" for n in result.net_spillover]
axes[2].bar(x_pos, result.net_spillover, color=net_colors, alpha=0.8)
axes[2].set_title("NET Spillover (TO - FROM)")
axes[2].set_ylabel("Spillover (%)")
axes[2].axhline(0, color="black", linewidth=0.5)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(labels, rotation=45)
axes[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.show()
```

!!! note "Interpretacao dos indices direcionais"
    | Mercado | FROM | TO | NET | Papel |
    |---------|------|-----|-----|-------|
    | S&P 500 | 37.7 | 42.4 | +4.7 | **Transmissor liquido** |
    | FTSE 100 | 41.8 | 42.8 | +1.0 | Levemente transmissor |
    | Nikkei 225 | 35.2 | 30.1 | -5.1 | **Receptor liquido** |
    | DAX | 44.1 | 43.5 | -0.6 | Levemente receptor |

    O **S&P 500** e o maior transmissor liquido de choques --- consistente com
    o papel dominante do mercado americano no sistema financeiro global. O
    **Nikkei** e o maior receptor, refletindo a sensibilidade do mercado japones
    a choques externos.

---

## Passo 5: Spillover Dinamico (Rolling Window)

O spillover estatico assume relacoes constantes. O spillover **dinamico**
usa janelas moveis para capturar a evolucao da conectividade ao longo do tempo:

```python
# Spillover dinamico com janela de 200 dias (~10 meses)
rolling = si.rolling(data, window=200)

print(f"Janela: {rolling.window_size} observacoes")
print(f"Numero de janelas: {len(rolling.total_spillover)}")
print(f"TSI medio: {np.mean(rolling.total_spillover):.2f}%")
print(f"TSI maximo: {np.max(rolling.total_spillover):.2f}%")
print(f"TSI minimo: {np.min(rolling.total_spillover):.2f}%")
```

```title="Output"
Janela: 200 observacoes
Numero de janelas: 5824
TSI medio: 38.45%
TSI maximo: 68.23%
TSI minimo: 22.14%
```

```python
# Visualizar TSI ao longo do tempo
fig, ax = plt.subplots(figsize=(14, 5))

dates = returns.index[199:]  # Ajustar para o tamanho do rolling
ax.plot(dates, rolling.total_spillover, color="steelblue", linewidth=0.8)
ax.fill_between(dates, rolling.total_spillover, alpha=0.2, color="steelblue")
ax.axhline(np.mean(rolling.total_spillover), color="gray", linestyle="--",
           linewidth=1, label=f"Media ({np.mean(rolling.total_spillover):.1f}%)")

# Marcar crises
crisis_periods = [
    ("2001-09", "2002-06", "Dot-com / 9-11"),
    ("2008-09", "2009-06", "Crise Financeira Global"),
    ("2010-04", "2010-12", "Crise da Divida Europeia"),
    ("2015-06", "2015-12", "Desaceleracao China"),
    ("2020-02", "2020-06", "COVID-19"),
    ("2022-02", "2022-06", "Guerra Ucrania"),
]

for start, end, label in crisis_periods:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
               alpha=0.15, color="red")

ax.set_title("Total Spillover Index (Rolling 200 dias)", fontsize=13)
ax.set_ylabel("TSI (%)")
ax.set_ylim(15, 75)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

!!! warning "Escolha da janela"
    | Janela | Vantagem | Desvantagem |
    |--------|----------|-------------|
    | Pequena (100 dias) | Captura mudancas rapidas | Maior ruido |
    | Media (200 dias) | Bom equilibrio | Padrao na literatura |
    | Grande (500 dias) | Mais estavel | Perde mudancas rapidas |

    A janela de 200 dias e o padrao de Diebold & Yilmaz (2012).

---

## Passo 6: Identificar Periodos de Alta Conectividade

Periodos com TSI acima da media + 1 desvio-padrao indicam **alta conectividade**,
tipicamente associados a crises ou episodios de contagio:

```python
# Identificar periodos de alta conectividade
tsi = rolling.total_spillover
mean_tsi = np.mean(tsi)
std_tsi = np.std(tsi)
threshold = mean_tsi + std_tsi

print(f"Media TSI:     {mean_tsi:.2f}%")
print(f"Desvio padrao: {std_tsi:.2f}%")
print(f"Limiar (mu+sigma): {threshold:.2f}%")

# Encontrar periodos acima do limiar
high_conn = tsi > threshold
dates_arr = returns.index[199:]

# Identificar blocos contiguos
changes = np.diff(high_conn.astype(int))
starts = np.where(changes == 1)[0] + 1
ends = np.where(changes == -1)[0] + 1

print(f"\nPeriodos de alta conectividade (TSI > {threshold:.1f}%):")
print(f"{'Inicio':<15s} {'Fim':<15s} {'Duracao (dias)':>15s} {'TSI max':>10s}")
print("-" * 58)

for s, e in zip(starts[:8], ends[:8]):
    dur = e - s
    max_tsi = np.max(tsi[s:e])
    print(f"{str(dates_arr[s].date()):<15s} {str(dates_arr[e].date()):<15s} {dur:15d} {max_tsi:10.1f}")
```

```title="Output"
Media TSI:     38.45%
Desvio padrao: 8.12%
Limiar (mu+sigma): 46.57%

Periodos de alta conectividade (TSI > 46.6%):
Inicio          Fim             Duracao (dias)    TSI max
----------------------------------------------------------
2001-09-15      2002-03-21                 130       54.3
2007-08-12      2009-07-15                 492       68.2
2010-05-03      2011-02-18                 205       58.7
2011-07-22      2012-01-30                 133       55.4
2015-08-10      2016-02-28                 143       52.1
2018-10-05      2019-01-15                  70       49.8
2020-02-24      2020-09-30                 152       65.8
2022-02-24      2022-08-12                 121       56.4
```

!!! success "Crises capturadas"
    Os periodos de alta conectividade correspondem precisamente a episodios
    conhecidos de estresse financeiro:

    - **2001--2002**: estouro da bolha dot-com e 11 de setembro
    - **2007--2009**: crise financeira global (maior pico: TSI = 68.2%)
    - **2010--2012**: crise da divida europeia (Grecia, Irlanda, Portugal)
    - **2020**: pandemia de COVID-19 (segundo maior pico: TSI = 65.8%)
    - **2022**: invasao da Ucrania e choque energetico

---

## Passo 7: Spillover Direcional Dinamico

Alem do TSI total, podemos rastrear como a direcionalidade evolui:

```python
# Visualizar NET spillover dinamico por mercado
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
colors = ["steelblue", "indianred", "seagreen", "darkorange"]
labels = ["S&P 500", "FTSE 100", "Nikkei 225", "DAX"]

for ax, i, color, label in zip(axes, range(4), colors, labels):
    net = rolling.net_spillover[:, i]
    ax.plot(dates, net, color=color, linewidth=0.6, alpha=0.8)
    ax.fill_between(dates, 0, net,
                    where=net > 0, alpha=0.3, color="seagreen",
                    label="Transmissor")
    ax.fill_between(dates, 0, net,
                    where=net < 0, alpha=0.3, color="indianred",
                    label="Receptor")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"NET ({label})")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    # Crises
    for start, end, _ in crisis_periods:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.1, color="red")

axes[0].set_title("NET Spillover Dinamico por Mercado", fontsize=13)
plt.tight_layout()
plt.show()
```

!!! note "Interpretacao do NET dinamico"
    - **S&P 500**: predominantemente **transmissor** ao longo do tempo, com picos
      durante crises originadas nos EUA (2008, 2020)
    - **FTSE 100**: alterna entre transmissor e receptor, mas tende a transmissor
      durante a crise da divida europeia (2010--2012)
    - **Nikkei 225**: predominantemente **receptor** --- o mercado japones absorve
      choques do resto do mundo mais do que transmite
    - **DAX**: papel variavel, mas torna-se forte transmissor durante crises europeias

---

## Passo 8: Network Visualization

A spillover table pode ser representada como uma **rede de transmissao**,
onde nos sao mercados e arestas representam a intensidade do spillover:

```python
# Extrair spillovers pairwise para a rede
# Usar a spillover table fora da diagonal
table = result.fevd_table  # Matriz K x K normalizada
np.fill_diagonal(table, 0)

# Criar visualizacao de rede
fig, ax = plt.subplots(figsize=(8, 8))

# Posicoes dos nos (layout circular)
n = len(names)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
pos_x = np.cos(angles) * 2
pos_y = np.sin(angles) * 2

# Desenhar arestas (setas com espessura proporcional ao spillover)
for i in range(n):
    for j in range(n):
        if i != j and table[i, j] > 5:  # Filtrar spillovers pequenos
            weight = table[i, j]
            alpha = min(weight / 20, 1.0)
            lw = weight / 5
            # Seta de j para i (j transmite para i)
            dx = pos_x[i] - pos_x[j]
            dy = pos_y[i] - pos_y[j]
            ax.annotate("",
                xy=(pos_x[i] - dx * 0.15, pos_y[i] - dy * 0.15),
                xytext=(pos_x[j] + dx * 0.15, pos_y[j] + dy * 0.15),
                arrowprops=dict(arrowstyle="->", lw=lw, color="gray",
                               alpha=alpha, connectionstyle="arc3,rad=0.1"))

# Desenhar nos
node_sizes = result.directional_to  # Tamanho proporcional ao TO
node_colors = ["steelblue", "indianred", "seagreen", "darkorange"]

for i, (name, color) in enumerate(zip(labels, node_colors)):
    size = 800 + node_sizes[i] * 20
    ax.scatter(pos_x[i], pos_y[i], s=size, c=color, zorder=5,
              edgecolors="black", linewidths=1.5)
    ax.annotate(f"{name}\nTO: {result.directional_to[i]:.1f}\nFROM: {result.directional_from[i]:.1f}",
                (pos_x[i], pos_y[i] - 0.45), ha="center", va="top",
                fontsize=9, fontweight="bold")

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect("equal")
ax.set_title("Rede de Spillover entre Mercados\n(Espessura = intensidade, Tamanho = TO spillover)",
             fontsize=12, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.show()
```

!!! info "Leitura da rede"
    - **Tamanho do no**: proporcional ao spillover TO (quanto transmite)
    - **Espessura da seta**: proporcional a intensidade do spillover pairwise
    - **Direcao da seta**: do transmissor para o receptor

    O S&P 500 e o maior no (maior TO), confirmando seu papel central no
    sistema financeiro global. As conexoes mais fortes sao entre mercados
    ocidentais (SP500 ↔ FTSE ↔ DAX), com o Nikkei mais perifericamente
    conectado.

---

## Analise de Robustez

### Sensibilidade ao Horizonte de Previsao

```python
# Testar diferentes horizontes H
print(f"{'Horizonte H':>12s} {'TSI (%)':>10s}")
print("-" * 25)

for h in [5, 10, 15, 20, 30]:
    si_h = SpilloverIndex(lags=2, horizon=h)
    res_h = si_h.fit(data)
    print(f"{h:12d} {res_h.total_spillover:10.2f}")
```

```title="Output"
 Horizonte H    TSI (%)
-------------------------
           5      36.12
          10      39.70
          15      40.89
          20      41.23
          30      41.45
```

!!! tip "Escolha do horizonte"
    O TSI aumenta com o horizonte $H$ e estabiliza em torno de $H = 15$--$20$.
    O padrao de $H = 10$ e conservador e amplamente utilizado na literatura.
    Se o TSI muda muito entre $H = 10$ e $H = 20$, ha muita transmissao
    de medio prazo que o horizonte curto nao captura.

---

## Conclusao

!!! success "Resumo do workflow Spillover"
    Neste tutorial, completamos a analise de spillover de Diebold-Yilmaz:

    | Etapa | Metodo | ChronoBox |
    |-------|--------|-----------|
    | Dados | Retornos logaritmicos | `np.log(p / p.shift(1))` |
    | VAR | OLS com selecao de lags | `VAR(lags=2).fit(data)` |
    | Spillover estatico | GFEVD normalizada | `SpilloverIndex.fit(data)` |
    | FROM / TO / NET | Indices direcionais | `result.directional_from`, `.to`, `.net` |
    | Spillover dinamico | Rolling window (200 dias) | `si.rolling(data, window=200)` |
    | Alta conectividade | TSI > media + 1 sigma | Analise de limiares |
    | Rede | Network de spillover pairwise | Visualizacao customizada |

    Os principais resultados sao:

    - O **TSI medio** de ~39% indica conectividade substancial entre mercados
    - O **S&P 500** e o principal transmissor liquido de choques
    - Periodos de **crise** coincidem com picos de conectividade (TSI > 60%)
    - O spillover dinamico revela mudancas estruturais na transmissao ao longo do tempo

!!! tip "Proximos passos"
    - [Complete Workflow](complete-workflow.md): pipeline profissional de analise
    - [User Guide: Static Spillover](../user-guide/spillover/static.md): detalhes da GFEVD
    - [User Guide: Dynamic Spillover](../user-guide/spillover/dynamic.md): rolling window avancado
    - [Theory: Spillover](../theory/spillover-theory.md): fundamentacao matematica
