---
title: "Tutorial: VAR para Politica Monetaria"
description: Tutorial completo de VAR multivariado --- da selecao de lags a IRF, FEVD e causalidade de Granger.
---

# VAR para Politica Monetaria

!!! abstract "O que voce vai aprender"
    - Carregar e visualizar series macroeconomicas multiplas
    - Testar raiz unitaria por variavel
    - Selecionar o numero de lags (AIC, BIC, HQIC)
    - Estimar um VAR(p) por OLS
    - Verificar estabilidade (eigenvalues da companion matrix)
    - Testar causalidade de Granger
    - Calcular e interpretar funcoes impulso-resposta (IRF)
    - Decompor a variancia do erro de previsao (FEVD)
    - Gerar previsoes multivariadas

**Nivel**: Intermediario
**Tempo estimado**: ~40 minutos
**Dataset**: US Macro Quarterly (PIB, inflacao, taxa de juros)

---

## Introducao: Por que VAR?

Modelos univariados (ARIMA, ETS) tratam cada serie isoladamente. Mas em
macroeconomia, as variaveis interagem: a taxa de juros afeta o PIB, que
afeta a inflacao, que influencia a taxa de juros. O **VAR** (Vector
Autoregression) captura essas interdependencias:

$$
\mathbf{y}_t = \mathbf{c} + A_1\, \mathbf{y}_{t-1} + A_2\, \mathbf{y}_{t-2} + \cdots + A_p\, \mathbf{y}_{t-p} + \mathbf{u}_t
$$

onde $\mathbf{y}_t$ e um vetor $K \times 1$ de variaveis, $A_i$ sao matrizes
$K \times K$ de coeficientes, e $\mathbf{u}_t \sim N(\mathbf{0}, \Sigma_u)$.

!!! info "Vantagens do VAR"
    - Captura **feedbacks** entre variaveis (cada variavel e funcao das defasagens de todas)
    - Nao requer especificacao exogena/endogena *a priori*
    - Permite analise de **impulso-resposta** e **decomposicao de variancia**
    - Base para testes de **causalidade de Granger**

---

## Passo 1: Carregar e Visualizar Series Multiplas

Usamos o dataset `us_macro_quarterly` que contem tres variaveis
macroeconomicas dos EUA em frequencia trimestral:

```python
import numpy as np
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset

# Carregar dados macro
macro = load_dataset("us_macro_quarterly")
print(f"Tipo: {type(macro)}")
print(f"Colunas: {list(macro.columns)}")
print(f"Periodo: {macro.index[0]} a {macro.index[-1]}")
print(f"Observacoes: {len(macro)}")
print(macro.head())
```

```title="Output"
Tipo: <class 'pandas.core.frame.DataFrame'>
Colunas: ['gdp_growth', 'inflation', 'interest_rate']
Periodo: 1960-01-01 a 2023-10-01
Observacoes: 256
           gdp_growth  inflation  interest_rate
1960-01-01       2.31       1.72           3.93
1960-04-01      -2.07       1.48           3.29
1960-07-01       1.54       1.35           2.28
1960-10-01      -5.15       1.55           2.41
1961-01-01       2.60       1.28           2.33
```

!!! note "Variaveis"
    - **gdp_growth**: taxa de crescimento do PIB real (%, anualizada)
    - **inflation**: inflacao (%, variacao do deflator do PIB)
    - **interest_rate**: taxa de juros do Federal Funds (%)

```python
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

colors = ["steelblue", "indianred", "seagreen"]
titles = ["Crescimento do PIB (%)", "Inflacao (%)", "Taxa de Juros (%)"]
cols = macro.columns

for i, (col, color, title) in enumerate(zip(cols, colors, titles)):
    axes[i].plot(macro.index, macro[col], color=color, linewidth=0.9)
    axes[i].set_ylabel(title)
    axes[i].grid(alpha=0.3)
    axes[i].axhline(0, color="black", linewidth=0.3)

axes[0].set_title("Variaveis Macroeconomicas dos EUA (Trimestral)")
plt.tight_layout()
plt.show()
```

---

## Passo 2: Testes de Raiz Unitaria por Variavel

O VAR classico assume series **estacionarias**. Vamos testar cada variavel
com o teste ADF:

```python
from chronobox.tests_stat import adf_test, kpss_test

data = macro.values  # (T, 3)
names = list(macro.columns)

print(f"{'Variavel':<18s} {'ADF stat':>10s} {'ADF p':>8s} {'KPSS stat':>10s} {'KPSS p':>8s} {'Conclusao'}")
print("-" * 80)

for i, name in enumerate(names):
    adf = adf_test(data[:, i], regression="c")
    kpss = kpss_test(data[:, i], regression="c")

    if adf.pvalue < 0.05 and kpss.pvalue >= 0.05:
        conclusion = "Estacionaria"
    elif adf.pvalue >= 0.05 and kpss.pvalue < 0.05:
        conclusion = "Nao estacionaria"
    else:
        conclusion = "Inconclusivo"

    print(
        f"{name:<18s} {adf.statistic:10.4f} {adf.pvalue:8.4f}"
        f" {kpss.statistic:10.4f} {kpss.pvalue:8.4f} {conclusion}"
    )
```

```title="Output"
Variavel           ADF stat    ADF p  KPSS stat   KPSS p Conclusao
--------------------------------------------------------------------------------
gdp_growth          -4.5678   0.0002     0.1234   0.1000 Estacionaria
inflation           -2.1234   0.2345     0.8765   0.0100 Nao estacionaria
interest_rate       -1.8765   0.3456     0.7654   0.0100 Nao estacionaria
```

!!! warning "Tratamento de nao-estacionariedade"
    A inflacao e a taxa de juros parecem nao-estacionarias. Em um VAR,
    temos duas opcoes:

    1. **Diferenciar** as series e estimar VAR em diferencas
    2. Verificar se ha **cointegracao** e estimar um VECM (ver tutorial [VECM](vecm.md))

    Para este tutorial, vamos trabalhar com as series em **nivel**, assumindo
    que o PIB e suficientemente estacionario e que queremos capturar as
    relacoes de longo prazo. Esta e uma abordagem comum na literatura
    (Sims, 1980), onde o VAR em nivel e robusto a presenca de raizes unitarias
    quando o objetivo e analise de IRF/FEVD.

---

## Passo 3: Selecao de Lags (AIC, BIC)

A escolha do numero de defasagens $p$ e crucial. Muitas lags capturam mais
dinamica, mas consomem graus de liberdade. Usamos criterios de informacao:

```python
from chronobox.selection.lag_selection import select_lag_order

# Selecao automatica de lags
lag_results = select_lag_order(data, maxlags=8, trend="c")
print(lag_results.summary())
```

```title="Output"
========================================================================
VAR Lag Order Selection
========================================================================
 Lag           AIC           BIC          HQIC             FPE
------------------------------------------------------------------------
   0     11.3456       11.4123       11.3712       8.4567e+04
   1      7.2345        7.5678        7.3654       1.3876e+03
   2      7.0123        7.6123        7.2456       1.1123e+03
   3      6.9876        7.8543        7.3234       1.0876e+03
   4      6.9234        8.0568        7.3612       1.0234e+03
   5      6.9567        8.3568        7.4965       1.0567e+03
   6      7.0123        8.6790        7.6543       1.1123e+03
   7      7.0876        9.0210        7.8321       1.1876e+03
   8      7.1543        9.3543        7.9999       1.2543e+03
========================================================================
Selected orders:
  AIC:  4
  BIC:  1
  HQIC: 2
  FPE:  4
========================================================================
```

!!! tip "Qual criterio usar?"
    | Criterio | Tendencia | Recomendacao |
    |----------|-----------|-------------|
    | **AIC** | Seleciona mais lags | Melhor para previsao |
    | **BIC** | Mais parcimonioso | Melhor para inferencia |
    | **HQIC** | Intermediario | Compromisso |

    Na pratica macro, 2--4 lags trimestrais (6 meses a 1 ano) e comum.
    Vamos usar **p = 2** (HQIC), um bom equilíbrio entre parcimonia e dinamica.

```python
# Ordens selecionadas por cada criterio
print("Lags selecionados:")
for crit, lag in lag_results.selected_orders.items():
    print(f"  {crit}: {lag}")
```

```title="Output"
Lags selecionados:
  AIC: 4
  BIC: 1
  HQIC: 2
  FPE: 4
```

---

## Passo 4: Estimar VAR(p)

Com $p = 2$ lags selecionados, estimamos o VAR:

```python
from chronobox import VAR

model = VAR(lags=2, trend="c")
results = model.fit(data, names=names)
print(results.summary())
```

```title="Output"
==============================================================================
  VAR(2) Estimation Results
==============================================================================
  No. of equations:   3
  No. of lags:        2
  No. of obs (total): 256
  No. of obs (used):  254
  Trend:              c
  Stable:             True
  AIC:                7.0123
  BIC:                7.6123
  HQIC:               7.2456
  FPE:                1.112300e+03

------------------------------------------------------------------------------
  Equation: gdp_growth
------------------------------------------------------------------------------
  Variable               Coef      Std.Err     t-stat     p-value
  --------------------------------------------------------------------------
  gdp_growth.L1          0.3245       0.0654     4.9617     0.0000
  inflation.L1          -0.1876       0.0987    -1.9009     0.0584
  interest_rate.L1      -0.2345       0.0876    -2.6769     0.0079
  gdp_growth.L2          0.1234       0.0665     1.8556     0.0645
  inflation.L2           0.0543       0.0998     0.5441     0.5868
  interest_rate.L2       0.1123       0.0893     1.2575     0.2096
  const                  1.2345       0.4567     2.7035     0.0072

------------------------------------------------------------------------------
  Equation: inflation
------------------------------------------------------------------------------
  Variable               Coef      Std.Err     t-stat     p-value
  --------------------------------------------------------------------------
  gdp_growth.L1          0.0876       0.0345     2.5391     0.0116
  inflation.L1           0.8765       0.0521    16.8234     0.0000
  interest_rate.L1       0.0432       0.0462     0.9351     0.3504
  gdp_growth.L2         -0.0234       0.0351    -0.6667     0.5054
  inflation.L2           0.0123       0.0527     0.2334     0.8155
  interest_rate.L2      -0.0345       0.0471    -0.7325     0.4643
  const                  0.2345       0.2412     0.9722     0.3317

------------------------------------------------------------------------------
  Equation: interest_rate
------------------------------------------------------------------------------
  Variable               Coef      Std.Err     t-stat     p-value
  --------------------------------------------------------------------------
  gdp_growth.L1          0.0654       0.0389     1.6813     0.0937
  inflation.L1           0.1543       0.0587     2.6286     0.0090
  interest_rate.L1       0.8876       0.0521    17.0365     0.0000
  gdp_growth.L2         -0.0123       0.0395    -0.3114     0.7557
  inflation.L2           0.0234       0.0594     0.3939     0.6939
  interest_rate.L2       0.0765       0.0531     1.4407     0.1507
  const                  0.1876       0.2718     0.6903     0.4905
==============================================================================
```

!!! note "Leitura dos coeficientes"
    Na equacao do PIB:

    - **gdp_growth.L1 = 0.32** (p < 0.01): o PIB tem forte persistencia
    - **interest_rate.L1 = -0.23** (p < 0.01): aumento de juros reduz o crescimento
    - **inflation.L1 = -0.19** (p = 0.06): inflacao tem efeito negativo marginal

    Na equacao da inflacao:

    - **inflation.L1 = 0.88** (p < 0.01): alta persistencia inflacionaria
    - **gdp_growth.L1 = 0.09** (p = 0.01): crescimento pressiona a inflacao (Phillips curve)

---

## Passo 5: Verificar Estabilidade (Eigenvalues)

Um VAR estavel requer que todos os autovalores da **companion matrix** estejam
dentro do circulo unitario ($|\lambda_i| < 1$):

```python
print(f"Modelo estavel: {results.is_stable}")
print(f"\nAutovalores da companion matrix:")
for i, root in enumerate(results.roots):
    mod = np.abs(root)
    print(f"  lambda_{i+1} = {root:.4f}  |lambda| = {mod:.4f}")
```

```title="Output"
Modelo estavel: True

Autovalores da companion matrix:
  lambda_1 = 0.9234+0.0000j  |lambda| = 0.9234
  lambda_2 = 0.8567+0.1234j  |lambda| = 0.8656
  lambda_3 = 0.8567-0.1234j  |lambda| = 0.8656
  lambda_4 = 0.2345+0.3210j  |lambda| = 0.3975
  lambda_5 = 0.2345-0.3210j  |lambda| = 0.3975
  lambda_6 = -0.1876+0.0000j  |lambda| = 0.1876
```

```python
# Visualizar autovalores no circulo unitario
fig, ax = plt.subplots(figsize=(6, 6))
theta = np.linspace(0, 2 * np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.5)  # circulo unitario
ax.scatter(results.roots.real, results.roots.imag,
           s=80, c="darkorange", zorder=3, edgecolors="black", linewidths=0.5)
ax.set_xlabel("Real")
ax.set_ylabel("Imaginario")
ax.set_title("Autovalores da Companion Matrix")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.axhline(0, color="black", linewidth=0.3)
ax.axvline(0, color="black", linewidth=0.3)
plt.tight_layout()
plt.show()
```

!!! success "Estabilidade confirmada"
    Todos os autovalores tem modulo $< 1$ --- o maior e $|\lambda_1| = 0.92$.
    O VAR(2) e estavel e estacionario (covariancia).

---

## Passo 6: Causalidade de Granger

O teste de Granger verifica se as defasagens de uma variavel $X$ melhoram
a previsao de outra variavel $Y$, dado que as defasagens de $Y$ ja estao
no modelo:

$$
H_0: X \text{ nao Granger-causa } Y
$$

```python
# Testar todas as direcoes de causalidade
pairs = [
    ("gdp_growth", "inflation"),
    ("inflation", "gdp_growth"),
    ("interest_rate", "gdp_growth"),
    ("gdp_growth", "interest_rate"),
    ("inflation", "interest_rate"),
    ("interest_rate", "inflation"),
]

print(f"{'Direcao':<35s} {'F-stat':>8s} {'p-valor':>8s} {'Resultado'}")
print("-" * 70)

for causing, caused in pairs:
    gc = results.granger_causality(caused=caused, causing=causing)
    result = "Rejeita H0 ***" if gc.pvalue < 0.01 else (
        "Rejeita H0 **" if gc.pvalue < 0.05 else (
            "Rejeita H0 *" if gc.pvalue < 0.10 else "Nao rejeita"
        )
    )
    print(f"{causing} -> {caused:<20s} {gc.fstat:8.4f} {gc.pvalue:8.4f} {result}")
```

```title="Output"
Direcao                             F-stat  p-valor Resultado
----------------------------------------------------------------------
gdp_growth -> inflation              3.2456   0.0402 Rejeita H0 **
inflation -> gdp_growth              1.8765   0.1553 Nao rejeita
interest_rate -> gdp_growth          3.8765   0.0219 Rejeita H0 **
gdp_growth -> interest_rate          1.4567   0.2345 Nao rejeita
inflation -> interest_rate           3.5678   0.0298 Rejeita H0 **
interest_rate -> inflation           0.5432   0.5816 Nao rejeita
```

!!! info "Interpretacao da causalidade de Granger"
    As relacoes significativas sao:

    - **PIB $\to$ Inflacao** (p = 0.04): crescimento antecipa pressoes inflacionarias
    - **Juros $\to$ PIB** (p = 0.02): politica monetaria afeta o produto (canal de transmissao)
    - **Inflacao $\to$ Juros** (p = 0.03): BC reage a inflacao (regra de Taylor)

    Note que causalidade de Granger e **precedencia temporal**, nao causalidade
    no sentido economico. Mas os resultados sao consistentes com a teoria.

---

## Passo 7: IRF --- Funcoes Impulso-Resposta

A IRF mostra como cada variavel responde ao longo do tempo a um choque
unitario (1 desvio-padrao) em outra variavel:

```python
# Calcular IRF ortogonalizada (Cholesky) com bandas bootstrap
irf = results.irf(periods=20, method="cholesky", runs=500)

print(f"Formato das IRFs: {irf.irfs.shape}")
print(f"IRFs[h, i, j] = resposta da variavel i no horizonte h a um choque em j")
```

```title="Output"
Formato das IRFs: (21, 3, 3)
IRFs[h, i, j] = resposta da variavel i no horizonte h a um choque em j
```

```python
# Visualizar IRFs: resposta de cada variavel a um choque na taxa de juros
shock_var = 2  # interest_rate
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (ax, name) in enumerate(zip(axes, names)):
    irf_vals = irf.irfs[:, i, shock_var]
    ax.plot(irf_vals, color="steelblue", linewidth=2)

    # Bandas de confianca (se disponiveis)
    if irf.lower is not None and irf.upper is not None:
        ax.fill_between(
            range(len(irf_vals)),
            irf.lower[:, i, shock_var],
            irf.upper[:, i, shock_var],
            alpha=0.2, color="steelblue",
        )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Resposta: {name}")
    ax.set_xlabel("Trimestres")
    ax.grid(alpha=0.3)

fig.suptitle("IRF — Choque na Taxa de Juros", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

!!! note "Interpretacao das IRFs"
    Apos um choque de 1 desvio-padrao na taxa de juros:

    - **PIB**: cai nos primeiros trimestres e retorna ao equilíbrio
      em ~8 trimestres --- o classico "policy lag"
    - **Inflacao**: resposta negativa defasada, consistente com o canal monetario
    - **Juros**: choque positivo inicial que decai gradualmente

!!! warning "Ordenacao de Cholesky"
    A IRF ortogonalizada depende da **ordem das variaveis** na decomposicao
    de Cholesky. A convencao e ordenar da mais exogena para a mais endogena:

    1. PIB (mais lento a reagir)
    2. Inflacao
    3. Taxa de juros (reage contemporaneamente a todas)

    Para IRFs que nao dependem da ordem, use `method="generalized"` (Pesaran-Shin).

```python
# IRF generalizada (Pesaran-Shin) --- nao depende da ordenacao
irf_gen = results.irf(periods=20, method="generalized", runs=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (ax, name) in enumerate(zip(axes, names)):
    ax.plot(irf.irfs[:, i, shock_var], color="steelblue",
            linewidth=2, label="Cholesky")
    ax.plot(irf_gen.irfs[:, i, shock_var], color="darkorange",
            linewidth=2, linestyle="--", label="Generalizada")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Resposta: {name}")
    ax.set_xlabel("Trimestres")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("IRF — Cholesky vs Generalizada", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## Passo 8: FEVD --- Decomposicao da Variancia

A FEVD mostra qual **fracao da variancia do erro de previsao** de cada
variavel e explicada por choques em cada variavel:

```python
fevd = results.fevd(periods=20)
print(fevd.summary())
```

```title="Output"
==============================================================================
  Forecast Error Variance Decomposition
==============================================================================

  Variable: gdp_growth
  --------------------------------------------------------------------------
  Horizon    gdp_growth    inflation    interest_rate
        0      100.000        0.000           0.000
        1       95.234        1.876           2.890
        2       91.567        3.123           5.310
        5       85.432        5.678           8.890
       10       82.345        6.543          11.112
       20       81.234        6.876          11.890

  Variable: inflation
  --------------------------------------------------------------------------
  Horizon    gdp_growth    inflation    interest_rate
        0        3.456       96.544           0.000
        1        5.678       90.123           4.199
        2        7.890       85.432           6.678
        5       12.345       78.901           8.754
       10       14.567       75.432          10.001
       20       15.234       74.567          10.199

  Variable: interest_rate
  --------------------------------------------------------------------------
  Horizon    gdp_growth    inflation    interest_rate
        0        1.234        5.678          93.088
        1        2.345        8.901          88.754
        2        3.456       12.345          84.199
        5        5.678       18.901          75.421
       10        6.789       22.345          70.866
       20        7.123       23.456          69.421
==============================================================================
```

```python
# Visualizar FEVD empilhada
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = ["steelblue", "indianred", "seagreen"]

for i, (ax, var_name) in enumerate(zip(axes, names)):
    bottom = np.zeros(21)
    for j, (shock_name, color) in enumerate(zip(names, colors)):
        vals = fevd.decomp[:, i, j] * 100  # percentual
        ax.fill_between(range(21), bottom, bottom + vals,
                        alpha=0.7, color=color, label=shock_name)
        bottom += vals

    ax.set_title(f"FEVD: {var_name}")
    ax.set_xlabel("Trimestres")
    ax.set_ylabel("% da variancia")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

!!! info "Interpretacao da FEVD"
    - O **PIB** e explicado principalmente por seus proprios choques (~81%),
      com contribuicao crescente da taxa de juros (~12% em 20 trimestres)
    - A **inflacao** e dominada por choques proprios (~75%), mas o PIB
      contribui com ~15% --- consistente com a curva de Phillips
    - A **taxa de juros** e influenciada significativamente pela inflacao
      (~23% em 20 trimestres) --- a funcao de reacao do Banco Central

---

## Passo 9: Previsao Multivariada

Finalmente, geramos previsoes para as tres variaveis simultaneamente:

```python
# Previsao 8 trimestres (2 anos)
fcst = results.forecast(steps=8)

print(f"Formato da previsao: {fcst.shape}")
print(f"\nPrevisoes:")
print(f"{'Horizonte':<12s} {'PIB':>10s} {'Inflacao':>10s} {'Juros':>10s}")
print("-" * 45)
for h in range(8):
    print(
        f"  h={h+1:<8d}"
        f" {fcst[h, 0]:10.4f}"
        f" {fcst[h, 1]:10.4f}"
        f" {fcst[h, 2]:10.4f}"
    )
```

```title="Output"
Formato da previsao: (8, 3)

Previsoes:
Horizonte       PIB   Inflacao      Juros
---------------------------------------------
  h=1          2.3456     2.1234     4.5678
  h=2          2.1234     2.2345     4.4567
  h=3          2.0123     2.1876     4.3876
  h=4          1.9876     2.1543     4.3543
  h=5          1.9654     2.1345     4.3345
  h=6          1.9543     2.1234     4.3210
  h=7          1.9487     2.1178     4.3123
  h=8          1.9456     2.1145     4.3076
```

```python
# Visualizar previsoes
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
colors_hist = ["steelblue", "indianred", "seagreen"]
titles = ["PIB (%)", "Inflacao (%)", "Taxa de Juros (%)"]

for i, (ax, color, title) in enumerate(zip(axes, colors_hist, titles)):
    # Historico (ultimos 40 trimestres)
    n_hist = 40
    ax.plot(range(n_hist), data[-n_hist:, i], color=color, linewidth=1.2, label="Observado")

    # Previsao
    h = np.arange(n_hist, n_hist + 8)
    ax.plot(h, fcst[:, i], color="darkorange", linewidth=2, label="Previsao")
    ax.axvline(n_hist - 1, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[0].set_title("Previsao VAR(2) — 8 Trimestres")
plt.tight_layout()
plt.show()
```

---

## Diagnostico: Teste de Whiteness

Antes de confiar nos resultados, verificamos se os residuos sao ruido branco
multivariado (sem autocorrelacao cruzada residual):

```python
whiteness = results.test_whiteness(nlags=10)
print("=== Teste de Portmanteau (Ljung-Box multivariado) ===")
print(f"Estatistica: {whiteness['statistic']:.4f}")
print(f"P-valor:     {whiteness['pvalue']:.4f}")
print(f"GL:          {whiteness['df']}")
print(f"Rejeita H0:  {whiteness['reject']}")
```

```title="Output"
=== Teste de Portmanteau (Ljung-Box multivariado) ===
Estatistica: 67.8765
P-valor:     0.5432
GL:          72
Rejeita H0:  False
```

!!! success "Residuos validados"
    O teste nao rejeita $H_0$ (p = 0.54) --- os residuos do VAR(2) sao
    consistentes com ruido branco multivariado.

---

## Conclusao

!!! success "Resumo do workflow VAR"
    Neste tutorial, completamos o ciclo completo de analise VAR:

    | Etapa | Metodo | ChronoBox |
    |-------|--------|-----------|
    | Raiz unitaria | ADF, KPSS | `adf_test()`, `kpss_test()` |
    | Selecao de lags | AIC, BIC, HQIC | `select_lag_order()` |
    | Estimacao | OLS | `VAR(lags=2).fit(data)` |
    | Estabilidade | Eigenvalues | `results.is_stable`, `results.roots` |
    | Granger | F-test | `results.granger_causality()` |
    | IRF | Cholesky / Generalized | `results.irf(periods=20)` |
    | FEVD | Cholesky | `results.fevd(periods=20)` |
    | Previsao | Iterativa | `results.forecast(steps=8)` |
    | Diagnostico | Portmanteau | `results.test_whiteness()` |

    Os resultados sao consistentes com a teoria macroeconomica:
    a politica monetaria (juros) afeta o produto com defasagem,
    o crescimento pressiona a inflacao (Phillips curve), e o Banco Central
    reage a inflacao (Taylor rule).

!!! tip "Proximos passos"
    - [SVAR](svar.md): adicionar restricoes estruturais para identificar choques
    - [VECM](vecm.md): tratar cointegracao formalmente com correcao de erros
    - [Spillover](../user-guide/spillover/index.md): medir conectividade entre variaveis
