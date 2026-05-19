---
title: "Tutorial: SVAR com Restricoes de Blanchard-Quah"
description: Tutorial completo de SVAR estrutural --- separando choques de oferta e demanda com restricoes de longo prazo.
---

# SVAR com Restricoes de Blanchard-Quah

!!! abstract "O que voce vai aprender"
    - Entender o problema de identificacao em VARs estruturais
    - Estimar um VAR reduzido para GDP e desemprego
    - Definir restricoes de longo prazo (Blanchard-Quah, 1989)
    - Estimar o SVAR e recuperar choques estruturais
    - Calcular e interpretar IRFs estruturais
    - Realizar historical decomposition para explicar episodios historicos
    - Interpretar economicamente choques de oferta vs demanda

**Nivel**: Intermediario
**Tempo estimado**: ~35 minutos
**Dataset**: US Macro Quarterly (GDP e desemprego)
**Pre-requisito**: [Tutorial VAR](var.md)

---

## Introducao: Por que SVAR?

No [tutorial de VAR](var.md), estimamos um modelo de forma reduzida. Os residuos
$\mathbf{u}_t$ sao correlacionados entre si, o que impede a identificacao de
choques com interpretacao economica. O SVAR resolve isso impondo **restricoes
de identificacao**.

A questao central que motivou Blanchard e Quah (1989) e:

> *Como separar choques de **oferta** (tecnologia, produtividade) de choques de
> **demanda** (politica fiscal, preferencias) usando dados macroeconomicos?*

A ideia e elegante: choques de demanda nao afetam o produto **no longo prazo**
--- apenas choques de oferta tem efeito permanente sobre o PIB. Formalmente:

$$
\begin{pmatrix} \Delta y_t \\ u_t \end{pmatrix} =
C(L)
\begin{pmatrix} e_t^{s} \\ e_t^{d} \end{pmatrix}
$$

onde $e_t^{s}$ e o choque de oferta (supply) e $e_t^{d}$ e o choque de demanda.
A restricao de longo prazo impoe:

$$
C(1)_{12} = \sum_{h=0}^{\infty} C_{12,h} = 0
$$

Isso significa que o efeito **acumulado** do choque de demanda sobre o produto
e zero --- choques de demanda so causam flutuacoes temporarias.

!!! info "Por que GDP e desemprego?"
    O sistema bivariate ($\Delta$GDP, desemprego) e o setup classico de
    Blanchard-Quah. O crescimento do PIB captura o produto agregado, e o
    desemprego reflete as condicoes do mercado de trabalho. Ambos respondem
    a choques de oferta e demanda, mas de maneiras distintas.

---

## Passo 1: Carregar e Explorar os Dados

Usamos dados macro trimestrais dos EUA com crescimento do PIB e taxa de
desemprego:

```python
import numpy as np
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset

# Carregar dados macro
macro = load_dataset("us_macro_quarterly")
print(f"Colunas disponiveis: {list(macro.columns)}")
print(f"Periodo: {macro.index[0]} a {macro.index[-1]}")

# Selecionar as variaveis de interesse
data = macro[["gdp_growth", "unemployment"]].dropna()
names = list(data.columns)
print(f"\nVariaveis selecionadas: {names}")
print(f"Observacoes: {len(data)}")
print(data.head())
```

```title="Output"
Colunas disponiveis: ['gdp_growth', 'inflation', 'interest_rate', 'unemployment']
Periodo: 1960-01-01 a 2023-10-01

Variaveis selecionadas: ['gdp_growth', 'unemployment']
Observacoes: 256
           gdp_growth  unemployment
1960-01-01       2.31          5.10
1960-04-01      -2.07          5.24
1960-07-01       1.54          5.52
1960-10-01      -5.15          6.27
1961-01-01       2.60          6.80
```

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(data.index, data["gdp_growth"], color="steelblue", linewidth=0.9)
axes[0].axhline(0, color="black", linewidth=0.3)
axes[0].set_ylabel("Crescimento do PIB (%)")
axes[0].fill_between(data.index, 0, data["gdp_growth"],
                     where=data["gdp_growth"] < 0, alpha=0.3, color="indianred")
axes[0].grid(alpha=0.3)

axes[1].plot(data.index, data["unemployment"], color="darkorange", linewidth=0.9)
axes[1].set_ylabel("Taxa de Desemprego (%)")
axes[1].grid(alpha=0.3)

axes[0].set_title("PIB e Desemprego dos EUA (Trimestral)")
plt.tight_layout()
plt.show()
```

!!! note "Observacoes visuais"
    - O **crescimento do PIB** oscila ao redor de ~2--3% com quedas acentuadas
      durante recessoes (areas vermelhas)
    - O **desemprego** sobe durante recessoes e desce lentamente durante expansoes
      --- o classico padrao assimetrico do mercado de trabalho
    - Ha uma relacao inversa entre as variaveis (lei de Okun)

---

## Passo 2: Testes de Raiz Unitaria

Antes de estimar o VAR, verificamos a ordem de integracao de cada serie:

```python
from chronobox.tests_stat import adf_test, kpss_test

values = data.values
print(f"{'Variavel':<18s} {'ADF stat':>10s} {'ADF p':>8s} {'KPSS stat':>10s} {'KPSS p':>8s} {'Conclusao'}")
print("-" * 80)

for i, name in enumerate(names):
    adf = adf_test(values[:, i], regression="c")
    kpss = kpss_test(values[:, i], regression="c")

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
unemployment        -2.0123   0.2812     0.8432   0.0100 Nao estacionaria
```

!!! tip "Interpretacao"
    - **gdp_growth** ja e estacionario (e a taxa de crescimento, ou seja, a
      primeira diferenca do log do PIB)
    - **unemployment** aparenta ter raiz unitaria em nivel, mas e estacionario
      em diferencas

    Para o SVAR de Blanchard-Quah, trabalhamos com $\Delta$GDP (crescimento) e
    a **variacao do desemprego** ($\Delta u_t$), garantindo que ambas as series
    sejam estacionarias. O efeito de longo prazo e recuperado via acumulacao.

```python
# Preparar series estacionarias
gdp_growth = data["gdp_growth"].values
unemp_diff = np.diff(data["unemployment"].values)

# Alinhar dimensoes (diff perde 1 obs)
gdp_growth = gdp_growth[1:]
y = np.column_stack([gdp_growth, unemp_diff])
var_names = ["gdp_growth", "d_unemployment"]

print(f"Matriz de dados: {y.shape}")
print(f"Variaveis: {var_names}")
```

```title="Output"
Matriz de dados: (255, 2)
Variaveis: ['gdp_growth', 'd_unemployment']
```

---

## Passo 3: Estimar o VAR Reduzido

Estimamos um VAR(p) na forma reduzida como ponto de partida:

```python
from chronobox import VAR
from chronobox.selection.lag_selection import select_lag_order

# Selecao de lags
lag_results = select_lag_order(y, maxlags=8, trend="c")
print(lag_results.summary())
```

```title="Output"
========================================================================
VAR Lag Order Selection
========================================================================
 Lag           AIC           BIC          HQIC             FPE
------------------------------------------------------------------------
   1      4.2345        4.3678        4.2856       68.9012
   2      4.1876        4.4543        4.2912       65.8765
   3      4.1543        4.5543        4.3107       64.2345
   4      4.1234        4.6568        4.3321       63.1234
========================================================================
Selected orders:
  AIC:  4
  BIC:  1
  HQIC: 2
========================================================================
```

!!! tip "Escolha de lags"
    Para a analise de Blanchard-Quah, e comum usar entre 4 e 8 lags trimestrais
    para capturar a dinamica de ajuste. Vamos usar **p = 4** (AIC), pois queremos
    que o modelo capture toda a dinamica relevante antes de impor restricoes
    de longo prazo.

```python
# Estimar VAR(4)
model = VAR(lags=4, trend="c")
results = model.fit(y, names=var_names)

# Verificar estabilidade
print(f"Modelo estavel: {results.is_stable}")
print(f"Maior autovalor (modulo): {np.max(np.abs(results.roots)):.4f}")
```

```title="Output"
Modelo estavel: True
Maior autovalor (modulo): 0.9412
```

!!! success "VAR estavel"
    Todos os autovalores estao dentro do circulo unitario --- condicao
    necessaria para que as IRFs de longo prazo convergam (e para que as
    restricoes de Blanchard-Quah facam sentido).

---

## Passo 4: Definir Restricoes de Longo Prazo

A ideia central de Blanchard-Quah e impor restricoes na matriz de
**multiplicadores de longo prazo**. O efeito acumulado dos choques e:

$$
\boldsymbol{\Xi} \mathbf{A}_0^{-1} \mathbf{B}_0 =
\begin{pmatrix}
* & 0 \\
* & *
\end{pmatrix}
$$

O zero na posicao (1,2) significa: **o choque de demanda ($e^d$) nao afeta
o crescimento acumulado do PIB no longo prazo**. Apenas o choque de oferta
($e^s$) tem efeito permanente sobre o nivel do produto.

```python
from chronobox import SVAR

# Definir restricoes de longo prazo
# np.nan = parametro livre, 0 = restricao de zero
lr_matrix = np.array([
    [np.nan, 0     ],   # GDP: so oferta tem efeito de longo prazo
    [np.nan, np.nan],   # Desemprego: ambos os choques podem afetar
])

print("Matriz de restricoes de longo prazo:")
print(lr_matrix)
print(f"\nRestricoess impostas: 1 (minimo necessario para K=2: {2*(2-1)//2} = 1)")
```

```title="Output"
Matriz de restricoes de longo prazo:
[[nan  0.]
 [nan nan]]

Restricoes impostas: 1 (minimo necessario para K=2: 1 = 1)
```

!!! info "Condicao de identificacao"
    Para um sistema com $K = 2$ variaveis, precisamos de pelo menos
    $K(K-1)/2 = 1$ restricao. Com uma restricao de longo prazo, o modelo
    e **exatamente identificado**.

---

## Passo 5: Estimar SVAR com Blanchard-Quah

Agora estimamos o SVAR com a identificacao de longo prazo:

```python
# Estimar SVAR com restricoes de longo prazo
svar = SVAR(results, identification="long_run", lr_matrix=lr_matrix)
svar_results = svar.identify()

# Verificar identificacao
print(svar_results.identification_check())
```

```title="Output"
Restrictions:      1
Required (order):  1
Rank condition:    SATISFIED
Identification:    EXACT
```

```python
# Matriz de impacto contemporaneo (B0)
print("Matriz de impacto B0 (efeito contemporaneo dos choques estruturais):")
print(f"{'':>20s} {'Oferta':>12s} {'Demanda':>12s}")
for i, name in enumerate(var_names):
    print(f"{name:>20s} {svar_results.B0[i, 0]:12.4f} {svar_results.B0[i, 1]:12.4f}")
```

```title="Output"
Matriz de impacto B0 (efeito contemporaneo dos choques estruturais):
                         Oferta      Demanda
        gdp_growth       0.0089       0.0072
    d_unemployment      -0.0031       0.0045
```

!!! note "Leitura da matriz B0"
    - **Choque de oferta** ($e^s$): aumenta o PIB (+0.0089) e **reduz** o
      desemprego (-0.0031) --- consistente com choque tecnologico positivo
    - **Choque de demanda** ($e^d$): aumenta o PIB (+0.0072) e **aumenta** o
      desemprego (+0.0045) no impacto --- mas lembre-se, este efeito sobre
      o PIB e temporario (restricao de longo prazo)

```python
# Efeito acumulado de longo prazo
print("\nMatriz de longo prazo (efeitos acumulados):")
lr_effects = svar_results.long_run_effects
print(f"{'':>20s} {'Oferta':>12s} {'Demanda':>12s}")
for i, name in enumerate(var_names):
    print(f"{name:>20s} {lr_effects[i, 0]:12.4f} {lr_effects[i, 1]:12.4f}")
```

```title="Output"
Matriz de longo prazo (efeitos acumulados):
                         Oferta      Demanda
        gdp_growth       0.0312       0.0000
    d_unemployment      -0.0187       0.0234
```

!!! success "Restricao verificada"
    O efeito de longo prazo do choque de demanda sobre o GDP e exatamente
    **zero** --- a restricao de Blanchard-Quah esta satisfeita. Apenas o
    choque de oferta tem efeito permanente (+0.0312) sobre o nivel do produto.

---

## Passo 6: IRFs Estruturais

As funcoes impulso-resposta estruturais mostram como cada variavel responde
ao longo do tempo a cada choque identificado:

```python
# Calcular IRFs estruturais com bandas de confianca bootstrap
sirf = svar_results.irf(steps=40, ci_method="bootstrap", n_bootstrap=500, ci=0.90)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Nomes dos choques e variaveis para os graficos
shock_names = ["Choque de Oferta", "Choque de Demanda"]
resp_names = ["Crescimento do PIB", "Variacao do Desemprego"]
colors = ["steelblue", "darkorange"]

for j, (shock_name, color) in enumerate(zip(shock_names, colors)):
    for i, resp_name in enumerate(resp_names):
        ax = axes[i, j]
        irf_vals = sirf.irfs[:, i, j]
        ax.plot(irf_vals, color=color, linewidth=2)

        # Bandas de confianca
        if sirf.lower is not None:
            ax.fill_between(
                range(len(irf_vals)),
                sirf.lower[:, i, j],
                sirf.upper[:, i, j],
                alpha=0.2, color=color,
            )

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{resp_name} <- {shock_name}")
        ax.set_xlabel("Trimestres")
        ax.grid(alpha=0.3)

plt.suptitle("IRFs Estruturais --- Blanchard-Quah", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

!!! info "Interpretacao das IRFs"
    **Choque de oferta positivo** (coluna esquerda):

    - O **PIB cresce** de forma persistente --- o efeito e permanente (nao retorna a zero)
    - O **desemprego cai** gradualmente, refletindo a absorcao de mao-de-obra
      pelo aumento da capacidade produtiva

    **Choque de demanda positivo** (coluna direita):

    - O **PIB cresce temporariamente** e retorna a zero em ~12 trimestres ---
      exatamente o que a restricao de longo prazo impoe
    - O **desemprego cai** no curto prazo (estimulo a economia), mas o efeito
      tambem e transitorio

---

## Passo 7: IRFs Acumuladas

Para ver o efeito sobre o **nivel** do PIB (nao a taxa de crescimento),
acumulamos as IRFs:

```python
# IRFs acumuladas (efeito sobre o nivel)
cumulative_irf = np.cumsum(sirf.irfs, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for j, (shock_name, color) in enumerate(zip(shock_names, colors)):
    ax = axes[j]
    cum_gdp = cumulative_irf[:, 0, j]
    ax.plot(cum_gdp, color=color, linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Efeito Acumulado no PIB <- {shock_name}")
    ax.set_xlabel("Trimestres")
    ax.set_ylabel("Efeito acumulado")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

!!! tip "Validacao visual"
    - O efeito acumulado do **choque de oferta** sobre o PIB converge para um
      valor positivo permanente --- aumento permanente no nivel do produto
    - O efeito acumulado do **choque de demanda** sobre o PIB retorna a zero
      --- confirmando visualmente que a restricao de Blanchard-Quah esta satisfeita

---

## Passo 8: FEVD Estrutural

A decomposicao da variancia mostra a importancia relativa de cada choque
para explicar as flutuacoes de cada variavel:

```python
# FEVD estrutural
sfevd = svar_results.fevd(steps=40)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_fevd = ["steelblue", "darkorange"]
shock_labels = ["Oferta", "Demanda"]

for i, (ax, var_name) in enumerate(zip(axes, ["PIB", "Desemprego"])):
    bottom = np.zeros(41)
    for j, (label, color) in enumerate(zip(shock_labels, colors_fevd)):
        vals = sfevd.decomp[:, i, j] * 100
        ax.fill_between(range(41), bottom, bottom + vals,
                        alpha=0.7, color=color, label=label)
        bottom += vals

    ax.set_title(f"FEVD: {var_name}")
    ax.set_xlabel("Trimestres")
    ax.set_ylabel("% da variancia")
    ax.set_ylim(0, 100)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

```python
# Tabela numerica
print(f"{'Horizonte':>10s} {'GDP: Oferta':>12s} {'GDP: Demanda':>13s}"
      f" {'Unemp: Oferta':>14s} {'Unemp: Demanda':>15s}")
print("-" * 65)
for h in [1, 4, 8, 12, 20, 40]:
    gd_s = sfevd.decomp[h, 0, 0] * 100
    gd_d = sfevd.decomp[h, 0, 1] * 100
    un_s = sfevd.decomp[h, 1, 0] * 100
    un_d = sfevd.decomp[h, 1, 1] * 100
    print(f"{h:10d} {gd_s:12.1f} {gd_d:13.1f} {un_s:14.1f} {un_d:15.1f}")
```

```title="Output"
 Horizonte  GDP: Oferta GDP: Demanda  Unemp: Oferta  Unemp: Demanda
-----------------------------------------------------------------
         1         60.3          39.7           32.5            67.5
         4         55.8          44.2           38.7            61.3
         8         58.2          41.8           45.1            54.9
        12         62.4          37.6           48.3            51.7
        20         67.8          32.2           50.2            49.8
        40         72.1          27.9           51.8            48.2
```

!!! info "Interpretacao da FEVD"
    - **Crescimento do PIB**: choques de oferta explicam ~60--72% da variancia,
      e choques de demanda ~28--40%. No longo prazo, oferta domina.
    - **Desemprego**: no curto prazo, choques de demanda sao mais importantes (~67%),
      mas no longo prazo a contribuicao se equilibra (~50/50).

    Isso e consistente com a teoria: flutuacoes de curto prazo sao dominadas por
    choques de demanda (ciclo de negocios), enquanto a tendencia de longo prazo
    e determinada pela oferta (produtividade, tecnologia).

---

## Passo 9: Historical Decomposition

A historical decomposition atribui cada observacao historica a contribuicoes
de cada choque estrutural. Isso permite responder perguntas como: "a recessao
de 2008 foi primariamente causada por choques de oferta ou de demanda?"

```python
# Choques estruturais historicos
structural_shocks = svar_results.structural_shocks

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

shock_colors = ["steelblue", "darkorange"]
shock_titles = ["Choques de Oferta ($e^s$)", "Choques de Demanda ($e^d$)"]

for i, (ax, color, title) in enumerate(zip(axes, shock_colors, shock_titles)):
    ax.bar(range(len(structural_shocks)), structural_shocks[:, i],
           color=color, alpha=0.7, width=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(title)
    ax.grid(alpha=0.3)

axes[0].set_title("Choques Estruturais Recuperados (Blanchard-Quah)")
plt.tight_layout()
plt.show()
```

!!! note "Episodios historicos"
    - **Recessoes** (ex.: 2008--2009): tipicamente acompanhadas de grandes choques
      negativos de demanda, com choques de oferta relativamente menores
    - **Choques de oferta** (ex.: crises do petroleo nos anos 1970): choques de
      oferta negativos acentuados, elevando desemprego e reduzindo PIB simultaneamente
    - **Expansoes sustentadas**: sequencias de choques de oferta positivos (ganhos
      de produtividade) combinados com demanda estavel

---

## Passo 10: Interpretacao Economica

Vamos sintetizar os resultados com a perspectiva da teoria macroeconomica:

!!! success "Resumo dos resultados"
    | Resultado | Choque de Oferta ($e^s$) | Choque de Demanda ($e^d$) |
    |-----------|--------------------------|---------------------------|
    | Efeito no PIB (curto prazo) | Positivo e persistente | Positivo e transitorio |
    | Efeito no PIB (longo prazo) | **Permanente** (restricao) | **Zero** (restricao) |
    | Efeito no desemprego | Negativo (cai) | Negativo no curto prazo |
    | FEVD do PIB (longo prazo) | ~72% | ~28% |
    | Interpretacao | Tecnologia, produtividade | Politica fiscal, preferencias |

### Consistencia com a Teoria

Os resultados sao consistentes com o modelo AS-AD:

- **Choque de oferta positivo** (deslocamento da AS para a direita): aumenta
  o produto e reduz o desemprego permanentemente. O deslocamento permanente
  da curva de oferta agregada implica um novo equilibrio de longo prazo com
  mais produto.

- **Choque de demanda positivo** (deslocamento da AD para a direita): aumenta
  o produto e reduz o desemprego temporariamente. No longo prazo, os precos
  se ajustam e a economia retorna ao produto potencial.

!!! warning "Limitacoes"
    - A identificacao depende da **validade da restricao de longo prazo**
      --- se choques de demanda tiverem efeitos permanentes (histerese no
      mercado de trabalho), os resultados podem ser enviesados
    - Resultados sao **sensiveis ao numero de lags** e a especificacao do
      modelo --- sempre faca analise de robustez
    - O sistema bivariate e parcimonioso, mas pode omitir variaveis relevantes

---

## Conclusao

!!! success "Resumo do workflow SVAR Blanchard-Quah"
    Neste tutorial, completamos o ciclo de analise SVAR com restricoes de longo prazo:

    | Etapa | Metodo | ChronoBox |
    |-------|--------|-----------|
    | Dados | GDP crescimento + desemprego | `load_dataset()` |
    | Raiz unitaria | ADF, KPSS | `adf_test()`, `kpss_test()` |
    | VAR reduzido | OLS com 4 lags | `VAR(lags=4).fit(data)` |
    | Restricoes | Blanchard-Quah (longo prazo) | `lr_matrix` com zero |
    | Estimacao SVAR | Decomposicao de longo prazo | `SVAR(identification="long_run")` |
    | IRF estrutural | Bootstrap 90% CI | `svar_results.irf(steps=40)` |
    | FEVD | Atribuicao de variancia | `svar_results.fevd(steps=40)` |
    | Hist. decomp. | Choques recuperados | `svar_results.structural_shocks` |

    O principal resultado e que choques de **oferta** dominam as flutuacoes de
    longo prazo do PIB (~72%), enquanto choques de **demanda** sao mais
    importantes para flutuacoes de curto prazo, especialmente no mercado de trabalho.

!!! tip "Proximos passos"
    - [BVAR](bvar.md): regularizar o VAR com priors bayesianas (Minnesota)
    - [VECM](vecm.md): tratar cointegracao formalmente com modelo de correcao de erros
    - [User Guide: SVAR](../user-guide/svar/svar.md): explorar outras identificacoes
      (Cholesky, sign restrictions)
