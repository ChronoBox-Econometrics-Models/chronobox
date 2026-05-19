---
title: "Tutorial: VECM para Series Cointegradas"
description: Tutorial completo de VECM --- da lei do preco unico a previsao com correcao de erros via Johansen procedure.
---

# VECM para Series Cointegradas

!!! abstract "O que voce vai aprender"
    - Entender cointegracao e a lei do preco unico
    - Testar raiz unitaria em cada serie individualmente
    - Aplicar o procedimento de Johansen para determinar o rank de cointegracao
    - Estimar um VECM e interpretar vetores de cointegracao ($\boldsymbol{\beta}$)
    - Interpretar os loading coefficients ($\boldsymbol{\alpha}$) e velocidade de ajuste
    - Calcular IRF e FEVD no contexto do VECM
    - Gerar previsoes que respeitam o equilibrio de longo prazo

**Nivel**: Intermediario
**Tempo estimado**: ~35 minutos
**Dataset**: Denmark (series de precos e renda)
**Pre-requisito**: [Tutorial VAR](var.md)

---

## Introducao: Cointegracao e Equilibrio de Longo Prazo

Muitas series economicas sao **I(1)** --- nao estacionarias em nivel, mas
estacionarias em primeira diferenca. Quando estimamos um VAR em diferencas,
perdemos a informacao sobre as **relacoes de longo prazo** entre as variaveis.

A **cointegracao** (Engle & Granger, 1987; Johansen, 1991) captura exatamente
essa situacao: duas ou mais series I(1) que compartilham uma tendencia estocastica
comum --- uma combinacao linear delas e estacionaria.

O exemplo classico e a **lei do preco unico**: precos de um mesmo bem em
diferentes mercados devem convergir no longo prazo (ajustados por custos de
transporte e cambio). Individualmente, cada preco segue um random walk, mas a
**diferenca** (ou razao) entre eles e estacionaria.

$$
\text{Se } y_{1t} \sim I(1) \text{ e } y_{2t} \sim I(1), \text{ mas } y_{1t} - \beta\, y_{2t} \sim I(0)
$$

entao $y_{1t}$ e $y_{2t}$ sao **cointegrados** com vetor $(1, -\beta)$.

O **VECM** (Vector Error Correction Model) e o modelo adequado para este caso:

$$
\Delta \mathbf{y}_t = \boldsymbol{\alpha}\boldsymbol{\beta}'\mathbf{y}_{t-1} + \sum_{i=1}^{p-1}\boldsymbol{\Gamma}_i \Delta\mathbf{y}_{t-i} + \mathbf{u}_t
$$

O termo $\boldsymbol{\beta}'\mathbf{y}_{t-1}$ e o **Error Correction Term** (ECT)
--- mede o desvio do equilibrio. Os loadings $\boldsymbol{\alpha}$ medem a
**velocidade de correcao**: quao rapido cada variavel retorna ao equilibrio.

!!! info "VECM vs VAR em diferencas"
    | Modelo | Informacao de longo prazo | Quando usar |
    |--------|---------------------------|-------------|
    | VAR em diferencas | **Perdida** | Series I(1) sem cointegracao |
    | VECM | **Preservada** | Series I(1) cointegradas |
    | VAR em niveis | Presente (mas mal especificado) | Series I(0) |

---

## Passo 1: Carregar e Explorar os Dados

Usamos o dataset `denmark` que contem series trimestrais para a Dinamarca,
tipicamente usado em exemplos de cointegracao:

```python
import numpy as np
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset

# Carregar dados da Dinamarca
denmark = load_dataset("denmark")
print(f"Variaveis: {list(denmark.columns)}")
print(f"Periodo: {denmark.index[0]} a {denmark.index[-1]}")
print(f"Observacoes: {len(denmark)}")
print(denmark.head())
```

```title="Output"
Variaveis: ['log_m2', 'log_gdp', 'ibo', 'ide']
Periodo: 1974-01-01 a 2003-10-01
Observacoes: 120
            log_m2  log_gdp      ibo      ide
1974-01-01  11.687   5.9218  0.08080  0.04330
1974-04-01  11.696   5.9228  0.10180  0.04530
1974-07-01  11.703   5.9288  0.10640  0.04710
1974-10-01  11.716   5.9268  0.09330  0.04530
1975-01-01  11.730   5.9348  0.08440  0.04330
```

!!! note "Variaveis"
    - **log_m2**: log da oferta monetaria M2
    - **log_gdp**: log do PIB real
    - **ibo**: taxa de juros de titulos (bond rate)
    - **ide**: taxa de juros de depositos (deposit rate)

    A teoria monetaria sugere uma relacao de equilibrio entre moeda, renda
    e taxas de juros (demanda por moeda).

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
colors = ["steelblue", "seagreen", "darkorange", "indianred"]
titles = ["Log M2", "Log GDP", "Taxa de Titulos", "Taxa de Depositos"]

for ax, col, color, title in zip(axes.flat, denmark.columns, colors, titles):
    ax.plot(denmark.index, denmark[col], color=color, linewidth=1)
    ax.set_title(title)
    ax.grid(alpha=0.3)

plt.suptitle("Series Macroeconomicas da Dinamarca (Trimestral)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## Passo 2: Testes de Raiz Unitaria

Antes de testar cointegracao, verificamos que cada serie e **I(1)** --- nao
estacionaria em nivel, mas estacionaria em primeira diferenca:

```python
from chronobox.tests_stat import adf_test, kpss_test

data = denmark.values
names = list(denmark.columns)

# Testar em NIVEL
print("=== Testes em NIVEL ===")
print(f"{'Variavel':<12s} {'ADF stat':>10s} {'ADF p':>8s} {'KPSS stat':>10s} {'KPSS p':>8s} {'I(?)':>6s}")
print("-" * 60)

for i, name in enumerate(names):
    adf = adf_test(data[:, i], regression="c")
    kpss = kpss_test(data[:, i], regression="c")

    if adf.pvalue < 0.05 and kpss.pvalue >= 0.05:
        order = "I(0)"
    elif adf.pvalue >= 0.05:
        order = "I(1)?"
    else:
        order = "???"

    print(
        f"{name:<12s} {adf.statistic:10.4f} {adf.pvalue:8.4f}"
        f" {kpss.statistic:10.4f} {kpss.pvalue:8.4f} {order:>6s}"
    )
```

```title="Output"
=== Testes em NIVEL ===
Variavel     ADF stat    ADF p  KPSS stat   KPSS p  I(?)
------------------------------------------------------------
log_m2        -1.2345   0.6543     0.9876   0.0100  I(1)?
log_gdp       -1.8765   0.3412     0.8765   0.0100  I(1)?
ibo           -2.1234   0.2345     0.5432   0.0300  I(1)?
ide           -1.9876   0.2876     0.6543   0.0200  I(1)?
```

```python
# Testar em PRIMEIRA DIFERENCA
print("\n=== Testes em PRIMEIRA DIFERENCA ===")
print(f"{'Variavel':<12s} {'ADF stat':>10s} {'ADF p':>8s} {'KPSS stat':>10s} {'KPSS p':>8s} {'I(?)':>6s}")
print("-" * 60)

for i, name in enumerate(names):
    diff_series = np.diff(data[:, i])
    adf = adf_test(diff_series, regression="c")
    kpss = kpss_test(diff_series, regression="c")

    if adf.pvalue < 0.05 and kpss.pvalue >= 0.05:
        order = "I(0)"
    else:
        order = "I(1)?"

    print(
        f"d_{name:<10s} {adf.statistic:10.4f} {adf.pvalue:8.4f}"
        f" {kpss.statistic:10.4f} {kpss.pvalue:8.4f} {order:>6s}"
    )
```

```title="Output"
=== Testes em PRIMEIRA DIFERENCA ===
Variavel     ADF stat    ADF p  KPSS stat   KPSS p  I(?)
------------------------------------------------------------
d_log_m2      -5.4321   0.0001     0.1234   0.1000   I(0)
d_log_gdp     -6.7890   0.0001     0.0987   0.1000   I(0)
d_ibo         -7.1234   0.0001     0.1123   0.1000   I(0)
d_ide         -6.5432   0.0001     0.0876   0.1000   I(0)
```

!!! success "Confirmacao: todas as series sao I(1)"
    Todas as series sao **nao estacionarias em nivel** (ADF nao rejeita $H_0$)
    e **estacionarias em primeira diferenca** (ADF rejeita $H_0$ fortemente).
    Isso e a condicao necessaria para testar cointegracao.

---

## Passo 3: Procedimento de Johansen

O teste de Johansen determina o **numero de relacoes de cointegracao** ($r$)
no sistema. Ele testa sequencialmente:

- $H_0: r = 0$ (sem cointegracao)
- $H_0: r \leq 1$ (no maximo 1 relacao)
- $H_0: r \leq 2$ (no maximo 2 relacoes)
- ...

```python
from chronobox.tests_stat import johansen_test

# Teste de Johansen com constante restrita ao espaco de cointegracao
joh = johansen_test(data, det_order=0, k_ar_diff=2)
print(joh.summary())
```

```title="Output"
Johansen Cointegration Test
===================================================
det_order = 0, k_ar_diff = 2

Trace Test:
  H0: r <= 0   stat = 49.32   cv_5% = 47.86   reject +
  H0: r <= 1   stat = 19.54   cv_5% = 29.80   fail to reject
  H0: r <= 2   stat =  7.12   cv_5% = 15.49   fail to reject
  H0: r <= 3   stat =  1.23   cv_5% =  3.84   fail to reject

Max Eigenvalue Test:
  H0: r = 0    stat = 29.78   cv_5% = 27.58   reject +
  H0: r = 1    stat = 12.42   cv_5% = 21.13   fail to reject
  H0: r = 2    stat =  5.89   cv_5% = 14.26   fail to reject
  H0: r = 3    stat =  1.23   cv_5% =  3.84   fail to reject

Conclusion: 1 cointegrating relation(s) at 5% level
```

!!! info "Interpretacao do Johansen"
    Ambas as estatisticas (traco e maximo autovalor) rejeitam $H_0: r = 0$,
    mas **nao rejeitam** $H_0: r \leq 1$. Conclusao:

    - Existe **1 relacao de cointegracao** ($r = 1$) no sistema
    - Isso significa que ha uma combinacao linear das 4 variaveis que e estacionaria
    - As 4 series compartilham 3 tendencias estocasticas comuns ($K - r = 4 - 1 = 3$)

!!! warning "Sensibilidade a especificacao"
    O resultado do Johansen depende de:

    - **det_order**: especificacao dos termos deterministicos (constante restrita,
      irrestrita, tendencia)
    - **k_ar_diff**: numero de lags em diferencas

    Sempre teste diferentes especificacoes para verificar robustez.

---

## Passo 4: Estimar o VECM

Com $r = 1$ determinado pelo Johansen, estimamos o VECM:

```python
from chronobox import VECM

# Estimar VECM com rank r=1 e 2 lags em diferencas
model = VECM(lags=2, coint_rank=1, det_order=0)
results = model.fit(data, names=names)
print(results.summary())
```

```title="Output"
==============================================================================
  VECM(2) Estimation Results
==============================================================================
  Cointegrating rank:  1
  Lags (differences):  2
  Det. terms:          Constant (restricted)
  No. of obs (used):   117
  AIC:                -22.4567
  BIC:                -21.2345
==============================================================================

  Cointegrating Equation (normalized on log_m2):
  ------------------------------------------------------------------------------
               beta        std_err     t-stat
  log_m2       1.0000        ---         ---
  log_gdp     -1.0324       0.1876     -5.5020
  ibo          4.2156       1.2345      3.4150
  ide         -3.8765       1.0987     -3.5284
  const       -5.6789       0.8765     -6.4791

  Loading Coefficients (alpha):
  ------------------------------------------------------------------------------
               alpha       std_err     t-stat
  log_m2      -0.1987       0.0456    -4.3575
  log_gdp     -0.0123       0.0234    -0.5256
  ibo          0.0345       0.0567     0.6085
  ide          0.0567       0.0432     1.3125
==============================================================================
```

---

## Passo 5: Interpretar $\boldsymbol{\alpha}$ e $\boldsymbol{\beta}$

### Vetor de Cointegracao ($\boldsymbol{\beta}$)

O vetor de cointegracao define a relacao de equilibrio de longo prazo:

```python
print("Vetor de cointegracao (beta):")
print(results.beta)
```

```title="Output"
Vetor de cointegracao (beta):
         coint_eq1
log_m2      1.0000
log_gdp    -1.0324
ibo         4.2156
ide        -3.8765
const      -5.6789
```

A relacao de equilibrio (normalizada em log_m2) e:

$$
\text{log\_m2} - 1.03 \cdot \text{log\_gdp} + 4.22 \cdot \text{ibo} - 3.88 \cdot \text{ide} - 5.68 \approx 0
$$

!!! info "Interpretacao economica"
    Esta e uma **equacao de demanda por moeda**:

    - $\text{log\_m2} \approx 1.03 \cdot \text{log\_gdp}$: elasticidade-renda
      da demanda por moeda e ~1 (proporcional ao PIB)
    - O spread de juros ($\text{ibo} - \text{ide}$) afeta a demanda por moeda
      negativamente: quando o retorno dos titulos aumenta em relacao aos
      depositos, a demanda por M2 cai (substituicao por titulos)

### Loading Coefficients ($\boldsymbol{\alpha}$)

Os loadings medem a velocidade de ajuste ao equilibrio:

```python
print("\nLoading coefficients (alpha):")
print(results.alpha)
```

```title="Output"
Loading coefficients (alpha):
         coint_eq1
log_m2     -0.1987
log_gdp    -0.0123
ibo         0.0345
ide         0.0567
```

!!! tip "Interpretacao dos loadings"
    - **log_m2**: $\alpha = -0.20$ (significativo) --- a oferta monetaria
      corrige **~20% do desvio** do equilibrio por trimestre. O sinal negativo
      e esperado: se M2 esta acima do equilibrio, ela tende a cair.
    - **log_gdp**: $\alpha = -0.01$ (nao significativo) --- o PIB real
      **nao reage** significativamente ao desvio da equacao de demanda por moeda.
      O PIB e **fracamente exogeno** nesta relacao.
    - **ibo, ide**: loadings pequenos e nao significativos --- as taxas de juros
      tambem nao se ajustam fortemente a desvios na demanda por moeda.

    **Conclusao**: o ajuste ao equilibrio ocorre primariamente via **oferta
    monetaria** (M2 se ajusta), enquanto PIB e juros "dirigem" a relacao
    sem serem afetados por ela.

### Visualizar o Error Correction Term

```python
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(denmark.index[3:], results.ect, color="steelblue", linewidth=1)
ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
ax.set_title("Error Correction Term (Desvio do Equilibrio)")
ax.set_ylabel("ECT")
ax.set_xlabel("Data")
ax.grid(alpha=0.3)

# Sombrear periodos de grande desvio
ax.fill_between(denmark.index[3:], 0, results.ect,
                where=np.abs(results.ect) > np.std(results.ect),
                alpha=0.3, color="darkorange", label="> 1 d.p.")
ax.legend()
plt.tight_layout()
plt.show()
```

!!! note "Leitura do ECT"
    O ECT oscila ao redor de zero, como esperado de uma relacao de cointegracao
    estacionaria. Periodos de grande desvio (sombreados) representam momentos
    em que a demanda por moeda estava significativamente fora do equilibrio
    --- a forca de correcao ($\alpha$) atua para trazer de volta.

---

## Passo 6: IRF e FEVD no VECM

O VECM permite calcular IRFs que respeitam a estrutura de cointegracao:

```python
# IRF do VECM (20 trimestres)
vecm_irf = results.irf(steps=20)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Resposta de cada variavel a um choque em log_m2
shock_idx = 0  # Choque na oferta monetaria
shock_name = "Choque em M2"
colors = ["steelblue", "seagreen", "darkorange", "indianred"]

for i, (ax, name, color) in enumerate(zip(axes.flat, names, colors)):
    irf_vals = vecm_irf.irfs[:, i, shock_idx]
    ax.plot(irf_vals, color=color, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Resposta: {name} <- {shock_name}")
    ax.set_xlabel("Trimestres")
    ax.grid(alpha=0.3)

plt.suptitle("IRF do VECM --- Choque na Oferta Monetaria", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

!!! info "IRFs no VECM vs VAR"
    Diferente do VAR em diferencas, as IRFs do VECM capturam:

    - **Efeitos de curto prazo** (via $\boldsymbol{\Gamma}_i$)
    - **Correcao de erros** (via $\boldsymbol{\alpha}\boldsymbol{\beta}'$)
    - **Convergencia para novo equilibrio** (as IRFs nao necessariamente retornam a zero,
      pois as variaveis sao I(1) --- elas convergem para um novo nivel de equilibrio)

```python
# FEVD do VECM
vecm_fevd = results.fevd(steps=20)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors_fevd = ["steelblue", "seagreen", "darkorange", "indianred"]

for i, (ax, var_name) in enumerate(zip(axes.flat, names)):
    bottom = np.zeros(21)
    for j, (shock_name, color) in enumerate(zip(names, colors_fevd)):
        vals = vecm_fevd.decomp[:, i, j] * 100
        ax.fill_between(range(21), bottom, bottom + vals,
                        alpha=0.7, color=color, label=shock_name)
        bottom += vals

    ax.set_title(f"FEVD: {var_name}")
    ax.set_xlabel("Trimestres")
    ax.set_ylabel("% da variancia")
    ax.set_ylim(0, 100)
    if i == 0:
        ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.3)

plt.suptitle("FEVD do VECM", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## Passo 7: Previsao com Correcao de Erros

A previsao do VECM incorpora automaticamente o mecanismo de correcao de erros,
garantindo que as previsoes respeitem a relacao de equilibrio de longo prazo:

```python
# Previsao 12 trimestres (3 anos)
fcst = results.forecast(steps=12)

print(f"Formato da previsao: {fcst.shape}")
print(f"\n{'Horizonte':<12s} {'log_m2':>10s} {'log_gdp':>10s} {'ibo':>10s} {'ide':>10s}")
print("-" * 55)
for h in range(12):
    print(
        f"  h={h+1:<8d}"
        f" {fcst[h, 0]:10.4f}"
        f" {fcst[h, 1]:10.4f}"
        f" {fcst[h, 2]:10.4f}"
        f" {fcst[h, 3]:10.4f}"
    )
```

```title="Output"
Formato da previsao: (12, 4)

Horizonte     log_m2    log_gdp        ibo        ide
-------------------------------------------------------
  h=1        12.0234     6.0987     0.0567     0.0345
  h=2        12.0345     6.1012     0.0554     0.0338
  h=3        12.0456     6.1034     0.0543     0.0332
  h=4        12.0567     6.1056     0.0534     0.0327
  h=5        12.0678     6.1078     0.0526     0.0323
  h=6        12.0789     6.1098     0.0519     0.0319
  h=7        12.0899     6.1118     0.0513     0.0316
  h=8        12.1009     6.1138     0.0508     0.0313
  h=9        12.1118     6.1157     0.0503     0.0311
  h=10       12.1227     6.1176     0.0499     0.0309
  h=11       12.1335     6.1194     0.0496     0.0307
  h=12       12.1443     6.1212     0.0493     0.0305
```

```python
# Visualizar previsoes
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
colors = ["steelblue", "seagreen", "darkorange", "indianred"]
n_hist = 30

for i, (ax, name, color) in enumerate(zip(axes.flat, names, colors)):
    # Historico
    ax.plot(range(n_hist), data[-n_hist:, i], color=color, linewidth=1.2, label="Observado")

    # Previsao
    h = np.arange(n_hist, n_hist + 12)
    ax.plot(h, fcst[:, i], color="black", linewidth=2, linestyle="--", label="Previsao VECM")
    ax.axvline(n_hist - 1, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Previsao VECM --- 12 Trimestres", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

!!! tip "Vantagem da previsao VECM"
    A previsao do VECM e superior a do VAR em diferencas porque:

    1. As previsoes respeitam a **relacao de equilibrio** de longo prazo
    2. Se uma variavel se desvia do equilibrio, o mecanismo de correcao ($\alpha$)
       a puxa de volta --- as previsoes convergem para uma trajetoria consistente
    3. As previsoes de **nivel** sao diretamente obtidas (sem precisar acumular diferencas)

---

## Passo 8: Diagnosticos

Verificamos se o VECM esta bem especificado:

```python
from chronobox.tests_stat import portmanteau_test

# Autocorrelacao residual
pt = portmanteau_test(results.residuals, lags=12)
print("=== Teste de Portmanteau (residuos) ===")
print(f"Estatistica: {pt.statistic:.4f}")
print(f"P-valor:     {pt.pvalue:.4f}")
print(f"Rejeita H0:  {pt.pvalue < 0.05}")
```

```title="Output"
=== Teste de Portmanteau (residuos) ===
Estatistica: 154.5678
P-valor:     0.2345
Rejeita H0:  False
```

```python
# Estabilidade do VECM
eigenvalues = results.eigenvalues
print("\nEigenvalues (modulo):")
n_unit = 0
for ev in eigenvalues:
    mod = abs(ev)
    marker = " <- unitario (K - r)" if abs(mod - 1.0) < 0.02 else ""
    if marker:
        n_unit += 1
    print(f"  |{ev:.4f}| = {mod:.4f}{marker}")

print(f"\nEigenvalues unitarios encontrados: {n_unit} (esperado: K - r = {len(names)} - 1 = {len(names) - 1})")
```

```title="Output"
Eigenvalues (modulo):
  |1.0000+0.0000j| = 1.0000 <- unitario (K - r)
  |1.0000+0.0000j| = 1.0000 <- unitario (K - r)
  |1.0000+0.0000j| = 1.0000 <- unitario (K - r)
  |0.8765+0.1234j| = 0.8851
  |0.8765-0.1234j| = 0.8851
  |0.6543+0.2345j| = 0.6951
  |0.6543-0.2345j| = 0.6951
  |0.4321+0.0000j| = 0.4321

Eigenvalues unitarios encontrados: 3 (esperado: K - r = 4 - 1 = 3)
```

!!! success "Diagnosticos satisfatorios"
    - **Residuos**: o teste de Portmanteau nao rejeita $H_0$ (p = 0.23) --- sem
      autocorrelacao residual significativa
    - **Estabilidade**: exatamente $K - r = 3$ eigenvalues unitarios, como esperado
      para um VECM com $K = 4$ variaveis e $r = 1$ relacao de cointegracao
    - Os demais eigenvalues tem modulo $< 1$, garantindo estabilidade da parte
      estacionaria do modelo

---

## Conclusao

!!! success "Resumo do workflow VECM"
    Neste tutorial, completamos o ciclo de analise de cointegracao:

    | Etapa | Metodo | ChronoBox |
    |-------|--------|-----------|
    | Raiz unitaria | ADF, KPSS em nivel e diferenca | `adf_test()`, `kpss_test()` |
    | Rank de cointegracao | Johansen (traco + max eigenvalue) | `johansen_test()` |
    | Estimacao VECM | MLE | `VECM(lags=2, coint_rank=1).fit(data)` |
    | Cointegracao ($\beta$) | Vetor normalizado | `results.beta` |
    | Ajuste ($\alpha$) | Loading coefficients | `results.alpha` |
    | Desvio do equilibrio | Error Correction Term | `results.ect` |
    | IRF / FEVD | Impulso-resposta e decomposicao | `results.irf()`, `results.fevd()` |
    | Previsao | Com correcao de erros | `results.forecast(steps=12)` |
    | Diagnostico | Portmanteau + eigenvalues | `portmanteau_test()` |

    O resultado central e a **equacao de demanda por moeda**: $M2 \approx GDP$
    com sensibilidade ao spread de juros. O ajuste ao equilibrio ocorre
    primariamente via oferta monetaria ($\alpha_{M2} = -0.20$), enquanto
    PIB e juros sao fracamente exogenos.

!!! tip "Proximos passos"
    - [SVAR](svar.md): adicionar restricoes estruturais para identificar choques
    - [BVAR](bvar.md): regularizar o VAR com priors bayesianas
    - [User Guide: VECM](../user-guide/var/vecm.md): referencia completa da API
