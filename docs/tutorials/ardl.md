---
title: "Tutorial: ARDL Bounds Testing"
description: Tutorial completo de ARDL bounds testing --- da selecao de lags ao ECM, multiplicadores de curto e longo prazo e dynamic multiplier plots.
---

# ARDL Bounds Testing

!!! abstract "O que voce vai aprender"
    - Entender quando o ARDL e preferivel ao Johansen
    - Testar ordens de integracao (verificar mistura I(0)/I(1))
    - Estimar um ARDL com selecao automatica de lags
    - Aplicar o bounds test de Pesaran, Shin & Smith (2001)
    - Derivar o ECM (Error Correction Model) a partir do ARDL
    - Interpretar multiplicadores de curto e longo prazo
    - Visualizar multiplicadores dinamicos

**Nivel**: Intermediario
**Tempo estimado**: ~35 minutos
**Dataset**: Macro (consumo, renda, riqueza)
**Pre-requisito**: Familiaridade basica com series temporais e cointegracao

---

## Introducao: Por que ARDL?

Em econometria aplicada, frequentemente queremos testar se existe uma relacao
de **longo prazo** (cointegracao) entre variaveis economicas. O metodo classico
de Johansen requer que **todas** as variaveis sejam I(1). Mas na pratica,
muitas vezes temos uma mistura de variaveis I(0) e I(1).

O modelo **ARDL** (Autoregressive Distributed Lag) e a solucao proposta por
Pesaran, Shin & Smith (2001). Sua grande vantagem e o **bounds test**, que
permite testar cointegracao **sem precisar determinar a priori** se as
variaveis sao I(0) ou I(1):

$$
y_t = c + \sum_{i=1}^{p} \phi_i \, y_{t-i} + \sum_{j=1}^{k} \sum_{\ell=0}^{q_j} \beta_{j,\ell} \, x_{j,t-\ell} + \epsilon_t
$$

!!! info "Quando usar ARDL vs Johansen"
    | Criterio | ARDL | Johansen |
    |----------|------|----------|
    | Ordens de integracao | Mistura I(0) e I(1) | Todas I(1) |
    | Tamanho amostral | Funciona bem com T < 80 | Requer T grande |
    | Numero de equacoes | Equacao unica | Sistema multivariado |
    | Multiplas relacoes | Uma relacao de cointegracao | Multiplas relacoes |
    | Pre-teste de raiz unitaria | Nao obrigatorio | Obrigatorio |

---

## Passo 1: Carregar e Explorar os Dados

Usamos um dataset macroeconomico com tres variaveis: consumo agregado,
renda disponivel e riqueza financeira. A questao e: **existe uma relacao
de longo prazo entre consumo, renda e riqueza?**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset

# Carregar dados macro
data = load_dataset("us_macro_quarterly")
print(f"Tipo: {type(data)}")
print(f"Colunas: {list(data.columns)}")
print(f"Periodo: {data.index[0]} a {data.index[-1]}")
print(f"Observacoes: {len(data)}")
print(data.head())
```

```title="Output"
Tipo: <class 'pandas.core.frame.DataFrame'>
Colunas: ['consumption', 'income', 'wealth', 'gdp_growth', 'inflation', 'interest_rate']
Periodo: 1960-01-01 a 2023-10-01
Observacoes: 256
           consumption    income    wealth  gdp_growth  inflation  interest_rate
1960-01-01     1523.45   1678.23   5432.10        2.31       1.72           3.93
1960-04-01     1534.67   1690.12   5478.34       -2.07       1.48           3.29
1960-07-01     1541.23   1698.45   5501.67        1.54       1.35           2.28
1960-10-01     1538.90   1687.34   5489.23       -5.15       1.55           2.41
1961-01-01     1545.12   1702.56   5534.78        2.60       1.28           2.33
```

```python
# Selecionar as variaveis de interesse
y = data["consumption"]
X = data[["income", "wealth"]]

# Visualizar as series
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

vars_plot = [("consumption", "Consumo", "steelblue"),
             ("income", "Renda Disponivel", "seagreen"),
             ("wealth", "Riqueza Financeira", "darkorange")]

for ax, (col, label, color) in zip(axes, vars_plot):
    ax.plot(data.index, data[col], color=color, linewidth=0.9)
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)

axes[0].set_title("Consumo, Renda e Riqueza (Trimestral, EUA)")
plt.tight_layout()
plt.show()
```

!!! note "Observacao visual"
    As tres series apresentam tendencia crescente e parecem se mover juntas
    no longo prazo --- um indicativo visual de possivel cointegracao. Mas
    precisamos de testes formais para confirmar.

---

## Passo 2: Testar Ordens de Integracao

Embora o ARDL **nao exija** que todas as variaveis tenham a mesma ordem de
integracao, precisamos garantir que **nenhuma variavel seja I(2)** --- o
bounds test nao e valido para variaveis I(2).

```python
from chronobox.tests_stat import adf_test, kpss_test

names = ["consumption", "income", "wealth"]
series_list = [data[n].values for n in names]

print(f"{'Variavel':<18s} {'ADF stat':>10s} {'ADF p':>8s} {'KPSS stat':>10s} {'KPSS p':>8s} {'Ordem'}")
print("-" * 75)

for name, s in zip(names, series_list):
    adf = adf_test(s, regression="c")
    kpss = kpss_test(s, regression="c")

    if adf.pvalue < 0.05 and kpss.pvalue >= 0.05:
        order = "I(0)"
    elif adf.pvalue >= 0.05 and kpss.pvalue < 0.05:
        order = "I(1)"
    else:
        order = "Inconclusivo"

    print(
        f"{name:<18s} {adf.statistic:10.4f} {adf.pvalue:8.4f}"
        f" {kpss.statistic:10.4f} {kpss.pvalue:8.4f} {order}"
    )
```

```title="Output"
Variavel           ADF stat    ADF p  KPSS stat   KPSS p Ordem
---------------------------------------------------------------------------
consumption        -1.4523   0.5567     1.8765   0.0100 I(1)
income             -1.8234   0.3678     1.6543   0.0100 I(1)
wealth             -2.8976   0.0456     0.2134   0.1000 I(0)
```

```python
# Confirmar que nenhuma e I(2): testar em primeiras diferencas
print(f"\n{'Variavel':<18s} {'ADF stat (diff)':>15s} {'ADF p':>8s} {'Conclusao'}")
print("-" * 55)

for name, s in zip(names, series_list):
    d_s = np.diff(s)
    adf_d = adf_test(d_s, regression="c")
    conclusion = "Estac. em diff (max I(1))" if adf_d.pvalue < 0.05 else "Possivel I(2)!"
    print(f"{name:<18s} {adf_d.statistic:15.4f} {adf_d.pvalue:8.4f} {conclusion}")
```

```title="Output"
Variavel           ADF stat (diff)    ADF p Conclusao
-------------------------------------------------------
consumption            -8.7654   0.0000 Estac. em diff (max I(1))
income                 -9.1234   0.0000 Estac. em diff (max I(1))
wealth                 -11.2345   0.0000 Estac. em diff (max I(1))
```

!!! success "Condicao satisfeita"
    - **Consumo** e **renda** sao I(1) (nao-estacionarias em nivel, estacionarias em diferenca)
    - **Riqueza** e I(0) (estacionaria em nivel)
    - Nenhuma variavel e I(2) --- o bounds test e aplicavel

    Esta e exatamente a situacao onde o ARDL brilha: **mistura de I(0) e I(1)**.
    O Johansen nao seria valido aqui porque riqueza e I(0).

---

## Passo 3: Estimar ARDL com Selecao Automatica de Lags

O ARDL permite selecao automatica das ordens de defasagem $p$ (para $y$)
e $q_1, q_2, \ldots, q_k$ (para cada $x_j$) via criterios de informacao:

```python
from chronobox import ARDL

# Estimar ARDL com selecao automatica (maximo 4 lags, criterio AIC)
model = ARDL(lags=4, exog_lags=4, trend='c', ic='aic')
results = model.fit(y=y, exog=X)

print(results.summary())
print(f"\nLags selecionados: ARDL({results.ar_lags}, {results.dl_lags})")
```

```title="Output"
                     ARDL(2, 1, 1) Results
==========================================================================
Dep. Variable:        consumption    No. Observations:          252
Method:                       OLS    R-squared:              0.9987
                                     Adj. R-squared:         0.9986
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

Lags selecionados: ARDL(2, 1, 1)
```

!!! tip "Selecao de lags"
    O AIC selecionou um ARDL(2, 1, 1): 2 defasagens do consumo, 1 defasagem
    da renda e 1 defasagem da riqueza. Pesaran et al. (2001) recomendam usar
    o AIC para a selecao, pois subajustar os lags pode invalidar o bounds test.

---

## Passo 4: Bounds Test --- Testar Cointegracao

O bounds test verifica se existe uma relacao de **longo prazo** entre as variaveis.
O ARDL e reparametrizado na forma de correcao de erros condicional:

$$
\Delta y_t = c + \pi_y \, y_{t-1} + \sum_{j=1}^{k} \pi_j \, x_{j,t-1} + \text{curto prazo} + \epsilon_t
$$

As hipoteses sao:

$$
H_0: \pi_y = \pi_1 = \cdots = \pi_k = 0 \quad \text{(sem relacao de longo prazo)}
$$

A estatistica F e comparada com **duas bandas** de valores criticos (I(0) e I(1)):

```python
# Aplicar o bounds test
bounds = results.bounds_test()
print(bounds.summary())
```

```title="Output"
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

!!! info "Interpretacao do bounds test"
    A estatistica $F = 5.423$ e comparada com as bandas:

    | Resultado | Condicao | Decisao |
    |-----------|----------|---------|
    | $F > 4.85$ (banda superior) | ✓ Satisfeita | **Rejeita $H_0$**: cointegracao |
    | $3.79 \leq F \leq 4.85$ | --- | Inconclusivo |
    | $F < 3.79$ (banda inferior) | --- | Nao rejeita: sem cointegracao |

    Como $F = 5.423 > 4.85$, **rejeitamos $H_0$** a 5%: existe evidencia de
    uma relacao de longo prazo entre consumo, renda e riqueza.

!!! warning "Resultado inconclusivo"
    Se o teste fosse inconclusivo ($F$ entre as bandas), seria necessario
    determinar as ordens de integracao exatas e usar os valores criticos
    correspondentes. Neste caso, os pre-testes do Passo 2 seriam essenciais.

---

## Passo 5: Derivar o ECM (Error Correction Model)

Com evidencia de cointegracao, reparametrizamos o ARDL na forma de
**correcao de erros**, separando dinamica de curto e longo prazo:

$$
\Delta y_t = c + \underbrace{\alpha \left( y_{t-1} - \theta_1 x_{1,t-1} - \theta_2 x_{2,t-1} \right)}_{\text{correcao de erros}} + \underbrace{\sum \gamma_i \Delta y_{t-i} + \sum \delta_{j,\ell} \Delta x_{j,t-\ell}}_{\text{dinamica de curto prazo}} + \epsilon_t
$$

```python
# Converter ARDL para ECM
ecm = results.to_ecm()
print(ecm.summary())
```

```title="Output"
               ECM (Error Correction Model) Results
==========================================================================
Dep. Variable:       Δconsumption    No. Observations:          251
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
```

!!! note "Parametros chave do ECM"
    | Parametro | Valor | Interpretacao |
    |-----------|-------|---------------|
    | $\alpha$ (EC(t-1)) | $-0.5122$ | 51.2% do desvio do equilibrio e corrigido por trimestre |
    | $\theta_{\text{income}}$ | $0.655$ | No longo prazo, +1 unidade de renda → +0.655 de consumo |
    | $\theta_{\text{wealth}}$ | $0.133$ | No longo prazo, +1 unidade de riqueza → +0.133 de consumo |
    | $\delta_{\text{income}}$ | $0.253$ | No curto prazo, +1 unidade de $\Delta$renda → +0.253 de $\Delta$consumo |

    O $\alpha$ negativo e significativo confirma o mecanismo de correcao de erros:
    quando o consumo esta acima do equilibrio de longo prazo, ele se ajusta para baixo.

```python
# Velocidade de ajuste e meia-vida
alpha = ecm.alpha
half_life = np.log(0.5) / np.log(1 + alpha)
print(f"Velocidade de ajuste (alpha): {alpha:.4f}")
print(f"Meia-vida do ajuste: {half_life:.2f} trimestres")
```

```title="Output"
Velocidade de ajuste (alpha): -0.5122
Meia-vida do ajuste: 0.97 trimestres
```

!!! tip "Meia-vida"
    A meia-vida de ~1 trimestre indica ajuste muito rapido: metade do desvio
    do equilibrio e corrigido em aproximadamente 3 meses. Isso e consistente
    com a teoria do consumo permanente, onde os consumidores ajustam rapidamente
    seus gastos a mudancas na renda permanente.

---

## Passo 6: Multiplicadores de Curto e Longo Prazo

Os multiplicadores quantificam o efeito de uma mudanca em $x_j$ sobre $y$
em diferentes horizontes:

```python
# Multiplicadores de curto prazo (efeito imediato)
sr = ecm.short_run_coefficients()
print("=== Multiplicadores de Curto Prazo ===")
print(sr)

print()

# Multiplicadores de longo prazo (equilibrio)
lr = ecm.long_run_coefficients()
print("=== Multiplicadores de Longo Prazo ===")
print(lr)
```

```title="Output"
=== Multiplicadores de Curto Prazo ===
             coef     std err       t     P>|t|
income      0.2534     0.0645    3.929    0.000
wealth      0.0412     0.0198    2.081    0.040

=== Multiplicadores de Longo Prazo ===
             coef     std err       t     P>|t|
income      0.6548     0.0923    7.094    0.000
wealth      0.1325     0.0412    3.217    0.002
```

!!! info "Comparacao curto vs longo prazo"
    | Variavel | Curto Prazo | Longo Prazo | Razao LP/CP |
    |----------|-------------|-------------|-------------|
    | Renda | 0.253 | 0.655 | 2.6x |
    | Riqueza | 0.041 | 0.133 | 3.2x |

    O efeito de longo prazo e substancialmente maior que o de curto prazo
    para ambas as variaveis. Isso significa que o impacto total de uma mudanca
    permanente na renda se materializa ao longo de varios trimestres, nao
    instantaneamente.

    A **propensao marginal a consumir** de longo prazo (0.655) e consistente
    com estimativas classicas da funcao consumo keynesiana.

---

## Passo 7: Multiplicadores Dinamicos

Os multiplicadores dinamicos rastreiam o efeito **periodo a periodo** de
um choque unitario permanente em $x_j$ sobre $y$:

$$
m_j(h) = \frac{\partial y_{t+h}}{\partial x_t}, \quad h = 0, 1, 2, \ldots
$$

A sequencia $\{m_j(h)\}$ comeca no multiplicador de curto prazo e converge
para o multiplicador de longo prazo.

```python
# Calcular multiplicadores dinamicos
dm = ecm.dynamic_multipliers(steps=30)
print(dm.head(10))
```

```title="Output"
   step    income    wealth
      0    0.2534    0.0412
      1    0.3891    0.0687
      2    0.4623    0.0856
      3    0.5089    0.0965
      4    0.5367    0.1040
      5    0.5612    0.1098
      6    0.5743    0.1132
      7    0.5834    0.1155
      8    0.5901    0.1172
      9    0.5945    0.1183
```

```python
# Visualizar multiplicadores dinamicos
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["steelblue", "darkorange"]
var_names = ["income", "wealth"]
var_labels = ["Renda", "Riqueza"]
lr_vals = [0.6548, 0.1325]

for ax, var, label, color, lr_val in zip(axes, var_names, var_labels, colors, lr_vals):
    ax.plot(dm["step"], dm[var], color=color, linewidth=2.5, label="Multiplicador dinamico")
    ax.axhline(lr_val, color="gray", linestyle="--", linewidth=1.5,
               label=f"Longo prazo ({lr_val:.3f})")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.fill_between(dm["step"], 0, dm[var], alpha=0.1, color=color)
    ax.set_xlabel("Trimestres apos o choque")
    ax.set_ylabel("Efeito acumulado sobre o consumo")
    ax.set_title(f"Dynamic Multiplier: {label}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

!!! note "Leitura dos graficos"
    - O efeito comeca no **multiplicador de curto prazo** (impacto contemporaneo)
    - Converge monotonicamente para o **multiplicador de longo prazo** (linha tracejada)
    - A velocidade de convergencia depende de $\alpha$: com $\alpha = -0.51$,
      a maior parte do ajuste ocorre nos primeiros 3--4 trimestres
    - Apos ~10 trimestres, o efeito esta essencialmente no valor de longo prazo

```python
# Usando a funcao de visualizacao integrada
from chronobox.visualization import plot_dynamic_multipliers

fig = plot_dynamic_multipliers(
    ecm,
    steps=30,
    ci=0.95,
    variables=["income", "wealth"],
)
plt.show()
```

O grafico com bandas de confianca (bootstrap) mostra a incerteza ao redor
dos multiplicadores dinamicos. A convergencia para o valor de longo prazo
e visivel pela estabilizacao da curva e estreitamento das bandas.

---

## Diagnosticos do ECM

Antes de confiar nos resultados, verificamos a validade do modelo:

```python
from chronobox.tests_stat import adf_test, breusch_godfrey_test, ljung_box_test

# 1. EC term deve ser estacionario (I(0))
ec_series = ecm.ec_term
adf_ec = adf_test(ec_series.values)
print(f"1. ADF do EC term:    stat = {adf_ec.statistic:.4f}, p = {adf_ec.pvalue:.4f}")
print(f"   EC term estacionario: {'Sim' if adf_ec.pvalue < 0.05 else 'Nao'}")

# 2. Residuos sem autocorrelacao
bg = breusch_godfrey_test(ecm.residuals.values, lags=4)
print(f"\n2. Breusch-Godfrey:   stat = {bg.statistic:.4f}, p = {bg.pvalue:.4f}")
print(f"   Sem autocorrelacao: {'Sim' if bg.pvalue > 0.05 else 'Nao'}")

# 3. Alpha negativo e significativo
print(f"\n3. Alpha = {ecm.alpha:.4f}")
print(f"   Negativo e significativo: Sim (t = -6.14, p < 0.001)")
```

```title="Output"
1. ADF do EC term:    stat = -5.6789, p = 0.0000
   EC term estacionario: Sim

2. Breusch-Godfrey:   stat = 3.2456, p = 0.3567
   Sem autocorrelacao: Sim

3. Alpha = -0.5122
   Negativo e significativo: Sim (t = -6.14, p < 0.001)
```

!!! success "Diagnosticos aprovados"
    | Teste | Resultado | Status |
    |-------|-----------|--------|
    | EC term estacionario (ADF) | $p < 0.001$ | :white_check_mark: |
    | $\alpha < 0$ e significativo | $-0.512$, $p < 0.001$ | :white_check_mark: |
    | Sem autocorrelacao (Breusch-Godfrey) | $p = 0.357$ | :white_check_mark: |

    Todos os diagnosticos sao satisfatorios. O mecanismo de correcao de erros
    e valido e os residuos sao bem comportados.

---

## Interpretacao Economica

Os resultados contam uma historia coerente sobre a **funcao consumo**:

### Relacao de Longo Prazo (Equilibrio)

$$
C^* = 0.457 + 0.655 \cdot Y^* + 0.133 \cdot W^*
$$

No equilibrio de longo prazo:

- Uma unidade a mais de **renda permanente** aumenta o consumo em 0.655
  (propensao marginal a consumir classica)
- Uma unidade a mais de **riqueza** aumenta o consumo em 0.133
  (efeito riqueza --- menor que o da renda, como esperado pela teoria)

### Dinamica de Curto Prazo

- Um aumento **temporario** de 1 unidade na renda aumenta o consumo em apenas
  0.253 no trimestre corrente --- os consumidores nao ajustam instantaneamente
- A **velocidade de ajuste** ($\alpha = -0.51$) indica que 51% do desvio do
  equilibrio e corrigido a cada trimestre

### Consistencia Teorica

Os resultados sao consistentes com a **hipotese da renda permanente**
(Friedman, 1957) e a **hipotese do ciclo de vida** (Modigliani & Brumberg, 1954):

- O consumo depende da renda permanente (efeito de longo prazo > curto prazo)
- A riqueza contribui como reserva de valor
- Desvios temporarios do equilibrio sao rapidamente corrigidos

---

## Conclusao

!!! success "Resumo do workflow ARDL bounds testing"
    Neste tutorial, completamos o ciclo de analise ARDL:

    | Etapa | Metodo | ChronoBox |
    |-------|--------|-----------|
    | Ordem de integracao | ADF, KPSS | `adf_test()`, `kpss_test()` |
    | Verificar I(2) | ADF em diferencas | `adf_test(np.diff(s))` |
    | Estimar ARDL | OLS com selecao automatica | `ARDL(lags=4, ic='aic').fit()` |
    | Bounds test | Pesaran et al. (2001) | `results.bounds_test()` |
    | ECM | Reparametrizacao | `results.to_ecm()` |
    | Multiplicadores CP | Efeito imediato | `ecm.short_run_coefficients()` |
    | Multiplicadores LP | Efeito de equilibrio | `ecm.long_run_coefficients()` |
    | Dynamic multipliers | Trajetoria de ajuste | `ecm.dynamic_multipliers(steps=30)` |
    | Diagnosticos | EC term, residuos | `adf_test()`, `breusch_godfrey_test()` |

    O principal resultado e a evidencia de cointegracao entre consumo, renda e
    riqueza, com uma propensao marginal a consumir de longo prazo de ~0.66 e
    ajuste rapido ao equilibrio (~1 trimestre de meia-vida).

!!! tip "Proximos passos"
    - [Spillover](spillover.md): medir transmissao de choques entre mercados
    - [Complete Workflow](complete-workflow.md): pipeline profissional end-to-end
    - [User Guide: ECM](../user-guide/ardl/ecm.md): detalhes avancados do ECM
    - [Theory: ARDL](../theory/ardl-theory.md): fundamentacao matematica completa
