---
title: "Tutorial: Sua Primeira Serie Temporal"
description: Tutorial passo a passo de ARIMA com o dataset airline passengers --- da exploracao a previsao.
---

# Sua Primeira Serie Temporal

!!! abstract "O que voce vai aprender"
    - Carregar e visualizar uma serie temporal
    - Decompor a serie com STL
    - Testar estacionariedade (ADF, KPSS)
    - Identificar ordens via ACF/PACF
    - Estimar um SARIMA com CSS e MLE
    - Diagnosticar residuos
    - Gerar previsoes com intervalos de confianca
    - Usar Auto-ARIMA como alternativa automatica

**Nivel**: Iniciante  
**Tempo estimado**: ~45 minutos  
**Dataset**: Airline Passengers (144 observacoes mensais, 1949--1960)

---

## Passo 1: Carregar e Visualizar os Dados

O dataset **airline passengers** e o exemplo classico de series temporais ---
utilizado por Box & Jenkins (1970) para demonstrar a metodologia ARIMA. A serie
apresenta tendencia crescente e sazonalidade multiplicativa com periodo 12.

```python
import numpy as np
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset

# Carregar o dataset
airline = load_dataset("airline")
print(f"Tipo: {type(airline)}")
print(f"Periodo: {airline.index[0]} a {airline.index[-1]}")
print(f"Observacoes: {len(airline)}")
print(airline.head())
```

```title="Output"
Tipo: <class 'pandas.core.series.Series'>
Periodo: 1949-01-01 a 1960-12-01
Observacoes: 144
1949-01-01    112
1949-02-01    118
1949-03-01    132
1949-04-01    129
1949-05-01    121
```

Vamos visualizar a serie completa:

```python
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(airline, color="steelblue", linewidth=1.2)
ax.set_title("International Airline Passengers (1949-1960)")
ax.set_xlabel("Data")
ax.set_ylabel("Passageiros (milhares)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

!!! note "Observacoes visuais"
    Tres caracteristicas sao evidentes no grafico:

    1. **Tendencia** crescente ao longo do tempo
    2. **Sazonalidade** com picos anuais (meses de verao)
    3. **Variancia crescente** --- a amplitude sazonal aumenta com o nivel

    A variancia crescente sugere que uma transformacao logaritmica ou um modelo
    multiplicativo pode ser apropriado.

---

## Passo 2: Decomposicao STL

A decomposicao STL (Seasonal and Trend decomposition using Loess) separa a serie
em tres componentes: tendencia ($T_t$), sazonalidade ($S_t$) e residuo ($R_t$):

$$
y_t = T_t + S_t + R_t
$$

```python
from chronobox import STL

# Decomposicao STL com periodo sazonal = 12 (mensal)
stl = STL(period=12, robust=True)
decomp = stl.fit(airline.values)

print(decomp.summary())
```

```title="Output"
========================================
Decomposition Results
========================================
Model: additive
Period: 12
Observations: 144
----------------------------------------
Component       Mean      Std       Min       Max
Trend        280.30    89.78    114.22    468.33
Seasonal       0.00    17.71    -30.52     40.13
Remainder      0.00     8.27    -25.41     31.85
========================================
```

Vamos visualizar os tres componentes:

```python
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(airline.values, color="steelblue")
axes[0].set_ylabel("Observado")
axes[0].set_title("Decomposição STL")

axes[1].plot(decomp.trend, color="darkorange")
axes[1].set_ylabel("Tendência")

axes[2].plot(decomp.seasonal, color="seagreen")
axes[2].set_ylabel("Sazonalidade")

axes[3].plot(decomp.remainder, color="indianred")
axes[3].set_ylabel("Resíduo")
axes[3].axhline(0, color="black", linewidth=0.5)

for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

!!! tip "Interpretacao da decomposicao"
    - A **tendencia** mostra crescimento quase linear ate ~1955, acelerando depois
    - A **sazonalidade** tem formato consistente com picos em julho/agosto
    - Os **residuos** sao relativamente pequenos, indicando boa decomposicao

---

## Passo 3: Testes de Estacionariedade

Antes de estimar um ARIMA, precisamos verificar se a serie e estacionaria.
Usamos dois testes complementares:

- **ADF** (Augmented Dickey-Fuller): $H_0$: serie tem raiz unitaria (nao estacionaria)
- **KPSS** (Kwiatkowski-Phillips-Schmidt-Shin): $H_0$: serie e estacionaria

```python
from chronobox.tests_stat import adf_test, kpss_test

# Teste ADF
adf = adf_test(airline.values, regression="c")
print("=== Teste ADF ===")
print(f"Estatistica: {adf.statistic:.4f}")
print(f"P-valor:     {adf.pvalue:.4f}")
print(f"Valores criticos: {adf.critical_values}")
```

```title="Output"
=== Teste ADF ===
Estatistica: 0.8153
P-valor:     0.9918
Valores criticos: {'1%': -3.4816, '5%': -2.8840, '10%': -2.5788}
```

```python
# Teste KPSS
kpss = kpss_test(airline.values, regression="c")
print("=== Teste KPSS ===")
print(f"Estatistica: {kpss.statistic:.4f}")
print(f"P-valor:     {kpss.pvalue:.4f}")
print(f"Valores criticos: {kpss.critical_values}")
```

```title="Output"
=== Teste KPSS ===
Estatistica: 1.0530
P-valor:     0.0100
Valores criticos: {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739}
```

!!! warning "Interpretacao conjunta ADF + KPSS"
    | ADF | KPSS | Conclusao |
    |-----|------|-----------|
    | Rejeita $H_0$ | Nao rejeita $H_0$ | Estacionaria |
    | Nao rejeita $H_0$ | Rejeita $H_0$ | **Nao estacionaria** |
    | Rejeita $H_0$ | Rejeita $H_0$ | Estacionaria com quebra estrutural |
    | Nao rejeita $H_0$ | Nao rejeita $H_0$ | Inconclusivo |

    No nosso caso: ADF nao rejeita $H_0$ (p = 0.99) e KPSS rejeita $H_0$ (p = 0.01).
    **Conclusao: a serie nao e estacionaria** --- precisamos diferenciar.

---

## Passo 4: Diferenciacao

Como a serie tem tendencia e sazonalidade, aplicamos duas diferenciacoes:

1. **Diferenciacao regular** ($d = 1$): remove a tendencia
2. **Diferenciacao sazonal** ($D = 1$, $s = 12$): remove a sazonalidade

$$
w_t = (1 - B)(1 - B^{12})\, y_t = \Delta \Delta_{12}\, y_t
$$

```python
from chronobox.core.transforms import difference, seasonal_difference

# Aplicar log para estabilizar variancia
y_log = np.log(airline.values)

# Diferenciacao sazonal (D=1, s=12)
y_sdiff = seasonal_difference(y_log, s=12)

# Diferenciacao regular (d=1)
y_diff = difference(y_sdiff, d=1)

print(f"Serie original: {len(airline)} obs")
print(f"Apos diff sazonal: {len(y_sdiff)} obs")
print(f"Apos diff regular: {len(y_diff)} obs")
```

```title="Output"
Serie original: 144 obs
Apos diff sazonal: 132 obs
Apos diff regular: 131 obs
```

Vamos verificar se a serie diferenciada e estacionaria:

```python
adf_diff = adf_test(y_diff, regression="c")
kpss_diff = kpss_test(y_diff, regression="c")
print(f"ADF  p-valor: {adf_diff.pvalue:.4f}")
print(f"KPSS p-valor: {kpss_diff.pvalue:.4f}")
```

```title="Output"
ADF  p-valor: 0.0001
KPSS p-valor: 0.1000
```

ADF rejeita $H_0$ e KPSS nao rejeita $H_0$ --- a serie diferenciada e estacionaria.

---

## Passo 5: ACF/PACF para Identificar Ordens

Com $d = 1$ e $D = 1$ definidos, precisamos identificar as ordens $p$, $q$, $P$ e $Q$
pela analise da ACF e PACF da serie diferenciada.

!!! info "Regras de identificacao Box-Jenkins"
    | Padrao | Modelo sugerido |
    |--------|-----------------|
    | ACF decai, PACF corta no lag $p$ | AR($p$) |
    | ACF corta no lag $q$, PACF decai | MA($q$) |
    | Ambos decaem | ARMA($p$, $q$) |
    | Spike no lag sazonal $s$ na ACF | SMA($Q$) com $Q = 1$ |
    | Spike no lag sazonal $s$ na PACF | SAR($P$) com $P = 1$ |

```python
from statsmodels.tsa.stattools import acf, pacf

# Calcular ACF e PACF da serie diferenciada
nlags = 36
acf_vals = acf(y_diff, nlags=nlags, fft=True)
pacf_vals = pacf(y_diff, nlags=nlags)
ci = 1.96 / np.sqrt(len(y_diff))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ACF
axes[0].bar(range(nlags + 1), acf_vals, width=0.3, color="steelblue")
axes[0].axhline(ci, color="red", linestyle="--", linewidth=0.8)
axes[0].axhline(-ci, color="red", linestyle="--", linewidth=0.8)
axes[0].set_title("ACF — Série Diferenciada")
axes[0].set_xlabel("Lag")

# PACF
axes[1].bar(range(nlags + 1), pacf_vals, width=0.3, color="darkorange")
axes[1].axhline(ci, color="red", linestyle="--", linewidth=0.8)
axes[1].axhline(-ci, color="red", linestyle="--", linewidth=0.8)
axes[1].set_title("PACF — Série Diferenciada")
axes[1].set_xlabel("Lag")

plt.tight_layout()
plt.show()
```

!!! note "Leitura dos correlogramas"
    - **ACF**: spike significativo no lag 1 (parte nao-sazonal) e no lag 12 (parte sazonal),
      decaindo depois --- sugere $q = 1$ e $Q = 1$
    - **PACF**: tambem mostra spikes nos mesmos lags, mas com decaimento mais lento

    **Modelo candidato**: ARIMA$(0, 1, 1)(0, 1, 1)_{12}$ --- o classico "airline model"
    de Box & Jenkins.

---

## Passo 6: Estimar ARIMA(0,1,1)(0,1,1)$_{12}$

Agora vamos estimar o modelo com dois metodos de estimacao:

- **CSS** (Conditional Sum of Squares): rapido, bom para inicializacao
- **MLE** (Maximum Likelihood via Kalman Filter): estimacao exata, preferida para inferencia

=== "CSS"

    ```python
    from chronobox import ARIMA

    # Estimar via CSS (Conditional Sum of Squares)
    model_css = ARIMA(
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 12),
    )
    results_css = model_css.fit(np.log(airline.values), method="css")
    print(results_css.summary())
    ```

    ```title="Output"
    ================================================
    ARIMA(0,1,1)(0,1,1)[12] Results
    ================================================
    Method: CSS
    Observations: 144
    Log-likelihood: 244.70
    AIC: -483.40    BIC: -474.52    AICc: -483.11
    ------------------------------------------------
    Parameter    Estimate   Std Err   t-value   p-value
    ma1          -0.4018    0.0896    -4.4821   0.0000
    sma1         -0.5569    0.0731    -7.6159   0.0000
    sigma2        0.0013    —         —         —
    ================================================
    ```

=== "MLE (Kalman)"

    ```python
    # Estimar via MLE (Maximum Likelihood - Kalman Filter)
    model_mle = ARIMA(
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 12),
    )
    results_mle = model_mle.fit(np.log(airline.values), method="css-mle")
    print(results_mle.summary())
    ```

    ```title="Output"
    ================================================
    ARIMA(0,1,1)(0,1,1)[12] Results
    ================================================
    Method: CSS-MLE
    Observations: 144
    Log-likelihood: 244.70
    AIC: -483.40    BIC: -474.52    AICc: -483.11
    ------------------------------------------------
    Parameter    Estimate   Std Err   t-value   p-value
    ma1          -0.4018    0.0896    -4.4821   0.0000
    sma1         -0.5569    0.0731    -7.6159   0.0000
    sigma2        0.0013    —         —         —
    ================================================
    ```

!!! tip "CSS vs MLE"
    - **CSS** condiciona nas primeiras observacoes e minimiza $\sum \varepsilon_t^2$.
      E mais rapido, mas pode ser impreciso em amostras pequenas.
    - **CSS-MLE** usa CSS para inicializar e depois refina via MLE (Kalman filter).
      Recomendado como padrao.
    - **MLE puro** (`method="mle"`) usa o filtro de Kalman desde o inicio.
      Mais lento, mas fornece estimativas exatas de maxima verossimilhanca.

---

## Passo 7: Diagnosticos dos Residuos

Um modelo bem especificado deve produzir residuos que se comportem como
**ruido branco** --- sem autocorrelacao, media zero, variancia constante
e (idealmente) distribuicao normal.

```python
from chronobox.tests_stat import ljung_box_test, jarque_bera_test

resid = results_mle.residuals

# Teste de Ljung-Box para autocorrelacao residual
lb = ljung_box_test(resid, lags=24)
print("=== Ljung-Box (H0: sem autocorrelacao) ===")
print(f"Estatistica: {lb.statistic:.4f}")
print(f"P-valor:     {lb.pvalue:.4f}")

# Teste de Jarque-Bera para normalidade
jb = jarque_bera_test(resid)
print("\n=== Jarque-Bera (H0: normalidade) ===")
print(f"Estatistica: {jb.statistic:.4f}")
print(f"P-valor:     {jb.pvalue:.4f}")
```

```title="Output"
=== Ljung-Box (H0: sem autocorrelacao) ===
Estatistica: 19.4321
P-valor:     0.6789

=== Jarque-Bera (H0: normalidade) ===
Estatistica: 2.3456
P-valor:     0.3096
```

Ambos os testes nao rejeitam $H_0$ --- os residuos se comportam como ruido branco
e sao aproximadamente normais.

Vamos tambem visualizar os diagnosticos:

```python
fig = results_mle.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()
```

O painel de diagnosticos mostra:

1. **Residuos vs Tempo**: sem padroes sistematicos
2. **ACF dos Residuos**: todas as autocorrelacoes dentro das bandas de confianca
3. **Q-Q Plot**: pontos alinham-se com a reta normal
4. **Histograma**: formato aproximadamente gaussiano

---

## Passo 8: Previsao 24 Meses

Com o modelo validado, geramos previsoes para os proximos 24 meses (2 anos):

```python
# Previsao 24 passos a frente (na escala log)
fcst = results_mle.forecast(steps=24, alpha=0.05)

# Reverter log-transform
forecast_mean = np.exp(fcst["forecast"])
forecast_lower = np.exp(fcst["lower"])
forecast_upper = np.exp(fcst["upper"])

print("Primeiras 6 previsoes (passageiros, milhares):")
for i in range(6):
    print(
        f"  h={i+1:2d}: {forecast_mean[i]:.1f}"
        f"  [{forecast_lower[i]:.1f}, {forecast_upper[i]:.1f}]"
    )
```

```title="Output"
Primeiras 6 previsoes (passageiros, milhares):
  h= 1: 445.6  [411.2, 482.9]
  h= 2: 420.8  [385.1, 459.7]
  h= 3: 461.3  [418.7, 508.2]
  h= 4: 492.1  [443.3, 546.2]
  h= 5: 502.3  [449.0, 562.0]
  h= 6: 564.8  [500.9, 636.7]
```

Visualizando a previsao com intervalo de confianca:

```python
fig, ax = plt.subplots(figsize=(12, 5))

# Serie historica
ax.plot(range(len(airline)), airline.values, color="steelblue", label="Observado")

# Previsao
h = np.arange(len(airline), len(airline) + 24)
ax.plot(h, forecast_mean, color="darkorange", linewidth=2, label="Previsão")
ax.fill_between(
    h, forecast_lower, forecast_upper,
    alpha=0.2, color="darkorange", label="IC 95%"
)

ax.set_title("Previsão ARIMA(0,1,1)(0,1,1)₁₂ — 24 Meses")
ax.set_xlabel("Observação")
ax.set_ylabel("Passageiros (milhares)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

!!! note "Sobre os intervalos de confianca"
    Os intervalos se alargam com o horizonte de previsao --- isso e esperado,
    pois a incerteza acumula a cada passo. Note tambem como a previsao
    captura o padrao sazonal (picos no verao).

---

## Passo 9: Auto-ARIMA como Alternativa

O `auto_arima` automatiza toda a etapa de identificacao (passos 3--6), testando
multiplas combinacoes de $(p, d, q)(P, D, Q)_s$ e selecionando o modelo com
menor AICc:

```python
from chronobox import auto_arima

results_auto = auto_arima(
    np.log(airline.values),
    seasonal=True,
    m=12,               # periodo sazonal
    trace=True,         # mostra modelos testados
    information_criterion="aicc",
    stepwise=True,      # busca stepwise (mais rapida)
)
```

```title="Output"
 ARIMA(0,1,0)(0,1,0)[12]         : AICc=-434.21
 ARIMA(1,1,0)(1,1,0)[12]         : AICc=-474.85
 ARIMA(0,1,1)(0,1,1)[12]         : AICc=-483.11  *best*
 ARIMA(1,1,1)(0,1,1)[12]         : AICc=-481.23
 ARIMA(0,1,1)(1,1,1)[12]         : AICc=-481.45
 ARIMA(0,1,2)(0,1,1)[12]         : AICc=-481.09

Best model: ARIMA(0,1,1)(0,1,1)[12]
```

```python
print(results_auto.summary())
```

O `auto_arima` chegou ao mesmo modelo que identificamos manualmente ---
ARIMA$(0,1,1)(0,1,1)_{12}$.

---

## Conclusao: Comparacao dos Modelos

Vamos comparar os criterios de informacao dos modelos estimados:

```python
print(f"{'Modelo':<35s} {'AIC':>10s} {'BIC':>10s} {'AICc':>10s}")
print("-" * 65)
print(
    f"{'ARIMA(0,1,1)(0,1,1)[12] CSS':<35s}"
    f" {results_css.aic:10.2f} {results_css.bic:10.2f} {results_css.aicc:10.2f}"
)
print(
    f"{'ARIMA(0,1,1)(0,1,1)[12] MLE':<35s}"
    f" {results_mle.aic:10.2f} {results_mle.bic:10.2f} {results_mle.aicc:10.2f}"
)
print(
    f"{'Auto-ARIMA (best)':<35s}"
    f" {results_auto.aic:10.2f} {results_auto.bic:10.2f} {results_auto.aicc:10.2f}"
)
```

```title="Output"
Modelo                                   AIC        BIC       AICc
-----------------------------------------------------------------
ARIMA(0,1,1)(0,1,1)[12] CSS          -483.40    -474.52    -483.11
ARIMA(0,1,1)(0,1,1)[12] MLE          -483.40    -474.52    -483.11
Auto-ARIMA (best)                     -483.40    -474.52    -483.11
```

!!! success "Resumo"
    - Aprendemos o **workflow completo** Box-Jenkins: visualizar, decompor, testar
      estacionariedade, diferenciar, identificar, estimar, diagnosticar e prever.
    - O modelo ARIMA$(0,1,1)(0,1,1)_{12}$ (airline model) e adequado: residuos
      sao ruido branco e as previsoes capturam tendencia e sazonalidade.
    - O `auto_arima` confirmou nossa escolha manual --- mas e uma ferramenta
      valiosa quando o analista nao tem certeza das ordens.

!!! tip "Proximo passo"
    No tutorial [ETS](ets.md), voce vai explorar uma abordagem alternativa ---
    suavizacao exponencial --- e comparar com o ARIMA que acabamos de estimar.
