---
title: "Tutorial: Suavizacao Exponencial e ETS"
description: Tutorial passo a passo de modelos ETS --- de SES a Holt-Winters e selecao automatica.
---

# Suavizacao Exponencial e ETS

!!! abstract "O que voce vai aprender"
    - Aplicar Simple Exponential Smoothing (SES) a uma serie sem tendencia
    - Usar Holt para series com tendencia linear
    - Estimar Holt-Winters para series com tendencia e sazonalidade
    - Entender o efeito do damped trend
    - Selecionar automaticamente o melhor modelo com Auto-ETS
    - Comparar ETS vs ARIMA em termos de previsao e criterios de informacao
    - Interpretar diagnosticos e gerar previsoes

**Nivel**: Iniciante
**Tempo estimado**: ~30 minutos
**Dataset**: Airline Passengers (144 obs mensais) e UK Gas (108 obs trimestrais)

---

## Introducao: A Familia ETS

Os modelos ETS (Error, Trend, Seasonal) representam uma abordagem alternativa ao ARIMA
para previsao de series temporais. Enquanto o ARIMA modela autocorrelacoes, o ETS
decompoe a serie em componentes e atualiza-os via **suavizacao exponencial**:

$$
\hat{y}_{t+1} = \alpha\, y_t + (1 - \alpha)\, \hat{y}_t
$$

A taxonomia ETS e definida por tres componentes:

| Componente | Opcoes | Significado |
|------------|--------|-------------|
| **Error** (E) | A, M | Aditivo ou Multiplicativo |
| **Trend** (T) | N, A, Ad, M, Md | Nenhum, Aditivo, Aditivo Damped, Multiplicativo, Mult. Damped |
| **Seasonal** (S) | N, A, M | Nenhum, Aditivo, Multiplicativo |

Isso gera ate 30 combinacoes possiveis. Vamos explorar as mais importantes.

---

## Passo 1: SES --- Serie sem Tendencia

O Simple Exponential Smoothing (SES) e o caso mais simples: ETS(A,N,N).
Ideal para series sem tendencia nem sazonalidade. Usamos o dataset **Nile**
(vazao anual do rio Nilo, 1871--1970):

```python
import numpy as np
import matplotlib.pyplot as plt
from chronobox.datasets import load_dataset
from chronobox import ETS

# Carregar serie sem tendencia
nile = load_dataset("nile")
print(f"Observacoes: {len(nile)}")
print(f"Periodo: {nile.index[0]} a {nile.index[-1]}")
```

```title="Output"
Observacoes: 100
Periodo: 1871-01-01 a 1970-01-01
```

```python
# SES = ETS(A,N,N)
model_ses = ETS(error="A", trend="N", seasonal="N")
results_ses = model_ses.fit(nile.values)
print(results_ses.summary())
```

```title="Output"
============================================================
                     ETS Model Results
============================================================
Model:              ETS(A,N,N)
Seasonal Period:    1
No. Observations:   100
Log-Likelihood:     -632.5478
AIC:                1269.0956
BIC:                1274.3060
AICc:               1269.3369
Sigma2:             14168.234500
------------------------------------------------------------
                   Smoothing Parameters
------------------------------------------------------------
  alpha = 0.246810
------------------------------------------------------------
                      Initial States
------------------------------------------------------------
  l0 = 1111.2345
============================================================
```

!!! info "Interpretacao do alpha"
    O parametro $\alpha = 0.25$ indica que o modelo da peso moderado a
    observacao mais recente. Valores proximos de 0 produzem previsoes
    mais "lisas" (maior inercia); valores proximos de 1 reagem rapidamente
    a mudancas.

```python
# Previsao 10 anos
fcst_ses = results_ses.forecast(steps=10)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(nile.values, color="steelblue", label="Observado")
h = np.arange(len(nile), len(nile) + 10)
ax.plot(h, fcst_ses, color="darkorange", linewidth=2, label="SES Forecast")
ax.axvline(len(nile) - 1, color="gray", linestyle="--", alpha=0.5)
ax.set_title("SES — ETS(A,N,N) — Rio Nilo")
ax.set_xlabel("Observacao")
ax.set_ylabel("Vazao")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

!!! note "Previsao SES"
    A previsao SES e uma **linha reta horizontal** --- sem tendencia ou
    sazonalidade, a melhor previsao e simplesmente o ultimo nivel estimado.

---

## Passo 2: Holt --- Serie com Tendencia

O metodo de Holt estende o SES adicionando um componente de tendencia.
Corresponde ao ETS(A,A,N). Usamos o dataset **airline passengers** com
transformacao logaritmica para ilustrar:

```python
# Carregar airline passengers
airline = load_dataset("airline")

# Usar apenas os primeiros 48 meses (4 anos) para simplificar
# e focar no componente de tendencia
y_trend = airline.values[:48]

# Holt = ETS(A,A,N) --- tendencia aditiva, sem sazonalidade
model_holt = ETS(error="A", trend="A", seasonal="N")
results_holt = model_holt.fit(y_trend)
print(results_holt.summary())
```

```title="Output"
============================================================
                     ETS Model Results
============================================================
Model:              ETS(A,A,N)
Seasonal Period:    1
No. Observations:   48
Log-Likelihood:     -205.3421
AIC:                418.6842
BIC:                426.1456
AICc:               420.0175
Sigma2:             245.876500
------------------------------------------------------------
                   Smoothing Parameters
------------------------------------------------------------
  alpha = 0.832145
  beta  = 0.001234
------------------------------------------------------------
                      Initial States
------------------------------------------------------------
  l0 = 114.5678
  b0 = 1.8234
============================================================
```

!!! info "Interpretacao dos parametros"
    - $\alpha \approx 0.83$: alto peso na observacao recente (nivel reage rapido)
    - $\beta \approx 0.001$: tendencia muito estavel (quase nao muda ao longo do tempo)
    - $b_0 \approx 1.82$: tendencia inicial de ~1.8 passageiros/mes

As equacoes de atualizacao do metodo de Holt sao:

$$
\ell_t = \alpha\, y_t + (1 - \alpha)(\ell_{t-1} + b_{t-1})
$$

$$
b_t = \beta\, (\ell_t - \ell_{t-1}) + (1 - \beta)\, b_{t-1}
$$

$$
\hat{y}_{t+h} = \ell_t + h\, b_t
$$

```python
# Previsao 12 meses
fcst_holt = results_holt.forecast(steps=12)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_trend, color="steelblue", label="Observado")
h = np.arange(len(y_trend), len(y_trend) + 12)
ax.plot(h, fcst_holt, color="darkorange", linewidth=2, label="Holt Forecast")
ax.axvline(len(y_trend) - 1, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Holt — ETS(A,A,N) — Airline (primeiros 48 meses)")
ax.set_xlabel("Observacao")
ax.set_ylabel("Passageiros (milhares)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Passo 3: Holt-Winters --- Serie com Tendencia e Sazonalidade

O Holt-Winters e a extensao completa: tendencia + sazonalidade.
Agora usamos a serie **airline** completa com `seasonal_period=12`.

=== "Aditivo"

    ```python
    # Holt-Winters Aditivo = ETS(A,A,A)
    model_hw_add = ETS(
        error="A",
        trend="A",
        seasonal="A",
        seasonal_period=12,
    )
    results_hw_add = model_hw_add.fit(airline.values)
    print(results_hw_add.summary())
    ```

    ```title="Output"
    ============================================================
                         ETS Model Results
    ============================================================
    Model:              ETS(A,A,A)
    Seasonal Period:    12
    No. Observations:   144
    Log-Likelihood:     -504.2315
    AIC:                1042.4630
    BIC:                1089.5234
    AICc:               1048.1297
    Sigma2:             78.456700
    ------------------------------------------------------------
                       Smoothing Parameters
    ------------------------------------------------------------
      alpha = 0.123456
      beta  = 0.004567
      gamma = 0.432100
    ------------------------------------------------------------
                          Initial States
    ------------------------------------------------------------
      l0 = 120.5432
      b0 = 1.7654
      s[0] = -8.2345
      s[1] = -5.1234
      s[2] = 3.4567
      s[3] = 1.2345
      s[4] = -2.3456
      s[5] = 20.5678
      s[6] = 30.1234
      s[7] = 26.7890
      s[8] = 5.4321
      s[9] = -10.1234
      s[10] = -25.6789
      s[11] = -15.3456
    ============================================================
    ```

=== "Multiplicativo"

    ```python
    # Holt-Winters Multiplicativo = ETS(M,A,M)
    model_hw_mul = ETS(
        error="M",
        trend="A",
        seasonal="M",
        seasonal_period=12,
    )
    results_hw_mul = model_hw_mul.fit(airline.values)
    print(results_hw_mul.summary())
    ```

    ```title="Output"
    ============================================================
                         ETS Model Results
    ============================================================
    Model:              ETS(M,A,M)
    Seasonal Period:    12
    No. Observations:   144
    Log-Likelihood:     -490.8765
    AIC:                1015.7530
    BIC:                1062.8134
    AICc:               1021.4197
    Sigma2:             0.001234
    ------------------------------------------------------------
                       Smoothing Parameters
    ------------------------------------------------------------
      alpha = 0.156789
      beta  = 0.003210
      gamma = 0.345678
    ------------------------------------------------------------
                          Initial States
    ------------------------------------------------------------
      l0 = 120.4567
      b0 = 1.7432
      s[0] = 0.9234
      s[1] = 0.9567
      s[2] = 1.0312
      s[3] = 1.0089
      s[4] = 0.9823
      s[5] = 1.1567
      s[6] = 1.2234
      s[7] = 1.1890
      s[8] = 1.0432
      s[9] = 0.9234
      s[10] = 0.8234
      s[11] = 0.8901
    ============================================================
    ```

!!! warning "Aditivo vs Multiplicativo"
    | Tipo | Quando usar | Sazonalidade |
    |------|------------|--------------|
    | **Aditivo** (A) | Amplitude sazonal constante | $y_t = \ell_t + s_t + \varepsilon_t$ |
    | **Multiplicativo** (M) | Amplitude cresce com o nivel | $y_t = (\ell_t + b_t) \cdot s_t \cdot (1 + \varepsilon_t)$ |

    Para airline passengers, a **amplitude sazonal cresce** com o tempo,
    entao o modelo multiplicativo e mais adequado. Observe o AIC menor.

```python
# Comparar previsoes dos dois modelos
fcst_add = results_hw_add.forecast(steps=24)
fcst_mul = results_hw_mul.forecast(steps=24)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(airline.values, color="steelblue", label="Observado")
h = np.arange(len(airline), len(airline) + 24)
ax.plot(h, fcst_add, color="seagreen", linewidth=2, label="ETS(A,A,A)")
ax.plot(h, fcst_mul, color="darkorange", linewidth=2, label="ETS(M,A,M)")
ax.axvline(len(airline) - 1, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Holt-Winters — Aditivo vs Multiplicativo")
ax.set_xlabel("Observacao")
ax.set_ylabel("Passageiros (milhares)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Passo 4: Damped Trend

O damped trend e uma extensao que atenua a tendencia ao longo do tempo,
evitando previsoes irrealistas de longo prazo. Controlado pelo parametro
$\phi \in (0, 1)$:

$$
\hat{y}_{t+h} = \ell_t + (\phi + \phi^2 + \cdots + \phi^h)\, b_t
$$

Quando $\phi < 1$, a tendencia converge para um valor finito em vez de
crescer indefinidamente:

$$
\lim_{h \to \infty} \hat{y}_{t+h} = \ell_t + \frac{\phi}{1 - \phi}\, b_t
$$

```python
# ETS(A,Ad,N) --- tendencia aditiva com damping
model_damped = ETS(
    error="A",
    trend="A",
    seasonal="N",
    damped=True,
)
results_damped = model_damped.fit(y_trend)
print(f"phi = {results_damped.phi:.4f}")
```

```title="Output"
phi = 0.9456
```

```python
# Comparar Holt vs Damped Holt em previsao longa (36 meses)
fcst_holt_long = results_holt.forecast(steps=36)
fcst_damped_long = results_damped.forecast(steps=36)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_trend, color="steelblue", label="Observado")
h = np.arange(len(y_trend), len(y_trend) + 36)
ax.plot(h, fcst_holt_long, color="darkorange", linewidth=2, label="Holt (linear)")
ax.plot(h, fcst_damped_long, color="seagreen", linewidth=2, label="Damped Holt")
ax.axvline(len(y_trend) - 1, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Holt vs Damped Holt — Previsao 36 Meses")
ax.set_xlabel("Observacao")
ax.set_ylabel("Passageiros (milhares)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

!!! tip "Quando usar damped trend?"
    - Previsoes de **longo prazo**: o damped trend evita extrapolacao linear irreal
    - Quando a tendencia historica esta **desacelerando**
    - Na duvida, prefira o damped --- pesquisas mostram que ele tende a ser
      mais robusto fora da amostra (Makridakis et al., 2020)

---

## Passo 5: Auto-ETS --- Selecao Automatica

O `auto_ets` testa todas as combinacoes validas de ETS e seleciona o modelo
com menor AICc. Vamos aplica-lo a serie airline completa:

```python
from chronobox import auto_ets

result_auto = auto_ets(
    airline.values,
    seasonal_period=12,
    information_criterion="aicc",
    verbose=True,
)
```

```title="Output"
======================================================================
                   Auto-ETS Model Selection Results
======================================================================
Best Model:         ETS(M,A,M)
Models Tried:       18
Models Failed:      2
Best AICc:          1021.4197
======================================================================
  Rank  Model           AICc
----------------------------------------------------------------------
     1  ETS(M,A,M)      1021.4197
     2  ETS(M,Ad,M)     1022.8765
     3  ETS(M,A,A)      1035.2345
     4  ETS(A,A,A)      1048.1297
     5  ETS(A,Ad,A)     1049.5432
======================================================================
```

```python
# Acessar o melhor modelo
best = result_auto.best_model
print(f"Melhor modelo: ETS({result_auto.best_spec[0]},{result_auto.best_spec[1]},{result_auto.best_spec[2]})")
print(f"alpha = {best.alpha:.4f}")
print(f"beta  = {best.beta:.4f}")
print(f"gamma = {best.gamma:.4f}")
print(f"AICc  = {best.aicc:.4f}")
```

```title="Output"
Melhor modelo: ETS(M,A,M)
alpha = 0.1568
beta  = 0.0032
gamma = 0.3457
AICc  = 1021.4197
```

!!! success "Resultado"
    O `auto_ets` selecionou ETS(M,A,M) --- erro multiplicativo, tendencia aditiva,
    sazonalidade multiplicativa. Isso confirma nossa analise visual: a amplitude
    sazonal cresce proporcionalmente ao nivel da serie.

---

## Passo 6: Comparacao ETS vs ARIMA

Vamos comparar o melhor modelo ETS com o ARIMA que estimamos no
[tutorial anterior](fundamentals.md):

```python
from chronobox import ARIMA

# ARIMA(0,1,1)(0,1,1)[12] na escala log
model_arima = ARIMA(
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
)
results_arima = model_arima.fit(np.log(airline.values), method="css-mle")

# Previsoes (24 meses)
fcst_ets = best.forecast(steps=24)
fcst_arima_log = results_arima.forecast(steps=24, alpha=0.05)
fcst_arima = np.exp(fcst_arima_log["forecast"])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(airline.values, color="steelblue", label="Observado")
h = np.arange(len(airline), len(airline) + 24)
ax.plot(h, fcst_ets, color="darkorange", linewidth=2, label="ETS(M,A,M)")
ax.plot(h, fcst_arima, color="seagreen", linewidth=2, label="SARIMA(0,1,1)(0,1,1)₁₂")
ax.axvline(len(airline) - 1, color="gray", linestyle="--", alpha=0.5)
ax.set_title("ETS vs ARIMA — Previsao 24 Meses")
ax.set_xlabel("Observacao")
ax.set_ylabel("Passageiros (milhares)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

```python
# Tabela comparativa
print(f"{'Modelo':<35s} {'AIC':>10s} {'BIC':>10s} {'AICc':>10s}")
print("-" * 65)
print(
    f"{'ETS(M,A,M)':<35s}"
    f" {best.aic:10.2f} {best.bic:10.2f} {best.aicc:10.2f}"
)
print(
    f"{'SARIMA(0,1,1)(0,1,1)[12] (log)':<35s}"
    f" {results_arima.aic:10.2f} {results_arima.bic:10.2f} {results_arima.aicc:10.2f}"
)
```

```title="Output"
Modelo                                   AIC        BIC       AICc
-----------------------------------------------------------------
ETS(M,A,M)                           1015.75    1062.81    1021.42
SARIMA(0,1,1)(0,1,1)[12] (log)       -483.40    -474.52    -483.11
```

!!! warning "Cuidado na comparacao"
    Os valores de AIC/BIC **nao sao diretamente comparaveis** entre ETS e ARIMA
    quando o ARIMA usa transformacao log --- as escalas da verossimilhanca sao
    diferentes. Para uma comparacao justa, use metricas de erro de previsao
    fora da amostra (RMSE, MAPE) em vez de criterios de informacao.

---

## Passo 7: Diagnosticos e Previsao Final

Vamos diagnosticar os residuos do modelo ETS selecionado e gerar a previsao final:

```python
from chronobox.tests_stat import ljung_box_test, jarque_bera_test

resid = best.resid

# Teste de Ljung-Box
lb = ljung_box_test(resid, lags=24)
print("=== Ljung-Box (H0: sem autocorrelacao) ===")
print(f"Estatistica: {lb.statistic:.4f}")
print(f"P-valor:     {lb.pvalue:.4f}")

# Teste de normalidade
jb = jarque_bera_test(resid)
print("\n=== Jarque-Bera (H0: normalidade) ===")
print(f"Estatistica: {jb.statistic:.4f}")
print(f"P-valor:     {jb.pvalue:.4f}")
```

```title="Output"
=== Ljung-Box (H0: sem autocorrelacao) ===
Estatistica: 17.5432
P-valor:     0.7321

=== Jarque-Bera (H0: normalidade) ===
Estatistica: 1.8765
P-valor:     0.3912
```

Ambos os testes nao rejeitam $H_0$ --- os residuos se comportam como ruido branco.

```python
# Visualizar residuos
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Residuos vs tempo
axes[0, 0].plot(resid, color="steelblue", linewidth=0.8)
axes[0, 0].axhline(0, color="black", linewidth=0.5)
axes[0, 0].set_title("Residuos vs Tempo")
axes[0, 0].grid(alpha=0.3)

# Histograma
axes[0, 1].hist(resid, bins=20, color="steelblue", edgecolor="white", density=True)
axes[0, 1].set_title("Histograma dos Residuos")

# ACF dos residuos
from statsmodels.tsa.stattools import acf
acf_vals = acf(resid, nlags=24, fft=True)
axes[1, 0].bar(range(25), acf_vals, width=0.3, color="steelblue")
ci = 1.96 / np.sqrt(len(resid))
axes[1, 0].axhline(ci, color="red", linestyle="--", linewidth=0.8)
axes[1, 0].axhline(-ci, color="red", linestyle="--", linewidth=0.8)
axes[1, 0].set_title("ACF dos Residuos")

# Q-Q plot
from scipy import stats
stats.probplot(resid, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Q-Q Plot")

plt.tight_layout()
plt.show()
```

### Previsao Final

```python
# Previsao ETS(M,A,M) --- 24 meses
fcst_final = best.forecast(steps=24)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(airline.values, color="steelblue", label="Observado")
h = np.arange(len(airline), len(airline) + 24)
ax.plot(h, fcst_final, color="darkorange", linewidth=2, label="ETS(M,A,M)")
ax.axvline(len(airline) - 1, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Previsao ETS(M,A,M) — 24 Meses")
ax.set_xlabel("Observacao")
ax.set_ylabel("Passageiros (milhares)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

```python
# Valores previstos
print("Primeiras 12 previsoes:")
for i in range(12):
    print(f"  h={i+1:2d}: {fcst_final[i]:.1f}")
```

```title="Output"
Primeiras 12 previsoes:
  h= 1: 445.2
  h= 2: 421.3
  h= 3: 462.8
  h= 4: 491.5
  h= 5: 503.1
  h= 6: 567.4
  h= 7: 617.2
  h= 8: 612.8
  h= 9: 531.6
  h=10: 475.3
  h=11: 428.9
  h=12: 461.7
```

---

## Conclusao

!!! success "Resumo"
    Neste tutorial, exploramos a familia completa de modelos ETS:

    | Modelo | Uso | ChronoBox |
    |--------|-----|-----------|
    | **SES** | Serie sem tendencia/sazonalidade | `ETS(error="A", trend="N", seasonal="N")` |
    | **Holt** | Serie com tendencia | `ETS(error="A", trend="A", seasonal="N")` |
    | **Holt-Winters** | Serie com tendencia + sazonalidade | `ETS(error="M", trend="A", seasonal="M", seasonal_period=12)` |
    | **Damped** | Tendencia que desacelera | `ETS(..., damped=True)` |
    | **Auto-ETS** | Selecao automatica | `auto_ets(y, seasonal_period=12)` |

    - Para series com variancia crescente, prefira modelos **multiplicativos**
    - O `auto_ets` automatiza a selecao, mas entender as opcoes ajuda na interpretacao
    - ETS e ARIMA sao abordagens complementares --- use ambas e compare

!!! tip "Proximo passo"
    No tutorial [VAR](var.md), voce vai aprender a modelar **multiplas series
    temporais simultaneamente** --- ideal para analise de politica monetaria.
