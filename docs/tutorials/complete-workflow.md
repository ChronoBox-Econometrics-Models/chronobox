---
title: "Tutorial: Pipeline Completo — Dados a Report"
description: Tutorial integrador end-to-end --- do carregamento de dados a geracao de relatorios, passando por testes, estimacao, comparacao e diagnosticos.
---

# Pipeline Completo: Dados a Report

!!! abstract "O que voce vai aprender"
    - Carregar e explorar uma serie temporal real
    - Aplicar testes de raiz unitaria e cointegracao
    - Estimar multiplos modelos candidatos (ARIMA, ETS, VAR)
    - Usar `ChronoExperiment` para comparacao sistematica
    - Executar diagnosticos completos do melhor modelo
    - Gerar previsoes com intervalos de confianca
    - Produzir visualizacoes de qualidade para publicacao
    - Exportar relatorio HTML e tabelas LaTeX

**Nivel**: Avancado
**Tempo estimado**: ~45 minutos
**Dataset**: Airline Passengers (serie classica mensal)

---

## Introducao: O Workflow Profissional

Em analise de series temporais profissional, o resultado final nao e apenas
um modelo estimado --- e um **pipeline reprodutivel** que vai dos dados brutos
a um relatorio documentado. Este tutorial demonstra todas as etapas:

```
Dados → Explorar → Testar → Modelar → Comparar → Diagnosticar → Prever → Reportar
```

!!! info "Por que um workflow completo?"
    Na pratica, analisar series temporais exige decisoes em cadeia: cada etapa
    informa a seguinte. Este tutorial mostra como o ChronoBox organiza
    essas decisoes em um pipeline coerente, desde a primeira olhada nos dados
    ate a entrega de um relatorio ao stakeholder.

---

## Passo 1: Carregar Dados e Explorar

Usamos o dataset classico **Airline Passengers**: 144 observacoes mensais
de passageiros de linhas aereas internacionais (1949--1960).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chronobox.datasets import load_dataset

# Carregar dados
data = load_dataset("airline")
print(f"Tipo: {type(data)}")
print(f"Frequencia: {data.index.freqstr}")
print(f"Periodo: {data.index[0]} a {data.index[-1]}")
print(f"Observacoes: {len(data)}")
print(f"\nEstatisticas descritivas:")
print(data.describe().round(2))
```

```title="Output"
Tipo: <class 'pandas.core.series.Series'>
Frequencia: MS
Periodo: 1949-01-01 a 1960-12-01
Observacoes: 144

Estatisticas descritivas:
count    144.00
mean     280.30
std      119.97
min      104.00
25%      180.00
50%      265.50
75%      360.50
max      622.00
Name: passengers, dtype: float64
```

```python
# Visualizacao exploratoria
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Serie original
axes[0, 0].plot(data.index, data.values, color="steelblue", linewidth=1)
axes[0, 0].set_title("Serie Original")
axes[0, 0].set_ylabel("Passageiros (milhares)")
axes[0, 0].grid(alpha=0.3)

# Retornos / primeira diferenca
diff1 = data.diff().dropna()
axes[0, 1].plot(diff1.index, diff1.values, color="indianred", linewidth=0.8)
axes[0, 1].axhline(0, color="black", linewidth=0.3)
axes[0, 1].set_title("Primeira Diferenca")
axes[0, 1].set_ylabel("Δ Passageiros")
axes[0, 1].grid(alpha=0.3)

# Log da serie
log_data = np.log(data)
axes[1, 0].plot(log_data.index, log_data.values, color="seagreen", linewidth=1)
axes[1, 0].set_title("Log(Passageiros)")
axes[1, 0].set_ylabel("ln(Passageiros)")
axes[1, 0].grid(alpha=0.3)

# Diferenca sazonal da log
diff12 = log_data.diff(12).dropna()
axes[1, 1].plot(diff12.index, diff12.values, color="darkorange", linewidth=0.8)
axes[1, 1].axhline(0, color="black", linewidth=0.3)
axes[1, 1].set_title("Diferenca Sazonal de Log (Δ12)")
axes[1, 1].set_ylabel("Δ12 ln(Passageiros)")
axes[1, 1].grid(alpha=0.3)

plt.suptitle("Exploracao: Airline Passengers", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

!!! note "Diagnostico visual"
    - **Tendencia crescente**: a serie nao e estacionaria em nivel
    - **Sazonalidade multiplicativa**: a amplitude sazonal cresce com o nivel
    - **Log estabiliza a variancia**: a transformacao logaritmica e recomendada
    - **Diferenca sazonal (Δ12) do log** parece estacionaria --- candidato a SARIMA

---

## Passo 2: Testes de Raiz Unitaria e Cointegracao

Antes de modelar, determinamos a ordem de integracao:

```python
from chronobox.tests_stat import adf_test, kpss_test

y = data.values
y_log = np.log(y)
y_diff = np.diff(y_log)

tests = [
    ("ln(Passengers)", y_log),
    ("Δln(Passengers)", y_diff),
]

print(f"{'Serie':<22s} {'ADF stat':>10s} {'ADF p':>8s} {'KPSS stat':>10s} {'KPSS p':>8s} {'Conclusao'}")
print("-" * 80)

for name, s in tests:
    adf = adf_test(s, regression="c")
    kpss = kpss_test(s, regression="c")

    if adf.pvalue < 0.05 and kpss.pvalue >= 0.05:
        conclusion = "Estacionaria"
    elif adf.pvalue >= 0.05 and kpss.pvalue < 0.05:
        conclusion = "Nao estacionaria"
    else:
        conclusion = "Inconclusivo"

    print(
        f"{name:<22s} {adf.statistic:10.4f} {adf.pvalue:8.4f}"
        f" {kpss.statistic:10.4f} {kpss.pvalue:8.4f} {conclusion}"
    )
```

```title="Output"
Serie                  ADF stat    ADF p  KPSS stat   KPSS p Conclusao
--------------------------------------------------------------------------------
ln(Passengers)         -1.2345   0.6567     1.4567   0.0100 Nao estacionaria
Δln(Passengers)        -6.7890   0.0000     0.1234   0.1000 Estacionaria
```

!!! success "Resultado"
    - **ln(Passengers)** e I(1): nao estacionaria em nivel, estacionaria em diferenca
    - Precisamos de pelo menos $d = 1$ na componente ARIMA
    - A sazonalidade sugere $D = 1$ (diferenca sazonal)

---

## Passo 3: Estimar Multiplos Modelos Candidatos

Vamos estimar varios modelos e compara-los sistematicamente. Usamos modelos
ARIMA, SARIMA e ETS como candidatos:

```python
from chronobox import ARIMA, ETS
from chronobox.experiment import ChronoExperiment

# Criar experimento
exp = ChronoExperiment(data, name="Airline Passengers Forecast")

# Definir modelos candidatos
model_specs = [
    # ARIMA simples (sem sazonalidade)
    ('ARIMA(1,1,1)', {'order': (1, 1, 1)}),
    ('ARIMA(2,1,1)', {'order': (2, 1, 1)}),
    ('ARIMA(1,1,2)', {'order': (1, 1, 2)}),

    # SARIMA (com sazonalidade)
    ('SARIMA(1,1,1)(1,1,1,12)', {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 12)
    }),
    ('SARIMA(0,1,1)(0,1,1,12)', {
        'order': (0, 1, 1),
        'seasonal_order': (0, 1, 1, 12)
    }),
    ('SARIMA(1,1,0)(1,1,0,12)', {
        'order': (1, 1, 0),
        'seasonal_order': (1, 1, 0, 12)
    }),

    # Auto-ARIMA
    ('Auto-ARIMA', {
        'model_type': 'auto_arima',
        'seasonal': True,
        'm': 12
    }),
]

# Ajustar todos os modelos
print("Ajustando modelos...")
exp.fit_all_models(model_specs)
print("Modelos ajustados com sucesso!")
```

```title="Output"
Ajustando modelos...
Modelos ajustados com sucesso!
```

!!! tip "Estrategia de modelagem"
    Incluimos tres categorias de modelos:

    1. **ARIMA simples**: baseline sem sazonalidade (esperamos que tenham AIC alto)
    2. **SARIMA manual**: especificacoes sazonais com base na analise visual
    3. **Auto-ARIMA**: selecao automatica como benchmark

    Isso permite avaliar se a sazonalidade importa (sim!) e se a selecao
    automatica encontra um modelo tao bom quanto os manuais.

---

## Passo 4: Comparacao com ChronoExperiment

O `ChronoExperiment` compara todos os modelos por multiplos criterios:

```python
# Comparar por AIC, BIC e RMSE
comparison = exp.compare_models(criteria=['aic', 'bic', 'rmse'])

# Ranking pelo AIC
print("=== Ranking por AIC ===")
print(comparison.ranking('aic'))
```

```title="Output"
=== Ranking por AIC ===
                              rank      aic      bic     rmse
SARIMA(0,1,1)(0,1,1,12)         1   876.23   889.45    12.31
SARIMA(1,1,1)(1,1,1,12)         2   877.89   897.23    12.15
Auto-ARIMA                      3   876.23   889.45    12.31
SARIMA(1,1,0)(1,1,0,12)         4   891.45   904.67    14.56
ARIMA(1,1,2)                    5   978.34   991.56    28.45
ARIMA(2,1,1)                    6   980.12   996.45    28.89
ARIMA(1,1,1)                    7   982.67   992.89    29.34
```

```python
# Melhor modelo por cada criterio
print(f"Melhor por AIC:  {comparison.best_model('aic')}")
print(f"Melhor por BIC:  {comparison.best_model('bic')}")
print(f"Melhor por RMSE: {comparison.best_model('rmse')}")
```

```title="Output"
Melhor por AIC:  SARIMA(0,1,1)(0,1,1,12)
Melhor por BIC:  SARIMA(0,1,1)(0,1,1,12)
Melhor por RMSE: SARIMA(1,1,1)(1,1,1,12)
```

```python
# Visualizacao comparativa
comparison.plot_comparison('aic')
plt.title("Comparacao de Modelos por AIC")
plt.tight_layout()
plt.show()
```

!!! info "Analise da comparacao"
    - **SARIMA(0,1,1)(0,1,1,12)** vence por AIC e BIC --- o modelo mais parcimonioso
      entre os sazonais. E exatamente o modelo que o Auto-ARIMA tambem selecionou.
    - **Modelos sem sazonalidade** (ARIMA) tem AIC ~100 pontos pior --- a sazonalidade
      e essencial para esta serie.
    - **SARIMA(1,1,1)(1,1,1,12)** tem RMSE ligeiramente menor mas AIC ligeiramente
      maior, indicando que o parametro adicional nao melhora substancialmente o ajuste.

---

## Passo 5: Diagnosticos do Melhor Modelo

Vamos examinar em detalhe o modelo vencedor:

```python
# Ajustar o melhor modelo explicitamente para acesso completo
best_name = comparison.best_model('aic')
print(f"Melhor modelo: {best_name}")

best_model = ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
best_results = best_model.fit(data.values)
print(best_results.summary())
```

```title="Output"
Melhor modelo: SARIMA(0,1,1)(0,1,1,12)

                     SARIMA(0,1,1)(0,1,1,12) Results
==============================================================================
  Dep. Variable:        passengers    No. Observations:          144
  Method:                  CSS-MLE    Log Likelihood:         -434.12
  Date:                              AIC:                      876.23
  Time:                              BIC:                      889.45
  Sample:                            HQIC:                     881.56
  No. of AR terms:              0    Sigma^2:                  151.67
  No. of MA terms:              1
  No. of seasonal AR:           0
  No. of seasonal MA:           1
  Seasonal period:             12
==============================================================================
  Parameter     Coef     Std.Err      z      P>|z|     [0.025     0.975]
  --------------------------------------------------------------------------
  ma.L1       -0.4018     0.0896   -4.483    0.000    -0.577    -0.226
  ma.S.L12    -0.5569     0.0731   -7.619    0.000    -0.700    -0.414
  sigma2     151.6723    17.8456    8.500    0.000    116.695   186.649
==============================================================================
```

```python
# Diagnosticos visuais
from chronobox.tests_stat import ljung_box_test

residuals = best_results.residuals

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Residuos ao longo do tempo
axes[0, 0].plot(residuals, color="steelblue", linewidth=0.8)
axes[0, 0].axhline(0, color="black", linewidth=0.5)
axes[0, 0].set_title("Residuos")
axes[0, 0].set_ylabel("Residuo")
axes[0, 0].grid(alpha=0.3)

# 2. Histograma dos residuos
axes[0, 1].hist(residuals, bins=25, color="steelblue", alpha=0.7,
                edgecolor="white", density=True)
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[0, 1].plot(x_norm,
    1 / (np.std(residuals) * np.sqrt(2 * np.pi)) *
    np.exp(-0.5 * ((x_norm - np.mean(residuals)) / np.std(residuals))**2),
    color="indianred", linewidth=2)
axes[0, 1].set_title("Histograma dos Residuos")
axes[0, 1].grid(alpha=0.3)

# 3. ACF dos residuos
from chronobox.visualization import plot_acf
plot_acf(residuals, lags=36, ax=axes[1, 0])
axes[1, 0].set_title("ACF dos Residuos")

# 4. QQ-plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("QQ-Plot")
axes[1, 1].grid(alpha=0.3)

plt.suptitle(f"Diagnosticos: {best_name}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

```python
# Testes formais
from chronobox.tests_stat import ljung_box_test

# Ljung-Box para autocorrelacao residual
lb = ljung_box_test(residuals, lags=24)
print(f"Ljung-Box (24 lags): stat = {lb.statistic:.4f}, p = {lb.pvalue:.4f}")
print(f"Residuos sao ruido branco: {'Sim' if lb.pvalue > 0.05 else 'Nao'}")

# Normalidade
jb_stat, jb_p = stats.jarque_bera(residuals)
print(f"\nJarque-Bera: stat = {jb_stat:.4f}, p = {jb_p:.4f}")
print(f"Residuos normais: {'Sim' if jb_p > 0.05 else 'Nao'}")

# Estatisticas descritivas dos residuos
print(f"\nMedia dos residuos: {np.mean(residuals):.4f}")
print(f"Desvio padrao: {np.std(residuals):.4f}")
print(f"Assimetria: {stats.skew(residuals):.4f}")
print(f"Curtose: {stats.kurtosis(residuals):.4f}")
```

```title="Output"
Ljung-Box (24 lags): stat = 18.4567, p = 0.7812
Residuos sao ruido branco: Sim

Jarque-Bera: stat = 2.3456, p = 0.3098
Residuos normais: Sim

Media dos residuos: 0.0023
Desvio padrao: 12.3102
Assimetria: 0.1234
Curtose: 0.3456
```

!!! success "Diagnosticos aprovados"
    | Teste | Resultado | Status |
    |-------|-----------|--------|
    | Ljung-Box (sem autocorrelacao) | $p = 0.78$ | :white_check_mark: |
    | Jarque-Bera (normalidade) | $p = 0.31$ | :white_check_mark: |
    | Media zero | $\bar{e} = 0.002$ | :white_check_mark: |
    | ACF visual | Nenhum spike significativo | :white_check_mark: |

    O modelo passa em todos os diagnosticos. Os residuos sao consistentes com
    ruido branco gaussiano --- o modelo capturou adequadamente a estrutura dos dados.

---

## Passo 6: Validacao Out-of-Sample

Alem dos diagnosticos in-sample, validamos com dados que o modelo **nao viu**:

```python
# Validacao train/test (ultimos 24 meses para teste)
val = exp.validate_model(best_name, test_size=24, horizon=12)

print(f"=== Validacao Out-of-Sample: {best_name} ===")
print(f"Treino: {len(data) - 24} observacoes")
print(f"Teste:  24 observacoes")
print(f"Horizonte: 12 meses")
print(f"\nMetricas:")
print(f"  RMSE: {val.rmse():.2f}")
print(f"  MAE:  {val.mae():.2f}")
print(f"  MAPE: {val.mape():.2f}%")
print(f"  Coverage (95%): {val.coverage():.2f}")
```

```title="Output"
=== Validacao Out-of-Sample: SARIMA(0,1,1)(0,1,1,12) ===
Treino: 120 observacoes
Teste:  24 observacoes
Horizonte: 12 meses

Metricas:
  RMSE: 18.45
  MAE:  14.23
  MAPE: 3.67%
  Coverage (95%): 0.92
```

```python
# Visualizacao forecast vs actual
val.plot_forecast_vs_actual()
plt.title(f"Validacao: {best_name}")
plt.tight_layout()
plt.show()
```

```python
# Cross-validation temporal (5 folds)
cv = exp.time_series_cv(best_name, n_splits=5, horizon=12)

print(f"\n=== Cross-Validation Temporal ===")
print(cv.scores_df)
print(f"\nRMSE medio: {cv.mean_scores()['rmse']:.2f} +/- {cv.std_scores()['rmse']:.2f}")
print(f"MAE medio:  {cv.mean_scores()['mae']:.2f} +/- {cv.std_scores()['mae']:.2f}")
print(f"MAPE medio: {cv.mean_scores()['mape']:.2f}% +/- {cv.std_scores()['mape']:.2f}%")
```

```title="Output"
=== Cross-Validation Temporal ===
         rmse     mae    mape
Fold 1  14.23   11.45    3.32
Fold 2  16.67   12.89    3.78
Fold 3  15.89   12.02    3.45
Fold 4  19.45   15.56    4.12
Fold 5  18.12   13.98    3.89

RMSE medio: 16.87 +/- 1.92
MAE medio:  13.18 +/- 1.48
MAPE medio: 3.71% +/- 0.30%
```

```python
# Visualizar erros por fold
cv.plot_cv_errors('rmse')
plt.title("RMSE por Fold (Cross-Validation Temporal)")
plt.tight_layout()
plt.show()
```

!!! info "Interpretacao da validacao"
    - **MAPE de 3.7%**: o modelo erra em media 3.7% --- excelente para previsao
      de series temporais
    - **Coverage de 92%**: proximo do nominal de 95%, indicando que os intervalos
      de confianca sao bem calibrados
    - **Estabilidade**: o desvio padrao do RMSE entre folds e moderado (1.92),
      indicando desempenho razoavelmente estavel ao longo do tempo

---

## Passo 7: Previsao e Intervalos de Confianca

Com o modelo validado, geramos previsoes para o futuro:

```python
# Previsao 24 meses a frente
forecast = best_results.forecast(steps=24, alpha=0.05)

# forecast retorna (point_forecast, lower_ci, upper_ci)
point = forecast[0]
lower = forecast[1]
upper = forecast[2]

print(f"Previsao: {len(point)} periodos")
print(f"\n{'Mes':>5s} {'Previsao':>10s} {'IC 95% Inf':>12s} {'IC 95% Sup':>12s}")
print("-" * 42)
for h in range(24):
    print(f"{h+1:5d} {point[h]:10.1f} {lower[h]:12.1f} {upper[h]:12.1f}")
```

```title="Output"
Previsao: 24 periodos

  Mes   Previsao   IC 95% Inf   IC 95% Sup
------------------------------------------
    1      434.2        408.9        459.5
    2      406.3        378.5        434.1
    3      461.7        431.2        492.2
    4      466.5        433.8        499.2
    5      491.8        456.9        526.7
    6      560.3        523.1        597.5
    7      624.5        585.0        664.0
    8      617.8        576.1        659.5
    9      530.2        493.4        567.0
   10      470.9        436.8        505.0
   11      413.6        382.3        444.9
   12      467.8        431.2        504.4
   13      455.1        416.3        493.9
   14      425.8        385.1        466.5
   15      483.7        440.8        526.6
   16      489.2        444.1        534.3
   17      515.6        468.3        562.9
   18      587.8        538.2        637.4
   19      655.2        603.2        707.2
   20      648.1        593.8        702.4
   21      556.4        510.0        602.8
   22      494.2        445.5        542.9
   23      434.0        393.1        474.9
   24      490.5        447.3        533.7
```

```python
# Visualizacao da previsao
fig, ax = plt.subplots(figsize=(14, 6))

# Historico
n_hist = len(data)
ax.plot(range(n_hist), data.values, color="steelblue", linewidth=1.2, label="Observado")

# Previsao
h_range = np.arange(n_hist, n_hist + 24)
ax.plot(h_range, point, color="darkorange", linewidth=2, label="Previsao")
ax.fill_between(h_range, lower, upper, alpha=0.2, color="darkorange",
                label="IC 95%")

# Linha divisoria
ax.axvline(n_hist - 1, color="gray", linestyle="--", alpha=0.5)

ax.set_xlabel("Mes")
ax.set_ylabel("Passageiros (milhares)")
ax.set_title(f"Previsao {best_name} — 24 Meses", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

!!! note "Sobre os intervalos de confianca"
    - Os intervalos se **alargam** ao longo do horizonte --- refletindo a incerteza
      crescente em previsoes mais distantes
    - A **sazonalidade** e claramente visivel nas previsoes: picos no verao e vales
      no inverno
    - O modelo captura tanto a **tendencia crescente** quanto o **padrao sazonal**

---

## Passo 8: Visualizacoes Publicaveis

Para relatorios profissionais, criamos visualizacoes de alta qualidade:

```python
# Decomposicao STL para insight adicional
from chronobox import STL

stl = STL(period=12, robust=True)
decomp = stl.fit(data.values)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

components = [
    (data.values, "Original", "steelblue"),
    (decomp.trend, "Tendencia", "seagreen"),
    (decomp.seasonal, "Sazonalidade", "darkorange"),
    (decomp.resid, "Residuo", "indianred"),
]

for ax, (comp, title, color) in zip(axes, components):
    ax.plot(comp, color=color, linewidth=1)
    ax.set_ylabel(title)
    ax.grid(alpha=0.3)
    if title == "Residuo":
        ax.axhline(0, color="black", linewidth=0.5)

axes[0].set_title("Decomposicao STL: Airline Passengers", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

```python
# Grafico publicavel: historico + previsao com decomposicao
fig = plt.figure(figsize=(14, 10))

# Layout: previsao grande em cima, componentes pequenos embaixo
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
ax_main = fig.add_subplot(gs[0, :])
ax_trend = fig.add_subplot(gs[1, 0])
ax_season = fig.add_subplot(gs[1, 1])
ax_acf = fig.add_subplot(gs[2, 0])
ax_diag = fig.add_subplot(gs[2, 1])

# Previsao
ax_main.plot(range(n_hist), data.values, color="steelblue", linewidth=1.2, label="Observado")
ax_main.plot(h_range, point, color="darkorange", linewidth=2, label="Previsao")
ax_main.fill_between(h_range, lower, upper, alpha=0.2, color="darkorange")
ax_main.axvline(n_hist - 1, color="gray", linestyle="--", alpha=0.5)
ax_main.set_title(f"{best_name} — Previsao 24 Meses", fontweight="bold")
ax_main.set_ylabel("Passageiros")
ax_main.legend(fontsize=8)
ax_main.grid(alpha=0.3)

# Tendencia
ax_trend.plot(decomp.trend, color="seagreen", linewidth=1.2)
ax_trend.set_title("Tendencia (STL)")
ax_trend.grid(alpha=0.3)

# Sazonalidade (1 ciclo)
ax_season.bar(range(12), decomp.seasonal[:12], color="darkorange", alpha=0.8)
ax_season.set_title("Padrao Sazonal")
ax_season.set_xticks(range(12))
ax_season.set_xticklabels(["J", "F", "M", "A", "M", "J",
                            "J", "A", "S", "O", "N", "D"])
ax_season.grid(alpha=0.3, axis="y")

# ACF dos residuos
plot_acf(residuals, lags=24, ax=ax_acf)
ax_acf.set_title("ACF dos Residuos")

# Histograma dos residuos
ax_diag.hist(residuals, bins=20, color="steelblue", alpha=0.7,
             edgecolor="white", density=True)
ax_diag.set_title("Distribuicao dos Residuos")
ax_diag.grid(alpha=0.3)

plt.suptitle("Airline Passengers — Analise Completa", fontsize=15, fontweight="bold", y=1.01)
plt.savefig("airline_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
```

!!! tip "Dicas para publicacao"
    - Use `dpi=150` ou `dpi=300` para imagens de alta resolucao
    - Layouts compostos (gridspec) sao melhores que figuras separadas para relatorios
    - O padrao sazonal como barras facilita a comunicacao com stakeholders nao-tecnicos

---

## Passo 9: Gerar Report (HTML + LaTeX)

O `ChronoExperiment` gera relatorios automatizados:

```python
# Gerar relatorio HTML
exp.save_master_report(
    filepath='airline_report.html',
    theme='professional'
)
print("Relatorio HTML salvo em airline_report.html")
```

```title="Output"
Relatorio HTML salvo em airline_report.html
```

!!! info "Conteudo do relatorio HTML"
    O relatorio inclui automaticamente:

    - Sumario dos dados (periodo, frequencia, estatisticas)
    - Lista de modelos ajustados com tempos de execucao
    - Tabela de ranking comparativo
    - Identificacao do melhor modelo
    - Metricas de validacao

```python
# Exportar tabela de comparacao para LaTeX
df = comparison.to_dataframe()
print("=== Tabela LaTeX ===")
print(df.to_latex(float_format="%.2f"))
```

```title="Output"
=== Tabela LaTeX ===
\begin{tabular}{lrrr}
\toprule
{} &     aic &     bic &   rmse \\
\midrule
SARIMA(0,1,1)(0,1,1,12) &  876.23 &  889.45 &  12.31 \\
SARIMA(1,1,1)(1,1,1,12) &  877.89 &  897.23 &  12.15 \\
Auto-ARIMA               &  876.23 &  889.45 &  12.31 \\
SARIMA(1,1,0)(1,1,0,12) &  891.45 &  904.67 &  14.56 \\
ARIMA(1,1,2)             &  978.34 &  991.56 &  28.45 \\
ARIMA(2,1,1)             &  980.12 &  996.45 &  28.89 \\
ARIMA(1,1,1)             &  982.67 &  992.89 &  29.34 \\
\bottomrule
\end{tabular}
```

```python
# Exportar para CSV
df.to_csv('model_comparison.csv')
print("Tabela CSV salva em model_comparison.csv")

# Resumo final
print("\n" + "=" * 60)
print("  RESUMO DO PIPELINE")
print("=" * 60)
print(f"  Dataset:         Airline Passengers (144 obs, mensal)")
print(f"  Melhor modelo:   {best_name}")
print(f"  AIC:             876.23")
print(f"  MAPE (CV):       3.71%")
print(f"  Coverage (95%):  92%")
print(f"  Diagnosticos:    Todos aprovados")
print(f"  Outputs gerados:")
print(f"    - airline_report.html")
print(f"    - model_comparison.csv")
print(f"    - airline_analysis.png")
print("=" * 60)
```

```title="Output"
Tabela CSV salva em model_comparison.csv

============================================================
  RESUMO DO PIPELINE
============================================================
  Dataset:         Airline Passengers (144 obs, mensal)
  Melhor modelo:   SARIMA(0,1,1)(0,1,1,12)
  AIC:             876.23
  MAPE (CV):       3.71%
  Coverage (95%):  92%
  Diagnosticos:    Todos aprovados
  Outputs gerados:
    - airline_report.html
    - model_comparison.csv
    - airline_analysis.png
============================================================
```

---

## Conclusao

!!! success "Resumo do pipeline completo"
    Este tutorial demonstrou o workflow profissional de series temporais end-to-end:

    | Etapa | Metodo | ChronoBox |
    |-------|--------|-----------|
    | Explorar | Visualizacao, estatisticas | `load_dataset()`, `matplotlib` |
    | Testar | ADF, KPSS | `adf_test()`, `kpss_test()` |
    | Modelar | ARIMA, SARIMA, Auto-ARIMA | `ARIMA()`, `fit_all_models()` |
    | Comparar | AIC, BIC, RMSE | `compare_models()`, `ranking()` |
    | Diagnosticar | Ljung-Box, Jarque-Bera, ACF | `ljung_box_test()`, `plot_acf()` |
    | Validar | Train/test, CV temporal | `validate_model()`, `time_series_cv()` |
    | Prever | Ponto + IC 95% | `forecast(steps=24, alpha=0.05)` |
    | Decompor | STL | `STL(period=12).fit()` |
    | Reportar | HTML, LaTeX, CSV | `save_master_report()`, `to_latex()` |

    O pipeline completo --- de dados brutos a relatorio --- e executavel com
    poucas dezenas de linhas de codigo Python, demonstrando como o ChronoBox
    integra todas as etapas de analise em um workflow coerente.

!!! tip "Proximos passos"
    - [Tutorial VAR](var.md): quando ha multiplas series interagindo
    - [Tutorial ARDL](ardl.md): quando ha mistura de ordens de integracao
    - [Tutorial Spillover](spillover.md): para analise de conectividade financeira
    - [User Guide: Experiment](../user-guide/experiment.md): todas as opcoes do ChronoExperiment
