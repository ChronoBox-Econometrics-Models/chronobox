---
title: ChronoExperiment
description: Workflow automatizado para comparacao de modelos, validacao e geracao de relatorios.
---

# ChronoExperiment

!!! info "Quick Reference"
    - **Classe**: `chronobox.experiment.ChronoExperiment`
    - **Import**: `from chronobox.experiment import ChronoExperiment`
    - **Workflow**: fit multiplos modelos $\rightarrow$ compare $\rightarrow$ validate $\rightarrow$ report

---

## Overview

O `ChronoExperiment` fornece um framework sistematico para **comparar
multiplos modelos de series temporais** em um unico workflow. Em vez de
ajustar, avaliar e comparar modelos manualmente, o ChronoExperiment
automatiza todo o processo:

1. **Fit**: ajusta multiplos modelos candidatos nos mesmos dados
2. **Compare**: compara modelos por criterios de informacao e metricas de erro
3. **Validate**: realiza validacao train/test e cross-validation temporal
4. **Report**: gera relatorios HTML com rankings e visualizacoes

### Quando usar

- Selecao de modelos entre ARIMA, SARIMA e Auto-ARIMA
- Comparacao sistematica via AIC, BIC, RMSE, MAE, MAPE
- Validacao out-of-sample com cross-validation temporal
- Geracao de relatorios automatizados para stakeholders

---

## Quick Example

```python
from chronobox.experiment import ChronoExperiment
from chronobox.datasets import load_dataset

# Carregar dados
data = load_dataset('airline')

# Criar experimento
exp = ChronoExperiment(data, name="Airline Forecast")

# Ajustar modelos candidatos
exp.fit_all_models([
    ('ARIMA(1,1,1)',  {'order': (1, 1, 1)}),
    ('ARIMA(0,1,1)',  {'order': (0, 1, 1)}),
    ('ARIMA(2,1,1)',  {'order': (2, 1, 1)}),
    ('SARIMA(0,1,1)(0,1,1,12)', {
        'order': (0, 1, 1),
        'seasonal_order': (0, 1, 1, 12)
    }),
])

# Comparar por AIC e BIC
comparison = exp.compare_models(criteria=['aic', 'bic'])
print(comparison.ranking())
print(f"\nMelhor modelo: {comparison.best_model()}")
```

---

## Guia Detalhado

### Construtor

```python
ChronoExperiment(
    data,                # Serie temporal (pd.Series ou pd.DataFrame)
    name="Experiment"    # Nome do experimento
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `data` | `pd.Series \| pd.DataFrame` | --- | Dados da serie temporal |
| `name` | `str` | `"Experiment"` | Nome identificador do experimento |

### Ajustando Modelos

#### `fit_model()` --- modelo individual

```python
result = exp.fit_model(
    name='ARIMA(1,1,1)',
    config={'order': (1, 1, 1)}
)
```

| Parametro | Tipo | Descricao |
|---|---|---|
| `name` | `str` | Nome identificador do modelo |
| `config` | `dict` | Configuracao do modelo (ver tabela abaixo) |

#### `fit_all_models()` --- multiplos modelos

```python
results = exp.fit_all_models([
    ('ARIMA(1,1,1)', {'order': (1, 1, 1)}),
    ('ARIMA(0,1,1)', {'order': (0, 1, 1)}),
    ('Auto-ARIMA',   {'model_type': 'auto_arima', 'seasonal': True, 'm': 12}),
])
```

### Configuracao de Modelos

A chave `model_type` no dicionario de configuracao determina o tipo de modelo:

| `model_type` | Chaves de config | Descricao |
|---|---|---|
| `"arima"` (default) | `order`, `seasonal_order` | ARIMA/SARIMA |
| `"auto_arima"` | `seasonal`, `m` | Selecao automatica |
| `"var"` | `maxlags` | VAR |
| `"vecm"` | `k_ar_diff`, `coint_rank` | VECM |
| `"ardl"` | `lags` | ARDL |

!!! tip "Modelos com `order`"
    Se a config contem a chave `order`, o tipo e inferido automaticamente
    como ARIMA, sem necessidade de especificar `model_type`.

---

## Comparacao de Modelos

### `compare_models()`

```python
comparison = exp.compare_models(
    criteria=['aic', 'bic', 'rmse']
)
```

| Parametro | Tipo | Default | Descricao |
|---|---|---|---|
| `criteria` | `list[str] \| None` | `['aic', 'bic']` | Criterios de comparacao |

**Criterios disponiveis**:

| Criterio | Descricao | Menor = Melhor |
|---|---|---|
| `'aic'` | Akaike Information Criterion | Sim |
| `'bic'` | Bayesian Information Criterion | Sim |
| `'aicc'` | AIC corrigido (amostras pequenas) | Sim |
| `'hqic'` | Hannan-Quinn Information Criterion | Sim |
| `'rmse'` | Root Mean Squared Error (in-sample) | Sim |
| `'loglike'` | Log-likelihood (negada para ranking) | Sim |

### `ComparisonResult`

| Metodo/Atributo | Tipo | Descricao |
|---|---|---|
| `ranking(criterion)` | `pd.DataFrame` | Ranking dos modelos pelo criterio |
| `best_model(criterion)` | `str` | Nome do melhor modelo |
| `to_dataframe()` | `pd.DataFrame` | Tabela completa de scores |
| `scores` | `pd.DataFrame` | DataFrame com modelos $\times$ criterios |
| `fit_times` | `dict[str, float]` | Tempo de ajuste por modelo (segundos) |
| `plot_comparison(criterion)` | `Axes` | Grafico de barras comparativo |

### Exemplo de Ranking

```python
comparison = exp.compare_models(criteria=['aic', 'bic', 'rmse'])

# Ranking pelo AIC
print(comparison.ranking('aic'))
```

```text
                              rank      aic      bic     rmse
SARIMA(0,1,1)(0,1,1,12)         1   876.23   889.45    12.31
ARIMA(1,1,1)                    2   912.56   921.78    15.67
ARIMA(0,1,1)                    3   915.34   921.45    16.02
ARIMA(2,1,1)                    4   914.12   926.34    15.89
```

```python
# Melhor modelo
print(f"Melhor por AIC: {comparison.best_model('aic')}")
print(f"Melhor por BIC: {comparison.best_model('bic')}")

# Visualizacao
comparison.plot_comparison('aic')
```

---

## Validacao

### Train/Test Split

```python
validation = exp.validate_model(
    model_name='ARIMA(1,1,1)',
    test_size=24,       # Ultimas 24 observacoes para teste
    horizon=12          # Prever 12 passos a frente
)

# Metricas out-of-sample
print(f"RMSE: {validation.rmse():.4f}")
print(f"MAE:  {validation.mae():.4f}")
print(f"MAPE: {validation.mape():.2f}%")

# Cobertura dos intervalos de confianca
print(f"Coverage (95%): {validation.coverage():.2f}")

# Visualizacao
validation.plot_forecast_vs_actual()
```

### `ValidationResult`

| Metodo/Atributo | Tipo | Descricao |
|---|---|---|
| `rmse()` | `float` | Root Mean Squared Error |
| `mae()` | `float` | Mean Absolute Error |
| `mape()` | `float` | Mean Absolute Percentage Error (%) |
| `coverage(alpha)` | `float` | Cobertura empirica do IC |
| `plot_forecast_vs_actual()` | `Axes` | Grafico forecast vs actual |
| `actuals` | `ndarray` | Valores observados no teste |
| `forecasts` | `ndarray` | Valores previstos |
| `fitted_model` | `Any` | Modelo ajustado nos dados de treino |

### Cross-Validation Temporal

A cross-validation temporal (expanding window) avalia a estabilidade do
modelo em multiplas particoes:

```python
cv = exp.time_series_cv(
    model_name='ARIMA(1,1,1)',
    n_splits=5,           # 5 folds
    horizon=12,           # Horizonte de previsao por fold
    min_train_size=None   # Minimo automatico
)

# Scores por fold
print(cv.scores_df)
```

```text
         rmse     mae    mape
Fold 1  14.23   11.45   8.32
Fold 2  15.67   12.89   9.15
Fold 3  13.89   11.02   7.98
Fold 4  16.45   13.56   9.87
Fold 5  14.78   11.98   8.65
```

```python
# Medias e desvios
print(f"RMSE medio: {cv.mean_scores()['rmse']:.2f} +/- {cv.std_scores()['rmse']:.2f}")

# Visualizacao
cv.plot_cv_errors(metric='rmse')
```

### `CVResult`

| Metodo/Atributo | Tipo | Descricao |
|---|---|---|
| `scores_df` | `pd.DataFrame` | Scores por fold |
| `mean_scores()` | `dict[str, float]` | Media de cada metrica |
| `std_scores()` | `dict[str, float]` | Desvio padrao de cada metrica |
| `plot_cv_errors(metric)` | `Axes` | Grafico de erros por fold |
| `n_splits` | `int` | Numero de folds |
| `fold_forecasts` | `list[ndarray]` | Previsoes por fold |
| `fold_actuals` | `list[ndarray]` | Valores reais por fold |

---

## Metricas

### Criterios de Informacao

Criterios in-sample que penalizam complexidade:

$$
\text{AIC} = -2\ln(\hat{L}) + 2k
$$

$$
\text{BIC} = -2\ln(\hat{L}) + k \ln(n)
$$

$$
\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}
$$

onde $\hat{L}$ e a verossimilhanca maximizada, $k$ e o numero de parametros,
e $n$ e o tamanho da amostra.

### Metricas de Erro

Metricas out-of-sample para avaliacao preditiva:

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)^2}
$$

$$
\text{MAE} = \frac{1}{n}\sum_{t=1}^{n}|y_t - \hat{y}_t|
$$

$$
\text{MAPE} = \frac{100}{n}\sum_{t=1}^{n}\left|\frac{y_t - \hat{y}_t}{y_t}\right|
$$

!!! warning "MAPE e zeros"
    MAPE retorna `inf` se todos os valores reais forem zero. Para series
    com valores proximos de zero, prefira RMSE ou MAE.

---

## Export de Relatorios

### Relatorio HTML

```python
# Gerar relatorio HTML completo
exp.save_master_report(
    filepath='report.html',
    theme='professional'   # 'professional', 'minimal', 'dark'
)
```

O relatorio inclui:

- Sumario dos dados
- Tabela de modelos ajustados com tempos de execucao
- Ranking comparativo
- Identificacao do melhor modelo

**Temas disponiveis**:

| Tema | Descricao |
|---|---|
| `'professional'` | Fundo branco, accent azul escuro |
| `'minimal'` | Fundo branco, monocromatico |
| `'dark'` | Fundo escuro, accent vermelho |

### Export para DataFrame

```python
# Tabela de scores como DataFrame (para LaTeX, CSV, etc.)
df = comparison.to_dataframe()

# Export para LaTeX
print(df.to_latex(float_format="%.2f"))

# Export para CSV
df.to_csv('model_comparison.csv')
```

---

## Exemplo Completo

```python
from chronobox.experiment import ChronoExperiment
from chronobox.datasets import load_dataset

# ── 1. Setup ──────────────────────────────────
data = load_dataset('airline')
exp = ChronoExperiment(data, name="Airline Passengers")

# ── 2. Definir candidatos ─────────────────────
model_specs = [
    ('ARIMA(1,1,1)',  {'order': (1, 1, 1)}),
    ('ARIMA(0,1,1)',  {'order': (0, 1, 1)}),
    ('ARIMA(2,1,0)',  {'order': (2, 1, 0)}),
    ('SARIMA(1,1,1)(1,1,1,12)', {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 12)
    }),
    ('SARIMA(0,1,1)(0,1,1,12)', {
        'order': (0, 1, 1),
        'seasonal_order': (0, 1, 1, 12)
    }),
    ('Auto-ARIMA', {
        'model_type': 'auto_arima',
        'seasonal': True,
        'm': 12
    }),
]

# ── 3. Ajustar todos ─────────────────────────
exp.fit_all_models(model_specs)

# ── 4. Comparar ──────────────────────────────
comparison = exp.compare_models(criteria=['aic', 'bic', 'rmse'])

# Ranking
print("=== Ranking por AIC ===")
print(comparison.ranking('aic'))

print(f"\nMelhor modelo (AIC): {comparison.best_model('aic')}")
print(f"Melhor modelo (BIC): {comparison.best_model('bic')}")

# Grafico comparativo
comparison.plot_comparison('aic')

# ── 5. Validar o melhor ──────────────────────
best = comparison.best_model('aic')
val = exp.validate_model(best, test_size=24, horizon=12)

print(f"\n=== Validacao: {best} ===")
print(f"RMSE: {val.rmse():.2f}")
print(f"MAE:  {val.mae():.2f}")
print(f"MAPE: {val.mape():.2f}%")

val.plot_forecast_vs_actual()

# ── 6. Cross-validation ─────────────────────
cv = exp.time_series_cv(best, n_splits=5, horizon=12)

print(f"\n=== CV: {best} ===")
print(f"RMSE medio: {cv.mean_scores()['rmse']:.2f} +/- {cv.std_scores()['rmse']:.2f}")

cv.plot_cv_errors('rmse')

# ── 7. Export ────────────────────────────────
exp.save_master_report('airline_report.html', theme='professional')
print("\nRelatorio salvo em airline_report.html")
```

---

## Equivalentes R

=== "chronobox (Python)"

    ```python
    from chronobox.experiment import ChronoExperiment

    exp = ChronoExperiment(data, name="Analysis")
    exp.fit_all_models([
        ('ARIMA(1,1,1)', {'order': (1,1,1)}),
        ('ARIMA(0,1,1)', {'order': (0,1,1)}),
    ])
    comp = exp.compare_models(criteria=['aic', 'bic'])
    print(comp.best_model())
    ```

=== "forecast (R)"

    ```r
    library(forecast)

    # Ajustar modelos individualmente
    m1 <- Arima(data, order = c(1,1,1))
    m2 <- Arima(data, order = c(0,1,1))

    # Comparar manualmente
    data.frame(
      Model = c("ARIMA(1,1,1)", "ARIMA(0,1,1)"),
      AIC = c(AIC(m1), AIC(m2)),
      BIC = c(BIC(m1), BIC(m2))
    )

    # Ou auto.arima para selecao automatica
    best <- auto.arima(data)
    ```

---

## Referencias

- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice*, 3rd ed. OTexts.
- Burnham, K. P. & Anderson, D. R. (2002). *Model Selection and Multimodel
  Inference: A Practical Information-Theoretic Approach*. Springer.
- Tashman, L. J. (2000). Out-of-sample Tests of Forecasting Accuracy: An
  Analysis and Review. *International Journal of Forecasting*, 16(4), 437--450.
