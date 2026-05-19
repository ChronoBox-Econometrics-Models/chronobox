---
title: "Ljung-Box Test"
description: "Teste Ljung-Box para autocorrelacao residual no chronobox — estatistica Q, distribuicao chi-quadrado, escolha de lags e exemplos praticos."
---

# Ljung-Box Test

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.specification.ljung_box_test()`
    **H₀:** Residuos sao white noise (sem autocorrelacao ate lag $h$)
    **H₁:** Existe autocorrelacao em algum lag ate $h$
    **Distribuicao:** $\chi^2(h - p - q)$ para ARIMA($p$,$d$,$q$)
    **Valores criticos:** Distribuicao $\chi^2$ padrao
    **R equivalente:** `Box.test(type="Ljung-Box")`

## Hipoteses

O teste Ljung-Box avalia se os residuos de um modelo apresentam autocorrelacao:

$$H_0: \rho_1 = \rho_2 = \cdots = \rho_h = 0 \quad \text{(residuos sao white noise)}$$

$$H_1: \exists \, k \leq h \text{ tal que } \rho_k \neq 0 \quad \text{(autocorrelacao presente)}$$

**Rejeitar H₀** indica que o modelo nao capturou toda a dinamica temporal — faltam lags ou termos MA.

**Nao rejeitar H₀** sugere que os residuos sao adequadamente nao-correlacionados.

!!! warning "Teste Conjunto"
    O Ljung-Box e um teste **portmanteau** (conjunto). Ele testa se *alguma* autocorrelacao ate o lag $h$ e significativa, mas **nao identifica qual lag** especifico. Se rejeitar, examine o correlograma (ACF) dos residuos para identificar a estrutura.

## Estatistica de Teste

A estatistica Q de Ljung-Box e uma versao corrigida da estatistica de Box-Pierce:

$$Q(h) = T(T+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{T-k}$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $T$ | Numero de observacoes |
| $h$ | Numero de lags testados |
| $\hat{\rho}_k$ | Autocorrelacao amostral no lag $k$ |
| $T - k$ | Correcao de pequena amostra (ausente em Box-Pierce) |

### Comparacao com Box-Pierce

A estatistica original de Box-Pierce e:

$$Q_{BP}(h) = T \sum_{k=1}^{h} \hat{\rho}_k^2$$

A versao Ljung-Box inclui o fator $(T+2)/(T-k)$ que melhora a aproximacao chi-quadrado em **amostras finitas**. Em amostras grandes, ambas convergem.

## Distribuicao

Sob H₀ (residuos sao white noise):

$$Q(h) \sim \chi^2(h - p - q)$$

onde $p + q$ sao os graus de liberdade consumidos pelo modelo ARIMA($p$,$d$,$q$).

!!! warning "Ajuste de Graus de Liberdade"
    Para residuos de um modelo ARIMA($p$,$d$,$q$), os graus de liberdade sao $h - p - q$, **nao** $h$. No chronobox, use o parametro `model_df=p+q` para este ajuste. Se `model_df=0` (padrao), a distribuicao usa $\chi^2(h)$ — adequado para testar uma serie bruta, nao residuos de um modelo.

## Escolha de $h$ (Numero de Lags)

A escolha de $h$ afeta o poder do teste:

| Regra | Formula | Quando Usar |
|:------|:--------|:------------|
| Regra de bolso | $h = 10$ | Padrao para a maioria das aplicacoes |
| Raiz de T | $h = \lfloor \sqrt{T} \rfloor$ | Amostras grandes |
| Dados sazonais | $h = 2s$ (ex: $h = 24$ para mensal) | Series sazonais |
| Minimo | $h > p + q$ | Garantir graus de liberdade positivos |

!!! tip "Recomendacao"
    Use $h = 10$ para dados nao-sazonais e $h = 2 \times \text{periodo}$ para dados sazonais. Sempre garanta que $h > p + q$ para que os graus de liberdade sejam positivos.

## Valores Criticos

Os valores criticos dependem dos graus de liberdade $df = h - p - q$:

| Significancia | $df = 5$ | $df = 10$ | $df = 15$ | $df = 20$ |
|:-------------|:---------|:----------|:----------|:----------|
| 5% | 11.07 | 18.31 | 25.00 | 31.41 |
| 1% | 15.09 | 23.21 | 30.58 | 37.57 |

*Rejeita-se H₀ quando $Q > \chi^2_{1-\alpha}(df)$.*

## Exemplo Pratico

### Residuos Sem Autocorrelacao

```python
import numpy as np
from chronobox.tests_stat.specification import ljung_box_test

# Residuos de um modelo bem especificado (white noise)
np.random.seed(42)
residuals = np.random.randn(200)

# Teste com 10 lags, sem ajuste de graus de liberdade
result = ljung_box_test(residuals, lags=10, model_df=0)
print(result.summary())
```

Saida esperada:

```
============================================================
  Ljung-Box Test
============================================================
  Test statistic : 8.1234
  p-value        : 0.6172

  H0: No serial correlation up to lag 10
  H1: Serial correlation present

  Critical Values:
      5% : 18.3070
      1% : 23.2093

  Decision (5%)  : Do not reject H0
============================================================
```

!!! tip "Interpretacao"
    $Q = 8.12 < 18.31$ (valor critico a 5%). Nao rejeitamos H₀: os residuos sao compataiveis com white noise.

### Residuos Com Autocorrelacao

```python
# Residuos com autocorrelacao (AR(1) nos erros)
np.random.seed(42)
T = 200
residuals_ar = np.zeros(T)
for t in range(1, T):
    residuals_ar[t] = 0.6 * residuals_ar[t - 1] + np.random.randn()

# Teste para residuos de ARIMA(1,0,0) — model_df = 1
result_ar = ljung_box_test(residuals_ar, lags=10, model_df=1)
print(result_ar.summary())
# Esperado: rejeita H0 (p-valor muito pequeno)
```

### Testando Multiplos Lags

```python
# Verificar consistencia em diferentes horizontes
for h in [5, 10, 15, 20]:
    r = ljung_box_test(residuals_ar, lags=h, model_df=1)
    print(f"h={h:2d}: Q={r.statistic:8.2f}, df={r.additional_info['df']:2d}, "
          f"p={r.pvalue:.4f}, {'Rejeita' if r.reject_at_5pct else 'OK'}")
```

### Acessando Resultados Programaticamente

```python
result = ljung_box_test(residuals, lags=10, model_df=1)

# Decisao automatica
if result.reject_at_5pct:
    print("Autocorrelacao detectada — respecificar o modelo")
    print(f"Autocorrelacoes: {result.additional_info['autocorrelations'][:5]}")
else:
    print(f"Residuos OK (Q={result.statistic:.4f}, p={result.pvalue:.4f})")

# Detalhes
print(f"Graus de liberdade: {result.additional_info['df']}")
print(f"Model df ajustado: {result.additional_info['model_df']}")
```

## Assinatura da Funcao

```python
ljung_box_test(
    residuals: NDArray,
    lags: int = 10,           # Numero de lags h
    model_df: int = 0         # p+q para ARIMA(p,d,q)
) -> TestResult
```

## Tabela de Decisao

| p-valor | Conclusao | Acao |
|:--------|:----------|:-----|
| $p > 0.10$ | Sem autocorrelacao | Modelo adequado |
| $0.05 < p \leq 0.10$ | Borderline | Investigar ACF dos residuos |
| $p \leq 0.05$ | **Autocorrelacao presente** | Aumentar lags ou adicionar termos MA |

## Limitacoes

1. **Teste conjunto** — nao identifica qual lag especifico e significativo
2. **Baixo poder** quando a autocorrelacao alterna de sinal em lags consecutivos
3. **Sensivel a escolha de $h$** — muitos lags reduzem o poder; poucos lags podem nao detectar autocorrelacao em lags distantes
4. **Nao funciona bem com lagged dependent variables** — use [Breusch-Godfrey](breusch-godfrey.md) neste caso

## Equivalente R

=== "stats"

    ```r
    # Ljung-Box test basico
    Box.test(residuals, lag = 10, type = "Ljung-Box")

    # Com ajuste para ARIMA(p,d,q)
    Box.test(residuals, lag = 10, type = "Ljung-Box", fitdf = p + q)

    # Equivalencia:
    # chronobox ljung_box_test(residuals, lags=10, model_df=p+q)
    # -> R Box.test(residuals, lag=10, type="Ljung-Box", fitdf=p+q)
    ```

=== "forecast"

    ```r
    library(forecast)

    # checkresiduals executa Ljung-Box automaticamente
    fit <- auto.arima(y)
    checkresiduals(fit)
    # Inclui grafico ACF + histograma + teste Ljung-Box
    ```

## See Also

- [Breusch-Godfrey](breusch-godfrey.md) — Alternativa via regressao auxiliar, funciona com lagged dependent variables
- [Specification Tests](index.md) — Visao geral de testes de especificacao

## Referencias

- Ljung, G.M. & Box, G.E.P. (1978). "On a measure of lack of fit in time series models." *Biometrika*, 65(2), 297-303.
- Box, G.E.P. & Pierce, D.A. (1970). "Distribution of residual autocorrelations in autoregressive-integrated moving average time series models." *JASA*, 65(332), 1509-1526.
- Harvey, A.C. (1993). *Time Series Models*. 2nd ed., MIT Press. Chapter 5.
