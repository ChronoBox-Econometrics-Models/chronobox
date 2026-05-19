---
title: "ADF Test"
description: "Teste Augmented Dickey-Fuller para raiz unitaria no chronobox — hipoteses, estatistica de teste, selecao de lags e exemplos praticos."
---

# Augmented Dickey-Fuller (ADF)

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.unit_root.adf_test()`
    **H₀:** Serie possui raiz unitaria ($\gamma = 0$)
    **H₁:** Serie e estacionaria ($\gamma < 0$)
    **Distribuicao:** Dickey-Fuller (nao-padrao)
    **Valores criticos:** MacKinnon (1996)
    **R equivalente:** `tseries::adf.test()`, `urca::ur.df()`

## Hipoteses

O teste ADF avalia se uma serie temporal possui raiz unitaria:

$$H_0: \gamma = 0 \quad \text{(raiz unitaria — serie nao estacionaria)}$$

$$H_1: \gamma < 0 \quad \text{(serie e estacionaria)}$$

**Rejeitar H₀** indica que a serie e estacionaria e pode ser modelada em niveis.
**Nao rejeitar H₀** sugere que a serie possui raiz unitaria e precisa ser diferenciada.

## Regressao de Teste

O ADF estima a seguinte regressao por OLS:

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $\Delta y_t = y_t - y_{t-1}$ | Primeira diferenca da serie |
| $\alpha$ | Constante (intercepto) |
| $\beta t$ | Tendencia linear deterministica |
| $\gamma y_{t-1}$ | **Coeficiente de interesse** — testa a raiz unitaria |
| $\delta_i \Delta y_{t-i}$ | Diferencas defasadas para corrigir autocorrelacao |
| $p$ | Numero de lags (selecionado por AIC/BIC) |

A inclusao dos termos $\Delta y_{t-i}$ e o que torna o teste **Augmented** — o teste DF original nao inclui esses termos e nao e robusto a autocorrelacao.

## Estatistica de Teste

A estatistica ADF e a razao t do coeficiente $\gamma$:

$$t_\gamma = \frac{\hat{\gamma}}{SE(\hat{\gamma})}$$

!!! warning "Distribuicao Nao-Padrao"
    Sob H₀, $t_\gamma$ **nao** segue uma distribuicao t de Student. Segue a distribuicao de Dickey-Fuller, que e assimetrica a esquerda. Os valores criticos sao **mais negativos** que os da t de Student. Usar valores criticos padrao levaria a rejeicao excessiva de H₀.

## Valores Criticos

Os valores criticos dependem de:

1. **Especificacao da regressao** (`'n'`, `'c'`, `'ct'`)
2. **Tamanho da amostra** ($T$)

O chronobox utiliza a **superficie de regressao de MacKinnon (1996)** para calcular valores criticos e p-valores exatos:

| Significancia | `regression='n'` | `regression='c'` | `regression='ct'` |
|:-------------|:-----------------|:-----------------|:------------------|
| 1% | ≈ -2.57 | ≈ -3.43 | ≈ -3.96 |
| 5% | ≈ -1.94 | ≈ -2.86 | ≈ -3.41 |
| 10% | ≈ -1.62 | ≈ -2.57 | ≈ -3.13 |

*Valores aproximados para T=100. Os valores exatos dependem de T.*

## Tres Especificacoes

=== "Constante (`'c'`)"

    $$\Delta y_t = \alpha + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

    **Quando usar:** Padrao para a maioria das series economicas (PIB, inflacao, juros).
    A serie pode ter nivel medio diferente de zero.

=== "Constante + Tendencia (`'ct'`)"

    $$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

    **Quando usar:** Series com tendencia deterministica visivel (log PIB, populacao).
    Valores criticos mais restritivos.

=== "Sem Termos Deterministicos (`'n'`)"

    $$\Delta y_t = \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

    **Quando usar:** Raramente. Apenas para series que oscilam em torno de zero (retornos, diferencas).

## Selecao Automatica de Lags

O numero de lags $p$ e crucial: poucos lags deixam autocorrelacao nos residuos; muitos lags reduzem o poder do teste.

| Metodo | Descricao | Parametro |
|:-------|:----------|:----------|
| AIC | Minimiza Akaike Information Criterion | `autolag="AIC"` |
| BIC | Minimiza Bayesian Information Criterion (mais parcimonioso) | `autolag="BIC"` |
| t-sig | Remove lags insignificantes sequencialmente | `autolag="t-sig"` |
| Manual | Usa `maxlag` diretamente | `autolag=None` |

O **lag maximo padrao** e calculado como:

$$p_{max} = \lfloor 12 \cdot (T/100)^{1/4} \rfloor$$

!!! tip "Recomendacao"
    Use `autolag="AIC"` (padrao) para a maioria das aplicacoes. O BIC tende a selecionar menos lags, o que e preferivel em amostras pequenas.

## Exemplo Pratico

### Serie Estacionaria

```python
import numpy as np
from chronobox.tests_stat.unit_root import adf_test

# Serie AR(1) estacionaria: phi = 0.7
np.random.seed(42)
T = 200
y_stationary = np.zeros(T)
for t in range(1, T):
    y_stationary[t] = 0.7 * y_stationary[t - 1] + np.random.randn()

result = adf_test(y_stationary, regression="c", autolag="AIC")
print(result.summary())
```

Saida esperada:

```
============================================================
  Augmented Dickey-Fuller Test
============================================================
  Test statistic : -5.234567
  p-value        : 0.000012
  Lags used      : 1

  H0: Unit root present (gamma = 0)
  H1: Series is stationary (gamma < 0)

  Critical Values:
      1% : -3.4580
      5% : -2.8694
     10% : -2.5710

  Decision (5%)  : Reject H0
============================================================
```

!!! tip "Interpretacao"
    Estatistica ADF (≈ -5.23) e mais negativa que o valor critico a 5% (≈ -2.87). Rejeitamos H₀: a serie **e estacionaria**.

### Serie com Raiz Unitaria

```python
# Random walk: y_t = y_{t-1} + eps_t
y_rw = np.cumsum(np.random.randn(200))

result_rw = adf_test(y_rw, regression="c")
print(result_rw.summary())
# Esperado: nao rejeita H0 (p-valor alto, estatistica proxima de zero)
```

### Comparando Especificacoes

```python
# Testar com diferentes especificacoes
for reg in ["n", "c", "ct"]:
    r = adf_test(y_rw, regression=reg)
    pval = f"{r.pvalue:.4f}" if r.pvalue is not None else "N/A"
    print(f"regression='{reg}': stat={r.statistic:.4f}, p={pval}, "
          f"lags={r.lags_used}, reject={r.reject_at_5pct}")
```

### Acessando Resultados Programaticamente

```python
result = adf_test(y_stationary, regression="c")

# Decisao automatica
if result.reject_at_5pct:
    print(f"Serie e I(0) — estacionaria (p={result.pvalue:.4f})")
else:
    print(f"Serie e I(1) — diferenciar (p={result.pvalue:.4f})")

# Informacoes adicionais
print(f"Lags selecionados: {result.lags_used}")
print(f"Coeficiente gamma: {result.additional_info['gamma']:.6f}")
print(f"Especificacao: {result.additional_info['regression']}")
print(f"Observacoes efetivas: {result.additional_info['nobs']}")
```

## Assinatura da Funcao

```python
adf_test(
    y: NDArray,
    regression: str = "c",       # 'n', 'c', ou 'ct'
    maxlag: int | None = None,   # None = 12*(T/100)^{1/4}
    autolag: str | None = "AIC"  # 'AIC', 'BIC', 't-sig', None
) -> TestResult
```

## Limitacoes

1. **Baixo poder** em amostras pequenas — pode falhar em rejeitar H₀ mesmo quando a serie e estacionaria perto da raiz unitaria
2. **Sensivel a quebras estruturais** — uma serie estacionaria com quebra pode parecer I(1). Use [Zivot-Andrews](zivot-andrews.md) neste caso
3. **Sensivel a escolha de lags** — usar `autolag` para selecao automatica
4. **Nao e robusto** a heteroscedasticidade — use [Phillips-Perron](pp.md) se suspeitar de variancia nao-constante

## Equivalente R

=== "tseries"

    ```r
    library(tseries)

    # Teste ADF basico
    adf.test(y, alternative = "stationary", k = 4)
    ```

=== "urca"

    ```r
    library(urca)

    # Teste ADF com controle total
    # type: "none", "drift" (constante), "trend" (constante + tendencia)
    # selectlags: "AIC", "BIC"
    result <- ur.df(y, type = "drift", selectlags = "AIC")
    summary(result)

    # Equivalencias de 'regression':
    # chronobox 'n'  → urca type = "none"
    # chronobox 'c'  → urca type = "drift"
    # chronobox 'ct' → urca type = "trend"
    ```

## See Also

- [Phillips-Perron](pp.md) — Alternativa nao-parametrica ao ADF
- [KPSS](kpss.md) — Teste de confirmacao com hipotese invertida
- [Unit Root Tests](index.md) — Visao geral de todos os testes

## Referencias

- Dickey, D.A. & Fuller, W.A. (1979). "Distribution of the estimators for autoregressive time series with a unit root." *JASA*, 74(366), 427-431.
- Dickey, D.A. & Fuller, W.A. (1981). "Likelihood ratio statistics for autoregressive time series with a unit root." *Econometrica*, 49(4), 1057-1072.
- MacKinnon, J.G. (1996). "Numerical distribution functions for unit root and cointegration tests." *Journal of Applied Econometrics*, 11(6), 601-618.
- Said, S.E. & Dickey, D.A. (1984). "Testing for unit roots in autoregressive-moving average models of unknown order." *Biometrika*, 71(3), 599-607.
