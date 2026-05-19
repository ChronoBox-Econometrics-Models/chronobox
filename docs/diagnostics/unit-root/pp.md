---
title: "Phillips-Perron Test"
description: "Teste Phillips-Perron para raiz unitaria no chronobox — correcao nao-parametrica de Newey-West, estatistica Z_t, comparacao com ADF e exemplos praticos."
---

# Phillips-Perron (PP)

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.unit_root.pp_test()`
    **H₀:** Serie possui raiz unitaria ($\rho = 1$)
    **H₁:** Serie e estacionaria ($\rho < 1$)
    **Distribuicao:** Dickey-Fuller (mesma do ADF)
    **Valores criticos:** MacKinnon (1996)
    **R equivalente:** `tseries::pp.test()`, `urca::ur.pp()`

## Hipoteses

$$H_0: \rho = 1 \quad \text{(raiz unitaria — serie nao estacionaria)}$$

$$H_1: \rho < 1 \quad \text{(serie e estacionaria)}$$

As hipoteses sao identicas as do [ADF](adf.md). A diferenca esta no **metodo de correcao** para autocorrelacao e heteroscedasticidade.

## Ideia Central

O teste de Dickey-Fuller simples estima:

$$y_t = \rho y_{t-1} + \varepsilon_t$$

e testa $\rho = 1$. O problema e que se os erros $\varepsilon_t$ sao autocorrelacionados ou heteroscedasticos, a estatistica t tem distribuicao incorreta.

- O **ADF** resolve isso adicionando **lags parametricos** ($\Delta y_{t-i}$) a regressao
- O **Phillips-Perron** resolve com uma **correcao nao-parametrica** da estatistica t usando o estimador de variancia de longo prazo de Newey-West

!!! tip "Vantagem do PP sobre o ADF"
    O PP nao requer especificar o numero de lags. Em vez disso, usa um **kernel** (Bartlett) para estimar a variancia de longo prazo. Isso o torna robusto a formas gerais de heteroscedasticidade e correlacao serial nos erros.

## Estatistica de Teste

### Regressao Base

O PP parte da regressao DF simples (sem lags aumentados):

$$\Delta y_t = \alpha + \gamma y_{t-1} + \varepsilon_t$$

onde $\gamma = \rho - 1$.

### Variancia de Longo Prazo (Newey-West)

O estimador de Newey-West com kernel de Bartlett calcula:

$$\hat{\lambda}^2 = \hat{\gamma}_0 + 2 \sum_{j=1}^{l} w_j \hat{\gamma}_j$$

onde:

- $\hat{\gamma}_j = \frac{1}{T} \sum_{t=j+1}^{T} \hat{\varepsilon}_t \hat{\varepsilon}_{t-j}$ e a autocovariancia amostral de lag $j$
- $w_j = 1 - \frac{j}{l+1}$ e o peso do kernel de Bartlett
- $l$ e a **bandwidth** (numero de lags do kernel)

### Estatistica $Z_t$

A estatistica PP corrige a estatistica DF:

$$Z_t = t_\gamma \sqrt{\frac{\hat{\gamma}_0}{\hat{\lambda}^2}} - \frac{T \cdot SE(\hat{\gamma}) \cdot (\hat{\lambda}^2 - \hat{\gamma}_0)}{2\sqrt{\hat{\lambda}^2} \cdot \hat{\sigma}}$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $t_\gamma$ | Estatistica t de DF (sem correcao) |
| $\hat{\gamma}_0$ | Variancia amostral dos residuos |
| $\hat{\lambda}^2$ | Variancia de longo prazo (Newey-West) |
| $SE(\hat{\gamma})$ | Erro padrao de $\hat{\gamma}$ |
| $\hat{\sigma}$ | Desvio padrao da regressao |

### Estatistica $Z_\alpha$

Uma estatistica alternativa (menos usada):

$$Z_\alpha = T\hat{\gamma} - \frac{T^2 \cdot SE(\hat{\gamma})^2}{2\hat{\sigma}^2}(\hat{\lambda}^2 - \hat{\gamma}_0)$$

O chronobox reporta $Z_t$ como estatistica principal e $Z_\alpha$ em `additional_info`.

## Distribuicao e Valores Criticos

Sob H₀, $Z_t$ tem a **mesma distribuicao assintotica** que a estatistica ADF. Portanto, os valores criticos de MacKinnon sao utilizados:

| Significancia | `regression='c'` | `regression='ct'` |
|:-------------|:-----------------|:------------------|
| 1% | ≈ -3.43 | ≈ -3.96 |
| 5% | ≈ -2.86 | ≈ -3.41 |
| 10% | ≈ -2.57 | ≈ -3.13 |

*Valores aproximados para T=100.*

## Bandwidth (Numero de Lags do Kernel)

| Opcao | Formula | Parametro |
|:------|:--------|:----------|
| Short | $l = \lfloor 4 \cdot (T/100)^{2/9} \rfloor$ | `lags="short"` (padrao) |
| Long | $l = \lfloor 12 \cdot (T/100)^{2/9} \rfloor$ | `lags="long"` |
| Manual | Valor direto | `lags=8` |

## Exemplo Pratico

### Teste Basico

```python
import numpy as np
from chronobox.tests_stat.unit_root import pp_test

# Serie estacionaria AR(1)
np.random.seed(42)
T = 200
y = np.zeros(T)
for t in range(1, T):
    y[t] = 0.7 * y[t - 1] + np.random.randn()

result = pp_test(y, regression="c")
print(result.summary())
```

Saida esperada:

```
============================================================
  Phillips-Perron Test
============================================================
  Test statistic : -7.123456
  p-value        : 0.000001
  Lags used      : 4

  H0: Unit root present (rho = 1)
  H1: Series is stationary (rho < 1)

  Critical Values:
      1% : -3.4580
      5% : -2.8694
     10% : -2.5710

  Decision (5%)  : Reject H0
============================================================
```

### Comparacao ADF vs PP

```python
from chronobox.tests_stat.unit_root import adf_test, pp_test

# Serie com heteroscedasticidade
np.random.seed(123)
T = 300
eps = np.random.randn(T) * np.sqrt(1 + 0.5 * np.arange(T) / T)
y_hetero = np.zeros(T)
for t in range(1, T):
    y_hetero[t] = 0.85 * y_hetero[t - 1] + eps[t]

adf_result = adf_test(y_hetero, regression="c")
pp_result = pp_test(y_hetero, regression="c")

print(f"ADF: stat={adf_result.statistic:.4f}, p={adf_result.pvalue:.4f}")
print(f"PP:  stat={pp_result.statistic:.4f}, p={pp_result.pvalue:.4f}")
# PP deve ter melhor desempenho com erros heteroscedasticos
```

### Acessando Informacoes Adicionais

```python
result = pp_test(y, regression="c")

# Estatistica Z_alpha
print(f"Z_t (principal): {result.statistic:.4f}")
print(f"Z_alpha: {result.additional_info['Z_alpha']:.4f}")

# Componentes da variancia
print(f"Variancia amostral (gamma_0): {result.additional_info['gamma_0']:.6f}")
print(f"Variancia de longo prazo (lambda^2): {result.additional_info['lambda_sq']:.6f}")
print(f"Bandwidth: {result.additional_info['bandwidth']}")
print(f"rho_hat: {result.additional_info['rho_hat']:.6f}")
```

## Assinatura da Funcao

```python
pp_test(
    y: NDArray,
    regression: str = "c",          # 'c' ou 'ct'
    lags: str | int | None = "short"  # 'short', 'long', int, None
) -> TestResult
```

!!! note
    O PP nao suporta `regression='n'` (sem constante). Use o [ADF](adf.md) para esse caso.

## ADF vs Phillips-Perron: Quando Usar Cada Um

| Criterio | ADF | PP |
|:---------|:----|:---|
| Autocorrelacao | Correcao parametrica (lags) | Correcao nao-parametrica (kernel) |
| Heteroscedasticidade | Nao robusto | Robusto (Newey-West) |
| Escolha de lags | Requer selecao de $p$ | Requer escolha de bandwidth |
| Amostras pequenas | Melhor desempenho | Pode ter distorcao de tamanho |
| Pratica | Mais utilizado | Usado para robustez |

!!! tip "Recomendacao"
    Use ADF como teste principal e PP como teste de robustez. Se os resultados divergirem, a serie pode ter padroes de variancia nao-constante que merecem investigacao.

## Equivalente R

=== "tseries"

    ```r
    library(tseries)

    # Teste PP basico
    pp.test(y, alternative = "stationary")
    ```

=== "urca"

    ```r
    library(urca)

    # Teste PP com controle de tipo e lags
    # type: "Z-tau" (equivalente a Z_t) ou "Z-alpha"
    # model: "constant", "trend"
    result <- ur.pp(y, type = "Z-tau", model = "constant", lags = "short")
    summary(result)

    # Equivalencias:
    # chronobox regression='c'   → urca model = "constant"
    # chronobox regression='ct'  → urca model = "trend"
    # chronobox lags='short'     → urca lags = "short"
    ```

## See Also

- [ADF Test](adf.md) — Alternativa parametrica ao PP
- [KPSS](kpss.md) — Teste de confirmacao com hipotese invertida
- [Unit Root Tests](index.md) — Visao geral de todos os testes

## Referencias

- Phillips, P.C.B. & Perron, P. (1988). "Testing for a unit root in time series regression." *Biometrika*, 75(2), 335-346.
- Newey, W.K. & West, K.D. (1987). "A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix." *Econometrica*, 55(3), 703-708.
- MacKinnon, J.G. (1996). "Numerical distribution functions for unit root and cointegration tests." *Journal of Applied Econometrics*, 11(6), 601-618.
