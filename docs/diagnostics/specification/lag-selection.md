---
title: "Lag Selection"
description: "Selecao de lags para modelos VAR no chronobox — criterios AIC, BIC, HQIC, FPE, teste LR sequencial e exemplos praticos."
---

# Lag Selection

!!! info "Quick Reference"
    **Funcao:** `chronobox.selection.lag_selection.select_lag_order()`
    **Objetivo:** Selecionar o numero otimo de lags para modelos VAR
    **Criterios:** AIC, BIC (SIC), HQIC, FPE, LR sequencial
    **Retorno:** `LagOrderResults` com tabela de criterios e ordens selecionadas
    **R equivalente:** `vars::VARselect()`

## Por Que Selecionar Lags?

A escolha do numero de lags $p$ em um modelo VAR($p$) e crucial:

| Poucos lags ($p$ baixo) | Muitos lags ($p$ alto) |
|:------------------------|:----------------------|
| Dinamica mal capturada | Perda de graus de liberdade |
| Residuos autocorrelacionados | Estimativas imprecisas (variancia alta) |
| Previsoes enviesadas | Sobreajuste (overfitting) |
| Testes de especificacao falham | Criterios de informacao penalizam |

O objetivo e encontrar o **equilibrio** entre viés (poucos lags) e variancia (muitos lags).

## Criterios de Informacao

Todos os criterios de informacao tem a forma:

$$IC(p) = \ln|\hat{\Sigma}_p| + C(p, T, K)$$

onde $\ln|\hat{\Sigma}_p|$ mede o ajuste e $C(p, T, K)$ e a **penalidade por complexidade**.

### AIC — Akaike Information Criterion

$$\text{AIC}(p) = \ln|\hat{\Sigma}_p| + \frac{2pK^2}{T}$$

- Penalidade proporcional ao numero de parametros
- Tende a **sobreestimar** a ordem em amostras finitas
- **Consistente para previsao** — minimiza o erro de previsao assintotico

### BIC — Bayesian Information Criterion (SIC)

$$\text{BIC}(p) = \ln|\hat{\Sigma}_p| + \frac{pK^2 \ln T}{T}$$

- Penalidade mais forte que AIC quando $T > 8$ ($\ln T > 2$)
- **Consistente** — converge para a ordem verdadeira quando $T \to \infty$
- Tende a selecionar modelos mais **parcimoniosos**

### HQIC — Hannan-Quinn Information Criterion

$$\text{HQIC}(p) = \ln|\hat{\Sigma}_p| + \frac{2pK^2 \ln(\ln T)}{T}$$

- Penalidade intermediaria entre AIC e BIC
- **Fortemente consistente** (mais que AIC, menos que BIC)
- Menos usado na pratica

### FPE — Final Prediction Error

$$\text{FPE}(p) = |\hat{\Sigma}_p| \left( \frac{T + Kp + d}{T - Kp - d} \right)^K$$

onde $d$ e o numero de termos deterministicos (0, 1 ou 2).

- Criterio original de Akaike (1969)
- Assintotica equivalente ao AIC
- Menos interpretavel; presente por completude

### Comparacao

| Criterio | Penalidade | Propriedade | Uso Recomendado |
|:---------|:-----------|:------------|:----------------|
| AIC | $2/T$ | Eficiente (minimiza perda de previsao) | **Previsao** |
| BIC | $\ln T / T$ | Consistente (identifica ordem verdadeira) | **Inferencia / estrutura** |
| HQIC | $2\ln(\ln T) / T$ | Fortemente consistente | Compromisso |
| FPE | ~$2/T$ | ~Equivalente a AIC | Historico |

!!! tip "Recomendacao Pratica"
    - Para **previsao**: use AIC (minimiza erro de previsao)
    - Para **inferencia estrutural**: use BIC (identifica o modelo verdadeiro)
    - Se AIC e BIC **discordam**, reporte ambos e use o contexto para decidir

## Teste LR Sequencial

Alem dos criterios de informacao, o teste de razao de verossimilhanca sequencial compara modelos aninhados:

$$LR(p \text{ vs } p-1) = (T - Kp - 1.5) \cdot \left( \ln|\hat{\Sigma}_{p-1}| - \ln|\hat{\Sigma}_p| \right)$$

$$LR \sim \chi^2(K^2)$$

### Procedimento

1. Comece com o lag maximo $p_{max}$
2. Teste $H_0: p = p_{max} - 1$ vs $H_1: p = p_{max}$
3. Se nao rejeita H₀, reduza $p_{max}$ por 1 e repita
4. Pare quando rejeitar H₀ — a ordem selecionada e $p$

## Tabela de Resultados

O chronobox apresenta os resultados em uma tabela padronizada:

```
========================================================================
VAR Lag Order Selection
========================================================================
 Lag           AIC           BIC          HQIC             FPE
------------------------------------------------------------------------
   0     -5.1234       -5.0891       -5.1098       5.952e-03
   1     -8.4567*      -8.2845       -8.3878*      2.134e-04*
   2     -8.4123       -8.1022*      -8.2882       2.198e-04
   3     -8.3890       -7.9410       -8.2096       2.245e-04
------------------------------------------------------------------------
* indicates lag order selected by the criterion

    AIC selects lag order 1
    BIC selects lag order 2
   HQIC selects lag order 1
    FPE selects lag order 1
========================================================================
```

## Exemplo Pratico

### Selecao de Lags para VAR

```python
import numpy as np
from chronobox.selection.lag_selection import select_lag_order

# Simular VAR(2) bivariado
np.random.seed(42)
T = 250
K = 2
y = np.zeros((T, K))

# VAR(2): y_t = A1*y_{t-1} + A2*y_{t-2} + eps
A1 = np.array([[0.5, 0.1], [0.2, 0.3]])
A2 = np.array([[0.2, 0.0], [0.0, 0.1]])

for t in range(2, T):
    y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + 0.5 * np.random.randn(K)

# Selecao de lags
result = select_lag_order(y, maxlags=8, trend="c")
print(result.summary())
```

### Acessando Resultados Individuais

```python
# Ordens selecionadas
for criterion, order in result.selected_orders.items():
    print(f"{criterion.upper():>5s}: lag = {order}")

# Valores dos criterios
for lag in sorted(result.aic.keys()):
    print(f"Lag {lag}: AIC={result.aic[lag]:.4f}, BIC={result.bic[lag]:.4f}")

# Teste LR sequencial
for lag in sorted(result.lr_stats.keys()):
    sig = "*" if result.lr_pvalues[lag] < 0.05 else ""
    print(f"LR({lag} vs {lag-1}): stat={result.lr_stats[lag]:.4f}, "
          f"p={result.lr_pvalues[lag]:.4f} {sig}")
```

### Usando na Estimacao do VAR

```python
from chronobox.models.var import VAR

# Selecionar lags
lag_result = select_lag_order(y, maxlags=10, trend="c")
optimal_lag = lag_result.selected_orders["aic"]

# Estimar VAR com a ordem selecionada
model = VAR(y, p=optimal_lag)
model.fit()
print(f"VAR({optimal_lag}) estimado com base no AIC")
```

### Comparando Criterios Visualmente

```python
import matplotlib.pyplot as plt

lags = sorted(result.aic.keys())
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AIC e BIC
axes[0].plot(lags, [result.aic[l] for l in lags], "bo-", label="AIC")
axes[0].plot(lags, [result.bic[l] for l in lags], "rs-", label="BIC")
axes[0].plot(lags, [result.hqic[l] for l in lags], "g^-", label="HQIC")
axes[0].set_xlabel("Lag order")
axes[0].set_ylabel("Information Criterion")
axes[0].set_title("Criterios de Informacao por Lag")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LR test p-values
lr_lags = sorted(result.lr_stats.keys())
axes[1].bar(lr_lags, [result.lr_pvalues[l] for l in lr_lags], color="steelblue")
axes[1].axhline(0.05, color="red", linestyle="--", label="5% significancia")
axes[1].set_xlabel("Lag order (p vs p-1)")
axes[1].set_ylabel("p-valor")
axes[1].set_title("Teste LR Sequencial")
axes[1].legend()

plt.tight_layout()
plt.show()
```

## Assinatura da Funcao

```python
select_lag_order(
    endog: NDArray,             # Dados multivariados (T, K)
    maxlags: int = 15,          # Lag maximo a considerar
    trend: str = "c"            # 'n', 'c', ou 'ct'
) -> LagOrderResults
```

## Recomendacoes Praticas

1. **Comece com $p_{max}$ razoavel** — regra de bolso: $p_{max} = \lfloor T^{1/3} \rfloor$ ou $p_{max} = 12 \cdot (T/100)^{1/4}$
2. **Verifique os residuos** — apos selecionar $p$, aplique [Ljung-Box](ljung-box.md) nos residuos para confirmar ausencia de autocorrelacao
3. **Considere o contexto** — dados trimestrais tipicamente requerem $p = 4$ ou $p = 8$; dados mensais, $p = 12$ ou $p = 24$
4. **Se AIC e BIC discordam** — use BIC para inferencia (impulso-resposta, causalidade) e AIC para previsao

!!! warning "Sobreparametrizacao em VAR"
    Um VAR($p$) com $K$ variaveis tem $K^2 p + Kd$ parametros. Para $K = 5$ e $p = 4$, sao **100 parametros** apenas nos coeficientes autoregressivos. Em amostras pequenas, BIC e fortemente preferivel ao AIC para evitar sobreajuste.

## Limitacoes

1. **Depende de $p_{max}$** — criterios de informacao so avaliam ate o lag maximo fornecido
2. **Assume linearidade** — nao detecta necessidade de termos nao-lineares
3. **Criterios podem discordar** — nao existe criterio universalmente superior
4. **Quebras estruturais** — se os coeficientes mudam ao longo do tempo, a selecao de lags pode ser instavel

## Equivalente R

=== "vars"

    ```r
    library(vars)

    # Selecao de lags para VAR
    VARselect(y, lag.max = 8, type = "const")

    # type: "none", "const" (constante), "trend", "both" (constante + tendencia)
    # Equivalencia:
    # chronobox select_lag_order(y, maxlags=8, trend="c")
    # -> R VARselect(y, lag.max=8, type="const")
    ```

=== "Base R (manual)"

    ```r
    # AIC manual para VAR(p)
    compute_aic <- function(y, p) {
        T <- nrow(y)
        K <- ncol(y)
        # Estimar VAR(p) e obter Sigma
        fit <- ar(y, order.max = p, method = "ols", aic = FALSE)
        resid <- na.omit(fit$resid)
        Sigma <- crossprod(resid) / nrow(resid)
        log(det(Sigma)) + 2 * p * K^2 / nrow(resid)
    }

    for (p in 1:8) {
        cat("Lag", p, ": AIC =", compute_aic(y, p), "\n")
    }
    ```

## See Also

- [Ljung-Box](ljung-box.md) — Validar residuos apos selecao de lags
- [Specification Tests](index.md) — Visao geral de testes de especificacao

## Referencias

- Akaike, H. (1974). "A new look at the statistical model identification." *IEEE Transactions on Automatic Control*, 19(6), 716-723.
- Schwarz, G. (1978). "Estimating the dimension of a model." *Annals of Statistics*, 6(2), 461-464.
- Hannan, E.J. & Quinn, B.G. (1979). "The determination of the order of an autoregression." *JRSS Series B*, 41(2), 190-195.
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Chapter 4.
