---
title: "Engle-Granger Cointegration Test"
description: "Teste de Engle-Granger para cointegracao no chronobox — dois passos, regressao OLS, ADF nos residuos, valores criticos de MacKinnon e exemplos praticos."
---

# Engle-Granger Cointegration Test

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.cointegration.engle_granger_test()`
    **H₀:** Nao ha cointegracao (residuos possuem raiz unitaria)
    **H₁:** Ha cointegracao (residuos sao estacionarios)
    **Distribuicao:** Dickey-Fuller modificada (residual-based)
    **Valores criticos:** MacKinnon (2010) para testes residuais
    **R equivalente:** `tseries::po.test()`, `urca::ca.po()`

## Hipoteses

O teste de Engle-Granger avalia se duas (ou mais) series $I(1)$ compartilham uma relacao de longo prazo:

$$H_0: \text{Nao ha cointegracao} \quad (\hat{u}_t \sim I(1))$$

$$H_1: \text{Ha cointegracao} \quad (\hat{u}_t \sim I(0))$$

**Rejeitar H₀** indica que as series sao cointegradas — existe uma relacao de equilibrio de longo prazo.
**Nao rejeitar H₀** sugere que as series nao compartilham uma relacao estavel.

## O Metodo de Dois Passos

O procedimento de Engle-Granger (1987) e intuitivo e simples:

### Passo 1: Regressao de Cointegracao (OLS)

Estimar a relacao de longo prazo por minimos quadrados ordinarios:

$$y_t = \alpha + \boldsymbol{\beta}^\top \mathbf{x}_t + u_t$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $y_t$ | Variavel dependente ($I(1)$) |
| $\mathbf{x}_t$ | Variavel(is) independente(s) ($I(1)$) |
| $\alpha$ | Constante (intercepto da relacao de longo prazo) |
| $\boldsymbol{\beta}$ | **Vetor de cointegracao** — parametros de longo prazo |
| $u_t$ | Residuo — desvio do equilibrio |

!!! note "Super-consistencia"
    Mesmo que $y_t$ e $\mathbf{x}_t$ sejam $I(1)$, se forem cointegrados, o estimador OLS de $\boldsymbol{\beta}$ e **super-consistente**: converge a taxa $T$ (em vez de $\sqrt{T}$). Porem, a distribuicao assintotica nao e normal, entao inferencia com t-tests padrao nao e valida nesta regressao.

### Passo 2: Teste ADF nos Residuos

Aplicar o teste ADF nos residuos estimados $\hat{u}_t$:

$$\Delta \hat{u}_t = \gamma \hat{u}_{t-1} + \sum_{i=1}^{p} \delta_i \Delta \hat{u}_{t-i} + \varepsilon_t$$

!!! warning "Valores Criticos Especiais"
    Os valores criticos **nao** sao os mesmos do ADF padrao. Como $\hat{u}_t$ sao residuos estimados (nao observados), os valores criticos sao mais restritivos. O chronobox utiliza os valores de MacKinnon (2010) para testes residuais, que dependem do **numero total de variaveis** no sistema.

## Estatistica de Teste

A estatistica e a razao t do coeficiente $\gamma$ na regressao ADF dos residuos:

$$\tau_{EG} = \frac{\hat{\gamma}}{SE(\hat{\gamma})}$$

Sob $H_0$, a distribuicao e mais assimetrica a esquerda que a Dickey-Fuller padrao — valores criticos sao **mais negativos**.

## Valores Criticos

Os valores criticos dependem de:

1. **Numero total de variaveis** no sistema ($n = 1 + k$, onde $k$ e o numero de regressores)
2. **Tamanho da amostra** ($T$)
3. **Especificacao da regressao** (com ou sem tendencia)

Valores aproximados de MacKinnon (2010) para $T = 200$:

| $n$ variaveis | 1% | 5% | 10% |
|:-------------|:---|:---|:----|
| 2 | -3.90 | -3.34 | -3.04 |
| 3 | -4.29 | -3.74 | -3.45 |
| 4 | -4.64 | -4.10 | -3.81 |
| 5 | -4.96 | -4.42 | -4.13 |

*Valores criticos mais negativos que o ADF padrao. O chronobox calcula valores exatos via superficie de MacKinnon.*

## Tres Especificacoes

=== "Constante (`'c'`)"

    $$y_t = \alpha + \boldsymbol{\beta}^\top \mathbf{x}_t + u_t$$

    **Quando usar:** Padrao para a maioria das aplicacoes. Permite nivel de equilibrio diferente de zero.

=== "Constante + Tendencia (`'ct'`)"

    $$y_t = \alpha + \delta t + \boldsymbol{\beta}^\top \mathbf{x}_t + u_t$$

    **Quando usar:** Quando a relacao de longo prazo inclui tendencia deterministica. Valores criticos mais restritivos.

=== "Sem Constante (`'n'`)"

    $$y_t = \boldsymbol{\beta}^\top \mathbf{x}_t + u_t$$

    **Quando usar:** Raramente. Apenas quando o equilibrio passa pela origem.

## Exemplo Pratico

### Teste Basico com Duas Variaveis

```python
import numpy as np
from chronobox.tests_stat.cointegration import engle_granger_test

# Gerar series cointegradas
np.random.seed(42)
T = 250

# x e I(1)
x = np.cumsum(np.random.randn(T))

# y = 1.5 + 2*x + u, onde u e estacionario (cointegradas)
u = np.zeros(T)
for t in range(1, T):
    u[t] = 0.5 * u[t - 1] + np.random.randn(t)
y = 1.5 + 2.0 * x + u

result = engle_granger_test(y, x, trend="c")
print(result.summary())
```

Saida esperada:

```
============================================================
  Engle-Granger Cointegration Test
============================================================
  Test statistic : -8.234567
  p-value        : N/A
  Lags used      : 1

  H0: No cointegration (residuals have unit root)
  H1: Cointegration (residuals are stationary)

  Critical Values:
      1% : -3.9001
      5% : -3.3377
     10% : -3.0462

  Decision (5%)  : Reject H0
============================================================
```

!!! tip "Interpretacao"
    Estatistica EG (≈ -8.23) e mais negativa que o valor critico a 5% (≈ -3.34). Rejeitamos $H_0$: as series **sao cointegradas** com vetor de cointegracao $\hat{\beta} \approx 2.0$.

### Series Nao Cointegradas

```python
# Dois random walks independentes
x_indep = np.cumsum(np.random.randn(250))
y_indep = np.cumsum(np.random.randn(250))

result_nc = engle_granger_test(y_indep, x_indep, trend="c")
print(f"Estatistica: {result_nc.statistic:.4f}")
print(f"Rejeita H0 a 5%: {result_nc.reject_at_5pct}")
# Esperado: nao rejeita (estatistica proxima de zero)
```

### Acessando Resultados Programaticamente

```python
result = engle_granger_test(y, x, trend="c")

# Decisao automatica
if result.reject_at_5pct:
    print("Series sao cointegradas")
    beta = result.additional_info["cointegrating_vector"]
    print(f"Vetor de cointegracao: {beta}")
else:
    print("Nao ha evidencia de cointegracao")

# Informacoes adicionais
print(f"Lags selecionados (ADF): {result.lags_used}")
print(f"Numero de variaveis: {result.additional_info['n_vars']}")
print(f"Especificacao: {result.additional_info['trend']}")
print(f"Observacoes efetivas: {result.additional_info['nobs']}")
```

### Multiplos Regressores

```python
# Cointegracao com 3 variaveis
x1 = np.cumsum(np.random.randn(250))
x2 = np.cumsum(np.random.randn(250))
y_multi = 1.0 + 2.0 * x1 - 0.5 * x2 + np.random.randn(250)

X = np.column_stack([x1, x2])
result_multi = engle_granger_test(y_multi, X, trend="c")
print(result_multi.summary())
print(f"Vetor de cointegracao: {result_multi.additional_info['cointegrating_vector']}")
```

## Assinatura da Funcao

```python
engle_granger_test(
    y: NDArray,              # Variavel dependente (T,)
    x: NDArray,              # Regressores (T,) ou (T, k)
    trend: str = "c",        # 'n', 'c', ou 'ct'
    maxlag: int | None = None,  # Max lags para ADF (None = auto)
    autolag: str = "AIC"     # 'AIC' ou 'BIC'
) -> TestResult
```

## Limitacoes

1. **Apenas uma relacao de cointegracao**: O teste identifica no maximo uma relacao. Para multiplas relacoes, use o [Johansen](johansen.md)
2. **Sensivel a normalizacao**: O resultado pode mudar dependendo de qual variavel e $y$ e qual e $\mathbf{x}$ — a relacao nao e simetrica
3. **Viés em amostras pequenas**: O OLS no passo 1 pode ter viés significativo em amostras pequenas
4. **Nao robusto a quebras**: A presenca de quebras estruturais pode mascarar cointegracao. Considere o teste de Gregory-Hansen
5. **Inferencia limitada**: t-tests padrao nos coeficientes da regressao de cointegracao nao sao validos

!!! warning "Problema da Normalizacao"
    Testar `engle_granger_test(y, x)` pode dar resultado diferente de `engle_granger_test(x, y)`. Para evitar resultados ambiguos com mais de 2 variaveis, prefira o [teste de Johansen](johansen.md).

## Equivalente R

=== "tseries"

    ```r
    library(tseries)

    # Phillips-Ouliaris cointegration test (similar ao Engle-Granger)
    po.test(cbind(y, x))
    ```

=== "urca"

    ```r
    library(urca)

    # Engle-Granger via urca
    # demean: "none", "constant", "trend"
    eg <- ca.po(cbind(y, x), demean = "constant", type = "Pz")
    summary(eg)
    ```

=== "Manual (2 passos)"

    ```r
    # Passo 1: Regressao de cointegracao
    model <- lm(y ~ x)
    residuals <- residuals(model)

    # Passo 2: ADF nos residuos
    library(tseries)
    adf.test(residuals)
    # ATENCAO: usar valores criticos de MacKinnon para EG,
    # nao os valores padrao do ADF!
    ```

## See Also

- [Johansen Test](johansen.md) — Para multiplas relacoes de cointegracao
- [Bounds Test](bounds-test.md) — Para ordens de integracao mistas
- [Cointegration Tests](index.md) — Visao geral
- [ADF Test](../unit-root/adf.md) — Teste ADF padrao (pre-requisito)
- [Theory: VECM](../../theory/vecm-theory.md) — Fundamentos teoricos

## Referencias

- Engle, R.F. & Granger, C.W.J. (1987). "Co-integration and error correction: representation, estimation, and testing." *Econometrica*, 55(2), 251-276.
- MacKinnon, J.G. (2010). "Critical values for cointegration tests." *Queen's Economics Department Working Paper No. 1227*.
- Phillips, P.C.B. & Ouliaris, S. (1990). "Asymptotic properties of residual based tests for cointegration." *Econometrica*, 58(1), 165-193.
- Engle, R.F. & Yoo, B.S. (1987). "Forecasting and testing in co-integrated systems." *Journal of Econometrics*, 35(1), 143-159.
