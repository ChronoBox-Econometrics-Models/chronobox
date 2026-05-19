---
title: "Breusch-Godfrey Test"
description: "Teste Breusch-Godfrey LM para correlacao serial no chronobox — regressao auxiliar, vantagens sobre Durbin-Watson, exemplos praticos."
---

# Breusch-Godfrey Test

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.specification.breusch_godfrey_test()`
    **H₀:** Sem correlacao serial ate ordem $p$
    **H₁:** Correlacao serial presente ate ordem $p$
    **Distribuicao:** $\chi^2(p)$ (LM) ou $F(p, T-k-p)$ (F-version)
    **Valores criticos:** Distribuicoes $\chi^2$ e $F$ padrao
    **R equivalente:** `lmtest::bgtest()`

## Hipoteses

O teste Breusch-Godfrey avalia se os residuos de uma regressao apresentam correlacao serial:

$$H_0: \alpha_1 = \alpha_2 = \cdots = \alpha_p = 0 \quad \text{(sem correlacao serial)}$$

$$H_1: \exists \, j \leq p \text{ tal que } \alpha_j \neq 0 \quad \text{(correlacao serial presente)}$$

**Rejeitar H₀** indica que os residuos possuem autocorrelacao — o modelo esta mal especificado.

**Nao rejeitar H₀** sugere que nao ha evidencia de correlacao serial ate a ordem testada.

## Vantagem Sobre Durbin-Watson

O teste de Durbin-Watson (DW) e o teste classico para autocorrelacao, mas possui **restricoes importantes**:

| Caracteristica | Durbin-Watson | Breusch-Godfrey |
|:---------------|:--------------|:----------------|
| Ordem de autocorrelacao | Apenas AR(1) | AR(1), ..., AR($p$) |
| Lagged dependent variables | **Invalido** | Valido |
| Regressores estocasticos | Problematico | Sem restricao |
| Distribuicao | Limites dL/dU | $\chi^2$ ou $F$ exata |

!!! warning "Quando Usar Breusch-Godfrey"
    Se o modelo inclui a variavel dependente defasada ($y_{t-1}$) como regressor — o que e comum em modelos ARIMA e VAR — o teste de Durbin-Watson e **enviesado para nao-rejeicao**. O Breusch-Godfrey e valido neste contexto.

## Regressao Auxiliar

O teste e baseado em uma **regressao auxiliar** dos residuos OLS:

$$\hat{e}_t = X_t' \gamma + \alpha_1 \hat{e}_{t-1} + \alpha_2 \hat{e}_{t-2} + \cdots + \alpha_p \hat{e}_{t-p} + v_t$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $\hat{e}_t$ | Residuos OLS do modelo original |
| $X_t$ | Regressores do modelo original (incluindo constante) |
| $\hat{e}_{t-j}$ | Residuos defasados (substituidos por zero para $t - j < 1$) |
| $\gamma, \alpha_j$ | Coeficientes estimados por OLS |
| $v_t$ | Erro da regressao auxiliar |

### Passos do Teste

1. Estimar o modelo original por OLS e obter residuos $\hat{e}_t$
2. Regredir $\hat{e}_t$ sobre $X_t$ e $\hat{e}_{t-1}, \ldots, \hat{e}_{t-p}$
3. Calcular $R^2$ da regressao auxiliar
4. Testar se $\alpha_1 = \cdots = \alpha_p = 0$

## Estatistica de Teste

### Versao LM ($\chi^2$)

$$LM = T \cdot R^2_{aux} \sim \chi^2(p)$$

onde $R^2_{aux}$ e o coeficiente de determinacao da regressao auxiliar.

### Versao F

$$F = \frac{R^2_{aux} / p}{(1 - R^2_{aux}) / (T - k - p)} \sim F(p, \, T - k - p)$$

onde $k$ e o numero de regressores do modelo original.

!!! note "Qual Versao Usar?"
    A versao **F** tende a ter melhores propriedades em amostras finitas. A versao **LM** ($\chi^2$) e assintotica. O chronobox reporta ambas no campo `additional_info`.

## Valores Criticos

Para a versao LM com $p$ lags:

| Significancia | $p = 1$ | $p = 2$ | $p = 4$ | $p = 12$ |
|:-------------|:--------|:--------|:--------|:---------|
| 5% | 3.84 | 5.99 | 9.49 | 21.03 |
| 1% | 6.63 | 9.21 | 13.28 | 26.22 |

*Rejeita-se H₀ quando $LM > \chi^2_{1-\alpha}(p)$.*

## Exemplo Pratico

### Modelo Bem Especificado

```python
import numpy as np
from chronobox.tests_stat.specification import breusch_godfrey_test

# Simular regressao: y = 1 + 2*x + epsilon
np.random.seed(42)
T = 200
x = np.random.randn(T)
y = 1.0 + 2.0 * x + np.random.randn(T)

# Estimar OLS manualmente
X = np.column_stack([np.ones(T), x])
beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
residuals = y - X @ beta_hat

# Teste de Breusch-Godfrey para autocorrelacao de ordem 4
result = breusch_godfrey_test(residuals, x_regressors=X, nlags=4)
print(result.summary())
```

Saida esperada:

```
============================================================
  Breusch-Godfrey Test
============================================================
  Test statistic : 3.2145
  p-value        : 0.5227

  H0: No serial correlation up to order 4
  H1: Serial correlation present

  Critical Values:
      5% : 9.4877
      1% : 13.2767

  Decision (5%)  : Do not reject H0
============================================================
```

!!! tip "Interpretacao"
    $LM = 3.21 < 9.49$ (valor critico a 5%). Nao rejeitamos H₀: sem evidencia de correlacao serial ate a ordem 4.

### Modelo Com Autocorrelacao nos Erros

```python
# Erros AR(1): e_t = 0.7 * e_{t-1} + v_t
np.random.seed(42)
T = 200
x = np.random.randn(T)
eps = np.zeros(T)
for t in range(1, T):
    eps[t] = 0.7 * eps[t - 1] + np.random.randn()
y_ar = 1.0 + 2.0 * x + eps

X = np.column_stack([np.ones(T), x])
beta_hat = np.linalg.solve(X.T @ X, X.T @ y_ar)
resid_ar = y_ar - X @ beta_hat

result_ar = breusch_godfrey_test(resid_ar, x_regressors=X, nlags=4)
print(f"LM = {result_ar.statistic:.4f}, p = {result_ar.pvalue:.6f}")
print(f"F  = {result_ar.additional_info['F_stat']:.4f}, "
      f"p(F) = {result_ar.additional_info['pvalue_F']:.6f}")
# Esperado: rejeita H0 (p-valor muito pequeno)
```

### Comparando Diferentes Ordens

```python
# Testar em diferentes ordens para ver onde a autocorrelacao aparece
for p in [1, 2, 4, 8, 12]:
    r = breusch_godfrey_test(resid_ar, x_regressors=X, nlags=p)
    print(f"p={p:2d}: LM={r.statistic:8.2f}, p={r.pvalue:.4f}, "
          f"R2_aux={r.additional_info['R_squared']:.4f}, "
          f"{'Rejeita' if r.reject_at_5pct else 'OK'}")
```

### Acessando Resultados Programaticamente

```python
result = breusch_godfrey_test(residuals, x_regressors=X, nlags=4)

# Decisao automatica
if result.reject_at_5pct:
    print(f"Correlacao serial detectada (LM={result.statistic:.4f})")
    print("Opcoes: aumentar lags, adicionar termos MA, usar HAC erros-padrao")
else:
    print(f"Residuos OK (p={result.pvalue:.4f})")

# Ambas as versoes do teste
print(f"LM (chi2):  stat={result.statistic:.4f}, p={result.pvalue:.4f}")
print(f"F-version:  stat={result.additional_info['F_stat']:.4f}, "
      f"p={result.additional_info['pvalue_F']:.4f}")
print(f"R2 auxiliar: {result.additional_info['R_squared']:.6f}")
```

## Assinatura da Funcao

```python
breusch_godfrey_test(
    residuals: NDArray,
    x_regressors: NDArray,    # Matriz de regressores (T, k)
    nlags: int = 1            # Ordem de autocorrelacao a testar
) -> TestResult
```

## Tabela de Decisao

| p-valor | Conclusao | Acao |
|:--------|:----------|:-----|
| $p > 0.10$ | Sem correlacao serial | Modelo adequado |
| $0.05 < p \leq 0.10$ | Borderline | Usar erros-padrao HAC por precaucao |
| $p \leq 0.05$ | **Correlacao serial presente** | Respecificar modelo ou usar HAC |

## Limitacoes

1. **Requer a matriz de regressores** — diferente do Ljung-Box que usa apenas residuos
2. **Assintotico** — a versao LM ($\chi^2$) pode ter distorcao de tamanho em amostras pequenas; prefira a versao F
3. **Teste conjunto** — como o Ljung-Box, nao identifica qual lag especifico e significativo
4. **Assume homocedasticidade** — sob heteroscedasticidade, a distribuicao pode nao ser exata

## Equivalente R

=== "lmtest"

    ```r
    library(lmtest)

    # Breusch-Godfrey test
    model <- lm(y ~ x)
    bgtest(model, order = 4)

    # Com tipo F
    bgtest(model, order = 4, type = "F")

    # Equivalencia:
    # chronobox breusch_godfrey_test(resid, X, nlags=4)
    # -> R bgtest(model, order=4)
    ```

=== "Base R (manual)"

    ```r
    # Regressao auxiliar manual
    model <- lm(y ~ x)
    e <- residuals(model)
    n <- length(e)

    # Criar residuos defasados
    e_lag1 <- c(0, e[-n])
    e_lag2 <- c(0, 0, e[-c(n-1, n)])

    # Regressao auxiliar
    aux <- lm(e ~ x + e_lag1 + e_lag2)
    r2 <- summary(aux)$r.squared

    # LM statistic
    LM <- n * r2
    p_value <- 1 - pchisq(LM, df = 2)
    cat("LM =", LM, "p-valor =", p_value, "\n")
    ```

## See Also

- [Ljung-Box](ljung-box.md) — Alternativa portmanteau, mais simples (nao requer regressores)
- [Specification Tests](index.md) — Visao geral de testes de especificacao

## Referencias

- Breusch, T.S. (1978). "Testing for autocorrelation in dynamic linear models." *Australian Economic Papers*, 17(31), 334-355.
- Godfrey, L.G. (1978). "Testing against general autoregressive and moving average error models when the regressors include lagged dependent variables." *Econometrica*, 46(6), 1293-1301.
- Godfrey, L.G. (1988). *Misspecification Tests in Econometrics*. Cambridge University Press.
