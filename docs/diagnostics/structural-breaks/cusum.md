---
title: "CUSUM Test"
description: "Testes CUSUM e CUSUM-SQ para estabilidade parametrica no chronobox — residuos recursivos, bandas de confianca e exemplos com plots."
---

# CUSUM e CUSUM-SQ

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.structural_breaks.cusum_test()`
    **H₀:** Parametros sao estaveis ao longo da amostra
    **H₁:** Existe instabilidade parametrica (quebra estrutural)
    **Distribuicao:** Movimento Browniano (limites tabelados)
    **Valores criticos:** Bandas lineares baseadas em $\pm 0.948 \sqrt{T - k}$ (CUSUM) ou limites curvos (CUSUM-SQ)
    **R equivalente:** `strucchange::efp()` + `strucchange::sctest()`

## Hipoteses

O CUSUM testa a estabilidade dos coeficientes de regressao ao longo do tempo:

$$H_0: \beta_t = \beta \quad \forall \, t \quad \text{(parametros estaveis)}$$

$$H_1: \beta_t \neq \beta \quad \text{para algum } t \quad \text{(instabilidade parametrica)}$$

**Rejeitar H₀** indica que os parametros do modelo mudaram em algum ponto da amostra — os coeficientes estimados sobre a amostra inteira nao sao confiaveis.

**Nao rejeitar H₀** sugere que o modelo e estavel e pode ser estimado sobre toda a amostra.

## Residuos Recursivos

O CUSUM e baseado em **residuos recursivos** (recursive residuals), que sao estimativas one-step-ahead:

$$w_t = \frac{y_t - x_t' \hat{\beta}_{t-1}}{\sqrt{1 + x_t' (X_{t-1}'X_{t-1})^{-1} x_t}}, \quad t = k+1, \ldots, T$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $\hat{\beta}_{t-1}$ | OLS estimado com as primeiras $t-1$ observacoes |
| $x_t$ | Vetor de regressores na observacao $t$ |
| $X_{t-1}$ | Matriz de regressores das primeiras $t-1$ observacoes |
| $k$ | Numero de regressores (incluindo constante) |

!!! note "Propriedade Fundamental"
    Sob H₀ (estabilidade), os residuos recursivos $w_t$ sao **iid** $N(0, \sigma^2)$. Se houver uma quebra, os residuos apos a quebra terao media sistematicamente diferente de zero.

## Estatistica CUSUM

A estatistica CUSUM e a soma acumulada dos residuos recursivos padronizados:

$$W_t = \frac{1}{\hat{\sigma}} \sum_{j=k+1}^{t} w_j, \quad t = k+1, \ldots, T$$

onde $\hat{\sigma}$ e o desvio padrao estimado dos residuos recursivos:

$$\hat{\sigma} = \sqrt{\frac{1}{T - k} \sum_{j=k+1}^{T} w_j^2}$$

### Bandas de Confianca

Sob H₀, $W_t$ segue aproximadamente um **movimento Browniano**. As bandas de confianca sao **linhas retas** que partem de zero em $t = k$ e aumentam linearmente:

$$\pm \, a \sqrt{T - k} \cdot \left(1 + 2 \cdot \frac{t - k}{T - k}\right)$$

onde $a$ depende do nivel de significancia:

| Significancia | $a$ |
|:-------------|:----|
| 1% | 1.143 |
| 5% | 0.948 |
| 10% | 0.850 |

!!! tip "Regra de Decisao"
    Se a curva $W_t$ **ultrapassar** as bandas de confianca em qualquer ponto, rejeita-se H₀. O ponto onde a curva cruza a banda indica aproximadamente a data da quebra.

## Estatistica CUSUM-SQ

O CUSUM-SQ (CUSUM of Squares) utiliza a soma acumulada dos **quadrados** dos residuos recursivos:

$$S_t = \frac{\sum_{j=k+1}^{t} w_j^2}{\sum_{j=k+1}^{T} w_j^2}, \quad t = k+1, \ldots, T$$

Note que $S_{k} = 0$ e $S_T = 1$ por construcao.

### Diferenca entre CUSUM e CUSUM-SQ

| Caracteristica | CUSUM | CUSUM-SQ |
|:---------------|:------|:---------|
| Baseado em | Residuos recursivos ($w_t$) | Quadrados dos residuos ($w_t^2$) |
| Detecta | Mudanca na **media** dos coeficientes | Mudanca na **variancia** ou desvios aleatorios |
| Bandas | Linhas retas | Curvas (baseadas na distribuicao Beta) |
| Poder | Melhor para quebras **abruptas** | Melhor para instabilidade **gradual** |
| Linha de referencia | Zero | Linha reta de 0 a 1 (sob H₀: $E[S_t] = (t-k)/(T-k)$) |

!!! warning "Complementaridade"
    CUSUM e CUSUM-SQ detectam tipos diferentes de instabilidade. E recomendavel aplicar **ambos** os testes. Um modelo pode passar no CUSUM (media dos parametros estavel) mas falhar no CUSUM-SQ (variancia instavel).

## Exemplo Pratico

### CUSUM com Serie Estavel

```python
import numpy as np
from chronobox.tests_stat.structural_breaks import cusum_test

# Serie estavel: y = 1 + 0.5*x + eps
np.random.seed(42)
T = 200
x = np.random.randn(T)
y = 1.0 + 0.5 * x + np.random.randn(T)

result = cusum_test(y, x, test_type="cusum")
print(result.summary())
```

Saida esperada:

```
============================================================
  CUSUM Test
============================================================
  Test statistic : 0.6234
  p-value        : 0.4521
  Max |CUSUM|    : 5.123

  H0: Parameters are stable
  H1: Structural instability present

  Critical Values (boundaries):
      1% : 15.234
      5% : 12.645
     10% : 11.334

  Decision (5%)  : Do not reject H0
============================================================
```

### CUSUM com Quebra Estrutural

```python
# Serie com quebra em t=100: beta muda de 0.5 para 2.0
y_break = np.where(np.arange(T) < 100,
                   1.0 + 0.5 * x,
                   1.0 + 2.0 * x) + np.random.randn(T)

result_break = cusum_test(y_break, x, test_type="cusum")
print(result_break.summary())
# Esperado: rejeita H0 — curva CUSUM ultrapassa as bandas
```

### Plot CUSUM

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CUSUM
cusum_result = cusum_test(y_break, x, test_type="cusum")
t_vals = cusum_result.additional_info["t_values"]
cusum_vals = cusum_result.additional_info["cusum_values"]
upper = cusum_result.additional_info["upper_band"]
lower = cusum_result.additional_info["lower_band"]

axes[0].plot(t_vals, cusum_vals, "b-", label="CUSUM")
axes[0].plot(t_vals, upper, "r--", label="5% banda superior")
axes[0].plot(t_vals, lower, "r--", label="5% banda inferior")
axes[0].axhline(0, color="gray", linestyle=":")
axes[0].axvline(100, color="green", linestyle="--", alpha=0.5, label="Quebra real")
axes[0].set_title("CUSUM Test")
axes[0].set_xlabel("Observacao")
axes[0].set_ylabel("CUSUM")
axes[0].legend()

# CUSUM-SQ
cusumsq_result = cusum_test(y_break, x, test_type="cusumsq")
s_vals = cusumsq_result.additional_info["cusumsq_values"]
upper_sq = cusumsq_result.additional_info["upper_band"]
lower_sq = cusumsq_result.additional_info["lower_band"]
ref_line = cusumsq_result.additional_info["reference_line"]

axes[1].plot(t_vals, s_vals, "b-", label="CUSUM-SQ")
axes[1].plot(t_vals, ref_line, "k:", label="E[S_t] sob H0")
axes[1].plot(t_vals, upper_sq, "r--", label="5% banda superior")
axes[1].plot(t_vals, lower_sq, "r--", label="5% banda inferior")
axes[1].axvline(100, color="green", linestyle="--", alpha=0.5, label="Quebra real")
axes[1].set_title("CUSUM-SQ Test")
axes[1].set_xlabel("Observacao")
axes[1].set_ylabel("CUSUM-SQ")
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Acessando Resultados Programaticamente

```python
result = cusum_test(y_break, x, test_type="cusum")

# Decisao automatica
if result.reject_at_5pct:
    print("Instabilidade detectada — modelo NAO e estavel")
else:
    print("Modelo e estavel ao longo da amostra")

# Informacoes adicionais
print(f"Estatistica maxima: {result.additional_info['max_cusum']:.4f}")
print(f"Tipo de teste: {result.additional_info['test_type']}")
print(f"Numero de regressores: {result.additional_info['k']}")
```

## Assinatura da Funcao

```python
cusum_test(
    y: NDArray,
    X: NDArray | None = None,   # Regressores (None = constante apenas)
    test_type: str = "cusum",   # 'cusum' ou 'cusumsq'
    alpha: float = 0.05         # Nivel de significancia
) -> TestResult
```

## Limitacoes

1. **Nao identifica a data exata** da quebra — apenas indica se houve instabilidade. Para datas, use [Bai-Perron](bai-perron.md)
2. **Requer modelo linear** — os residuos recursivos sao baseados em OLS
3. **Baixo poder para quebras no inicio/fim** da amostra — o CUSUM e mais sensivel a quebras no meio
4. **Sensivel ao numero de regressores** — muitos regressores reduzem o numero efetivo de residuos recursivos ($T - k$)
5. **Nao distingue o tipo de quebra** — mudanca em intercepto, inclinacao ou variancia produzem sinais semelhantes

## Equivalente R

=== "strucchange"

    ```r
    library(strucchange)

    # CUSUM
    # efp = Empirical Fluctuation Process
    cusum_proc <- efp(y ~ x, type = "Rec-CUSUM")
    plot(cusum_proc)           # Plot com bandas
    sctest(cusum_proc)         # Teste formal

    # CUSUM-SQ
    cusumsq_proc <- efp(y ~ x, type = "Rec-CUSUM", functional = "maxL2")
    # Ou diretamente:
    cusumsq_proc <- efp(y ~ x, type = "OLS-CUSUM")
    plot(cusumsq_proc)
    sctest(cusumsq_proc)

    # Equivalencias:
    # chronobox test_type="cusum"   -> strucchange type="Rec-CUSUM"
    # chronobox test_type="cusumsq" -> strucchange type="OLS-CUSUM"
    ```

=== "Base R (manual)"

    ```r
    # Residuos recursivos manualmente
    n <- length(y)
    k <- 2  # numero de parametros
    w <- numeric(n - k)

    for (t in (k + 1):n) {
      fit <- lm(y[1:(t-1)] ~ x[1:(t-1)])
      pred <- predict(fit, newdata = data.frame(x = x[t]))
      h_t <- 1 + x[t] %*% solve(t(X[1:(t-1),]) %*% X[1:(t-1),]) %*% x[t]
      w[t - k] <- (y[t] - pred) / sqrt(h_t)
    }

    sigma_hat <- sd(w)
    cusum <- cumsum(w) / sigma_hat
    ```

## See Also

- [Chow Test](chow.md) — Teste com data de quebra conhecida
- [Bai-Perron](bai-perron.md) — Multiplas quebras com datas estimadas
- [Structural Break Tests](index.md) — Visao geral
- [Zivot-Andrews](../unit-root/zivot-andrews.md) — Raiz unitaria com quebra estrutural

## Referencias

- Brown, R.L., Durbin, J. & Evans, J.M. (1975). "Techniques for testing the constancy of regression relationships over time." *JRSS-B*, 37(2), 149-192.
- Ploberger, W. & Kramer, W. (1992). "The CUSUM test with OLS residuals." *Econometrica*, 60(2), 271-285.
- Zeileis, A. (2004). "Alternative boundaries for CUSUM tests." *Statistical Papers*, 45(1), 123-131.
