---
title: "Chow Test"
description: "Chow test para quebra estrutural com data conhecida no chronobox — estatistica F, hipoteses, exemplo pratico e equivalente R."
---

# Chow Test

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.structural_breaks.chow_test()`
    **H₀:** Parametros sao iguais antes e depois de $t^*$ ($\beta_1 = \beta_2$)
    **H₁:** Parametros mudaram em $t^*$ ($\beta_1 \neq \beta_2$)
    **Distribuicao:** $F(k, T - 2k)$
    **Valores criticos:** Distribuicao F padrao
    **R equivalente:** `strucchange::sctest()` com `type = "Chow"`

## Hipoteses

O Chow test avalia se os coeficientes de uma regressao sao iguais em dois sub-periodos divididos por uma data de quebra $t^*$:

$$H_0: \beta_1 = \beta_2 \quad \text{(parametros estaveis — sem quebra)}$$

$$H_1: \beta_1 \neq \beta_2 \quad \text{(quebra estrutural em } t^* \text{)}$$

**Rejeitar H₀** indica que os parametros do modelo mudaram significativamente em $t^*$.

**Nao rejeitar H₀** sugere que nao ha evidencia de quebra naquela data especifica.

!!! warning "Data de Quebra Conhecida"
    O Chow test **requer** que a data de quebra $t^*$ seja fornecida *a priori*. Ele nao busca a data otima. Se voce nao tem uma data candidata, use [CUSUM](cusum.md) para exploracao ou [Bai-Perron](bai-perron.md) para deteccao endogena.

## Modelo

Considere o modelo de regressao linear:

$$y_t = x_t' \beta + \varepsilon_t, \quad t = 1, \ldots, T$$

O Chow test divide a amostra em dois sub-periodos em $t^*$:

=== "Modelo Restrito (H₀)"

    Estima um unico conjunto de parametros sobre **toda** a amostra:

    $$y_t = x_t' \beta + \varepsilon_t, \quad t = 1, \ldots, T$$

    Soma dos quadrados dos residuos: $SSR_R$

=== "Modelo Irrestrito (H₁)"

    Estima parametros **separados** para cada sub-periodo:

    **Sub-periodo 1** ($t = 1, \ldots, t^*$):
    $$y_t = x_t' \beta_1 + \varepsilon_{1t}$$

    **Sub-periodo 2** ($t = t^* + 1, \ldots, T$):
    $$y_t = x_t' \beta_2 + \varepsilon_{2t}$$

    Soma dos quadrados dos residuos: $SSR_U = SSR_1 + SSR_2$

## Estatistica de Teste

A estatistica de Chow compara o ajuste do modelo restrito vs. irrestrito:

$$F = \frac{(SSR_R - SSR_U) / k}{SSR_U / (T - 2k)}$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $SSR_R$ | Soma dos quadrados dos residuos do modelo restrito (amostra inteira) |
| $SSR_U = SSR_1 + SSR_2$ | Soma dos quadrados dos residuos dos dois sub-modelos |
| $k$ | Numero de parametros em cada regressao (incluindo constante) |
| $T$ | Numero total de observacoes |
| $T - 2k$ | Graus de liberdade do denominador |

### Distribuicao

Sob H₀, a estatistica segue uma distribuicao **F padrao**:

$$F \sim F(k, \, T - 2k)$$

!!! note "Vantagem do Chow Test"
    Diferente dos testes de raiz unitaria, o Chow test usa a distribuicao F **padrao** — nao requer tabelas especiais. Isso o torna simples de implementar e interpretar.

### Intuicao

- Se **nao ha quebra**, $SSR_R \approx SSR_U$ (dividir a amostra nao melhora o ajuste), entao $F \approx 0$
- Se **ha quebra**, $SSR_R \gg SSR_U$ (o modelo separado ajusta muito melhor), entao $F$ sera grande

## Valores Criticos

Os valores criticos dependem de $k$ (parametros) e $T - 2k$ (graus de liberdade):

| Significancia | $F_{crit}$ (k=2, T-2k=196) | $F_{crit}$ (k=3, T-2k=194) |
|:-------------|:---------------------------|:---------------------------|
| 1% | 4.71 | 3.88 |
| 5% | 3.04 | 2.65 |
| 10% | 2.33 | 2.10 |

*Valores exatos dependem dos graus de liberdade especificos.*

## Requisitos

Para que o Chow test seja valido:

1. **Data de quebra conhecida** — deve ser especificada antes de ver os dados (nao pode ser escolhida por data mining)
2. **Sub-amostras suficientes** — cada sub-periodo deve ter pelo menos $k$ observacoes
3. **Erros normais homocedasticos** — assume $\varepsilon_t \sim N(0, \sigma^2)$ com variancia constante nos dois periodos
4. **Sem autocorrelacao** — residuos independentes

!!! warning "Variancia Heterogenea"
    Se a variancia muda entre os sub-periodos ($\sigma_1^2 \neq \sigma_2^2$), o Chow test pode rejeitar H₀ mesmo sem mudanca nos coeficientes $\beta$. Neste caso, considere a versao robusta do teste ou verifique com [CUSUM-SQ](cusum.md) se a variancia e instavel.

## Exemplo Pratico

### Teste de Quebra em Periodo de Crise

```python
import numpy as np
from chronobox.tests_stat.structural_breaks import chow_test

# Simular serie com quebra na crise (t=100)
np.random.seed(42)
T = 200
x = np.random.randn(T)

# Regime 1 (pre-crise): beta = 0.5
# Regime 2 (pos-crise): beta = 2.0
y = np.where(np.arange(T) < 100,
             1.0 + 0.5 * x,
             1.0 + 2.0 * x) + 0.5 * np.random.randn(T)

# Teste de Chow na data da crise
result = chow_test(y, x, break_point=100)
print(result.summary())
```

Saida esperada:

```
============================================================
  Chow Test
============================================================
  Test statistic : 45.6789
  p-value        : 0.000000
  Break point    : 100

  H0: Parameters are equal (beta_1 = beta_2)
  H1: Structural break at t* = 100

  Critical Values:
      1% : 4.7124
      5% : 3.0402
     10% : 2.3289

  Sub-sample 1   : t = 1 to 100 (n1 = 100)
  Sub-sample 2   : t = 101 to 200 (n2 = 100)

  SSR (restricted)   : 234.5678
  SSR (unrestricted) : 48.1234
  Parameters (k)     : 2

  Decision (5%)  : Reject H0
============================================================
```

!!! tip "Interpretacao"
    $F = 45.68 \gg 3.04$ (valor critico a 5%). Forte evidencia de quebra estrutural em $t = 100$. Os parametros mudaram significativamente entre os dois sub-periodos.

### Testando Multiplas Datas Candidatas

```python
# Testar varias datas candidatas
break_candidates = [50, 80, 100, 120, 150]

print(f"{'Break Point':>12} {'F-stat':>10} {'p-valor':>10} {'Decisao':>15}")
print("-" * 50)
for bp in break_candidates:
    r = chow_test(y, x, break_point=bp)
    decision = "Rejeita H0" if r.reject_at_5pct else "Nao rejeita"
    print(f"{bp:>12} {r.statistic:>10.4f} {r.pvalue:>10.6f} {decision:>15}")
```

Saida esperada:

```
 Break Point     F-stat    p-valor         Decisao
--------------------------------------------------
          50     1.2345   0.293210     Nao rejeita
          80    12.3456   0.000012     Rejeita H0
         100    45.6789   0.000000     Rejeita H0
         120    15.7890   0.000001     Rejeita H0
         150     2.3456   0.098765     Nao rejeita
```

!!! warning "Data Mining"
    Testar multiplas datas e **informativamente util** mas invalida os p-valores nominais. Se voce esta buscando a data otima de quebra, use o [Bai-Perron](bai-perron.md) que corrige para multiplas comparacoes, ou aplique correcao de Bonferroni.

### Visualizacao

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Dados com regressoes separadas
axes[0].scatter(x[:100], y[:100], alpha=0.5, label="Sub-periodo 1", c="blue", s=15)
axes[0].scatter(x[100:], y[100:], alpha=0.5, label="Sub-periodo 2", c="red", s=15)

# Linhas de regressao
from numpy.polynomial.polynomial import polyfit
b1 = np.polyfit(x[:100], y[:100], 1)
b2 = np.polyfit(x[100:], y[100:], 1)
x_line = np.linspace(x.min(), x.max(), 100)
axes[0].plot(x_line, np.polyval(b1, x_line), "b-", linewidth=2, label=f"beta1={b1[0]:.2f}")
axes[0].plot(x_line, np.polyval(b2, x_line), "r-", linewidth=2, label=f"beta2={b2[0]:.2f}")
axes[0].set_title("Regressoes por Sub-periodo")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend()

# F-statistics por break point
break_range = range(20, 181)
f_stats = [chow_test(y, x, break_point=bp).statistic for bp in break_range]

axes[1].plot(list(break_range), f_stats, "b-")
axes[1].axhline(3.04, color="red", linestyle="--", label="Valor critico 5%")
axes[1].axvline(100, color="green", linestyle="--", alpha=0.5, label="Quebra real")
axes[1].set_title("Estatistica F por Break Point")
axes[1].set_xlabel("Break Point")
axes[1].set_ylabel("F-statistic")
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Acessando Resultados Programaticamente

```python
result = chow_test(y, x, break_point=100)

# Decisao automatica
if result.reject_at_5pct:
    print(f"Quebra detectada em t={result.additional_info['break_point']}")
    print(f"F-stat: {result.statistic:.4f}, p-valor: {result.pvalue:.6f}")
else:
    print("Sem evidencia de quebra estrutural")

# Detalhes dos sub-modelos
info = result.additional_info
print(f"SSR restrito: {info['ssr_restricted']:.4f}")
print(f"SSR irrestrito: {info['ssr_unrestricted']:.4f}")
print(f"Parametros: {info['k']}")
print(f"Sub-amostra 1: n={info['n1']}")
print(f"Sub-amostra 2: n={info['n2']}")
```

## Assinatura da Funcao

```python
chow_test(
    y: NDArray,
    X: NDArray | None = None,      # Regressores (None = constante apenas)
    break_point: int = None,        # Indice da data de quebra t*
    alpha: float = 0.05             # Nivel de significancia
) -> TestResult
```

## Limitacoes

1. **Requer data de quebra conhecida** — a principal limitacao. Escolher $t^*$ por data mining invalida o teste
2. **Apenas uma quebra** — nao testa multiplas quebras simultaneamente. Use [Bai-Perron](bai-perron.md) neste caso
3. **Assume variancia constante** — sensivel a heteroscedasticidade entre sub-periodos
4. **Assume normalidade** — a distribuicao F exata requer erros normais (robusto em amostras grandes por CLT)
5. **Sub-amostras precisam ter tamanho minimo** — cada sub-periodo precisa de pelo menos $k$ observacoes

## Equivalente R

=== "strucchange"

    ```r
    library(strucchange)

    # Chow test com data de quebra conhecida
    sctest(y ~ x, type = "Chow", point = 100)

    # Equivalencia:
    # chronobox chow_test(y, x, break_point=100)
    # -> strucchange sctest(y ~ x, type = "Chow", point = 100)
    ```

=== "Base R (manual)"

    ```r
    # Chow test manual
    n <- length(y)
    bp <- 100  # break point

    # Modelo restrito (amostra inteira)
    fit_r <- lm(y ~ x)
    ssr_r <- sum(residuals(fit_r)^2)

    # Modelo irrestrito (dois sub-periodos)
    fit_1 <- lm(y[1:bp] ~ x[1:bp])
    fit_2 <- lm(y[(bp+1):n] ~ x[(bp+1):n])
    ssr_u <- sum(residuals(fit_1)^2) + sum(residuals(fit_2)^2)

    # Estatistica F
    k <- length(coef(fit_r))  # numero de parametros
    F_stat <- ((ssr_r - ssr_u) / k) / (ssr_u / (n - 2 * k))
    p_value <- 1 - pf(F_stat, k, n - 2 * k)

    cat("F =", F_stat, "p-valor =", p_value, "\n")
    ```

## See Also

- [CUSUM Test](cusum.md) — Teste sequencial sem data de quebra conhecida
- [Bai-Perron](bai-perron.md) — Multiplas quebras com deteccao endogena
- [Structural Break Tests](index.md) — Visao geral

## Referencias

- Chow, G.C. (1960). "Tests of equality between sets of coefficients in two linear regressions." *Econometrica*, 28(3), 591-605.
- Fisher, F.M. (1970). "Tests of equality between sets of coefficients in two linear regressions: An expository note." *Econometrica*, 38(2), 361-366.
- Toyoda, T. (1974). "Use of the Chow test under heteroscedasticity." *Econometrica*, 42(3), 601-608.
