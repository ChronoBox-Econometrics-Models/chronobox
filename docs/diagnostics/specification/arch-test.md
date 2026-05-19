---
title: "ARCH-LM Test"
description: "Teste ARCH-LM para heterocedasticidade condicional no chronobox — regressao dos residuos ao quadrado, estatistica LM e exemplos praticos."
---

# ARCH-LM Test

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.specification.arch_lm_test()`
    **H₀:** Sem efeitos ARCH (homocedasticidade condicional)
    **H₁:** Efeitos ARCH presentes (heterocedasticidade condicional)
    **Distribuicao:** $\chi^2(q)$
    **Valores criticos:** Distribuicao $\chi^2$ padrao
    **R equivalente:** `FinTS::ArchTest()`, `lmtest::bptest()`

## Hipoteses

O teste ARCH-LM avalia se a variancia dos residuos depende de valores passados (efeitos ARCH):

$$H_0: \alpha_1 = \alpha_2 = \cdots = \alpha_q = 0 \quad \text{(sem efeitos ARCH — variancia constante)}$$

$$H_1: \exists \, j \leq q \text{ tal que } \alpha_j \neq 0 \quad \text{(efeitos ARCH presentes)}$$

**Rejeitar H₀** indica que a volatilidade dos residuos e **previsivel** — a variancia muda ao longo do tempo de forma sistematica. Isso sugere que um modelo GARCH pode ser necessario.

**Nao rejeitar H₀** sugere que a variancia e constante (homocedastica).

## O Que Sao Efeitos ARCH?

ARCH (**A**uto**R**egressive **C**onditional **H**eteroskedasticity) significa que a variancia condicional $\sigma_t^2$ depende de erros passados ao quadrado:

$$\sigma_t^2 = \text{Var}(\varepsilon_t | \mathcal{F}_{t-1}) = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2 + \cdots + \alpha_q \varepsilon_{t-q}^2$$

Na pratica, efeitos ARCH significam que:

- Periodos de **alta volatilidade** tendem a ser seguidos por alta volatilidade
- Periodos de **baixa volatilidade** tendem a ser seguidos por baixa volatilidade
- E o fenomeno de **volatility clustering**, comum em series financeiras

## Regressao Auxiliar

O teste regride os residuos ao quadrado sobre seus proprios lags:

$$\hat{e}_t^2 = \alpha_0 + \alpha_1 \hat{e}_{t-1}^2 + \alpha_2 \hat{e}_{t-2}^2 + \cdots + \alpha_q \hat{e}_{t-q}^2 + v_t$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $\hat{e}_t^2$ | Residuos ao quadrado do modelo original |
| $\alpha_0$ | Constante |
| $\alpha_j \hat{e}_{t-j}^2$ | Residuos ao quadrado defasados |
| $q$ | Numero de lags ARCH |
| $v_t$ | Erro da regressao auxiliar |

### Passos do Teste

1. Estimar o modelo original e obter residuos $\hat{e}_t$
2. Elevar ao quadrado: $\hat{e}_t^2$
3. Regredir $\hat{e}_t^2$ sobre $\hat{e}_{t-1}^2, \ldots, \hat{e}_{t-q}^2$
4. Calcular $R^2$ da regressao auxiliar

## Estatistica de Teste

$$LM = T \cdot R^2 \sim \chi^2(q)$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $T$ | Numero efetivo de observacoes ($T - q$ na regressao auxiliar) |
| $R^2$ | Coeficiente de determinacao da regressao de $\hat{e}_t^2$ |
| $q$ | Numero de lags ARCH testados |

### Intuicao

- Se **nao ha efeitos ARCH**, $\hat{e}_t^2$ nao e previsto por seus lags → $R^2 \approx 0$ → $LM \approx 0$
- Se **ha efeitos ARCH**, $\hat{e}_t^2$ e parcialmente previsto por seus lags → $R^2 > 0$ → $LM$ grande

## Valores Criticos

| Significancia | $q = 1$ | $q = 2$ | $q = 5$ | $q = 10$ |
|:-------------|:--------|:--------|:--------|:---------|
| 5% | 3.84 | 5.99 | 11.07 | 18.31 |
| 1% | 6.63 | 9.21 | 15.09 | 23.21 |

*Rejeita-se H₀ quando $LM > \chi^2_{1-\alpha}(q)$.*

## Exemplo Pratico

### Residuos Sem Efeitos ARCH

```python
import numpy as np
from chronobox.tests_stat.specification import arch_lm_test

# Residuos homocedasticos (sem ARCH)
np.random.seed(42)
residuals = np.random.randn(500)

result = arch_lm_test(residuals, nlags=5)
print(result.summary())
```

Saida esperada:

```
============================================================
  ARCH-LM Test
============================================================
  Test statistic : 2.3456
  p-value        : 0.7994

  H0: No ARCH effects up to order 5
  H1: ARCH effects present

  Critical Values:
      5% : 11.0705
      1% : 15.0863

  Decision (5%)  : Do not reject H0
============================================================
```

!!! tip "Interpretacao"
    $LM = 2.35 < 11.07$ (valor critico a 5%). Nao rejeitamos H₀: sem evidencia de efeitos ARCH.

### Residuos Com Efeitos ARCH

```python
# Simular processo ARCH(1): sigma_t^2 = 0.2 + 0.7 * e_{t-1}^2
np.random.seed(42)
T = 500
eps = np.zeros(T)
sigma2 = np.zeros(T)
sigma2[0] = 1.0

for t in range(1, T):
    sigma2[t] = 0.2 + 0.7 * eps[t - 1] ** 2
    eps[t] = np.sqrt(sigma2[t]) * np.random.randn()

result_arch = arch_lm_test(eps, nlags=5)
print(f"LM = {result_arch.statistic:.4f}, p = {result_arch.pvalue:.6f}")
print(f"R2 = {result_arch.additional_info['R_squared']:.4f}")
# Esperado: rejeita H0 fortemente
```

### Testando Diferentes Ordens

```python
# Qual ordem de ARCH?
print(f"{'q':>3} {'LM':>10} {'R2':>8} {'p-valor':>10} {'Decisao':>12}")
print("-" * 48)
for q in [1, 2, 5, 10, 15]:
    r = arch_lm_test(eps, nlags=q)
    print(f"{q:3d} {r.statistic:10.4f} {r.additional_info['R_squared']:8.4f} "
          f"{r.pvalue:10.6f} {'Rejeita' if r.reject_at_5pct else 'OK':>12}")
```

### Acessando Resultados Programaticamente

```python
result = arch_lm_test(residuals, nlags=5)

if result.reject_at_5pct:
    print("Efeitos ARCH detectados — considere modelagem GARCH")
    print(f"LM = {result.statistic:.4f}, p = {result.pvalue:.6f}")
    print(f"R2 da regressao auxiliar: {result.additional_info['R_squared']:.4f}")
else:
    print("Variancia constante — OLS e eficiente")
```

## Assinatura da Funcao

```python
arch_lm_test(
    residuals: NDArray,
    nlags: int = 1             # Ordem ARCH (q)
) -> TestResult
```

## Implicacoes Praticas

| Resultado | Implicacao | Acao |
|:----------|:-----------|:-----|
| Nao rejeita H₀ | Variancia constante | OLS e eficiente, erros-padrao validos |
| Rejeita H₀ | Volatility clustering | Usar modelos GARCH, ou erros-padrao robustos (HAC) |

!!! warning "ARCH e Intervalos de Confianca"
    Se efeitos ARCH estao presentes, os **intervalos de confianca** do modelo serao incorretos — subestimados em periodos de alta volatilidade e superestimados em periodos de baixa volatilidade. A previsao pontual nao e afetada, mas as bandas de confianca sao.

## Limitacoes

1. **Detecta apenas ARCH linear** — nao detecta formas nao-lineares de heterocedasticidade (ex: EGARCH, GJR-GARCH)
2. **Sensivel a outliers** — residuos ao quadrado amplificam outliers, podendo levar a rejeicao espuria
3. **Assintotico** — em amostras pequenas, pode ter distorcao de tamanho
4. **Nao indica a ordem** — se rejeitado, a ordem GARCH otima deve ser selecionada separadamente

## Equivalente R

=== "FinTS"

    ```r
    library(FinTS)

    # ARCH-LM test
    ArchTest(residuals, lags = 5)

    # Equivalencia:
    # chronobox arch_lm_test(residuals, nlags=5)
    # -> R ArchTest(residuals, lags=5)
    ```

=== "Base R (manual)"

    ```r
    # ARCH-LM test manual
    e2 <- residuals^2
    n <- length(e2)
    q <- 5  # lags

    # Regressao auxiliar
    dep <- e2[(q+1):n]
    regressors <- matrix(NA, nrow = n - q, ncol = q)
    for (j in 1:q) {
        regressors[, j] <- e2[(q+1-j):(n-j)]
    }
    aux <- lm(dep ~ regressors)
    r2 <- summary(aux)$r.squared

    # LM statistic
    LM <- (n - q) * r2
    p_value <- 1 - pchisq(LM, df = q)
    cat("LM =", LM, "p-valor =", p_value, "\n")
    ```

## See Also

- [Jarque-Bera](jarque-bera.md) — Teste de normalidade (residuos ARCH tendem a ter caudas pesadas)
- [Specification Tests](index.md) — Visao geral de testes de especificacao

## Referencias

- Engle, R.F. (1982). "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation." *Econometrica*, 50(4), 987-1007.
- Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.
- Tsay, R.S. (2010). *Analysis of Financial Time Series*. 3rd ed., Wiley. Chapter 3.
