---
title: "Jarque-Bera Test"
description: "Teste Jarque-Bera para normalidade dos residuos no chronobox — skewness, kurtosis, estatistica JB e exemplos praticos."
---

# Jarque-Bera Test

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.specification.jarque_bera_test()`
    **H₀:** Residuos sao normalmente distribuidos ($S = 0$, $K = 3$)
    **H₁:** Residuos nao sao normais ($S \neq 0$ ou $K \neq 3$)
    **Distribuicao:** $\chi^2(2)$
    **Valores criticos:** 5.991 (5%), 9.210 (1%)
    **R equivalente:** `tseries::jarque.bera.test()`

## Hipoteses

O teste Jarque-Bera avalia se os residuos seguem uma distribuicao normal:

$$H_0: S = 0 \text{ e } K = 3 \quad \text{(normalidade)}$$

$$H_1: S \neq 0 \text{ ou } K \neq 3 \quad \text{(nao-normalidade)}$$

onde $S$ e a skewness (assimetria) e $K$ e a kurtosis.

**Rejeitar H₀** indica que os residuos nao sao normais — podem ter assimetria, caudas pesadas, ou ambos.

**Nao rejeitar H₀** sugere que a hipotese de normalidade nao pode ser rejeitada.

## Componentes: Skewness e Kurtosis

### Skewness (Assimetria)

$$S = \frac{1}{T} \sum_{t=1}^{T} \left( \frac{\hat{e}_t - \bar{e}}{\hat{\sigma}} \right)^3$$

| Valor | Significado |
|:------|:------------|
| $S = 0$ | Distribuicao simetrica (como a normal) |
| $S > 0$ | Cauda direita mais longa (assimetria positiva) |
| $S < 0$ | Cauda esquerda mais longa (assimetria negativa) |

### Kurtosis (Curtose)

$$K = \frac{1}{T} \sum_{t=1}^{T} \left( \frac{\hat{e}_t - \bar{e}}{\hat{\sigma}} \right)^4$$

| Valor | Significado |
|:------|:------------|
| $K = 3$ | Mesma curtose que a normal (**mesocurtica**) |
| $K > 3$ | Caudas mais pesadas que a normal (**leptocurtica**) |
| $K < 3$ | Caudas mais leves que a normal (**platicurtica**) |

!!! note "Kurtosis vs. Excesso de Kurtosis"
    O Jarque-Bera usa kurtosis ($K$) com referencia 3. Alguns softwares reportam **excesso de kurtosis** ($K - 3$), com referencia 0. O chronobox reporta ambas em `additional_info`.

## Estatistica de Teste

$$JB = \frac{T}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right)$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $T$ | Numero de observacoes |
| $S$ | Skewness amostral |
| $K$ | Kurtosis amostral |
| $S^2$ | Contribuicao da assimetria |
| $(K-3)^2/4$ | Contribuicao do excesso de kurtosis |

### Distribuicao

Sob H₀ (normalidade):

$$JB \sim \chi^2(2)$$

Os dois graus de liberdade correspondem as duas restricoes testadas simultaneamente: $S = 0$ e $K = 3$.

### Intuicao

- Se $S \approx 0$ e $K \approx 3$: $JB \approx 0$ → nao rejeita H₀
- Se $S$ e grande ou $K$ difere muito de 3: $JB$ sera grande → rejeita H₀

## Valores Criticos

A distribuicao $\chi^2(2)$ tem valores criticos fixos:

| Significancia | Valor Critico |
|:-------------|:-------------|
| 10% | 4.605 |
| 5% | **5.991** |
| 1% | 9.210 |

*Rejeita-se H₀ quando $JB > \chi^2_{1-\alpha}(2)$.*

## Importancia para Inferencia

A normalidade dos residuos e necessaria para:

| Aplicacao | Sem Normalidade |
|:----------|:----------------|
| Testes t e F | Validos apenas aproximadamente (pelo CLT em amostras grandes) |
| Intervalos de confianca | Cobertura incorreta em amostras finitas |
| Previsao pontual | **Nao afetada** — OLS e BLUE independente de normalidade |
| Intervalos de previsao | Construidos sob normalidade; incorretos se caudas sao pesadas |
| Testes de raiz unitaria | Distribuicoes dependem de normalidade |

!!! tip "Normalidade e o Menos Critico"
    Entre as tres hipoteses classicas (sem autocorrelacao, homocedasticidade, normalidade), a normalidade e a **menos critica**. Em amostras grandes ($T > 100$), o CLT garante que os estimadores sao aproximadamente normais mesmo sem normalidade dos erros. Foque primeiro em autocorrelacao e heterocedasticidade.

## Exemplo Pratico

### Residuos Normais

```python
import numpy as np
from chronobox.tests_stat.specification import jarque_bera_test

# Residuos normais
np.random.seed(42)
residuals = np.random.randn(300)

result = jarque_bera_test(residuals)
print(result.summary())
```

Saida esperada:

```
============================================================
  Jarque-Bera Test
============================================================
  Test statistic : 1.2345
  p-value        : 0.5393

  H0: Normality (skewness=0, kurtosis=3)
  H1: Non-normality

  Critical Values:
      5% : 5.9915
      1% : 9.2103

  Decision (5%)  : Do not reject H0
============================================================
```

!!! tip "Interpretacao"
    $JB = 1.23 < 5.99$ (valor critico a 5%). Nao rejeitamos H₀: os residuos sao compativeis com a distribuicao normal.

### Residuos Com Caudas Pesadas

```python
# Residuos t-Student com 3 graus de liberdade (caudas pesadas)
np.random.seed(42)
residuals_t = np.random.standard_t(df=3, size=300)

result_t = jarque_bera_test(residuals_t)
print(f"JB = {result_t.statistic:.4f}, p = {result_t.pvalue:.6f}")
print(f"Skewness = {result_t.additional_info['skewness']:.4f}")
print(f"Kurtosis = {result_t.additional_info['kurtosis']:.4f}")
print(f"Excesso  = {result_t.additional_info['excess_kurtosis']:.4f}")
# Esperado: rejeita H0 (kurtosis > 3)
```

### Residuos Assimetricos

```python
# Distribuicao log-normal (assimetria positiva)
np.random.seed(42)
residuals_skew = np.random.lognormal(0, 0.5, 300)
residuals_skew -= residuals_skew.mean()  # centralizar

result_skew = jarque_bera_test(residuals_skew)
print(f"JB = {result_skew.statistic:.4f}, p = {result_skew.pvalue:.6f}")
print(f"Skewness = {result_skew.additional_info['skewness']:.4f}")
print(f"Kurtosis = {result_skew.additional_info['kurtosis']:.4f}")
# Esperado: rejeita H0 (assimetria positiva)
```

### Decompondo as Contribuicoes

```python
result = jarque_bera_test(residuals_t)
T = result.additional_info['nobs']
S = result.additional_info['skewness']
K = result.additional_info['kurtosis']

contrib_skew = T / 6 * S**2
contrib_kurt = T / 6 * (K - 3)**2 / 4

print(f"Contribuicao da skewness:  {contrib_skew:.4f} "
      f"({100*contrib_skew/result.statistic:.1f}%)")
print(f"Contribuicao da kurtosis:  {contrib_kurt:.4f} "
      f"({100*contrib_kurt/result.statistic:.1f}%)")
print(f"Total (JB):                {result.statistic:.4f}")
```

### Acessando Resultados Programaticamente

```python
result = jarque_bera_test(residuals)

if result.reject_at_5pct:
    info = result.additional_info
    print(f"Nao-normalidade detectada (JB={result.statistic:.4f})")
    if abs(info['skewness']) > 0.5:
        print(f"  Assimetria significativa: S={info['skewness']:.4f}")
    if abs(info['excess_kurtosis']) > 1:
        print(f"  Caudas pesadas: excesso kurtosis={info['excess_kurtosis']:.4f}")
    print("  Considere: erros robustos, bootstrap, ou transformacao da variavel")
else:
    print(f"Normalidade OK (p={result.pvalue:.4f})")
```

## Assinatura da Funcao

```python
jarque_bera_test(
    residuals: NDArray         # Dados ou residuos (T,)
) -> TestResult
```

## Tabela de Decisao

| p-valor | Conclusao | Acao |
|:--------|:----------|:-----|
| $p > 0.10$ | Normalidade plausivel | Inferencia padrao valida |
| $0.05 < p \leq 0.10$ | Borderline | Verificar com QQ-plot; inferencia aproximada |
| $p \leq 0.05$ | **Nao-normalidade** | Usar bootstrap ou erros robustos para inferencia |

## Limitacoes

1. **Sensivel ao tamanho da amostra** — em amostras grandes ($T > 500$), rejeita H₀ por desvios pequenos e praticamente irrelevantes
2. **Nao identifica a fonte** — para distinguir entre skewness e kurtosis, examine os componentes individuais
3. **Baixo poder** contra certas alternativas — distribuicoes com skewness e kurtosis proximas da normal mas formato diferente podem nao ser detectadas
4. **Outliers** — poucos outliers podem causar rejeicao mesmo que o restante dos dados seja normal

## Equivalente R

=== "tseries"

    ```r
    library(tseries)

    # Jarque-Bera test
    jarque.bera.test(residuals)

    # Equivalencia:
    # chronobox jarque_bera_test(residuals)
    # -> R jarque.bera.test(residuals)
    ```

=== "moments"

    ```r
    library(moments)

    # Componentes individuais
    cat("Skewness:", skewness(residuals), "\n")
    cat("Kurtosis:", kurtosis(residuals), "\n")

    # Jarque-Bera manual
    T <- length(residuals)
    S <- skewness(residuals)
    K <- kurtosis(residuals)
    JB <- T/6 * (S^2 + (K-3)^2/4)
    p_value <- 1 - pchisq(JB, df = 2)
    cat("JB =", JB, "p-valor =", p_value, "\n")
    ```

## See Also

- [ARCH-LM](arch-test.md) — Caudas pesadas frequentemente coexistem com efeitos ARCH
- [Specification Tests](index.md) — Visao geral de testes de especificacao

## Referencias

- Jarque, C.M. & Bera, A.K. (1980). "Efficient tests for normality, homoscedasticity and serial independence of regression residuals." *Economics Letters*, 6(3), 255-259.
- Jarque, C.M. & Bera, A.K. (1987). "A test for normality of observations and regression residuals." *International Statistical Review*, 55(2), 163-172.
- Bera, A.K. & Jarque, C.M. (1981). "Efficient tests for normality, homoscedasticity and serial independence of regression residuals: Monte Carlo evidence." *Economics Letters*, 7(4), 313-318.
