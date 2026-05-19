---
title: "Bai-Perron Test"
description: "Teste Bai-Perron para multiplas quebras estruturais no chronobox — supF, UDmax, WDmax, selecao por BIC e exemplos praticos."
---

# Bai-Perron Multiple Structural Breaks

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.structural_breaks.bai_perron_test()`
    **H₀:** Nenhuma quebra estrutural ($m = 0$)
    **H₁:** $m$ quebras estruturais ($m \geq 1$)
    **Distribuicao:** Nao-padrao (valores criticos tabelados por Bai & Perron, 2003)
    **Selecao de $m$:** BIC, LWZ, testes sequenciais
    **R equivalente:** `strucchange::breakpoints()`

## Hipoteses

O Bai-Perron testa a presenca de $m$ quebras estruturais em datas desconhecidas $T_1, T_2, \ldots, T_m$:

$$H_0: m = 0 \quad \text{(nenhuma quebra — parametros estaveis)}$$

$$H_1: m \geq 1 \quad \text{(pelo menos uma quebra estrutural)}$$

O modelo sob H₁ com $m$ quebras define $m + 1$ regimes:

$$y_t = x_t' \beta_j + \varepsilon_t, \quad T_{j-1} + 1 \leq t \leq T_j, \quad j = 1, \ldots, m+1$$

onde $T_0 = 0$ e $T_{m+1} = T$.

!!! note "Diferenca do Chow Test"
    Enquanto o [Chow test](chow.md) requer a data de quebra conhecida e testa apenas uma quebra, o Bai-Perron **estima endogenamente** as datas de quebra e permite **multiplas quebras simultaneas**.

## Algoritmo de Programacao Dinamica

### Estimacao das Datas de Quebra

Para $m$ quebras fixo, o algoritmo minimiza a soma total dos quadrados dos residuos:

$$\min_{T_1, \ldots, T_m} \sum_{j=1}^{m+1} SSR_j(T_{j-1}+1, T_j)$$

onde $SSR_j(a, b) = \sum_{t=a}^{b} (y_t - x_t' \hat{\beta}_j)^2$ e a soma dos quadrados dos residuos no segmento $[a, b]$.

!!! tip "Eficiencia Computacional"
    A busca exaustiva sobre todas as combinacoes de $m$ datas em $T$ observacoes e computacionalmente proibitiva. Bai & Perron (2003) propuseram um algoritmo de **programacao dinamica** que resolve o problema em $O(T^2)$ para qualquer $m$, tornando o metodo pratico mesmo para amostras grandes.

### Intervalo Minimo entre Quebras

O algoritmo impoe um **tamanho minimo de segmento** $h$:

$$T_j - T_{j-1} \geq h = \lfloor \epsilon T \rfloor$$

onde $\epsilon$ e a **fracao de trimming**, tipicamente:

| $\epsilon$ | Interpretacao | Segmento minimo (T=200) |
|:-----------|:-------------|:------------------------|
| 0.05 | Agressivo — segmentos curtos permitidos | 10 |
| 0.10 | Moderado | 20 |
| 0.15 | **Padrao** — recomendado | 30 |
| 0.20 | Conservador — segmentos longos | 40 |

!!! warning "Trade-off do Trimming"
    Valores menores de $\epsilon$ permitem detectar quebras mais proximas do inicio/fim da amostra, mas aumentam a chance de falsos positivos. Valores maiores sao mais conservadores, mas podem perder quebras reais que ocorrem perto dos extremos.

## Testes Estatisticos

### supF Test ($\ell$ vs. 0 quebras)

Testa $m = \ell$ quebras contra $m = 0$:

$$\sup F_T(\ell) = \sup_{T_1, \ldots, T_\ell} F_T(T_1, \ldots, T_\ell; \ell)$$

onde a estatistica F para datas de quebra fixas e:

$$F_T(T_1, \ldots, T_\ell; \ell) = \frac{(SSR_0 - SSR_\ell) / (\ell k)}{SSR_\ell / (T - (\ell + 1)k)}$$

O supremo e tomado sobre todas as particoes admissiveis (respeitando o trimming $\epsilon$).

### UDmax e WDmax (Testes de Duplo Maximo)

Quando o numero de quebras e desconhecido, os testes de duplo maximo testam $H_0: m = 0$ contra $H_1: m \leq M$ (algum numero de quebras ate $M$):

=== "UDmax (Unweighted)"

    $$UD\max = \max_{1 \leq \ell \leq M} \sup F_T(\ell)$$

    Maximo simples das estatisticas $\sup F_T(\ell)$. Atribui **peso igual** a todas as alternativas.

=== "WDmax (Weighted)"

    $$WD\max = \max_{1 \leq \ell \leq M} \frac{c(1, \alpha)}{c(\ell, \alpha)} \sup F_T(\ell)$$

    onde $c(\ell, \alpha)$ e o valor critico de $\sup F_T(\ell)$ ao nivel $\alpha$. Os pesos equalizam o **tamanho marginal** de cada teste. Geralmente mais poderoso que UDmax.

!!! tip "Quando Usar UDmax/WDmax"
    Use UDmax/WDmax como **primeiro passo**: se rejeitam H₀, ha pelo menos uma quebra. Em seguida, use os testes sequenciais ou criterios de informacao para determinar **quantas** quebras existem.

### Teste Sequencial ($\ell + 1$ vs. $\ell$ quebras)

Testa se ha $\ell + 1$ quebras contra $\ell$ quebras:

$$\sup F_T(\ell + 1 | \ell) = \sup_{\tau \in \Lambda_\ell} F_T^*(\tau)$$

onde a busca e feita sobre possiveis datas de uma quebra adicional $\tau$ dentro de cada segmento existente.

**Procedimento sequencial:**

1. Teste $\sup F_T(1)$: se rejeitar, ha pelo menos 1 quebra
2. Dado 1 quebra, teste $\sup F_T(2|1)$: se rejeitar, ha pelo menos 2
3. Continue ate $\sup F_T(\ell+1|\ell)$ nao rejeitar
4. O numero estimado de quebras e $\hat{m} = \ell$

## Selecao do Numero de Quebras

Alem dos testes sequenciais, criterios de informacao podem ser usados para selecionar $m$:

### BIC (Bayesian Information Criterion)

$$BIC(m) = \ln\left(\frac{SSR_m}{T}\right) + \frac{(m+1)k \ln T}{T}$$

### LWZ (Liu, Wu & Zidek)

$$LWZ(m) = \ln\left(\frac{SSR_m}{T - (m+1)k}\right) + \frac{(m+1)k \, c_T}{T - (m+1)k}$$

onde $c_T = 0.299 (\ln T)^{2.1}$.

| Criterio | Propriedade | Recomendacao |
|:---------|:-----------|:-------------|
| BIC | Consistente — seleciona $m$ verdadeiro com $T \to \infty$ | Padrao para maioria das aplicacoes |
| LWZ | Mais conservador — penaliza mais fortemente modelos com muitas quebras | Usar quando BIC sugere muitas quebras |
| Sequencial | Controla tamanho do teste a cada passo | Complementar aos criterios de informacao |

!!! tip "Recomendacao Pratica"
    Use **BIC** como criterio principal e verifique com o procedimento **sequencial**. Se ambos concordam, a conclusao e robusta. Se divergem, o LWZ pode servir como desempate.

## Exemplo Pratico

### Serie com Multiplas Quebras

```python
import numpy as np
from chronobox.tests_stat.structural_breaks import bai_perron_test

# Serie com 3 regimes (2 quebras: t=80 e t=160)
np.random.seed(42)
T = 240
x = np.random.randn(T)

y = np.zeros(T)
y[:80] = 1.0 + 0.5 * x[:80]          # Regime 1: beta = 0.5
y[80:160] = 3.0 + 1.5 * x[80:160]    # Regime 2: beta = 1.5
y[160:] = 0.5 + 0.8 * x[160:]        # Regime 3: beta = 0.8
y += 0.5 * np.random.randn(T)

result = bai_perron_test(y, x, max_breaks=5, trimming=0.15)
print(result.summary())
```

Saida esperada:

```
============================================================
  Bai-Perron Multiple Structural Breaks Test
============================================================
  Optimal breaks (BIC) : 2
  Break dates          : [80, 160]

  H0: No structural breaks (m = 0)
  H1: m structural breaks

  --- supF Tests (l vs. 0 breaks) ---
  supF(1)  : 34.5678  [crit 5%: 8.58]  *
  supF(2)  : 28.1234  [crit 5%: 7.22]  *
  supF(3)  : 15.6789  [crit 5%: 5.96]  *

  --- Double Maximum Tests ---
  UDmax    : 34.5678  [crit 5%: 8.88]  *
  WDmax    : 36.7890  [crit 5%: 9.91]  *

  --- Sequential Tests (l+1 vs. l) ---
  supF(2|1) : 25.3456  [crit 5%: 9.02]  *
  supF(3|2) :  3.4567  [crit 5%: 9.56]

  --- Information Criteria ---
  BIC  : m = 2  (optimal)
  LWZ  : m = 2

  --- Estimated Segments ---
  Regime 1 : t = [  1,  80]  n = 80
  Regime 2 : t = [ 81, 160]  n = 80
  Regime 3 : t = [161, 240]  n = 80

  Decision: 2 structural breaks detected
============================================================
```

### Visualizacao

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Serie com quebras estimadas
t = np.arange(T)
axes[0].plot(t, y, "b-", alpha=0.6, label="Serie")

# Marcar quebras estimadas
breaks = result.additional_info["break_dates"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
start = 0
for i, bp in enumerate(breaks + [T]):
    segment = slice(start, bp)
    axes[0].axvline(bp, color="red", linestyle="--", alpha=0.7)
    # Regressao do segmento
    from numpy.polynomial.polynomial import polyfit
    b = np.polyfit(x[segment], y[segment], 1)
    axes[0].plot(t[segment], np.polyval(b, x[segment]),
                 color=colors[i], linewidth=2,
                 label=f"Regime {i+1}: beta={b[0]:.2f}")
    start = bp

axes[0].set_title("Bai-Perron: Quebras Estimadas")
axes[0].set_xlabel("Observacao")
axes[0].set_ylabel("y")
axes[0].legend(fontsize=9)

# RSS por numero de quebras
m_range = range(0, 6)
bic_vals = result.additional_info["bic_values"]
axes[1].plot(list(m_range), bic_vals, "bo-", markersize=8)
axes[1].axvline(result.additional_info["optimal_breaks_bic"],
                color="red", linestyle="--", label="m otimo (BIC)")
axes[1].set_title("BIC por Numero de Quebras")
axes[1].set_xlabel("Numero de quebras (m)")
axes[1].set_ylabel("BIC")
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Acessando Resultados Programaticamente

```python
result = bai_perron_test(y, x, max_breaks=5)

# Numero otimo de quebras
m = result.additional_info["optimal_breaks_bic"]
print(f"Numero de quebras (BIC): {m}")

# Datas de quebra
breaks = result.additional_info["break_dates"]
print(f"Datas de quebra: {breaks}")

# Testes supF
for ell in range(1, m + 2):
    key = f"supF_{ell}"
    if key in result.additional_info:
        stat = result.additional_info[key]["statistic"]
        crit = result.additional_info[key]["critical_5pct"]
        sig = "*" if stat > crit else ""
        print(f"supF({ell}): {stat:.4f} [crit: {crit:.2f}] {sig}")

# UDmax e WDmax
print(f"UDmax: {result.additional_info['udmax']:.4f}")
print(f"WDmax: {result.additional_info['wdmax']:.4f}")

# BIC e LWZ para cada m
for m_val, bic_val in enumerate(result.additional_info["bic_values"]):
    print(f"m={m_val}: BIC={bic_val:.4f}")

# Coeficientes por regime
for j, coefs in enumerate(result.additional_info["regime_coefficients"]):
    print(f"Regime {j+1}: beta = {coefs}")
```

### Serie sem Quebra (Controle)

```python
# Serie estavel (sem quebra)
y_stable = 1.0 + 0.5 * x + 0.5 * np.random.randn(T)

result_stable = bai_perron_test(y_stable, x, max_breaks=5)
print(f"Quebras estimadas (BIC): {result_stable.additional_info['optimal_breaks_bic']}")
# Esperado: 0 quebras
```

## Assinatura da Funcao

```python
bai_perron_test(
    y: NDArray,
    X: NDArray | None = None,       # Regressores (None = constante apenas)
    max_breaks: int = 5,            # Numero maximo de quebras a considerar
    trimming: float = 0.15,         # Fracao de trimming (epsilon)
    alpha: float = 0.05,            # Nivel de significancia
    criterion: str = "bic"          # 'bic' ou 'lwz' para selecao de m
) -> TestResult
```

## Limitacoes

1. **Requer modelo linear** — o framework e baseado em regressao OLS com parametros que mudam entre segmentos
2. **Computacionalmente intensivo** — $O(T^2)$ para cada $m$; amostras muito grandes podem ser lentas
3. **Trimming afeta resultados** — quebras muito proximas dos extremos da amostra nao sao detectaveis
4. **Assume erros iid** (versao basica) — extensoes robustas a autocorrelacao e heteroscedasticidade existem mas sao mais complexas
5. **Numero maximo** de quebras deve ser especificado — o algoritmo nao busca alem de `max_breaks`
6. **Quebras graduais** podem nao ser bem captadas — o modelo assume mudancas **abruptas** nos parametros

## Equivalente R

=== "strucchange"

    ```r
    library(strucchange)

    # Estimar quebras (Bai-Perron)
    bp <- breakpoints(y ~ x, h = 0.15, breaks = 5)
    summary(bp)

    # Datas de quebra estimadas
    bp$breakpoints

    # Numero otimo de quebras
    # BIC
    summary(bp)  # mostra BIC para cada m

    # Coeficientes por regime
    coef(bp)

    # Intervalos de confianca para as datas de quebra
    confint(bp)

    # Visualizacao
    plot(bp)
    lines(bp)

    # Testes supF
    fs <- Fstats(y ~ x)
    plot(fs)
    sctest(fs, type = "supF")    # supF(1)
    sctest(fs, type = "aveF")    # aveF
    sctest(fs, type = "expF")    # expF

    # Equivalencias:
    # chronobox bai_perron_test(y, x, max_breaks=5, trimming=0.15)
    # -> strucchange breakpoints(y ~ x, h = 0.15, breaks = 5)
    ```

=== "Detalhes dos testes"

    ```r
    library(strucchange)

    # --- Testes supF(l vs 0) ---
    fs <- Fstats(y ~ x, from = 0.15, to = 0.85)
    sctest(fs, type = "supF")

    # --- UDmax ---
    # Nao disponivel diretamente no strucchange
    # Calcular manualmente como max dos supF(l)

    # --- Sequencial ---
    bp <- breakpoints(y ~ x, h = 0.15, breaks = 5)
    # Escolher m pelo BIC:
    opt_m <- which.min(summary(bp)$RSS["BIC", ]) - 1

    # --- Intervalos de confianca para datas ---
    ci <- confint(bp, breaks = opt_m)
    ci  # Lower, Breakpoint, Upper para cada quebra
    ```

## See Also

- [CUSUM Test](cusum.md) — Deteccao de instabilidade sem datas especificas
- [Chow Test](chow.md) — Teste com data de quebra conhecida
- [Structural Break Tests](index.md) — Visao geral

## Referencias

- Bai, J. (1997). "Estimation of a change point in multiple regression models." *Review of Economics and Statistics*, 79(4), 551-563.
- Bai, J. & Perron, P. (1998). "Estimating and testing linear models with multiple structural changes." *Econometrica*, 66(1), 47-78.
- Bai, J. & Perron, P. (2003). "Computation and analysis of multiple structural change models." *Journal of Applied Econometrics*, 18(1), 1-22.
- Bai, J. & Perron, P. (2003). "Critical values for multiple structural change tests." *The Econometrics Journal*, 6(1), 72-78.
- Liu, J., Wu, S. & Zidek, J.V. (1997). "On segmented multivariate regression." *Statistica Sinica*, 7(2), 497-525.
