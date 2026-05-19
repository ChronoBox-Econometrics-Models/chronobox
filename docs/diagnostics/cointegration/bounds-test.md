---
title: "ARDL Bounds Test"
description: "Teste de Bounds (PSS) para cointegracao no chronobox — ARDL-ECM, bandas criticas I(0)/I(1), tres regioes de decisao e exemplos praticos."
---

# ARDL Bounds Test

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.cointegration.bounds_test()`
    **H₀:** Nao ha relacao de niveis ($\pi_{yy} = 0$ e $\boldsymbol{\pi}_{yx} = \mathbf{0}$)
    **H₁:** Existe relacao de niveis (cointegracao)
    **Distribuicao:** F nao-padrao com bandas I(0) e I(1)
    **Valores criticos:** Pesaran, Shin & Smith (2001)
    **R equivalente:** `ARDL::bounds_test()`, `dynamac::pssbounds()`

## Hipoteses

O Bounds test (Pesaran, Shin & Smith, 2001) testa a existencia de uma relacao de longo prazo no contexto de um modelo ARDL-ECM:

$$H_0: \pi_{yy} = 0 \text{ e } \boldsymbol{\pi}_{yx} = \mathbf{0} \quad \text{(nao ha relacao de niveis)}$$

$$H_1: \pi_{yy} \neq 0 \text{ ou } \boldsymbol{\pi}_{yx} \neq \mathbf{0} \quad \text{(existe relacao de niveis)}$$

**Rejeitar H₀** indica cointegracao — existe uma relacao de longo prazo entre as variaveis.

## Por Que Usar o Bounds Test?

O Bounds test tem uma **vantagem fundamental** sobre Johansen e Engle-Granger:

!!! note "Vantagem Principal"
    O teste e valido **independentemente** de as variaveis serem $I(0)$, $I(1)$, ou uma mistura de ambas. Nao requer pre-teste de raiz unitaria para determinar a ordem de integracao.

Comparacao com outros testes:

| Caracteristica | Johansen | Engle-Granger | **Bounds Test** |
|:--------------|:---------|:--------------|:----------------|
| Variaveis $I(0)/I(1)$ mistas | Nao | Nao | **Sim** |
| Multiplas relacoes | Sim | Nao | Nao |
| Pre-teste de raiz unitaria | Necessario | Necessario | **Nao necessario** |
| Amostras pequenas | Baixo poder | Viés | **Melhor desempenho** |

## O Modelo ARDL-ECM

O teste baseia-se na estimacao de uma equacao condicional de correcao de erros (ECM) derivada do modelo ARDL:

$$\Delta y_t = c_0 + \pi_{yy} y_{t-1} + \boldsymbol{\pi}_{yx}^\top \mathbf{x}_{t-1} + \sum_{i=1}^{p-1} \phi_i \Delta y_{t-i} + \sum_{j=0}^{q-1} \boldsymbol{\theta}_j^\top \Delta \mathbf{x}_{t-j} + \varepsilon_t$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $\pi_{yy} y_{t-1}$ | Nivel defasado da variavel dependente |
| $\boldsymbol{\pi}_{yx}^\top \mathbf{x}_{t-1}$ | Niveis defasados dos regressores |
| $\phi_i \Delta y_{t-i}$ | Dinamica de curto prazo (diferencas de $y$) |
| $\boldsymbol{\theta}_j^\top \Delta \mathbf{x}_{t-j}$ | Dinamica de curto prazo (diferencas de $\mathbf{x}$) |
| $c_0$ | Constante |

Os termos em **niveis** ($y_{t-1}$, $\mathbf{x}_{t-1}$) capturam a relacao de longo prazo. Se todos os coeficientes de niveis sao zero, nao ha relacao de longo prazo.

## Estatistica de Teste

### F-test (Principal)

A estatistica F testa a significancia conjunta dos termos em niveis:

$$F = \frac{(SSR_R - SSR_U) / m}{SSR_U / (T - k)}$$

onde:

| Termo | Descricao |
|:------|:----------|
| $SSR_R$ | Soma dos quadrados dos residuos do modelo **restrito** ($\pi_{yy} = 0, \boldsymbol{\pi}_{yx} = \mathbf{0}$) |
| $SSR_U$ | Soma dos quadrados dos residuos do modelo **irrestrito** |
| $m$ | Numero de restricoes ($= 1 + k$, onde $k$ e o numero de regressores) |
| $T$ | Numero de observacoes |
| $k$ | Numero total de parametros no modelo irrestrito |

### t-test (Complementar)

Testa apenas o coeficiente de $y_{t-1}$:

$$t = \frac{\hat{\pi}_{yy}}{SE(\hat{\pi}_{yy})}$$

## Bandas Criticas e Tres Regioes de Decisao

A inovacao do Bounds test sao as **duas bandas criticas**:

- **Banda inferior (I(0) bound)**: Calculada assumindo que todos os regressores sao $I(0)$
- **Banda superior (I(1) bound)**: Calculada assumindo que todos os regressores sao $I(1)$

A decisao e baseada em **tres regioes**:

```
            I(0) bound                I(1) bound
                |                         |
    Nao rejeita |      Inconclusivo       | Rejeita H0
    H0          |                         | (cointegracao)
  ──────────────┼─────────────────────────┼──────────────
                |                         |
        F < lower                  F > upper
```

| Regiao | Condicao | Decisao |
|:-------|:---------|:--------|
| **Rejeita H₀** | $F > \text{upper bound}$ | Cointegracao, independente da ordem de integracao |
| **Nao rejeita H₀** | $F < \text{lower bound}$ | Sem cointegracao, independente da ordem de integracao |
| **Inconclusivo** | $\text{lower} \leq F \leq \text{upper}$ | Resultado depende da ordem de integracao — realizar testes de raiz unitaria |

!!! warning "Regiao Inconclusiva"
    Se o resultado cair na regiao inconclusiva, a decisao depende da ordem de integracao das variaveis. Nesse caso, realize [testes de raiz unitaria](../unit-root/index.md) e use Johansen ou Engle-Granger conforme apropriado.

## Casos Deterministicos

O Bounds test tem diferentes especificacoes deterministicas:

=== "Caso 3: Intercepto irrestrito, sem tendencia"

    $$\Delta y_t = c_0 + \pi_{yy} y_{t-1} + \boldsymbol{\pi}_{yx}^\top \mathbf{x}_{t-1} + \ldots + \varepsilon_t$$

    **Quando usar:** Caso mais comum. Adequado para a maioria das series macroeconomicas.

=== "Caso 5: Intercepto irrestrito + tendencia"

    $$\Delta y_t = c_0 + c_1 t + \pi_{yy} y_{t-1} + \boldsymbol{\pi}_{yx}^\top \mathbf{x}_{t-1} + \ldots + \varepsilon_t$$

    **Quando usar:** Quando as variaveis em niveis apresentam tendencia deterministica.

## Valores Criticos

Valores criticos de Pesaran, Shin & Smith (2001) para o F-test a 5% (Caso 3):

| $k$ regressores | I(0) bound | I(1) bound |
|:----------------|:-----------|:-----------|
| 1 | 4.94 | 5.73 |
| 2 | 3.79 | 4.85 |
| 3 | 3.23 | 4.35 |
| 4 | 2.86 | 4.01 |
| 5 | 2.62 | 3.79 |

*Valores para amostras grandes. O chronobox armazena tabelas completas para diferentes niveis de significancia.*

## Exemplo Pratico

### Teste Basico

```python
import numpy as np
from chronobox.tests_stat.cointegration import bounds_test

# Gerar dados com relacao de longo prazo
np.random.seed(42)
T = 200

# x pode ser I(0) ou I(1) — o Bounds test funciona em ambos os casos
x = np.cumsum(np.random.randn(T))  # I(1)

# y tem relacao de longo prazo com x
e = np.random.randn(T)
y = np.zeros(T)
for t in range(1, T):
    y[t] = 0.8 * y[t - 1] + 0.5 * x[t] - 0.3 * x[t - 1] + e[t]

result = bounds_test(y, x, case=3)
print(result.summary())
```

Saida esperada:

```
============================================================
  PSS ARDL Bounds Test
============================================================
  F-statistic    : 12.345678
  t-statistic    : -4.567890
  Lags used      : 2

  H0: No levels relationship (pi_yy = 0, pi_yx = 0)
  H1: Levels relationship exists

  Critical Bounds (5%):
      I(0) bound : 4.94
      I(1) bound : 5.73

  Decision (5%)  : Cointegration
============================================================
```

!!! tip "Interpretacao"
    $F = 12.35 > 5.73$ (upper bound a 5%). A estatistica F excede a banda superior, rejeitamos $H_0$: existe uma relacao de longo prazo entre as variaveis, **independentemente** de serem $I(0)$ ou $I(1)$.

### Acessando Resultados Programaticamente

```python
result = bounds_test(y, x, case=3)

# Estatisticas
print(f"F-statistic: {result.additional_info['f_statistic']:.4f}")
print(f"t-statistic: {result.additional_info['t_statistic']:.4f}")

# Bandas criticas
print(f"Lower bound (I(0)): {result.additional_info['lower_bound']:.4f}")
print(f"Upper bound (I(1)): {result.additional_info['upper_bound']:.4f}")

# Decisao
print(f"Decisao: {result.additional_info['decision']}")
# 'cointegration', 'no cointegration', ou 'inconclusive'

# Decisao automatica
decision = result.additional_info["decision"]
if decision == "cointegration":
    print("Cointegracao confirmada — estimar ARDL-ECM")
elif decision == "no cointegration":
    print("Sem cointegracao — usar ARDL sem ECM")
else:
    print("Inconclusivo — realizar testes de raiz unitaria")
```

### Multiplos Regressores

```python
# Bounds test com 3 regressores
x1 = np.cumsum(np.random.randn(200))
x2 = np.random.randn(200)           # I(0) — sem problema!
x3 = np.cumsum(np.random.randn(200))

X = np.column_stack([x1, x2, x3])
y_multi = 0.5 * x1 - 0.3 * x2 + 0.2 * x3 + np.random.randn(200)

result_multi = bounds_test(y_multi, X, case=3)
print(result_multi.summary())
```

### Selecao Automatica de Lags

```python
# Lags selecionados automaticamente via AIC
result_auto = bounds_test(y, x, lags=None, case=3)
print(f"Lags selecionados: {result_auto.lags_used}")

# Lags fixos
result_fixed = bounds_test(y, x, lags=4, case=3)
print(f"Lags fixos: {result_fixed.lags_used}")
```

## Assinatura da Funcao

```python
bounds_test(
    y: NDArray,             # Variavel dependente (T,)
    x: NDArray,             # Regressores (T,) ou (T, k)
    lags: int | None = None, # Lags ECM (None = selecao via AIC)
    case: int = 3           # 3: intercepto sem tendencia; 5: com tendencia
) -> TestResult
```

## Limitacoes

1. **Apenas uma relacao de longo prazo**: Assim como Engle-Granger, detecta no maximo uma relacao. Para multiplas relacoes, use [Johansen](johansen.md)
2. **Requer I(0) ou I(1)**: Nao e valido para variaveis $I(2)$ ou ordens superiores
3. **Sensivel a lags**: A selecao de lags afeta o resultado — use criterios de informacao
4. **Regiao inconclusiva**: Pode nao produzir conclusao definitiva; neste caso, complementar com outros testes
5. **Amostras muito pequenas**: Em amostras com $T < 80$, os valores criticos assintoticos podem nao ser confiaveis — considere valores criticos de amostras finitas (Narayan, 2005)

!!! tip "Quando Preferir o Bounds Test"
    - Variaveis com ordens de integracao **mistas** ($I(0)$ e $I(1)$)
    - Incerteza sobre a ordem de integracao (testes de raiz unitaria ambiguos)
    - Amostras pequenas a moderadas
    - Modelos ARDL ja estimados ou planejados

## Equivalente R

=== "ARDL"

    ```r
    library(ARDL)

    # Estimar modelo ARDL
    model <- ardl(y ~ x1 + x2, data = df, order = c(2, 1, 1))

    # Bounds test
    bounds_test(model)
    # Retorna: F-statistic, t-statistic, bandas criticas, decisao
    ```

=== "dynamac"

    ```r
    library(dynamac)

    # PSS Bounds test
    pssbounds(
      obs = nrow(df),
      fstat = 12.35,     # F calculada
      tstat = -4.57,     # t calculada
      case = 3,          # intercepto irrestrito, sem tendencia
      k = 2              # numero de regressores
    )
    ```

=== "Manual"

    ```r
    # Passo 1: Estimar o ECM irrestrito
    ecm <- lm(diff(y) ~ y_lag1 + x_lag1 + diff_y_lag1 + diff_x + diff_x_lag1)

    # Passo 2: Estimar o ECM restrito (sem termos em niveis)
    ecm_r <- lm(diff(y) ~ diff_y_lag1 + diff_x + diff_x_lag1)

    # Passo 3: F-test
    anova(ecm_r, ecm)
    # Comparar F com tabelas de PSS (2001)
    ```

## See Also

- [Johansen Test](johansen.md) — Para multiplas relacoes de cointegracao
- [Engle-Granger Test](engle-granger.md) — Alternativa para series puramente $I(1)$
- [Cointegration Tests](index.md) — Visao geral
- [Unit Root Tests](../unit-root/index.md) — Determinar ordem de integracao (quando necessario)
- [Theory: ARDL](../../theory/ardl-theory.md) — Fundamentos teoricos ARDL

## Referencias

- Pesaran, M.H., Shin, Y. & Smith, R.J. (2001). "Bounds testing approaches to the analysis of level relationships." *Journal of Applied Econometrics*, 16(3), 289-326.
- Pesaran, M.H. & Shin, Y. (1999). "An autoregressive distributed-lag modelling approach to cointegration analysis." In *Econometrics and Economic Theory in the 20th Century: The Ragnar Frisch Centennial Symposium*, 371-413. Cambridge University Press.
- Narayan, P.K. (2005). "The saving and investment nexus for China: evidence from cointegration tests." *Applied Economics*, 37(17), 1979-1990.
- Kripfganz, S. & Schneider, D.C. (2020). "Response surface regressions for critical value bounds and approximate p-values in equilibrium correction models." *Oxford Bulletin of Economics and Statistics*, 82(6), 1456-1481.
