---
title: "KPSS Test"
description: "Teste KPSS de estacionaridade no chronobox — hipotese nula invertida, estatistica LM, level vs trend stationarity e uso conjunto com ADF."
---

# KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.unit_root.kpss_test()`
    **H₀:** Serie e estacionaria ($\sigma_u^2 = 0$)
    **H₁:** Serie possui raiz unitaria ($\sigma_u^2 > 0$)
    **Distribuicao:** KPSS (propria — nao e Dickey-Fuller)
    **Valores criticos:** Kwiatkowski et al. (1992), Tabela 1
    **R equivalente:** `tseries::kpss.test()`, `urca::ur.kpss()`

!!! warning "Hipotese Nula Invertida"
    O KPSS e o **unico** teste de raiz unitaria neste pacote onde H₀ e **estacionaridade**.
    
    - **Rejeitar H₀** → evidencia de raiz unitaria (nao-estacionaria)
    - **Nao rejeitar H₀** → sem evidencia contra estacionaridade

    Isso e o **oposto** do ADF e Phillips-Perron!

## Hipoteses

$$H_0: \sigma_u^2 = 0 \quad \text{(serie e estacionaria)}$$

$$H_1: \sigma_u^2 > 0 \quad \text{(serie possui raiz unitaria)}$$

A logica: o KPSS decompoe a serie em tendencia deterministica + random walk + erro estacionario. Se a variancia do componente random walk ($\sigma_u^2$) e zero, nao ha raiz unitaria.

## Modelo

O KPSS decompoe a serie como:

$$y_t = \xi t + r_t + \varepsilon_t$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $\xi t$ | Tendencia deterministica (linear) |
| $r_t = r_{t-1} + u_t$ | Random walk: $u_t \sim (0, \sigma_u^2)$ |
| $\varepsilon_t$ | Erro estacionario |

- Se $\sigma_u^2 = 0$: o componente random walk e constante → serie e estacionaria em torno da tendencia
- Se $\sigma_u^2 > 0$: ha uma raiz unitaria estocastica

## Estatistica de Teste

A estatistica KPSS e baseada nas **somas parciais** dos residuos da regressao OLS de $y_t$ sobre os termos deterministicos:

### Passo 1: Regressao OLS

=== "Level Stationarity (`'c'`)"

    $$y_t = \alpha + \varepsilon_t$$

    Testa se a serie e estacionaria em torno de uma **media constante**.

=== "Trend Stationarity (`'ct'`)"

    $$y_t = \alpha + \beta t + \varepsilon_t$$

    Testa se a serie e estacionaria em torno de uma **tendencia linear**.

### Passo 2: Somas Parciais

Calcular as somas parciais dos residuos:

$$S_t = \sum_{i=1}^{t} \hat{\varepsilon}_i, \quad t = 1, \ldots, T$$

### Passo 3: Variancia de Longo Prazo

Estimar $\hat{\lambda}^2$ via Newey-West com kernel de Bartlett:

$$\hat{\lambda}^2 = \hat{\gamma}_0 + 2\sum_{j=1}^{l} \left(1 - \frac{j}{l+1}\right) \hat{\gamma}_j$$

### Passo 4: Estatistica LM

$$\eta = \frac{1}{T^2} \cdot \frac{\sum_{t=1}^{T} S_t^2}{\hat{\lambda}^2}$$

!!! note "Regra de Rejeicao"
    Rejeita H₀ de estacionaridade se $\eta > c_\alpha$ (valor critico **superior**). Isso e oposto ao ADF/PP onde rejeitamos se a estatistica e mais **negativa** que o valor critico.

## Valores Criticos

Os valores criticos assintóticos vêm de Kwiatkowski et al. (1992):

=== "Level Stationarity (`'c'`)"

    | Significancia | Valor Critico |
    |:-------------|:-------------|
    | 10% | 0.347 |
    | 5% | 0.463 |
    | 2.5% | 0.574 |
    | 1% | 0.739 |

=== "Trend Stationarity (`'ct'`)"

    | Significancia | Valor Critico |
    |:-------------|:-------------|
    | 10% | 0.119 |
    | 5% | 0.146 |
    | 2.5% | 0.176 |
    | 1% | 0.216 |

!!! note
    Os valores criticos para trend stationarity sao **menores** porque a tendencia absorve parte da variacao, tornando o teste mais sensivel.

## Exemplo Pratico

### Serie Estacionaria

```python
import numpy as np
from chronobox.tests_stat.unit_root import kpss_test

# Serie estacionaria AR(1)
np.random.seed(42)
T = 200
y_stat = np.zeros(T)
for t in range(1, T):
    y_stat[t] = 0.5 * y_stat[t - 1] + np.random.randn()

result = kpss_test(y_stat, regression="c")
print(result.summary())
```

Saida esperada:

```
============================================================
  KPSS Test
============================================================
  Test statistic : 0.123456
  p-value        : 0.150000
  Lags used      : 4

  H0: Series is level-stationary
  H1: Unit root present (non-stationary)

  Critical Values:
      1% : 0.7390
      5% : 0.4630
     10% : 0.3470

  Decision (5%)  : Fail to reject H0
============================================================
```

!!! tip "Interpretacao"
    Estatistica KPSS (≈ 0.12) e **menor** que o valor critico a 5% (0.463). **Nao rejeitamos** H₀ de estacionaridade — consistente com a serie ser I(0).

### Serie com Raiz Unitaria

```python
# Random walk
y_rw = np.cumsum(np.random.randn(200))

result_rw = kpss_test(y_rw, regression="c")
print(f"Estatistica: {result_rw.statistic:.4f}")
print(f"Rejeita H0 (estacionaridade): {result_rw.reject_at_5pct}")
# Esperado: rejeita H0 (estatistica >> 0.463)
```

### Estrategia ADF + KPSS

```python
from chronobox.tests_stat.unit_root import adf_test, kpss_test

def classificar_integracao(y, nome="Serie"):
    """Classifica a ordem de integracao usando ADF + KPSS."""
    adf = adf_test(y, regression="c")
    kpss = kpss_test(y, regression="c")

    print(f"\n--- {nome} ---")
    print(f"ADF:  stat={adf.statistic:.4f}, p={adf.pvalue:.4f}, "
          f"rejeita={adf.reject_at_5pct}")
    print(f"KPSS: stat={kpss.statistic:.4f}, p={kpss.pvalue:.4f}, "
          f"rejeita={kpss.reject_at_5pct}")

    if not adf.reject_at_5pct and kpss.reject_at_5pct:
        print("→ I(1) confirmado")
    elif adf.reject_at_5pct and not kpss.reject_at_5pct:
        print("→ I(0) confirmado")
    else:
        print("→ Resultado ambiguo — investigar mais")

# Testar
np.random.seed(42)
classificar_integracao(
    0.5 * np.random.randn(200).cumsum() + np.random.randn(200) * 3,
    "Serie mista"
)
```

### Level vs Trend Stationarity

```python
# Serie com tendencia deterministica
t = np.arange(200)
y_trend = 0.05 * t + np.random.randn(200) * 2

# Level stationarity: vai rejeitar (a tendencia parece raiz unitaria)
result_level = kpss_test(y_trend, regression="c")
print(f"Level:  stat={result_level.statistic:.4f}, rejeita={result_level.reject_at_5pct}")

# Trend stationarity: nao deve rejeitar (tendencia e deterministica)
result_trend = kpss_test(y_trend, regression="ct")
print(f"Trend:  stat={result_trend.statistic:.4f}, rejeita={result_trend.reject_at_5pct}")
```

!!! warning "Cuidado com a Especificacao"
    Se a serie tem tendencia deterministica, use `regression='ct'`. Com `regression='c'`, a tendencia sera confundida com raiz unitaria, levando a rejeicao espuria de H₀.

## Assinatura da Funcao

```python
kpss_test(
    y: NDArray,
    regression: str = "c",      # 'c' (level) ou 'ct' (trend)
    nlags: int | None = None    # None = 4*(T/100)^{2/9}
) -> TestResult
```

## KPSS vs ADF: Complementaridade

| Aspecto | ADF | KPSS |
|:--------|:----|:-----|
| H₀ | Raiz unitaria | **Estacionaridade** |
| Rejeicao indica | Estacionaria | **Raiz unitaria** |
| Direcao da estatistica | Mais negativa = mais evidencia | Mais positiva = mais evidencia |
| Distribuicao | Dickey-Fuller | KPSS (propria) |
| Proposito | Teste principal | **Confirmacao** |

## Equivalente R

=== "tseries"

    ```r
    library(tseries)

    # Teste KPSS
    # null = "Level" ou "Trend"
    kpss.test(y, null = "Level")
    kpss.test(y, null = "Trend")

    # Equivalencias:
    # chronobox regression='c'  → tseries null = "Level"
    # chronobox regression='ct' → tseries null = "Trend"
    ```

=== "urca"

    ```r
    library(urca)

    # Teste KPSS com controle de bandwidth
    # type: "mu" (level) ou "tau" (trend)
    # lags: "short", "long", ou "nil"
    result <- ur.kpss(y, type = "mu", lags = "short")
    summary(result)

    # Equivalencias:
    # chronobox regression='c'  → urca type = "mu"
    # chronobox regression='ct' → urca type = "tau"
    ```

## See Also

- [ADF Test](adf.md) — Teste principal de raiz unitaria
- [Phillips-Perron](pp.md) — Alternativa nao-parametrica ao ADF
- [Unit Root Tests](index.md) — Visao geral e estrategia de teste

## Referencias

- Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root." *Journal of Econometrics*, 54(1-3), 159-178.
- Hobijn, B., Franses, P.H. & Ooms, M. (2004). "Generalizations of the KPSS-test for stationarity." *Statistica Neerlandica*, 58(4), 483-502.
