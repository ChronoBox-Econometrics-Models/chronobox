---
title: "Zivot-Andrews Test"
description: "Teste Zivot-Andrews de raiz unitaria com quebra estrutural endogena no chronobox — tres modelos de quebra, selecao endogena da data e exemplos praticos."
---

# Zivot-Andrews

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.unit_root.zivot_andrews_test()`
    **H₀:** Raiz unitaria (sem quebra estrutural)
    **H₁:** Serie estacionaria com uma quebra estrutural
    **Distribuicao:** Zivot-Andrews (propria — valores criticos especificos)
    **Valores criticos:** Zivot & Andrews (1992), Tabela 4
    **R equivalente:** `urca::ur.za()`

## Motivacao

Os testes ADF e Phillips-Perron podem **falhar em rejeitar** a hipotese de raiz unitaria quando a serie e de fato estacionaria mas sofreu uma **quebra estrutural** (mudanca no nivel ou na tendencia).

Perron (1989) demonstrou que confundir uma quebra estrutural com raiz unitaria e um problema serio. O teste de Zivot-Andrews resolve isso permitindo uma quebra em data **desconhecida**, estimada endogenamente a partir dos dados.

!!! note "Quando Usar"
    Use Zivot-Andrews quando:
    
    - O ADF nao rejeita H₀, mas voce suspeita de uma quebra estrutural
    - A serie passou por eventos como crises, mudancas de politica ou reformas
    - Voce quer testar raiz unitaria **controlando** para uma quebra

## Hipoteses

$$H_0: \text{Raiz unitaria (sem quebra estrutural)}$$

$$H_1: \text{Serie estacionaria com uma quebra estrutural em data desconhecida}$$

A data da quebra $T_B$ e selecionada **endogenamente** como o ponto que produz a maior evidencia contra H₀ (ou seja, a estatistica t mais negativa para $\gamma$).

## Tres Modelos de Quebra

### Modelo A: Quebra no Intercepto

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \theta DU_t + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

onde $DU_t = 1$ se $t > T_B$, e $0$ caso contrario.

**Uso:** Mudancas abruptas no **nivel** da serie (ex.: desvalorizacao cambial, mudanca de regime de precos).

### Modelo B: Quebra na Tendencia

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \phi DT_t + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

onde $DT_t = (t - T_B)$ se $t > T_B$, e $0$ caso contrario.

**Uso:** Mudancas na **taxa de crescimento** (ex.: desaceleracao do PIB, mudanca de produtividade).

### Modelo C: Quebra no Intercepto e na Tendencia

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \theta DU_t + \phi DT_t + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

**Uso:** Mudancas simultaneas no **nivel e na taxa de crescimento** (ex.: crise financeira com recuperacao em taxa diferente). Este e o modelo mais geral e o **padrao** no chronobox.

## Selecao Endogena da Data de Quebra

O algoritmo busca a data de quebra que **minimiza** a estatistica t de $\gamma$:

$$T_B^* = \arg\min_{T_B \in [\lambda T, (1-\lambda)T]} t_\gamma(T_B)$$

onde $\lambda$ e o parametro de trimming (padrao: 15%).

**Procedimento:**

1. Para cada data candidata $T_B$ na janela $[\lambda T, (1-\lambda)T]$:
    - Construir as dummies $DU_t$ e/ou $DT_t$
    - Estimar a regressao por OLS
    - Selecionar lags por AIC/BIC
    - Calcular $t_\gamma(T_B)$
2. Selecionar $T_B^*$ que minimiza $t_\gamma$
3. Comparar $\min(t_\gamma)$ com valores criticos de Zivot-Andrews

## Valores Criticos

Os valores criticos sao **especificos** para o teste ZA e mais negativos que os do ADF (porque a busca sobre $T_B$ introduz um vies de selecao):

| Significancia | Modelo A (intercepto) | Modelo B (tendencia) | Modelo C (ambos) |
|:-------------|:---------------------|:--------------------|:-----------------|
| 1% | -5.34 | -4.93 | -5.57 |
| 5% | -4.80 | -4.42 | -5.08 |
| 10% | -4.58 | -4.11 | -4.82 |

!!! warning "Nao Use Valores Criticos do ADF"
    Os valores criticos do ADF sao **invalidos** para o teste ZA. A busca endogena sobre a data de quebra desloca a distribuicao, exigindo valores criticos mais restritivos.

## Exemplo Pratico

### Serie com Quebra Estrutural

```python
import numpy as np
from chronobox.tests_stat.unit_root import zivot_andrews_test, adf_test

# Serie estacionaria com quebra no nivel em t=100
np.random.seed(42)
T = 250
y = np.zeros(T)
for t in range(1, T):
    y[t] = 0.7 * y[t - 1] + np.random.randn()
# Adicionar quebra no nivel
y[100:] += 5.0  # salto de 5 unidades em t=100

# ADF pode falhar em detectar estacionaridade
adf = adf_test(y, regression="c")
print(f"ADF: stat={adf.statistic:.4f}, p={adf.pvalue:.4f}, "
      f"rejeita={adf.reject_at_5pct}")
# Provavelmente nao rejeita H0 (confunde quebra com raiz unitaria)

# Zivot-Andrews detecta a quebra e rejeita H0
za = zivot_andrews_test(y, model="a")  # quebra no intercepto
print(f"\nZA:  stat={za.statistic:.4f}, rejeita={za.reject_at_5pct}")
print(f"Data da quebra: t={za.additional_info['break_date']}")
print(f"Fracao: {za.additional_info['break_fraction']:.2f}")
```

### Comparando Modelos

```python
# Testar com os tres modelos
for model_name, model_code in [("Intercepto", "a"),
                                ("Tendencia", "b"),
                                ("Ambos", "c")]:
    result = zivot_andrews_test(y, model=model_code)
    print(f"Modelo {model_name} ({model_code}): "
          f"stat={result.statistic:.4f}, "
          f"break={result.additional_info['break_date']}, "
          f"rejeita={result.reject_at_5pct}")
```

### Resultado Completo

```python
result = zivot_andrews_test(y, model="c", trim=0.15)
print(result.summary())

# Acessar informacoes programaticamente
print(f"\nEstatistica: {result.statistic:.4f}")
print(f"Data da quebra (indice): {result.additional_info['break_date']}")
print(f"Fracao da amostra: {result.additional_info['break_fraction']:.3f}")
print(f"Modelo: {result.additional_info['model']}")
print(f"Lags usados: {result.lags_used}")

# Valores criticos
for level, cv in result.critical_values.items():
    marker = " ← rejeita" if result.statistic < cv else ""
    print(f"  {level}: {cv:.2f}{marker}")
```

## Assinatura da Funcao

```python
zivot_andrews_test(
    y: NDArray,
    model: str = "c",            # 'a' (intercepto), 'b' (tendencia), 'c' (ambos)
    maxlag: int | None = None,   # None = 12*(T/100)^{1/4}
    trim: float = 0.15,          # Fracao a excluir das pontas
    autolag: str = "AIC"         # 'AIC' ou 'BIC'
) -> TestResult
```

## Limitacoes

1. **Apenas uma quebra**: Se a serie tem multiplas quebras, use [Lee-Strazicich](lee-strazicich.md)
2. **H₀ nao inclui quebra**: Sob H₀, nao ha quebra. Se a serie tem raiz unitaria **com** quebra, o teste pode rejeitar incorretamente (problema de tamanho). O Lee-Strazicich resolve isso
3. **Busca exhaustiva**: Computacionalmente mais lento que ADF para series longas
4. **p-valor**: Nao disponivel (apenas valores criticos tabelados)

## Equivalente R

```r
library(urca)

# Teste Zivot-Andrews
# model: "intercept", "trend", "both"
result <- ur.za(y, model = "both", lag = NULL)
summary(result)

# Equivalencias:
# chronobox model='a' → urca model = "intercept"
# chronobox model='b' → urca model = "trend"
# chronobox model='c' → urca model = "both"
```

## See Also

- [ADF Test](adf.md) — Teste padrao (sem quebra)
- [Lee-Strazicich](lee-strazicich.md) — Teste com duas quebras (size-correct)
- [Unit Root Tests](index.md) — Visao geral e comparacao

## Referencias

- Zivot, E. & Andrews, D.W.K. (1992). "Further evidence on the great crash, the oil-price shock, and the unit-root hypothesis." *Journal of Business & Economic Statistics*, 10(3), 251-270.
- Perron, P. (1989). "The great crash, the oil price shock, and the unit root hypothesis." *Econometrica*, 57(6), 1361-1401.
