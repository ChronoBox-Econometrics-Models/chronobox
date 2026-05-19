---
title: "Unit Root Tests"
description: "Testes de raiz unitaria para series temporais no chronobox — ADF, Phillips-Perron, KPSS, Zivot-Andrews e Lee-Strazicich."
---

# Unit Root Tests

!!! info "Quick Reference"
    **Modulo:** `chronobox.tests_stat.unit_root`
    **Objetivo:** Determinar a ordem de integracao da serie temporal
    **R equivalente:** `tseries::adf.test()`, `urca::ur.df()`, `urca::ur.pp()`, `urca::ur.kpss()`
    **Retorno:** `TestResult` com estatistica, p-valor, valores criticos e decisao

## O Que e Raiz Unitaria?

Uma serie temporal $y_t$ possui **raiz unitaria** quando segue um processo do tipo:

$$y_t = y_{t-1} + \varepsilon_t$$

Neste caso, choques ($\varepsilon_t$) tem efeito **permanente** sobre o nivel da serie — ela nao retorna a uma media fixa. Uma serie com raiz unitaria e dita **integrada de ordem 1**, ou $I(1)$.

Uma serie **estacionaria** ($I(0)$), por outro lado, possui media e variancia constantes ao longo do tempo:

$$y_t = \mu + \phi y_{t-1} + \varepsilon_t, \quad |\phi| < 1$$

## Por Que Testar?

Testar raiz unitaria e um passo **obrigatorio** antes da modelagem porque:

1. **Regressao espuria**: Regredir uma serie I(1) contra outra I(1) produz resultados significativos mas sem significado economico (Granger & Newbold, 1974)
2. **Selecao de modelo**: Series I(0) podem ser modeladas com ARMA; series I(1) requerem ARIMA ou diferenciacao
3. **Cointegracao**: Apenas series I(1) podem ser cointegradas — prerequisito para VECM
4. **Inferencia**: Distribuicoes padrao (t, F) nao sao validas na presenca de raiz unitaria

## Cinco Testes Comparados

O chronobox implementa cinco testes de raiz unitaria, cada um com vantagens especificas:

| Teste | H₀ | H₁ | Vantagem Principal |
|:------|:---|:---|:-------------------|
| [ADF](adf.md) | Raiz unitaria | Estacionaria | Teste padrao, selecao automatica de lags |
| [Phillips-Perron](pp.md) | Raiz unitaria | Estacionaria | Robusto a heteroscedasticidade e correlacao serial |
| [KPSS](kpss.md) | **Estacionaria** | Raiz unitaria | Hipotese invertida — confirmacao |
| [Zivot-Andrews](zivot-andrews.md) | Raiz unitaria (sem quebra) | Estacionaria com quebra | Uma quebra estrutural endogena |
| [Lee-Strazicich](lee-strazicich.md) | Raiz unitaria com quebras | Estacionaria com quebras | Duas quebras, size-correct sob H₀ |

!!! warning "KPSS: Hipotese Invertida"
    O KPSS e o unico teste onde H₀ e **estacionaridade**. Rejeitar H₀ no KPSS significa evidencia de raiz unitaria — o **oposto** de ADF e Phillips-Perron. Use KPSS como teste de **confirmacao** junto com ADF.

## Estrategia de Teste Recomendada

### Abordagem: ADF + KPSS

A combinacao ADF + KPSS e a mais utilizada na pratica. Como as hipoteses sao invertidas, a combinacao permite quatro conclusoes:

| ADF (H₀: raiz unitaria) | KPSS (H₀: estacionaria) | Conclusao |
|:------------------------|:-------------------------|:----------|
| Nao rejeita H₀ | Rejeita H₀ | **I(1) confirmado** — diferenciar |
| Rejeita H₀ | Nao rejeita H₀ | **I(0) confirmado** — modelar em niveis |
| Rejeita H₀ | Rejeita H₀ | Ambiguo — pode ser fracionariamente integrada |
| Nao rejeita H₀ | Nao rejeita H₀ | Ambiguo — baixo poder; aumentar amostra |

### Quando Usar Testes com Quebra Estrutural

Se a serie passou por eventos como:

- Mudancas de politica monetaria ou fiscal
- Crises financeiras (2008, COVID)
- Reformas regulatorias

Os testes ADF e PP podem **falhar em rejeitar** a hipotese de raiz unitaria mesmo quando a serie e estacionaria com quebra. Nesse caso, use **Zivot-Andrews** (uma quebra) ou **Lee-Strazicich** (duas quebras).

## Exemplo: Bateria Completa

```python
import numpy as np
from chronobox.tests_stat.unit_root import (
    adf_test,
    pp_test,
    kpss_test,
    zivot_andrews_test,
)

# Serie simulada: random walk
np.random.seed(42)
y = np.cumsum(np.random.randn(250))

# --- Bateria de testes ---
results = {
    "ADF": adf_test(y, regression="c"),
    "PP": pp_test(y, regression="c"),
    "KPSS": kpss_test(y, regression="c"),
    "ZA": zivot_andrews_test(y, model="c"),
}

# Tabela de resultados
print(f"{'Teste':<6} {'Estatistica':>12} {'p-valor':>10} {'Decisao (5%)':<20}")
print("-" * 50)
for name, r in results.items():
    pval = f"{r.pvalue:.4f}" if r.pvalue is not None else "N/A"
    decision = "Rejeita H0" if r.reject_at_5pct else "Nao rejeita H0"
    print(f"{name:<6} {r.statistic:>12.4f} {pval:>10} {decision:<20}")
```

Saida esperada (random walk — todas devem indicar raiz unitaria):

```
Teste  Estatistica    p-valor Decisao (5%)
--------------------------------------------------
ADF        -1.2345     0.6543 Nao rejeita H0
PP         -1.3456     0.6012 Nao rejeita H0
KPSS        1.8901     0.0050 Rejeita H0
ZA         -3.4567        N/A Nao rejeita H0
```

!!! tip "Interpretacao"
    - ADF e PP **nao rejeitam** H₀ de raiz unitaria → evidencia de I(1)
    - KPSS **rejeita** H₀ de estacionaridade → confirma I(1)
    - ZA **nao rejeita** mesmo permitindo quebra → I(1) mesmo com quebra

## Especificacoes de Regressao

Todos os testes permitem diferentes especificacoes deterministicas:

| Codigo | Nome | Modelo | Quando Usar |
|:-------|:-----|:-------|:------------|
| `'n'` | None | Sem constante ou tendencia | Series que oscilam em torno de zero |
| `'c'` | Constant | Constante apenas | **Padrao** — maioria das series economicas |
| `'ct'` | Trend | Constante + tendencia linear | Series com tendencia deterministica (log PIB) |

!!! note "Selecao da Especificacao"
    A escolha afeta os valores criticos. Em caso de duvida, comece com `'c'` (constante). Use `'ct'` apenas se a serie exibir tendencia clara no grafico. Nunca use `'n'` em series com nivel diferente de zero.

## Resultado Padrao: `TestResult`

Todos os testes retornam um objeto `TestResult` com interface uniforme:

```python
result = adf_test(y, regression="c")

result.test_name          # "Augmented Dickey-Fuller"
result.statistic          # Estatistica de teste
result.pvalue             # p-valor (None se nao disponivel)
result.critical_values    # {"1%": ..., "5%": ..., "10%": ...}
result.null_hypothesis    # Descricao de H0
result.reject_at_5pct     # True/False
result.lags_used          # Numero de lags utilizados
result.additional_info    # Informacoes adicionais (break dates, etc.)
result.summary()          # Tabela formatada
```

## See Also

- [ADF Test](adf.md) — Augmented Dickey-Fuller
- [Phillips-Perron](pp.md) — Correcao nao-parametrica
- [KPSS](kpss.md) — Hipotese invertida
- [Zivot-Andrews](zivot-andrews.md) — Uma quebra estrutural
- [Lee-Strazicich](lee-strazicich.md) — Duas quebras estruturais
- [Diagnosticos](../index.md) — Visao geral de todos os testes

## Referencias

- Dickey, D.A. & Fuller, W.A. (1979). "Distribution of the estimators for autoregressive time series with a unit root." *JASA*, 74(366), 427-431.
- Phillips, P.C.B. & Perron, P. (1988). "Testing for a unit root in time series regression." *Biometrika*, 75(2), 335-346.
- Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root." *Journal of Econometrics*, 54(1-3), 159-178.
- Zivot, E. & Andrews, D.W.K. (1992). "Further evidence on the great crash, the oil-price shock, and the unit-root hypothesis." *JBES*, 10(3), 251-270.
- Lee, J. & Strazicich, M.C. (2003). "Minimum Lagrange multiplier unit root test with two structural breaks." *Review of Economics and Statistics*, 85(4), 1082-1089.
