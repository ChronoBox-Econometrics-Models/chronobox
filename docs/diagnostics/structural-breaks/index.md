---
title: "Structural Break Tests"
description: "Testes de quebras estruturais no chronobox — CUSUM, Chow test e Bai-Perron para deteccao de instabilidade parametrica."
---

# Structural Break Tests

!!! info "Quick Reference"
    **Modulo:** `chronobox.tests_stat.structural_breaks`
    **Objetivo:** Detectar mudancas nos parametros do modelo ao longo do tempo
    **R equivalente:** `strucchange::efp()`, `strucchange::sctest()`, `strucchange::breakpoints()`
    **Retorno:** `TestResult` com estatistica, p-valor, datas de quebra e decisao

## O Que Sao Quebras Estruturais?

Uma **quebra estrutural** ocorre quando os parametros de um modelo de regressao mudam em algum ponto da amostra. Considere o modelo linear:

$$y_t = x_t' \beta_t + \varepsilon_t$$

Se $\beta_t$ e constante para todo $t$, o modelo e **estavel**. Uma quebra estrutural em $t^*$ significa que:

$$\beta_t = \begin{cases} \beta_1 & \text{se } t \leq t^* \\ \beta_2 & \text{se } t > t^* \end{cases}$$

com $\beta_1 \neq \beta_2$.

## Impacto na Modelagem

Quebras estruturais sao um problema sério porque:

1. **Viés nas estimativas**: Estimar $\beta$ sobre toda a amostra produz uma media ponderada entre $\beta_1$ e $\beta_2$ — nenhuma das quais e o parametro verdadeiro
2. **Previsoes erradas**: Um modelo estimado no regime 1 produz previsoes ruins no regime 2
3. **Testes invalidos**: Testes de raiz unitaria (ADF, PP) perdem poder na presenca de quebras — uma serie estacionaria com quebra pode parecer I(1)
4. **Intervalos de confianca incorretos**: A variancia estimada e inflada, distorcendo a inferencia

!!! warning "Quebras em Dados Macroeconomicos"
    Quebras estruturais sao **muito comuns** em series macroeconomicas. Mudancas de politica monetaria, crises financeiras (2008, COVID-19), reformas tributarias e choques de oferta frequentemente alteram as relacoes entre variaveis. Sempre teste estabilidade antes de confiar nos resultados de um modelo estimado em janela longa.

## Tres Abordagens

O chronobox implementa tres abordagens complementares para detectar quebras estruturais:

| Teste | Abordagem | Data de Quebra | Numero de Quebras |
|:------|:----------|:---------------|:------------------|
| [CUSUM](cusum.md) | Sequencial | Nao especifica data | Detecta instabilidade geral |
| [Chow](chow.md) | Data conhecida | **Fornecida pelo usuario** | Uma |
| [Bai-Perron](bai-perron.md) | Endogena | **Estimada pelo algoritmo** | Multiplas |

### Quando Usar Cada Teste

=== "CUSUM / CUSUM-SQ"

    **Use quando:** Voce quer verificar a estabilidade geral dos parametros sem ter uma hipotese sobre **quando** a quebra ocorreu.

    - Baseado na soma acumulada de residuos recursivos
    - Produz um grafico visual com bandas de confianca
    - Bom como teste exploratório inicial
    - Nao identifica a data exata da quebra

=== "Chow Test"

    **Use quando:** Voce tem uma **data candidata** para a quebra (ex: inicio de uma crise, mudanca de politica).

    - Testa se os parametros mudaram em uma data especifica
    - Baseado em estatistica F — simples e intuitivo
    - Requer que a data de quebra seja conhecida *a priori*
    - Nao e adequado para buscar a data de quebra

=== "Bai-Perron"

    **Use quando:** Voce suspeita de **multiplas quebras** e quer estimar **quantas** e **onde** ocorreram.

    - Detecta multiplas quebras de forma endogena
    - Usa programacao dinamica para otimizar globalmente
    - Seleciona o numero otimo de quebras via BIC/LWZ
    - O mais completo, porem mais complexo

## Exemplo Rapido

```python
import numpy as np
from chronobox.tests_stat.structural_breaks import cusum_test, chow_test

# Serie com quebra estrutural em t=100
np.random.seed(42)
T = 200
x = np.random.randn(T)
y = np.where(np.arange(T) < 100,
             1.0 + 0.5 * x,        # Regime 1: beta = 0.5
             1.0 + 2.0 * x         # Regime 2: beta = 2.0
             ) + 0.5 * np.random.randn(T)

# CUSUM: detecta instabilidade geral
cusum_result = cusum_test(y, x)
print(cusum_result.summary())

# Chow: testa quebra na data conhecida t=100
chow_result = chow_test(y, x, break_point=100)
print(chow_result.summary())
```

## Relacao com Outros Diagnosticos

Os testes de quebra estrutural complementam outros diagnosticos do chronobox:

- **[Unit Root Tests](../unit-root/index.md)**: Quebras podem mascarar estacionaridade. Use [Zivot-Andrews](../unit-root/zivot-andrews.md) ou [Lee-Strazicich](../unit-root/lee-strazicich.md) se suspeitar de quebra + raiz unitaria
- **[VAR Stability](../var-stability/index.md)**: Para modelos VAR, a estabilidade dos coeficientes e verificada via raizes do polinomio caracteristico
- **[Cointegration Tests](../cointegration/index.md)**: Quebras afetam testes de cointegracao — considere testes com quebra (Gregory-Hansen)

## See Also

- [CUSUM Test](cusum.md) — Soma acumulada de residuos recursivos
- [Chow Test](chow.md) — Teste com data de quebra conhecida
- [Bai-Perron](bai-perron.md) — Multiplas quebras endogenas
- [Diagnosticos](../index.md) — Visao geral de todos os testes

## Referencias

- Chow, G.C. (1960). "Tests of equality between sets of coefficients in two linear regressions." *Econometrica*, 28(3), 591-605.
- Brown, R.L., Durbin, J. & Evans, J.M. (1975). "Techniques for testing the constancy of regression relationships over time." *JRSS-B*, 37(2), 149-192.
- Bai, J. & Perron, P. (1998). "Estimating and testing linear models with multiple structural changes." *Econometrica*, 66(1), 47-78.
- Bai, J. & Perron, P. (2003). "Computation and analysis of multiple structural change models." *Journal of Applied Econometrics*, 18(1), 1-22.
- Andrews, D.W.K. (1993). "Tests for parameter instability and structural change with unknown change point." *Econometrica*, 61(4), 821-856.
