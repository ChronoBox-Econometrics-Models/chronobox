---
title: "Lee-Strazicich Test"
description: "Teste Lee-Strazicich LM de raiz unitaria com uma ou duas quebras estruturais endogenas no chronobox — size-correct sob H0, modelos A e C, exemplos praticos."
---

# Lee-Strazicich

!!! info "Quick Reference"
    **Funcao:** `chronobox.tests_stat.unit_root.lee_strazicich_test()`
    **H₀:** Raiz unitaria **com** quebra(s) estrutural(is)
    **H₁:** Serie estacionaria com quebra(s) estrutural(is)
    **Distribuicao:** Lee-Strazicich (propria)
    **Valores criticos:** Lee & Strazicich (2003, 2004)
    **R equivalente:** `breaks::ur.ls()` (pacote nao-padrao)

## Motivacao

O teste de [Zivot-Andrews](zivot-andrews.md) tem uma limitacao importante: sua hipotese nula e raiz unitaria **sem** quebra. Se a serie de fato tem raiz unitaria **com** quebra, o teste pode rejeitar incorretamente H₀, levando a conclusoes erroneas.

O teste de Lee-Strazicich resolve isso com uma formulacao onde H₀ **inclui** as quebras:

| Teste | H₀ | Problema |
|:------|:---|:---------|
| Zivot-Andrews | Raiz unitaria sem quebra | Pode rejeitar espuriamente se ha raiz unitaria com quebra |
| **Lee-Strazicich** | **Raiz unitaria com quebras** | **Size-correct** — tamanho correto sob H₀ |

!!! tip "Vantagem Principal"
    O Lee-Strazicich e **size-correct** sob H₀: o tamanho empirico do teste e proximo ao nivel nominal, independentemente de haver quebras sob a hipotese nula. Isso o torna mais confiavel que o Zivot-Andrews.

## Hipoteses

$$H_0: \text{Raiz unitaria com quebra(s) estrutural(is)}$$

$$H_1: \text{Serie estacionaria com quebra(s) estrutural(is)}$$

A diferenca fundamental: **ambas** as hipoteses permitem quebras estruturais.

## Dois Modelos de Quebra

### Modelo A ("break"): Quebra no Intercepto

Permite mudancas abruptas no **nivel** da serie em cada data de quebra.

Sob H₁, a serie segue:

$$y_t = \alpha + \delta_1 DU_{1t} + \delta_2 DU_{2t} + \beta t + \varepsilon_t$$

onde $DU_{jt} = 1$ se $t > T_{Bj}$, e $0$ caso contrario.

**Uso:** Choques permanentes de nivel (desvalorizacao, mudanca de patamar de precos).

### Modelo C ("crash"): Quebra no Intercepto e na Tendencia

Permite mudancas no **nivel e na taxa de crescimento** em cada data de quebra.

$$y_t = \alpha + \delta_1 DU_{1t} + \delta_2 DU_{2t} + \beta t + \gamma_1 DT_{1t} + \gamma_2 DT_{2t} + \varepsilon_t$$

onde $DT_{jt} = (t - T_{Bj})$ se $t > T_{Bj}$, e $0$ caso contrario.

**Uso:** Mudancas estruturais completas (crises com recuperacao em taxa diferente).

## Estatistica de Teste (LM)

O teste utiliza o **principio do Multiplicador de Lagrange (LM)**:

### Passo 1: Estimar Tendencia sob H₀

Sob H₀ (raiz unitaria), estimar $\hat{\delta}$ da regressao em primeiras diferencas:

$$\Delta y_t = \delta' \Delta Z_t + \varepsilon_t$$

onde $Z_t$ contem os termos deterministicos (constante, tendencia, dummies de quebra).

### Passo 2: Serie Detrended

Calcular a serie detrended sob H₀:

$$\tilde{S}_t = y_t - \hat{\psi}_x - \hat{\delta}' Z_t$$

onde $\hat{\psi}_x = y_1 - \hat{\delta}' Z_1$.

### Passo 3: Regressao LM

Estimar:

$$\Delta y_t = \delta' \Delta Z_t + \phi \tilde{S}_{t-1} + \sum_{j=1}^{p} \psi_j \Delta \tilde{S}_{t-j} + \varepsilon_t$$

### Passo 4: Estatistica

$$\tau_{LM} = \frac{\hat{\phi}}{SE(\hat{\phi})}$$

A data(s) de quebra e selecionada(s) para **minimizar** $\tau_{LM}$:

$$T_B^* = \arg\min_{T_B} \tau_{LM}(T_B)$$

## Valores Criticos

Os valores criticos dependem do numero de quebras e do modelo:

=== "1 Quebra"

    | Significancia | Modelo A ("break") | Modelo C ("crash") |
    |:-------------|:------------------|:------------------|
    | 1% | -4.239 | -5.11 |
    | 5% | -3.566 | -4.50 |
    | 10% | -3.211 | -4.21 |

=== "2 Quebras"

    | Significancia | Modelo A ("break") | Modelo C ("crash") |
    |:-------------|:------------------|:------------------|
    | 1% | -4.545 | -5.823 |
    | 5% | -3.842 | -5.286 |
    | 10% | -3.504 | -4.989 |

!!! note
    Com 2 quebras no Modelo C, os valores criticos sao bastante restritivos (≈ -5.3 a 5%), refletindo a maior flexibilidade do modelo.

## Exemplo Pratico

### Uma Quebra Estrutural

```python
import numpy as np
from chronobox.tests_stat.unit_root import lee_strazicich_test

# Serie estacionaria com quebra de nivel em t=120
np.random.seed(42)
T = 250
y = np.zeros(T)
for t in range(1, T):
    y[t] = 0.7 * y[t - 1] + np.random.randn()
y[120:] += 4.0  # quebra de nivel

result = lee_strazicich_test(y, model="break", breaks=1)
print(result.summary())

# Acessar informacoes da quebra
print(f"\nData da quebra: t={result.additional_info['break_dates']}")
print(f"Fracao: {result.additional_info['break_fractions']}")
```

### Duas Quebras Estruturais

```python
# Serie com duas quebras
y_2breaks = np.zeros(300)
for t in range(1, 300):
    y_2breaks[t] = 0.6 * y_2breaks[t - 1] + np.random.randn()
y_2breaks[80:] += 3.0   # primeira quebra
y_2breaks[200:] -= 4.0  # segunda quebra

result_2 = lee_strazicich_test(y_2breaks, model="break", breaks=2)
print(result_2.summary())

print(f"\nQuebras detectadas: {result_2.additional_info['break_dates']}")
print(f"Fracoes: {[f'{f:.2f}' for f in result_2.additional_info['break_fractions']]}")
```

### Comparacao com Zivot-Andrews

```python
from chronobox.tests_stat.unit_root import zivot_andrews_test, lee_strazicich_test

# Serie com raiz unitaria E quebra (caso problematico para ZA)
np.random.seed(42)
y_rw_break = np.cumsum(np.random.randn(250))
y_rw_break[100:] += 5.0  # quebra em random walk

# Zivot-Andrews pode rejeitar espuriamente
za = zivot_andrews_test(y_rw_break, model="a")
print(f"ZA:  stat={za.statistic:.4f}, rejeita={za.reject_at_5pct}")
# Pode rejeitar H0 incorretamente (confunde quebra com estacionaridade)

# Lee-Strazicich e size-correct
ls = lee_strazicich_test(y_rw_break, model="break", breaks=1)
print(f"LS:  stat={ls.statistic:.4f}, rejeita={ls.reject_at_5pct}")
# Nao deve rejeitar H0 (raiz unitaria com quebra)
```

### Modelo Crash (Intercepto + Tendencia)

```python
# Serie com quebra na tendencia
T = 300
t = np.arange(T, dtype=float)
y_trend_break = np.zeros(T)
for i in range(1, T):
    y_trend_break[i] = 0.8 * y_trend_break[i - 1] + np.random.randn()
y_trend_break += 0.05 * t
y_trend_break[150:] += 3.0 + 0.1 * (t[150:] - 150)  # quebra nivel + tendencia

result_crash = lee_strazicich_test(y_trend_break, model="crash", breaks=1)
print(result_crash.summary())
```

## Assinatura da Funcao

```python
lee_strazicich_test(
    y: NDArray,
    model: str = "break",       # 'break' (intercepto) ou 'crash' (intercepto + tendencia)
    breaks: int = 1,            # 1 ou 2 quebras
    trim: float = 0.15,         # Fracao a excluir das pontas
    maxlag: int | None = None   # None = 12*(T/100)^{1/4}
) -> TestResult
```

### Campos Adicionais em `additional_info`

| Campo | Tipo | Descricao |
|:------|:-----|:----------|
| `break_dates` | `list[int]` | Indices das datas de quebra |
| `break_fractions` | `list[float]` | Fracoes da amostra ($T_B/T$) |
| `model` | `str` | Modelo utilizado (`'break'` ou `'crash'`) |
| `n_breaks` | `int` | Numero de quebras |
| `nobs` | `int` | Tamanho da amostra |

## Zivot-Andrews vs Lee-Strazicich

| Aspecto | Zivot-Andrews | Lee-Strazicich |
|:--------|:-------------|:---------------|
| H₀ | Raiz unitaria **sem** quebra | Raiz unitaria **com** quebra(s) |
| Metodo | ADF modificado | LM |
| Quebras | 1 | 1 ou 2 |
| Size sob H₀ | Distorcido se ha quebras sob H₀ | **Correto** |
| Modelo intercepto | Model A | `model='break'` |
| Modelo tendencia | Model B | — |
| Modelo ambos | Model C | `model='crash'` |
| p-valor | Nao disponivel | Nao disponivel |

## Equivalente R

```r
# Pacote 'breaks' (nao CRAN — disponivel via GitHub)
# install.packages("breaks")
library(breaks)

# Lee-Strazicich com 1 quebra
result <- ur.ls(y, model = "break", breaks = 1)

# Lee-Strazicich com 2 quebras
result2 <- ur.ls(y, model = "crash", breaks = 2)

# Alternativa: implementacao manual usando o pacote 'strucchange'
# para busca de quebras + teste LM
```

!!! note
    Nao existe implementacao canonica do Lee-Strazicich no R base ou nos pacotes padrao (urca, tseries). Implementacoes estao disponiveis em pacotes como `breaks` ou via codigo customizado.

## See Also

- [Zivot-Andrews](zivot-andrews.md) — Teste com uma quebra (H₀ sem quebra)
- [ADF Test](adf.md) — Teste padrao sem quebras
- [Unit Root Tests](index.md) — Visao geral e comparacao

## Referencias

- Lee, J. & Strazicich, M.C. (2003). "Minimum Lagrange multiplier unit root test with two structural breaks." *Review of Economics and Statistics*, 85(4), 1082-1089.
- Lee, J. & Strazicich, M.C. (2004). "Minimum LM unit root test with one structural break." Manuscript, Appalachian State University.
- Perron, P. (1989). "The great crash, the oil price shock, and the unit root hypothesis." *Econometrica*, 57(6), 1361-1401.
