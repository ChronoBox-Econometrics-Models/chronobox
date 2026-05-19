---
title: Tutorials
description: Tutoriais praticos e completos para aprender econometria de series temporais com ChronoBox.
---

# Tutorials

Bem-vindo a secao de tutoriais do ChronoBox. Diferente do [User Guide](../user-guide/index.md),
que serve como referencia por modelo, os tutoriais sao **workflows end-to-end** com dados reais
e narrativa didatica. Cada tutorial e auto-contido e executavel.

---

## Learning Paths

Escolha o caminho que melhor se adapta ao seu nivel de experiencia:

=== "Iniciante"

    Ideal para quem esta comecando com series temporais ou migrando de R/Stata para Python.

    ```mermaid
    graph LR
        A[Fundamentals] --> B[ETS]
        style A fill:#26a69a,color:#fff
        style B fill:#26a69a,color:#fff
    ```

    | # | Tutorial | Descricao | Tempo |
    |---|----------|-----------|-------|
    | 1 | [Fundamentals](fundamentals.md) | Sua primeira serie temporal --- ARIMA passo a passo | ~45 min |
    | 2 | [ETS](ets.md) | Suavizacao exponencial e modelos ETS | ~30 min |

=== "Intermediario"

    Para quem ja domina modelos univariados e quer explorar sistemas multivariados.

    ```mermaid
    graph LR
        A[VAR] --> B[SVAR]
        B --> C[VECM]
        style A fill:#42a5f5,color:#fff
        style B fill:#42a5f5,color:#fff
        style C fill:#42a5f5,color:#fff
    ```

    | # | Tutorial | Descricao | Tempo |
    |---|----------|-----------|-------|
    | 1 | [VAR](var.md) | VAR para politica monetaria --- IRF, FEVD, Granger | ~40 min |
    | 2 | [SVAR](svar.md) | Identificacao estrutural e choques ortogonais | ~35 min |
    | 3 | [VECM](vecm.md) | Cointegracao e correcao de erros | ~35 min |

=== "Avancado"

    Para pesquisadores e profissionais que precisam de ferramentas sofisticadas.

    ```mermaid
    graph LR
        A[BVAR] --> B[Filters]
        B --> C[Spillover]
        C --> D[Complete Workflow]
        style A fill:#ab47bc,color:#fff
        style B fill:#ab47bc,color:#fff
        style C fill:#ab47bc,color:#fff
        style D fill:#ab47bc,color:#fff
    ```

    | # | Tutorial | Descricao | Tempo |
    |---|----------|-----------|-------|
    | 1 | [BVAR](bvar.md) | Priors bayesianas e regularizacao | ~40 min |
    | 2 | [Filters](filters.md) | Extracao de ciclos economicos | ~30 min |
    | 3 | [Spillover](spillover.md) | Conectividade e transmissao de choques | ~35 min |
    | 4 | [Complete Workflow](complete-workflow.md) | Projeto completo de pesquisa aplicada | ~60 min |

---

## Todos os Tutoriais

| Tutorial | Nivel | Modelos | Dataset | Tempo |
|----------|-------|---------|---------|-------|
| [Fundamentals](fundamentals.md) | Iniciante | ARIMA, Auto-ARIMA, STL | Airline Passengers | ~45 min |
| [ETS](ets.md) | Iniciante | SES, Holt, Holt-Winters, Auto-ETS | Airline, UK Gas | ~30 min |
| [VAR](var.md) | Intermediario | VAR, IRF, FEVD, Granger | US Macro Quarterly | ~40 min |
| [SVAR](svar.md) | Intermediario | SVAR, Cholesky, Sign Restrictions | US Macro Quarterly | ~35 min |
| [VECM](vecm.md) | Intermediario | VECM, Johansen, ECM | Denmark | ~35 min |
| [BVAR](bvar.md) | Avancado | BayesianVAR, Minnesota Prior | Canada | ~40 min |
| [Filters](filters.md) | Avancado | HP, BK, CF, Hamilton | US GDP | ~30 min |
| [Spillover](spillover.md) | Avancado | Spillover, DY2012 | Exchange Rates | ~35 min |
| [Complete Workflow](complete-workflow.md) | Avancado | Multiplos | Macro Brasil | ~60 min |

---

## Convencoes dos Tutoriais

Todos os tutoriais seguem a mesma estrutura:

!!! info "Estrutura padrao"
    1. **Objetivo** --- o que voce vai aprender
    2. **Dados** --- carregar e explorar o dataset
    3. **Passos numerados** --- workflow completo com codigo executavel
    4. **Conclusao** --- resumo e proximos passos

Os blocos de codigo incluem o output esperado para que voce possa verificar
seus resultados:

```python
# codigo executavel
from chronobox import ARIMA
model = ARIMA(order=(1, 0, 0))
```

```title="Output"
ARIMA(1,0,0) model fitted successfully
```

!!! tip "Dica"
    Todos os datasets utilizados nos tutoriais estao disponiveis via
    `chronobox.datasets.load_dataset()`. Nao e necessario baixar dados externos.
