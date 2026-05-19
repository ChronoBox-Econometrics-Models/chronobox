---
title: "SVAR: Fundamentos Teoricos"
description: "Teoria completa de modelos SVAR — problema de identificacao, restricoes de curto e longo prazo, sign restrictions, decomposicao de Cholesky, estimacao e IRF estruturais."
---

# SVAR: Fundamentos Teoricos

!!! abstract "Resumo"
    Esta pagina apresenta a teoria matematica dos modelos VAR Estruturais (SVAR),
    focando no problema central de identificacao. Cobre restricoes de exclusao de
    curto prazo, restricoes de longo prazo (Blanchard-Quah), sign restrictions
    (Uhlig), a decomposicao de Cholesky como caso especial, e a estimacao via
    MLE e metodos de momentos. O tratamento segue Kilian & Lutkepohl (2017).

---

## 1. Do VAR Reduzido ao VAR Estrutural

### 1.1 Forma Reduzida

O VAR($p$) na forma reduzida:

$$
\mathbf{y}_t = \boldsymbol{\nu} + \mathbf{A}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{u}_t, \quad \mathbf{u}_t \sim (\mathbf{0}, \boldsymbol{\Sigma}_u)
$$

Os erros $\mathbf{u}_t$ sao contemporaneamente correlacionados ($\boldsymbol{\Sigma}_u$ nao e diagonal),
impedindo a interpretacao causal de choques individuais.

### 1.2 Forma Estrutural

O modelo estrutural:

$$
\mathbf{A}_0\, \mathbf{y}_t = \boldsymbol{\nu}^* + \mathbf{A}_1^* \mathbf{y}_{t-1} + \cdots + \mathbf{A}_p^* \mathbf{y}_{t-p} + \mathbf{B}\, \boldsymbol{\eps}_t
$$

onde:

- $\mathbf{A}_0$ e uma matriz $K \times K$ invertivel de **relacoes contemporaneas**
- $\boldsymbol{\eps}_t \sim (\mathbf{0}, \mathbf{I}_K)$ sao os **choques estruturais** (nao correlacionados, com variancia unitaria)
- $\mathbf{B}$ e uma matriz $K \times K$ de impacto dos choques

### 1.3 Relacao entre Formas

Pre-multiplicando por $\mathbf{A}_0^{-1}$:

$$
\mathbf{y}_t = \mathbf{A}_0^{-1}\boldsymbol{\nu}^* + \mathbf{A}_0^{-1}\mathbf{A}_1^* \mathbf{y}_{t-1} + \cdots + \mathbf{A}_0^{-1}\mathbf{B}\, \boldsymbol{\eps}_t
$$

Portanto:

$$
\mathbf{u}_t = \mathbf{A}_0^{-1}\, \mathbf{B}\, \boldsymbol{\eps}_t
$$

e a restricao fundamental:

$$
\boldsymbol{\Sigma}_u = \E[\mathbf{u}_t \mathbf{u}_t^\top] = \mathbf{A}_0^{-1}\, \mathbf{B}\, \mathbf{B}^\top\, \mathbf{A}_0^{-\top}
$$

---

## 2. O Problema de Identificacao

### 2.1 Contagem de Parametros

A matriz $\boldsymbol{\Sigma}_u$ e simetrica e fornece $K(K+1)/2$ equacoes distintas.
O modelo SVAR geral $\mathbf{A}_0 \mathbf{u}_t = \mathbf{B}\boldsymbol{\eps}_t$ possui $2K^2$
parametros livres (elementos de $\mathbf{A}_0$ e $\mathbf{B}$).

O **deficit de identificacao** e:

$$
2K^2 - \frac{K(K+1)}{2} = \frac{K(3K-1)}{2}
$$

!!! warning "Problema Fundamental"
    Sao necessarias pelo menos $K(3K-1)/2$ restricoes adicionais para identificar
    o modelo. Sem restricoes, infinitas combinacoes de $\mathbf{A}_0$ e $\mathbf{B}$
    geram a mesma $\boldsymbol{\Sigma}_u$.

### 2.2 Modelo AB

No modelo AB (Amisano & Giannini, 1997):

$$
\mathbf{A}\, \mathbf{u}_t = \mathbf{B}\, \boldsymbol{\eps}_t
$$

com a restricao $\boldsymbol{\Sigma}_u = \mathbf{A}^{-1}\mathbf{B}\mathbf{B}^\top\mathbf{A}^{-\top}$.

Caso especial comum: **modelo A** ($\mathbf{B} = \mathbf{I}_K$), onde $\mathbf{A}\mathbf{u}_t = \boldsymbol{\eps}_t$
e $\boldsymbol{\Sigma}_u = \mathbf{A}^{-1}\mathbf{A}^{-\top}$.

### 2.3 Condicao de Ordem e Condicao de Rank

**Condicao de ordem** (necessaria): o numero de restricoes deve ser pelo menos
$K(3K-1)/2$ (para o modelo AB) ou $K(K-1)/2$ (para o modelo A com $\mathbf{B} = \mathbf{I}_K$).

**Condicao de rank** (necessaria e suficiente): a matriz jacobiana das restricoes
deve ter rank igual ao numero de parametros a eliminar. Verificar a condicao de
rank requer a analise caso a caso da estrutura de restricoes.

---

## 3. Restricoes de Exclusao (Curto Prazo)

### 3.1 Intuicao

Restricoes de curto prazo impoe zeros na matriz de impacto contemporaneo
$\mathbf{A}_0^{-1}\mathbf{B}$, significando que certos choques estruturais nao afetam
certas variaveis no **periodo do choque**.

### 3.2 Exemplo: Sistema Bivariado

Para $K = 2$ com o modelo A ($\mathbf{B} = \mathbf{I}_2$):

$$
\begin{pmatrix} 1 & 0 \\ a_{21} & 1 \end{pmatrix} \begin{pmatrix} u_{1t} \\ u_{2t} \end{pmatrix} = \begin{pmatrix} \eps_{1t} \\ \eps_{2t} \end{pmatrix}
$$

A restricao $a_{12} = 0$ significa que $y_2$ nao afeta $y_1$ contemporaneamente.
A inversa:

$$
\mathbf{A}^{-1} = \begin{pmatrix} 1 & 0 \\ -a_{21} & 1 \end{pmatrix}
$$

de modo que $\eps_{1t}$ afeta $y_2$ contemporaneamente (via $-a_{21}$), mas $\eps_{2t}$
nao afeta $y_1$.

### 3.3 Restricoes via Teoria Economica

!!! note "Exemplos de Restricoes Economicas"
    - **Rigidez de precos**: choques de demanda nao afetam o produto no mesmo trimestre
    - **Informacao**: decisoes de politica monetaria precedem a observacao do PIB contemporaneo
    - **Mercados financeiros**: precos de ativos respondem contemporaneamente a todos os choques (sem restricao de zero)

---

## 4. Decomposicao de Cholesky como Caso Especial

### 4.1 Definicao

A decomposicao de Cholesky $\boldsymbol{\Sigma}_u = \mathbf{P}\mathbf{P}^\top$, onde $\mathbf{P}$
e triangular inferior com diagonal positiva, corresponde ao modelo SVAR com:

$$
\mathbf{A}_0 = \mathbf{P}^{-1}, \quad \mathbf{B} = \mathbf{I}_K
$$

### 4.2 Estrutura Recursiva

A decomposicao de Cholesky impoe uma **estrutura causal recursiva**: a variavel
ordenada primeiro nao e afetada contemporaneamente por nenhuma outra, a segunda
so e afetada pela primeira, etc.

Para $K = 3$:

$$
\mathbf{P} = \begin{pmatrix}
p_{11} & 0 & 0 \\
p_{21} & p_{22} & 0 \\
p_{31} & p_{32} & p_{33}
\end{pmatrix}
$$

### 4.3 Exatamente Identificado

A decomposicao de Cholesky impoe $K(K-1)/2$ restricoes de zero (os elementos acima
da diagonal), que e exatamente o numero necessario para identificacao no modelo A.

!!! warning "Dependencia da Ordenacao"
    A decomposicao de Cholesky depende criticamente da **ordenacao das variaveis**.
    Diferentes ordenacoes produzem diferentes identificacoes estruturais e,
    potencialmente, diferentes conclusoes. A ordenacao deve ser justificada por
    teoria economica, nao escolhida por conveniencia.

---

## 5. Restricoes de Longo Prazo (Blanchard-Quah)

### 5.1 Motivacao

Blanchard & Quah (1989) propuseram identificar choques estruturais por seus
efeitos **acumulados de longo prazo**, em vez de impactos contemporaneos.

### 5.2 Multiplicador de Longo Prazo

A representacao VMA($\infty$) do VAR estavel:

$$
\mathbf{y}_t = \boldsymbol{\mu} + \sum_{i=0}^{\infty} \boldsymbol{\Phi}_i\, \mathbf{u}_{t-i}
$$

Em diferencas (para variaveis $I(1)$):

$$
\diff \mathbf{y}_t = \sum_{i=0}^{\infty} \boldsymbol{\Phi}_i\, \mathbf{u}_{t-i}
$$

O efeito acumulado (de longo prazo) dos choques reduzidos sobre os niveis:

$$
\mathbf{C}(1) = \sum_{i=0}^{\infty} \boldsymbol{\Phi}_i = (\mathbf{I}_K - \mathbf{A}_1 - \cdots - \mathbf{A}_p)^{-1}
$$

Para os choques estruturais, o **multiplicador de longo prazo** e:

$$
\boldsymbol{\Xi} = \mathbf{C}(1)\, \mathbf{A}_0^{-1}\, \mathbf{B}
$$

### 5.3 Restricoes

A identificacao de Blanchard-Quah impoe zeros em $\boldsymbol{\Xi}$. Por exemplo,
para distinguir choques de oferta e demanda num sistema bivariado $(y_t, \pi_t)$:

$$
\boldsymbol{\Xi} = \begin{pmatrix} \xi_{11} & 0 \\ \xi_{21} & \xi_{22} \end{pmatrix}
$$

A restricao $\xi_{12} = 0$ significa que o choque de demanda ($\eps_{2t}$) nao tem efeito
permanente sobre o produto — apenas choques de oferta ($\eps_{1t}$) afetam o produto
no longo prazo.

### 5.4 Implementacao

**Passo 1.** Estimar o VAR em diferencas e obter $\hat{\mathbf{A}}_i$ e $\hat{\boldsymbol{\Sigma}}_u$.

**Passo 2.** Computar $\hat{\mathbf{C}}(1) = (\mathbf{I}_K - \hat{\mathbf{A}}_1 - \cdots - \hat{\mathbf{A}}_p)^{-1}$.

**Passo 3.** Computar $\hat{\boldsymbol{\Xi}}\hat{\boldsymbol{\Xi}}^\top = \hat{\mathbf{C}}(1)\, \hat{\boldsymbol{\Sigma}}_u\, \hat{\mathbf{C}}(1)^\top$.

**Passo 4.** Obter $\hat{\boldsymbol{\Xi}}$ via decomposicao de Cholesky de $\hat{\boldsymbol{\Xi}}\hat{\boldsymbol{\Xi}}^\top$.

**Passo 5.** A matriz de impacto contemporaneo: $\hat{\mathbf{A}}_0^{-1}\hat{\mathbf{B}} = \hat{\mathbf{C}}(1)^{-1}\hat{\boldsymbol{\Xi}}$.

---

## 6. Sign Restrictions (Uhlig, 2005)

### 6.1 Motivacao

As restricoes de zero (curto ou longo prazo) sao frequentemente controversas.
Uhlig (2005) propoe identificar choques restringindo apenas o **sinal** das IRFs
em determinados horizontes.

### 6.2 Formulacao

Seja $\mathbf{Q}$ uma matriz ortogonal ($\mathbf{Q}\mathbf{Q}^\top = \mathbf{I}_K$). Se
$\hat{\boldsymbol{\Sigma}}_u = \hat{\mathbf{P}}\hat{\mathbf{P}}^\top$ (Cholesky), entao
$\tilde{\mathbf{P}} = \hat{\mathbf{P}}\mathbf{Q}$ tambem satisfaz
$\hat{\boldsymbol{\Sigma}}_u = \tilde{\mathbf{P}}\tilde{\mathbf{P}}^\top$.

Cada coluna $\mathbf{q}_j$ de $\mathbf{Q}$ define um candidato a choque estrutural com IRF:

$$
\boldsymbol{\Theta}_i^{(j)} = \hat{\boldsymbol{\Phi}}_i\, \hat{\mathbf{P}}\, \mathbf{q}_j
$$

### 6.3 Algoritmo

1. Estimar o VAR e computar $\hat{\mathbf{P}}$ e $\hat{\boldsymbol{\Phi}}_i$
2. Gerar uma matriz ortogonal aleatoria $\mathbf{Q}$ (via decomposicao QR de uma matriz aleatoria gaussiana)
3. Computar as IRFs candidatas $\boldsymbol{\Theta}_i^{(j)} = \hat{\boldsymbol{\Phi}}_i \hat{\mathbf{P}} \mathbf{q}_j$
4. Verificar se as restricoes de sinal sao satisfeitas para os horizontes especificados
5. Se sim, armazenar a rotacao; se nao, descartar
6. Repetir passos 2-5 um grande numero de vezes
7. Reportar a mediana e os percentis das IRFs retidas

### 6.4 Identificacao por Conjuntos

!!! warning "Identificacao Parcial"
    Sign restrictions tipicamente nao **identificam pontualmente** os choques estruturais.
    O resultado e um **conjunto identificado** de modelos observacionalmente equivalentes
    que satisfazem as restricoes. Isso implica:

    - IRFs sao reportadas como bandas, nao como funcoes pontuais
    - A mediana das IRFs retidas **nao** e uma estimativa consistente da IRF verdadeira
    - Restricoes adicionais (zero + sinal) podem reduzir o conjunto identificado

---

## 7. Estimacao do SVAR

### 7.1 MLE para o Modelo A

Para o modelo A ($\mathbf{B} = \mathbf{I}_K$), a log-verossimilhanca concentrada:

$$
\ln L(\mathbf{A}_0) = \text{const.} + T \ln|\det(\mathbf{A}_0)| - \frac{T}{2}\, \text{tr}\left(\mathbf{A}_0^\top \mathbf{A}_0\, \hat{\boldsymbol{\Sigma}}_u\right)
$$

onde $\hat{\boldsymbol{\Sigma}}_u$ e a covariancia residual do VAR reduzido.

A maximizacao e sujeita as restricoes de identificacao e realizada via algoritmos
de otimizacao numerica com restricoes (scoring algorithm de Amisano & Giannini, 1997).

### 7.2 MLE para o Modelo AB

Para o modelo AB geral:

$$
\ln L(\mathbf{A}, \mathbf{B}) = \text{const.} + T \ln|\det(\mathbf{A})| - T \ln|\det(\mathbf{B})| - \frac{T}{2}\, \text{tr}\left(\mathbf{B}^{-\top}\mathbf{A}^\top \mathbf{A}\, \hat{\boldsymbol{\Sigma}}_u\, \mathbf{B}^{-1}\right)
$$

### 7.3 Inferencia

=== "Assintotica"

    Os estimadores MLE dos parametros estruturais sao assintoticamente normais:

    $$
    \sqrt{T}(\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}_0) \xrightarrow{d} N(\mathbf{0}, \boldsymbol{\Sigma}_\theta)
    $$

    onde $\boldsymbol{\Sigma}_\theta$ e obtida via o inverso da informacao de Fisher observada.

=== "Bootstrap"

    O bootstrap residual e frequentemente preferido na pratica:

    1. Estimar o VAR reduzido e os parametros estruturais
    2. Gerar pseudo-amostras via reamostragem dos residuos
    3. Re-estimar o modelo completo em cada replicacao
    4. Construir intervalos de confianca via percentis

---

## 8. IRF Estruturais vs. Forma Reduzida

### 8.1 IRF na Forma Reduzida

$$
\mathbf{y}_{t+i} = \cdots + \boldsymbol{\Phi}_i\, \mathbf{u}_t + \cdots
$$

O elemento $(k, \ell)$ de $\boldsymbol{\Phi}_i$ mede a resposta de $y_k$ a um choque
unitario em $u_\ell$, mas os choques $u_\ell$ sao correlacionados e **nao tem
interpretacao estrutural**.

### 8.2 IRF Estrutural

$$
\mathbf{y}_{t+i} = \cdots + \boldsymbol{\Theta}_i\, \boldsymbol{\eps}_t + \cdots
$$

onde $\boldsymbol{\Theta}_i = \boldsymbol{\Phi}_i\, \mathbf{A}_0^{-1}\, \mathbf{B}$.

O elemento $(k, j)$ de $\boldsymbol{\Theta}_i$ mede a resposta de $y_k$ a um choque
estrutural unitario $\eps_j$, que e nao correlacionado com os demais choques
e possui interpretacao economica (e.g., choque de politica monetaria, choque
de oferta).

### 8.3 Multiplicador Acumulado

A IRF acumulada ate o horizonte $h$:

$$
\boldsymbol{\Xi}_h = \sum_{i=0}^{h} \boldsymbol{\Theta}_i
$$

O limite $\boldsymbol{\Xi}_\infty = \lim_{h \to \infty} \boldsymbol{\Xi}_h = \mathbf{C}(1)\, \mathbf{A}_0^{-1}\, \mathbf{B}$
e o multiplicador de longo prazo utilizado nas restricoes de Blanchard-Quah.

---

## 9. FEVD Estrutural

A decomposicao da variancia do erro de previsao com choques estruturais:

$$
\omega_{kj}(h) = \frac{\sum_{i=0}^{h-1} [\boldsymbol{\Theta}_i]_{kj}^2}{\sum_{i=0}^{h-1} \sum_{\ell=1}^{K} [\boldsymbol{\Theta}_i]_{k\ell}^2}
$$

Interpretacao: $\omega_{kj}(h)$ mede a fracao da variancia do erro de previsao $h$
passos a frente da variavel $k$ que e explicada pelo choque estrutural $j$.

Por construcao: $\sum_{j=1}^{K} \omega_{kj}(h) = 1$.

---

## 10. Decomposicao Historica

A **decomposicao historica** (historical decomposition) atribui cada observacao
aos choques estruturais individuais:

$$
\mathbf{y}_t = \underbrace{\boldsymbol{\mathcal{A}}^t \mathbf{Y}_0}_{\text{condicao inicial}} + \sum_{j=1}^{K}\; \underbrace{\sum_{i=0}^{t-1} \boldsymbol{\Theta}_i\, \mathbf{e}_j\, \eps_{j,t-i}}_{\text{contribuicao do choque } j}
$$

onde $\mathbf{e}_j$ e o $j$-esimo vetor canonico. Isso permite avaliar a importancia
relativa de cada choque em episodios historicos especificos (e.g., "quanto da
recessao de 2008 foi causada por choques financeiros?").

---

## Referencias

- Amisano, G. & Giannini, C. (1997). *Topics in Structural VAR Econometrics*. 2nd ed., Springer.
- Blanchard, O. J. & Quah, D. (1989). "The Dynamic Effects of Aggregate Demand and Supply Disturbances." *American Economic Review*, 79(4), 655-673.
- Christiano, L. J., Eichenbaum, M. & Evans, C. L. (1999). "Monetary Policy Shocks: What Have We Learned and to What End?" In *Handbook of Macroeconomics*, Vol. 1A, Elsevier.
- Kilian, L. & Lutkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.
- Rubio-Ramirez, J. F., Waggoner, D. F. & Zha, T. (2010). "Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference." *Review of Economic Studies*, 77(2), 665-696.
- Sims, C. A. (1980). "Macroeconomics and Reality." *Econometrica*, 48(1), 1-48.
- Uhlig, H. (2005). "What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure." *Journal of Monetary Economics*, 52(2), 381-419.

---

## Veja Tambem

- [VAR](var-theory.md) — Modelo VAR reduzido e suas propriedades
- [VECM & Cointegracao](vecm-theory.md) — SVAR com variaveis cointegradas
- [ARIMA](arima-theory.md) — Modelos univariados como caso especial
