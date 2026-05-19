* =============================================================================
* 01_svar_cholesky_ab_validation.do
* Validacao cruzada: SVAR com identificacao Cholesky e modelo AB usando Stata
* Compara resultados com chronobox (Python) e R (vars/svars)
* =============================================================================
* Comandos: var, matrix, svar, irf create, irf table
* Dataset: us_macro_quarterly.csv
* Variaveis: gdp, inflation, fed_funds, unemployment (sistema 4 variaveis)
* Ordenacao Cholesky: gdp -> inflation -> fed_funds -> unemployment
* Tolerancia esperada: < 1e-3 para IRFs estruturais
* =============================================================================
*
* NOTA SOBRE SVAR NO STATA:
* O comando `svar` do Stata implementa o modelo AB (Amisano & Giannini, 1997):
*   A * u_t = B * e_t
* onde u_t sao residuos da forma reduzida e e_t sao choques estruturais.
*
* Para identificacao Cholesky (recursiva):
*   - A = lower triangular com 1s na diagonal
*   - B = diagonal (desvios padrao dos choques estruturais)
*   Isso equivale a decompor Sigma_u = A^{-1} B B' (A^{-1})'
*
* Para modelo AB com restricoes parciais:
*   - Restricoes customizadas em A e B
*   - Estimacao por ML (scoring algorithm)
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de diretorios ----------------------------------------------
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

* Criar diretorio de saida
capture mkdir "`output_dir'"

display "======================================================================="
display "SVAR Cholesky & AB Model Validation (Stata)"
display "======================================================================="
display ""

* =============================================================================
* PARTE 1: Carregar dados e configurar serie temporal
* =============================================================================

display "--- Carregar Dados ---"

* Importar CSV com dados macroeconomicos trimestrais
import delimited "`data_dir'/us_macro_quarterly.csv", clear

display "Dataset: us_macro_quarterly.csv"
display "Observacoes: " _N
display "Variaveis: gdp, inflation, fed_funds, unemployment"
display ""

* Configurar indice temporal
gen t = _n
tsset t, quarterly

* Estatisticas descritivas
summarize gdp inflation fed_funds unemployment

* =============================================================================
* PARTE 2: Estimacao VAR forma reduzida
* =============================================================================

display ""
display "--- Estimacao VAR(4) forma reduzida ---"

* Estimar VAR(4) com constante
* Mesma especificacao usada no R (vars::VAR) e Python (chronobox)
var gdp inflation fed_funds unemployment, lags(1/4)

display "VAR(4) estimado com sucesso"
display "N. obs: " e(N)
display "N. equacoes: " e(neqs)
display "N. lags: 4"
display ""

* Exibir matriz de covariancia residual (Sigma_u)
display "--- Matriz de Covariancia Residual (Sigma_u) ---"
matrix sigma_u = e(Sigma)
matrix list sigma_u, format(%12.8f)
display ""

* Salvar Sigma_u para comparacao
preserve
clear
svmat sigma_u, names(col)
gen str20 variable = ""
replace variable = "gdp" in 1
replace variable = "inflation" in 2
replace variable = "fed_funds" in 3
replace variable = "unemployment" in 4
order variable
export delimited using "`output_dir'/sigma_u.csv", replace
display "Salvo: sigma_u.csv"
restore

* =============================================================================
* PARTE 3: SVAR com identificacao Cholesky (recursiva)
* =============================================================================

display ""
display "--- SVAR Cholesky (Identificacao Recursiva) ---"
display ""
display "Ordenacao: gdp -> inflation -> fed_funds -> unemployment"
display "A = lower triangular (1s na diagonal)"
display "B = diagonal"
display ""

* Definir restricoes para modelo Cholesky (recursivo)
* A: lower triangular com 1s na diagonal
* Formato Stata: NA => parametro livre; valor fixo => restricao
*
* A = | 1   0   0   0 |    B = | b11  0    0    0   |
*     | a21 1   0   0 |        | 0    b22  0    0   |
*     | a31 a32 1   0 |        | 0    0    b33  0   |
*     | a41 a42 a43 1 |        | 0    0    0    b44 |

* Definir matriz A (4x4)
matrix A_chol = (1, 0, 0, 0 \ ., 1, 0, 0 \ ., ., 1, 0 \ ., ., ., 1)

* Definir matriz B (4x4) - diagonal
matrix B_chol = (., 0, 0, 0 \ 0, ., 0, 0 \ 0, 0, ., 0 \ 0, 0, 0, .)

display "Restricoes A (. = livre, valor = fixo):"
matrix list A_chol
display ""
display "Restricoes B (. = livre, valor = fixo):"
matrix list B_chol
display ""

* Estimar SVAR com restricoes Cholesky
svar gdp inflation fed_funds unemployment, lags(1/4) aeq(A_chol) beq(B_chol)

display "SVAR Cholesky estimado com sucesso"
display ""

* Exibir matrizes estruturais estimadas
display "--- Matriz A Estimada (Cholesky) ---"
matrix A_est_chol = e(A)
matrix list A_est_chol, format(%12.8f)
display ""

display "--- Matriz B Estimada (Cholesky) ---"
matrix B_est_chol = e(B)
matrix list B_est_chol, format(%12.8f)
display ""

* Salvar matrizes A e B estimadas
preserve
clear
svmat A_est_chol, names(col)
gen str20 variable = ""
replace variable = "gdp" in 1
replace variable = "inflation" in 2
replace variable = "fed_funds" in 3
replace variable = "unemployment" in 4
order variable
export delimited using "`output_dir'/svar_A_cholesky.csv", replace
display "Salvo: svar_A_cholesky.csv"
restore

preserve
clear
svmat B_est_chol, names(col)
gen str20 variable = ""
replace variable = "gdp" in 1
replace variable = "inflation" in 2
replace variable = "fed_funds" in 3
replace variable = "unemployment" in 4
order variable
export delimited using "`output_dir'/svar_B_cholesky.csv", replace
display "Salvo: svar_B_cholesky.csv"
restore

* --- IRFs estruturais (Cholesky) ---------------------------------------------
display ""
display "--- IRF Estrutural (Cholesky) ---"

* Criar conjunto de IRFs
* O irf create gera IRFs a partir do ultimo modelo SVAR estimado
irf create svar_chol, set("`output_dir'/svar_chol_irf", replace) step(20)

* Exibir tabela de IRFs
* sirf = structural IRF (impulso-resposta estrutural)
display "IRFs estruturais (Cholesky), horizonte 0-20:"
display ""

irf table sirf, irf(svar_chol) impulse(gdp inflation fed_funds unemployment) ///
    response(gdp inflation fed_funds unemployment) step(20)

* Exportar IRFs para CSV usando irf table
* Salvar resultados como dataset
preserve
irf table sirf, irf(svar_chol) impulse(gdp inflation fed_funds unemployment) ///
    response(gdp inflation fed_funds unemployment) step(20) saving("`output_dir'/svar_irf_cholesky_raw", replace)

* Carregar IRFs salvas e exportar como CSV
use "`output_dir'/svar_irf_cholesky_raw", clear
export delimited using "`output_dir'/svar_irf_cholesky.csv", replace
display "Salvo: svar_irf_cholesky.csv"
restore

* =============================================================================
* PARTE 4: SVAR com modelo AB (restricoes parciais)
* =============================================================================

display ""
display "--- SVAR Modelo AB (Restricoes Parciais) ---"
display ""
display "Restricoes AB customizadas:"
display "  A: lower triangular com 1s na diagonal"
display "     a42, a43 sao livres (unemployment depende de fed_funds e inflation)"
display "  B: diagonal (mesma estrutura que Cholesky)"
display ""

* Modelo AB: mesma estrutura Cholesky mas explicita a parametrizacao AB
* Isso e equivalente ao modelo AB do R (vars::SVAR)
*
* A = | 1    0    0   0 |    B = | b11  0    0    0   |
*     | a21  1    0   0 |        | 0    b22  0    0   |
*     | a31  a32  1   0 |        | 0    0    b33  0   |
*     | a41  a42  a43 1 |        | 0    0    0    b44 |
*
* Nota: No R, usamos restricoes parciais em A (a42=0 no R, aqui a42=livre)
* Para manter consistencia exata, vamos usar a mesma especificacao do R:

* A matrix: identical to R specification
* a21, a31, a32, a41 livres; a42=0, a43=0
matrix A_ab = (1, 0, 0, 0 \ ., 1, 0, 0 \ ., ., 1, 0 \ ., 0, 0, 1)

* B diagonal
matrix B_ab = (., 0, 0, 0 \ 0, ., 0, 0 \ 0, 0, ., 0 \ 0, 0, 0, .)

display "Restricoes A para modelo AB (. = livre):"
matrix list A_ab
display ""
display "Restricoes B para modelo AB (. = livre):"
matrix list B_ab
display ""

* Estimar SVAR com modelo AB
svar gdp inflation fed_funds unemployment, lags(1/4) aeq(A_ab) beq(B_ab)

display "SVAR modelo AB estimado com sucesso"
display ""

* Exibir matrizes estruturais
display "--- Matriz A Estimada (AB model) ---"
matrix A_est_ab = e(A)
matrix list A_est_ab, format(%12.8f)
display ""

display "--- Matriz B Estimada (AB model) ---"
matrix B_est_ab = e(B)
matrix list B_est_ab, format(%12.8f)
display ""

* Calcular A^{-1}B (matriz de impacto estrutural)
matrix A_inv = inv(A_est_ab)
matrix A_inv_B = A_inv * B_est_ab
display "--- A^{-1}B (Matriz de Impacto Estrutural - AB model) ---"
matrix list A_inv_B, format(%12.8f)
display ""

* Salvar matrizes AB
preserve
clear
svmat A_est_ab, names(col)
gen str20 variable = ""
replace variable = "gdp" in 1
replace variable = "inflation" in 2
replace variable = "fed_funds" in 3
replace variable = "unemployment" in 4
order variable
export delimited using "`output_dir'/svar_A_ab.csv", replace
display "Salvo: svar_A_ab.csv"
restore

preserve
clear
svmat B_est_ab, names(col)
gen str20 variable = ""
replace variable = "gdp" in 1
replace variable = "inflation" in 2
replace variable = "fed_funds" in 3
replace variable = "unemployment" in 4
order variable
export delimited using "`output_dir'/svar_B_ab.csv", replace
display "Salvo: svar_B_ab.csv"
restore

preserve
clear
svmat A_inv_B, names(col)
gen str20 variable = ""
replace variable = "gdp" in 1
replace variable = "inflation" in 2
replace variable = "fed_funds" in 3
replace variable = "unemployment" in 4
order variable
export delimited using "`output_dir'/svar_A_inv_B.csv", replace
display "Salvo: svar_A_inv_B.csv"
restore

* --- IRFs do modelo AB -------------------------------------------------------
display ""
display "--- IRF Estrutural (AB model) ---"

irf create svar_ab, set("`output_dir'/svar_ab_irf", replace) step(20)

irf table sirf, irf(svar_ab) impulse(gdp inflation fed_funds unemployment) ///
    response(gdp inflation fed_funds unemployment) step(20)

* Exportar IRFs do modelo AB
preserve
irf table sirf, irf(svar_ab) impulse(gdp inflation fed_funds unemployment) ///
    response(gdp inflation fed_funds unemployment) step(20) saving("`output_dir'/svar_irf_ab_raw", replace)

use "`output_dir'/svar_irf_ab_raw", clear
export delimited using "`output_dir'/svar_irf_ab.csv", replace
display "Salvo: svar_irf_ab.csv"
restore

* =============================================================================
* PARTE 5: Comparacao Cholesky vs AB
* =============================================================================

display ""
display "--- Comparacao: Cholesky vs AB Model ---"
display ""

* Comparar diagonais de B
display "Diagonal de B (Cholesky):"
forvalues i = 1/4 {
    display "  B[`i',`i'] = " B_est_chol[`i',`i']
}
display ""

display "Diagonal de B (AB model):"
forvalues i = 1/4 {
    display "  B[`i',`i'] = " B_est_ab[`i',`i']
}
display ""

* Diferenca maxima entre matrizes de impacto
* Cholesky impact = A_chol^{-1} * B_chol
matrix A_inv_chol = inv(A_est_chol)
matrix impact_chol = A_inv_chol * B_est_chol

display "Matriz de impacto (Cholesky): A^{-1}B"
matrix list impact_chol, format(%12.8f)
display ""

display "Matriz de impacto (AB model): A^{-1}B"
matrix list A_inv_B, format(%12.8f)
display ""

* =============================================================================
* PARTE 6: Notas de implementacao
* =============================================================================

display ""
display "======================================================================="
display "NOTAS DE IMPLEMENTACAO"
display "======================================================================="
display ""
display "1. O comando svar do Stata usa o modelo A*u = B*e (Amisano & Giannini)"
display "2. Identificacao Cholesky: A = lower triangular, B = diagonal"
display "3. Na parametrizacao Stata, '.' indica parametro livre"
display "4. O R (vars::SVAR) usa scoring algorithm, Stata tambem usa ML"
display "5. Pequenas diferencas esperadas entre plataformas por:"
display "   - Algoritmos de otimizacao diferentes"
display "   - Tolerancias de convergencia diferentes"
display "   - Parametrizacao da log-likelihood"
display ""
display "Tolerancia esperada para comparacao:"
display "  - IRFs Cholesky: < 1e-3 (decomposicao analitica)"
display "  - IRFs AB model: < 1e-2 (depende de convergencia ML)"
display ""
display "=== Fim da validacao SVAR Cholesky/AB (Stata) ==="
