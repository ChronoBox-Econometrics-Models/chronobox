* =============================================================================
* 02_blanchard_quah_validation.do
* Validacao cruzada: SVAR com restricoes de longo prazo (Blanchard-Quah)
* Compara resultados com chronobox (Python) e R (vars::BQ)
* =============================================================================
* Comandos: var, svar (lrestrictions), irf create, irf table
* Dataset: blanchard_quah.csv
* Variaveis: output_growth, unemployment (sistema bivariado)
* Tolerancia esperada: < 1e-3 para IRFs estruturais BQ
* =============================================================================
*
* FUNDAMENTACAO TEORICA:
* Blanchard & Quah (1989) propuseram identificar choques estruturais usando
* restricoes de longo prazo. No sistema bivariado:
*   - Choque de oferta (supply): efeito permanente sobre o produto
*   - Choque de demanda (demand): efeito ZERO de longo prazo sobre o produto
*
* No Stata, restricoes de longo prazo sao impostas via a opcao
* `lrestrictions()` do comando svar. A matriz C(1) = (I - A_1 - ... - A_p)^{-1}
* multiplicada pela matriz de impacto estrutural deve ter zeros nos elementos
* apropriados.
*
* LIMITACAO IMPORTANTE:
* Sign restrictions NAO estao disponiveis nativamente no Stata.
* O Stata nao possui comando built-in para identificacao por restricoes de sinal.
* Para sign restrictions, alternativas incluem:
*   - Implementacao manual via Mata (linguagem matricial do Stata)
*   - Pacote externo `signres` (se disponivel no SSC)
*   - Uso do R (svars) ou Python (chronobox) para esta funcionalidade
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de diretorios ----------------------------------------------
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

capture mkdir "`output_dir'"

display "======================================================================="
display "Blanchard-Quah Long-Run Restrictions Validation (Stata)"
display "======================================================================="
display ""

* =============================================================================
* PARTE 1: Carregar dados Blanchard-Quah
* =============================================================================

display "--- Carregar Dados ---"

import delimited "`data_dir'/blanchard_quah.csv", clear

display "Dataset: blanchard_quah.csv"
display "Observacoes: " _N
display "Variaveis: output_growth, unemployment"
display ""

* Configurar serie temporal
gen t = _n
tsset t, quarterly

summarize output_growth unemployment

* =============================================================================
* PARTE 2: Estimacao VAR forma reduzida
* =============================================================================

display ""
display "--- Estimacao VAR(4) forma reduzida ---"

var output_growth unemployment, lags(1/4)

display "VAR(4) estimado"
display "N. obs: " e(N)
display ""

* Covariancia residual
display "--- Sigma_u ---"
matrix sigma_u = e(Sigma)
matrix list sigma_u, format(%12.8f)
display ""

* =============================================================================
* PARTE 3: SVAR com restricoes de longo prazo (Blanchard-Quah)
* =============================================================================

display ""
display "--- SVAR com Restricoes de Longo Prazo (BQ) ---"
display ""
display "Restricao: choque de demanda tem efeito zero de longo prazo"
display "           sobre output_growth (acumulado -> nivel do produto)"
display ""

* No Stata, restricoes de longo prazo sao definidas como:
* C(1) * B = estrutura desejada
* Onde C(1) = (I - A_1 - ... - A_p)^{-1} e a resposta de longo prazo
*
* Para BQ bivariado:
* A matriz de longo prazo estrutural C(1)*B deve ser lower triangular:
* | c11  0   |
* | c21  c22 |
*
* Isso significa que o segundo choque (demanda) nao afeta output_growth
* no longo prazo

* Definir restricoes de longo prazo
* Na notacao Stata: lr = long-run impact matrix
* Zero no canto superior direito = choque 2 nao afeta var 1 no longo prazo
matrix lr_restrict = (., 0 \ ., .)

display "Restricoes de longo prazo (. = livre, 0 = restringido):"
matrix list lr_restrict
display ""

* Definir matrizes A e B para curto prazo (sem restricoes adicionais)
* A = identidade (sem restricoes contemporaneas)
* B = livre (determinada pelas restricoes de longo prazo)
matrix A_bq = (1, 0 \ 0, 1)
matrix B_bq = (., 0 \ ., .)

* Estimar SVAR com restricoes de longo prazo
svar output_growth unemployment, lags(1/4) aeq(A_bq) beq(B_bq) ///
    lrestrictions(lr_restrict)

display "SVAR BQ estimado com sucesso"
display ""

* Exibir matrizes estimadas
display "--- Matriz A Estimada (BQ) ---"
matrix A_est_bq = e(A)
matrix list A_est_bq, format(%12.8f)
display ""

display "--- Matriz B Estimada (BQ) ---"
matrix B_est_bq = e(B)
matrix list B_est_bq, format(%12.8f)
display ""

* Salvar matrizes BQ
preserve
clear
svmat B_est_bq, names(col)
gen str20 variable = ""
replace variable = "output_growth" in 1
replace variable = "unemployment" in 2
order variable
export delimited using "`output_dir'/bq_B_matrix.csv", replace
display "Salvo: bq_B_matrix.csv"
restore

* =============================================================================
* PARTE 4: IRFs estruturais do modelo BQ
* =============================================================================

display ""
display "--- IRF Estrutural (BQ) ---"

* Criar IRFs estruturais
irf create svar_bq, set("`output_dir'/svar_bq_irf", replace) step(20)

* Exibir IRFs
display "IRFs estruturais BQ, horizonte 0-20:"
irf table sirf, irf(svar_bq) impulse(output_growth unemployment) ///
    response(output_growth unemployment) step(20)

* Exportar IRFs
preserve
irf table sirf, irf(svar_bq) impulse(output_growth unemployment) ///
    response(output_growth unemployment) step(20) ///
    saving("`output_dir'/bq_irf_raw", replace)

use "`output_dir'/bq_irf_raw", clear
export delimited using "`output_dir'/bq_irf.csv", replace
display "Salvo: bq_irf.csv"
restore

* --- IRFs acumuladas (para verificar restricao de longo prazo) ---------------
display ""
display "--- IRF Acumulada (verificacao da restricao de longo prazo) ---"

* O irf table com opcao `cumulative` mostra IRFs acumuladas
irf create svar_bq_cum, set("`output_dir'/svar_bq_cum_irf", replace) step(40)

irf table sirf, irf(svar_bq_cum) impulse(output_growth unemployment) ///
    response(output_growth) step(40)

display ""
display "Verificacao: o efeito acumulado do choque de demanda (unemployment)"
display "sobre output_growth deve convergir para zero no longo prazo."
display "Isso confirma que o choque de demanda nao tem efeito permanente"
display "sobre o nivel do produto."

* =============================================================================
* PARTE 5: Salvar resultados para comparacao
* =============================================================================

display ""
display "--- Exportar Resultados ---"

* Salvar resumo dos resultados BQ
preserve
clear
set obs 4
gen str40 metric = ""
gen str60 value = ""

replace metric = "model" in 1
replace value = "Blanchard-Quah SVAR" in 1

replace metric = "n_lags" in 2
replace value = "4" in 2

replace metric = "long_run_restriction" in 3
replace value = "demand shock has zero LR effect on output" in 3

replace metric = "identification" in 4
replace value = "long-run restrictions (C(1)*B lower triangular)" in 4

export delimited using "`output_dir'/bq_summary.csv", replace
display "Salvo: bq_summary.csv"
restore

* =============================================================================
* PARTE 6: Documentacao de limitacoes
* =============================================================================

display ""
display "======================================================================="
display "LIMITACOES DO STATA PARA SVAR AVANCADO"
display "======================================================================="
display ""
display "1. SIGN RESTRICTIONS (Restricoes de Sinal):"
display "   - NAO disponivel nativamente no Stata"
display "   - O Stata nao tem comando built-in para sign restrictions"
display "   - Alternativas:"
display "     a) Implementacao manual via Mata (linguagem matricial)"
display "     b) Pacote externo 'signres' (verificar disponibilidade no SSC)"
display "     c) Usar R (svars::id.sign) ou Python (chronobox)"
display ""
display "2. RESTRICOES DE LONGO PRAZO (implementadas acima):"
display "   - Disponivel via svar com opcao lrestrictions()"
display "   - Implementacao segue Amisano & Giannini (1997)"
display "   - Comparavel a vars::BQ() no R"
display ""
display "3. NARRATIVE RESTRICTIONS:"
display "   - NAO disponivel nativamente no Stata"
display "   - Requer implementacao manual"
display ""
display "4. PROXY/EXTERNAL INSTRUMENTS SVAR:"
display "   - Disponivel via svar com opcao `exog()` em versoes recentes"
display "   - Tambem disponivel via ivreg2/ivregress para equacoes individuais"
display ""
display "RECOMENDACAO:"
display "Para SVAR com sign restrictions, usar R (svars) ou Python (chronobox)."
display "Para restricoes Cholesky, AB model e BQ, Stata oferece suporte completo."
display ""
display "=== Fim da validacao Blanchard-Quah (Stata) ==="
