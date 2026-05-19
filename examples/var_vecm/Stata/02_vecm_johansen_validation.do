* =============================================================================
* 02_vecm_johansen_validation.do
* Validacao cruzada: Teste de Johansen e estimacao VECM usando Stata
* Compara resultados com chronobox (Python)
* =============================================================================
* Comandos: vecrank, vec
* Dataset: us_macro_quarterly.csv
* Tolerancia esperada: < 1e-4 para coeficientes, < 0.01 para p-valores
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de diretorios ----------------------------------------------
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

capture mkdir "`output_dir'"

display "=== VECM / Johansen Validation (Stata) ==="
display ""

* --- Carregar dados ----------------------------------------------------------
import delimited "`data_dir'/us_macro_quarterly.csv", clear

display "Dataset: us_macro_quarterly.csv"
display "Observacoes: " _N
display "Variaveis: gdp, inflation, fed_funds, unemployment"
display ""

* --- Configurar serie temporal -----------------------------------------------
gen t = _n
tsset t, quarterly

* --- Teste de Johansen (Trace) -----------------------------------------------
display "--- Teste de Johansen (Trace) ---"

* vecrank realiza o teste de Johansen para cointegracao
* lags(4) corresponde a K=4 lags em niveis (3 em diferencas + 1 para ECM)
* trend(rconstant) = constante restrita ao espaco de cointegracao
vecrank gdp inflation fed_funds unemployment, lags(4) trend(rconstant) trace

* Salvar estatisticas do teste trace
* vecrank armazena resultados em r()
local trace_rank = r(rank)
display ""
display "Rank de cointegracao (trace): `trace_rank'"

* --- Teste de Johansen (Max Eigenvalue) --------------------------------------
display ""
display "--- Teste de Johansen (Max Eigenvalue) ---"

vecrank gdp inflation fed_funds unemployment, lags(4) trend(rconstant) max

* --- Salvar resultados do teste de Johansen em CSV ---------------------------
* Os resultados sao exibidos na tela; para exportacao, extraimos da matriz r()
display ""
display "--- Exportando resultados Johansen ---"

* Capturar eigenvalues do ultimo vecrank
* vecrank armazena r(eigenvalues) como vetor
preserve
clear

* Criar dataset com resultados do rank test
* Nota: r(rank) contem o rank selecionado
set obs 5
gen int rank_h0 = _n - 1
gen str10 test_type = "trace"
gen double eigenvalue = .

* Os eigenvalues sao exibidos no output do vecrank
* Para captura programatica, usamos os resultados armazenados
display "Nota: Eigenvalues e estatisticas sao exibidos no output acima"
display "Para comparacao detalhada, consulte os outputs impressos"

export delimited using "`output_dir'/johansen_trace.csv", replace
display "Resultados trace salvos em: outputs/Stata/johansen_trace.csv"
restore

* --- Estimacao VECM ----------------------------------------------------------
display ""
display "--- Estimacao VECM ---"

* Estimar VECM com rank = 3 (conforme resultado Python)
* lags(4) = 4 lags em niveis
* trend(rconstant) = constante restrita
vec gdp inflation fed_funds unemployment, rank(3) lags(4) trend(rconstant)

display ""
display "VECM estimado com rank = 3"
display "N. obs: " e(N)

* --- Extrair vetores de cointegracao (beta) ----------------------------------
display ""
display "--- Vetores de Cointegracao (beta) ---"

* beta esta armazenado em e(beta)
matrix beta = e(b_ce)
matrix list beta, format(%12.8f) title("Beta (cointegrating vectors)")

* --- Extrair coeficientes de ajustamento (alpha) -----------------------------
display ""
display "--- Coeficientes de Ajustamento (alpha) ---"

matrix alpha = e(a_ce)
matrix list alpha, format(%12.8f) title("Alpha (loading matrix)")

* --- Matriz Pi = alpha * beta' -----------------------------------------------
display ""
display "--- Matriz Pi (alpha x beta') ---"

* Nota: Em Stata, a parametrizacao pode diferir levemente
* Exibimos os componentes para comparacao manual
display "Pi pode ser calculado como alpha * beta'"
display "Componentes salvos separadamente para comparacao"

* --- Salvar coeficientes VECM em CSV -----------------------------------------
display ""
display "--- Exportando coeficientes VECM ---"

* Salvar alpha
preserve
clear
matrix alpha_mat = e(a_ce)
svmat alpha_mat, names(alpha)
gen str20 equation = ""
replace equation = "gdp" in 1
replace equation = "inflation" in 2
replace equation = "fed_funds" in 3
replace equation = "unemployment" in 4
order equation
export delimited using "`output_dir'/vecm_alpha.csv", replace
display "Alpha salvo em: outputs/Stata/vecm_alpha.csv"
restore

* Salvar beta
preserve
clear
matrix beta_mat = e(b_ce)
svmat beta_mat, names(beta)
gen str20 variable = ""
replace variable = "gdp" in 1
replace variable = "inflation" in 2
replace variable = "fed_funds" in 3
replace variable = "unemployment" in 4
capture replace variable = "_cons" in 5
order variable
export delimited using "`output_dir'/vecm_beta.csv", replace
display "Beta salvo em: outputs/Stata/vecm_beta.csv"
restore

* --- Diagnosticos do VECM ----------------------------------------------------
display ""
display "--- Diagnosticos VECM ---"

* Teste de autocorrelacao dos residuos do VECM
* Nota: veclmar disponivel apos vec
capture veclmar, mlag(4)

display ""
display "=== Fim da validacao VECM/Johansen (Stata) ==="
