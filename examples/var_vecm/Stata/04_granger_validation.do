* =============================================================================
* 04_granger_validation.do
* Validacao cruzada: Testes de causalidade de Granger usando Stata
* Compara resultados com chronobox (Python)
* =============================================================================
* Comandos: var, vargranger
* Dataset: us_macro_quarterly.csv
* Tolerancia esperada: < 0.01 para p-valores
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de diretorios ----------------------------------------------
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

capture mkdir "`output_dir'"

display "=== Granger Causality Validation (Stata) ==="
display ""

* --- Carregar dados ----------------------------------------------------------
import delimited "`data_dir'/us_macro_quarterly.csv", clear

display "Dataset: us_macro_quarterly.csv"
display "Observacoes: " _N
display ""

* --- Configurar serie temporal -----------------------------------------------
gen t = _n
tsset t, quarterly

* --- Estimar VAR(4) para testes de Granger -----------------------------------
display "--- Estimacao VAR(4) ---"

* Usamos 4 lags conforme a especificacao Python
var gdp inflation fed_funds unemployment, lags(1/4)

display "Modelo VAR(4) estimado"
display "N. obs: " e(N)
display "N. equacoes: " e(neqs)
display ""

* --- Testes de Causalidade de Granger ----------------------------------------
display "--- Testes de Causalidade de Granger ---"
display "H0: X nao Granger-causa Y"
display "Nivel de significancia: 5%"
display ""

* vargranger executa todos os testes de Granger pairwise e conjuntos
* dentro do VAR estimado
vargranger

* --- Extrair resultados pairwise para CSV ------------------------------------
display ""
display "--- Exportando resultados Granger para CSV ---"

* Para exportacao programatica dos p-valores, fazemos testes individuais
* usando test/testparm apos var

* Lista de variaveis
local varlist "gdp inflation fed_funds unemployment"

* Contar pares (4 variaveis * 3 outras = 12 testes pairwise)
local n_tests = 12
local row = 0

* Criar dataset temporario para resultados
preserve
clear
set obs `n_tests'

gen str20 causing = ""
gen str20 caused = ""
gen double chi2_stat = .
gen int df = .
gen double pvalue = .
gen byte reject_5pct = .

restore

* Reestimar VAR para ter acesso aos coeficientes
quietly var gdp inflation fed_funds unemployment, lags(1/4)

* Testes pairwise: para cada par (X -> Y), testamos se os coeficientes
* de X na equacao de Y sao conjuntamente zero
* Usamos a abordagem de VARs bivariados para consistencia com Python/R

display ""
display "--- Testes Pairwise (bivariados com 4 lags) ---"

local row = 0

* Criar arquivo temporario para salvar resultados
tempname results_handle
tempfile results_file

postfile `results_handle' str20 causing str20 caused double(chi2_stat df pvalue) byte reject_5pct using `results_file'

foreach causing of local varlist {
    foreach caused of local varlist {
        if "`causing'" == "`caused'" continue

        * Estimar VAR bivariado
        quietly var `caused' `causing', lags(1/4)

        * Testar se todos os lags de `causing' sao zero na equacao de `caused'
        * H0: L.`causing' = L2.`causing' = L3.`causing' = L4.`causing' = 0
        quietly test ///
            [#1]L.`causing' ///
            [#1]L2.`causing' ///
            [#1]L3.`causing' ///
            [#1]L4.`causing'

        local chi2 = r(chi2)
        local p = r(p)
        local test_df = r(df)
        local reject = (`p' < 0.05)

        display "  `causing' -> `caused': chi2 = " %10.4f `chi2' ///
                ", df = " `test_df' ///
                ", p = " %10.6f `p' ///
                cond(`reject', " *", "")

        post `results_handle' ("`causing'") ("`caused'") (`chi2') (`test_df') (`p') (`reject')
    }
}

postclose `results_handle'

* Carregar resultados e exportar
preserve
use `results_file', clear

display ""
display "--- Matriz de P-valores ---"
list, noobs clean

* Contar relacoes significativas
count if reject_5pct == 1
local n_sig = r(N)
count
local n_total = r(N)
display ""
display "Relacoes significativas a 5%: `n_sig' de `n_total'"

export delimited using "`output_dir'/granger_results.csv", replace
display ""
display "Resultados Granger salvos em: outputs/Stata/granger_results.csv"
restore

* --- Testes de Granger no VAR multivariado -----------------------------------
display ""
display "--- Testes de Granger no VAR(4) Multivariado ---"

* Reestimar VAR completo
quietly var gdp inflation fed_funds unemployment, lags(1/4)

* vargranger no contexto multivariado
vargranger

display ""
display "=== Fim da validacao Granger (Stata) ==="
