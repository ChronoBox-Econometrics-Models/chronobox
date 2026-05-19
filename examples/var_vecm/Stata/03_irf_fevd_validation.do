* =============================================================================
* 03_irf_fevd_validation.do
* Validacao cruzada: IRF e FEVD usando Stata
* Compara resultados com chronobox (Python)
* =============================================================================
* Comandos: var, irf create, irf table
* Dataset: us_macro_quarterly.csv
* Tolerancia esperada: < 1e-4 para IRF/FEVD
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de diretorios ----------------------------------------------
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

capture mkdir "`output_dir'"

display "=== IRF & FEVD Validation (Stata) ==="
display ""

* --- Carregar dados ----------------------------------------------------------
import delimited "`data_dir'/us_macro_quarterly.csv", clear

display "Dataset: us_macro_quarterly.csv"
display "Observacoes: " _N
display ""

* --- Configurar serie temporal -----------------------------------------------
gen t = _n
tsset t, quarterly

* --- Estimar VAR(1) ----------------------------------------------------------
display "--- Estimacao VAR(1) ---"
var gdp inflation fed_funds unemployment, lags(1)

display "Modelo VAR(1) estimado"
display "N. obs: " e(N)
display ""

* --- IRF Ortogonalizada (Cholesky) -------------------------------------------
display "--- IRF Ortogonalizada (Cholesky) ---"

* Numero de periodos adiante
local n_ahead = 20

* Criar arquivo IRF
* set = 0 indica ortogonalizada (Cholesky), que e o padrao
* step() define o horizonte maximo
irf create var_irf, set("`output_dir'/var_irf.irf") step(`n_ahead') replace

* --- Exibir tabelas IRF para cada combinacao impulso-resposta ----------------
display ""
display "--- Tabelas IRF (ortogonalizadas) ---"

* IRF ortogonalizada: impulso em cada variavel, resposta em cada variavel
* oirf = orthogonalized impulse-response function
irf table oirf, impulse(gdp inflation fed_funds unemployment) ///
    response(gdp inflation fed_funds unemployment) ///
    set("`output_dir'/var_irf.irf") irf(var_irf)

* --- Exportar IRF para CSV ---------------------------------------------------
display ""
display "--- Exportando IRF para CSV ---"

* Carregar dados do IRF para exportacao
preserve
use "`output_dir'/var_irf.irf", clear

* O arquivo .irf contem variaveis como:
* step, irfname, impulse, response, oirf, etc.
* Renomear e filtrar para formato compativel com Python/R
keep step irfname impulse response oirf

* Renomear para consistencia
rename step horizon
rename oirf irf

* Filtrar apenas o IRF que criamos
keep if irfname == "var_irf"
drop irfname

* Converter impulse e response de numericos para nomes
* Nota: Stata armazena como variaveis string no arquivo IRF
* Se necessario, decodificar labels

export delimited using "`output_dir'/irf_results.csv", replace
display "IRF salvo em: outputs/Stata/irf_results.csv"
restore

* --- FEVD (Forecast Error Variance Decomposition) ----------------------------
display ""
display "--- FEVD ---"

* Exibir tabela FEVD
irf table fevd, impulse(gdp inflation fed_funds unemployment) ///
    response(gdp inflation fed_funds unemployment) ///
    set("`output_dir'/var_irf.irf") irf(var_irf)

* --- Exportar FEVD para CSV --------------------------------------------------
display ""
display "--- Exportando FEVD para CSV ---"

preserve
use "`output_dir'/var_irf.irf", clear

* Manter apenas dados de FEVD
keep step irfname impulse response fevd

rename step horizon
rename impulse shock

keep if irfname == "var_irf"
drop irfname

export delimited using "`output_dir'/fevd_results.csv", replace
display "FEVD salvo em: outputs/Stata/fevd_results.csv"
restore

* --- IRF com intervalos de confianca (bootstrap) -----------------------------
display ""
display "--- IRF com Bootstrap CI ---"

* Criar IRF com bootstrap para intervalos de confianca
* reps() define o numero de replicacoes bootstrap
* bsp = bootstrap percentile CI
irf create var_irf_bs, set("`output_dir'/var_irf_bs.irf") ///
    step(`n_ahead') bs reps(500) replace

* Exibir IRF com intervalos de confianca
irf table oirf, impulse(gdp) response(gdp inflation fed_funds unemployment) ///
    set("`output_dir'/var_irf_bs.irf") irf(var_irf_bs) ///
    level(95)

* --- Exportar IRF com CI para CSV --------------------------------------------
preserve
use "`output_dir'/var_irf_bs.irf", clear

keep step irfname impulse response oirf oirf_lb oirf_ub
rename step horizon
rename oirf irf
rename oirf_lb lower
rename oirf_ub upper

keep if irfname == "var_irf_bs"
drop irfname

export delimited using "`output_dir'/irf_results_with_ci.csv", replace
display "IRF com CI salvo em: outputs/Stata/irf_results_with_ci.csv"
restore

* --- Resumo ------------------------------------------------------------------
display ""
display "--- Resumo ---"
display "IRF e FEVD calculados para VAR(1) com `n_ahead' periodos"
display "Decomposicao de Cholesky (ortogonalizada)"
display "Bootstrap CI: 500 replicacoes, 95% intervalo"
display ""
display "Arquivos gerados:"
display "  outputs/Stata/irf_results.csv"
display "  outputs/Stata/fevd_results.csv"
display "  outputs/Stata/irf_results_with_ci.csv"
display "  outputs/Stata/var_irf.irf"
display "  outputs/Stata/var_irf_bs.irf"

display ""
display "=== Fim da validacao IRF & FEVD (Stata) ==="
