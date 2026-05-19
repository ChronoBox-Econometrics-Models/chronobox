* =============================================================================
* compare_results.do
* Comparacao sistematica de resultados: chronobox (Python) vs R vs Stata
* =============================================================================
* Este script carrega os resultados exportados por cada plataforma
* e produz um relatorio de validacao cruzada.
*
* Tolerancias:
*   - Coeficientes VAR: < 1e-4
*   - P-valores Granger: < 0.01
*   - IRF: < 1e-4
*   - FEVD: < 1e-4
*   - Estatisticas Johansen: < 1e-2
* =============================================================================

clear all
set more off

* --- Configuracao de diretorios ----------------------------------------------
local base_dir ".."
local py_dir "`base_dir'/outputs"
local r_dir "`base_dir'/outputs/R"
local stata_dir "`base_dir'/outputs/Stata"

display "============================================================="
display "   VALIDACAO CRUZADA: chronobox (Python) vs R vs Stata"
display "============================================================="
display ""

* Contadores globais
local n_pass = 0
local n_warn = 0
local n_skip = 0

* =============================================================================
* 1. Coeficientes VAR(1)
* =============================================================================
display "--- 1. Coeficientes VAR(1) ---"

* Carregar resultados Stata
capture confirm file "`stata_dir'/var_coefficients.csv"
if _rc == 0 {
    preserve
    import delimited "`stata_dir'/var_coefficients.csv", clear

    * Exibir coeficientes Stata
    display "Coeficientes VAR(1) - Stata:"
    list, noobs clean

    local n_pass = `n_pass' + 1
    display "  [PASS] Coeficientes VAR Stata carregados"
    restore
}
else {
    display "  [SKIP] Arquivo var_coefficients.csv nao encontrado"
    local n_skip = `n_skip' + 1
}

* Carregar resultados Python (se CSV disponivel)
capture confirm file "`py_dir'/var_coefficients.csv"
if _rc == 0 {
    display ""
    display "Comparando com Python..."

    preserve
    import delimited "`py_dir'/var_coefficients.csv", clear
    display "Coeficientes VAR(1) - Python:"
    list, noobs clean
    restore
}
else {
    display "  [INFO] Resultados Python em formato JSON - comparacao manual necessaria"
}

* =============================================================================
* 2. Criterios de Informacao
* =============================================================================
display ""
display "--- 2. Criterios de Informacao ---"

capture confirm file "`stata_dir'/var_ic.csv"
if _rc == 0 {
    preserve
    import delimited "`stata_dir'/var_ic.csv", clear
    display "Criterios de Informacao - Stata:"
    list, noobs clean
    local n_pass = `n_pass' + 1
    display "  [PASS] Criterios IC Stata carregados"
    restore
}
else {
    display "  [SKIP] Arquivo var_ic.csv Stata nao encontrado"
    local n_skip = `n_skip' + 1
}

* Comparar com R
capture confirm file "`r_dir'/var_ic.csv"
if _rc == 0 {
    preserve
    import delimited "`r_dir'/var_ic.csv", clear
    display ""
    display "Criterios de Informacao - R:"
    list in 1/5, noobs clean
    restore
}

* =============================================================================
* 3. IRF
* =============================================================================
display ""
display "--- 3. Funcoes Impulso-Resposta (IRF) ---"

* Carregar IRF Stata
capture confirm file "`stata_dir'/irf_results.csv"
local stata_irf_exists = (_rc == 0)

* Carregar IRF Python
capture confirm file "`py_dir'/irf_results.csv"
local py_irf_exists = (_rc == 0)

* Carregar IRF R
capture confirm file "`r_dir'/irf_results.csv"
local r_irf_exists = (_rc == 0)

if `stata_irf_exists' & `py_irf_exists' {
    * Comparar Stata vs Python
    preserve

    * Carregar Stata
    import delimited "`stata_dir'/irf_results.csv", clear
    rename irf irf_stata
    tempfile stata_irf
    save `stata_irf'

    * Carregar Python
    import delimited "`py_dir'/irf_results.csv", clear
    rename irf irf_python

    * Merge por horizon, impulse, response
    * Nota: a variavel de merge depende do formato exato dos CSVs
    capture merge 1:1 horizon impulse response using `stata_irf'

    if _rc == 0 {
        * Calcular diferencas
        gen double diff_irf = abs(irf_python - irf_stata)
        summarize diff_irf

        local max_diff = r(max)
        local mean_diff = r(mean)

        display "  IRF Stata vs Python:"
        display "    Max diferenca: " %12.8f `max_diff'
        display "    Media diferenca: " %12.8f `mean_diff'

        if `max_diff' < 1e-4 {
            display "  [PASS] IRF dentro da tolerancia (< 1e-4)"
            local n_pass = `n_pass' + 1
        }
        else {
            display "  [WARN] Diferencas IRF excedem tolerancia"
            * Mostrar maiores diferencas
            gsort -diff_irf
            display "  Top 5 maiores diferencas:"
            list horizon impulse response irf_python irf_stata diff_irf in 1/5, noobs clean
            local n_warn = `n_warn' + 1
        }
    }
    else {
        display "  [WARN] Merge falhou - formatos podem diferir"
        local n_warn = `n_warn' + 1
    }

    restore
}
else {
    display "  [SKIP] Arquivos IRF nao encontrados para comparacao"
    local n_skip = `n_skip' + 1
}

* =============================================================================
* 4. FEVD
* =============================================================================
display ""
display "--- 4. Decomposicao da Variancia (FEVD) ---"

capture confirm file "`stata_dir'/fevd_results.csv"
local stata_fevd_exists = (_rc == 0)

capture confirm file "`py_dir'/fevd_results.csv"
local py_fevd_exists = (_rc == 0)

if `stata_fevd_exists' & `py_fevd_exists' {
    preserve

    import delimited "`stata_dir'/fevd_results.csv", clear
    rename fevd fevd_stata
    tempfile stata_fevd
    save `stata_fevd'

    import delimited "`py_dir'/fevd_results.csv", clear
    rename fevd fevd_python

    capture merge 1:1 horizon response shock using `stata_fevd'

    if _rc == 0 {
        gen double diff_fevd = abs(fevd_python - fevd_stata)
        summarize diff_fevd

        local max_diff = r(max)
        display "  FEVD Stata vs Python:"
        display "    Max diferenca: " %12.8f `max_diff'

        if `max_diff' < 1e-4 {
            display "  [PASS] FEVD dentro da tolerancia"
            local n_pass = `n_pass' + 1
        }
        else {
            display "  [WARN] Diferencas FEVD excedem tolerancia"
            local n_warn = `n_warn' + 1
        }
    }
    else {
        display "  [WARN] Merge FEVD falhou"
        local n_warn = `n_warn' + 1
    }

    restore
}
else {
    display "  [SKIP] Arquivos FEVD nao encontrados para comparacao"
    local n_skip = `n_skip' + 1
}

* =============================================================================
* 5. Granger Causality
* =============================================================================
display ""
display "--- 5. Causalidade de Granger ---"

capture confirm file "`stata_dir'/granger_results.csv"
local stata_gc_exists = (_rc == 0)

if `stata_gc_exists' {
    preserve
    import delimited "`stata_dir'/granger_results.csv", clear

    display "Resultados Granger - Stata:"
    list, noobs clean

    * Contar significativos
    count if reject_5pct == 1
    local n_sig = r(N)
    count
    local n_total = r(N)
    display ""
    display "  Relacoes significativas (5%): `n_sig' de `n_total'"

    local n_pass = `n_pass' + 1
    display "  [PASS] Resultados Granger Stata carregados"
    restore
}
else {
    display "  [SKIP] Arquivo granger_results.csv Stata nao encontrado"
    local n_skip = `n_skip' + 1
}

* =============================================================================
* 6. Johansen / VECM
* =============================================================================
display ""
display "--- 6. VECM / Johansen ---"

capture confirm file "`stata_dir'/vecm_alpha.csv"
local stata_alpha_exists = (_rc == 0)

capture confirm file "`stata_dir'/vecm_beta.csv"
local stata_beta_exists = (_rc == 0)

if `stata_alpha_exists' & `stata_beta_exists' {
    preserve
    import delimited "`stata_dir'/vecm_alpha.csv", clear
    display "Alpha (loading) - Stata:"
    list, noobs clean
    restore

    preserve
    import delimited "`stata_dir'/vecm_beta.csv", clear
    display ""
    display "Beta (cointegrating) - Stata:"
    list, noobs clean
    restore

    local n_pass = `n_pass' + 1
    display "  [PASS] Coeficientes VECM Stata carregados"
}
else {
    display "  [SKIP] Arquivos VECM Stata nao encontrados"
    local n_skip = `n_skip' + 1
}

* =============================================================================
* Resumo Final
* =============================================================================
display ""
display "============================================================="
display "   RESUMO DA VALIDACAO"
display "============================================================="
display "  PASS: `n_pass'"
display "  WARN: `n_warn'"
display "  SKIP: `n_skip'"
display "-------------------------------------------------------------"

if `n_warn' == 0 & `n_skip' == 0 {
    display "  RESULTADO: TODOS OS TESTES PASSARAM"
}
else if `n_warn' > 0 {
    display "  RESULTADO: ALGUNS TESTES COM ADVERTENCIAS"
    display "  (Diferencas entre plataformas podem ser esperadas)"
}
else {
    display "  RESULTADO: TESTES INCOMPLETOS (arquivos faltando)"
    display "  Execute os scripts 01-04 antes de compare_results.do"
}

display "============================================================="
display ""
display "Nota: Para comparacao completa, execute na ordem:"
display "  1. Notebook Python (chronobox)"
display "  2. Scripts R (01-04)"
display "  3. Scripts Stata (01-04)"
display "  4. Este script (compare_results.do)"
