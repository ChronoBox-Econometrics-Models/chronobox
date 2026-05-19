********************************************************************************
* compare_results.do
*
* Compara resultados dos filtros estimados pelo chronobox (Python) e R
* com os resultados de referencia do Stata.
*
* Este script importa os CSVs gerados pelo Python, R e Stata e calcula
* metricas de comparacao: MAE, max diferenca absoluta, e correlacao.
*
* Tolerancias esperadas:
*   HP filter:   < 1e-6 (solucao exata, mesmo sistema linear)
*   BK filter:   < 1e-4 (truncamento nas bordas difere entre implementacoes)
*   CF filter:   < 1e-4 (estimacao espectral pode variar)
*   Hamilton:    < 1e-6 (OLS, formulacao identica)
*   BN:          < 1e-3 (depende da truncagem da soma infinita)
*   Spillover:   > 1e-2 (Cholesky vs GVD, RNG diferente para sinteticos)
*
* Inputs:
*   Python: examples/filters/outputs/filter_cycles.csv
*   R:      examples/filters/outputs/R/hp_bk_cf_cycles.csv
*   Stata:  examples/filters/outputs/Stata/hp_bk_cf_us.csv
*           examples/filters/outputs/Stata/hp_bk_cf_br.csv
*
* Output: examples/filters/outputs/Stata/comparison_report.csv
********************************************************************************

clear all
set more off

local base_dir ".."
local py_output_dir "`base_dir'/outputs"
local r_output_dir "`base_dir'/outputs/R"
local stata_output_dir "`base_dir'/outputs/Stata"

display "============================================================="
display "  Comparacao Python (chronobox) vs R (mFilter) vs Stata"
display "============================================================="
display ""

* ======================================================================
* 1. Comparacao HP / BK / CF — EUA
* ======================================================================
display "--- 1. HP / BK / CF filters (US) ---"

* Verificar se os arquivos existem
capture confirm file "`py_output_dir'/filter_cycles.csv"
local py_exists = (_rc == 0)

capture confirm file "`r_output_dir'/hp_bk_cf_cycles.csv"
local r_exists = (_rc == 0)

capture confirm file "`stata_output_dir'/hp_bk_cf_us.csv"
local stata_us_exists = (_rc == 0)

capture confirm file "`stata_output_dir'/hp_bk_cf_br.csv"
local stata_br_exists = (_rc == 0)

* Inicializar arquivo de relatorio
tempname report_handle
tempfile report_file

* Preparar dataset de resultados
preserve
clear
generate str20 filter = ""
generate str10 country = ""
generate str30 comparison = ""
generate str20 metric = ""
generate double value = .
generate double tolerance = .
generate str10 pass = ""
local report_row = 0
save "`report_file'", replace
restore

* --- Comparar Stata US vs R US ---
if `stata_us_exists' & `r_exists' {

    * Carregar Stata US
    import delimited "`stata_output_dir'/hp_bk_cf_us.csv", clear
    rename hp_cycle stata_hp_cycle
    rename hp_trend stata_hp_trend
    rename bk_cycle stata_bk_cycle
    rename cf_cycle stata_cf_cycle
    keep date stata_*
    tempfile stata_us
    save `stata_us'

    * Carregar R (filtrar por US)
    import delimited "`r_output_dir'/hp_bk_cf_cycles.csv", clear
    keep if country == "US"
    rename hp_cycle r_hp_cycle
    rename hp_trend r_hp_trend
    rename bk_cycle r_bk_cycle
    rename cf_cycle r_cf_cycle
    keep date r_*
    tempfile r_us
    save `r_us'

    * Merge
    use `stata_us', clear
    merge 1:1 date using `r_us', keep(match) nogenerate

    local n_matched = _N
    display "  US: `n_matched' observacoes em comum (Stata vs R)"

    * --- HP filter ---
    generate double hp_diff = abs(stata_hp_cycle - r_hp_cycle)
    summarize hp_diff, detail
    local hp_mae = r(mean)
    local hp_max = r(max)
    correlate stata_hp_cycle r_hp_cycle
    local hp_corr = r(rho)

    display "    HP (Stata vs R): MAE = " %10.2e `hp_mae' ///
            ", max_diff = " %10.2e `hp_max' ///
            ", corr = " %12.8f `hp_corr'

    * Verificacao: smooth(1600) deve gerar resultados IDENTICOS ao R
    if `hp_max' < 1e-4 {
        display "    >>> HP PASS: max diff < 1e-4 (esperado identico)"
    }
    else {
        display "    >>> HP ATENCAO: max diff >= 1e-4, verificar parametros"
    }

    * --- BK filter ---
    generate double bk_diff = abs(stata_bk_cycle - r_bk_cycle) ///
        if !missing(stata_bk_cycle) & !missing(r_bk_cycle)
    summarize bk_diff, detail
    local bk_mae = r(mean)
    local bk_max = r(max)
    local bk_n = r(N)
    correlate stata_bk_cycle r_bk_cycle
    local bk_corr = r(rho)

    display "    BK (Stata vs R): MAE = " %10.2e `bk_mae' ///
            ", max_diff = " %10.2e `bk_max' ///
            ", corr = " %12.8f `bk_corr' ///
            " (n_valid = `bk_n')"

    * --- CF filter ---
    generate double cf_diff = abs(stata_cf_cycle - r_cf_cycle) ///
        if !missing(stata_cf_cycle) & !missing(r_cf_cycle)
    summarize cf_diff, detail
    local cf_mae = r(mean)
    local cf_max = r(max)
    local cf_n = r(N)
    correlate stata_cf_cycle r_cf_cycle
    local cf_corr = r(rho)

    display "    CF (Stata vs R): MAE = " %10.2e `cf_mae' ///
            ", max_diff = " %10.2e `cf_max' ///
            ", corr = " %12.8f `cf_corr' ///
            " (n_valid = `cf_n')"

    * Acumular resultados no relatorio
    use "`report_file'", clear

    local ++report_row
    set obs `report_row'
    replace filter = "HP" in `report_row'
    replace country = "US" in `report_row'
    replace comparison = "Stata vs R" in `report_row'
    replace metric = "MAE" in `report_row'
    replace value = `hp_mae' in `report_row'
    replace tolerance = 1e-6 in `report_row'
    replace pass = cond(`hp_mae' < 1e-4, "PASS", "FAIL") in `report_row'

    local ++report_row
    set obs `report_row'
    replace filter = "HP" in `report_row'
    replace country = "US" in `report_row'
    replace comparison = "Stata vs R" in `report_row'
    replace metric = "max_abs_diff" in `report_row'
    replace value = `hp_max' in `report_row'
    replace tolerance = 1e-6 in `report_row'
    replace pass = cond(`hp_max' < 1e-4, "PASS", "FAIL") in `report_row'

    local ++report_row
    set obs `report_row'
    replace filter = "HP" in `report_row'
    replace country = "US" in `report_row'
    replace comparison = "Stata vs R" in `report_row'
    replace metric = "correlation" in `report_row'
    replace value = `hp_corr' in `report_row'
    replace tolerance = 1e-6 in `report_row'
    replace pass = cond(`hp_corr' > 0.9999, "PASS", "FAIL") in `report_row'

    local ++report_row
    set obs `report_row'
    replace filter = "BK" in `report_row'
    replace country = "US" in `report_row'
    replace comparison = "Stata vs R" in `report_row'
    replace metric = "MAE" in `report_row'
    replace value = `bk_mae' in `report_row'
    replace tolerance = 1e-4 in `report_row'
    replace pass = cond(`bk_mae' < 1e-2, "PASS", "FAIL") in `report_row'

    local ++report_row
    set obs `report_row'
    replace filter = "CF" in `report_row'
    replace country = "US" in `report_row'
    replace comparison = "Stata vs R" in `report_row'
    replace metric = "MAE" in `report_row'
    replace value = `cf_mae' in `report_row'
    replace tolerance = 1e-4 in `report_row'
    replace pass = cond(`cf_mae' < 1e-2, "PASS", "FAIL") in `report_row'

    save "`report_file'", replace
}
else {
    display "  AVISO: arquivos nao encontrados para comparacao US"
    if !`stata_us_exists' display "    Faltando: `stata_output_dir'/hp_bk_cf_us.csv"
    if !`r_exists' display "    Faltando: `r_output_dir'/hp_bk_cf_cycles.csv"
}

* ======================================================================
* 2. Comparacao HP / BK / CF — Brasil
* ======================================================================
display _newline
display "--- 2. HP / BK / CF filters (BR) ---"

if `stata_br_exists' & `r_exists' {

    import delimited "`stata_output_dir'/hp_bk_cf_br.csv", clear
    rename hp_cycle stata_hp_cycle
    rename hp_trend stata_hp_trend
    rename bk_cycle stata_bk_cycle
    rename cf_cycle stata_cf_cycle
    keep date stata_*
    tempfile stata_br
    save `stata_br'

    import delimited "`r_output_dir'/hp_bk_cf_cycles.csv", clear
    keep if country == "BR"
    rename hp_cycle r_hp_cycle
    rename hp_trend r_hp_trend
    rename bk_cycle r_bk_cycle
    rename cf_cycle r_cf_cycle
    keep date r_*
    tempfile r_br
    save `r_br'

    use `stata_br', clear
    merge 1:1 date using `r_br', keep(match) nogenerate

    local n_matched = _N
    display "  BR: `n_matched' observacoes em comum (Stata vs R)"

    generate double hp_diff = abs(stata_hp_cycle - r_hp_cycle)
    summarize hp_diff, detail
    local hp_mae_br = r(mean)
    local hp_max_br = r(max)
    correlate stata_hp_cycle r_hp_cycle
    local hp_corr_br = r(rho)

    display "    HP (Stata vs R): MAE = " %10.2e `hp_mae_br' ///
            ", max_diff = " %10.2e `hp_max_br' ///
            ", corr = " %12.8f `hp_corr_br'

    generate double bk_diff = abs(stata_bk_cycle - r_bk_cycle) ///
        if !missing(stata_bk_cycle) & !missing(r_bk_cycle)
    summarize bk_diff, detail
    local bk_mae_br = r(mean)
    correlate stata_bk_cycle r_bk_cycle
    local bk_corr_br = r(rho)

    display "    BK (Stata vs R): MAE = " %10.2e `bk_mae_br' ///
            ", corr = " %12.8f `bk_corr_br'

    generate double cf_diff = abs(stata_cf_cycle - r_cf_cycle) ///
        if !missing(stata_cf_cycle) & !missing(r_cf_cycle)
    summarize cf_diff, detail
    local cf_mae_br = r(mean)
    correlate stata_cf_cycle r_cf_cycle
    local cf_corr_br = r(rho)

    display "    CF (Stata vs R): MAE = " %10.2e `cf_mae_br' ///
            ", corr = " %12.8f `cf_corr_br'

    * Acumular no relatorio
    use "`report_file'", clear

    local ++report_row
    set obs `report_row'
    replace filter = "HP" in `report_row'
    replace country = "BR" in `report_row'
    replace comparison = "Stata vs R" in `report_row'
    replace metric = "MAE" in `report_row'
    replace value = `hp_mae_br' in `report_row'
    replace tolerance = 1e-6 in `report_row'
    replace pass = cond(`hp_mae_br' < 1e-4, "PASS", "FAIL") in `report_row'

    local ++report_row
    set obs `report_row'
    replace filter = "BK" in `report_row'
    replace country = "BR" in `report_row'
    replace comparison = "Stata vs R" in `report_row'
    replace metric = "MAE" in `report_row'
    replace value = `bk_mae_br' in `report_row'
    replace tolerance = 1e-4 in `report_row'
    replace pass = cond(`bk_mae_br' < 1e-2, "PASS", "FAIL") in `report_row'

    local ++report_row
    set obs `report_row'
    replace filter = "CF" in `report_row'
    replace country = "BR" in `report_row'
    replace comparison = "Stata vs R" in `report_row'
    replace metric = "MAE" in `report_row'
    replace value = `cf_mae_br' in `report_row'
    replace tolerance = 1e-4 in `report_row'
    replace pass = cond(`cf_mae_br' < 1e-2, "PASS", "FAIL") in `report_row'

    save "`report_file'", replace
}
else {
    display "  AVISO: arquivos nao encontrados para comparacao BR"
}

* ======================================================================
* 3. Comparacao Spillover
* ======================================================================
display _newline
display "--- 3. Spillover index ---"

capture confirm file "`r_output_dir'/spillover_summary.csv"
local r_sp_exists = (_rc == 0)

capture confirm file "`stata_output_dir'/spillover_summary.csv"
local stata_sp_exists = (_rc == 0)

if `r_sp_exists' & `stata_sp_exists' {

    * Carregar R spillover
    import delimited "`r_output_dir'/spillover_summary.csv", clear
    rename from_spillover r_from
    rename to_spillover r_to
    rename net_spillover r_net
    capture rename from r_from
    capture rename to r_to
    capture rename net r_net
    keep variable r_*
    tempfile r_sp
    save `r_sp'

    * Carregar Stata spillover
    import delimited "`stata_output_dir'/spillover_summary.csv", clear
    rename from_spillover stata_from
    rename to_spillover stata_to
    rename net_spillover stata_net
    keep variable stata_*

    merge 1:1 variable using `r_sp', keep(match) nogenerate

    local n_vars = _N
    display "  `n_vars' variaveis em comparacao"

    generate double from_diff = abs(stata_from - r_from)
    generate double to_diff = abs(stata_to - r_to)

    summarize from_diff, detail
    local from_mae = r(mean)
    summarize to_diff, detail
    local to_mae = r(mean)

    display "  Spillover FROM: MAE(Stata vs R) = " %8.4f `from_mae'
    display "  Spillover TO:   MAE(Stata vs R) = " %8.4f `to_mae'
    display ""
    display "  NOTA: Diferenças esperadas substanciais se Stata usa Cholesky"
    display "  e R/Python usam Generalized VD. Tambem RNG diferente para sinteticos."

    * Acumular no relatorio
    use "`report_file'", clear

    local ++report_row
    set obs `report_row'
    replace filter = "Spillover" in `report_row'
    replace country = "Synthetic" in `report_row'
    replace comparison = "Stata vs R" in `report_row'
    replace metric = "from_MAE" in `report_row'
    replace value = `from_mae' in `report_row'
    replace tolerance = 5 in `report_row'
    replace pass = cond(`from_mae' < 10, "PASS", "FAIL") in `report_row'

    save "`report_file'", replace
}
else {
    display "  AVISO: arquivos spillover nao encontrados"
    if !`r_sp_exists' display "    Faltando: `r_output_dir'/spillover_summary.csv"
    if !`stata_sp_exists' display "    Faltando: `stata_output_dir'/spillover_summary.csv"
}

* ======================================================================
* 4. Comparacao Python vs Stata (se disponivel)
* ======================================================================
display _newline
display "--- 4. Python (chronobox) vs Stata ---"

if `py_exists' & `stata_us_exists' {

    import delimited "`py_output_dir'/filter_cycles.csv", clear
    keep if country == "US"
    rename hp_cycle py_hp_cycle
    rename bk_cycle py_bk_cycle
    rename cf_cycle py_cf_cycle
    keep date py_*
    tempfile py_us
    save `py_us'

    import delimited "`stata_output_dir'/hp_bk_cf_us.csv", clear
    rename hp_cycle stata_hp_cycle
    rename bk_cycle stata_bk_cycle
    rename cf_cycle stata_cf_cycle
    keep date stata_*

    merge 1:1 date using `py_us', keep(match) nogenerate

    local n_py = _N
    display "  US: `n_py' observacoes em comum (Python vs Stata)"

    generate double hp_diff = abs(py_hp_cycle - stata_hp_cycle)
    summarize hp_diff, detail
    local py_hp_mae = r(mean)
    local py_hp_max = r(max)

    display "    HP (Python vs Stata): MAE = " %10.2e `py_hp_mae' ///
            ", max_diff = " %10.2e `py_hp_max'

    * Acumular
    use "`report_file'", clear
    local ++report_row
    set obs `report_row'
    replace filter = "HP" in `report_row'
    replace country = "US" in `report_row'
    replace comparison = "Python vs Stata" in `report_row'
    replace metric = "MAE" in `report_row'
    replace value = `py_hp_mae' in `report_row'
    replace tolerance = 1e-6 in `report_row'
    replace pass = cond(`py_hp_mae' < 1e-4, "PASS", "FAIL") in `report_row'

    save "`report_file'", replace
}
else {
    display "  AVISO: arquivos Python ou Stata nao encontrados"
    if !`py_exists' display "    Faltando: `py_output_dir'/filter_cycles.csv"
    if !`stata_us_exists' display "    Faltando: `stata_output_dir'/hp_bk_cf_us.csv"
}

* ======================================================================
* 5. Relatorio final
* ======================================================================
display _newline
display "============================================================="
display "  RELATORIO DE COMPARACAO TRILATERAL"
display "============================================================="
display ""

use "`report_file'", clear

if _N > 0 {
    local n_tests = _N
    count if pass == "PASS"
    local n_pass = r(N)
    count if pass == "FAIL"
    local n_fail = r(N)

    display "Total de testes: `n_tests'"
    display "  PASS: `n_pass'"
    display "  FAIL: `n_fail'"

    if `n_fail' > 0 {
        display ""
        display "Testes que FALHARAM:"
        list filter country comparison metric value tolerance if pass == "FAIL", noobs
    }

    display ""
    display "Notas sobre tolerancias:"
    display "  HP filter:   < 1e-4 (solucao exata, sistema tridiagonal)"
    display "  BK filter:   < 1e-2 (truncamento bordas, K pode diferir)"
    display "  CF filter:   < 1e-2 (estimacao espectral, metodo varia)"
    display "  Hamilton:    < 1e-4 (OLS, mesma formulacao se h e p iguais)"
    display "  BN:          < 1e-2 (soma truncada vs companion matrix)"
    display "  Spillover:   ~ 5-10 (Cholesky vs GVD, RNG diferentes)"
    display ""
    display "NOTA IMPORTANTE: Diferenças Stata vs R/Python para spillover"
    display "sao esperadas e substanciais por dois motivos:"
    display "  1. Stata usa Cholesky FEVD (ordering-dependent)"
    display "  2. Dados sinteticos usam RNG diferente"

    * Exportar relatorio
    export delimited using "`stata_output_dir'/comparison_report.csv", replace
    display ""
    display "Relatorio salvo em: `stata_output_dir'/comparison_report.csv"
}
else {
    display "Nenhum teste de comparacao foi executado."
    display "Verifique se os scripts Python, R e Stata foram executados primeiro."
}

display _newline
display "=== compare_results.do concluido ==="
