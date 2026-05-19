* ==============================================================================
* compare_results.do
* Comparacao automatica de resultados: Python (chronobox) vs R (forecast) vs Stata
*
* Objetivo: Importar outputs das tres plataformas e calcular diferencas
*           absolutas e relativas para validacao cruzada triangular.
*
* Inputs:
*   - examples/arima/outputs/arima_coefficients.json  (Python - via CSV convertido)
*   - examples/arima/outputs/R/arima_coefficients_R.csv  (R)
*   - examples/arima/outputs/Stata/arima_nile_coefficients.csv  (Stata)
*   - examples/arima/outputs/Stata/arima_airline_coefficients.csv  (Stata)
*
* Output:
*   - examples/arima/outputs/Stata/comparison_python_r_stata.csv
*
* Nota: Stata nao le JSON nativamente. Os resultados Python devem estar
*       disponiveis em CSV (gerados pelos notebooks) ou este script compara
*       apenas R vs Stata.
*
* Dependencias: Stata 14+
* ==============================================================================

clear all
set more off
set seed 42

* --- Caminhos -----------------------------------------------------------------
local output_dir    "../outputs"
local r_dir         "../outputs/R"
local stata_dir     "../outputs/Stata"

display _newline(2)
display "================================================================"
display "  COMPARACAO DE RESULTADOS: Python vs R vs Stata"
display "================================================================"

* ==============================================================================
* SECAO 1: Carregar resultados R
* ==============================================================================

display _newline(2)
display "=== 1. Carregando resultados R ==="

* Verificar se os arquivos R existem
capture confirm file "`r_dir'/arima_coefficients_R.csv"
local r_coefs_exist = (_rc == 0)

if (`r_coefs_exist') {
    import delimited "`r_dir'/arima_coefficients_R.csv", clear varnames(1)
    display "Resultados R carregados: " _N " linhas"
    list dataset model param value aic bic in 1/10, separator(0)

    * Renomear para merge
    rename value value_r
    rename aic aic_r
    rename bic bic_r
    rename sigma2 sigma2_r
    rename loglik loglik_r

    tempfile r_results
    save `r_results'
}
else {
    display "AVISO: arima_coefficients_R.csv nao encontrado."
    display "       Execute os scripts R primeiro."
}

* ==============================================================================
* SECAO 2: Carregar resultados Stata
* ==============================================================================

display _newline(2)
display "=== 2. Carregando resultados Stata ==="

* Carregar Nile
capture confirm file "`stata_dir'/arima_nile_coefficients.csv"
local stata_nile_exist = (_rc == 0)

if (`stata_nile_exist') {
    import delimited "`stata_dir'/arima_nile_coefficients.csv", clear varnames(1)
    display "Resultados Stata (Nile): " _N " linhas"
    tempfile stata_nile
    save `stata_nile'
}

* Carregar Airline
capture confirm file "`stata_dir'/arima_airline_coefficients.csv"
local stata_airline_exist = (_rc == 0)

if (`stata_airline_exist') {
    import delimited "`stata_dir'/arima_airline_coefficients.csv", clear varnames(1)
    display "Resultados Stata (Airline): " _N " linhas"
    tempfile stata_airline
    save `stata_airline'
}

* Combinar Nile + Airline do Stata
if (`stata_nile_exist' & `stata_airline_exist') {
    use `stata_nile', clear
    append using `stata_airline'
    display "Resultados Stata combinados: " _N " linhas"

    rename value value_stata
    rename aic aic_stata
    rename bic bic_stata

    tempfile stata_results
    save `stata_results'
}
else if (`stata_nile_exist') {
    use `stata_nile', clear
    rename value value_stata
    rename aic aic_stata
    rename bic bic_stata
    tempfile stata_results
    save `stata_results'
}
else if (`stata_airline_exist') {
    use `stata_airline', clear
    rename value value_stata
    rename aic aic_stata
    rename bic bic_stata
    tempfile stata_results
    save `stata_results'
}
else {
    display "AVISO: Resultados Stata nao encontrados."
    display "       Execute 01_arima_validation.do primeiro."
}

* ==============================================================================
* SECAO 3: Merge R vs Stata
* ==============================================================================

display _newline(2)
display "=== 3. Comparacao R vs Stata ==="

if (`r_coefs_exist' & (`stata_nile_exist' | `stata_airline_exist')) {
    * Merge por dataset + model + param
    use `stata_results', clear
    merge 1:1 dataset model param using `r_results'

    * Mostrar resultado do merge
    tab _merge
    display _newline(1)

    * Manter apenas matches
    keep if _merge == 3
    drop _merge

    * Calcular diferencas
    generate double abs_diff = abs(value_stata - value_r)
    generate double rel_diff = abs_diff / abs(value_r) if abs(value_r) > 1e-10
    generate double aic_diff = abs(aic_stata - aic_r) if aic_r != . & aic_stata != .

    * Mostrar comparacao
    display _newline(1)
    display "Comparacao de coeficientes (R vs Stata):"
    display _newline(1)
    list dataset model param value_r value_stata abs_diff rel_diff, separator(0)

    * Resumo
    display _newline(1)
    summarize abs_diff, detail
    local max_diff = r(max)
    local mean_diff = r(mean)

    display _newline(1)
    display "Maxima diferenca absoluta em coeficientes: " %12.6e `max_diff'
    display "Media das diferencas absolutas:            " %12.6e `mean_diff'
    display "Dentro da tolerancia 1e-3?                 " cond(`max_diff' < 1e-3, "SIM", "NAO (verificar)")

    * Comparacao de AIC
    display _newline(1)
    display "Comparacao de AIC:"
    list dataset model aic_r aic_stata aic_diff if aic_diff != ., separator(0)

    * Exportar comparacao
    export delimited using "`stata_dir'/comparison_r_vs_stata.csv", replace
    display _newline(1)
    display "Salvo: comparison_r_vs_stata.csv"
}
else {
    display "Nao e possivel comparar: faltam resultados R e/ou Stata."
    display "Execute os scripts correspondentes primeiro."
}

* ==============================================================================
* SECAO 4: Carregar resultados Python (se CSV disponivel)
* ==============================================================================

display _newline(2)
display "=== 4. Resultados Python ==="

* Nota: Os resultados Python sao salvos em JSON pelo chronobox.
* Como Stata nao le JSON nativamente, verificamos se existe um CSV equivalente.
* Os notebooks Python devem exportar arima_coefficients.csv para compatibilidade.

capture confirm file "`output_dir'/arima_forecasts.csv"
local py_fc_exist = (_rc == 0)

if (`py_fc_exist') {
    display "Previsoes Python encontradas."

    * Comparar previsoes se Stata tambem tiver
    capture confirm file "`stata_dir'/arima_forecasts_airline.csv"
    local stata_fc_exist = (_rc == 0)

    if (`stata_fc_exist') {
        display "Comparando previsoes Python vs Stata para Airline..."

        * Carregar previsoes Python (filtrar airline ARIMA(1,1,1))
        import delimited "`output_dir'/arima_forecasts.csv", clear varnames(1)
        keep if dataset == "airline" & model == "ARIMA(1,1,1)"
        rename forecast forecast_py
        keep step forecast_py
        tempfile py_fc
        save `py_fc'

        * Carregar previsoes Stata
        import delimited "`stata_dir'/arima_forecasts_airline.csv", clear varnames(1)
        rename forecast forecast_stata
        keep step forecast_stata

        * Merge
        merge 1:1 step using `py_fc'
        keep if _merge == 3
        drop _merge

        generate double fc_diff = abs(forecast_py - forecast_stata)

        display _newline(1)
        display "Comparacao de previsoes Airline ARIMA(1,1,1):"
        list step forecast_py forecast_stata fc_diff, separator(0)

        summarize fc_diff
        display "Maxima diferenca em previsoes: " %12.6e r(max)
    }
}
else {
    display "Previsoes Python nao encontradas em formato CSV."
    display "Nota: Os resultados Python sao armazenados em JSON."
    display "      Para comparacao completa, converta para CSV ou use Python/R."
}

* ==============================================================================
* SECAO 5: Comparacao SARIMA (se disponivel)
* ==============================================================================

display _newline(2)
display "=== 5. Comparacao SARIMA ==="

capture confirm file "`r_dir'/sarima_results_R.csv"
local r_sarima_exist = (_rc == 0)

capture confirm file "`stata_dir'/sarima_airline_results.csv"
local stata_sarima_exist = (_rc == 0)

if (`r_sarima_exist' & `stata_sarima_exist') {
    * Carregar R SARIMA
    import delimited "`r_dir'/sarima_results_R.csv", clear varnames(1)
    keep if dataset == "airline"
    rename value value_r
    rename aic aic_r
    keep dataset model param value_r aic_r
    tempfile r_sarima
    save `r_sarima'

    * Carregar Stata SARIMA
    import delimited "`stata_dir'/sarima_airline_results.csv", clear varnames(1)
    rename value value_stata
    rename aic aic_stata
    keep dataset model param value_stata aic_stata

    * Merge
    merge 1:1 dataset model param using `r_sarima'

    display _newline(1)
    display "Comparacao SARIMA Airline (R vs Stata):"
    keep if _merge == 3
    drop _merge

    generate double abs_diff = abs(value_stata - value_r)
    list dataset model param value_r value_stata abs_diff, separator(0)

    display _newline(1)
    display "Diferencas de AIC nos modelos SARIMA:"
    * Mostrar AIC por modelo (uma vez por modelo)
    bysort model: generate first = (_n == 1)
    list model aic_r aic_stata if first, separator(0)
    drop first
}
else {
    display "Resultados SARIMA incompletos para comparacao."
    if (!`r_sarima_exist') display "  - Falta: sarima_results_R.csv"
    if (!`stata_sarima_exist') display "  - Falta: sarima_airline_results.csv"
}

* ==============================================================================
* SECAO 6: Sumario geral
* ==============================================================================

display _newline(2)
display "================================================================"
display "  SUMARIO DA VALIDACAO CRUZADA TRIANGULAR"
display "================================================================"
display _newline(1)

display "Plataformas comparadas: Python (chronobox) / R (forecast) / Stata"
display _newline(1)
display "Notas:"
display "  1. Diferencas pequenas (< 1e-4 em coeficientes) sao esperadas"
display "     devido a diferencas de implementacao numerica."
display "  2. Diferencas maiores podem ocorrer quando:"
display "     - Metodos de otimizacao diferem (CSS, CSS-ML, ML)"
display "     - Parametrizacao difere (constante incluida/excluida)"
display "     - Convergencia para diferentes maximos locais"
display "  3. AIC pode diferir por constantes aditivas entre plataformas."
display "     Comparacoes de AIC sao validas DENTRO de cada plataforma."
display "  4. Para ARFIMA, comparar apenas metodos equivalentes:"
display "     - ML vs ML (Stata arfima vs R fracdiff)"
display "     - GPH vs GPH (Python vs R fdGPH)"
display _newline(1)
display "Tolerancias recomendadas:"
display "  - Coeficientes AR/MA: < 1e-3 (aceitavel)"
display "  - AIC/BIC: diferencas absolutas < 2.0 (aceitavel)"
display "  - Previsoes: < 1e-2 (aceitavel para series macro)"

display _newline(2)
display "=== Script compare_results.do concluido com sucesso ==="
