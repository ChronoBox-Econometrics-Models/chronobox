* =============================================================================
* compare_results.do
* Compara resultados de suavizacao exponencial entre Python, R e Stata
*
* Carrega CSVs gerados pelos tres ambientes e produz tabela comparativa.
* Para funcionar, execute primeiro:
*   - Notebooks Python (chronobox): examples/ets/solutions/
*   - Scripts R (forecast): examples/ets/R/
*   - Scripts Stata: 01_ets_validation.do, 02_hw_validation.do,
*                    03_auto_ets_validation.do
*
* NOTA SOBRE COMPARABILIDADE:
*   - Python (chronobox) e R (forecast) implementam o framework ETS completo
*     com modelos aditivos e multiplicativos, damped trend, e selecao
*     automatica via AICc.
*   - Stata (tssmooth) implementa apenas suavizacao classica (SES, Holt,
*     HW aditivo). Comparacoes diretas sao possiveis apenas para estes
*     metodos basicos.
* =============================================================================

clear all
set more off

* --- Configuracao ---
local base_dir ".."
local output_dir "`base_dir'/outputs"
local stata_dir "`output_dir'/Stata"
local r_dir "`output_dir'/R"
local python_dir "`output_dir'/Python"

display _newline
display "========================================================"
display " Comparacao Cruzada: Python (chronobox) vs R vs Stata"
display "========================================================"
display _newline

* =============================================================================
* 1. Verificar disponibilidade de resultados
* =============================================================================

display "--- Verificando resultados disponiveis ---"
display _newline

local stata_ok = 0
local r_ok = 0
local python_ok = 0

* Stata
capture confirm file "`stata_dir'/ets_basic_results.csv"
if _rc == 0 {
    local stata_ok = 1
    display "  [OK] Resultados Stata encontrados"
}
else {
    display "  [FALTA] Resultados Stata - execute 01_ets_validation.do primeiro"
}

* R
capture confirm file "`r_dir'/ets_coefficients.csv"
if _rc == 0 {
    local r_ok = 1
    display "  [OK] Resultados R encontrados"
}
else {
    display "  [FALTA] Resultados R - execute os scripts R primeiro"
}

* Python
capture confirm file "`python_dir'/ets_coefficients.csv"
if _rc == 0 {
    local python_ok = 1
    display "  [OK] Resultados Python encontrados"
}
else {
    display "  [FALTA] Resultados Python - execute os notebooks primeiro"
}

display _newline

* =============================================================================
* 2. Carregar resultados Stata
* =============================================================================

if `stata_ok' {
    display "=== Resultados Stata ==="
    display _newline

    import delimited "`stata_dir'/ets_basic_results.csv", clear
    display "  Modelos basicos (ETS):"
    display _col(5) "Modelo" _col(25) "Alpha" _col(35) "Beta" ///
        _col(45) "RMSE" _col(58) "MAE" _col(70) "MAPE"
    display _col(5) strrep("-", 70)
    forvalues i = 1/`=_N' {
        display _col(5) model[`i'] ///
            _col(25) %6.4f alpha[`i'] ///
            _col(35) %6.4f beta[`i'] ///
            _col(45) %10.4f rmse[`i'] ///
            _col(58) %10.4f mae[`i'] ///
            _col(70) %6.2f mape[`i'] "%"
    }
    display _newline

    * HW results
    capture confirm file "`stata_dir'/hw_results.csv"
    if _rc == 0 {
        import delimited "`stata_dir'/hw_results.csv", clear
        display "  Holt-Winters:"
        display _col(5) "Metodo" _col(30) "Alpha" _col(40) "Beta" ///
            _col(50) "Gamma" _col(60) "RMSE"
        display _col(5) strrep("-", 65)
        forvalues i = 1/`=_N' {
            display _col(5) method[`i'] ///
                _col(30) %6.4f alpha[`i'] ///
                _col(40) %6.4f beta[`i'] ///
                _col(50) %6.4f gamma[`i'] ///
                _col(60) %10.4f rmse[`i']
        }
        display _newline
    }

    * Grid search summary
    capture confirm file "`stata_dir'/grid_search_summary.csv"
    if _rc == 0 {
        import delimited "`stata_dir'/grid_search_summary.csv", clear
        display "  Grid Search - Melhores modelos:"
        display _col(5) "Metodo" _col(25) "Parametros" _col(60) "MSE" _col(75) "RMSE"
        display _col(5) strrep("-", 75)
        forvalues i = 1/`=_N' {
            display _col(5) method[`i'] ///
                _col(25) best_params[`i'] ///
                _col(60) %10.4f best_mse[`i'] ///
                _col(75) %10.4f best_rmse[`i']
        }
        display _newline
    }
}

* =============================================================================
* 3. Carregar resultados R
* =============================================================================

if `r_ok' {
    display "=== Resultados R (forecast) ==="
    display _newline

    import delimited "`r_dir'/ets_coefficients.csv", clear
    display "  Coeficientes ETS:"
    display _col(5) "Modelo" _col(18) "Alpha" _col(28) "Beta" ///
        _col(38) "Gamma" _col(48) "Phi" _col(58) "AICc"
    display _col(5) strrep("-", 60)
    forvalues i = 1/`=_N' {
        display _col(5) model[`i'] ///
            _col(18) %6.4f alpha[`i'] ///
            _col(28) %6.4f beta[`i'] ///
            _col(38) %6.4f gamma[`i'] ///
            _col(48) %6.4f phi[`i'] ///
            _col(58) %10.2f aicc[`i']
    }
    display _newline

    * Metricas R
    capture confirm file "`r_dir'/ets_metrics.csv"
    if _rc == 0 {
        import delimited "`r_dir'/ets_metrics.csv", clear
        display "  Metricas de erro (R):"
        display _col(5) "Modelo" _col(18) "RMSE" _col(30) "MAE" _col(42) "MAPE"
        display _col(5) strrep("-", 45)
        forvalues i = 1/`=_N' {
            display _col(5) model[`i'] ///
                _col(18) %10.4f rmse[`i'] ///
                _col(30) %10.4f mae[`i'] ///
                _col(42) %6.2f mape[`i'] "%"
        }
        display _newline
    }
}

* =============================================================================
* 4. Carregar resultados Python
* =============================================================================

if `python_ok' {
    display "=== Resultados Python (chronobox) ==="
    display _newline

    import delimited "`python_dir'/ets_coefficients.csv", clear
    display "  Coeficientes ETS:"
    display _col(5) "Modelo" _col(18) "Alpha" _col(28) "Beta" ///
        _col(38) "Gamma" _col(48) "Phi"
    display _col(5) strrep("-", 50)
    forvalues i = 1/`=_N' {
        display _col(5) model[`i'] ///
            _col(18) %6.4f alpha[`i'] ///
            _col(28) %6.4f beta[`i'] ///
            _col(38) %6.4f gamma[`i'] ///
            _col(48) %6.4f phi[`i']
    }
    display _newline
}

* =============================================================================
* 5. Comparacao direta (quando disponivel)
* =============================================================================

display "========================================================"
display " Tabela Comparativa"
display "========================================================"
display _newline

* Comparar SES - todos os tres devem ter alpha para SES
display "--- Suavizacao Exponencial Simples (SES / ETS(A,N,N)) ---"
display _newline
display _col(5) "Parametro" _col(20) "Python" _col(35) "R" _col(50) "Stata"
display _col(5) strrep("-", 55)

if `python_ok' {
    import delimited "`python_dir'/ets_coefficients.csv", clear
    * Procurar modelo ANN ou SES
    quietly count if model == "ANN" | model == "SES"
    if r(N) > 0 {
        quietly levelsof alpha if model == "ANN" | model == "SES", local(py_alpha)
    }
}

if `r_ok' {
    import delimited "`r_dir'/ets_coefficients.csv", clear
    quietly count if model == "ANN" | model == "SES"
    if r(N) > 0 {
        quietly levelsof alpha if model == "ANN" | model == "SES", local(r_alpha)
    }
}

if `stata_ok' {
    import delimited "`stata_dir'/ets_basic_results.csv", clear
    quietly count if model == "SES_opt"
    if r(N) > 0 {
        quietly levelsof alpha if model == "SES_opt", local(stata_alpha)
    }
}

display _col(5) "Alpha" ///
    _col(20) "`py_alpha'" ///
    _col(35) "`r_alpha'" ///
    _col(50) "`stata_alpha'"

display _newline

* =============================================================================
* 6. Nota sobre diferencas esperadas
* =============================================================================

display "========================================================"
display " Notas sobre diferencas esperadas"
display "========================================================"
display _newline
display "  1. OTIMIZACAO: Cada ferramenta usa otimizador diferente."
display "     Python (scipy.optimize), R (optim/nlminb), Stata (proprio)."
display "     Parametros podem diferir em 1e-3 a 1e-2."
display _newline
display "  2. INICIALIZACAO: Valores iniciais de nivel, tendencia e"
display "     sazonalidade podem diferir entre implementacoes."
display _newline
display "  3. ESCOPO DE MODELOS:"
display "     - Python (chronobox): ETS completo (30 modelos), auto-selecao"
display "     - R (forecast): ETS completo (30 modelos), auto-selecao via AICc"
display "     - Stata (tssmooth): SES, Holt, HW aditivo apenas (3 metodos)"
display _newline
display "  4. MODELOS EXCLUSIVOS Python/R (nao disponiveis no Stata):"
display "     - Modelos multiplicativos: ETS(M,*,*)"
display "     - Damped trend: ETS(*,Ad,*)"
display "     - Multiplicativo sazonal: ETS(*,*,M)"
display "     - Theta method"
display _newline
display "  5. METRICAS: Stata usa MSE; Python/R reportam AICc."
display "     MSE e RMSE sao comparaveis diretamente."
display _newline

* =============================================================================
* 7. Exportar tabela resumo
* =============================================================================

clear
set obs 6
generate str15 method = ""
generate str10 python_available = ""
generate str10 r_available = ""
generate str10 stata_available = ""

replace method = "SES" in 1
replace python_available = "Sim" in 1
replace r_available = "Sim" in 1
replace stata_available = "Sim" in 1

replace method = "Holt" in 2
replace python_available = "Sim" in 2
replace r_available = "Sim" in 2
replace stata_available = "Sim" in 2

replace method = "HW_aditivo" in 3
replace python_available = "Sim" in 3
replace r_available = "Sim" in 3
replace stata_available = "Sim" in 3

replace method = "HW_multiplic" in 4
replace python_available = "Sim" in 4
replace r_available = "Sim" in 4
replace stata_available = "Nao" in 4

replace method = "Damped_trend" in 5
replace python_available = "Sim" in 5
replace r_available = "Sim" in 5
replace stata_available = "Nao" in 5

replace method = "Auto_ETS" in 6
replace python_available = "Sim" in 6
replace r_available = "Sim" in 6
replace stata_available = "Nao" in 6

display "  Disponibilidade de metodos por plataforma:"
list, noobs separator(0)

export delimited "`output_dir'/Stata/comparison_availability.csv", replace
display _newline
display "Salvo: comparison_availability.csv"

display _newline
display "========================================================"
display " compare_results.do concluido!"
display "========================================================"
display _newline
display "Para comparacao completa, execute os scripts na seguinte ordem:"
display "  1. Notebooks Python (chronobox) -> outputs/Python/"
display "  2. Scripts R (forecast) -> outputs/R/"
display "  3. Scripts Stata (01, 02, 03) -> outputs/Stata/"
display "  4. Este script (compare_results.do)"
