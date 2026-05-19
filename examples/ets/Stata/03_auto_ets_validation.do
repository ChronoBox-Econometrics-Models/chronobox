* =============================================================================
* 03_auto_ets_validation.do
* Grid search manual de parametros de suavizacao exponencial
*
* IMPORTANTE: Stata NAO possui funcionalidade auto-ETS nativa.
*   - Nao ha equivalente ao ets() do R (pacote forecast) que seleciona
*     automaticamente o melhor modelo ETS via AICc.
*   - Nao ha equivalente ao auto_ets() do chronobox (Python).
*   - Este script implementa um grid search manual sobre alpha, beta
*     e gamma para encontrar os melhores parametros por MSE.
*
* LIMITACOES DO STATA PARA SELECAO AUTOMATICA:
*   1. tssmooth nao reporta AIC/AICc/BIC - usamos MSE como criterio.
*   2. Apenas modelos aditivos sao suportados (nao multiplicativos).
*   3. Nao ha damped trend disponivel.
*   4. O espaco de modelos e muito menor que o framework ETS completo
*      (30 modelos no ETS vs 3 metodos no tssmooth).
*   5. tssmooth otimiza parametros internamente, mas nao compara entre
*      metodos (SES vs Holt vs HW).
*
* Datasets: airline.csv
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao ---
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

capture mkdir "`output_dir'"

display _newline
display "============================================="
display " Auto ETS - Grid Search Manual"
display "============================================="
display _newline

* =============================================================================
* 1. Carregar dados
* =============================================================================

import delimited "`data_dir'/airline.csv", clear

generate month_id = _n
tsset month_id
rename passengers y

local n_obs = _N

* =============================================================================
* 2. Grid search para SES (apenas alpha)
* =============================================================================

display "=== Grid Search: SES (tssmooth exponential) ==="
display "Variando alpha de 0.05 a 0.95"
display _newline

* Armazenar resultados em arquivo temporario
tempfile ses_results
postfile ses_handle double(alpha mse rmse) using `ses_results'

* Tambem incluir alpha otimizado
tssmooth exponential ses_auto = y
local alpha_auto = r(alpha)
generate resid_auto = y - ses_auto
generate resid_sq_auto = resid_auto^2
quietly summarize resid_sq_auto
local mse_auto = r(mean)
post ses_handle (`alpha_auto') (`mse_auto') (sqrt(`mse_auto'))
drop ses_auto resid_auto resid_sq_auto

* Grid sobre alpha
forvalues a = 5(5)95 {
    local alpha = `a' / 100
    quietly tssmooth exponential ses_grid = y, parms(`alpha')
    quietly generate resid = y - ses_grid
    quietly generate resid_sq = resid^2
    quietly summarize resid_sq
    local mse = r(mean)
    post ses_handle (`alpha') (`mse') (sqrt(`mse'))
    drop ses_grid resid resid_sq
}

postclose ses_handle

* Carregar e mostrar resultados SES
preserve
use `ses_results', clear
sort mse
display "Top 5 valores de alpha (por MSE):"
display _col(5) "Alpha" _col(15) "MSE" _col(30) "RMSE"
display _col(5) "-----" _col(15) "----------" _col(30) "----------"
forvalues i = 1/5 {
    local a = alpha[`i']
    local m = mse[`i']
    local r = rmse[`i']
    display _col(5) %5.3f `a' _col(15) %10.4f `m' _col(30) %10.4f `r'
}
display _newline
display "Melhor alpha SES: " %5.3f alpha[1] " (MSE=" %10.4f mse[1] ")"
local best_alpha_ses = alpha[1]
local best_mse_ses = mse[1]
local best_rmse_ses = rmse[1]
export delimited "`output_dir'/grid_search_ses.csv", replace
restore

* =============================================================================
* 3. Grid search para Holt (alpha e beta)
* =============================================================================

display _newline
display "=== Grid Search: Holt (tssmooth dexponential) ==="
display "Variando alpha e beta de 0.05 a 0.95"
display _newline

tempfile holt_results
postfile holt_handle double(alpha beta mse rmse) using `holt_results'

* Alpha otimizado
quietly tssmooth dexponential holt_auto = y
local alpha_h_auto = r(alpha)
local beta_h_auto = r(beta)
quietly generate resid = y - holt_auto
quietly generate resid_sq = resid^2
quietly summarize resid_sq
local mse_h_auto = r(mean)
post holt_handle (`alpha_h_auto') (`beta_h_auto') (`mse_h_auto') (sqrt(`mse_h_auto'))
drop holt_auto resid resid_sq

* Grid sobre alpha e beta (mais esparso para viabilidade)
forvalues a = 5(10)95 {
    forvalues b = 5(10)95 {
        local alpha = `a' / 100
        local beta = `b' / 100
        capture quietly tssmooth dexponential holt_grid = y, parms(`alpha' `beta')
        if _rc == 0 {
            quietly generate resid = y - holt_grid
            quietly generate resid_sq = resid^2
            quietly summarize resid_sq
            local mse = r(mean)
            post holt_handle (`alpha') (`beta') (`mse') (sqrt(`mse'))
            drop holt_grid resid resid_sq
        }
        else {
            capture drop holt_grid
        }
    }
}

postclose holt_handle

* Carregar e mostrar resultados Holt
preserve
use `holt_results', clear
sort mse
display "Top 5 combinacoes alpha/beta (por MSE):"
display _col(5) "Alpha" _col(15) "Beta" _col(25) "MSE" _col(40) "RMSE"
display _col(5) "-----" _col(15) "-----" _col(25) "----------" _col(40) "----------"
forvalues i = 1/5 {
    local a = alpha[`i']
    local b = beta[`i']
    local m = mse[`i']
    local r = rmse[`i']
    display _col(5) %5.3f `a' _col(15) %5.3f `b' _col(25) %10.4f `m' _col(40) %10.4f `r'
}
display _newline
display "Melhor: alpha=" %5.3f alpha[1] " beta=" %5.3f beta[1] ///
    " (MSE=" %10.4f mse[1] ")"
local best_alpha_holt = alpha[1]
local best_beta_holt = beta[1]
local best_mse_holt = mse[1]
local best_rmse_holt = rmse[1]
export delimited "`output_dir'/grid_search_holt.csv", replace
restore

* =============================================================================
* 4. Grid search para HW sazonal (alpha, beta, gamma)
* =============================================================================

display _newline
display "=== Grid Search: HW sazonal (tssmooth shwinters) ==="
display "Variando alpha, beta, gamma (grid esparso)"
display _newline

tempfile hw_results
postfile hw_handle double(alpha beta gamma mse rmse) using `hw_results'

* Otimizado
quietly tssmooth shwinters hw_auto = y, period(12)
local alpha_s_auto = r(alpha)
local beta_s_auto = r(beta)
local gamma_s_auto = r(gamma)
quietly generate resid = y - hw_auto
quietly generate resid_sq = resid^2
quietly summarize resid_sq
local mse_s_auto = r(mean)
post hw_handle (`alpha_s_auto') (`beta_s_auto') (`gamma_s_auto') ///
    (`mse_s_auto') (sqrt(`mse_s_auto'))
drop hw_auto resid resid_sq

* Grid esparso (alpha, beta, gamma cada um em {0.1, 0.2, 0.3, 0.5, 0.7, 0.9})
foreach a in 0.1 0.2 0.3 0.5 0.7 0.9 {
    foreach b in 0.05 0.1 0.2 0.3 0.5 {
        foreach g in 0.05 0.1 0.2 0.3 0.5 {
            capture quietly tssmooth shwinters hw_grid = y, ///
                period(12) parms(`a' `b' `g')
            if _rc == 0 {
                quietly generate resid = y - hw_grid
                quietly generate resid_sq = resid^2
                quietly summarize resid_sq
                local mse = r(mean)
                post hw_handle (`a') (`b') (`g') (`mse') (sqrt(`mse'))
                drop hw_grid resid resid_sq
            }
            else {
                capture drop hw_grid
            }
        }
    }
}

postclose hw_handle

* Carregar e mostrar resultados HW
preserve
use `hw_results', clear
sort mse
display "Top 5 combinacoes alpha/beta/gamma (por MSE):"
display _col(3) "Alpha" _col(13) "Beta" _col(23) "Gamma" ///
    _col(33) "MSE" _col(48) "RMSE"
display _col(3) "-----" _col(13) "-----" _col(23) "-----" ///
    _col(33) "----------" _col(48) "----------"
forvalues i = 1/5 {
    local a = alpha[`i']
    local b = beta[`i']
    local g = gamma[`i']
    local m = mse[`i']
    local r = rmse[`i']
    display _col(3) %5.3f `a' _col(13) %5.3f `b' _col(23) %5.3f `g' ///
        _col(33) %10.4f `m' _col(48) %10.4f `r'
}
display _newline
display "Melhor: alpha=" %5.3f alpha[1] " beta=" %5.3f beta[1] ///
    " gamma=" %5.3f gamma[1] " (MSE=" %10.4f mse[1] ")"
local best_alpha_hw = alpha[1]
local best_beta_hw = beta[1]
local best_gamma_hw = gamma[1]
local best_mse_hw = mse[1]
local best_rmse_hw = rmse[1]
export delimited "`output_dir'/grid_search_hw_seasonal.csv", replace
restore

* =============================================================================
* 5. Comparacao dos melhores modelos de cada metodo
* =============================================================================

display _newline
display "============================================="
display " Comparacao dos melhores modelos"
display "============================================="
display _newline
display _col(3) "Metodo" _col(25) "Parametros" _col(55) "MSE" _col(70) "RMSE"
display _col(3) strrep("-", 75)

display _col(3) "SES" ///
    _col(25) "a=" %4.2f `best_alpha_ses' ///
    _col(55) %10.4f `best_mse_ses' ///
    _col(70) %10.4f `best_rmse_ses'

display _col(3) "Holt" ///
    _col(25) "a=" %4.2f `best_alpha_holt' " b=" %4.2f `best_beta_holt' ///
    _col(55) %10.4f `best_mse_holt' ///
    _col(70) %10.4f `best_rmse_holt'

display _col(3) "HW sazonal" ///
    _col(25) "a=" %4.2f `best_alpha_hw' " b=" %4.2f `best_beta_hw' ///
    " g=" %4.2f `best_gamma_hw' ///
    _col(55) %10.4f `best_mse_hw' ///
    _col(70) %10.4f `best_rmse_hw'

* =============================================================================
* 6. Exportar resumo
* =============================================================================

clear
set obs 3
generate str20 method = ""
generate double best_mse = .
generate double best_rmse = .
generate str50 best_params = ""

replace method = "SES" in 1
replace best_mse = `best_mse_ses' in 1
replace best_rmse = `best_rmse_ses' in 1
replace best_params = "alpha=" + string(`best_alpha_ses', "%5.3f") in 1

replace method = "Holt" in 2
replace best_mse = `best_mse_holt' in 2
replace best_rmse = `best_rmse_holt' in 2
replace best_params = "alpha=" + string(`best_alpha_holt', "%5.3f") + ///
    " beta=" + string(`best_beta_holt', "%5.3f") in 2

replace method = "HW_seasonal" in 3
replace best_mse = `best_mse_hw' in 3
replace best_rmse = `best_rmse_hw' in 3
replace best_params = "alpha=" + string(`best_alpha_hw', "%5.3f") + ///
    " beta=" + string(`best_beta_hw', "%5.3f") + ///
    " gamma=" + string(`best_gamma_hw', "%5.3f") in 3

export delimited "`output_dir'/grid_search_summary.csv", replace
display _newline
display "Salvo: grid_search_summary.csv"

display _newline
display "============================================="
display " 03_auto_ets_validation.do concluido!"
display "============================================="
display _newline
display "LIMITACOES DOCUMENTADAS:"
display "  1. Stata NAO possui auto-ETS (selecao automatica de modelo)."
display "  2. Grid search manual e necessario para comparar parametros."
display "  3. Criterio de selecao: MSE (nao AICc, que nao esta disponivel)."
display "  4. Apenas modelos aditivos foram testados (tssmooth limitacao)."
display "  5. Damped trend NAO disponivel no tssmooth."
display "  6. O grid search aqui e mais grosso que a otimizacao numerica"
display "     usada pelo R (forecast) e Python (chronobox), portanto os"
display "     parametros otimos podem diferir ligeiramente."
