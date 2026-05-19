* =============================================================================
* 02_hw_validation.do
* Validacao cruzada: Holt-Winters (sazonal e nao-sazonal)
*
* Usa tssmooth hwinters (Holt-Winters nao-sazonal com trend)
* e tssmooth shwinters (Holt-Winters sazonal) para comparar
* com resultados do chronobox (Python) e forecast (R).
*
* LIMITACOES DO STATA PARA HOLT-WINTERS:
*   - tssmooth hwinters: Holt-Winters sem sazonalidade (equivale a Holt
*     com otimizacao diferente de dexponential).
*   - tssmooth shwinters: Holt-Winters sazonal aditivo apenas.
*     Stata NAO suporta Holt-Winters multiplicativo nativamente.
*   - Nao ha opcao de damped trend em tssmooth hwinters/shwinters.
*   - Criterios de informacao (AIC/AICc/BIC) nao sao reportados.
*   - Previsoes fora da amostra requerem manipulacao manual
*     (estender a serie com missings e usar predict).
*
* Datasets: airline.csv (mensal, sazonal)
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de diretorios ---
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

capture mkdir "`output_dir'"

display _newline
display "============================================="
display " Holt-Winters Validation"
display "============================================="
display _newline

* =============================================================================
* 1. AIRLINE DATASET - Holt-Winters nao-sazonal (hwinters)
* =============================================================================

display "--- Carregando airline.csv ---"
import delimited "`data_dir'/airline.csv", clear

generate month_id = _n
tsset month_id
rename passengers y

local n_obs = _N
display "Observacoes: `n_obs'"

* --- Holt-Winters nao-sazonal (otimizado) ---
display _newline
display "=== Holt-Winters nao-sazonal (hwinters, otimizado) ==="
* tssmooth hwinters ajusta alpha e beta
tssmooth hwinters hw_ns = y

local alpha_hw = r(alpha)
local beta_hw = r(beta)
display "Alpha: " %9.4f `alpha_hw'
display "Beta:  " %9.4f `beta_hw'

* Metricas in-sample
generate resid_hw = y - hw_ns
generate resid_sq = resid_hw^2
quietly summarize resid_sq
local rmse_hw = sqrt(r(mean))
generate resid_abs = abs(resid_hw)
quietly summarize resid_abs
local mae_hw = r(mean)
generate pct_err = abs(resid_hw / y) * 100
quietly summarize pct_err
local mape_hw = r(mean)
drop resid_sq resid_abs pct_err

display "RMSE: " %9.4f `rmse_hw'
display "MAE:  " %9.4f `mae_hw'
display "MAPE: " %9.4f `mape_hw' "%"

* =============================================================================
* 2. AIRLINE DATASET - Holt-Winters sazonal aditivo (shwinters)
* =============================================================================

display _newline
display "=== Holt-Winters sazonal aditivo (shwinters, periodo=12) ==="
* shwinters requer especificar o periodo sazonal
tssmooth shwinters hw_add = y, period(12)

local alpha_add = r(alpha)
local beta_add = r(beta)
local gamma_add = r(gamma)
display "Alpha: " %9.4f `alpha_add'
display "Beta:  " %9.4f `beta_add'
display "Gamma: " %9.4f `gamma_add'

* Metricas
generate resid_add = y - hw_add
generate resid_sq = resid_add^2
quietly summarize resid_sq
local rmse_add = sqrt(r(mean))
generate resid_abs = abs(resid_add)
quietly summarize resid_abs
local mae_add = r(mean)
generate pct_err = abs(resid_add / y) * 100
quietly summarize pct_err
local mape_add = r(mean)
drop resid_sq resid_abs pct_err

display "RMSE: " %9.4f `rmse_add'
display "MAE:  " %9.4f `mae_add'
display "MAPE: " %9.4f `mape_add' "%"

* =============================================================================
* 3. Holt-Winters sazonal com parametros fixos
* =============================================================================

display _newline
display "=== HW sazonal aditivo (alpha=0.2, beta=0.1, gamma=0.1) ==="
tssmooth shwinters hw_fixed = y, period(12) parms(0.2 0.1 0.1)

local alpha_fix = 0.2
local beta_fix = 0.1
local gamma_fix = 0.1

generate resid_fix = y - hw_fixed
generate resid_sq = resid_fix^2
quietly summarize resid_sq
local rmse_fix = sqrt(r(mean))
generate resid_abs = abs(resid_fix)
quietly summarize resid_abs
local mae_fix = r(mean)
generate pct_err = abs(resid_fix / y) * 100
quietly summarize pct_err
local mape_fix = r(mean)
drop resid_sq resid_abs pct_err

display "RMSE: " %9.4f `rmse_fix'
display "MAE:  " %9.4f `mae_fix'
display "MAPE: " %9.4f `mape_fix' "%"

* =============================================================================
* 4. Previsoes fora da amostra (h=24)
* =============================================================================

display _newline
display "=== Previsoes fora da amostra (h=24) ==="
display "Nota: tssmooth nao gera forecasts automaticamente."
display "      Estendemos a serie com missings e re-ajustamos."

* Salvar dados originais
preserve

* Estender a serie com 24 periodos adicionais
local new_n = `n_obs' + 24
set obs `new_n'
replace month_id = _n if month_id == .
tsset month_id

* Re-aplicar shwinters na serie estendida
* Os valores smoothed nos periodos com y=. sao as previsoes
tssmooth shwinters hw_fcast = y, period(12)

* Extrair previsoes (ultimos 24 periodos)
display _newline
display "Previsoes HW aditivo (h=1 a h=24):"
list month_id hw_fcast in `=`n_obs'+1'/`new_n', noobs

* Salvar previsoes
generate h = _n - `n_obs' if _n > `n_obs'
keep if h != .
keep h hw_fcast
rename hw_fcast forecast_hw_additive
export delimited "`output_dir'/hw_forecasts.csv", replace
display "Salvo: hw_forecasts.csv"

restore

* =============================================================================
* 5. Exportar resultados consolidados
* =============================================================================

display _newline
display "--- Exportando resultados ---"

clear
set obs 4
generate str30 method = ""
generate double alpha = .
generate double beta = .
generate double gamma = .
generate double rmse = .
generate double mae = .
generate double mape = .

replace method = "HW_nonseasonal" in 1
replace alpha = `alpha_hw' in 1
replace beta = `beta_hw' in 1
replace rmse = `rmse_hw' in 1
replace mae = `mae_hw' in 1
replace mape = `mape_hw' in 1

replace method = "HW_additive_opt" in 2
replace alpha = `alpha_add' in 2
replace beta = `beta_add' in 2
replace gamma = `gamma_add' in 2
replace rmse = `rmse_add' in 2
replace mae = `mae_add' in 2
replace mape = `mape_add' in 2

replace method = "HW_additive_fixed" in 3
replace alpha = `alpha_fix' in 3
replace beta = `beta_fix' in 3
replace gamma = `gamma_fix' in 3
replace rmse = `rmse_fix' in 3
replace mae = `mae_fix' in 3
replace mape = `mape_fix' in 3

replace method = "HW_multiplicative" in 4
replace method = "NOT_SUPPORTED" in 4
* Stata tssmooth shwinters nao suporta Holt-Winters multiplicativo

export delimited "`output_dir'/hw_results.csv", replace
display "Salvo: hw_results.csv"

display _newline
display "============================================="
display " 02_hw_validation.do concluido!"
display "============================================="
display _newline
display "NOTA: tssmooth hwinters = Holt-Winters sem sazonalidade"
display "      tssmooth shwinters = Holt-Winters sazonal ADITIVO apenas"
display "      Stata NAO suporta HW multiplicativo via tssmooth."
display "      Stata NAO suporta damped trend via tssmooth."
display "      Para HW multiplicativo, considerar stsmooth (Stata 17+)"
display "      ou pacotes contribuidos."
