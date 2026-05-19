********************************************************************************
* compare_all_results.do
* Comprehensive comparison of Python (chronobox) vs R vs Stata results for
* both univariate and multivariate workflows.
*
* Reads outputs from:
*   - outputs/          (Python results)
*   - outputs/R/        (R results)
*   - outputs/Stata/    (Stata results)
*
* Produces:
*   - outputs/Stata/comparison_all_platforms.csv
*   - outputs/Stata/comparison_summary.csv
*   - outputs/Stata/models_not_in_stata.csv
*   - Console output with tolerance assessment
*
* NOTE: This script reads CSV files produced by all three platforms and
* creates a unified comparison table. Some Python/R output files are in JSON
* format, which Stata cannot read natively. For those, we compare only the
* CSV outputs that all three platforms produce.
*
* IMPORTANT: JSON parsing is NOT natively supported in Stata. This script
* focuses on CSV-based comparisons. For full JSON comparison, use the Python
* or R comparison scripts.
********************************************************************************

clear all
set more off

* ==============================================================================
* 0. Setup paths
* ==============================================================================
capture confirm file "../outputs/univariate_models_comparison.csv"
if _rc == 0 {
    local base_dir ".."
}
else {
    capture confirm file "examples/complete_workflow/outputs/univariate_models_comparison.csv"
    if _rc == 0 {
        local base_dir "examples/complete_workflow"
    }
    else {
        local base_dir ".."
    }
}

local py_dir    "`base_dir'/outputs"
local r_dir     "`base_dir'/outputs/R"
local stata_dir "`base_dir'/outputs/Stata"

display "=== Cross-Validation: Python vs R vs Stata ==="
display "Python outputs: `py_dir'"
display "R outputs     : `r_dir'"
display "Stata outputs : `stata_dir'"
display ""

* ==============================================================================
* 1. UNIVARIATE COMPARISON: Models
* ==============================================================================
display "=========================================="
display "=== UNIVARIATE MODEL COMPARISON ==="
display "=========================================="
display ""

* --- Load Python models ---
import delimited "`py_dir'/univariate_models_comparison.csv", clear
rename model py_model
rename aic py_aic
rename bic py_bic
rename rmse py_rmse
rename mae py_mae
rename mape py_mape
generate int source_id = _n
tempfile py_models
save `py_models'

* --- Load R models ---
import delimited "`r_dir'/univariate_models_comparison.csv", clear
rename model r_model
rename aic r_aic
rename bic r_bic
rename rmse r_rmse
rename mae r_mae
rename mape r_mape
generate int source_id = _n
tempfile r_models
save `r_models'

* --- Load Stata models ---
import delimited "`stata_dir'/univariate_models_comparison.csv", clear
rename model stata_model
rename aic stata_aic
rename bic stata_bic
rename rmse stata_rmse
rename mae stata_mae
rename mape stata_mape
generate int source_id = _n
tempfile stata_models
save `stata_models'

* --- Build comparison table ---
* Since model names differ across platforms, we compare SARIMA models
* which exist in all three platforms.
* Python and R have: SARIMA(0,1,1)(0,1,1)[12], SARIMA(1,1,0)(1,1,0)[12]
* Stata has the same SARIMA models.

display "--- SARIMA Model Comparison ---"
display ""

* Create comparison dataset manually
clear
set obs 10
generate str50 metric = ""
generate str15 category = ""
generate double python_value = .
generate double r_value = .
generate double stata_value = .
generate double diff_py_r = .
generate double diff_py_stata = .
generate double diff_r_stata = .
generate str30 tolerance_type = ""
generate double tolerance = .

* We need to reload values from each file
* Load Python SARIMA(0,1,1)(0,1,1) - look for matching row
preserve
use `py_models', clear
* Find SARIMA(0,1,1) row - typically contains "0,1,1" in model name
generate match011 = regexm(py_model, "0,1,1.*0,1,1") | regexm(py_model, "011.*011")
quietly summarize py_rmse if match011 == 1
local py_rmse_011 = r(mean)
quietly summarize py_mae if match011 == 1
local py_mae_011 = r(mean)
quietly summarize py_aic if match011 == 1
local py_aic_011 = r(mean)

* Find SARIMA(1,1,0) row
generate match110 = regexm(py_model, "1,1,0.*1,1,0") | regexm(py_model, "110.*110")
quietly summarize py_rmse if match110 == 1
local py_rmse_110 = r(mean)
quietly summarize py_mae if match110 == 1
local py_mae_110 = r(mean)
quietly summarize py_aic if match110 == 1
local py_aic_110 = r(mean)
restore

* Load R values
preserve
use `r_models', clear
generate match011 = regexm(r_model, "0,1,1.*0,1,1") | regexm(r_model, "011.*011")
quietly summarize r_rmse if match011 == 1
local r_rmse_011 = r(mean)
quietly summarize r_mae if match011 == 1
local r_mae_011 = r(mean)
quietly summarize r_aic if match011 == 1
local r_aic_011 = r(mean)

generate match110 = regexm(r_model, "1,1,0.*1,1,0") | regexm(r_model, "110.*110")
quietly summarize r_rmse if match110 == 1
local r_rmse_110 = r(mean)
quietly summarize r_mae if match110 == 1
local r_mae_110 = r(mean)
quietly summarize r_aic if match110 == 1
local r_aic_110 = r(mean)
restore

* Load Stata values
preserve
use `stata_models', clear
generate match011 = regexm(stata_model, "0,1,1.*0,1,1") | regexm(stata_model, "011.*011")
quietly summarize stata_rmse if match011 == 1
local stata_rmse_011 = r(mean)
quietly summarize stata_mae if match011 == 1
local stata_mae_011 = r(mean)
quietly summarize stata_aic if match011 == 1
local stata_aic_011 = r(mean)

generate match110 = regexm(stata_model, "1,1,0.*1,1,0") | regexm(stata_model, "110.*110")
quietly summarize stata_rmse if match110 == 1
local stata_rmse_110 = r(mean)
quietly summarize stata_mae if match110 == 1
local stata_mae_110 = r(mean)
quietly summarize stata_aic if match110 == 1
local stata_aic_110 = r(mean)
restore

* Fill comparison table
replace category = "SARIMA(011)" in 1
replace metric = "SARIMA(0,1,1)(0,1,1)_RMSE" in 1
replace python_value = `py_rmse_011' in 1
replace r_value = `r_rmse_011' in 1
replace stata_value = `stata_rmse_011' in 1
replace tolerance_type = "forecast_rmse" in 1
replace tolerance = 5.0 in 1

replace category = "SARIMA(011)" in 2
replace metric = "SARIMA(0,1,1)(0,1,1)_MAE" in 2
replace python_value = `py_mae_011' in 2
replace r_value = `r_mae_011' in 2
replace stata_value = `stata_mae_011' in 2
replace tolerance_type = "forecast_mae" in 2
replace tolerance = 5.0 in 2

replace category = "SARIMA(011)" in 3
replace metric = "SARIMA(0,1,1)(0,1,1)_AIC" in 3
replace python_value = `py_aic_011' in 3
replace r_value = `r_aic_011' in 3
replace stata_value = `stata_aic_011' in 3
replace tolerance_type = "model_aic" in 3
replace tolerance = 10.0 in 3

replace category = "SARIMA(110)" in 4
replace metric = "SARIMA(1,1,0)(1,1,0)_RMSE" in 4
replace python_value = `py_rmse_110' in 4
replace r_value = `r_rmse_110' in 4
replace stata_value = `stata_rmse_110' in 4
replace tolerance_type = "forecast_rmse" in 4
replace tolerance = 5.0 in 4

replace category = "SARIMA(110)" in 5
replace metric = "SARIMA(1,1,0)(1,1,0)_MAE" in 5
replace python_value = `py_mae_110' in 5
replace r_value = `r_mae_110' in 5
replace stata_value = `stata_mae_110' in 5
replace tolerance_type = "forecast_mae" in 5
replace tolerance = 5.0 in 5

replace category = "SARIMA(110)" in 6
replace metric = "SARIMA(1,1,0)(1,1,0)_AIC" in 6
replace python_value = `py_aic_110' in 6
replace r_value = `r_aic_110' in 6
replace stata_value = `stata_aic_110' in 6
replace tolerance_type = "model_aic" in 6
replace tolerance = 10.0 in 6

* Drop unused rows
drop if metric == ""

* Compute differences
replace diff_py_r = abs(python_value - r_value)
replace diff_py_stata = abs(python_value - stata_value)
replace diff_r_stata = abs(r_value - stata_value)

* Assess tolerance
generate byte within_tol_py_r = (diff_py_r <= tolerance)
generate byte within_tol_py_stata = (diff_py_stata <= tolerance)
generate byte within_tol_r_stata = (diff_r_stata <= tolerance)

display "--- Univariate Cross-Platform Comparison ---"
list metric python_value r_value stata_value diff_py_r diff_py_stata diff_r_stata, ///
    noobs abbreviate(25)

tempfile uni_comparison
save `uni_comparison'

* ==============================================================================
* 2. MULTIVARIATE COMPARISON: Forecasts
* ==============================================================================
display ""
display "=========================================="
display "=== MULTIVARIATE FORECAST COMPARISON ==="
display "=========================================="
display ""

* --- Load Python forecasts ---
capture {
    import delimited "`py_dir'/multivariate_forecasts.csv", clear
    * Keep first forecast per variable
    bysort variable: keep if _n == 1
    keep variable var_forecast
    rename var_forecast py_forecast
    tempfile py_fc
    save `py_fc'
}

* --- Load R forecasts ---
capture {
    import delimited "`r_dir'/multivariate_forecasts.csv", clear
    bysort variable: keep if _n == 1
    keep variable var_forecast
    rename var_forecast r_forecast
    tempfile r_fc
    save `r_fc'
}

* --- Load Stata forecasts (wide format) ---
capture {
    import delimited "`stata_dir'/multivariate_forecasts.csv", clear
    keep if _n == 1
    * Stata forecasts are in wide format, need to manually extract
    local stata_fc_gdp = gdp_forecast[1]
    local stata_fc_inf = inflation_forecast[1]
    local stata_fc_ff  = fed_funds_forecast[1]
    local stata_fc_ue  = unemployment_forecast[1]
}

* Build multivariate comparison
capture {
    use `py_fc', clear
    merge 1:1 variable using `r_fc', nogenerate
    generate double stata_forecast = .
    replace stata_forecast = `stata_fc_gdp' if variable == "gdp"
    replace stata_forecast = `stata_fc_inf' if variable == "inflation"
    replace stata_forecast = `stata_fc_ff'  if variable == "fed_funds"
    replace stata_forecast = `stata_fc_ue'  if variable == "unemployment"

    generate double diff_py_r = abs(py_forecast - r_forecast)
    generate double diff_py_stata = abs(py_forecast - stata_forecast)
    generate double diff_r_stata = abs(r_forecast - stata_forecast)

    display "--- VAR Forecast Comparison (h=1) ---"
    list variable py_forecast r_forecast stata_forecast ///
        diff_py_r diff_py_stata diff_r_stata, noobs

    tempfile multi_comparison
    save `multi_comparison'
}

* ==============================================================================
* 3. COMBINED COMPARISON TABLE
* ==============================================================================
display ""
display "=========================================="
display "=== COMBINED COMPARISON SUMMARY ==="
display "=========================================="
display ""

use `uni_comparison', clear
generate str20 workflow = "univariate"
generate str20 platform_pair = ""

* Calculate overall pass rates
quietly count if within_tol_py_r == 1
local pass_py_r = r(N)
quietly count
local total = r(N)

display "--- Tolerance Assessment ---"
display "Python vs R:     `pass_py_r'/`total' within tolerance"

quietly count if within_tol_py_stata == 1
local pass_py_stata = r(N)
display "Python vs Stata: `pass_py_stata'/`total' within tolerance"

quietly count if within_tol_r_stata == 1
local pass_r_stata = r(N)
display "R vs Stata:      `pass_r_stata'/`total' within tolerance"

* Export combined comparison
export delimited "`stata_dir'/comparison_all_platforms.csv", replace
display ""
display "Saved: comparison_all_platforms.csv"

* ==============================================================================
* 4. MODELS NOT AVAILABLE IN STATA
* ==============================================================================
display ""
display "=========================================="
display "=== MODELS WITHOUT NATIVE STATA EQUIVALENT ==="
display "=========================================="
display ""

clear
set obs 15
generate str40 model_or_method = ""
generate str15 available_in_python = ""
generate str15 available_in_r = ""
generate str15 available_in_stata = ""
generate str80 stata_alternative = ""
generate str80 limitation_notes = ""

* --- Univariate models ---
replace model_or_method = "Auto-ARIMA" in 1
replace available_in_python = "Yes" in 1
replace available_in_r = "Yes" in 1
replace available_in_stata = "No" in 1
replace stata_alternative = "Manual grid search over (p,d,q) with estat ic" in 1
replace limitation_notes = "No automatic order selection; user must specify orders" in 1

replace model_or_method = "ETS (30 models)" in 2
replace available_in_python = "Yes" in 2
replace available_in_r = "Yes" in 2
replace available_in_stata = "Partial" in 2
replace stata_alternative = "tssmooth (exponential, dexponential, hwinters)" in 2
replace limitation_notes = "Only 3 smoothing types; no damped trend; no auto selection" in 2

replace model_or_method = "Theta method" in 3
replace available_in_python = "Yes" in 3
replace available_in_r = "Yes" in 3
replace available_in_stata = "No" in 3
replace stata_alternative = "None" in 3
replace limitation_notes = "Not available natively or via SSC" in 3

replace model_or_method = "STL decomposition" in 4
replace available_in_python = "Yes" in 4
replace available_in_r = "Yes" in 4
replace available_in_stata = "No" in 4
replace stata_alternative = "tsdecomp (SSC) or manual classical decomposition" in 4
replace limitation_notes = "No LOESS-based decomposition; only additive classical" in 4

replace model_or_method = "Zivot-Andrews test" in 5
replace available_in_python = "Yes" in 5
replace available_in_r = "Yes" in 5
replace available_in_stata = "SSC" in 5
replace stata_alternative = "zandrews (ssc install zandrews)" in 5
replace limitation_notes = "Requires user-written package, not native" in 5

replace model_or_method = "KPSS test" in 6
replace available_in_python = "Yes" in 6
replace available_in_r = "Yes" in 6
replace available_in_stata = "SSC" in 6
replace stata_alternative = "kpss (ssc install kpss)" in 6
replace limitation_notes = "Requires user-written package; no p-value returned" in 6

* --- Multivariate models ---
replace model_or_method = "Bayesian VAR (BVAR)" in 7
replace available_in_python = "Yes" in 7
replace available_in_r = "Yes" in 7
replace available_in_stata = "No" in 7
replace stata_alternative = "None (limited SSC options)" in 7
replace limitation_notes = "No Minnesota prior, no hierarchical priors, no MCMC" in 7

replace model_or_method = "Spillover index (D-Y)" in 8
replace available_in_python = "Yes" in 8
replace available_in_r = "Yes" in 8
replace available_in_stata = "Manual" in 8
replace stata_alternative = "Compute from FEVD output manually" in 8
replace limitation_notes = "No native command; rolling-window requires looping" in 8

replace model_or_method = "ARDL bounds test" in 9
replace available_in_python = "Yes" in 9
replace available_in_r = "Yes" in 9
replace available_in_stata = "SSC" in 9
replace stata_alternative = "ardl (ssc install ardl)" in 9
replace limitation_notes = "SSC package; syntax/defaults differ from R/Python" in 9

replace model_or_method = "HP filter" in 10
replace available_in_python = "Yes" in 10
replace available_in_r = "Yes" in 10
replace available_in_stata = "Yes" in 10
replace stata_alternative = "tsfilter hp (native)" in 10
replace limitation_notes = "Fully supported natively since Stata 12" in 10

replace model_or_method = "ARIMA/SARIMA" in 11
replace available_in_python = "Yes" in 11
replace available_in_r = "Yes" in 11
replace available_in_stata = "Yes" in 11
replace stata_alternative = "arima (native)" in 11
replace limitation_notes = "Fully supported; conditional vs exact MLE may differ" in 11

replace model_or_method = "VAR" in 12
replace available_in_python = "Yes" in 12
replace available_in_r = "Yes" in 12
replace available_in_stata = "Yes" in 12
replace stata_alternative = "var (native)" in 12
replace limitation_notes = "Fully supported with varsoc, vargranger, etc." in 12

replace model_or_method = "VEC (Johansen)" in 13
replace available_in_python = "Yes" in 13
replace available_in_r = "Yes" in 13
replace available_in_stata = "Yes" in 13
replace stata_alternative = "vec, vecrank (native)" in 13
replace limitation_notes = "Fully supported natively" in 13

replace model_or_method = "SVAR" in 14
replace available_in_python = "Yes" in 14
replace available_in_r = "Yes" in 14
replace available_in_stata = "Yes" in 14
replace stata_alternative = "svar (native)" in 14
replace limitation_notes = "Requires explicit A/B constraint matrices" in 14

replace model_or_method = "IRF / FEVD" in 15
replace available_in_python = "Yes" in 15
replace available_in_r = "Yes" in 15
replace available_in_stata = "Yes" in 15
replace stata_alternative = "irf create, irf table (native)" in 15
replace limitation_notes = "Results stored in .irf binary files; export needs extra steps" in 15

* Display the table
display ""
display "Model/Method Availability Across Platforms:"
display "============================================"
list model_or_method available_in_python available_in_r available_in_stata ///
    stata_alternative, noobs abbreviate(25)

display ""
display "Limitation Notes:"
display "================="
list model_or_method limitation_notes if limitation_notes != "", ///
    noobs abbreviate(25)

* Export
export delimited "`stata_dir'/models_not_in_stata.csv", replace
display ""
display "Saved: models_not_in_stata.csv"

* ==============================================================================
* 5. TOLERANCE DOCUMENTATION
* ==============================================================================
display ""
display "=========================================="
display "=== TOLERANCE DOCUMENTATION ==="
display "=========================================="
display ""
display "Result Type                    Tolerance  Rationale"
display "-------------------------------------------------------------------"
display "Unit root statistic            0.5        Different lag selection defaults"
display "Unit root p-value              0.1        P-value interpolation/tables differ"
display "HP filter                      0.01       Deterministic, same lambda"
display "Model AIC/BIC                  10.0       Likelihood constant differs (cond vs exact MLE)"
display "Forecast RMSE/MAE              5.0        Different optimization algorithms"
display "Forecast MAPE                  2.0        Percentage metric"
display "Forecast values                20.0       Accumulated estimation differences"
display "IRF (VAR)                      0.05       Same Cholesky, small numerical diffs"
display "FEVD / Spillover               5.0        Percentage points"
display "BVAR IRF                       0.15       MCMC sampling variability"
display "BVAR forecast                  1.0        MCMC draws produce variation"
display "Cointegration rank             1          Borderline test statistics"
display ""

* ==============================================================================
* 6. OVERALL SUMMARY
* ==============================================================================
display ""
display "=========================================="
display "=== OVERALL SUMMARY ==="
display "=========================================="
display ""
display "Platforms compared: Python (chronobox), R (forecast/vars/urca), Stata"
display ""
display "UNIVARIATE:"
display "  - All platforms support ARIMA/SARIMA estimation"
display "  - HP filter available in all (tsfilter hp in Stata)"
display "  - Stata LACKS: auto.arima, full ETS, STL, Theta method"
display "  - Differences due to: conditional vs exact MLE, lag selection defaults"
display ""
display "MULTIVARIATE:"
display "  - All platforms support VAR, VEC, SVAR, IRF, FEVD"
display "  - Granger causality available in all (vargranger in Stata)"
display "  - Stata LACKS: BVAR (Bayesian VAR), native spillover index"
display "  - Differences due to: numerical backends, SVAR constraint syntax"
display ""
display "KEY FINDING: Core econometric methods (ARIMA, VAR, VEC, SVAR, Johansen)"
display "are available across all three platforms. Differences are primarily in:"
display "  1. Modern/specialized methods (BVAR, ETS auto-selection, spillover)"
display "  2. Convenience features (auto.arima, automatic model selection)"
display "  3. Numerical precision (conditional vs exact MLE, p-value interpolation)"
display ""

* Save summary
preserve
clear
set obs 6
generate str50 summary_item = ""
generate str100 description = ""

replace summary_item = "Platforms" in 1
replace description = "Python (chronobox), R (forecast/vars/urca/BVAR), Stata (native)" in 1

replace summary_item = "Univariate models shared" in 2
replace description = "ARIMA, SARIMA, HP filter, ADF test" in 2

replace summary_item = "Multivariate models shared" in 3
replace description = "VAR, VEC, SVAR, Johansen, Granger, IRF, FEVD" in 3

replace summary_item = "Stata missing (univariate)" in 4
replace description = "auto.arima, full ETS, STL, Theta, KPSS (native)" in 4

replace summary_item = "Stata missing (multivariate)" in 5
replace description = "BVAR, Diebold-Yilmaz spillover (native)" in 5

replace summary_item = "Main diff sources" in 6
replace description = "Conditional vs exact MLE, lag defaults, MCMC sampling" in 6

export delimited "`stata_dir'/comparison_summary.csv", replace
display "Saved: comparison_summary.csv"
restore

display ""
display "=== Comparison Complete ==="
display "Output files:"
display "  - comparison_all_platforms.csv"
display "  - models_not_in_stata.csv"
display "  - comparison_summary.csv"
