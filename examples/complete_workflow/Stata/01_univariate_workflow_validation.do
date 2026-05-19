********************************************************************************
* 01_univariate_workflow_validation.do
* Complete univariate time series workflow in Stata for cross-validation with
* chronobox (Python) and R.
*
* Pipeline:
*   1. Load airline passengers dataset
*   2. Unit root tests: dfuller, kpss
*   3. HP filter: tsfilter hp
*   4. ARIMA / SARIMA estimation
*   5. Exponential smoothing: tssmooth
*   6. Information criteria comparison: estat ic
*   7. Forecast and export results
*
* Dataset: airline.csv (144 monthly observations, 1949m1-1960m12)
* Seed: set seed 42
*
* LIMITATIONS DOCUMENTED:
*   - Stata's kpss requires the user-written -kpss- command (ssc install kpss)
*   - Stata's tssmooth does not support full ETS(A,Ad,M) model selection like
*     R's ets() or Python's statsmodels. Only simple/double/Holt-Winters
*     exponential smoothing are available natively.
*   - Stata does not have an automatic ARIMA order selection equivalent to
*     R's auto.arima() or Python's pm.auto_arima(). Users must specify orders.
*   - Stata's ARIMA uses conditional MLE by default (different from R/Python
*     exact MLE), which can cause small numerical differences in AIC/BIC.
*   - Stata does not have a native Theta method or STL decomposition.
*   - Zivot-Andrews test requires -zandrews- from SSC.
*   - MAPE is not a built-in post-estimation statistic in Stata.
********************************************************************************

clear all
set more off
set seed 42

* ==============================================================================
* 0. Setup paths
* ==============================================================================
* Detect script location and set paths
local script_dir "`c(pwd)'"

* Try to find the data relative to common locations
capture confirm file "../data/airline.csv"
if _rc == 0 {
    local base_dir ".."
}
else {
    capture confirm file "examples/complete_workflow/data/airline.csv"
    if _rc == 0 {
        local base_dir "examples/complete_workflow"
    }
    else {
        * Default: assume running from examples/complete_workflow/Stata/
        local base_dir ".."
    }
}

local data_dir    "`base_dir'/data"
local output_dir  "`base_dir'/outputs/Stata"

* Create output directory
capture mkdir "`output_dir'"

display "=== Univariate Workflow Validation (Stata) ==="
display "Data dir  : `data_dir'"
display "Output dir: `output_dir'"
display ""

* ==============================================================================
* 1. Load data
* ==============================================================================
import delimited "`data_dir'/airline.csv", clear

* Parse date and set time series
generate date_stata = date(date, "YMD")
format date_stata %td
generate month_id = mofd(date_stata)
format month_id %tm
tsset month_id

* Generate log of passengers
generate log_passengers = ln(passengers)

local n_obs = _N
display "Loaded airline dataset: `n_obs' observations"

* ==============================================================================
* 2. Unit root tests
* ==============================================================================
display ""
display "--- Unit Root Tests ---"

* --- ADF on raw series ---
* Stata's dfuller uses the ADF regression with automatic lag selection
dfuller passengers, lags(4)
local adf_raw_stat = r(Zt)
local adf_raw_pval = r(p)
display "ADF (raw):  stat = `adf_raw_stat', p-value = `adf_raw_pval'"

* --- ADF on log series ---
dfuller log_passengers, lags(4)
local adf_log_stat = r(Zt)
local adf_log_pval = r(p)
display "ADF (log):  stat = `adf_log_stat', p-value = `adf_log_pval'"

* --- KPSS on raw series ---
* NOTE: kpss requires user-written command: ssc install kpss
* If not installed, this will fail gracefully
capture noisily kpss passengers, maxlag(4)
if _rc == 0 {
    local kpss_raw_stat = r(kpss)
    local kpss_raw_pval = .
    display "KPSS (raw): stat = `kpss_raw_stat'"
    display "  NOTE: Stata kpss does not report p-values directly."
    display "  Compare test statistic to critical values in output above."
}
else {
    display "KPSS test not available. Install with: ssc install kpss"
    local kpss_raw_stat = .
    local kpss_raw_pval = .
}

* --- KPSS on log series ---
capture noisily kpss log_passengers, maxlag(4)
if _rc == 0 {
    local kpss_log_stat = r(kpss)
    local kpss_log_pval = .
    display "KPSS (log): stat = `kpss_log_stat'"
}
else {
    local kpss_log_stat = .
    local kpss_log_pval = .
}

* --- ADF on first difference of log ---
generate diff_log = D.log_passengers
dfuller diff_log, lags(4)
local adf_dlog_stat = r(Zt)
local adf_dlog_pval = r(p)
display "ADF (diff log):  stat = `adf_dlog_stat', p-value = `adf_dlog_pval'"

* --- ADF on seasonal difference of first difference of log ---
generate diff_seas_log = D12.log_passengers
generate diff_diff_seas_log = D.diff_seas_log
dfuller diff_diff_seas_log, lags(4)
local adf_dslog_stat = r(Zt)
local adf_dslog_pval = r(p)
display "ADF (diff+seas log): stat = `adf_dslog_stat', p-value = `adf_dslog_pval'"

* --- Save unit root test results ---
preserve
clear
set obs 6
generate str30 test_name = ""
generate double statistic = .
generate double pvalue = .
generate str20 conclusion = ""

replace test_name = "adf_raw" in 1
replace statistic = `adf_raw_stat' in 1
replace pvalue = `adf_raw_pval' in 1
replace conclusion = cond(`adf_raw_pval' < 0.05, "Stationary", "Non-stationary") in 1

replace test_name = "adf_log" in 2
replace statistic = `adf_log_stat' in 2
replace pvalue = `adf_log_pval' in 2
replace conclusion = cond(`adf_log_pval' < 0.05, "Stationary", "Non-stationary") in 2

replace test_name = "kpss_raw" in 3
replace statistic = `kpss_raw_stat' in 3
replace pvalue = `kpss_raw_pval' in 3
replace conclusion = "See critical values" in 3

replace test_name = "kpss_log" in 4
replace statistic = `kpss_log_stat' in 4
replace pvalue = `kpss_log_pval' in 4
replace conclusion = "See critical values" in 4

replace test_name = "adf_diff_log" in 5
replace statistic = `adf_dlog_stat' in 5
replace pvalue = `adf_dlog_pval' in 5
replace conclusion = cond(`adf_dlog_pval' < 0.05, "Stationary", "Non-stationary") in 5

replace test_name = "adf_diff_seasonal_log" in 6
replace statistic = `adf_dslog_stat' in 6
replace pvalue = `adf_dslog_pval' in 6
replace conclusion = cond(`adf_dslog_pval' < 0.05, "Stationary", "Non-stationary") in 6

export delimited "`output_dir'/univariate_unit_root_tests.csv", replace
display "Saved: univariate_unit_root_tests.csv"
restore

* ==============================================================================
* 3. HP Filter
* ==============================================================================
display ""
display "--- HP Filter ---"

* tsfilter hp requires tsset, which we already did
* lambda = 14400 for monthly data (Ravn-Uhlig rule)
tsfilter hp hp_cycle = log_passengers, smooth(14400) trend(hp_trend)

* Save HP filter results
preserve
keep date passengers log_passengers hp_trend hp_cycle
rename log_passengers original
export delimited "`output_dir'/univariate_hp_filter.csv", replace
display "Saved: univariate_hp_filter.csv"
display "  HP trend range: [" %9.4f hp_trend[1] ", " %9.4f hp_trend[_N] "]"
restore

* ==============================================================================
* 4. Model Estimation
* ==============================================================================
display ""
display "--- Model Estimation ---"

* Train/test split: train = first 132 obs, test = last 12 obs
local n_train = `n_obs' - 12
local n_test  = 12
display "Train: `n_train' obs, Test: `n_test' obs"

* Get the month_id boundary for the split
local split_month = month_id[`n_train']
local first_test  = month_id[`n_train' + 1]
display "Training ends at: " %tm `split_month'
display "Testing starts at: " %tm `first_test'

* --- ARIMA(0,1,1)(0,1,1)12 (Airline model) ---
display ""
display "--- SARIMA(0,1,1)(0,1,1)12 ---"
arima passengers if _n <= `n_train', arima(0,1,1) sarima(0,1,1,12)
estat ic
* Store AIC and BIC
matrix ic_011 = r(S)
local aic_011 = ic_011[1,5]
local bic_011 = ic_011[1,6]
display "  AIC = `aic_011', BIC = `bic_011'"

* Forecast out-of-sample
predict yhat_011, y dynamic(`first_test')

* Calculate forecast errors for test period
generate err_011 = passengers - yhat_011 if _n > `n_train'
generate err2_011 = err_011^2 if _n > `n_train'
generate abserr_011 = abs(err_011) if _n > `n_train'
generate pct_011 = abs(err_011 / passengers) * 100 if _n > `n_train'

quietly summarize err2_011
local rmse_011 = sqrt(r(mean))
quietly summarize abserr_011
local mae_011 = r(mean)
quietly summarize pct_011
local mape_011 = r(mean)

display "  Test RMSE = `rmse_011'"
display "  Test MAE  = `mae_011'"
display "  Test MAPE = `mape_011'"

* --- ARIMA(1,1,0)(1,1,0)12 ---
display ""
display "--- SARIMA(1,1,0)(1,1,0)12 ---"
arima passengers if _n <= `n_train', arima(1,1,0) sarima(1,1,0,12)
estat ic
matrix ic_110 = r(S)
local aic_110 = ic_110[1,5]
local bic_110 = ic_110[1,6]
display "  AIC = `aic_110', BIC = `bic_110'"

predict yhat_110, y dynamic(`first_test')
generate err_110 = passengers - yhat_110 if _n > `n_train'
generate err2_110 = err_110^2 if _n > `n_train'
generate abserr_110 = abs(err_110) if _n > `n_train'
generate pct_110 = abs(err_110 / passengers) * 100 if _n > `n_train'

quietly summarize err2_110
local rmse_110 = sqrt(r(mean))
quietly summarize abserr_110
local mae_110 = r(mean)
quietly summarize pct_110
local mape_110 = r(mean)

display "  Test RMSE = `rmse_110'"
display "  Test MAE  = `mae_110'"
display "  Test MAPE = `mape_110'"

* --- ARIMA(1,1,1)(1,1,1)12 ---
display ""
display "--- SARIMA(1,1,1)(1,1,1)12 ---"
arima passengers if _n <= `n_train', arima(1,1,1) sarima(1,1,1,12)
estat ic
matrix ic_111 = r(S)
local aic_111 = ic_111[1,5]
local bic_111 = ic_111[1,6]
display "  AIC = `aic_111', BIC = `bic_111'"

predict yhat_111, y dynamic(`first_test')
generate err_111 = passengers - yhat_111 if _n > `n_train'
generate err2_111 = err_111^2 if _n > `n_train'
generate abserr_111 = abs(err_111) if _n > `n_train'
generate pct_111 = abs(err_111 / passengers) * 100 if _n > `n_train'

quietly summarize err2_111
local rmse_111 = sqrt(r(mean))
quietly summarize abserr_111
local mae_111 = r(mean)
quietly summarize pct_111
local mape_111 = r(mean)

display "  Test RMSE = `rmse_111'"
display "  Test MAE  = `mae_111'"
display "  Test MAPE = `mape_111'"

* ==============================================================================
* 5. Exponential Smoothing (tssmooth)
* ==============================================================================
display ""
display "--- Exponential Smoothing ---"

* --- Single exponential smoothing ---
* NOTE: tssmooth only works on training sample. We estimate on full data and
* compare in-sample fit, since tssmooth does not natively support out-of-sample
* dynamic forecasting like arima's predict with dynamic().
display ""
display "--- tssmooth exponential (single) ---"
tssmooth exponential sm_single = passengers if _n <= `n_train', parms(0.3)

* --- Double exponential smoothing (Holt) ---
display ""
display "--- tssmooth dexponential (Holt) ---"
tssmooth dexponential sm_double = passengers if _n <= `n_train', parms(0.3 0.1)

* --- Holt-Winters (multiplicative seasonal) ---
display ""
display "--- tssmooth hwinters (Holt-Winters) ---"
* NOTE: tssmooth hwinters fits additive seasonality only in some Stata versions.
* Multiplicative seasonality is supported via the "multiplicative" option in
* Stata 17+. For older versions, only additive is available.
capture tssmooth hwinters sm_hw = passengers if _n <= `n_train', ///
    parms(0.3 0.1 0.1) sn0(average)
if _rc != 0 {
    display "tssmooth hwinters failed. This may require Stata 17+ or specific options."
    display "Limitation: Stata's exponential smoothing is less flexible than R/Python."
}

* ==============================================================================
* 6. Save models comparison
* ==============================================================================
display ""
display "--- Models Comparison ---"

preserve
clear
set obs 3
generate str40 Model = ""
generate double AIC = .
generate double BIC = .
generate double RMSE = .
generate double MAE = .
generate double MAPE = .

replace Model = "SARIMA(0,1,1)(0,1,1)[12]" in 1
replace AIC = `aic_011' in 1
replace BIC = `bic_011' in 1
replace RMSE = `rmse_011' in 1
replace MAE = `mae_011' in 1
replace MAPE = `mape_011' in 1

replace Model = "SARIMA(1,1,0)(1,1,0)[12]" in 2
replace AIC = `aic_110' in 2
replace BIC = `bic_110' in 2
replace RMSE = `rmse_110' in 2
replace MAE = `mae_110' in 2
replace MAPE = `mape_110' in 2

replace Model = "SARIMA(1,1,1)(1,1,1)[12]" in 3
replace AIC = `aic_111' in 3
replace BIC = `bic_111' in 3
replace RMSE = `rmse_111' in 3
replace MAE = `mae_111' in 3
replace MAPE = `mape_111' in 3

export delimited "`output_dir'/univariate_models_comparison.csv", replace
display "Saved: univariate_models_comparison.csv"
restore

* ==============================================================================
* 7. Save forecasts (24-step ahead from full series for comparison)
* ==============================================================================
display ""
display "--- 24-Step-Ahead ARIMA Forecast (full series) ---"

* Re-estimate on full series
arima passengers, arima(0,1,1) sarima(0,1,1,12)

* Extend dataset for forecasting
local last_month = month_id[_N]
local n_forecast = 24
local new_n = _N + `n_forecast'
set obs `new_n'

* Fill in future months
forvalues i = 1/`n_forecast' {
    local row = `n_obs' + `i'
    quietly replace month_id = `last_month' + `i' in `row'
}
tsset month_id

* Dynamic forecast from first out-of-sample period
local first_oos = `last_month' + 1
predict yhat_full, y dynamic(`first_oos')

* Save forecasts
preserve
keep if _n > `n_obs'
generate date_fc = dofm(month_id)
format date_fc %td
keep date_fc yhat_full
rename yhat_full forecast
rename date_fc date
generate str30 model = "SARIMA(0,1,1)(0,1,1)[12]"

* NOTE: Stata ARIMA does not produce prediction intervals as easily as R/Python.
* Limitation: confidence intervals require manual computation or additional commands.
generate double lower_95 = .
generate double upper_95 = .

export delimited "`output_dir'/univariate_forecasts.csv", replace
display "Saved: univariate_forecasts.csv"
restore

* ==============================================================================
* 8. Documentation of Stata limitations for univariate analysis
* ==============================================================================
display ""
display "=== STATA LIMITATIONS: Univariate Analysis ==="
display ""
display "1. NO auto.arima equivalent: Stata has no automatic ARIMA order"
display "   selection. Users must manually specify (p,d,q)(P,D,Q)s orders."
display "   Workaround: loop over candidate models and compare AIC/BIC."
display ""
display "2. KPSS test not native: Requires user-written -kpss- (ssc install kpss)."
display "   The command does not return p-values directly; users compare to"
display "   critical value tables."
display ""
display "3. Zivot-Andrews test not native: Requires -zandrews- (ssc install zandrews)."
display ""
display "4. tssmooth limitations:"
display "   - Only simple, double (Holt), and Holt-Winters smoothing."
display "   - No automatic model selection like R's ets() (30 models)."
display "   - No damped trend option in all versions."
display "   - Multiplicative seasonality only in Stata 17+."
display ""
display "5. No STL decomposition: Stata has no native STL. Only additive"
display "   classical decomposition via -tsdecomp- (SSC) or manual approach."
display ""
display "6. No Theta method: Not available natively or via SSC."
display ""
display "7. Prediction intervals: ARIMA prediction intervals require manual"
display "   computation. The -predict- command after -arima- gives point"
display "   forecasts by default."
display ""
display "8. AIC/BIC scale: Stata's -arima- uses conditional MLE by default,"
display "   which can produce different AIC/BIC values compared to R/Python"
display "   exact MLE. Use -arima ..., technique(nr)- or -method(mle)- for"
display "   closer comparisons."
display ""
display "=== Univariate Workflow Complete ==="
display "Output files:"
display "  - univariate_unit_root_tests.csv"
display "  - univariate_hp_filter.csv"
display "  - univariate_models_comparison.csv"
display "  - univariate_forecasts.csv"
