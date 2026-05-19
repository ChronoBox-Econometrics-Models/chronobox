********************************************************************************
* 02_multivariate_workflow_validation.do
* Complete multivariate time series workflow in Stata for cross-validation
* with chronobox (Python) and R.
*
* Pipeline:
*   1. Load US macro quarterly dataset (gdp, inflation, fed_funds, unemployment)
*   2. Unit root tests: dfuller for each variable
*   3. Johansen cointegration test: vecrank
*   4. VAR estimation: var
*   5. VEC estimation: vec
*   6. Structural VAR: svar
*   7. IRF and FEVD: irf create, irf table
*   8. Granger causality: vargranger
*   9. Export results
*
* Dataset: us_macro_quarterly.csv (quarterly, 1975Q1 onwards)
* Seed: set seed 42
*
* LIMITATIONS DOCUMENTED:
*   - Stata's VAR lag selection uses varsoc (not VARselect as in R).
*   - Stata SVAR requires explicit constraint matrices A and B.
*   - Stata does not have a native Bayesian VAR (BVAR) command. The -bvar-
*     package from SSC is limited compared to R's BVAR or Python's implementations.
*   - Spillover index (Diebold-Yilmaz) is not a native command. Must be computed
*     manually from FEVD output.
*   - ARDL is available via -ardl- (SSC) but syntax differs from R/Python.
*   - Rolling-window spillover requires manual looping.
*   - Stata's irf table exports to .irf files, not directly to CSV. Requires
*     post-processing to get CSV output.
********************************************************************************

clear all
set more off
set seed 42

* ==============================================================================
* 0. Setup paths
* ==============================================================================
capture confirm file "../data/us_macro_quarterly.csv"
if _rc == 0 {
    local base_dir ".."
}
else {
    capture confirm file "examples/complete_workflow/data/us_macro_quarterly.csv"
    if _rc == 0 {
        local base_dir "examples/complete_workflow"
    }
    else {
        local base_dir ".."
    }
}

local data_dir    "`base_dir'/data"
local output_dir  "`base_dir'/outputs/Stata"
capture mkdir "`output_dir'"

display "=== Multivariate Workflow Validation (Stata) ==="
display "Data dir  : `data_dir'"
display "Output dir: `output_dir'"
display ""

* ==============================================================================
* 1. Load data
* ==============================================================================
import delimited "`data_dir'/us_macro_quarterly.csv", clear

* Parse date and set quarterly time series
generate date_stata = date(date, "YMD")
format date_stata %td
generate qtr_id = qofd(date_stata)
format qtr_id %tq
tsset qtr_id

local n_obs = _N
local var_names "gdp inflation fed_funds unemployment"
display "Loaded US macro dataset: `n_obs' observations, 4 variables"
display "Variables: `var_names'"

* ==============================================================================
* 2. Unit root tests (ADF for each variable)
* ==============================================================================
display ""
display "--- Unit Root Tests ---"

* We store results in a temporary file
tempname memhold
tempfile unit_root_file

postfile `memhold' str20 variable str10 test double statistic double pvalue ///
    str15 conclusion using `unit_root_file', replace

foreach v of local var_names {
    * ADF test on levels
    quietly dfuller `v', lags(4)
    local adf_stat = r(Zt)
    local adf_pval = r(p)
    local adf_concl = cond(`adf_pval' < 0.05, "Stationary", "Non-stationary")

    display "  `v': ADF stat=`adf_stat' (p=`adf_pval') => `adf_concl'"

    post `memhold' ("`v'") ("adf_level") (`adf_stat') (`adf_pval') ("`adf_concl'")

    * KPSS test (if installed)
    capture quietly kpss `v', maxlag(4)
    if _rc == 0 {
        local kpss_stat = r(kpss)
        display "  `v': KPSS stat=`kpss_stat'"
        post `memhold' ("`v'") ("kpss_level") (`kpss_stat') (.) ("See crit. val.")
    }
}

postclose `memhold'

* Save unit root results
preserve
use `unit_root_file', clear
export delimited "`output_dir'/multivariate_unit_root_tests.csv", replace
display "Saved: multivariate_unit_root_tests.csv"
restore

* ==============================================================================
* 3. Johansen cointegration test (vecrank)
* ==============================================================================
display ""
display "--- Johansen Cointegration Test ---"

* vecrank performs the Johansen trace and max-eigenvalue tests
* Using lag(2) for K=2 as in the R script (VAR with 1 lag => VEC with K-1=1
* lag of differences, but vecrank's lag option refers to the VAR level lag)
vecrank gdp inflation fed_funds unemployment, lags(2) trend(constant)

* NOTE: vecrank displays the trace statistic and critical values.
* The cointegration rank is determined by the number of rejections.
* Stata does not store the rank in r() by default; users read the table.
* We capture the trace statistics from the output.

* Store vecrank results
local trace_stat_r0 = r(trace_0)
local trace_stat_r1 = r(trace_1)
local trace_stat_r2 = r(trace_2)
local trace_stat_r3 = r(trace_3)

display "Trace statistics:"
display "  r=0: `trace_stat_r0'"
display "  r=1: `trace_stat_r1'"
display "  r=2: `trace_stat_r2'"
display "  r=3: `trace_stat_r3'"

* Save cointegration results
preserve
clear
set obs 4
generate str10 null_hypothesis = ""
generate double trace_statistic = .
generate str30 note = ""

replace null_hypothesis = "r=0" in 1
replace trace_statistic = `trace_stat_r0' in 1

replace null_hypothesis = "r<=1" in 2
replace trace_statistic = `trace_stat_r1' in 2

replace null_hypothesis = "r<=2" in 3
replace trace_statistic = `trace_stat_r2' in 3

replace null_hypothesis = "r<=3" in 4
replace trace_statistic = `trace_stat_r3' in 4

replace note = "Johansen trace test, lags(2), trend(constant)" in 1

export delimited "`output_dir'/multivariate_cointegration.csv", replace
display "Saved: multivariate_cointegration.csv"
restore

* ==============================================================================
* 4. VAR estimation
* ==============================================================================
display ""
display "--- VAR Model Selection and Estimation ---"

* Lag selection using varsoc
varsoc gdp inflation fed_funds unemployment, maxlag(8)

* Estimate VAR(1) for consistency with Python/R
var gdp inflation fed_funds unemployment, lags(1/1)

* Display information criteria
estat ic

* Store model results
matrix ic_var = r(S)
local var_aic = ic_var[1,5]
local var_bic = ic_var[1,6]
display "VAR(1) AIC = `var_aic', BIC = `var_bic'"

* ==============================================================================
* 5. Granger causality
* ==============================================================================
display ""
display "--- Granger Causality ---"

vargranger

* Save Granger causality results
* NOTE: vargranger does not store results in r() or e() in a structured way.
* We capture key statistics from the output.
* Limitation: Programmatic extraction requires parsing logs.

* ==============================================================================
* 6. VEC estimation (if cointegration found)
* ==============================================================================
display ""
display "--- VEC Estimation ---"
display "NOTE: Estimating VEC with rank=1 (assuming at least 1 cointegrating relation)"

* vec estimates a vector error-correction model
vec gdp inflation fed_funds unemployment, lags(2) rank(1)

display "VEC model estimated with rank=1"

* ==============================================================================
* 7. Structural VAR (SVAR)
* ==============================================================================
display ""
display "--- Structural VAR ---"

* Re-estimate reduced-form VAR for SVAR
quietly var gdp inflation fed_funds unemployment, lags(1/1)

* SVAR with short-run (Cholesky) restrictions
* In Stata, svar requires specifying constraint matrices.
* For Cholesky decomposition, we use a lower-triangular A matrix and diagonal B.

* Define A matrix (lower triangular, diagonal = 1)
matrix A = (1,0,0,0 \ .,1,0,0 \ .,.,1,0 \ .,.,.,1)
matrix B = (.,0,0,0 \ 0,.,0,0 \ 0,0,.,0 \ 0,0,0,.)

svar gdp inflation fed_funds unemployment, lags(1/1) aeq(A) beq(B)

display "SVAR estimated with Cholesky decomposition (lower-triangular A)"

* ==============================================================================
* 8. IRF and FEVD
* ==============================================================================
display ""
display "--- IRF and FEVD ---"

* Create IRF results (10 periods ahead, orthogonalized = structural)
irf create svar_irf, set(svar_results) step(10) replace

* Display IRF table
irf table oirf, irf(svar_irf) noci

* Display FEVD table
irf table fevd, irf(svar_irf) noci

* ==============================================================================
* 9. Extract and save IRF to CSV
* ==============================================================================
display ""
display "--- Extracting IRF to CSV ---"

* Load the IRF dataset to export
preserve
use svar_results.irf, clear

* Keep only the orthogonalized IRF columns
* The IRF dataset has columns: step, irfname, impulse, response, oirf, ...
keep step impulse response oirf fevd

rename step period
rename oirf var_irf

export delimited "`output_dir'/multivariate_irf.csv", replace
display "Saved: multivariate_irf.csv"
restore

* ==============================================================================
* 10. FEVD at horizon 10 and Spillover Index
* ==============================================================================
display ""
display "--- FEVD and Spillover Index ---"

* Load IRF dataset to compute spillover from FEVD
preserve
use svar_results.irf, clear

* Keep FEVD at step 10
keep if step == 10
keep impulse response fevd

* Reshape to matrix form for spillover computation
display "FEVD at horizon 10:"
list impulse response fevd, noobs

* Compute spillover index manually
* Total spillover = (sum of off-diagonal FEVD / N) * 100
* NOTE: This is a simplified Diebold-Yilmaz computation
local n_var = 4
local total_fevd = 0
local own_fevd = 0

* Sum all FEVD values
quietly summarize fevd
local total_fevd = r(sum)

* Sum own (diagonal) FEVD values
local own_fevd = 0
foreach v in gdp inflation fed_funds unemployment {
    quietly summarize fevd if impulse == "`v'" & response == "`v'"
    local own_fevd = `own_fevd' + r(mean)
}

local total_spillover = (`total_fevd' - `own_fevd') / `n_var' * 100
display ""
display "Total spillover index: `total_spillover'%"
display "  (Sum FEVD = `total_fevd', Own FEVD = `own_fevd')"

* Save FEVD matrix
export delimited "`output_dir'/multivariate_fevd_h10.csv", replace
display "Saved: multivariate_fevd_h10.csv"
restore

* Save spillover results
preserve
clear
set obs 1
generate double total_spillover = `total_spillover'
generate int var_lags = 1
generate int horizon = 10
generate str50 method = "Diebold-Yilmaz from Cholesky FEVD"
generate str50 variables = "gdp,inflation,fed_funds,unemployment"

export delimited "`output_dir'/multivariate_spillover.csv", replace
display "Saved: multivariate_spillover.csv"
restore

* ==============================================================================
* 11. VAR Forecasts (8 quarters ahead)
* ==============================================================================
display ""
display "--- VAR Forecasts ---"

* Re-estimate VAR on full sample
quietly var gdp inflation fed_funds unemployment, lags(1/1)

* Extend dataset
local last_qtr = qtr_id[_N]
local h_forecast = 8
local new_n = _N + `h_forecast'
set obs `new_n'

forvalues i = 1/`h_forecast' {
    local row = `n_obs' + `i'
    quietly replace qtr_id = `last_qtr' + `i' in `row'
}
tsset qtr_id

* Dynamic forecast
local first_oos = `last_qtr' + 1
foreach v in gdp inflation fed_funds unemployment {
    predict fc_`v', equation(`v') dynamic(`first_oos')
}

* Save forecasts
preserve
keep if _n > `n_obs'
generate date_fc = dofq(qtr_id)
format date_fc %td

* Reshape to long format for comparison
stack fc_gdp fc_inflation fc_fed_funds fc_unemployment, ///
    into(var_forecast) wide clear
* NOTE: stack may not produce the desired format. Use manual approach instead.
restore

* Manual approach for forecast export
preserve
keep if _n > `n_obs'
generate date_fc = dofq(qtr_id)
format date_fc %td

* Export wide format
keep date_fc fc_gdp fc_inflation fc_fed_funds fc_unemployment
rename fc_gdp gdp_forecast
rename fc_inflation inflation_forecast
rename fc_fed_funds fed_funds_forecast
rename fc_unemployment unemployment_forecast

export delimited "`output_dir'/multivariate_forecasts.csv", replace
display "Saved: multivariate_forecasts.csv"
restore

* ==============================================================================
* 12. Cleanup temporary IRF files
* ==============================================================================
capture erase svar_results.irf

* ==============================================================================
* 13. Documentation of Stata limitations for multivariate analysis
* ==============================================================================
display ""
display "=== STATA LIMITATIONS: Multivariate Analysis ==="
display ""
display "1. NO Bayesian VAR (BVAR) native command: Stata has no built-in"
display "   BVAR estimation. The SSC -bvar- package (if available) has limited"
display "   features compared to R's BVAR package or Python's implementations."
display "   Minnesota priors, hierarchical priors not natively supported."
display ""
display "2. Spillover index not native: Diebold-Yilmaz spillover must be"
display "   computed manually from FEVD output. No -spillover- command exists."
display "   Rolling-window spillover requires manual looping over subsamples."
display ""
display "3. SVAR constraints: Stata's -svar- requires explicit A and B matrices"
display "   with specific constraint syntax. This is more verbose than R's"
display "   vars::SVAR() or Python's chronobox SVAR interface."
display ""
display "4. IRF export: Stata stores IRF results in .irf binary files."
display "   Converting to CSV requires opening the .irf file as a dataset and"
display "   exporting. Less convenient than R/Python direct matrix output."
display ""
display "5. vecrank programmatic access: The Johansen test results from"
display "   -vecrank- are displayed but not all values are stored in r()."
display "   Programmatic extraction of the rank can require parsing logs."
display ""
display "6. vargranger output: Results are displayed but not stored in a"
display "   structured matrix. Programmatic comparison with R/Python requires"
display "   log parsing or manual recording."
display ""
display "7. Forecast intervals: VAR forecast confidence intervals require"
display "   additional computation. The -fcast compute- command provides this"
display "   but with different syntax than R/Python."
display ""
display "8. No ARDL native bounds test: While -ardl- exists on SSC, the"
display "   Pesaran bounds test implementation may differ from R's dynamac"
display "   or Python's chronobox."
display ""
display "=== Multivariate Workflow Complete ==="
display "Output files:"
display "  - multivariate_unit_root_tests.csv"
display "  - multivariate_cointegration.csv"
display "  - multivariate_irf.csv"
display "  - multivariate_fevd_h10.csv"
display "  - multivariate_spillover.csv"
display "  - multivariate_forecasts.csv"
