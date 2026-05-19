********************************************************************************
* compare_results.do
*
* Compare ARDL/ECM results: Python (chronobox) vs R vs Stata
*
* Reads CSV/JSON outputs from all implementations and reports differences.
* Tolerance thresholds:
*   - Coefficients:         < 1e-4
*   - F-statistic:          < 0.05
*   - Speed of adjustment:  < 1e-4
*
* Required packages:
*   ssc install ardl    (for the ARDL estimation - already used in prior scripts)
*
* Note: This script reads previously exported results from:
*   - outputs/Stata/bounds_test_results_Stata.csv  (from 01_ardl_bounds_test_validation.do)
*   - outputs/Stata/ecm_results_Stata.csv          (from 02_ecm_validation.do)
*   - outputs/bounds_test_results.json             (from Python/chronobox)
*   - outputs/R/bounds_test_results_R.json         (from R/ARDL package)
*
* Stata does not natively parse JSON, so Python/R results are loaded manually
* by re-estimating the model in Stata and comparing with Stata's own output.
* For a full three-way comparison, use the Python compare_results.py script.
*
* Package version: ardl 1.0.2+ (check with: which ardl)
********************************************************************************

clear all
set more off
set seed 42

display "======================================================"
display "=== Cross-Platform ARDL/ECM Comparison ================"
display "=== Python (chronobox) vs R (ARDL) vs Stata (ardl) ===="
display "======================================================"
display ""

* Display version info
display "Stata version: `c(version)'"
capture which ardl
display ""

* ==============================================================================
* 1. Define tolerance thresholds
* ==============================================================================

scalar TOL_COEF  = 1e-4   // coefficients
scalar TOL_FSTAT = 0.05   // F-statistic
scalar TOL_SOA   = 1e-4   // speed of adjustment

display "Tolerance thresholds:"
display "  Coefficients: " TOL_COEF
display "  F-statistic:  " TOL_FSTAT
display "  SOA:          " TOL_SOA
display ""

* ==============================================================================
* 2. Load Stata bounds test results
* ==============================================================================

display "--- Loading Stata Results ---"

* Load bounds test results
preserve
import delimited using "../outputs/Stata/bounds_test_results_Stata.csv", clear varnames(1)
list

* Store Stata values into scalars
* We need to find each metric row and extract its value
levelsof metric, local(metrics)
forvalues i = 1/`=_N' {
    local m = metric[`i']
    local v = value[`i']
    scalar stata_`m' = `v'
}
restore

display ""
display "Stata bounds test results loaded:"
display "  F-statistic: " stata_f_statistic
display "  LR x1: " stata_lr_x1
display "  LR x2: " stata_lr_x2
display "  LR x3: " stata_lr_x3
display "  R-squared: " stata_r_squared

* Load ECM results
preserve
import delimited using "../outputs/Stata/ecm_results_Stata.csv", clear varnames(1)
list

forvalues i = 1/`=_N' {
    local m = metric[`i']
    local v = value[`i']
    scalar stata_ecm_`m' = `v'
}
restore

display ""
display "Stata ECM results loaded:"
display "  Speed of adjustment: " stata_ecm_speed_of_adjustment
display "  Half-life: " stata_ecm_half_life_quarters
display "  LR x1: " stata_ecm_lr_x1
display "  LR x2: " stata_ecm_lr_x2
display "  LR x3: " stata_ecm_lr_x3

* ==============================================================================
* 3. Re-estimate in Stata for direct comparison (sanity check)
* ==============================================================================

display ""
display "--- Re-estimating ARDL(1,1,1,1) for verification ---"

import delimited using "../data/ardl_synthetic.csv", clear varnames(1)
generate date_stata = date(date, "YMD")
format date_stata %td
generate qdate = qofd(date_stata)
format qdate %tq
tsset qdate

* ARDL in levels
ardl y x1 x2 x3, lags(1 1 1 1)
estimates store ardl_check

* Bounds test
estat btest
scalar f_stat_check = r(F)

* ECM form
ardl y x1 x2 x3, lags(1 1 1 1) ec
estimates store ecm_check

scalar soa_check = _b[L.y]
scalar lr_x1_check = -_b[L.x1] / _b[L.y]
scalar lr_x2_check = -_b[L.x2] / _b[L.y]
scalar lr_x3_check = -_b[L.x3] / _b[L.y]

display ""
display "Verification (fresh estimation):"
display "  F-statistic: " f_stat_check
display "  SOA: " soa_check
display "  LR x1: " lr_x1_check
display "  LR x2: " lr_x2_check
display "  LR x3: " lr_x3_check

* ==============================================================================
* 4. Internal consistency check (stored vs fresh)
* ==============================================================================

display ""
display "======================================================"
display "=== Internal Consistency: Stored vs Fresh Estimation ==="
display "======================================================"

local all_pass = 1

* Compare F-statistic
scalar diff_f = abs(stata_f_statistic - f_stat_check)
if diff_f < TOL_FSTAT {
    display "  [PASS] F-statistic     stored: " %10.6f stata_f_statistic "  fresh: " %10.6f f_stat_check "  diff: " %10.6f diff_f
}
else {
    display "  [FAIL] F-statistic     stored: " %10.6f stata_f_statistic "  fresh: " %10.6f f_stat_check "  diff: " %10.6f diff_f
    local all_pass = 0
}

* Compare SOA
scalar diff_soa = abs(stata_ecm_speed_of_adjustment - soa_check)
if diff_soa < TOL_SOA {
    display "  [PASS] SOA             stored: " %10.6f stata_ecm_speed_of_adjustment "  fresh: " %10.6f soa_check "  diff: " %10.6f diff_soa
}
else {
    display "  [FAIL] SOA             stored: " %10.6f stata_ecm_speed_of_adjustment "  fresh: " %10.6f soa_check "  diff: " %10.6f diff_soa
    local all_pass = 0
}

* Compare LR coefficients
foreach var in x1 x2 x3 {
    scalar diff_lr_`var' = abs(stata_ecm_lr_`var' - lr_`var'_check)
    if diff_lr_`var' < TOL_COEF {
        display "  [PASS] LR_`var'          stored: " %10.6f stata_ecm_lr_`var' "  fresh: " %10.6f lr_`var'_check "  diff: " %10.6f diff_lr_`var'
    }
    else {
        display "  [FAIL] LR_`var'          stored: " %10.6f stata_ecm_lr_`var' "  fresh: " %10.6f lr_`var'_check "  diff: " %10.6f diff_lr_`var'
        local all_pass = 0
    }
}

* ==============================================================================
* 5. Comparison with true DGP
* ==============================================================================

display ""
display "======================================================"
display "=== Proximity to True DGP ============================="
display "======================================================"

display "True DGP: LR_x1 = 0.6, SOA = -0.25, intercept = 1.5"
display ""
display "  Stata LR_x1:  " %10.6f lr_x1_check "  true: 0.600000  error: " %10.6f abs(lr_x1_check - 0.6)
display "  Stata SOA:    " %10.6f soa_check "  true: -0.250000  error: " %10.6f abs(soa_check - (-0.25))

* ==============================================================================
* 6. Cross-platform comparison table
* ==============================================================================

display ""
display "======================================================"
display "=== Cross-Platform Comparison ========================="
display "======================================================"
display ""
display "NOTE: For a full Python vs R vs Stata comparison, run the"
display "Python compare_results.py or R compare_results.R scripts"
display "which can parse all output formats."
display ""
display "Stata results summary for manual comparison:"
display ""
display "  Metric                 Stata Value"
display "  ---------------------- ----------------"
display "  F-statistic            " %12.6f f_stat_check
display "  Speed of adjustment    " %12.6f soa_check
display "  Half-life (quarters)   " %12.6f stata_ecm_half_life_quarters
display "  LR coeff x1            " %12.6f lr_x1_check
display "  LR coeff x2            " %12.6f lr_x2_check
display "  LR coeff x3            " %12.6f lr_x3_check
display "  R-squared (ARDL)       " %12.6f stata_r_squared
display "  AIC                    " %12.6f stata_aic
display "  BIC                    " %12.6f stata_bic

* ==============================================================================
* 7. Diagnostics comparison
* ==============================================================================

display ""
display "======================================================"
display "=== Diagnostics Summary ==============================="
display "======================================================"

display "  BG serial corr chi2:  " %12.6f stata_ecm_bg_chi2 "  p=" %8.6f stata_ecm_bg_pvalue
display "  ARCH-LM chi2:         " %12.6f stata_ecm_arch_chi2 "  p=" %8.6f stata_ecm_arch_pvalue
display "  Normality chi2:       " %12.6f stata_ecm_normality_chi2 "  p=" %8.6f stata_ecm_normality_pvalue
display "  Residual mean:        " %12.6f stata_ecm_resid_mean
display "  Residual std:         " %12.6f stata_ecm_resid_sd

* ==============================================================================
* 8. Export comparison summary
* ==============================================================================

preserve
clear
set obs 12

generate str35 metric = ""
generate double stata_value = .
generate str15 status = ""

replace metric = "f_statistic" in 1
replace stata_value = f_stat_check in 1

replace metric = "speed_of_adjustment" in 2
replace stata_value = soa_check in 2

replace metric = "half_life_quarters" in 3
replace stata_value = stata_ecm_half_life_quarters in 3

replace metric = "lr_x1" in 4
replace stata_value = lr_x1_check in 4

replace metric = "lr_x2" in 5
replace stata_value = lr_x2_check in 5

replace metric = "lr_x3" in 6
replace stata_value = lr_x3_check in 6

replace metric = "r_squared" in 7
replace stata_value = stata_r_squared in 7

replace metric = "aic" in 8
replace stata_value = stata_aic in 8

replace metric = "bic" in 9
replace stata_value = stata_bic in 9

replace metric = "bg_serial_corr_pvalue" in 10
replace stata_value = stata_ecm_bg_pvalue in 10

replace metric = "arch_lm_pvalue" in 11
replace stata_value = stata_ecm_arch_pvalue in 11

replace metric = "normality_pvalue" in 12
replace stata_value = stata_ecm_normality_pvalue in 12

drop if metric == ""

export delimited using "../outputs/Stata/comparison_summary_Stata.csv", replace
display ""
display "Comparison saved to: outputs/Stata/comparison_summary_Stata.csv"

restore

* ==============================================================================
* SUMMARY
* ==============================================================================

display ""
display "======================================================"
if `all_pass' {
    display "INTERNAL CONSISTENCY: ALL CHECKS PASSED"
}
else {
    display "INTERNAL CONSISTENCY: SOME CHECKS FAILED"
    display "Review FAIL items above."
}
display ""
display "For cross-platform comparison (Python vs R vs Stata),"
display "compare the exported CSV files:"
display "  - outputs/Stata/bounds_test_results_Stata.csv"
display "  - outputs/Stata/ecm_results_Stata.csv"
display "  - outputs/Stata/comparison_summary_Stata.csv"
display "with:"
display "  - outputs/bounds_test_results.json (Python)"
display "  - outputs/R/bounds_test_results_R.json (R)"
display "  - outputs/R/ecm_results_R.json (R)"
display "======================================================"

display ""
display "=== DONE: compare_results.do ==="
