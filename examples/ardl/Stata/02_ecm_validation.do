********************************************************************************
* 02_ecm_validation.do
*
* ECM Validation - Reference implementation in Stata
* Uses the ardl command with ec option for Error Correction Model estimation.
*
* Purpose:  Cross-validate chronobox ECM/ARDL results.
* Dataset:  examples/ardl/data/ardl_synthetic.csv (seed=42, 200 quarterly obs)
* Output:   examples/ardl/outputs/Stata/ecm_results_Stata.csv
*
* Required packages:
*   ssc install ardl     (Demirgunes, 2019 - community ARDL package)
*   Package version used: ardl 1.0.2+ (check with: which ardl)
*
* Built-in post-estimation diagnostics:
*   estat bgodfrey  - Breusch-Godfrey serial correlation test
*   estat archlm    - ARCH-LM heteroskedasticity test
*   estat imtest    - Information matrix test (White's test)
*   predict, resid  - Residuals for normality testing
*
* Tolerance: coefficients < 1e-4; F-statistic < 0.05
********************************************************************************

clear all
set more off
set seed 42

* Display Stata and package version information
display "=== Stata Version ==="
display "Stata version: `c(version)'"
display "Stata edition: `c(edition_real)'"
display ""

* Check if ardl is installed
capture which ardl
if _rc != 0 {
    display as error "Package 'ardl' is not installed."
    display as error "Install with: ssc install ardl"
    display as error "Or: net install ardl, from(http://fmwww.bc.edu/repec/bocode/a/)"
    exit 198
}
display "ardl package: installed"
which ardl

* ==============================================================================
* 1. Load and prepare data
* ==============================================================================

display ""
display "=== Loading Data ==="

import delimited using "../data/ardl_synthetic.csv", clear varnames(1)

describe
summarize y x1 x2 x3

* Parse date and set time series
generate date_stata = date(date, "YMD")
format date_stata %td
generate qdate = qofd(date_stata)
format qdate %tq
tsset qdate

display "Observations: `=_N'"
display "Time range: " %tq qdate[1] " to " %tq qdate[_N]

* ==============================================================================
* 2. ARDL(1,1,1,1) in levels (baseline)
* ==============================================================================

display ""
display "======================================================"
display "=== ARDL(1,1,1,1) in levels (baseline) ==============="
display "======================================================"

ardl y x1 x2 x3, lags(1 1 1 1)
estimates store ardl_levels

* ==============================================================================
* 3. Error Correction Model (ECM) via ardl, ec
* ==============================================================================

display ""
display "======================================================"
display "=== ECM Estimation: ardl with ec option ==============="
display "======================================================"

* The ec option transforms the ARDL into an unrestricted ECM (UECM)
* dy_t = c + pi_yy * y_{t-1} + pi_yx1 * x1_{t-1} + pi_yx2 * x2_{t-1}
*        + pi_yx3 * x3_{t-1} + short-run dynamics + e_t
*
* Speed of adjustment = pi_yy (coefficient on L.y)
* Long-run coefficients = -pi_yxj / pi_yy

ardl y x1 x2 x3, lags(1 1 1 1) ec
estimates store ecm_model

* Store key results
scalar r2_ecm = e(r2)
scalar r2a_ecm = e(r2_a)
scalar nobs_ecm = e(N)
scalar aic_ecm = e(aic)
scalar bic_ecm = e(bic)

display ""
display "ECM fit statistics:"
display "  R-squared: " r2_ecm
display "  Adj R-squared: " r2a_ecm
display "  AIC: " aic_ecm
display "  BIC: " bic_ecm

* ==============================================================================
* 4. Extract speed of adjustment and long-run coefficients
* ==============================================================================

display ""
display "======================================================"
display "=== Speed of Adjustment and Long-run Coefficients ====="
display "======================================================"

* In the ECM form from ardl, ec:
*   L.y       = speed of adjustment (pi_yy), should be negative
*   L.x1      = pi_yx1 (level coefficient on x1)
*   L.x2      = pi_yx2 (level coefficient on x2)
*   L.x3      = pi_yx3 (level coefficient on x3)
*   D.x1      = short-run effect of x1
*   D.x2      = short-run effect of x2
*   D.x3      = short-run effect of x3

scalar ect_coef = _b[L.y]
scalar pi_x1 = _b[L.x1]
scalar pi_x2 = _b[L.x2]
scalar pi_x3 = _b[L.x3]
scalar ecm_const = _b[_cons]

display "Speed of adjustment (pi_yy): " ect_coef
display "  Expected: negative (mean-reverting)"
if ect_coef >= 0 {
    display as error "  WARNING: Non-negative ECT coefficient!"
}

* Half-life of adjustment
scalar half_life = ln(0.5) / ln(1 + ect_coef)
display "Half-life (quarters): " half_life

* Long-run coefficients: theta_j = -pi_yxj / pi_yy
scalar lr_x1 = -pi_x1 / ect_coef
scalar lr_x2 = -pi_x2 / ect_coef
scalar lr_x3 = -pi_x3 / ect_coef
scalar lr_const = -ecm_const / ect_coef

display ""
display "Long-run coefficients (from ECM):"
display "  x1: " lr_x1
display "  x2: " lr_x2
display "  x3: " lr_x3
display "  const: " lr_const

* Short-run coefficients
display ""
display "Short-run coefficients:"
capture display "  D.x1: " _b[D.x1]
capture display "  D.x2: " _b[D.x2]
capture display "  D.x3: " _b[D.x3]

* ==============================================================================
* 5. Comparison with true DGP
* ==============================================================================

display ""
display "======================================================"
display "=== Comparison with True DGP =========================="
display "======================================================"

display "True DGP parameters:"
display "  Long-run x1 = 0.6"
display "  Speed of adjustment = -0.25"
display "  Intercept = 1.5"
display ""
display "Estimated vs True:"
display "  LR x1:  estimated = " lr_x1 "  true = 0.6  error = " abs(lr_x1 - 0.6)
display "  SOA:    estimated = " ect_coef "  true = -0.25  error = " abs(ect_coef - (-0.25))
display "  Const:  estimated = " lr_const "  true = 1.5  error = " abs(lr_const - 1.5)

* ==============================================================================
* 6. Post-estimation diagnostics
* ==============================================================================

display ""
display "======================================================"
display "=== Post-estimation Diagnostics ======================="
display "======================================================"

* Restore ECM estimates for diagnostics
estimates restore ecm_model

* --- 6a. Breusch-Godfrey serial correlation test ---
display ""
display "--- Breusch-Godfrey Serial Correlation Test (4 lags) ---"
estat bgodfrey, lags(1 2 3 4)

* Store BG test results
scalar bg_chi2 = r(chi2)
scalar bg_p = r(p)
scalar bg_df = r(df)
display ""
display "BG chi2(" bg_df "): " bg_chi2
display "BG p-value: " bg_p
if bg_p > 0.05 {
    display "  => No serial correlation at 5% level"
}
else {
    display "  => Evidence of serial correlation at 5% level"
}

* --- 6b. ARCH-LM test for heteroskedasticity ---
display ""
display "--- ARCH-LM Test (4 lags) ---"
estat archlm, lags(1 2 3 4)

scalar arch_chi2 = r(chi2)
scalar arch_p = r(p)
display ""
display "ARCH chi2: " arch_chi2
display "ARCH p-value: " arch_p
if arch_p > 0.05 {
    display "  => No ARCH effects at 5% level"
}
else {
    display "  => Evidence of ARCH effects at 5% level"
}

* --- 6c. Residual normality (Jarque-Bera via sktest) ---
display ""
display "--- Residual Normality Test ---"
predict double resid_ecm, residuals
sktest resid_ecm

scalar jb_chi2 = r(chi2)
scalar jb_p = r(P_chi2)
display ""
display "Normality chi2: " jb_chi2
display "Normality p-value: " jb_p
if jb_p > 0.05 {
    display "  => Residuals are normally distributed at 5% level"
}
else {
    display "  => Residuals deviate from normality at 5% level"
}

* --- 6d. Residual summary statistics ---
display ""
display "--- Residual Summary Statistics ---"
summarize resid_ecm, detail

scalar resid_mean = r(mean)
scalar resid_sd = r(sd)
scalar resid_min = r(min)
scalar resid_max = r(max)
scalar resid_skew = r(skewness)
scalar resid_kurt = r(kurtosis)

display "Residual mean: " resid_mean
display "Residual std:  " resid_sd
display "Residual skew: " resid_skew
display "Residual kurt: " resid_kurt

* ==============================================================================
* 7. Bounds test on ECM
* ==============================================================================

display ""
display "======================================================"
display "=== Bounds Test from ECM =============================="
display "======================================================"

* Perform bounds test
estat btest

scalar f_stat_ecm = r(F)
display "F-statistic (from ECM): " f_stat_ecm

* ==============================================================================
* 8. Export results
* ==============================================================================

display ""
display "======================================================"
display "=== Exporting Results ================================="
display "======================================================"

preserve
clear
set obs 25

generate str30 metric = ""
generate double value = .

* ECM coefficients
replace metric = "speed_of_adjustment" in 1
replace value = ect_coef in 1

replace metric = "half_life_quarters" in 2
replace value = half_life in 2

replace metric = "lr_x1" in 3
replace value = lr_x1 in 3

replace metric = "lr_x2" in 4
replace value = lr_x2 in 4

replace metric = "lr_x3" in 5
replace value = lr_x3 in 5

replace metric = "lr_const" in 6
replace value = lr_const in 6

replace metric = "pi_x1" in 7
replace value = pi_x1 in 7

replace metric = "pi_x2" in 8
replace value = pi_x2 in 8

replace metric = "pi_x3" in 9
replace value = pi_x3 in 9

* Model fit
replace metric = "r_squared" in 10
replace value = r2_ecm in 10

replace metric = "adj_r_squared" in 11
replace value = r2a_ecm in 11

replace metric = "aic" in 12
replace value = aic_ecm in 12

replace metric = "bic" in 13
replace value = bic_ecm in 13

replace metric = "nobs" in 14
replace value = nobs_ecm in 14

* F-test
replace metric = "f_statistic" in 15
replace value = f_stat_ecm in 15

* Diagnostics
replace metric = "bg_chi2" in 16
replace value = bg_chi2 in 16

replace metric = "bg_pvalue" in 17
replace value = bg_p in 17

replace metric = "arch_chi2" in 18
replace value = arch_chi2 in 18

replace metric = "arch_pvalue" in 19
replace value = arch_p in 19

replace metric = "normality_chi2" in 20
replace value = jb_chi2 in 20

replace metric = "normality_pvalue" in 21
replace value = jb_p in 21

replace metric = "resid_mean" in 22
replace value = resid_mean in 22

replace metric = "resid_sd" in 23
replace value = resid_sd in 23

replace metric = "resid_skewness" in 24
replace value = resid_skew in 24

replace metric = "resid_kurtosis" in 25
replace value = resid_kurt in 25

drop if metric == ""

export delimited using "../outputs/Stata/ecm_results_Stata.csv", replace
display "Results saved to: outputs/Stata/ecm_results_Stata.csv"

restore

* ==============================================================================
* 9. Export all ECM coefficients
* ==============================================================================

preserve
clear

estimates restore ecm_model
matrix b = e(b)
matrix V = e(V)
local names : colnames b
local ncols = colsof(b)

set obs `ncols'
generate str30 coefficient = ""
generate double estimate = .
generate double std_error = .
generate double t_stat = .

forvalues i = 1/`ncols' {
    local cname : word `i' of `names'
    replace coefficient = "`cname'" in `i'
    replace estimate = b[1,`i'] in `i'
    replace std_error = sqrt(V[`i',`i']) in `i'
    replace t_stat = b[1,`i'] / sqrt(V[`i',`i']) in `i'
}

export delimited using "../outputs/Stata/ecm_coefficients_Stata.csv", replace
display "ECM coefficients saved to: outputs/Stata/ecm_coefficients_Stata.csv"

restore

display ""
display "=== DONE: 02_ecm_validation.do ==="
