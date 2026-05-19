********************************************************************************
* 01_ardl_bounds_test_validation.do
*
* ARDL Bounds Test Validation - Reference implementation in Stata
* Uses the community-contributed ardl command for ARDL estimation and
* Pesaran-Shin-Smith bounds testing.
*
* Purpose:  Cross-validate chronobox ARDL bounds test results.
* Dataset:  examples/ardl/data/ardl_synthetic.csv (seed=42, 200 quarterly obs)
* Output:   examples/ardl/outputs/Stata/bounds_test_results_Stata.csv
*
* Required packages:
*   ssc install ardl     (Demirgunes, 2019 - community ARDL package)
*   Package version used: ardl 1.0.2+ (check with: which ardl)
*
* Tolerance: F-statistic < 0.05 difference; coefficients < 1e-4
*
* Note on Stata ardl package:
*   The ardl command was written by Tarik Demirgunes (2019).
*   It implements Pesaran, Shin and Smith (2001) bounds testing approach.
*   Citation: Demirgunes, T. (2019). ardl: Stata module to perform ARDL
*   bounds testing for cointegration. SSC Archive.
********************************************************************************

clear all
set more off
set seed 42

* Display Stata and package version information
display "=== Stata Version ==="
display "Stata version: `c(version)'"
display "Stata edition: `c(edition_real)'"
display ""

* Check if ardl is installed; provide install instructions if not
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

* Import CSV data
import delimited using "../data/ardl_synthetic.csv", clear varnames(1)

* Display data overview
describe
summarize y x1 x2 x3

* Parse date and set time series
generate date_stata = date(date, "YMD")
format date_stata %td

* Create quarterly date variable from the date string
* Data is quarterly starting 1970Q1
generate qdate = qofd(date_stata)
format qdate %tq
tsset qdate

display "Observations: `=_N'"
display "Time range: " %tq qdate[1] " to " %tq qdate[_N]

* ==============================================================================
* 2. ARDL with fixed lag specification: ARDL(1,1,1,1)
* ==============================================================================

display ""
display "=============================================="
display "=== ARDL(1,1,1,1) - Matching chronobox spec =="
display "=============================================="

* Estimate ARDL(1,1,1,1) - matching the specification used by chronobox
ardl y x1 x2 x3, lags(1 1 1 1)

* Store results
estimates store ardl_1111
scalar aic_1111 = e(aic)
scalar bic_1111 = e(bic)
scalar nobs_1111 = e(N)
scalar r2_1111 = e(r2)
scalar r2a_1111 = e(r2_a)

display ""
display "AIC: " aic_1111
display "BIC: " bic_1111
display "R-squared: " r2_1111
display "Adj R-squared: " r2a_1111
display "Observations: " nobs_1111

* Extract coefficients
matrix b_1111 = e(b)
matrix list b_1111

* ==============================================================================
* 3. Automatic ARDL order selection (AIC)
* ==============================================================================

display ""
display "=============================================="
display "=== Automatic ARDL Order Selection (AIC) ====="
display "=============================================="

* Use ardl with aic option for automatic lag selection
* maxlags(4) sets maximum lag order to 4 for all variables
ardl y x1 x2 x3, aic maxlags(4)
estimates store ardl_auto_aic

scalar aic_auto = e(aic)
scalar bic_auto_fromaic = e(bic)

display ""
display "Auto-selected model (AIC):"
display "  AIC: " aic_auto
display "  BIC: " bic_auto_fromaic

* ==============================================================================
* 4. Automatic ARDL order selection (BIC)
* ==============================================================================

display ""
display "=============================================="
display "=== Automatic ARDL Order Selection (BIC) ====="
display "=============================================="

ardl y x1 x2 x3, bic maxlags(4)
estimates store ardl_auto_bic

scalar aic_auto_frombic = e(aic)
scalar bic_auto_bic = e(bic)

display ""
display "Auto-selected model (BIC):"
display "  AIC: " aic_auto_frombic
display "  BIC: " bic_auto_bic

* ==============================================================================
* 5. Bounds F-test (Pesaran, Shin and Smith, 2001)
* ==============================================================================

display ""
display "=============================================="
display "=== Bounds F-test (ARDL 1,1,1,1) ============"
display "=============================================="

* Restore the ARDL(1,1,1,1) estimates for bounds test
estimates restore ardl_1111

* Perform Pesaran-Shin-Smith bounds test
* estat btest: F-test for existence of long-run relationship
* Case 3: unrestricted intercept, no trend (most common)
estat btest

* Store F-statistic
scalar f_stat = r(F)
display ""
display "F-statistic: " f_stat

* Extract critical value bands
* r(cv) contains critical values at various significance levels
* Columns: I(0) and I(1) bounds
* Rows: 10%, 5%, 2.5%, 1% significance levels
matrix cv_table = r(cv)
display ""
display "Critical value bands:"
matrix list cv_table

* Store critical values for export
scalar cv_10_I0 = cv_table[1,1]
scalar cv_10_I1 = cv_table[1,2]
scalar cv_05_I0 = cv_table[2,1]
scalar cv_05_I1 = cv_table[2,2]
scalar cv_025_I0 = cv_table[3,1]
scalar cv_025_I1 = cv_table[3,2]
scalar cv_01_I0 = cv_table[4,1]
scalar cv_01_I1 = cv_table[4,2]

* Determine conclusion at 5% level
display ""
display "=== F-test Decision (5% level) ==="
if f_stat > cv_05_I1 {
    display "Conclusion: REJECT null -- evidence of long-run relationship"
    local conclusion_5pct "reject"
}
else if f_stat < cv_05_I0 {
    display "Conclusion: FAIL TO REJECT null -- no cointegration"
    local conclusion_5pct "fail_to_reject"
}
else {
    display "Conclusion: INCONCLUSIVE -- F-stat between I(0) and I(1) bounds"
    local conclusion_5pct "inconclusive"
}

* ==============================================================================
* 6. Long-run coefficients from ARDL(1,1,1,1)
* ==============================================================================

display ""
display "=============================================="
display "=== Long-run Coefficients ===================="
display "=============================================="

* Restore ARDL(1,1,1,1)
estimates restore ardl_1111

* Extract coefficients for long-run calculation
* ARDL(1,1,1,1): y_t = c + phi*y_{t-1} + beta0_x1*x1_t + beta1_x1*x1_{t-1} + ...
* Long-run: theta_j = (beta0_j + beta1_j) / (1 - phi)

* Get coefficient names and values
matrix b = e(b)
matrix list b

* Extract individual coefficients
* Note: coefficient naming depends on ardl package version
* Common pattern: L.y, x1, L.x1, x2, L.x2, x3, L.x3, _cons
scalar phi_1 = _b[L.y]
scalar b0_x1 = _b[x1]
scalar b1_x1 = _b[L.x1]
scalar b0_x2 = _b[x2]
scalar b1_x2 = _b[L.x2]
scalar b0_x3 = _b[x3]
scalar b1_x3 = _b[L.x3]
scalar intercept = _b[_cons]

display "phi (L.y) = " phi_1
display "x1 = " b0_x1 ", L.x1 = " b1_x1
display "x2 = " b0_x2 ", L.x2 = " b1_x2
display "x3 = " b0_x3 ", L.x3 = " b1_x3
display "intercept = " intercept

* Compute long-run multipliers
scalar lr_x1 = (b0_x1 + b1_x1) / (1 - phi_1)
scalar lr_x2 = (b0_x2 + b1_x2) / (1 - phi_1)
scalar lr_x3 = (b0_x3 + b1_x3) / (1 - phi_1)
scalar lr_const = intercept / (1 - phi_1)

display ""
display "Long-run multipliers:"
display "  x1: " lr_x1
display "  x2: " lr_x2
display "  x3: " lr_x3
display "  const: " lr_const

* ==============================================================================
* 7. Export results
* ==============================================================================

display ""
display "=============================================="
display "=== Exporting Results ========================"
display "=============================================="

* Export bounds test results to CSV
preserve
clear
set obs 1

generate str20 metric = ""
generate double value = .

* Expand to hold all results
set obs 20

* F-test results
replace metric = "f_statistic" in 1
replace value = f_stat in 1

replace metric = "cv_10pct_I0" in 2
replace value = cv_10_I0 in 2

replace metric = "cv_10pct_I1" in 3
replace value = cv_10_I1 in 3

replace metric = "cv_05pct_I0" in 4
replace value = cv_05_I0 in 4

replace metric = "cv_05pct_I1" in 5
replace value = cv_05_I1 in 5

replace metric = "cv_025pct_I0" in 6
replace value = cv_025_I0 in 6

replace metric = "cv_025pct_I1" in 7
replace value = cv_025_I1 in 7

replace metric = "cv_01pct_I0" in 8
replace value = cv_01_I0 in 8

replace metric = "cv_01pct_I1" in 9
replace value = cv_01_I1 in 9

* Long-run coefficients
replace metric = "lr_x1" in 10
replace value = lr_x1 in 10

replace metric = "lr_x2" in 11
replace value = lr_x2 in 11

replace metric = "lr_x3" in 12
replace value = lr_x3 in 12

replace metric = "lr_const" in 13
replace value = lr_const in 13

* Model fit
replace metric = "r_squared" in 14
replace value = r2_1111 in 14

replace metric = "adj_r_squared" in 15
replace value = r2a_1111 in 15

replace metric = "aic" in 16
replace value = aic_1111 in 16

replace metric = "bic" in 17
replace value = bic_1111 in 17

replace metric = "nobs" in 18
replace value = nobs_1111 in 18

* Short-run coefficients
replace metric = "phi_Ly" in 19
replace value = phi_1 in 19

replace metric = "intercept" in 20
replace value = intercept in 20

* Drop empty rows
drop if metric == ""

export delimited using "../outputs/Stata/bounds_test_results_Stata.csv", replace
display "Results saved to: outputs/Stata/bounds_test_results_Stata.csv"

restore

* ==============================================================================
* 8. Export raw coefficients
* ==============================================================================

preserve
clear

estimates restore ardl_1111
matrix b = e(b)
local names : colnames b
local ncols = colsof(b)

set obs `ncols'
generate str30 coefficient = ""
generate double estimate = .

forvalues i = 1/`ncols' {
    local cname : word `i' of `names'
    replace coefficient = "`cname'" in `i'
    replace estimate = b[1,`i'] in `i'
}

export delimited using "../outputs/Stata/ardl_coefficients_Stata.csv", replace
display "Coefficients saved to: outputs/Stata/ardl_coefficients_Stata.csv"

restore

display ""
display "=== DONE: 01_ardl_bounds_test_validation.do ==="
