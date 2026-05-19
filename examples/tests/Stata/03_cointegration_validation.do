********************************************************************************
* 03_cointegration_validation.do
* Cross-validation of cointegration tests
* Engle-Granger: regress + dfuller on residuals (manual two-step)
* Johansen: vecrank (built-in)
*
* Compares results with Python chronobox and R (urca) outputs.
* Uses GDP datasets loaded from CSV (identical across platforms).
*
* SSC packages required: none (vecrank, regress, dfuller are built-in)
*
* Note: Stata's vecrank implements the Johansen (1988, 1995) trace and
* max-eigenvalue tests. Engle-Granger is implemented manually via OLS
* regression followed by ADF test on residuals.
********************************************************************************

clear all
set more off
set seed 42

* --- Configuration ---
local data_dir "/home/guhaase/projetos/chronobox/examples/tests/data"
local out_dir  "/home/guhaase/projetos/chronobox/examples/tests/outputs/Stata"

capture mkdir "`out_dir'"

display _newline
display "======================================================================"
display "Cointegration Tests Validation (Stata)"
display "======================================================================"
display _newline

* --- Initialize output file ---
local outfile "`out_dir'/cointegration_results.csv"
capture file close fout
file open fout using "`outfile'", write replace
file write fout "test,system,type,statistic,cv_5pct,rank,reject_at_5pct,decision" _n

********************************************************************************
* 1. Load and merge GDP data
********************************************************************************
display "--- Loading GDP data ---"
display _newline

* Load US GDP
import delimited "`data_dir'/us_gdp_quarterly.csv", clear
generate date_q = quarterly(substr(date, 1, 7), "YQ")
format date_q %tq
rename log_gdp log_gdp_us
rename gdp_growth gdp_growth_us
rename gdp_real gdp_real_us
keep date_q log_gdp_us gdp_growth_us
tempfile us_data
save `us_data', replace

* Load Brazil GDP
import delimited "`data_dir'/brazil_gdp.csv", clear
generate date_q = quarterly(substr(date, 1, 7), "YQ")
format date_q %tq
rename log_gdp log_gdp_br
rename gdp_growth gdp_growth_br
rename gdp_real gdp_real_br
keep date_q log_gdp_br gdp_growth_br

* Merge on date
merge 1:1 date_q using `us_data', nogenerate keep(match)
tsset date_q

display "  Merged GDP data: " _N " observations"
display _newline

********************************************************************************
* 2. Integration check (prerequisite)
********************************************************************************
display "--- Integration Check ---"
display _newline
display "  Both series must be I(1) for cointegration analysis."
display _newline

* ADF on US GDP level
display "  ADF on US GDP level (drift):"
dfuller log_gdp_us, drift lags(4)
local stat_us_level = r(Zt)
local pval_us_level = r(p)
display "    stat = " %9.4f `stat_us_level' ", p = " %9.6f `pval_us_level'

* ADF on US GDP first difference
generate dlog_us = D.log_gdp_us
display "  ADF on US GDP 1st diff (drift):"
dfuller dlog_us, drift lags(4)
local stat_us_diff = r(Zt)
local pval_us_diff = r(p)
display "    stat = " %9.4f `stat_us_diff' ", p = " %9.6f `pval_us_diff'
display _newline

* ADF on Brazil GDP level
display "  ADF on Brazil GDP level (drift):"
dfuller log_gdp_br, drift lags(4)
local stat_br_level = r(Zt)
local pval_br_level = r(p)
display "    stat = " %9.4f `stat_br_level' ", p = " %9.6f `pval_br_level'

* ADF on Brazil GDP first difference
generate dlog_br = D.log_gdp_br
display "  ADF on Brazil GDP 1st diff (drift):"
dfuller dlog_br, drift lags(4)
local stat_br_diff = r(Zt)
local pval_br_diff = r(p)
display "    stat = " %9.4f `stat_br_diff' ", p = " %9.6f `pval_br_diff'
display _newline

display "  Integration check results:"
display "    US GDP level: p = " %9.6f `pval_us_level' " => " ///
    cond(`pval_us_level' < 0.05, "stationary (unexpected)", "unit root (OK)")
display "    US GDP diff:  p = " %9.6f `pval_us_diff'  " => " ///
    cond(`pval_us_diff' < 0.05, "stationary (OK)", "unit root (unexpected)")
display "    BR GDP level: p = " %9.6f `pval_br_level' " => " ///
    cond(`pval_br_level' < 0.05, "stationary (unexpected)", "unit root (OK)")
display "    BR GDP diff:  p = " %9.6f `pval_br_diff'  " => " ///
    cond(`pval_br_diff' < 0.05, "stationary (OK)", "unit root (unexpected)")
display _newline

file write fout "Integration,Log PIB EUA (nivel),ADF," %12.6f (`stat_us_level') ",,," ///
    cond(`pval_us_level' >= 0.05, "1", "0") ",I(1) check" _n
file write fout "Integration,Log PIB EUA (1a diff),ADF," %12.6f (`stat_us_diff') ",,," ///
    cond(`pval_us_diff' < 0.05, "1", "0") ",I(0) check" _n
file write fout "Integration,Log PIB Brasil (nivel),ADF," %12.6f (`stat_br_level') ",,," ///
    cond(`pval_br_level' >= 0.05, "1", "0") ",I(1) check" _n
file write fout "Integration,Log PIB Brasil (1a diff),ADF," %12.6f (`stat_br_diff') ",,," ///
    cond(`pval_br_diff' < 0.05, "1", "0") ",I(0) check" _n

********************************************************************************
* 3. Engle-Granger Cointegration Test (manual two-step)
********************************************************************************
display "--- Engle-Granger Cointegration Test (manual) ---"
display _newline
display "  Step 1: OLS regression y = a + b*x + e"
display "  Step 2: ADF test on residuals e"
display "  H0: no cointegration (residuals have unit root)"
display _newline

* --- EG Test: Brazil GDP ~ US GDP ---
display "  Engle-Granger: Log PIB Brasil ~ Log PIB EUA"
display _newline

* Step 1: Cointegrating regression
display "  Step 1: OLS regression"
regress log_gdp_br log_gdp_us
local eg_beta = _b[log_gdp_us]
local eg_const = _b[_cons]
local eg_r2 = e(r2)
display "    beta = " %9.4f `eg_beta' ", const = " %9.4f `eg_const' ", R2 = " %9.4f `eg_r2'
display _newline

* Get residuals
predict eg_resid, residuals

* Step 2: ADF test on residuals
display "  Step 2: ADF on residuals (no constant — residuals have zero mean)"
dfuller eg_resid, noconstant lags(4)
local eg_stat = r(Zt)
local eg_pval = r(p)
display "    ADF(resid) stat = " %9.4f `eg_stat' ", p = " %9.6f `eg_pval'
display _newline

* Engle-Granger critical values (MacKinnon 1996, 2 variables with constant)
* 1%: -3.96, 5%: -3.37, 10%: -3.07
local eg_cv1 = -3.96
local eg_cv5 = -3.37
local eg_cv10 = -3.07
local eg_reject = cond(`eg_stat' < `eg_cv5', 1, 0)
local eg_decision = cond(`eg_reject', "Cointegrated", "Not cointegrated")

display "  Engle-Granger critical values (MacKinnon, 2 vars):"
display "    1%: `eg_cv1', 5%: `eg_cv5', 10%: `eg_cv10'"
display "    Statistic: " %9.4f `eg_stat' " => `eg_decision'"
display _newline

file write fout "Engle-Granger,PIB Brasil ~ PIB EUA,ADF on residuals," ///
    %12.6f (`eg_stat') "," %12.6f (`eg_cv5') ",,`eg_reject',`eg_decision'" _n

* --- EG with trend ---
display "  Engle-Granger: Log PIB Brasil ~ Log PIB EUA + trend"
generate time_trend = _n
regress log_gdp_br log_gdp_us time_trend
local eg_beta_ct = _b[log_gdp_us]
local eg_trend = _b[time_trend]
display "    beta = " %9.4f `eg_beta_ct' ", trend = " %9.6f `eg_trend'

predict eg_resid_ct, residuals
dfuller eg_resid_ct, noconstant lags(4)
local eg_stat_ct = r(Zt)
local eg_reject_ct = cond(`eg_stat_ct' < `eg_cv5', 1, 0)
local eg_decision_ct = cond(`eg_reject_ct', "Cointegrated", "Not cointegrated")
display "    ADF(resid) stat = " %9.4f `eg_stat_ct' " => `eg_decision_ct'"
display _newline

file write fout "Engle-Granger,PIB Brasil ~ PIB EUA + trend,ADF on residuals," ///
    %12.6f (`eg_stat_ct') "," %12.6f (`eg_cv5') ",,`eg_reject_ct',`eg_decision_ct'" _n

* --- EG in reverse direction: US ~ Brazil ---
display "  Engle-Granger: Log PIB EUA ~ Log PIB Brasil"
regress log_gdp_us log_gdp_br
local eg_beta_rev = _b[log_gdp_br]
predict eg_resid_rev, residuals
dfuller eg_resid_rev, noconstant lags(4)
local eg_stat_rev = r(Zt)
local eg_reject_rev = cond(`eg_stat_rev' < `eg_cv5', 1, 0)
local eg_decision_rev = cond(`eg_reject_rev', "Cointegrated", "Not cointegrated")
display "    beta = " %9.4f `eg_beta_rev'
display "    ADF(resid) stat = " %9.4f `eg_stat_rev' " => `eg_decision_rev'"
display "    Note: EG test can give different results depending on normalization"
display _newline

file write fout "Engle-Granger,PIB EUA ~ PIB Brasil,ADF on residuals," ///
    %12.6f (`eg_stat_rev') "," %12.6f (`eg_cv5') ",,`eg_reject_rev',`eg_decision_rev'" _n

********************************************************************************
* 4. Johansen Cointegration Test (vecrank)
********************************************************************************
display "--- Johansen Cointegration Test (vecrank) ---"
display _newline
display "  vecrank performs Johansen trace and max-eigenvalue tests."
display "  H0: rank(Pi) = r vs H1: rank(Pi) > r"
display _newline

* --- Johansen on GDP pair (trace test) ---
display "  Johansen: Log PIB Brasil, Log PIB EUA (trace, K=4, const):"
vecrank log_gdp_br log_gdp_us, trend(constant) lags(4)

* Extract results from stored values
local jo_n = e(N)
local jo_k = e(lags)

* vecrank stores trace statistics and critical values
* r(trace_0) and r(trace_1) for bivariate system
* We can access eigenvalues and rank from the output
display _newline
display "    Observations: `jo_n', Lags: `jo_k'"
display "    (See trace and max-eigenvalue statistics in output above)"
display _newline

file write fout "Johansen,PIB Brasil x PIB EUA,trace,see_output,,see_output,see_output,vecrank lags(4) trend(constant)" _n

* --- Johansen with different lag orders ---
display "  Johansen with different lag orders:"
display _newline

foreach k of numlist 2 4 6 8 {
    display "  vecrank lags(`k'):"
    quietly vecrank log_gdp_br log_gdp_us, trend(constant) lags(`k')
    display "    N = " e(N) ", lags = `k'"
    file write fout "Johansen,PIB Brasil x PIB EUA (K=`k'),trace,see_output,,see_output,see_output,vecrank lags(`k')" _n
}
display _newline

* --- Johansen with different trend specifications ---
display "  Johansen with different trend specifications (lags=4):"
display _newline

display "  vecrank trend(none) — no deterministic components:"
quietly vecrank log_gdp_br log_gdp_us, trend(none) lags(4)
file write fout "Johansen,PIB BR x EUA (trend=none),trace,see_output,,see_output,see_output,vecrank trend(none)" _n

display "  vecrank trend(constant) — restricted constant:"
quietly vecrank log_gdp_br log_gdp_us, trend(constant) lags(4)
file write fout "Johansen,PIB BR x EUA (trend=constant),trace,see_output,,see_output,see_output,vecrank trend(constant)" _n

display "  vecrank trend(trend) — restricted trend:"
quietly vecrank log_gdp_br log_gdp_us, trend(trend) lags(4)
file write fout "Johansen,PIB BR x EUA (trend=trend),trace,see_output,,see_output,see_output,vecrank trend(trend)" _n

display _newline

* --- Full vecrank output for main specification ---
display "  Full Johansen output (lags=4, trend=constant):"
display _newline
vecrank log_gdp_br log_gdp_us, trend(constant) lags(4) levela
display _newline

********************************************************************************
* 5. Synthetic Cointegrated Data
********************************************************************************
display "--- Synthetic Cointegrated Data ---"
display _newline
display "  Note: Stata's RNG differs from Python/R. Qualitative comparison only."
display _newline

clear
set obs 200
set seed 42
generate t = _n
tsset t

* Generate common stochastic trend (random walk)
generate eps_x = rnormal(0, 1)
generate x = sum(eps_x)

* Generate cointegrated y = 1.5 * x + stationary error
generate eps_eq = rnormal(0, 0.5)
generate eq_error = 0 in 1
replace eq_error = 0.3 * eq_error[_n-1] + eps_eq in 2/l
generate y_coint = 1.5 * x + eq_error

* Generate independent random walk
generate eps_indep = rnormal(0, 1)
generate y_indep = sum(eps_indep)

* EG test on cointegrated pair
display "  EG on cointegrated pair (y = 1.5*x + stationary):"
regress y_coint x
local eg_beta_syn = _b[x]
predict resid_coint, residuals
dfuller resid_coint, noconstant lags(4)
local eg_stat_syn = r(Zt)
local eg_reject_syn = cond(`eg_stat_syn' < -3.37, 1, 0)
display "    beta = " %9.4f `eg_beta_syn' " (true = 1.5)"
display "    ADF(resid) stat = " %9.4f `eg_stat_syn' " => " ///
    cond(`eg_reject_syn', "Cointegrated", "Not cointegrated")
file write fout "Engle-Granger,Par cointegrado Stata (beta=1.5),ADF on residuals," ///
    %12.6f (`eg_stat_syn') ",-3.37,,`eg_reject_syn',synthetic" _n
display _newline

* EG test on independent pair
display "  EG on independent pair (spurious regression):"
regress y_indep x
local eg_beta_indep = _b[x]
predict resid_indep, residuals
dfuller resid_indep, noconstant lags(4)
local eg_stat_indep = r(Zt)
local eg_reject_indep = cond(`eg_stat_indep' < -3.37, 1, 0)
display "    beta = " %9.4f `eg_beta_indep' " (spurious)"
display "    ADF(resid) stat = " %9.4f `eg_stat_indep' " => " ///
    cond(`eg_reject_indep', "Cointegrated", "Not cointegrated")
file write fout "Engle-Granger,Series independentes Stata,ADF on residuals," ///
    %12.6f (`eg_stat_indep') ",-3.37,,`eg_reject_indep',synthetic" _n
display _newline

* Johansen on cointegrated pair
display "  Johansen on cointegrated pair:"
vecrank y_coint x, trend(constant) lags(2)
file write fout "Johansen,Par cointegrado Stata,trace,see_output,,see_output,see_output,synthetic" _n
display _newline

* Johansen on independent pair
display "  Johansen on independent pair:"
vecrank y_indep x, trend(constant) lags(2)
file write fout "Johansen,Series independentes Stata,trace,see_output,,see_output,see_output,synthetic" _n
display _newline

********************************************************************************
* 6. Close output and summary
********************************************************************************
file close fout

display "======================================================================"
display "SUMMARY"
display "======================================================================"
display _newline
display "Results exported to: `outfile'"
display _newline
display "Tests performed:"
display "  - Engle-Granger (manual): regress + dfuller on residuals"
display "  - Johansen (vecrank): trace and max-eigenvalue tests"
display "  - Integration pre-check (dfuller on levels and differences)"
display _newline
display "Specifications tested:"
display "  - EG: constant, constant+trend, reverse direction"
display "  - Johansen: trend(none), trend(constant), trend(trend)"
display "  - Johansen: lags = 2, 4, 6, 8"
display _newline
display "Expected results (GDP data):"
display "  - Both series are I(1) => cointegration analysis is valid"
display "  - EG and Johansen may disagree on GDP cointegration"
display "  - Results should match qualitatively with Python and R"
display _newline
display "Done."
