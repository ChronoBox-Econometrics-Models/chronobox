********************************************************************************
* 01_unit_root_validation.do
* Cross-validation of ADF and Phillips-Perron unit root tests
* Commands: dfuller, pperron
*
* Compares results with Python chronobox and R (urca/tseries) outputs.
* Uses GDP datasets loaded from CSV (identical across platforms).
*
* SSC packages required: none (dfuller and pperron are built-in)
*
* Note: Stata does not have a native way to generate synthetic data with
* the same RNG as Python/R, so we focus on real GDP data for exact
* comparison. Synthetic series are generated for illustration only.
********************************************************************************

clear all
set more off
set seed 42

* --- Configuration ---
local data_dir "/home/guhaase/projetos/chronobox/examples/tests/data"
local out_dir  "/home/guhaase/projetos/chronobox/examples/tests/outputs/Stata"

* Create output directory
capture mkdir "`out_dir'"

display _newline
display "======================================================================"
display "Unit Root Validation: ADF and Phillips-Perron Tests (Stata)"
display "======================================================================"
display _newline

* --- Initialize output file ---
local outfile "`out_dir'/unit_root_results.csv"
capture file close fout
file open fout using "`outfile'", write replace
file write fout "test,series,regression,statistic,pvalue,lags_used,reject_at_5pct,decision" _n

********************************************************************************
* 1. Load and prepare US GDP data
********************************************************************************
display "--- Loading US GDP data ---"
display _newline

import delimited "`data_dir'/us_gdp_quarterly.csv", clear
generate date_q = quarterly(substr(date, 1, 7), "YQ")
format date_q %tq
tsset date_q

* Rename for convenience
rename log_gdp log_gdp_us
rename gdp_growth gdp_growth_us

* Save US GDP
tempfile us_data
save `us_data', replace

display "  US GDP: " _N " observations"
display _newline

********************************************************************************
* 2. ADF Tests on US GDP (dfuller)
********************************************************************************
display "--- ADF Tests on US GDP (dfuller) ---"
display _newline

* --- ADF on log GDP level: no constant, no trend ---
display "  ADF US GDP level (noconstant):"
dfuller log_gdp_us, noconstant lags(4)
local stat = r(Zt)
local pval = r(p)
local lags = 4
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "ADF,Log PIB EUA (nivel),noconstant," %12.6f (`stat') "," %12.6f (`pval') ",`lags',`reject',`decision'" _n
display _newline

* --- ADF on log GDP level: with drift (constant) ---
display "  ADF US GDP level (drift/constant):"
dfuller log_gdp_us, drift lags(4)
local stat = r(Zt)
local pval = r(p)
local lags = 4
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "ADF,Log PIB EUA (nivel),drift," %12.6f (`stat') "," %12.6f (`pval') ",`lags',`reject',`decision'" _n
display _newline

* --- ADF on log GDP level: with trend ---
display "  ADF US GDP level (trend):"
dfuller log_gdp_us, trend lags(4)
local stat = r(Zt)
local pval = r(p)
local lags = 4
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "ADF,Log PIB EUA (nivel),trend," %12.6f (`stat') "," %12.6f (`pval') ",`lags',`reject',`decision'" _n
display _newline

* --- ADF on first difference of log GDP ---
display "  ADF US GDP first difference (drift):"
generate dlog_gdp_us = D.log_gdp_us
dfuller dlog_gdp_us, drift lags(4)
local stat = r(Zt)
local pval = r(p)
local lags = 4
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "ADF,Log PIB EUA (1a diferenca),drift," %12.6f (`stat') "," %12.6f (`pval') ",`lags',`reject',`decision'" _n
display _newline

* --- ADF with different lag specifications ---
display "  ADF US GDP level - lag sensitivity:"
foreach lag of numlist 1 2 4 8 {
    quietly dfuller log_gdp_us, drift lags(`lag')
    local stat = r(Zt)
    local pval = r(p)
    local reject = cond(`pval' < 0.05, 1, 0)
    display "    lags=`lag': stat = " %9.4f `stat' ", p = " %9.6f `pval'
    file write fout "ADF,Log PIB EUA (nivel) lag=`lag',drift," %12.6f (`stat') "," %12.6f (`pval') ",`lag',`reject',lag sensitivity" _n
}
display _newline

********************************************************************************
* 3. Phillips-Perron Tests on US GDP (pperron)
********************************************************************************
display "--- Phillips-Perron Tests on US GDP (pperron) ---"
display _newline

* --- PP on log GDP level: no constant ---
display "  PP US GDP level (noconstant):"
pperron log_gdp_us, noconstant lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "PP,Log PIB EUA (nivel),noconstant," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',`decision'" _n
display _newline

* --- PP on log GDP level: with drift ---
display "  PP US GDP level (drift):"
pperron log_gdp_us, lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "PP,Log PIB EUA (nivel),drift," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',`decision'" _n
display _newline

* --- PP on log GDP level: with trend ---
display "  PP US GDP level (trend):"
pperron log_gdp_us, trend lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "PP,Log PIB EUA (nivel),trend," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',`decision'" _n
display _newline

* --- PP on first difference ---
display "  PP US GDP first difference (drift):"
pperron dlog_gdp_us, lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "PP,Log PIB EUA (1a diferenca),drift," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',`decision'" _n
display _newline

********************************************************************************
* 4. Load and prepare Brazil GDP data
********************************************************************************
display "--- Loading Brazil GDP data ---"
display _newline

import delimited "`data_dir'/brazil_gdp.csv", clear
generate date_q = quarterly(substr(date, 1, 7), "YQ")
format date_q %tq
tsset date_q

rename log_gdp log_gdp_br
rename gdp_growth gdp_growth_br

display "  Brazil GDP: " _N " observations"
display _newline

* Generate first difference
generate dlog_gdp_br = D.log_gdp_br

********************************************************************************
* 5. ADF Tests on Brazil GDP
********************************************************************************
display "--- ADF Tests on Brazil GDP (dfuller) ---"
display _newline

* --- ADF on log GDP level: drift ---
display "  ADF Brazil GDP level (drift):"
dfuller log_gdp_br, drift lags(4)
local stat = r(Zt)
local pval = r(p)
local lags = 4
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "ADF,Log PIB Brasil (nivel),drift," %12.6f (`stat') "," %12.6f (`pval') ",`lags',`reject',`decision'" _n
display _newline

* --- ADF on log GDP level: trend ---
display "  ADF Brazil GDP level (trend):"
dfuller log_gdp_br, trend lags(4)
local stat = r(Zt)
local pval = r(p)
local lags = 4
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "ADF,Log PIB Brasil (nivel),trend," %12.6f (`stat') "," %12.6f (`pval') ",`lags',`reject',`decision'" _n
display _newline

* --- ADF on first difference ---
display "  ADF Brazil GDP first difference (drift):"
dfuller dlog_gdp_br, drift lags(4)
local stat = r(Zt)
local pval = r(p)
local lags = 4
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "ADF,Log PIB Brasil (1a diferenca),drift," %12.6f (`stat') "," %12.6f (`pval') ",`lags',`reject',`decision'" _n
display _newline

* --- ADF with different lag specifications ---
display "  ADF Brazil GDP level - lag sensitivity:"
foreach lag of numlist 1 2 4 8 {
    quietly dfuller log_gdp_br, drift lags(`lag')
    local stat = r(Zt)
    local pval = r(p)
    local reject = cond(`pval' < 0.05, 1, 0)
    display "    lags=`lag': stat = " %9.4f `stat' ", p = " %9.6f `pval'
    file write fout "ADF,Log PIB Brasil (nivel) lag=`lag',drift," %12.6f (`stat') "," %12.6f (`pval') ",`lag',`reject',lag sensitivity" _n
}
display _newline

********************************************************************************
* 6. Phillips-Perron Tests on Brazil GDP
********************************************************************************
display "--- Phillips-Perron Tests on Brazil GDP (pperron) ---"
display _newline

* --- PP on log GDP level: drift ---
display "  PP Brazil GDP level (drift):"
pperron log_gdp_br, lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "PP,Log PIB Brasil (nivel),drift," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',`decision'" _n
display _newline

* --- PP on log GDP level: trend ---
display "  PP Brazil GDP level (trend):"
pperron log_gdp_br, trend lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "PP,Log PIB Brasil (nivel),trend," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',`decision'" _n
display _newline

* --- PP on first difference ---
display "  PP Brazil GDP first difference (drift):"
pperron dlog_gdp_br, lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
local decision = cond(`reject', "Reject H0 (stationary)", "Fail to reject H0 (unit root)")
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval' " => `decision'"
file write fout "PP,Log PIB Brasil (1a diferenca),drift," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',`decision'" _n
display _newline

********************************************************************************
* 7. Synthetic Data Illustration
********************************************************************************
display "--- Synthetic Data: Unit Root Processes ---"
display _newline
display "  Note: Stata's RNG differs from Python/R, so synthetic data results"
display "  are for illustration and qualitative comparison only."
display _newline

* Generate I(0) stationary process (AR(1) with phi=0.5)
clear
set obs 200
set seed 42
generate t = _n
tsset t

generate eps = rnormal(0, 1)
generate y_i0 = 0 in 1
replace y_i0 = 0.5 * y_i0[_n-1] + eps in 2/l

* Generate I(1) random walk
generate y_i1 = sum(rnormal(0, 1))

display "  ADF on I(0) synthetic (drift):"
dfuller y_i0, drift lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval'
file write fout "ADF,I(0) sintetica Stata,drift," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',synthetic" _n
display _newline

display "  ADF on I(1) synthetic (drift):"
dfuller y_i1, drift lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval'
file write fout "ADF,I(1) sintetica Stata,drift," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',synthetic" _n
display _newline

display "  ADF on diff(I(1)) synthetic (drift):"
generate dy_i1 = D.y_i1
dfuller dy_i1, drift lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval'
file write fout "ADF,I(1) sintetica Stata 1a diferenca,drift," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',synthetic" _n
display _newline

display "  PP on I(0) synthetic:"
pperron y_i0, lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval'
file write fout "PP,I(0) sintetica Stata,drift," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',synthetic" _n
display _newline

display "  PP on I(1) synthetic:"
pperron y_i1, lags(4)
local stat = r(Zt)
local pval = r(p)
local reject = cond(`pval' < 0.05, 1, 0)
display "    stat = " %9.4f `stat' ", p = " %9.6f `pval'
file write fout "PP,I(1) sintetica Stata,drift," %12.6f (`stat') "," %12.6f (`pval') ",4,`reject',synthetic" _n
display _newline

********************************************************************************
* 8. Close output and summary
********************************************************************************
file close fout

display "======================================================================"
display "SUMMARY"
display "======================================================================"
display _newline
display "Results exported to: `outfile'"
display _newline
display "Tests performed:"
display "  - ADF (dfuller): noconstant, drift, trend specifications"
display "  - PP (pperron): noconstant, drift, trend specifications"
display "  - Lag sensitivity analysis (lags = 1, 2, 4, 8)"
display "  - Both US and Brazil GDP series (level + first difference)"
display "  - Synthetic I(0) and I(1) processes for illustration"
display _newline
display "Expected results (GDP data):"
display "  - Log GDP (level): fail to reject H0 => unit root"
display "  - Log GDP (1st diff): reject H0 => stationary"
display "  - Qualitative decisions should match Python and R"
display _newline
display "Done."
