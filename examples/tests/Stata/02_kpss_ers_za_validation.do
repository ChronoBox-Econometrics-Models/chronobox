********************************************************************************
* 02_kpss_ers_za_validation.do
* Cross-validation of KPSS, ERS (DF-GLS), and Zivot-Andrews tests
* Commands: kpss (ssc), dfgls (built-in), zandrews (ssc)
*
* Compares results with Python chronobox and R (urca) outputs.
* Uses GDP datasets loaded from CSV (identical across platforms).
*
* SSC packages required:
*   ssc install kpss      — KPSS stationarity test
*   ssc install zandrews  — Zivot-Andrews unit root test with structural break
*
* Built-in commands used:
*   dfgls — Elliott-Rothenberg-Stock DF-GLS test
*
* Note: Run "ssc install kpss" and "ssc install zandrews" before executing
* this script if these packages are not already installed.
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
display "KPSS, ERS (DF-GLS), and Zivot-Andrews Validation (Stata)"
display "======================================================================"
display _newline

* --- Check SSC packages ---
display "SSC packages required:"
display "  - kpss: ssc install kpss"
display "  - zandrews: ssc install zandrews"
display "  - dfgls: built-in (no installation needed)"
display _newline

* --- Initialize output file ---
local outfile "`out_dir'/kpss_ers_za_results.csv"
capture file close fout
file open fout using "`outfile'", write replace
file write fout "test,series,regression,statistic,cv_1pct,cv_5pct,cv_10pct,reject_at_5pct,decision" _n

********************************************************************************
* 1. Load and prepare US GDP data
********************************************************************************
display "--- Loading US GDP data ---"
display _newline

import delimited "`data_dir'/us_gdp_quarterly.csv", clear
generate date_q = quarterly(substr(date, 1, 7), "YQ")
format date_q %tq
tsset date_q

rename log_gdp log_gdp_us
rename gdp_growth gdp_growth_us
generate dlog_gdp_us = D.log_gdp_us

display "  US GDP: " _N " observations"
display _newline

********************************************************************************
* 2. KPSS Tests on US GDP
********************************************************************************
display "--- KPSS Tests on US GDP ---"
display _newline
display "  Note: KPSS tests H0: stationarity. Reject => unit root evidence."
display _newline

* --- KPSS on log GDP level (level stationarity) ---
display "  KPSS US GDP level (level stationarity):"
capture noisily kpss log_gdp_us, maxlag(4) notrend
if _rc == 0 {
    * kpss stores results in r()
    * Extract test statistic — kpss reports the statistic in the output
    display "    (See output above for test statistic and critical values)"
    file write fout "KPSS,Log PIB EUA (nivel),mu,see_output,,,,see_output,see_output" _n
}
else {
    display "    KPSS command not found. Install with: ssc install kpss"
    file write fout "KPSS,Log PIB EUA (nivel),mu,NOT_INSTALLED,,,,NA,ssc install kpss" _n
}
display _newline

* --- KPSS on log GDP level (trend stationarity) ---
display "  KPSS US GDP level (trend stationarity):"
capture noisily kpss log_gdp_us, maxlag(4)
if _rc == 0 {
    display "    (See output above for test statistic and critical values)"
    file write fout "KPSS,Log PIB EUA (nivel),tau,see_output,,,,see_output,see_output" _n
}
display _newline

* --- KPSS on first difference ---
display "  KPSS US GDP first difference (level stationarity):"
capture noisily kpss dlog_gdp_us, maxlag(4) notrend
if _rc == 0 {
    display "    (See output above)"
    file write fout "KPSS,Log PIB EUA (1a diferenca),mu,see_output,,,,see_output,see_output" _n
}
display _newline

********************************************************************************
* 3. DF-GLS (ERS) Tests on US GDP
********************************************************************************
display "--- DF-GLS (ERS) Tests on US GDP ---"
display _newline
display "  Note: dfgls tests H0: unit root (like ADF but with GLS detrending)."
display "  More powerful than standard ADF, especially with trend."
display _newline

* --- DF-GLS on log GDP level ---
display "  DF-GLS US GDP level:"
dfgls log_gdp_us, maxlag(8)
* dfgls reports results including optimal lag by MAIC
* Extract from stored results
local stat_dfgls = r(tstat)
display "    DF-GLS statistic = " %9.4f `stat_dfgls'
file write fout "DFGLS,Log PIB EUA (nivel),constant," %12.6f (`stat_dfgls') ",,,,,see_table_output" _n
display _newline

* --- DF-GLS on first difference ---
display "  DF-GLS US GDP first difference:"
dfgls dlog_gdp_us, maxlag(8)
local stat_dfgls = r(tstat)
display "    DF-GLS statistic = " %9.4f `stat_dfgls'
file write fout "DFGLS,Log PIB EUA (1a diferenca),constant," %12.6f (`stat_dfgls') ",,,,,see_table_output" _n
display _newline

********************************************************************************
* 4. Zivot-Andrews Tests on US GDP
********************************************************************************
display "--- Zivot-Andrews Tests on US GDP ---"
display _newline
display "  Note: zandrews tests H0: unit root allowing for one structural break."
display "  Models: intercept break, trend break, or both."
display _newline

* --- ZA on log GDP level (break in intercept) ---
display "  ZA US GDP level (break in intercept):"
capture noisily zandrews log_gdp_us, break(intercept) maxlags(4)
if _rc == 0 {
    local za_stat = r(Zt)
    local za_break = r(breakpoint)
    display "    ZA statistic = " %9.4f `za_stat' ", break at t = " `za_break'
    file write fout "ZA,Log PIB EUA (nivel),intercept," %12.6f (`za_stat') ",,,," `za_break' ",see_output" _n
}
else {
    display "    zandrews command not found. Install with: ssc install zandrews"
    file write fout "ZA,Log PIB EUA (nivel),intercept,NOT_INSTALLED,,,,NA,ssc install zandrews" _n
}
display _newline

* --- ZA on log GDP level (break in trend) ---
display "  ZA US GDP level (break in trend):"
capture noisily zandrews log_gdp_us, break(trend) maxlags(4)
if _rc == 0 {
    local za_stat = r(Zt)
    local za_break = r(breakpoint)
    display "    ZA statistic = " %9.4f `za_stat' ", break at t = " `za_break'
    file write fout "ZA,Log PIB EUA (nivel),trend," %12.6f (`za_stat') ",,,," `za_break' ",see_output" _n
}
display _newline

* --- ZA on log GDP level (break in both) ---
display "  ZA US GDP level (break in both intercept and trend):"
capture noisily zandrews log_gdp_us, break(both) maxlags(4)
if _rc == 0 {
    local za_stat = r(Zt)
    local za_break = r(breakpoint)
    display "    ZA statistic = " %9.4f `za_stat' ", break at t = " `za_break'
    file write fout "ZA,Log PIB EUA (nivel),both," %12.6f (`za_stat') ",,,," `za_break' ",see_output" _n
}
display _newline

* --- ZA on first difference ---
display "  ZA US GDP first difference (break in both):"
capture noisily zandrews dlog_gdp_us, break(both) maxlags(4)
if _rc == 0 {
    local za_stat = r(Zt)
    local za_break = r(breakpoint)
    display "    ZA statistic = " %9.4f `za_stat' ", break at t = " `za_break'
    file write fout "ZA,Log PIB EUA (1a diferenca),both," %12.6f (`za_stat') ",,,," `za_break' ",see_output" _n
}
display _newline

********************************************************************************
* 5. Load and prepare Brazil GDP data
********************************************************************************
display "--- Loading Brazil GDP data ---"
display _newline

import delimited "`data_dir'/brazil_gdp.csv", clear
generate date_q = quarterly(substr(date, 1, 7), "YQ")
format date_q %tq
tsset date_q

rename log_gdp log_gdp_br
rename gdp_growth gdp_growth_br
generate dlog_gdp_br = D.log_gdp_br

display "  Brazil GDP: " _N " observations"
display _newline

********************************************************************************
* 6. KPSS Tests on Brazil GDP
********************************************************************************
display "--- KPSS Tests on Brazil GDP ---"
display _newline

display "  KPSS Brazil GDP level (level stationarity):"
capture noisily kpss log_gdp_br, maxlag(4) notrend
if _rc == 0 {
    file write fout "KPSS,Log PIB Brasil (nivel),mu,see_output,,,,see_output,see_output" _n
}
display _newline

display "  KPSS Brazil GDP level (trend stationarity):"
capture noisily kpss log_gdp_br, maxlag(4)
if _rc == 0 {
    file write fout "KPSS,Log PIB Brasil (nivel),tau,see_output,,,,see_output,see_output" _n
}
display _newline

display "  KPSS Brazil GDP first difference (level stationarity):"
capture noisily kpss dlog_gdp_br, maxlag(4) notrend
if _rc == 0 {
    file write fout "KPSS,Log PIB Brasil (1a diferenca),mu,see_output,,,,see_output,see_output" _n
}
display _newline

********************************************************************************
* 7. DF-GLS (ERS) Tests on Brazil GDP
********************************************************************************
display "--- DF-GLS (ERS) Tests on Brazil GDP ---"
display _newline

display "  DF-GLS Brazil GDP level:"
dfgls log_gdp_br, maxlag(8)
local stat_dfgls = r(tstat)
display "    DF-GLS statistic = " %9.4f `stat_dfgls'
file write fout "DFGLS,Log PIB Brasil (nivel),constant," %12.6f (`stat_dfgls') ",,,,,see_table_output" _n
display _newline

display "  DF-GLS Brazil GDP first difference:"
dfgls dlog_gdp_br, maxlag(8)
local stat_dfgls = r(tstat)
display "    DF-GLS statistic = " %9.4f `stat_dfgls'
file write fout "DFGLS,Log PIB Brasil (1a diferenca),constant," %12.6f (`stat_dfgls') ",,,,,see_table_output" _n
display _newline

********************************************************************************
* 8. Zivot-Andrews Tests on Brazil GDP
********************************************************************************
display "--- Zivot-Andrews Tests on Brazil GDP ---"
display _newline

* --- ZA Brazil GDP - all three models ---
foreach model in intercept trend both {
    display "  ZA Brazil GDP level (break in `model'):"
    capture noisily zandrews log_gdp_br, break(`model') maxlags(4)
    if _rc == 0 {
        local za_stat = r(Zt)
        local za_break = r(breakpoint)
        display "    ZA statistic = " %9.4f `za_stat' ", break at t = " `za_break'
        file write fout "ZA,Log PIB Brasil (nivel),`model'," %12.6f (`za_stat') ",,,," `za_break' ",see_output" _n
    }
    display _newline
}

* --- ZA Brazil GDP first difference ---
display "  ZA Brazil GDP first difference (break in both):"
capture noisily zandrews dlog_gdp_br, break(both) maxlags(4)
if _rc == 0 {
    local za_stat = r(Zt)
    local za_break = r(breakpoint)
    display "    ZA statistic = " %9.4f `za_stat' ", break at t = " `za_break'
    file write fout "ZA,Log PIB Brasil (1a diferenca),both," %12.6f (`za_stat') ",,,," `za_break' ",see_output" _n
}
display _newline

********************************************************************************
* 9. Synthetic Data Illustration
********************************************************************************
display "--- Synthetic Data: KPSS, DF-GLS, ZA ---"
display _newline
display "  Note: Stata's RNG differs from Python/R. Qualitative comparison only."
display _newline

clear
set obs 200
set seed 42
generate t = _n
tsset t

* I(0) stationary process
generate eps = rnormal(0, 1)
generate y_i0 = 0 in 1
replace y_i0 = 0.5 * y_i0[_n-1] + eps in 2/l

* I(1) random walk
generate y_i1 = sum(rnormal(0, 1))

* Near unit root (phi = 0.95)
generate eps2 = rnormal(0, 1)
generate y_near = 0 in 1
replace y_near = 0.95 * y_near[_n-1] + eps2 in 2/l

* Structural break series
generate eps3 = rnormal(0, 1)
generate y_break = cond(t <= 100, 0 + eps3, 3 + eps3)

* DF-GLS on synthetic data
display "  DF-GLS on I(0) synthetic:"
dfgls y_i0, maxlag(4)
local stat_dfgls = r(tstat)
display "    DF-GLS statistic = " %9.4f `stat_dfgls'
file write fout "DFGLS,I(0) sintetica Stata,constant," %12.6f (`stat_dfgls') ",,,,,synthetic" _n
display _newline

display "  DF-GLS on I(1) synthetic:"
dfgls y_i1, maxlag(4)
local stat_dfgls = r(tstat)
display "    DF-GLS statistic = " %9.4f `stat_dfgls'
file write fout "DFGLS,I(1) sintetica Stata,constant," %12.6f (`stat_dfgls') ",,,,,synthetic" _n
display _newline

display "  DF-GLS on near-UR (phi=0.95) synthetic:"
dfgls y_near, maxlag(4)
local stat_dfgls = r(tstat)
display "    DF-GLS statistic = " %9.4f `stat_dfgls'
file write fout "DFGLS,Near UR (phi=0.95) Stata,constant," %12.6f (`stat_dfgls') ",,,,,synthetic" _n
display _newline

* KPSS on synthetic data
display "  KPSS on I(0) synthetic:"
capture noisily kpss y_i0, maxlag(4) notrend
if _rc == 0 {
    file write fout "KPSS,I(0) sintetica Stata,mu,see_output,,,,see_output,synthetic" _n
}
display _newline

display "  KPSS on I(1) synthetic:"
capture noisily kpss y_i1, maxlag(4) notrend
if _rc == 0 {
    file write fout "KPSS,I(1) sintetica Stata,mu,see_output,,,,see_output,synthetic" _n
}
display _newline

* ZA on structural break series
display "  ZA on structural break synthetic (break in intercept):"
capture noisily zandrews y_break, break(intercept) maxlags(4)
if _rc == 0 {
    local za_stat = r(Zt)
    local za_break = r(breakpoint)
    display "    ZA statistic = " %9.4f `za_stat' ", break at t = " `za_break'
    file write fout "ZA,Quebra estrutural Stata,intercept," %12.6f (`za_stat') ",,,," `za_break' ",synthetic" _n
}
display _newline

********************************************************************************
* 10. Close output and summary
********************************************************************************
file close fout

display "======================================================================"
display "SUMMARY"
display "======================================================================"
display _newline
display "Results exported to: `outfile'"
display _newline
display "Tests performed:"
display "  - KPSS (ssc): level and trend stationarity"
display "  - DF-GLS/ERS (dfgls): built-in, GLS-detrended unit root test"
display "  - Zivot-Andrews (ssc): unit root with structural break"
display _newline
display "SSC packages required:"
display "  ssc install kpss"
display "  ssc install zandrews"
display _newline
display "Expected results (GDP data):"
display "  - KPSS: reject H0 on levels (non-stationary), fail on diffs"
display "  - DF-GLS: fail to reject on levels, reject on diffs"
display "  - ZA: identifies potential break dates in GDP series"
display _newline
display "Done."
