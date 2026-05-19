********************************************************************************
* 04_breaks_validation.do
* Cross-validation of structural break tests
* Commands: estat sbcusum (built-in), Chow test (manual F-test)
*
* Compares results with Python chronobox and R (strucchange) outputs.
* Uses GDP datasets loaded from CSV (identical across platforms).
*
* SSC packages that may be useful (documented but not required):
*   ssc install multbreak  — Bai-Perron multiple structural breaks
*
* Built-in commands used:
*   estat sbcusum — CUSUM test after regress (cumulative sum of residuals)
*   regress      — OLS for Chow test manual implementation
*   constraint   — for Chow test via constrained regression
*
* Note on Bai-Perron: The "multbreak" package from SSC implements the
* Bai-Perron (1998, 2003) test for multiple structural breaks. Install
* with "ssc install multbreak". This script documents its usage but does
* not require it to run the core tests.
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
display "Structural Breaks Validation: CUSUM, Chow, Bai-Perron (Stata)"
display "======================================================================"
display _newline

* --- Initialize output file ---
local outfile "`out_dir'/breaks_results.csv"
capture file close fout
file open fout using "`outfile'", write replace
file write fout "test,series,break_point,statistic,pvalue,reject_at_5pct,decision" _n

********************************************************************************
* 1. Generate synthetic data with structural breaks
********************************************************************************
display "--- Generating Synthetic Data ---"
display _newline

clear
set obs 200
set seed 42
generate t = _n
tsset t

* Level break at midpoint (t=100)
generate eps1 = rnormal(0, 1)
generate y_level_break = cond(t <= 100, 0 + eps1, 3 + eps1)

* Trend break at midpoint
generate eps2 = rnormal(0, 1)
generate y_trend_break = cond(t <= 100, eps2, 0.05 * (t - 100) + eps2)

* Stable series (no break)
generate eps3 = rnormal(0, 1)
generate y_stable = eps3

display "  Generated 200-obs synthetic series:"
display "    y_level_break: level shift of +3 at t=100"
display "    y_trend_break: trend break at t=100 (slope 0 -> 0.05)"
display "    y_stable: no break (control)"
display _newline

********************************************************************************
* 2. Chow Test (manual F-test)
********************************************************************************
display "--- Chow Tests (manual F-test implementation) ---"
display _newline
display "  Chow test: H0 = no structural break at specified point"
display "  F = [(SSR_full - SSR_1 - SSR_2) / k] / [(SSR_1 + SSR_2) / (n - 2k)]"
display _newline

* --- Chow test program ---
capture program drop chow_test
program define chow_test, rclass
    syntax varlist(min=1 max=1), break_point(integer) [trend]

    local y `varlist'
    local n = _N
    local bp = `break_point'

    * Full model
    if "`trend'" != "" {
        quietly regress `y' t
    }
    else {
        quietly regress `y'
    }
    local ssr_full = e(rss)
    local k = e(df_m) + 1   // number of parameters (including constant)

    * Sub-sample 1: t <= bp
    if "`trend'" != "" {
        quietly regress `y' t if t <= `bp'
    }
    else {
        quietly regress `y' if t <= `bp'
    }
    local ssr1 = e(rss)

    * Sub-sample 2: t > bp
    if "`trend'" != "" {
        quietly regress `y' t if t > `bp'
    }
    else {
        quietly regress `y' if t > `bp'
    }
    local ssr2 = e(rss)

    * F-statistic
    local f_stat = ((`ssr_full' - `ssr1' - `ssr2') / `k') / ///
                   ((`ssr1' + `ssr2') / (`n' - 2 * `k'))
    local df1 = `k'
    local df2 = `n' - 2 * `k'
    local p_value = 1 - F(`df1', `df2', `f_stat')

    return scalar F = `f_stat'
    return scalar p = `p_value'
    return scalar df1 = `df1'
    return scalar df2 = `df2'
    return scalar ssr_full = `ssr_full'
    return scalar ssr1 = `ssr1'
    return scalar ssr2 = `ssr2'
end

* --- Chow test on level break (correct point) ---
display "  Chow: Level break at t=100 (correct point):"
chow_test y_level_break, break_point(100)
local f_stat = r(F)
local p_val = r(p)
local reject = cond(`p_val' < 0.05, 1, 0)
display "    F = " %9.4f `f_stat' ", p = " %9.6f `p_val' " => " ///
    cond(`reject', "BREAK DETECTED", "NO BREAK")
file write fout "Chow,Nivel (ponto correto),100," %12.6f (`f_stat') "," ///
    %12.6f (`p_val') ",`reject'," cond(`reject', "Quebra detectada", "Sem quebra") _n
display _newline

* --- Chow test on level break (wrong point) ---
display "  Chow: Level break at t=50 (wrong point):"
chow_test y_level_break, break_point(50)
local f_stat = r(F)
local p_val = r(p)
local reject = cond(`p_val' < 0.05, 1, 0)
display "    F = " %9.4f `f_stat' ", p = " %9.6f `p_val' " => " ///
    cond(`reject', "BREAK DETECTED", "NO BREAK")
file write fout "Chow,Nivel (ponto errado t=50),50," %12.6f (`f_stat') "," ///
    %12.6f (`p_val') ",`reject'," cond(`reject', "Quebra detectada", "Sem quebra") _n
display _newline

* --- Chow test on trend break ---
display "  Chow: Trend break at t=100 (correct point, with trend):"
chow_test y_trend_break, break_point(100) trend
local f_stat = r(F)
local p_val = r(p)
local reject = cond(`p_val' < 0.05, 1, 0)
display "    F = " %9.4f `f_stat' ", p = " %9.6f `p_val' " => " ///
    cond(`reject', "BREAK DETECTED", "NO BREAK")
file write fout "Chow,Tendencia (ponto correto),100," %12.6f (`f_stat') "," ///
    %12.6f (`p_val') ",`reject'," cond(`reject', "Quebra detectada", "Sem quebra") _n
display _newline

* --- Chow test on stable series ---
display "  Chow: Stable series at t=100 (control):"
chow_test y_stable, break_point(100)
local f_stat = r(F)
local p_val = r(p)
local reject = cond(`p_val' < 0.05, 1, 0)
display "    F = " %9.4f `f_stat' ", p = " %9.6f `p_val' " => " ///
    cond(`reject', "BREAK DETECTED", "NO BREAK")
file write fout "Chow,Estavel (controle),100," %12.6f (`f_stat') "," ///
    %12.6f (`p_val') ",`reject'," cond(`reject', "Quebra detectada", "Sem quebra") _n
display _newline

* --- Chow scan (sequential F-tests) ---
display "  Chow scan: Testing level break at multiple points:"
foreach bp of numlist 30 50 70 80 90 100 110 120 140 160 {
    quietly chow_test y_level_break, break_point(`bp')
    local f_stat = r(F)
    local p_val = r(p)
    local reject = cond(`p_val' < 0.05, 1, 0)
    display "    bp=`bp': F = " %9.4f `f_stat' ", p = " %9.6f `p_val' ///
        cond(`reject', " ***", "")
    file write fout "Chow_scan,Nivel [bp=`bp'],`bp'," %12.6f (`f_stat') "," ///
        %12.6f (`p_val') ",`reject',scan" _n
}
display _newline

********************************************************************************
* 3. CUSUM Test (estat sbcusum after regress)
********************************************************************************
display "--- CUSUM Tests (estat sbcusum) ---"
display _newline
display "  estat sbcusum: CUSUM test for parameter stability after OLS"
display "  H0: no structural instability"
display _newline

* --- CUSUM on level break ---
display "  CUSUM on level break series:"
regress y_level_break t
estat sbcusum
file write fout "CUSUM,Nivel (quebra),,,see_output,,estat sbcusum" _n
display _newline

* --- CUSUM on stable series ---
display "  CUSUM on stable series:"
regress y_stable t
estat sbcusum
file write fout "CUSUM,Estavel (controle),,,see_output,,estat sbcusum" _n
display _newline

* --- CUSUM on trend break ---
display "  CUSUM on trend break series:"
regress y_trend_break t
estat sbcusum
file write fout "CUSUM,Tendencia (quebra),,,see_output,,estat sbcusum" _n
display _newline

********************************************************************************
* 4. Chow test on GDP data
********************************************************************************
display "--- Chow Tests on GDP Data ---"
display _newline

* Load Brazil GDP
import delimited "`data_dir'/brazil_gdp.csv", clear
generate date_q = quarterly(substr(date, 1, 7), "YQ")
format date_q %tq
tsset date_q
rename gdp_growth gdp_growth_br
rename log_gdp log_gdp_br
generate t = _n

display "  Brazil GDP growth: " _N " observations"
display _newline

* Chow test on Brazil GDP growth at t=36 (~2003-Q1)
display "  Chow: PIB Brasil crescimento at t=36 (~2003-Q1):"
chow_test gdp_growth_br, break_point(36)
local f_stat = r(F)
local p_val = r(p)
local reject = cond(`p_val' < 0.05, 1, 0)
display "    F = " %9.4f `f_stat' ", p = " %9.6f `p_val' " => " ///
    cond(`reject', "BREAK DETECTED", "NO BREAK")
file write fout "Chow,PIB Brasil (break=36 ~2003-Q1),36," %12.6f (`f_stat') "," ///
    %12.6f (`p_val') ",`reject',GDP" _n
display _newline

* CUSUM on Brazil GDP growth
display "  CUSUM on Brazil GDP growth:"
regress gdp_growth_br t
estat sbcusum
file write fout "CUSUM,PIB Brasil crescimento,,,see_output,,estat sbcusum GDP" _n
display _newline

* Save Brazil data
tempfile br_data
save `br_data', replace

* Load US GDP
import delimited "`data_dir'/us_gdp_quarterly.csv", clear
generate date_q = quarterly(substr(date, 1, 7), "YQ")
format date_q %tq
tsset date_q
rename gdp_growth gdp_growth_us
rename log_gdp log_gdp_us
generate t = _n

display "  US GDP growth: " _N " observations"
display _newline

* Chow test at 2008-Q3 financial crisis (~t=137)
local bp_crisis = min(137, _N - 10)
display "  Chow: PIB EUA crescimento at t=`bp_crisis' (~2008-Q3 crisis):"
chow_test gdp_growth_us, break_point(`bp_crisis')
local f_stat = r(F)
local p_val = r(p)
local reject = cond(`p_val' < 0.05, 1, 0)
display "    F = " %9.4f `f_stat' ", p = " %9.6f `p_val' " => " ///
    cond(`reject', "BREAK DETECTED", "NO BREAK")
file write fout "Chow,PIB EUA (2008-Q3 crise),`bp_crisis'," %12.6f (`f_stat') "," ///
    %12.6f (`p_val') ",`reject',GDP" _n
display _newline

* Chow test at 2020-Q1 COVID (~t=183)
local bp_covid = min(183, _N - 10)
display "  Chow: PIB EUA crescimento at t=`bp_covid' (~2020-Q1 COVID):"
chow_test gdp_growth_us, break_point(`bp_covid')
local f_stat = r(F)
local p_val = r(p)
local reject = cond(`p_val' < 0.05, 1, 0)
display "    F = " %9.4f `f_stat' ", p = " %9.6f `p_val' " => " ///
    cond(`reject', "BREAK DETECTED", "NO BREAK")
file write fout "Chow,PIB EUA (2020-Q1 COVID),`bp_covid'," %12.6f (`f_stat') "," ///
    %12.6f (`p_val') ",`reject',GDP" _n
display _newline

* CUSUM on US GDP growth
display "  CUSUM on US GDP growth:"
regress gdp_growth_us t
estat sbcusum
file write fout "CUSUM,PIB EUA crescimento,,,see_output,,estat sbcusum GDP" _n
display _newline

********************************************************************************
* 5. Two-break series
********************************************************************************
display "--- Two Breaks Series ---"
display _newline

clear
set obs 300
set seed 42
generate t = _n
tsset t

generate eps = rnormal(0, 1)
generate y_2breaks = cond(t <= 100, 0 + eps, cond(t <= 200, 3 + eps, 1 + eps))

display "  Series with 2 breaks at t=100 and t=200:"
display "    Regime 1 (t<=100): mean=0"
display "    Regime 2 (100<t<=200): mean=3"
display "    Regime 3 (t>200): mean=1"
display _newline

* Chow at break 1
display "  Chow at t=100 (first break):"
chow_test y_2breaks, break_point(100)
local f_stat = r(F)
local p_val = r(p)
local reject = cond(`p_val' < 0.05, 1, 0)
display "    F = " %9.4f `f_stat' ", p = " %9.6f `p_val'
file write fout "Chow,2 quebras (t=100),100," %12.6f (`f_stat') "," ///
    %12.6f (`p_val') ",`reject',two_breaks" _n

* Chow at break 2
display "  Chow at t=200 (second break):"
chow_test y_2breaks, break_point(200)
local f_stat = r(F)
local p_val = r(p)
local reject = cond(`p_val' < 0.05, 1, 0)
display "    F = " %9.4f `f_stat' ", p = " %9.6f `p_val'
file write fout "Chow,2 quebras (t=200),200," %12.6f (`f_stat') "," ///
    %12.6f (`p_val') ",`reject',two_breaks" _n
display _newline

* CUSUM on two-break series
display "  CUSUM on two-break series:"
regress y_2breaks t
estat sbcusum
file write fout "CUSUM,2 quebras,,,see_output,,estat sbcusum" _n
display _newline

********************************************************************************
* 6. Bai-Perron documentation (multbreak)
********************************************************************************
display "--- Bai-Perron Multiple Structural Breaks ---"
display _newline
display "  The multbreak package (ssc) implements Bai-Perron (1998, 2003)"
display "  for detecting multiple structural breaks."
display _newline
display "  Installation:"
display "    ssc install multbreak"
display _newline
display "  Usage example:"
display "    multbreak y, breaks(3) trim(0.15)"
display _newline
display "  This performs:"
display "    - Sequential supF tests for 1 vs 0, 2 vs 1, ... breaks"
display "    - UDmax and WDmax tests"
display "    - Estimation of break dates with confidence intervals"
display "    - BIC/LWZ information criteria for optimal number of breaks"
display _newline

* Try multbreak if installed
display "  Attempting multbreak on two-break series..."
capture noisily {
    * multbreak y_2breaks, breaks(5) trim(0.15)
    display "    multbreak not attempted (uncomment to run if installed)"
}
display _newline

file write fout "Bai-Perron,Documentation,,,,NA,ssc install multbreak" _n

********************************************************************************
* 7. Chow test via constraint (alternative implementation)
********************************************************************************
display "--- Chow Test via Constraint (alternative) ---"
display _newline
display "  Alternative Chow test using dummy variables and constraint test."
display _newline

* Reload synthetic data
clear
set obs 200
set seed 42
generate t = _n
tsset t

generate eps1 = rnormal(0, 1)
generate y_level_break = cond(t <= 100, 0 + eps1, 3 + eps1)

* Create dummy and interaction
generate d_post = (t > 100)
generate d_post_t = d_post * t

* Full model with dummy
display "  Chow via constraint: Level break at t=100"
regress y_level_break t d_post d_post_t
test d_post d_post_t
local f_chow_alt = r(F)
local p_chow_alt = r(p)
local reject_alt = cond(`p_chow_alt' < 0.05, 1, 0)
display "    F (joint test) = " %9.4f `f_chow_alt' ", p = " %9.6f `p_chow_alt'
display "    => " cond(`reject_alt', "BREAK DETECTED", "NO BREAK")
display _newline

file write fout "Chow_constraint,Nivel (dummy method),100," ///
    %12.6f (`f_chow_alt') "," %12.6f (`p_chow_alt') ",`reject_alt',alternative method" _n

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
display "  - Chow test (manual F-test): SSR decomposition"
display "  - Chow test (constraint): dummy variable + joint F-test"
display "  - CUSUM (estat sbcusum): cumulative sum of residuals"
display "  - Chow scan: sequential F-tests at multiple breakpoints"
display _newline
display "SSC packages documented:"
display "  ssc install multbreak  — Bai-Perron multiple structural breaks"
display _newline
display "Data tested:"
display "  - Synthetic: level break, trend break, stable, two breaks"
display "  - GDP: Brazil growth (break ~2003-Q1), US growth (2008 crisis, COVID)"
display _newline
display "Expected results:"
display "  - Level break at t=100: Chow and CUSUM should detect"
display "  - Stable series: no break detected (control)"
display "  - GDP: potential breaks at crisis dates"
display _newline
display "Done."
