********************************************************************************
* compare_results.do
* Compare Stata results with Python (chronobox) and R outputs
*
* Loads CSV outputs from Stata scripts and JSON outputs from Python/R,
* presenting a unified comparison table.
*
* Since Stata cannot natively parse JSON, this script:
* 1. Loads Stata CSV outputs
* 2. Displays summary tables
* 3. Documents expected Python/R values for manual comparison
*
* For automated comparison, use the Python notebook or R script.
********************************************************************************

clear all
set more off

* --- Configuration ---
local out_dir "/home/guhaase/projetos/chronobox/examples/tests/outputs"
local stata_dir "`out_dir'/Stata"

display _newline
display "======================================================================"
display "Cross-Platform Comparison: Python (chronobox) vs R vs Stata"
display "======================================================================"
display _newline

********************************************************************************
* 1. Load and display Stata unit root results
********************************************************************************
display "======================================================================"
display "1. Unit Root Tests (ADF & Phillips-Perron)"
display "======================================================================"
display _newline

capture {
    import delimited "`stata_dir'/unit_root_results.csv", clear
    display "  Stata unit root results loaded: " _N " rows"
    display _newline

    * Display GDP results only (comparable across platforms)
    display "  --- GDP Series (identical data across platforms) ---"
    display _newline
    list test series regression statistic pvalue reject_at_5pct ///
        if strpos(series, "PIB") > 0 & strpos(decision, "lag") == 0, ///
        separator(0) noobs abbreviate(20)
    display _newline
}
if _rc != 0 {
    display "  WARNING: unit_root_results.csv not found."
    display "  Run 01_unit_root_validation.do first."
    display _newline
}

********************************************************************************
* 2. Load and display Stata KPSS/ERS/ZA results
********************************************************************************
display "======================================================================"
display "2. KPSS, ERS (DF-GLS), and Zivot-Andrews Tests"
display "======================================================================"
display _newline

capture {
    import delimited "`stata_dir'/kpss_ers_za_results.csv", clear
    display "  Stata KPSS/ERS/ZA results loaded: " _N " rows"
    display _newline

    * Display GDP results
    display "  --- GDP Series ---"
    display _newline
    list test series regression statistic ///
        if strpos(series, "PIB") > 0, ///
        separator(0) noobs abbreviate(20)
    display _newline
}
if _rc != 0 {
    display "  WARNING: kpss_ers_za_results.csv not found."
    display "  Run 02_kpss_ers_za_validation.do first."
    display _newline
}

********************************************************************************
* 3. Load and display Stata cointegration results
********************************************************************************
display "======================================================================"
display "3. Cointegration Tests (Engle-Granger & Johansen)"
display "======================================================================"
display _newline

capture {
    import delimited "`stata_dir'/cointegration_results.csv", clear
    display "  Stata cointegration results loaded: " _N " rows"
    display _newline

    display "  --- All Results ---"
    display _newline
    list test system type statistic reject_at_5pct decision, ///
        separator(0) noobs abbreviate(20)
    display _newline
}
if _rc != 0 {
    display "  WARNING: cointegration_results.csv not found."
    display "  Run 03_cointegration_validation.do first."
    display _newline
}

********************************************************************************
* 4. Load and display Stata breaks results
********************************************************************************
display "======================================================================"
display "4. Structural Break Tests (Chow & CUSUM)"
display "======================================================================"
display _newline

capture {
    import delimited "`stata_dir'/breaks_results.csv", clear
    display "  Stata breaks results loaded: " _N " rows"
    display _newline

    * Display non-scan results
    display "  --- Main Results (excluding scan) ---"
    display _newline
    list test series break_point statistic pvalue reject_at_5pct ///
        if test != "Chow_scan", ///
        separator(0) noobs abbreviate(20)
    display _newline
}
if _rc != 0 {
    display "  WARNING: breaks_results.csv not found."
    display "  Run 04_breaks_validation.do first."
    display _newline
}

********************************************************************************
* 5. Expected Python/R values for comparison
********************************************************************************
display "======================================================================"
display "5. Expected Values from Python and R (for manual comparison)"
display "======================================================================"
display _newline

display "  The following are reference values from Python (chronobox) and R"
display "  outputs stored in JSON files. Since Stata cannot parse JSON natively,"
display "  these are documented here for manual comparison."
display _newline

display "  --- ADF Test on GDP Data ---"
display _newline
display "  Python/R JSON files:"
display "    `out_dir'/adf_pp_results.json"
display "    `out_dir'/R/unit_root_results.json"
display _newline
display "  Key comparisons (GDP level, drift/constant):"
display "    Log PIB EUA (nivel):     compare ADF statistic"
display "    Log PIB Brasil (nivel):  compare ADF statistic"
display "    Log PIB EUA (1a diff):   compare ADF statistic (should reject)"
display "    Log PIB Brasil (1a diff): compare ADF statistic (should reject)"
display _newline

display "  --- PP Test on GDP Data ---"
display _newline
display "  Key comparisons:"
display "    Log PIB EUA (nivel):     compare PP Z(t) statistic"
display "    Log PIB Brasil (nivel):  compare PP Z(t) statistic"
display _newline

display "  --- KPSS Test ---"
display _newline
display "  Python/R JSON files:"
display "    `out_dir'/kpss_ers_za_results.json"
display "    `out_dir'/R/kpss_ers_za_results.json"
display _newline
display "  Key comparisons:"
display "    Log PIB EUA (nivel):     KPSS should reject (non-stationary)"
display "    Log PIB EUA (1a diff):   KPSS should not reject (stationary)"
display _newline

display "  --- Cointegration ---"
display _newline
display "  Python/R JSON files:"
display "    `out_dir'/cointegration_results.json"
display "    `out_dir'/R/cointegration_results.json"
display _newline
display "  Key comparisons:"
display "    EG: PIB Brasil ~ PIB EUA: compare ADF(resid) statistic"
display "    Johansen: trace(r=0) statistic"
display _newline

display "  --- Structural Breaks ---"
display _newline
display "  Python/R JSON files:"
display "    `out_dir'/breaks_results.json"
display "    `out_dir'/R/breaks_results.json"
display _newline
display "  Key comparisons:"
display "    Chow PIB Brasil (bp=36): compare F statistic"
display "    CUSUM: compare qualitative decisions"
display _newline

********************************************************************************
* 6. Qualitative Decision Comparison Table
********************************************************************************
display "======================================================================"
display "6. Qualitative Decision Comparison"
display "======================================================================"
display _newline

display "  Expected qualitative agreement across platforms:"
display _newline
display "  " _dup(78) "-"
display "  Series                          | ADF    | PP     | KPSS   | ERS"
display "  " _dup(78) "-"
display "  Log PIB EUA (nivel)             | FAIL   | FAIL   | REJECT | FAIL"
display "  Log PIB EUA (1a diferenca)      | REJECT | REJECT | FAIL   | REJECT"
display "  Log PIB Brasil (nivel)          | FAIL   | FAIL   | REJECT | FAIL"
display "  Log PIB Brasil (1a diferenca)   | REJECT | REJECT | FAIL   | REJECT"
display "  " _dup(78) "-"
display _newline
display "  Legend:"
display "    ADF/PP/ERS: H0 = unit root.  REJECT => stationary"
display "    KPSS:       H0 = stationary. REJECT => unit root evidence"
display _newline
display "  All four tests should agree:"
display "    Levels:      non-stationary (I(1))"
display "    Differences: stationary (I(0))"
display _newline

display "  " _dup(78) "-"
display "  Cointegration                   | Python | R      | Stata"
display "  " _dup(78) "-"
display "  PIB Brasil ~ PIB EUA (EG)       | check  | check  | check"
display "  PIB Brasil x PIB EUA (Johansen) | check  | check  | check"
display "  " _dup(78) "-"
display _newline

display "  " _dup(78) "-"
display "  Structural Breaks               | Python | R      | Stata"
display "  " _dup(78) "-"
display "  Chow PIB Brasil (bp=36)         | check  | check  | check"
display "  CUSUM PIB Brasil                | check  | check  | check"
display "  CUSUM PIB EUA                   | check  | check  | check"
display "  " _dup(78) "-"
display _newline

********************************************************************************
* 7. Notes on cross-platform differences
********************************************************************************
display "======================================================================"
display "7. Notes on Cross-Platform Differences"
display "======================================================================"
display _newline

display "  Potential sources of numerical differences:"
display _newline
display "  1. Lag selection: Stata's dfuller uses fixed lags; Python's"
display "     statsmodels uses AIC/BIC; R's urca has selectlags option"
display _newline
display "  2. Bandwidth estimation: PP and KPSS kernel bandwidth"
display "     estimators differ (Newey-West vs Andrews)"
display _newline
display "  3. Critical values: Table interpolation methods differ"
display "     across implementations (MacKinnon vs Osterwald-Lenum)"
display _newline
display "  4. Johansen: Stata's vecrank uses different normalizations"
display "     and small-sample corrections than R's ca.jo"
display _newline
display "  5. P-values: Some tests (KPSS, ZA) report interpolated"
display "     p-values differently or only provide critical values"
display _newline
display "  6. Synthetic data: RNG differences across platforms mean"
display "     synthetic data is NOT comparable — only GDP data (from"
display "     identical CSV files) provides exact comparisons"
display _newline
display "  Tolerance guidelines:"
display "    - Test statistics: |diff| < 1e-4 (for identical data)"
display "    - P-values: |diff| < 0.01"
display "    - Qualitative decisions: should match in all cases"
display _newline

display "======================================================================"
display "Done. For automated comparison, use:"
display "  Python: examples/tests/notebooks/05_cross_validation.ipynb"
display "  R:      examples/tests/R/compare_results.R"
display "======================================================================"
