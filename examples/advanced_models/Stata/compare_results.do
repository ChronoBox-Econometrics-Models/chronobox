********************************************************************************
* compare_results.do
* Compare Python (chronobox) vs Stata results for advanced models
*
* Purpose: Load output CSVs from both Python and Stata runs, compute
*          agreement metrics where possible, and produce a summary report.
*
* Inputs:
*   examples/advanced_models/outputs/          (Python results)
*   examples/advanced_models/outputs/Stata/    (Stata results)
*
* Outputs:
*   examples/advanced_models/outputs/Stata/comparison_report_stata.txt
*   examples/advanced_models/outputs/Stata/comparison_summary_stata.csv
*
* Note: Due to Stata's limitations with advanced models (FAVAR, TVP-VAR,
* GVAR), comparisons are partial. This script focuses on what CAN be
* compared: factor extraction (PCA), rolling coefficient trends, and
* individual country VAR coefficients.
********************************************************************************

clear all
set more off
set seed 42
version 16

* --- Paths -------------------------------------------------------------------
local base_dir ".."
local py_out_dir "`base_dir'/outputs"
local stata_out_dir "`base_dir'/outputs/Stata"

* --- Start Report ------------------------------------------------------------
display as text ""
display as text "=============================================="
display as text " Comparison: Python (chronobox) vs Stata"
display as text " Advanced Models Validation"
display as text "=============================================="

log using "`stata_out_dir'/comparison_report_stata.txt", text replace

display as text ""
display as text "======================================================================"
display as text " COMPARISON REPORT: Python (chronobox) vs Stata"
display as text " Advanced Models Validation"
display as text "======================================================================"
display as text ""
display as text " Generated: $S_DATE $S_TIME"
display as text ""

* Prepare summary storage
tempfile summary_data
preserve
    clear
    generate str15 model = ""
    generate str40 metric = ""
    generate double value = .
    generate str10 status = ""
    save `summary_data', replace
restore

********************************************************************************
* 1. FAVAR Comparison: PCA Factors
*
* Compare PCA factors extracted by Python and Stata. Due to sign/scale
* indeterminacy in PCA, we compute absolute correlations.
********************************************************************************
display as text ""
display as text "----------------------------------------------------------------------"
display as text " 1. FAVAR - Factor-Augmented VAR"
display as text "----------------------------------------------------------------------"
display as text ""

* Check if Python factor file exists
capture confirm file "`py_out_dir'/favar_factors.csv"
local py_factors_exist = (_rc == 0)

* Check if Stata factor file exists
capture confirm file "`stata_out_dir'/favar_factors_stata.csv"
local stata_factors_exist = (_rc == 0)

if `py_factors_exist' & `stata_factors_exist' {
    display as text "Factor comparison:"

    * Load Python factors
    preserve
        import delimited "`py_out_dir'/favar_factors.csv", clear
        describe
        local py_nobs = _N
        display as text "  Python factors: `py_nobs' observations"

        * Rename for merge
        rename factor_1 py_factor_1
        rename factor_2 py_factor_2
        rename factor_3 py_factor_3

        tempfile py_factors
        save `py_factors', replace
    restore

    * Load Stata factors
    preserve
        import delimited "`stata_out_dir'/favar_factors_stata.csv", clear
        local stata_nobs = _N
        display as text "  Stata factors : `stata_nobs' observations"

        rename factor_1 stata_factor_1
        rename factor_2 stata_factor_2
        rename factor_3 stata_factor_3

        * Merge on observation number (dates may differ in format)
        generate obs_id = _n

        tempfile stata_factors
        save `stata_factors', replace
    restore

    * Since date formats may differ, merge on observation order
    * (both should have the same number of observations from same data)
    preserve
        use `py_factors', clear
        generate obs_id = _n
        merge 1:1 obs_id using `stata_factors', nogenerate

        * Compute correlations between all factor pairs
        display as text ""
        display as text "  Cross-correlation matrix |cor(Py, Stata)|:"
        display as text "  (Absolute values due to PCA sign indeterminacy)"
        display as text ""

        * Compute pairwise correlations
        forvalues i = 1/3 {
            forvalues j = 1/3 {
                quietly correlate py_factor_`i' stata_factor_`j'
                local cor_`i'_`j' = abs(r(rho))
                local cor_`i'_`j'_fmt : display %6.4f `cor_`i'_`j''
            }
        }

        display as text "           Stata_F1  Stata_F2  Stata_F3"
        display as text "  Py_F1    `cor_1_1_fmt'    `cor_1_2_fmt'    `cor_1_3_fmt'"
        display as text "  Py_F2    `cor_2_1_fmt'    `cor_2_2_fmt'    `cor_2_3_fmt'"
        display as text "  Py_F3    `cor_3_1_fmt'    `cor_3_2_fmt'    `cor_3_3_fmt'"

        * Find best matches
        display as text ""
        display as text "  Best factor matches:"
        forvalues i = 1/3 {
            * Find max correlation across Stata factors
            local best_cor = 0
            local best_j = 1
            forvalues j = 1/3 {
                if `cor_`i'_`j'' > `best_cor' {
                    local best_cor = `cor_`i'_`j''
                    local best_j = `j'
                }
            }
            local status = cond(`best_cor' > 0.5, "PASS", "CHECK")
            local best_cor_fmt : display %6.4f `best_cor'
            display as text "    Py_F`i' -> Stata_F`best_j' (|cor| = `best_cor_fmt') [`status']"

            * Save to summary
            preserve
                use `summary_data', clear
                local new_n = _N + 1
                set obs `new_n'
                quietly replace model = "FAVAR" in `new_n'
                quietly replace metric = "factor_`i'_abs_cor" in `new_n'
                quietly replace value = `best_cor' in `new_n'
                quietly replace status = "`status'" in `new_n'
                save `summary_data', replace
            restore
        }

        display as text ""
        display as text "  Note: PCA factors have sign and ordering indeterminacy."
        display as text "  High |cor| > 0.8 indicates excellent agreement."
        display as text "  |cor| > 0.5 is acceptable for cross-validation."
    restore
}
else {
    display as text "  SKIPPED: Factor files not found."
    if !`py_factors_exist' {
        display as text "    Missing: `py_out_dir'/favar_factors.csv"
    }
    if !`stata_factors_exist' {
        display as text "    Missing: `stata_out_dir'/favar_factors_stata.csv"
        display as text "    Run 01_favar_validation.do first."
    }
}

********************************************************************************
* 2. TVP-VAR Comparison: Coefficient Paths
*
* Compare rolling window coefficients (Stata) with Kalman filter
* coefficients (Python). These are fundamentally different methods,
* so we compare trends, not exact values.
********************************************************************************
display as text ""
display as text "----------------------------------------------------------------------"
display as text " 2. TVP-VAR - Time-Varying Parameter VAR"
display as text "----------------------------------------------------------------------"
display as text ""

capture confirm file "`py_out_dir'/tvp_coefficients.csv"
local py_tvp_exist = (_rc == 0)

capture confirm file "`stata_out_dir'/tvp_rolling_coefficients_stata.csv"
local stata_tvp_exist = (_rc == 0)

if `py_tvp_exist' & `stata_tvp_exist' {
    display as text "Coefficient path comparison:"
    display as text "  Python method: Kalman filter / MLE (smooth coefficients)"
    display as text "  Stata method : Rolling window VAR (step-function coefficients)"
    display as text ""
    display as text "  NOTE: These methods produce fundamentally different output."
    display as text "  Rolling window gives discrete jumps; Kalman gives smooth paths."
    display as text "  Compare general trends, not point estimates."
    display as text ""

    * Load Python coefficients
    preserve
        import delimited "`py_out_dir'/tvp_coefficients.csv", clear
        local py_tvp_nobs = _N
        local py_tvp_ncols = c(k)
        display as text "  Python: `py_tvp_nobs' obs x `py_tvp_ncols' columns"
        describe, short
    restore

    * Load Stata coefficients
    preserve
        import delimited "`stata_out_dir'/tvp_rolling_coefficients_stata.csv", clear
        local stata_tvp_nobs = _N
        local stata_tvp_ncols = c(k)
        display as text "  Stata : `stata_tvp_nobs' obs x `stata_tvp_ncols' columns"
        describe, short

        display as text ""
        display as text "  Stata coefficient summary (rolling window):"
        summarize, separator(0)
    restore

    * Record that both files exist
    preserve
        use `summary_data', clear
        local new_n = _N + 1
        set obs `new_n'
        quietly replace model = "TVP-VAR" in `new_n'
        quietly replace metric = "both_files_exist" in `new_n'
        quietly replace value = 1 in `new_n'
        quietly replace status = "PASS" in `new_n'
        save `summary_data', replace
    restore

    display as text ""
    display as text "  Detailed comparison requires date alignment and common columns."
    display as text "  For visual comparison, plot both coefficient paths together."
    display as text "  Focus on: (1) trend direction, (2) regime changes, (3) magnitudes."
}
else {
    display as text "  SKIPPED: Coefficient files not found."
    if !`py_tvp_exist' {
        display as text "    Missing: `py_out_dir'/tvp_coefficients.csv"
    }
    if !`stata_tvp_exist' {
        display as text "    Missing: `stata_out_dir'/tvp_rolling_coefficients_stata.csv"
        display as text "    Run 02_tvp_var_validation.do first."
    }
}

* --- TVP-VAR Volatility ---
capture confirm file "`py_out_dir'/tvp_volatility.csv"
local py_vol_exist = (_rc == 0)

capture confirm file "`stata_out_dir'/tvp_rolling_volatility_stata.csv"
local stata_vol_exist = (_rc == 0)

if `py_vol_exist' & `stata_vol_exist' {
    display as text ""
    display as text "Volatility comparison:"

    preserve
        import delimited "`stata_out_dir'/tvp_rolling_volatility_stata.csv", clear
        display as text "  Stata rolling volatility:"
        summarize, separator(0)
    restore

    preserve
        use `summary_data', clear
        local new_n = _N + 1
        set obs `new_n'
        quietly replace model = "TVP-VAR" in `new_n'
        quietly replace metric = "volatility_files_exist" in `new_n'
        quietly replace value = 1 in `new_n'
        quietly replace status = "PASS" in `new_n'
        save `summary_data', replace
    restore
}

********************************************************************************
* 3. GVAR Comparison: Country Coefficients
*
* Compare country-level VARX coefficients from Stata with chronobox
* GVAR coefficients. This is the most direct comparison possible,
* since Stata cannot produce the full GVAR global solution.
********************************************************************************
display as text ""
display as text "----------------------------------------------------------------------"
display as text " 3. GVAR - Global VAR"
display as text "----------------------------------------------------------------------"
display as text ""

capture confirm file "`stata_out_dir'/gvar_coefficients_stata.csv"
local stata_gvar_exist = (_rc == 0)

if `stata_gvar_exist' {
    display as text "Country VARX coefficients (Stata):"

    preserve
        import delimited "`stata_out_dir'/gvar_coefficients_stata.csv", clear

        display as text "  Total coefficient rows: " _N
        display as text ""

        * Summary by country
        display as text "  Coefficients per country:"
        tabulate country

        * Display significant coefficients (|t| > 2)
        display as text ""
        display as text "  Significant coefficients (|t| > 2.0):"
        list country equation variable coef t_stat if abs(t_stat) > 2.0, ///
            separator(0) noobs abbreviate(20)

        * Count significant foreign variable effects
        display as text ""
        display as text "  Foreign (star) variable significance:"
        generate is_star = strpos(variable, "_star") > 0
        generate is_sig = abs(t_stat) > 2.0
        tabulate is_star is_sig, row

        drop is_star is_sig
    restore

    preserve
        use `summary_data', clear
        local new_n = _N + 1
        set obs `new_n'
        quietly replace model = "GVAR" in `new_n'
        quietly replace metric = "country_varx_estimated" in `new_n'
        quietly replace value = 1 in `new_n'
        quietly replace status = "PASS" in `new_n'
        save `summary_data', replace
    restore

    display as text ""
    display as text "  NOTE: Stata provides country-level VARX only."
    display as text "  Global IRFs and spillover tables require the stacked GVAR system,"
    display as text "  which is NOT available in native Stata."
    display as text "  Compare: coefficient signs, foreign variable significance patterns."
}
else {
    display as text "  SKIPPED: Stata GVAR results not found."
    display as text "  Run 03_gvar_validation.do first."
}

********************************************************************************
* Overall Summary
********************************************************************************
display as text ""
display as text "======================================================================"
display as text " OVERALL COMPARISON SUMMARY"
display as text "======================================================================"
display as text ""
display as text "Methodology differences (by design):"
display as text ""
display as text "  FAVAR:"
display as text "    Python: PCA + VAR (two-step, synthetic factors)"
display as text "    Stata : PCA + VAR (same two-step approach)"
display as text "    -> Most comparable of the three models"
display as text "    -> Factor sign/order indeterminacy is expected"
display as text ""
display as text "  TVP-VAR:"
display as text "    Python: Kalman filter / MLE (smooth coefficient paths)"
display as text "    Stata : Rolling window VAR (discrete step-function)"
display as text "    -> Fundamentally different estimation approaches"
display as text "    -> Compare trends and regimes, not point estimates"
display as text ""
display as text "  GVAR:"
display as text "    Python: Full GVAR with stacking and global IRFs"
display as text "    Stata : Country-level VARX only (no global solution)"
display as text "    -> Only country-specific coefficients can be compared"
display as text "    -> Spillover analysis NOT possible in Stata"
display as text ""
display as text "Stata Limitations Summary:"
display as text "  - NO native FAVAR command (PCA+VAR approximation works)"
display as text "  - NO native TVP-VAR command (rolling window is crude proxy)"
display as text "  - NO native GVAR command (country VARX only, no stacking)"
display as text "  - For full advanced model validation, use R (bvarsv, BGVAR)"
display as text "    or Python (chronobox)"

log close

display as text ""
display as text "  Report saved: `stata_out_dir'/comparison_report_stata.txt"

* --- Save summary CSV --------------------------------------------------------
preserve
    use `summary_data', clear
    drop if model == ""

    if _N > 0 {
        export delimited using "`stata_out_dir'/comparison_summary_stata.csv", replace
        display as text "  Summary saved: `stata_out_dir'/comparison_summary_stata.csv"

        display as text ""
        display as text "Comparison Summary:"
        list, separator(0) noobs
    }
    else {
        display as text ""
        display as text "  No comparison metrics computed."
        display as text "  Run individual validation scripts first."
    }
restore

display as text ""
display as text "=== Comparison Complete ==="
display as text ""
display as text "For full cross-validation of advanced models, see also:"
display as text "  R scripts  : examples/advanced_models/R/"
display as text "  Python     : examples/advanced_models/notebooks/"
display as text "  Solutions  : examples/advanced_models/solutions/"
