********************************************************************************
* 03_gvar_validation.do
* GVAR (Global VAR) validation in Stata
*
* Purpose: Cross-validate chronobox GVAR results using Stata.
*
* IMPORTANT LIMITATION:
*   Stata does NOT have a native GVAR command. The GVAR methodology
*   (Pesaran, Schuermann & Weiner, 2004; Dees, di Mauro, Pesaran &
*   Smith, 2007) requires:
*     1. Country-specific VARX models with foreign variables
*     2. A trade weight matrix to construct foreign variables
*     3. Stacking of country models into a global system
*
*   This script implements a PARTIAL approximation:
*     - Estimate separate VAR models per country
*     - Include trade-weighted foreign variables as exogenous regressors
*     - Compare country-level results with chronobox GVAR output
*
*   The full GVAR stacking and global IRF computation is NOT feasible
*   in Stata without specialized packages.
*
* SSC Packages:
*   - ssc install gvar     : user-written GVAR (check availability)
*   - ssc install xtvar    : panel VAR (related but not identical)
*   Alternative: GVAR Toolbox for Matlab (Smith & Galesi, 2014)
*
* Data: examples/advanced_models/data/us_macro_panel.csv
*       (80 quarters x 5 countries: US, UK, DE, JP, BR)
*       (4 variables: gdp, inflation, interest_rate, unemployment)
*
* Outputs: examples/advanced_models/outputs/Stata/
*   - gvar_country_var_stata.txt    : country-by-country VAR results
*   - gvar_coefficients_stata.csv   : estimated coefficients
*   - gvar_summary_stata.txt        : summary and limitations
*
* Comparison notes:
*   - Separate country VARs are NOT equivalent to a full GVAR
*   - The global solution (stacked system) affects IRFs and FEVD
*   - Compare individual country coefficient signs and magnitudes
*   - Spillover analysis requires the full stacked system
********************************************************************************

clear all
set more off
set seed 42
version 16

* --- Paths -------------------------------------------------------------------
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"
capture mkdir "`output_dir'"

* --- Load data ---------------------------------------------------------------
display as text ""
display as text "=============================================="
display as text " GVAR Validation (Country VARs) - Stata"
display as text "=============================================="
display as text ""
display as text "--- Step 0: Loading data ---"

import delimited "`data_dir'/us_macro_panel.csv", clear

* Inspect the data
display as text "  Total rows: " _N
tabulate country
summarize gdp inflation interest_rate unemployment

* Encode country as numeric for panel operations
encode country, generate(country_id)

* Parse dates
generate date_q = quarterly(substr(date, 1, 4) + "q" + ///
    string(ceil(real(substr(date, 6, 7)) / 3)), "YQ")
format date_q %tq

* Set panel structure
xtset country_id date_q
display as text "  Panel set: country_id x date_q"

********************************************************************************
* Step 1: Construct Trade Weight Matrix
*
* The GVAR approach requires a weight matrix W where W(i,j) is the weight
* of country j in constructing country i's foreign variables.
* Here we use equal weights (1/(N-1)) for simplicity, consistent with
* the Python and R validation scripts.
*
* In practice, weights are derived from bilateral trade data.
********************************************************************************
display as text ""
display as text "--- Step 1: Trade Weight Matrix ---"
display as text "  Using equal weights: w(i,j) = 1/(N-1) = 0.25"
display as text "  (5 countries, exclude own country)"
display as text ""
display as text "  Weight matrix:"
display as text "       US    UK    DE    JP    BR"
display as text "  US  0.00  0.25  0.25  0.25  0.25"
display as text "  UK  0.25  0.00  0.25  0.25  0.25"
display as text "  DE  0.25  0.25  0.00  0.25  0.25"
display as text "  JP  0.25  0.25  0.25  0.00  0.25"
display as text "  BR  0.25  0.25  0.25  0.25  0.00"

********************************************************************************
* Step 2: Construct Foreign (Star) Variables
*
* For each country i, the foreign variable x*_i is the weighted average
* of x_j for all j != i:
*   x*_i(t) = sum_{j!=i} w(i,j) * x_j(t)
*
* With equal weights: x*_i(t) = mean of x_j(t) for j != i
*
* These star variables enter as exogenous variables in each country's
* VARX model: y_i(t) = A_i * y_i(t-1) + B_i * y*_i(t) + u_i(t)
********************************************************************************
display as text ""
display as text "--- Step 2: Constructing Foreign (Star) Variables ---"

* For each variable, compute the cross-country mean excluding own country
* This equals the equal-weight foreign variable
foreach var of varlist gdp inflation interest_rate unemployment {
    * Compute overall mean by date
    bysort date_q: egen `var'_mean = mean(`var')

    * Foreign (star) variable: exclude own country
    * x*_i = (sum_all - x_i) / (N-1) = (N * mean - x_i) / (N-1)
    local n_countries = 5
    generate `var'_star = (`n_countries' * `var'_mean - `var') / (`n_countries' - 1)

    drop `var'_mean
}

display as text "  Created star variables: gdp_star, inflation_star, etc."
summarize *_star

********************************************************************************
* Step 3: Country-by-Country VARX Estimation
*
* Estimate a VAR for each country separately, with star variables as
* exogenous regressors. This is the first step of the GVAR approach.
*
* Note: In a full GVAR, these country models are stacked into a global
* system. Here we only estimate the individual models.
*
* Stata's -var- command with -exog()- option handles VARX estimation.
********************************************************************************
display as text ""
display as text "--- Step 3: Country-by-Country VARX Estimation ---"

log using "`output_dir'/gvar_country_var_stata.txt", text replace
display as text ""
display as text "GVAR Validation - Country VARX Results (Stata)"
display as text "================================================"
display as text ""
display as text "Method: Separate VARX(1) per country"
display as text "Domestic variables: gdp, inflation, interest_rate, unemployment"
display as text "Foreign variables: gdp_star, inflation_star, interest_rate_star, unemployment_star"
display as text "Weight matrix: equal weights (0.25)"
display as text ""

* Store results for CSV export
tempfile results_all
preserve
    clear
    generate str5 country = ""
    generate str30 equation = ""
    generate str30 variable = ""
    generate coef = .
    generate se = .
    generate t_stat = .
    generate p_value = .
    save `results_all', replace
restore

* Estimate VAR for each country
levelsof country, local(countries)
foreach cc of local countries {
    display as text ""
    display as text "=============================="
    display as text " Country: `cc'"
    display as text "=============================="

    * Estimate VARX(1) with star variables as exogenous
    * Note: Stata requires the data to be filtered for each country
    * since we're using cross-sectional VAR, not panel VAR
    preserve
        keep if country == "`cc'"
        tsset date_q

        * Lag selection for this country
        display as text ""
        display as text "Lag selection:"
        quietly varsoc gdp inflation interest_rate unemployment, maxlag(4)
        varsoc gdp inflation interest_rate unemployment, maxlag(4)

        * Estimate VARX(1) with foreign variables as exogenous
        display as text ""
        display as text "VARX(1) estimation:"
        var gdp inflation interest_rate unemployment, ///
            lags(1) ///
            exog(gdp_star inflation_star interest_rate_star unemployment_star)

        * Stability check
        display as text ""
        display as text "Stability:"
        varstable

        * Granger causality
        display as text ""
        display as text "Granger causality:"
        vargranger

        * Extract coefficients for CSV
        * Store key domestic VAR coefficients
        foreach eq in gdp inflation interest_rate unemployment {
            foreach xvar in gdp inflation interest_rate unemployment {
                local b = _b[`eq':L.`xvar']
                local se_val = _se[`eq':L.`xvar']
                local t_val = `b' / `se_val'
                local p_val = 2 * ttail(e(df_r), abs(`t_val'))

                preserve
                    use `results_all', clear
                    local new_n = _N + 1
                    set obs `new_n'
                    quietly replace country = "`cc'" in `new_n'
                    quietly replace equation = "`eq'" in `new_n'
                    quietly replace variable = "L.`xvar'" in `new_n'
                    quietly replace coef = `b' in `new_n'
                    quietly replace se = `se_val' in `new_n'
                    quietly replace t_stat = `t_val' in `new_n'
                    quietly replace p_value = `p_val' in `new_n'
                    save `results_all', replace
                restore
            }

            * Also store star variable coefficients
            foreach xvar in gdp_star inflation_star interest_rate_star unemployment_star {
                capture local b = _b[`eq':`xvar']
                if _rc == 0 {
                    local se_val = _se[`eq':`xvar']
                    local t_val = `b' / `se_val'
                    local p_val = 2 * ttail(e(df_r), abs(`t_val'))

                    preserve
                        use `results_all', clear
                        local new_n = _N + 1
                        set obs `new_n'
                        quietly replace country = "`cc'" in `new_n'
                        quietly replace equation = "`eq'" in `new_n'
                        quietly replace variable = "`xvar'" in `new_n'
                        quietly replace coef = `b' in `new_n'
                        quietly replace se = `se_val' in `new_n'
                        quietly replace t_stat = `t_val' in `new_n'
                        quietly replace p_value = `p_val' in `new_n'
                        save `results_all', replace
                    restore
                }
            }
        }
    restore
}

log close

display as text ""
display as text "  Saved gvar_country_var_stata.txt"

********************************************************************************
* Step 4: Export Coefficients
********************************************************************************
display as text ""
display as text "--- Step 4: Exporting coefficients ---"

preserve
    use `results_all', clear
    drop if country == ""
    export delimited using "`output_dir'/gvar_coefficients_stata.csv", replace
    display as text "  Saved gvar_coefficients_stata.csv"
    display as text "  Rows: " _N
restore

********************************************************************************
* Step 5: Summary and Limitations
********************************************************************************
display as text ""
display as text "--- Step 5: Summary ---"

log using "`output_dir'/gvar_summary_stata.txt", text replace
display as text ""
display as text "GVAR Validation Summary (Stata)"
display as text "================================"
display as text ""
display as text "Method: Country-specific VARX(1) with trade-weighted foreign variables"
display as text "Countries: US, UK, DE, JP, BR"
display as text "Variables: gdp, inflation, interest_rate, unemployment"
display as text "Foreign vars: equal-weighted cross-country averages (excluding own)"
display as text "Seed: 42"
display as text ""
display as text "CRITICAL LIMITATIONS:"
display as text "====================="
display as text ""
display as text "1. NO NATIVE GVAR COMMAND IN STATA"
display as text "   Stata does not have a built-in GVAR estimation command."
display as text "   The var command estimates single-country VAR/VARX models."
display as text ""
display as text "2. MISSING: GLOBAL STACKING"
display as text "   A true GVAR stacks all country VARX models into one global"
display as text "   system using a link matrix. This global solution is needed for:"
display as text "   - Global IRFs (how a US shock affects other countries)"
display as text "   - Forecast Error Variance Decomposition (spillovers)"
display as text "   - Forecasting with cross-country feedback"
display as text "   This script does NOT implement the global stacking."
display as text ""
display as text "3. SEPARATE ESTIMATION BIAS"
display as text "   Estimating country VARs separately ignores the simultaneous"
display as text "   determination of domestic and foreign variables. A full GVAR"
display as text "   accounts for this through the global solution."
display as text ""
display as text "4. SPILLOVER ANALYSIS NOT POSSIBLE"
display as text "   Without the stacked global system, we cannot compute:"
display as text "   - Generalized IRFs (Pesaran & Shin, 1998)"
display as text "   - FEVD-based spillover indices (Diebold & Yilmaz, 2012)"
display as text "   - Connectedness measures"
display as text ""
display as text "SSC PACKAGES:"
display as text "============="
display as text "  ssc describe gvar    : Check if a user-written GVAR exists"
display as text "  ssc install xtvar    : Panel VAR (Abrigo & Love, 2016)"
display as text "                         Related but not identical to GVAR"
display as text "  ssc install pvar     : Panel VAR alternative"
display as text ""
display as text "ALTERNATIVE SOFTWARE:"
display as text "====================="
display as text "  - GVAR Toolbox (Matlab): Smith & Galesi (2014)"
display as text "    https://sites.google.com/site/gaborpesticontextgvar/"
display as text "  - BGVAR (R package): Bayesian GVAR, Crespo Cuaresma et al."
display as text "  - chronobox (Python): Frequentist GVAR implementation"
display as text ""
display as text "COMPARISON NOTES:"
display as text "================="
display as text "  What CAN be compared:"
display as text "  - Individual country VARX coefficient signs and magnitudes"
display as text "  - Significance of foreign (star) variable coefficients"
display as text "  - Direction of cross-country spillovers from exogenous terms"
display as text ""
display as text "  What CANNOT be compared:"
display as text "  - Global IRFs (requires stacked system)"
display as text "  - FEVD / spillover table (requires stacked system)"
display as text "  - Forecast accuracy (requires global solution)"
display as text ""
display as text "REFERENCES:"
display as text "==========="
display as text "  Pesaran, M.H., Schuermann, T., & Weiner, S.M. (2004)."
display as text "    Modeling Regional Interdependencies Using a Global"
display as text "    Error-Correcting Macroeconometric Model. JBES, 22(2)."
display as text ""
display as text "  Dees, S., di Mauro, F., Pesaran, M.H., & Smith, L.V. (2007)."
display as text "    Exploring the International Linkages of the Euro Area."
display as text "    Journal of Applied Econometrics, 22(1)."
log close

display as text "  Saved gvar_summary_stata.txt"

********************************************************************************
* Summary
********************************************************************************
display as text ""
display as text "=============================================="
display as text " GVAR Validation Complete"
display as text "=============================================="
display as text ""
display as text "Output files:"
display as text "  `output_dir'/gvar_country_var_stata.txt"
display as text "  `output_dir'/gvar_coefficients_stata.csv"
display as text "  `output_dir'/gvar_summary_stata.txt"
display as text ""
display as text "LIMITATIONS:"
display as text "  1. Stata has NO native GVAR command"
display as text "  2. Only country-level VARX estimated (no global stacking)"
display as text "  3. Cannot compute global IRFs or spillover indices"
display as text "  4. For full GVAR: use GVAR Toolbox (Matlab), BGVAR (R),"
display as text "     or chronobox (Python)"
