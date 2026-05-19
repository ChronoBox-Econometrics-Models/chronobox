********************************************************************************
* 02_tvp_var_validation.do
* TVP-VAR (Time-Varying Parameter VAR) validation in Stata
*
* Purpose: Cross-validate chronobox TVP-VAR results using Stata.
*
* IMPORTANT LIMITATION:
*   Stata does NOT have a native TVP-VAR command. Time-varying parameter
*   models require state-space estimation (Kalman filter) or Bayesian
*   MCMC methods that are not part of Stata's standard VAR toolkit.
*
*   This script implements a ROLLING WINDOW VAR as an approximation:
*     - Estimate a standard VAR on a moving window of fixed size
*     - Extract coefficients at each window position
*     - The resulting coefficient paths approximate time-variation
*
*   This is NOT equivalent to a true TVP-VAR (which uses a Kalman filter
*   or Gibbs sampler for smooth coefficient evolution), but it captures
*   the key idea of parameters changing over time.
*
* SSC Packages (for true TVP-VAR):
*   - ssc install tvpvar   : if available (check SSC; limited support)
*   - Stata 17+ has some state-space capabilities (sspace command)
*     but these require manual specification for TVP-VAR
*
* Data: examples/advanced_models/data/us_macro_quarterly.csv
*       (200 quarterly obs: inflation, gdp, fed_funds)
*
* Outputs: examples/advanced_models/outputs/Stata/
*   - tvp_rolling_coefficients_stata.csv : time-varying coefficients
*   - tvp_rolling_volatility_stata.csv   : time-varying residual volatility
*   - tvp_var_results_stata.txt          : estimation log
*
* Comparison notes:
*   - Rolling window VAR is a crude approximation to TVP-VAR
*   - Python (chronobox) uses Kalman filter; R (bvarsv) uses Bayesian MCMC
*   - Expect coefficient TRENDS to agree, not exact values
*   - Turning points in coefficient paths are the key comparison metric
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
display as text " TVP-VAR Validation (Rolling Window) - Stata"
display as text "=============================================="
display as text ""
display as text "--- Step 0: Loading data ---"

import delimited "`data_dir'/us_macro_quarterly.csv", clear

* Parse date and set time series
generate date_q = quarterly(substr(date, 1, 4) + "q" + ///
    string(ceil(real(substr(date, 6, 7)) / 3)), "YQ")
format date_q %tq
tsset date_q

* Use 3 variables to match chronobox TVP-VAR example
keep date date_q inflation gdp fed_funds
display as text "  Observations: " _N
display as text "  Variables: inflation, gdp, fed_funds"
summarize inflation gdp fed_funds, separator(0)

********************************************************************************
* Step 1: Full-sample VAR as baseline
*
* Before running the rolling window, estimate a standard VAR on the full
* sample. This provides a benchmark against which to judge time-variation.
********************************************************************************
display as text ""
display as text "--- Step 1: Full-sample VAR (baseline) ---"

* Lag selection
varsoc inflation gdp fed_funds, maxlag(8)

* Estimate VAR(1) for consistency with TVP-VAR (typically lag 1)
var inflation gdp fed_funds, lags(1)
display as text "  Full-sample VAR(1) estimated."
display as text "  This is the constant-parameter benchmark."

********************************************************************************
* Step 2: Rolling Window VAR
*
* Estimate VAR(1) on a rolling window of size W=60 (15 years of quarterly
* data). At each step, we shift the window forward by 1 observation and
* re-estimate. The coefficient path approximates time-varying parameters.
*
* Window size choice:
*   - Too small (W<40): noisy coefficient estimates
*   - Too large (W>80): over-smoothed, misses short-lived changes
*   - W=60 is a standard choice for quarterly macro data
*
* Note: This is computationally intensive. Stata's -rolling- command
* could be used, but manual looping gives more control over output.
********************************************************************************
display as text ""
display as text "--- Step 2: Rolling Window VAR(1) ---"

local window_size = 60
local n_obs = _N
local n_windows = `n_obs' - `window_size' + 1

display as text "  Window size: `window_size' observations"
display as text "  Number of windows: `n_windows'"
display as text "  Estimating..."

* Prepare temporary dataset to store coefficients
* VAR(1) with 3 variables and constant: 3 equations x 4 params = 12 coefficients
* Plus 3 residual standard deviations = 15 values per window

tempfile coef_data vol_data
preserve

    * Create coefficient storage dataset
    clear
    set obs `n_windows'

    generate date_q = .
    format date_q %tq

    * Intercepts
    generate c_inflation = .
    generate c_gdp = .
    generate c_fed_funds = .

    * Coefficients: A(response, lag_variable)
    * Equation for inflation
    generate A_inflation_inflation = .
    generate A_inflation_gdp = .
    generate A_inflation_fed_funds = .

    * Equation for gdp
    generate A_gdp_inflation = .
    generate A_gdp_gdp = .
    generate A_gdp_fed_funds = .

    * Equation for fed_funds
    generate A_fed_funds_inflation = .
    generate A_fed_funds_gdp = .
    generate A_fed_funds_fed_funds = .

    * Residual standard deviations
    generate sigma_inflation = .
    generate sigma_gdp = .
    generate sigma_fed_funds = .

    save `coef_data', replace

restore

* Run rolling window estimation
forvalues w = 1/`n_windows' {

    * Define window boundaries
    local start = `w'
    local end = `w' + `window_size' - 1

    * Display progress every 20 windows
    if mod(`w', 20) == 1 {
        display as text "    Window `w' of `n_windows' (obs `start' to `end')"
    }

    * Estimate VAR(1) on the current window
    quietly var inflation gdp fed_funds if _n >= `start' & _n <= `end', lags(1)

    * Store the date corresponding to the END of the window
    * (this is the "current" date for the coefficient estimates)
    local end_date = date_q[`end']

    * Extract coefficients from each equation
    * Equation 1: inflation
    local c_infl    = _b[inflation:_cons]
    local a_infl_infl = _b[inflation:L.inflation]
    local a_infl_gdp  = _b[inflation:L.gdp]
    local a_infl_ff   = _b[inflation:L.fed_funds]

    * Equation 2: gdp
    local c_gdp     = _b[gdp:_cons]
    local a_gdp_infl  = _b[gdp:L.inflation]
    local a_gdp_gdp   = _b[gdp:L.gdp]
    local a_gdp_ff    = _b[gdp:L.fed_funds]

    * Equation 3: fed_funds
    local c_ff      = _b[fed_funds:_cons]
    local a_ff_infl   = _b[fed_funds:L.inflation]
    local a_ff_gdp    = _b[fed_funds:L.gdp]
    local a_ff_ff     = _b[fed_funds:L.fed_funds]

    * Extract residual standard deviations
    * Use e(Sigma) matrix
    matrix Sigma = e(Sigma)
    local sigma_infl = sqrt(Sigma[1,1])
    local sigma_gdp  = sqrt(Sigma[2,2])
    local sigma_ff   = sqrt(Sigma[3,3])

    * Store results in the coefficient dataset
    preserve
        use `coef_data', clear

        quietly replace date_q = `end_date' in `w'
        quietly replace c_inflation = `c_infl' in `w'
        quietly replace c_gdp = `c_gdp' in `w'
        quietly replace c_fed_funds = `c_ff' in `w'

        quietly replace A_inflation_inflation = `a_infl_infl' in `w'
        quietly replace A_inflation_gdp = `a_infl_gdp' in `w'
        quietly replace A_inflation_fed_funds = `a_infl_ff' in `w'

        quietly replace A_gdp_inflation = `a_gdp_infl' in `w'
        quietly replace A_gdp_gdp = `a_gdp_gdp' in `w'
        quietly replace A_gdp_fed_funds = `a_gdp_ff' in `w'

        quietly replace A_fed_funds_inflation = `a_ff_infl' in `w'
        quietly replace A_fed_funds_gdp = `a_ff_gdp' in `w'
        quietly replace A_fed_funds_fed_funds = `a_ff_ff' in `w'

        quietly replace sigma_inflation = `sigma_infl' in `w'
        quietly replace sigma_gdp = `sigma_gdp' in `w'
        quietly replace sigma_fed_funds = `sigma_ff' in `w'

        save `coef_data', replace
    restore
}

display as text "  Rolling estimation complete."

********************************************************************************
* Step 3: Export Coefficient Paths
*
* Save the time-varying coefficient and volatility estimates to CSV.
* These can be compared with chronobox (Kalman filter) and R (bvarsv).
********************************************************************************
display as text ""
display as text "--- Step 3: Exporting results ---"

preserve
    use `coef_data', clear

    * Generate string date for export
    generate date_str = string(date_q, "%tq")

    * Export coefficients
    drop sigma_inflation sigma_gdp sigma_fed_funds
    rename date_str date
    drop date_q
    order date c_* A_*
    export delimited using "`output_dir'/tvp_rolling_coefficients_stata.csv", replace
    display as text "  Saved tvp_rolling_coefficients_stata.csv"
restore

preserve
    use `coef_data', clear

    * Export volatility
    generate date_str = string(date_q, "%tq")
    keep date_str sigma_*
    rename date_str date
    order date sigma_*
    export delimited using "`output_dir'/tvp_rolling_volatility_stata.csv", replace
    display as text "  Saved tvp_rolling_volatility_stata.csv"
restore

********************************************************************************
* Step 4: Save Summary Log
********************************************************************************
display as text ""
display as text "--- Step 4: Summary ---"

log using "`output_dir'/tvp_var_results_stata.txt", text replace
display as text ""
display as text "TVP-VAR Validation - Rolling Window VAR Results (Stata)"
display as text "========================================================"
display as text ""
display as text "Method   : Rolling window VAR(1)"
display as text "Window   : `window_size' observations"
display as text "Variables: inflation, gdp, fed_funds"
display as text "Total obs: `n_obs'"
display as text "Windows  : `n_windows'"
display as text "Seed     : 42"
display as text ""
display as text "--- Full-sample VAR(1) (baseline) ---"
quietly var inflation gdp fed_funds, lags(1)
var inflation gdp fed_funds, lags(1)
display as text ""
display as text "--- Coefficient Summary Statistics ---"
display as text "(rolling window estimates)"

preserve
    use `coef_data', clear
    summarize c_* A_* sigma_*, separator(0) detail
restore

display as text ""
display as text "--- Limitations ---"
display as text "1. Stata has NO native TVP-VAR command"
display as text "2. Rolling window VAR is a discrete approximation to smooth TVP"
display as text "3. Window edges create boundary effects (abrupt changes)"
display as text "4. No formal smoothness prior (unlike Bayesian TVP-VAR)"
display as text "5. Coefficient paths are step functions, not continuous curves"
display as text ""
display as text "--- SSC Packages ---"
display as text "Check SSC for user-written TVP-VAR implementations:"
display as text "  ssc describe tvpvar   (if available)"
display as text "  ssc describe sspace   (state-space helper, if available)"
display as text ""
display as text "Stata 17+ state-space approach (sspace command):"
display as text "  Requires manual specification of the TVP-VAR state-space form."
display as text "  Transition equation: beta_t = beta_{t-1} + eta_t"
display as text "  Measurement equation: y_t = X_t * beta_t + epsilon_t"
display as text "  See Stata manual [TS] sspace for details."
display as text ""
display as text "--- Comparison Notes ---"
display as text "Python (chronobox): Kalman filter / MLE for smooth coefficients"
display as text "R (bvarsv)        : Bayesian MCMC (Primiceri 2005)"
display as text "Stata (this)      : Rolling window OLS (discrete approximation)"
display as text ""
display as text "Compare:"
display as text "  - General trends in coefficient paths"
display as text "  - Identification of structural breaks / regime changes"
display as text "  - Relative coefficient magnitudes across equations"
display as text "  - Volatility regimes (high vs low variance periods)"
display as text "Do NOT expect numerical agreement with smooth TVP-VAR methods."
log close

display as text "  Saved tvp_var_results_stata.txt"

********************************************************************************
* Summary
********************************************************************************
display as text ""
display as text "=============================================="
display as text " TVP-VAR Validation Complete"
display as text "=============================================="
display as text ""
display as text "Output files:"
display as text "  `output_dir'/tvp_rolling_coefficients_stata.csv"
display as text "  `output_dir'/tvp_rolling_volatility_stata.csv"
display as text "  `output_dir'/tvp_var_results_stata.txt"
display as text ""
display as text "LIMITATIONS:"
display as text "  1. Stata has NO native TVP-VAR command"
display as text "  2. Rolling window VAR is a crude approximation"
display as text "  3. No Kalman filter smoothing or Bayesian shrinkage"
display as text "  4. For true TVP-VAR, use Python (chronobox) or R (bvarsv)"
display as text ""
display as text "SSC ALTERNATIVES:"
display as text "  ssc install tvpvar     (if available, check with: ssc describe tvpvar)"
display as text "  ssc install bayesmixedlogit  (Bayesian methods, general)"
display as text "  Stata 17: sspace command for manual state-space specification"
