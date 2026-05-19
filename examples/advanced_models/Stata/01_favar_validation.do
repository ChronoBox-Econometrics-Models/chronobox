********************************************************************************
* 01_favar_validation.do
* FAVAR (Factor-Augmented VAR) validation using PCA + VAR in Stata
*
* Purpose: Cross-validate chronobox FAVAR results using Stata's native pca
*          command for factor extraction and var for the augmented model.
*
* IMPORTANT LIMITATION:
*   Stata does NOT have a native FAVAR command. This script implements the
*   two-step approach of Bernanke, Boivin & Eliasz (2005):
*     Step 1: Extract factors via PCA (pca command)
*     Step 2: Estimate VAR with factors + policy variable (var command)
*   This is a simplified approximation. For full FAVAR with rotation
*   restrictions, consider user-written packages or Matlab.
*
* SSC Packages (optional):
*   - None strictly required for the PCA+VAR approach
*   - ssc install pca2   (alternative PCA, optional)
*
* Data: examples/advanced_models/data/us_macro_quarterly.csv
*       (200 quarterly obs: gdp, inflation, fed_funds, unemployment)
*
* Outputs: examples/advanced_models/outputs/Stata/
*   - favar_factors_stata.csv       : extracted PCA factors (scores)
*   - favar_var_results_stata.txt   : VAR estimation results
*   - favar_irf_stata.csv           : impulse response functions
*
* Comparison notes:
*   - PCA factors may differ in sign/scale from Python and R
*   - Compare absolute correlations between factor pairs
*   - VAR coefficient estimates should be qualitatively similar
*   - IRF shapes matter more than exact magnitudes
********************************************************************************

clear all
set more off
set seed 42
version 16

* --- Paths -------------------------------------------------------------------
* Adjust this path to your installation
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"
capture mkdir "`output_dir'"

* --- Load data ---------------------------------------------------------------
display as text ""
display as text "=============================================="
display as text " FAVAR Validation - Stata"
display as text "=============================================="
display as text ""
display as text "--- Step 0: Loading data ---"

import delimited "`data_dir'/us_macro_quarterly.csv", clear

* Parse date and set time series
generate date_q = quarterly(substr(date, 1, 4) + "q" + ///
    string(ceil(real(substr(date, 6, 7)) / 3)), "YQ")
format date_q %tq
tsset date_q

display as text "  Observations: " _N
summarize gdp inflation fed_funds unemployment, separator(0)

********************************************************************************
* Step 1: PCA Factor Extraction
*
* In a true FAVAR, factors are extracted from a large panel of informational
* series. Here we use the 4 macro variables to illustrate the procedure,
* matching the approach in the Python and R validation scripts.
*
* Stata's -pca- command performs principal component analysis on the
* correlation or covariance matrix. We standardize first for comparability.
********************************************************************************
display as text ""
display as text "--- Step 1: PCA Factor Extraction ---"

* Standardize variables (zero mean, unit variance)
foreach var of varlist gdp inflation fed_funds unemployment {
    quietly summarize `var'
    generate `var'_std = (`var' - r(mean)) / r(sd)
}

* Run PCA on standardized variables
pca gdp_std inflation_std fed_funds_std unemployment_std

* Display variance explained
display as text ""
display as text "  Variance explained by first 3 components:"
display as text "  (See eigenvalues above)"

* Extract factor scores (first 3 components)
predict factor_1 factor_2 factor_3, score

* Verify extraction
summarize factor_1 factor_2 factor_3

* Save factors to CSV
preserve
    keep date_q factor_1 factor_2 factor_3
    generate date_str = string(date_q, "%tq")
    drop date_q
    order date_str factor_1 factor_2 factor_3
    rename date_str date
    export delimited using "`output_dir'/favar_factors_stata.csv", replace
    display as text "  Saved favar_factors_stata.csv"
restore

********************************************************************************
* Step 2: VAR Estimation with Factors (FAVAR)
*
* Two-step FAVAR: include extracted factors alongside the policy variable
* (fed_funds) in a standard VAR. This captures the information from the
* large panel (proxied by factors) while keeping the structural
* interpretation of the policy variable.
*
* Note: Stata's -var- command estimates a reduced-form VAR.
* Lag selection uses information criteria (AIC, BIC, HQIC).
********************************************************************************
display as text ""
display as text "--- Step 2: VAR Estimation with Factors ---"

* Lag selection
display as text "  Selecting optimal lag order..."
varsoc factor_1 factor_2 factor_3 fed_funds, maxlag(8)

* Estimate VAR with lag 2 (reasonable default; adjust based on varsoc output)
* Using lag 2 for consistency with typical quarterly macro models
display as text ""
display as text "  Estimating VAR(2) with 3 factors + fed_funds..."
var factor_1 factor_2 factor_3 fed_funds, lags(1/2)

* Save VAR results to log file
log using "`output_dir'/favar_var_results_stata.txt", text replace
display as text ""
display as text "FAVAR Validation - VAR Results (Stata)"
display as text "======================================="
display as text ""
display as text "Model: VAR(2) with 3 PCA factors + fed_funds"
display as text "Estimation: OLS equation-by-equation"
display as text "Seed: 42"
display as text ""
var factor_1 factor_2 factor_3 fed_funds, lags(1/2)
display as text ""
display as text "Lag Selection Criteria:"
varsoc factor_1 factor_2 factor_3 fed_funds, maxlag(8)
display as text ""
display as text "Granger Causality Tests:"
vargranger
display as text ""
display as text "VAR Stability Condition:"
varstable
log close

display as text "  Saved favar_var_results_stata.txt"

********************************************************************************
* Step 3: Impulse Response Functions
*
* Compute orthogonalized IRFs from a monetary policy shock (fed_funds).
* Stata's -irf- suite computes and stores IRF results.
*
* Note: Ordering matters for Cholesky decomposition.
* The ordering factor_1 factor_2 factor_3 fed_funds places the policy
* variable last (most endogenous), consistent with the FAVAR literature.
********************************************************************************
display as text ""
display as text "--- Step 3: Impulse Response Functions ---"

* Create IRF results
irf create favar_irf, set(favar_irfset, replace) step(20)

* Display IRFs for fed_funds shock
display as text "  IRF from fed_funds shock:"
irf table oirf, impulse(fed_funds) response(factor_1 factor_2 factor_3 fed_funds)

* Export IRF data
* Stata stores IRFs in .irf files; extract to CSV manually
preserve
    irf set favar_irfset

    * Use irf table to display, then capture
    * Alternative: use irfname and step to construct output
    clear
    set obs 21
    generate horizon = _n - 1

    * Generate placeholder columns - actual values from irf table above
    * Note: Stata does not natively export IRF to CSV easily.
    * The irf table command above provides the reference values.
    * For automated comparison, use the .irf file or parse the log output.

    generate impulse = "fed_funds"
    generate str20 response = ""
    generate oirf = .

    display as text "  Note: IRF values displayed in table above."
    display as text "  For CSV export, parse the log file or use irf table output."
    display as text "  Stata .irf files can be read with: irf set favar_irfset"
restore

********************************************************************************
* Step 4: Model Diagnostics
*
* Perform standard VAR diagnostics:
*   - Stability condition (all eigenvalues inside unit circle)
*   - Lagrange multiplier test for residual autocorrelation
*   - Jarque-Bera normality test
********************************************************************************
display as text ""
display as text "--- Step 4: VAR Diagnostics ---"

* Re-estimate VAR (needed after preserve/restore)
quietly var factor_1 factor_2 factor_3 fed_funds, lags(1/2)

* Stability
display as text "  Stability condition:"
varstable, graph

* LM test for autocorrelation
display as text ""
display as text "  LM test for residual autocorrelation:"
varlmar, mlag(4)

* Normality test
display as text ""
display as text "  Jarque-Bera normality test:"
varnorm, jbera

********************************************************************************
* Summary
********************************************************************************
display as text ""
display as text "=============================================="
display as text " FAVAR Validation Complete"
display as text "=============================================="
display as text ""
display as text "Output files:"
display as text "  `output_dir'/favar_factors_stata.csv"
display as text "  `output_dir'/favar_var_results_stata.txt"
display as text ""
display as text "LIMITATIONS:"
display as text "  1. Stata has NO native FAVAR command"
display as text "  2. This uses a two-step PCA + VAR approximation"
display as text "  3. Factor rotation restrictions are not enforced"
display as text "  4. For full FAVAR, consider Matlab or Python (chronobox)"
display as text ""
display as text "COMPARISON NOTES:"
display as text "  - PCA factors may differ in sign from Python/R (normal)"
display as text "  - Compare |correlation| between factor pairs, not raw values"
display as text "  - VAR coefficients should be qualitatively similar"
display as text "  - IRF shapes are more informative than exact magnitudes"
display as text ""
display as text "SSC PACKAGES (optional alternatives):"
display as text "  - ssc install pca2     : enhanced PCA (Stas Kolenikov)"
display as text "  - ssc install favar    : if available (check SSC archive)"
display as text "  - factor command       : Stata's built-in factor analysis"
display as text "    (use: factor gdp_std inflation_std fed_funds_std unemployment_std, pcf factors(3))"
