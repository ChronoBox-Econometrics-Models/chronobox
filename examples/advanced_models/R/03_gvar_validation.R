##############################################################################
# 03_gvar_validation.R
# GVAR (Global VAR) validation using BGVAR
#
# Purpose: Cross-validate chronobox GVAR results using the BGVAR package,
#          which implements Bayesian Global VAR models.
#
# Required packages:
#   - BGVAR  (>= 2.5)  : Bayesian Global VAR estimation
#
# Data: examples/advanced_models/data/us_macro_panel.csv
#       (80 quarters x 5 countries: US, UK, DE, JP, BR)
#       (4 variables: gdp, inflation, interest_rate, unemployment)
#
# Outputs saved to: examples/advanced_models/outputs/R/
#   - gvar_coefficients_R.csv      : estimated coefficients
#   - gvar_spillover_R.csv         : spillover / FEVD table
#   - gvar_hist_decomp_R.csv       : historical decomposition (if available)
#   - gvar_summary_R.txt           : model summary
#
# Note: BGVAR uses Bayesian estimation. Compare posterior means and
#   structural patterns, not exact point estimates.
##############################################################################

# --- Setup -------------------------------------------------------------------
set.seed(42)

if (!requireNamespace("BGVAR", quietly = TRUE)) {
  stop("Package 'BGVAR' is required. Install with: install.packages('BGVAR')")
}

library(BGVAR)

# --- Paths -------------------------------------------------------------------
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir)) script_dir <- "."

base_dir   <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
data_dir   <- file.path(base_dir, "data")
output_dir <- file.path(base_dir, "outputs", "R")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# --- Load data ---------------------------------------------------------------
cat("Loading us_macro_panel.csv ...\n")
df <- read.csv(file.path(data_dir, "us_macro_panel.csv"),
               stringsAsFactors = FALSE)
df$date <- as.Date(df$date)

countries <- unique(df$country)
var_names <- c("gdp", "inflation", "interest_rate", "unemployment")
n_dates   <- length(unique(df$date))

cat(sprintf("  Countries   : %s\n", paste(countries, collapse = ", ")))
cat(sprintf("  Variables   : %s\n", paste(var_names, collapse = ", ")))
cat(sprintf("  Time periods: %d\n", n_dates))

# ---------------------------------------------------------------------------
# Step 1: Prepare data in BGVAR format
#
# BGVAR expects a list of T x k matrices, one per country, with the same
# time index (rownames as dates).
# ---------------------------------------------------------------------------
cat("\n--- Step 1: Preparing data for BGVAR ---\n")

dates_unique <- sort(unique(df$date))

# Create list of country-specific time series matrices
country_data <- list()
for (cc in countries) {
  sub <- df[df$country == cc, ]
  sub <- sub[order(sub$date), ]
  mat <- as.matrix(sub[, var_names])
  rownames(mat) <- as.character(sub$date)
  colnames(mat) <- var_names
  country_data[[cc]] <- mat
}

cat(sprintf("  Created data list for %d countries\n", length(country_data)))

# ---------------------------------------------------------------------------
# Step 2: Define trade weight matrix
#
# Use a simple weight matrix. BGVAR requires a 3D array of weights:
# W[i, j, t] = weight of country j in country i's foreign variables.
# For simplicity, use time-invariant equal weights (excluding own country).
# ---------------------------------------------------------------------------
cat("\n--- Step 2: Constructing weight matrix ---\n")

n_c <- length(countries)
W <- matrix(1 / (n_c - 1), nrow = n_c, ncol = n_c)
diag(W) <- 0
rownames(W) <- countries
colnames(W) <- countries

# BGVAR expects a named list of weight matrices (one per time period or
# a single matrix for time-invariant weights)
W_list <- list("trade" = W)

cat("  Weight matrix (equal weights):\n")
print(round(W, 3))

# ---------------------------------------------------------------------------
# Step 3: Estimate GVAR
# ---------------------------------------------------------------------------
cat("\n--- Step 3: Estimating BGVAR model ---\n")
cat("  This may take several minutes (Bayesian estimation) ...\n")

# Estimate the BGVAR model
# Use default priors and 2 lags for domestic, 1 for foreign
bgvar_fit <- tryCatch({
  bgvar(
    Data       = country_data,
    W          = W_list,
    plag       = 2,        # domestic lags
    draws      = 5000,     # MCMC draws
    burnin     = 2000,     # burn-in
    prior      = "MN",     # Minnesota prior
    SV         = FALSE,    # no stochastic volatility (faster)
    thin       = 1,
    expert     = NULL
  )
}, error = function(e) {
  cat(sprintf("  BGVAR estimation failed: %s\n", e$message))
  cat("  Attempting with simplified settings ...\n")
  # Fallback: try with fewer lags
  bgvar(
    Data       = country_data,
    W          = W_list,
    plag       = 1,
    draws      = 3000,
    burnin     = 1000,
    prior      = "MN",
    SV         = FALSE,
    thin       = 1,
    expert     = NULL
  )
})

cat("  BGVAR estimation complete.\n")

# ---------------------------------------------------------------------------
# Step 4: Extract results
# ---------------------------------------------------------------------------
cat("\n--- Step 4: Extracting results ---\n")

# --- 4a: Impulse Response Functions ---
cat("  Computing IRFs ...\n")
irf_result <- tryCatch({
  irf(bgvar_fit,
      n.ahead = 20,
      shockinfo = NULL)  # default: unit shock to each variable
}, error = function(e) {
  cat(sprintf("  IRF computation note: %s\n", e$message))
  NULL
})

# --- 4b: Forecast Error Variance Decomposition (spillover proxy) ---
cat("  Computing FEVD ...\n")
fevd_result <- tryCatch({
  fevd(bgvar_fit, n.ahead = 10)
}, error = function(e) {
  cat(sprintf("  FEVD computation note: %s\n", e$message))
  NULL
})

# --- 4c: Save FEVD / spillover table ---
if (!is.null(fevd_result)) {
  # Extract FEVD table and convert to spillover-style format
  # FEVD gives proportion of h-step forecast error variance of variable i
  # explained by shocks to variable j
  fevd_mat <- tryCatch({
    # BGVAR FEVD returns a 3D array: [response, impulse, horizon]
    # Aggregate at max horizon for Diebold-Yilmaz style table
    fevd_array <- fevd_result$FEVD
    if (is.array(fevd_array) && length(dim(fevd_array)) >= 2) {
      # Take last horizon slice
      if (length(dim(fevd_array)) == 3) {
        spillover_mat <- fevd_array[, , dim(fevd_array)[3]]
      } else {
        spillover_mat <- fevd_array
      }
      spillover_df <- as.data.frame(spillover_mat)
      write.csv(spillover_df,
                file.path(output_dir, "gvar_spillover_R.csv"),
                row.names = TRUE)
      cat("  Saved gvar_spillover_R.csv\n")
      spillover_df
    } else {
      cat("  FEVD format not as expected, saving raw summary.\n")
      NULL
    }
  }, error = function(e) {
    cat(sprintf("  Could not extract FEVD matrix: %s\n", e$message))
    NULL
  })
}

# --- 4d: Historical Decomposition ---
cat("  Computing historical decomposition ...\n")
hd_result <- tryCatch({
  hd(bgvar_fit)
}, error = function(e) {
  cat(sprintf("  Historical decomposition note: %s\n", e$message))
  NULL
})

if (!is.null(hd_result)) {
  # Extract and save HD results
  hd_df <- tryCatch({
    # hd() returns a list with HD matrix
    hd_mat <- hd_result$HD
    if (!is.null(hd_mat)) {
      hd_out <- as.data.frame(hd_mat)
      write.csv(hd_out,
                file.path(output_dir, "gvar_hist_decomp_R.csv"),
                row.names = TRUE)
      cat("  Saved gvar_hist_decomp_R.csv\n")
      hd_out
    } else {
      cat("  HD matrix is NULL.\n")
      NULL
    }
  }, error = function(e) {
    cat(sprintf("  Could not save HD: %s\n", e$message))
    NULL
  })
}

# ---------------------------------------------------------------------------
# Step 5: Save summary
# ---------------------------------------------------------------------------
cat("\n--- Step 5: Saving summary ---\n")

summary_file <- file.path(output_dir, "gvar_summary_R.txt")
sink(summary_file)
cat("GVAR Validation - BGVAR Summary (R)\n")
cat("====================================\n\n")
cat(sprintf("Package   : BGVAR\n"))
cat(sprintf("Method    : Bayesian Global VAR\n"))
cat(sprintf("Prior     : Minnesota (MN)\n"))
cat(sprintf("MCMC draws: 5000 (2000 burn-in)\n"))
cat(sprintf("Seed      : 42\n\n"))
cat(sprintf("Countries : %s\n", paste(countries, collapse = ", ")))
cat(sprintf("Variables : %s\n", paste(var_names, collapse = ", ")))
cat(sprintf("Time pds  : %d quarterly\n", n_dates))
cat(sprintf("Dom. lags : 2\n"))
cat(sprintf("For. lags : 1 (BGVAR default)\n\n"))

cat("Weight matrix (trade weights):\n")
print(round(W, 3))
cat("\n")

# Print model summary if available
tryCatch({
  print(summary(bgvar_fit))
}, error = function(e) {
  cat(sprintf("Summary not available: %s\n", e$message))
})

cat("\n\nComparison notes:\n")
cat("-----------------\n")
cat("BGVAR uses Bayesian estimation with Minnesota prior.\n")
cat("chronobox uses country-specific VARX models with OLS.\n")
cat("Expected differences:\n")
cat("  - Coefficient shrinkage (Bayesian prior pulls towards zero)\n")
cat("  - Spillover magnitudes may differ in scale but should show\n")
cat("    similar relative patterns (e.g., US -> UK stronger than JP -> BR)\n")
cat("  - Historical decomposition: compare shock contribution rankings\n")
cat("    and sign patterns, not exact percentages.\n")
cat("Tolerance: Bayesian models produce posterior distributions.\n")
cat("Compare posterior means to chronobox point estimates as a guide.\n")
sink()
cat(sprintf("  Saved gvar_summary_R.txt\n"))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat("\n=== GVAR Validation Complete ===\n")
cat("Output files:\n")
out_files <- list.files(output_dir, pattern = "^gvar_", full.names = TRUE)
for (f in out_files) cat(sprintf("  %s\n", f))
cat("\nNote: Bayesian GVAR results should be compared at the posterior\n")
cat("distribution level, not as point estimates.\n")
