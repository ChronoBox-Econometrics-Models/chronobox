##############################################################################
# 02_tvp_var_validation.R
# TVP-VAR (Time-Varying Parameter VAR) validation using bvarsv
#
# Purpose: Cross-validate chronobox TVP-VAR results using the bvarsv package
#          which implements Primiceri (2005) TVP-VAR with stochastic volatility.
#
# Required packages:
#   - bvarsv  (>= 1.1)  : Bayesian TVP-VAR with stochastic volatility
#
# Data: examples/advanced_models/data/us_macro_quarterly.csv
#       (200 quarterly obs: gdp, inflation, fed_funds, unemployment)
#       We use 3 variables (inflation, gdp, fed_funds) to match chronobox.
#
# Outputs saved to: examples/advanced_models/outputs/R/
#   - tvp_coefficients_R.csv  : posterior means of time-varying coefficients
#   - tvp_volatility_R.csv    : posterior means of time-varying volatility
#   - tvp_summary_R.txt       : model summary and diagnostics
#
# Note on comparison: bvarsv is Bayesian (MCMC), chronobox uses a Kalman
#   filter / MLE approach. Results will differ in levels but should show
#   similar time-varying patterns. Compare trends and turning points,
#   not exact point estimates.
##############################################################################

# --- Setup -------------------------------------------------------------------
set.seed(42)

if (!requireNamespace("bvarsv", quietly = TRUE)) {
  stop("Package 'bvarsv' is required. Install with: install.packages('bvarsv')")
}

library(bvarsv)

# --- Paths -------------------------------------------------------------------
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir)) script_dir <- "."

base_dir   <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
data_dir   <- file.path(base_dir, "data")
output_dir <- file.path(base_dir, "outputs", "R")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# --- Load data ---------------------------------------------------------------
cat("Loading us_macro_quarterly.csv ...\n")
df <- read.csv(file.path(data_dir, "us_macro_quarterly.csv"),
               stringsAsFactors = FALSE)
df$date <- as.Date(df$date)

# Use 3 variables to match chronobox TVP-VAR example
var_names <- c("inflation", "gdp", "fed_funds")
Y <- as.matrix(df[, var_names])
dates <- df$date

cat(sprintf("  Observations: %d\n", nrow(Y)))
cat(sprintf("  Variables   : %s\n", paste(var_names, collapse = ", ")))

# ---------------------------------------------------------------------------
# Step 1: Estimate TVP-VAR with stochastic volatility using bvarsv
#
# bvar.sv.tvp() fits a Bayesian TVP-VAR(1) via Gibbs sampling.
# Model:
#   y_t = c_t + B_t * y_{t-1} + u_t,   u_t ~ N(0, H_t)
# where B_t and H_t evolve over time.
# ---------------------------------------------------------------------------
cat("\n--- Step 1: Estimating TVP-VAR with bvarsv ---\n")
cat("  This may take a few minutes (MCMC sampling) ...\n")

# MCMC settings
nrep  <- 5000   # total draws
nburn <- 2000   # burn-in

tvp_fit <- bvar.sv.tvp(Y,
                       nrep  = nrep,
                       nburn = nburn,
                       tau   = 40)   # training sample size

cat("  MCMC estimation complete.\n")
cat(sprintf("  Total draws: %d, Burn-in: %d, Effective: %d\n",
            nrep, nburn, nrep - nburn))

# ---------------------------------------------------------------------------
# Step 2: Extract time-varying coefficients (posterior means)
#
# bvarsv stores coefficients in tvp_fit$Beta.postmean
# Dimensions: (T x (k*k + k)) where k = number of variables
#   - first k columns: intercepts
#   - remaining k*k columns: VAR(1) coefficients
# ---------------------------------------------------------------------------
cat("\n--- Step 2: Extracting time-varying coefficients ---\n")

k <- ncol(Y)
T_eff <- nrow(tvp_fit$Beta.postmean)

# Dates for the effective sample (after lag + training)
# bvarsv uses tau observations as training, plus 1 lag
start_idx <- nrow(Y) - T_eff + 1
dates_eff <- dates[start_idx:nrow(Y)]

# Build coefficient dataframe
# Beta.postmean columns: intercept_1, ..., intercept_k, B_11, B_12, ..., B_kk
coef_df <- data.frame(date = dates_eff)

# Intercepts
for (i in 1:k) {
  col_name <- paste0("intercept_", var_names[i])
  coef_df[[col_name]] <- tvp_fit$Beta.postmean[, i]
}

# VAR(1) coefficients A(i,j) = effect of variable j on equation i
for (i in 1:k) {
  for (j in 1:k) {
    col_name <- paste0("A_", var_names[i], "_", var_names[j])
    col_idx <- k + (i - 1) * k + j
    coef_df[[col_name]] <- tvp_fit$Beta.postmean[, col_idx]
  }
}

write.csv(coef_df, file.path(output_dir, "tvp_coefficients_R.csv"),
          row.names = FALSE)
cat(sprintf("  Saved tvp_coefficients_R.csv (%d rows)\n", nrow(coef_df)))

# ---------------------------------------------------------------------------
# Step 3: Extract time-varying volatility (posterior means)
#
# H.postmean contains posterior means of the covariance matrix elements.
# Dimensions: (T x k*(k+1)/2) for the lower triangle of H_t
# ---------------------------------------------------------------------------
cat("\n--- Step 3: Extracting time-varying volatility ---\n")

vol_df <- data.frame(date = dates_eff)

# H.postmean stores the unique elements of the covariance matrix
# For k=3: H_11, H_21, H_22, H_31, H_32, H_33
idx <- 1
for (i in 1:k) {
  for (j in 1:i) {
    col_name <- paste0("sigma_", var_names[i], "_", var_names[j])
    vol_df[[col_name]] <- tvp_fit$H.postmean[, idx]
    idx <- idx + 1
  }
}

write.csv(vol_df, file.path(output_dir, "tvp_volatility_R.csv"),
          row.names = FALSE)
cat(sprintf("  Saved tvp_volatility_R.csv (%d rows)\n", nrow(vol_df)))

# ---------------------------------------------------------------------------
# Step 4: Save summary
# ---------------------------------------------------------------------------
cat("\n--- Step 4: Saving summary ---\n")

summary_file <- file.path(output_dir, "tvp_summary_R.txt")
sink(summary_file)
cat("TVP-VAR Validation - bvarsv Summary (R)\n")
cat("========================================\n\n")
cat(sprintf("Package : bvarsv\n"))
cat(sprintf("Method  : Primiceri (2005) TVP-VAR with stochastic volatility\n"))
cat(sprintf("MCMC    : %d draws, %d burn-in\n", nrep, nburn))
cat(sprintf("Training: tau = 40\n"))
cat(sprintf("Seed    : 42\n\n"))
cat(sprintf("Variables: %s\n", paste(var_names, collapse = ", ")))
cat(sprintf("Total obs: %d\n", nrow(Y)))
cat(sprintf("Effective: %d time points\n\n", T_eff))

cat("Coefficient statistics (posterior means, averaged over time):\n")
cat("-------------------------------------------------------------\n")
for (col in names(coef_df)[-1]) {
  vals <- coef_df[[col]]
  cat(sprintf("  %-30s  mean=%.4f  sd=%.4f  range=[%.4f, %.4f]\n",
              col, mean(vals), sd(vals), min(vals), max(vals)))
}

cat("\nVolatility statistics (posterior means, averaged over time):\n")
cat("-------------------------------------------------------------\n")
for (col in names(vol_df)[-1]) {
  vals <- vol_df[[col]]
  cat(sprintf("  %-30s  mean=%.4f  sd=%.4f  range=[%.4f, %.4f]\n",
              col, mean(vals), sd(vals), min(vals), max(vals)))
}

cat("\n\nComparison notes:\n")
cat("-----------------\n")
cat("bvarsv uses Bayesian MCMC (Gibbs sampler) while chronobox uses\n")
cat("Kalman filter / MLE. Exact coefficient values WILL differ.\n")
cat("Compare: (1) trends and turning points in coefficient paths,\n")
cat("         (2) relative magnitudes across equations,\n")
cat("         (3) volatility regimes (high vs low vol periods).\n")
cat("Tolerance: focus on qualitative pattern agreement, not point estimates.\n")
sink()
cat(sprintf("  Saved tvp_summary_R.txt\n"))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat("\n=== TVP-VAR Validation Complete ===\n")
cat("Output files:\n")
cat(sprintf("  %s\n", file.path(output_dir, "tvp_coefficients_R.csv")))
cat(sprintf("  %s\n", file.path(output_dir, "tvp_volatility_R.csv")))
cat(sprintf("  %s\n", file.path(output_dir, "tvp_summary_R.txt")))
cat("\nNote: Bayesian posteriors vs frequentist estimates -- compare\n")
cat("trends and patterns, not exact point values.\n")
