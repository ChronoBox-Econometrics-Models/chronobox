##############################################################################
# 01_favar_validation.R
# FAVAR (Factor-Augmented VAR) validation using PCA + VAR
#
# Purpose: Cross-validate chronobox FAVAR results using base R / stats
#          PCA for factor extraction and vars::VAR for the augmented model.
#
# Required packages:
#   - vars   (>= 1.5-6)  : VAR estimation and IRF
#   - stats  (base R)     : prcomp() for PCA
#
# Data: examples/advanced_models/data/us_macro_quarterly.csv
#       (200 quarterly obs, 4 variables: gdp, inflation, fed_funds, unemployment)
#
# Outputs saved to: examples/advanced_models/outputs/R/
#   - favar_factors_R.csv       : extracted PCA factors
#   - favar_irf_R.csv           : impulse response functions
#   - favar_var_summary_R.txt   : VAR model summary
##############################################################################

# --- Setup -------------------------------------------------------------------
set.seed(42)

if (!requireNamespace("vars", quietly = TRUE)) {
  stop("Package 'vars' is required. Install with: install.packages('vars')")
}

library(vars)

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

cat(sprintf("  Observations: %d\n", nrow(df)))
cat(sprintf("  Variables   : %s\n", paste(names(df)[-1], collapse = ", ")))

# ---------------------------------------------------------------------------
# Step 1: PCA for factor extraction
#
# In a true FAVAR we would have a large panel of "informational" series from
# which latent factors are extracted. Here we use the 4 macro variables
# themselves to illustrate the procedure (consistent with what the chronobox
# example does with its synthetic factor model data).
# ---------------------------------------------------------------------------
cat("\n--- Step 1: PCA factor extraction ---\n")

# Standardise the data (zero mean, unit variance)
X <- as.matrix(df[, c("gdp", "inflation", "fed_funds", "unemployment")])
X_scaled <- scale(X)

n_factors <- 3  # match chronobox default

pca_fit <- prcomp(X_scaled, center = FALSE, scale. = FALSE)  # already scaled

# Extract factors (scores)
factors <- pca_fit$x[, 1:n_factors]
colnames(factors) <- paste0("factor_", 1:n_factors)

cat(sprintf("  Variance explained by %d factors: %.1f%%\n",
            n_factors,
            100 * sum(pca_fit$sdev[1:n_factors]^2) / sum(pca_fit$sdev^2)))

# Save factors
factors_df <- data.frame(date = df$date, factors)
write.csv(factors_df, file.path(output_dir, "favar_factors_R.csv"),
          row.names = FALSE)
cat("  Saved favar_factors_R.csv\n")

# ---------------------------------------------------------------------------
# Step 2: VAR with factors (FAVAR estimation)
#
# Include extracted factors alongside the policy variable (fed_funds) in a
# VAR. This is the two-step FAVAR approach of Bernanke, Boivin & Eliasz (2005).
# ---------------------------------------------------------------------------
cat("\n--- Step 2: VAR estimation with factors ---\n")

# Build the FAVAR dataset: factors + policy variable
favar_data <- ts(cbind(factors, fed_funds = df$fed_funds),
                 start = c(1975, 1), frequency = 4)

# Select lag order (BIC)
lag_select <- VARselect(favar_data, lag.max = 8, type = "const")
optimal_lag <- lag_select$selection["SC(n)"]
cat(sprintf("  Optimal lag (BIC): %d\n", optimal_lag))

# Estimate VAR
var_model <- VAR(favar_data, p = optimal_lag, type = "const")

# Save summary
summary_file <- file.path(output_dir, "favar_var_summary_R.txt")
sink(summary_file)
cat("FAVAR Validation - VAR Summary (R)\n")
cat("===================================\n\n")
cat(sprintf("Lag order: %d (BIC)\n", optimal_lag))
cat(sprintf("Variables: %s\n", paste(colnames(favar_data), collapse = ", ")))
cat(sprintf("Observations used: %d\n\n", nrow(favar_data) - optimal_lag))
print(summary(var_model))
sink()
cat(sprintf("  Saved favar_var_summary_R.txt\n"))

# ---------------------------------------------------------------------------
# Step 3: Impulse Response Functions
#
# Compute orthogonalised IRFs from a monetary policy shock (fed_funds).
# ---------------------------------------------------------------------------
cat("\n--- Step 3: Impulse Response Functions ---\n")

n_ahead <- 20
irf_result <- irf(var_model, impulse = "fed_funds", response = NULL,
                  n.ahead = n_ahead, ortho = TRUE, boot = TRUE,
                  runs = 500, seed = 42)

# Collect IRF into a data.frame
irf_rows <- list()
for (resp_name in names(irf_result$irf)) {
  irf_vals  <- irf_result$irf[[resp_name]]
  resp_cols <- colnames(irf_vals)
  for (j in seq_along(resp_cols)) {
    for (h in seq_len(nrow(irf_vals))) {
      irf_rows[[length(irf_rows) + 1]] <- data.frame(
        horizon  = h - 1,
        impulse  = "fed_funds",
        response = resp_cols[j],
        irf      = irf_vals[h, j],
        lower    = irf_result$Lower[[resp_name]][h, j],
        upper    = irf_result$Upper[[resp_name]][h, j],
        stringsAsFactors = FALSE
      )
    }
  }
}
irf_df <- do.call(rbind, irf_rows)

write.csv(irf_df, file.path(output_dir, "favar_irf_R.csv"),
          row.names = FALSE)
cat(sprintf("  Saved favar_irf_R.csv (%d rows)\n", nrow(irf_df)))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat("\n=== FAVAR Validation Complete ===\n")
cat("Output files:\n")
cat(sprintf("  %s\n", file.path(output_dir, "favar_factors_R.csv")))
cat(sprintf("  %s\n", file.path(output_dir, "favar_irf_R.csv")))
cat(sprintf("  %s\n", file.path(output_dir, "favar_var_summary_R.txt")))
cat("\nNote: Factor signs and scales may differ between R and Python.\n")
cat("Compare factor correlations (absolute values) rather than raw values.\n")
cat("IRF shapes and magnitudes should be qualitatively similar.\n")
