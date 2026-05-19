###############################################################################
# 01_svar_cholesky_ab_validation.R
#
# Cross-validation script for SVAR Cholesky and AB model identification
# against chronobox Python results.
#
# Packages used:
#   - vars (>= 1.5-6): VAR estimation, SVAR() for AB model, irf()
#   - svars (>= 1.3.11): id.chol() for Cholesky identification
#
# Implementation differences between packages:
#   - vars::SVAR() uses scoring algorithm to estimate A and B matrices
#     with user-specified restrictions (NA = free, 0 = restricted)
#   - svars::id.chol() performs Cholesky decomposition of the residual
#     covariance matrix directly (equivalent to recursive identification)
#   - Both should produce identical IRFs under Cholesky ordering, but
#     the AB model from vars::SVAR() may differ slightly due to
#     optimization vs. direct decomposition
#
# Tolerance: < 1e-3 for structural IRF comparisons
#
# Data: examples/svar/data/us_macro_quarterly.csv
#   Variables: gdp, inflation, fed_funds, unemployment (4-variable system)
#   Ordering: gdp -> inflation -> fed_funds -> unemployment (Cholesky)
###############################################################################

library(vars)
library(svars)
library(jsonlite)

set.seed(42)

cat("=" %+% rep("=", 69) %+% "\n")
cat("SVAR Cholesky & AB Model Validation (R)\n")
cat("=" %+% rep("=", 69) %+% "\n\n")

# Helper for string concatenation
`%+%` <- function(a, b) paste0(a, b)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data_dir <- file.path(dirname(getwd()), "data")
if (!file.exists(file.path(data_dir, "us_macro_quarterly.csv"))) {
  data_dir <- file.path(getwd(), "examples", "svar", "data")
}
if (!file.exists(file.path(data_dir, "us_macro_quarterly.csv"))) {
  data_dir <- file.path(getwd(), "data")
}

cat("Loading data from:", data_dir, "\n")
df <- read.csv(file.path(data_dir, "us_macro_quarterly.csv"))

# Prepare time series (drop date column)
y <- ts(df[, c("gdp", "inflation", "fed_funds", "unemployment")],
         start = c(1975, 1), frequency = 4)

cat("Observations:", nrow(df), "\n")
cat("Variables:", paste(colnames(df)[-1], collapse = ", "), "\n\n")

# ---------------------------------------------------------------------------
# 2. Estimate reduced-form VAR(4)
# ---------------------------------------------------------------------------
cat("--- Estimating VAR(4) ---\n")
var_model <- VAR(y, p = 4, type = "const")
cat("VAR(4) estimated successfully\n")
cat("Residual covariance matrix (Sigma_u):\n")
sigma_u <- summary(var_model)$covres
print(round(sigma_u, 6))
cat("\n")

# ---------------------------------------------------------------------------
# 3. Cholesky identification via svars::id.chol()
# ---------------------------------------------------------------------------
cat("--- Cholesky Identification (svars::id.chol) ---\n")
svar_chol <- id.chol(var_model)

# Extract B matrix (structural impact matrix = lower Cholesky factor)
B_chol <- svar_chol$B
cat("Structural impact matrix B (Cholesky):\n")
print(round(B_chol, 6))
cat("\n")

# Compute IRFs
irf_chol <- irf(svar_chol, n.ahead = 20, cumulative = FALSE, boot = FALSE)

# Extract IRF matrix: response of each variable to each shock
n_ahead <- 21  # 0 to 20
n_vars <- 4
var_names <- c("gdp", "inflation", "fed_funds", "unemployment")
shock_names <- c("GDP shock", "Inflation shock", "Monetary shock", "Unemp. shock")

irf_matrix <- matrix(NA, nrow = n_ahead * n_vars, ncol = n_vars + 2)
colnames(irf_matrix) <- c("horizon", "response_variable",
                           shock_names)

row_idx <- 1
for (h in 0:(n_ahead - 1)) {
  for (v in 1:n_vars) {
    irf_matrix[row_idx, 1] <- h
    irf_matrix[row_idx, 2] <- var_names[v]
    for (s in 1:n_vars) {
      # irf object: irf$irf[[shock_name]] is a matrix (horizon x response)
      shock_label <- var_names[s]
      irf_matrix[row_idx, s + 2] <- irf_chol$irf[[shock_label]][h + 1, v]
    }
    row_idx <- row_idx + 1
  }
}

irf_df <- as.data.frame(irf_matrix, stringsAsFactors = FALSE)
irf_df$horizon <- as.integer(irf_df$horizon)
for (s in shock_names) {
  irf_df[[s]] <- as.numeric(irf_df[[s]])
}

cat("IRF sample (first 8 rows):\n")
print(head(irf_df, 8))
cat("\n")

# ---------------------------------------------------------------------------
# 4. AB Model identification via vars::SVAR()
# ---------------------------------------------------------------------------
cat("--- AB Model Identification (vars::SVAR) ---\n")

# A matrix: lower triangular with 1s on diagonal
# Ordering: gdp, inflation, fed_funds, unemployment
Amat <- diag(n_vars)
Amat[2, 1] <- NA  # inflation <- gdp
Amat[3, 1] <- NA  # fed_funds <- gdp
Amat[3, 2] <- NA  # fed_funds <- inflation
Amat[4, 1] <- NA  # unemployment <- gdp

# B matrix: diagonal (structural shock std devs)
Bmat <- diag(NA, n_vars)
Bmat[upper.tri(Bmat)] <- 0
Bmat[lower.tri(Bmat)] <- 0

cat("A matrix restrictions (NA = free):\n")
print(Amat)
cat("\nB matrix restrictions (NA = free):\n")
print(Bmat)
cat("\n")

svar_ab <- SVAR(var_model, Amat = Amat, Bmat = Bmat,
                max.iter = 1000, conv.crit = 1e-8)

cat("Estimated A matrix:\n")
print(round(svar_ab$A, 6))
cat("\nEstimated B matrix:\n")
print(round(svar_ab$B, 6))
cat("\n")

# Compute A_inv * B = structural impact matrix for AB model
A_inv_B <- solve(svar_ab$A) %*% svar_ab$B
cat("A^{-1}B (structural impact matrix from AB model):\n")
print(round(A_inv_B, 6))
cat("\n")

# IRFs from AB model
irf_ab <- irf(svar_ab, n.ahead = 20, cumulative = FALSE, boot = FALSE)

# ---------------------------------------------------------------------------
# 5. Compare Cholesky vs AB model
# ---------------------------------------------------------------------------
cat("--- Comparison: Cholesky vs AB Model ---\n")

# Compare impact matrices
diff_B <- abs(B_chol - A_inv_B)
cat("Max absolute difference in impact matrices:", max(diff_B), "\n")

# Compare IRFs at horizon 0
cat("Impact (h=0) comparison:\n")
cat("  Cholesky diagonal:", round(diag(B_chol), 6), "\n")
cat("  AB model B diagonal:", round(diag(svar_ab$B), 6), "\n\n")

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
output_dir <- file.path(dirname(data_dir), "outputs", "R")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Save IRF CSV
write.csv(irf_df, file.path(output_dir, "svar_irf_cholesky.csv"),
          row.names = FALSE)
cat("Saved: svar_irf_cholesky.csv\n")

# Save B matrix JSON
B_results <- list(
  method = "cholesky",
  description = "R validation: Structural impact matrices from Cholesky and AB identification",
  variable_names = var_names,
  shock_names = shock_names,
  B0_inv_cholesky = as.list(as.data.frame(t(B_chol))),
  A_matrix_ab = as.list(as.data.frame(t(svar_ab$A))),
  B_matrix_ab = as.list(as.data.frame(t(svar_ab$B))),
  A_inv_B = as.list(as.data.frame(t(A_inv_B))),
  sigma_u = as.list(as.data.frame(t(sigma_u))),
  n_obs = nrow(df),
  n_lags = 4,
  cholesky_diagonal = diag(B_chol),
  ab_B_diagonal = diag(svar_ab$B),
  packages = list(
    vars = as.character(packageVersion("vars")),
    svars = as.character(packageVersion("svars"))
  )
)

write(toJSON(B_results, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "svar_B_matrix.json"))
cat("Saved: svar_B_matrix.json\n")

# ---------------------------------------------------------------------------
# 7. Load and compare with Python results
# ---------------------------------------------------------------------------
cat("\n--- Comparison with Python (chronobox) results ---\n")

py_irf_file <- file.path(dirname(data_dir), "outputs", "svar_irf_cholesky.csv")
if (file.exists(py_irf_file)) {
  py_irf <- read.csv(py_irf_file)

  # Compare first few horizons
  cat("Comparing IRFs (first 5 horizons, GDP shock -> GDP response):\n")
  for (h in 0:4) {
    r_val <- irf_df[irf_df$horizon == h & irf_df$response_variable == "gdp",
                     "GDP shock"]
    py_val <- py_irf[py_irf$horizon == h & py_irf$response_variable == "gdp",
                      "GDP.shock"]
    diff <- abs(r_val - py_val)
    status <- ifelse(diff < 1e-3, "PASS", "FAIL")
    cat(sprintf("  h=%d: R=%.6f, Py=%.6f, diff=%.2e [%s]\n",
                h, r_val, py_val, diff, status))
  }
} else {
  cat("Python IRF file not found at:", py_irf_file, "\n")
  cat("Skipping cross-validation comparison.\n")
}

py_B_file <- file.path(dirname(data_dir), "outputs", "svar_B_matrix.json")
if (file.exists(py_B_file)) {
  py_B <- fromJSON(py_B_file)

  cat("\nComparing B matrix diagonals:\n")
  py_chol_diag <- py_B$comparison$cholesky_impact_diagonal
  r_chol_diag <- diag(B_chol)
  for (i in 1:n_vars) {
    diff <- abs(r_chol_diag[i] - py_chol_diag[i])
    status <- ifelse(diff < 1e-3, "PASS", "FAIL")
    cat(sprintf("  %s: R=%.6f, Py=%.6f, diff=%.2e [%s]\n",
                var_names[i], r_chol_diag[i], py_chol_diag[i], diff, status))
  }
} else {
  cat("Python B matrix file not found. Skipping.\n")
}

cat("\n")
cat("Tolerance threshold: < 1e-3 for structural IRFs\n")
cat("Implementation note: vars::SVAR() uses ML scoring; svars::id.chol()\n")
cat("uses direct Cholesky decomposition. Small differences expected in AB\n")
cat("model due to optimization convergence vs. analytic solution.\n")
cat("\nDone.\n")
