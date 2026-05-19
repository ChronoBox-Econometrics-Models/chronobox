###############################################################################
# 02_blanchard_quah_sign_validation.R
#
# Cross-validation script for Blanchard-Quah long-run restrictions and
# sign restriction identification against chronobox Python results.
#
# Packages used:
#   - vars (>= 1.5-6): VAR estimation, BQ() for Blanchard-Quah decomposition
#   - svars (>= 1.3.11): id.ngml() for non-Gaussian ML identification,
#     id.dc() for distance covariance identification
#
# Implementation differences:
#   - vars::BQ() implements the Blanchard-Quah (1989) decomposition using
#     the long-run impact matrix. Requires exactly identified system (K shocks
#     for K variables) with (K*(K-1))/2 long-run zero restrictions.
#   - svars sign restrictions: svars does not have built-in sign restrictions.
#     We implement a simple accept/reject algorithm using random rotation
#     matrices (Rubio-Ramirez et al., 2010).
#   - chronobox uses its own implementation which may differ in normalization
#     conventions and random draw algorithms.
#
# Tolerance: < 1e-3 for BQ IRFs; sign restriction results compared
#   qualitatively (median IRFs, acceptance rates)
#
# Data:
#   - blanchard_quah.csv: output_growth, unemployment (2-variable BQ system)
#   - sign_restriction.csv: output, inflation, rate (3-variable system)
###############################################################################

library(vars)
library(svars)
library(jsonlite)

set.seed(42)

cat("=======================================================================\n")
cat("Blanchard-Quah & Sign Restriction Validation (R)\n")
cat("=======================================================================\n\n")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data_dir <- file.path(dirname(getwd()), "data")
if (!file.exists(file.path(data_dir, "blanchard_quah.csv"))) {
  data_dir <- file.path(getwd(), "examples", "svar", "data")
}
if (!file.exists(file.path(data_dir, "blanchard_quah.csv"))) {
  data_dir <- file.path(getwd(), "data")
}

cat("Loading data from:", data_dir, "\n\n")

# --- Blanchard-Quah data ---
bq_df <- read.csv(file.path(data_dir, "blanchard_quah.csv"))
bq_y <- ts(bq_df[, c("output_growth", "unemployment")],
            start = c(1950, 1), frequency = 4)
cat("BQ data: ", nrow(bq_df), "obs,", ncol(bq_df) - 1, "variables\n")

# --- Sign restriction data ---
sign_df <- read.csv(file.path(data_dir, "sign_restriction.csv"))
sign_y <- ts(sign_df[, c("output", "inflation", "rate")],
             start = c(1975, 1), frequency = 4)
cat("Sign restriction data:", nrow(sign_df), "obs,", ncol(sign_df) - 1, "variables\n\n")

# ===========================================================================
# PART A: Blanchard-Quah Long-Run Restrictions
# ===========================================================================
cat("=== PART A: Blanchard-Quah Decomposition ===\n\n")

# ---------------------------------------------------------------------------
# 2. Estimate VAR and apply BQ decomposition
# ---------------------------------------------------------------------------
cat("--- Estimating VAR(4) for BQ system ---\n")
bq_var <- VAR(bq_y, p = 4, type = "const")
cat("VAR(4) estimated\n\n")

cat("--- Applying Blanchard-Quah decomposition ---\n")
bq_model <- BQ(bq_var)
cat("BQ decomposition complete\n\n")

# The BQ decomposition imposes that demand shocks have zero long-run
# effect on output (output_growth). Only supply shocks affect output
# permanently.

cat("Structural impact matrix (SIGMA.U from BQ):\n")
print(round(bq_model$LROVER, 6))
cat("\n")

# ---------------------------------------------------------------------------
# 3. Compute IRFs from BQ model
# ---------------------------------------------------------------------------
n_ahead <- 21  # horizons 0 to 20
bq_irf <- irf(bq_model, n.ahead = 20, cumulative = FALSE, boot = FALSE)

var_names_bq <- c("GDP growth", "Unemployment")
shock_names_bq <- c("Supply shock", "Demand shock")

# Build IRF dataframe
irf_rows <- list()
for (h in 0:20) {
  for (v in 1:2) {
    row <- data.frame(
      horizon = h,
      response_variable = var_names_bq[v],
      stringsAsFactors = FALSE
    )
    for (s in 1:2) {
      shock_var <- c("output_growth", "unemployment")[s]
      row[[shock_names_bq[s]]] <- bq_irf$irf[[shock_var]][h + 1, v]
    }
    irf_rows[[length(irf_rows) + 1]] <- row
  }
}
bq_irf_df <- do.call(rbind, irf_rows)

cat("BQ IRF sample (first 10 rows):\n")
print(head(bq_irf_df, 10))
cat("\n")

# Verify long-run restriction: cumulative supply shock effect on GDP growth
# should be non-zero; cumulative demand shock on GDP growth should be ~0
cum_irf <- irf(bq_model, n.ahead = 40, cumulative = TRUE, boot = FALSE)
cum_supply_gdp <- tail(cum_irf$irf$output_growth[, 1], 1)
cum_demand_gdp <- tail(cum_irf$irf$output_growth[, 2], 1)  # should NOT be 0 for cumulative of growth
cat("Long-run cumulative IRF check (h=40):\n")
cat("  Cumulative supply -> GDP growth:", round(cum_supply_gdp, 6), "\n")
cat("  Cumulative demand -> GDP growth:", round(cum_demand_gdp, 6), "\n")
cat("  (Demand shock should have zero PERMANENT effect on output level,\n")
cat("   which means zero long-run cumulative effect on output growth)\n\n")

# ===========================================================================
# PART B: Sign Restrictions
# ===========================================================================
cat("=== PART B: Sign Restrictions (Accept/Reject) ===\n\n")

# ---------------------------------------------------------------------------
# 4. Estimate VAR for sign restriction system
# ---------------------------------------------------------------------------
cat("--- Estimating VAR(4) for sign restriction system ---\n")
sign_var <- VAR(sign_y, p = 4, type = "const")
cat("VAR(4) estimated\n\n")

# ---------------------------------------------------------------------------
# 5. Sign restriction identification via accept/reject
# ---------------------------------------------------------------------------
# We implement the Rubio-Ramirez et al. (2010) algorithm:
# 1. Compute Cholesky factor P of Sigma_u
# 2. Draw random orthogonal matrix Q
# 3. Candidate impact matrix = P * Q
# 4. Check sign restrictions; if satisfied, keep the draw

cat("--- Sign Restriction Identification ---\n")
cat("Restrictions for monetary shock (shock 3):\n")
cat("  output < 0, inflation < 0, rate > 0 at h=0..3\n\n")

sigma_sign <- summary(sign_var)$covres
P <- t(chol(sigma_sign))  # Lower Cholesky factor

# Get VAR coefficient matrices for IRF computation
K <- 3
p <- 4
coef_mats <- Acoef(sign_var)  # List of K x K coefficient matrices

# Function to compute IRFs given an impact matrix
compute_irf <- function(impact_mat, coef_list, n_ahead, K) {
  Phi <- array(0, dim = c(K, K, n_ahead))
  Phi[, , 1] <- diag(K)

  # Companion form IRFs
  for (h in 2:n_ahead) {
    for (j in 1:min(h - 1, length(coef_list))) {
      Phi[, , h] <- Phi[, , h] + coef_list[[j]] %*% Phi[, , h - j]
    }
  }

  # Structural IRFs = Phi_h * impact_mat
  irf_out <- array(0, dim = c(K, K, n_ahead))
  for (h in 1:n_ahead) {
    irf_out[, , h] <- Phi[, , h] %*% impact_mat
  }
  return(irf_out)
}

# Random orthogonal matrix via QR decomposition of random normal matrix
random_orthogonal <- function(K) {
  Z <- matrix(rnorm(K * K), K, K)
  qr_dec <- qr(Z)
  Q <- qr.Q(qr_dec)
  # Ensure positive diagonal (Haar measure)
  R <- qr.R(qr_dec)
  Q <- Q %*% diag(sign(diag(R)))
  return(Q)
}

# Accept/reject loop
n_draws <- 10000
n_accepted <- 0
max_accept <- 500
accepted_impacts <- list()
accepted_irfs <- list()

# Sign restrictions: monetary shock (column 3) at horizons 0-3
# output (row 1) < 0, inflation (row 2) < 0, rate (row 3) > 0
check_horizon <- 4  # Check h=0,1,2,3

for (d in 1:n_draws) {
  Q <- random_orthogonal(K)
  candidate <- P %*% Q

  # Compute IRFs for this candidate
  candidate_irf <- compute_irf(candidate, coef_mats, check_horizon, K)

  # Check sign restrictions for shock 3 (monetary)
  valid <- TRUE
  for (h in 1:check_horizon) {
    if (candidate_irf[1, 3, h] >= 0) { valid <- FALSE; break }  # output < 0
    if (candidate_irf[2, 3, h] >= 0) { valid <- FALSE; break }  # inflation < 0
    if (candidate_irf[3, 3, h] <= 0) { valid <- FALSE; break }  # rate > 0
  }

  if (valid) {
    n_accepted <- n_accepted + 1
    accepted_impacts[[n_accepted]] <- candidate
    accepted_irfs[[n_accepted]] <- compute_irf(candidate, coef_mats, 21, K)
    if (n_accepted >= max_accept) break
  }
}

cat(sprintf("Accepted %d / %d draws (%.1f%%)\n\n",
            n_accepted, min(d, n_draws), 100 * n_accepted / min(d, n_draws)))

# Compute median IRFs across accepted draws
if (n_accepted > 0) {
  sign_var_names <- c("output", "inflation", "rate")

  # Extract monetary shock IRFs (shock 3) across all accepted draws
  monetary_irf_array <- array(0, dim = c(n_accepted, 3, 21))
  for (i in 1:n_accepted) {
    for (h in 1:21) {
      monetary_irf_array[i, , h] <- accepted_irfs[[i]][, 3, h]
    }
  }

  median_monetary_irf <- apply(monetary_irf_array, c(2, 3), median)
  q16_monetary_irf <- apply(monetary_irf_array, c(2, 3), quantile, probs = 0.16)
  q84_monetary_irf <- apply(monetary_irf_array, c(2, 3), quantile, probs = 0.84)

  cat("Median monetary shock IRF (h=0..4):\n")
  for (h in 0:4) {
    cat(sprintf("  h=%d: output=%.4f, inflation=%.4f, rate=%.4f\n",
                h, median_monetary_irf[1, h + 1],
                median_monetary_irf[2, h + 1],
                median_monetary_irf[3, h + 1]))
  }
  cat("\n")
}

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
output_dir <- file.path(dirname(data_dir), "outputs", "R")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Save BQ IRFs
write.csv(bq_irf_df, file.path(output_dir, "bq_irf.csv"), row.names = FALSE)
cat("Saved: bq_irf.csv\n")

# Save sign restriction results
sign_results <- list(
  description = "R validation: Sign restriction identification results",
  variable_names = sign_var_names,
  n_draws = min(d, n_draws),
  n_accepted = n_accepted,
  acceptance_rate = n_accepted / min(d, n_draws),
  restrictions = list(
    description = "Monetary shock: output<0, inflation<0, rate>0 for h=0..3",
    horizons_checked = 4
  ),
  median_impact_monetary = if (n_accepted > 0) median_monetary_irf[, 1] else NULL,
  packages = list(
    vars = as.character(packageVersion("vars")),
    svars = as.character(packageVersion("svars"))
  )
)
write(toJSON(sign_results, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "sign_restrictions_results.json"))
cat("Saved: sign_restrictions_results.json\n")

# ---------------------------------------------------------------------------
# 7. Compare with Python results
# ---------------------------------------------------------------------------
cat("\n--- Comparison with Python (chronobox) results ---\n")

py_bq_file <- file.path(dirname(data_dir), "outputs", "bq_irf.csv")
if (file.exists(py_bq_file)) {
  py_bq <- read.csv(py_bq_file)

  cat("Comparing BQ IRFs (Supply shock -> GDP growth, first 5 horizons):\n")
  for (h in 0:4) {
    r_val <- bq_irf_df[bq_irf_df$horizon == h &
                          bq_irf_df$response_variable == "GDP growth",
                        "Supply shock"]
    py_val <- py_bq[py_bq$horizon == h &
                      py_bq$response_variable == "GDP growth",
                    "Supply.shock"]
    diff <- abs(r_val - py_val)
    status <- ifelse(diff < 1e-3, "PASS", "FAIL")
    cat(sprintf("  h=%d: R=%.6f, Py=%.6f, diff=%.2e [%s]\n",
                h, r_val, py_val, diff, status))
  }
} else {
  cat("Python BQ IRF file not found. Skipping.\n")
}

py_sign_file <- file.path(dirname(data_dir), "outputs",
                           "sign_restrictions_results.json")
if (file.exists(py_sign_file)) {
  py_sign <- fromJSON(py_sign_file)

  cat("\nSign restriction comparison (qualitative):\n")
  cat(sprintf("  Python accepted draws: %d\n", py_sign$n_accepted_draws))
  cat(sprintf("  R accepted draws: %d\n", n_accepted))
  cat("  Note: Exact match not expected due to different random draws.\n")
  cat("  Comparison is qualitative: signs and magnitude ranges should agree.\n")

  if (n_accepted > 0 && !is.null(py_sign$strong_restrictions$median_impact)) {
    py_impact <- py_sign$strong_restrictions$median_impact
    r_impact <- median_monetary_irf[, 1]
    cat("\n  Median monetary impact comparison:\n")
    for (i in 1:min(length(r_impact), length(py_impact))) {
      cat(sprintf("    %s: R=%.4f, Py=%.4f (same sign: %s)\n",
                  sign_var_names[i], r_impact[i], py_impact[i],
                  ifelse(sign(r_impact[i]) == sign(py_impact[i]), "YES", "NO")))
    }
  }
} else {
  cat("Python sign restriction file not found. Skipping.\n")
}

cat("\n")
cat("Tolerance: < 1e-3 for BQ structural IRFs\n")
cat("Sign restrictions: qualitative comparison (signs must agree)\n")
cat("Implementation note: BQ() in vars uses the C(1) long-run matrix\n")
cat("from the MA representation. Sign restrictions use accept/reject with\n")
cat("random orthogonal rotations (Rubio-Ramirez et al., 2010).\n")
cat("\nDone.\n")
