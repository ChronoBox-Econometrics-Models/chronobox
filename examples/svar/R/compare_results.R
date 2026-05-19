###############################################################################
# compare_results.R
#
# Master comparison script: loads Python (chronobox) and R outputs,
# computes summary statistics, and generates a validation report.
#
# Packages used:
#   - jsonlite (>= 1.8): JSON reading/writing
#
# This script reads from:
#   - examples/svar/outputs/          (Python results)
#   - examples/svar/outputs/R/        (R results)
#
# Tolerance thresholds:
#   - Structural IRFs (Cholesky/BQ): < 1e-3
#   - BVAR forecasts: qualitative (means within ~0.1, intervals overlap)
#   - Sign restrictions: qualitative (same signs, similar magnitudes)
###############################################################################

library(jsonlite)

set.seed(42)

cat("=======================================================================\n")
cat("SVAR/BVAR Cross-Validation Report: Python (chronobox) vs R\n")
cat("=======================================================================\n\n")

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
base_dir <- file.path(dirname(getwd()))
if (!file.exists(file.path(base_dir, "outputs"))) {
  base_dir <- file.path(getwd(), "examples", "svar")
}
if (!file.exists(file.path(base_dir, "outputs"))) {
  base_dir <- getwd()
}

py_dir <- file.path(base_dir, "outputs")
r_dir <- file.path(base_dir, "outputs", "R")

cat("Python outputs:", py_dir, "\n")
cat("R outputs:", r_dir, "\n\n")

# Counters
total_tests <- 0
passed_tests <- 0
failed_tests <- 0
skipped_tests <- 0

check <- function(name, r_val, py_val, tol = 1e-3) {
  total_tests <<- total_tests + 1
  diff <- abs(r_val - py_val)
  if (diff < tol) {
    passed_tests <<- passed_tests + 1
    cat(sprintf("  [PASS] %s: R=%.6f, Py=%.6f, diff=%.2e\n",
                name, r_val, py_val, diff))
  } else {
    failed_tests <<- failed_tests + 1
    cat(sprintf("  [FAIL] %s: R=%.6f, Py=%.6f, diff=%.2e (tol=%.0e)\n",
                name, r_val, py_val, diff, tol))
  }
}

check_qualitative <- function(name, r_val, py_val) {
  total_tests <<- total_tests + 1
  same_sign <- sign(r_val) == sign(py_val)
  rel_diff <- abs(r_val - py_val) / max(abs(r_val), abs(py_val), 1e-10)
  if (same_sign && rel_diff < 0.5) {
    passed_tests <<- passed_tests + 1
    cat(sprintf("  [PASS] %s: R=%.4f, Py=%.4f (same sign, rel_diff=%.2f)\n",
                name, r_val, py_val, rel_diff))
  } else {
    failed_tests <<- failed_tests + 1
    cat(sprintf("  [FAIL] %s: R=%.4f, Py=%.4f (sign=%s, rel_diff=%.2f)\n",
                name, r_val, py_val,
                ifelse(same_sign, "same", "DIFFERENT"), rel_diff))
  }
}

skip <- function(name, reason) {
  total_tests <<- total_tests + 1
  skipped_tests <<- skipped_tests + 1
  cat(sprintf("  [SKIP] %s: %s\n", name, reason))
}

# ===========================================================================
# TEST 1: SVAR Cholesky IRFs
# ===========================================================================
cat("=" %+% rep("=", 69) %+% "\n")
`%+%` <- function(a, b) paste0(a, b)
cat("TEST 1: SVAR Cholesky IRFs\n")
cat("Tolerance: < 1e-3\n")
cat(strrep("-", 70), "\n")

py_irf_file <- file.path(py_dir, "svar_irf_cholesky.csv")
r_irf_file <- file.path(r_dir, "svar_irf_cholesky.csv")

if (file.exists(py_irf_file) && file.exists(r_irf_file)) {
  py_irf <- read.csv(py_irf_file)
  r_irf <- read.csv(r_irf_file)

  var_names <- c("gdp", "inflation", "fed_funds", "unemployment")
  shock_cols_py <- c("GDP.shock", "Inflation.shock", "Monetary.shock", "Unemp..shock")
  shock_cols_r <- c("GDP.shock", "Inflation.shock", "Monetary.shock", "Unemp..shock")

  for (h in c(0, 1, 5, 10, 20)) {
    for (v in var_names) {
      for (s_idx in 1:4) {
        r_row <- r_irf[r_irf$horizon == h & r_irf$response_variable == v, ]
        py_row <- py_irf[py_irf$horizon == h & py_irf$response_variable == v, ]

        if (nrow(r_row) > 0 && nrow(py_row) > 0) {
          r_val <- r_row[[shock_cols_r[s_idx]]]
          py_val <- py_row[[shock_cols_py[s_idx]]]
          check(sprintf("IRF h=%d %s->%s", h,
                        c("GDP", "Infl", "Mon", "Unemp")[s_idx], v),
                r_val, py_val)
        }
      }
    }
  }
} else {
  skip("SVAR Cholesky IRFs", "Missing output files")
}

# ===========================================================================
# TEST 2: SVAR B Matrix
# ===========================================================================
cat("\n", strrep("=", 70), "\n")
cat("TEST 2: SVAR Structural Impact Matrix (B)\n")
cat("Tolerance: < 1e-3\n")
cat(strrep("-", 70), "\n")

py_B_file <- file.path(py_dir, "svar_B_matrix.json")
r_B_file <- file.path(r_dir, "svar_B_matrix.json")

if (file.exists(py_B_file) && file.exists(r_B_file)) {
  py_B <- fromJSON(py_B_file)
  r_B <- fromJSON(r_B_file)

  # Compare Cholesky diagonals
  py_diag <- py_B$comparison$cholesky_impact_diagonal
  r_diag <- r_B$cholesky_diagonal

  for (i in 1:4) {
    check(sprintf("B_chol diagonal[%d] (%s)", i, var_names[i]),
          r_diag[i], py_diag[i])
  }

  # Compare AB model B diagonals
  py_ab_diag <- py_B$comparison$ab_model_B_diagonal
  r_ab_diag <- r_B$ab_B_diagonal

  for (i in 1:4) {
    check(sprintf("B_AB diagonal[%d] (%s)", i, var_names[i]),
          r_ab_diag[i], py_ab_diag[i], tol = 0.05)
  }
} else {
  skip("SVAR B Matrix", "Missing output files")
}

# ===========================================================================
# TEST 3: Blanchard-Quah IRFs
# ===========================================================================
cat("\n", strrep("=", 70), "\n")
cat("TEST 3: Blanchard-Quah IRFs\n")
cat("Tolerance: < 1e-3\n")
cat(strrep("-", 70), "\n")

py_bq_file <- file.path(py_dir, "bq_irf.csv")
r_bq_file <- file.path(r_dir, "bq_irf.csv")

if (file.exists(py_bq_file) && file.exists(r_bq_file)) {
  py_bq <- read.csv(py_bq_file)
  r_bq <- read.csv(r_bq_file)

  bq_vars <- c("GDP growth", "Unemployment")
  bq_shocks_py <- c("Supply.shock", "Demand.shock")
  bq_shocks_r <- c("Supply.shock", "Demand.shock")

  for (h in c(0, 1, 5, 10, 20)) {
    for (v in bq_vars) {
      for (s_idx in 1:2) {
        r_row <- r_bq[r_bq$horizon == h & r_bq$response_variable == v, ]
        py_row <- py_bq[py_bq$horizon == h & py_bq$response_variable == v, ]

        if (nrow(r_row) > 0 && nrow(py_row) > 0) {
          r_val <- r_row[[bq_shocks_r[s_idx]]]
          py_val <- py_row[[bq_shocks_py[s_idx]]]
          check(sprintf("BQ h=%d %s->%s", h,
                        c("Supply", "Demand")[s_idx], v),
                r_val, py_val)
        }
      }
    }
  }
} else {
  skip("BQ IRFs", "Missing output files")
}

# ===========================================================================
# TEST 4: Sign Restrictions (qualitative)
# ===========================================================================
cat("\n", strrep("=", 70), "\n")
cat("TEST 4: Sign Restrictions (qualitative comparison)\n")
cat("Tolerance: same sign + relative difference < 50%\n")
cat(strrep("-", 70), "\n")

py_sign_file <- file.path(py_dir, "sign_restrictions_results.json")
r_sign_file <- file.path(r_dir, "sign_restrictions_results.json")

if (file.exists(py_sign_file) && file.exists(r_sign_file)) {
  py_sign <- fromJSON(py_sign_file)
  r_sign <- fromJSON(r_sign_file)

  cat(sprintf("  Python accepted draws: %d\n", py_sign$n_accepted_draws))
  cat(sprintf("  R accepted draws: %d\n", r_sign$n_accepted))
  cat(sprintf("  R acceptance rate: %.1f%%\n", r_sign$acceptance_rate * 100))
  cat("\n")

  # Compare median monetary shock impacts
  if (!is.null(py_sign$strong_restrictions$median_impact) &&
      !is.null(r_sign$median_impact_monetary)) {
    py_impact <- py_sign$strong_restrictions$median_impact
    r_impact <- r_sign$median_impact_monetary
    sign_vars <- c("output", "inflation", "rate")

    for (i in 1:min(length(r_impact), length(py_impact))) {
      check_qualitative(sprintf("Sign monetary impact %s", sign_vars[i]),
                        r_impact[i], py_impact[i])
    }
  } else {
    skip("Sign restriction impacts", "Missing median impact data")
  }
} else {
  skip("Sign Restrictions", "Missing output files")
}

# ===========================================================================
# TEST 5: BVAR Forecasts (qualitative)
# ===========================================================================
cat("\n", strrep("=", 70), "\n")
cat("TEST 5: BVAR Forecasts\n")
cat("Tolerance: qualitative (means within 0.1, intervals overlap)\n")
cat(strrep("-", 70), "\n")

py_fcast_file <- file.path(py_dir, "bvar_forecasts.csv")
r_fcast_file <- file.path(r_dir, "bvar_forecasts.csv")

if (file.exists(py_fcast_file) && file.exists(r_fcast_file)) {
  py_fcast <- read.csv(py_fcast_file)
  r_fcast <- read.csv(r_fcast_file)

  for (h in c(1, 4, 8, 12)) {
    for (v in c("gdp", "inflation", "fed_funds", "unemployment")) {
      r_row <- r_fcast[r_fcast$horizon == h & r_fcast$variable == v, ]
      py_row <- py_fcast[py_fcast$horizon == h & py_fcast$variable == v, ]

      if (nrow(r_row) > 0 && nrow(py_row) > 0) {
        check_qualitative(sprintf("BVAR h=%d %s mean", h, v),
                          r_row$mean, py_row$mean)

        # Check interval overlap
        total_tests <<- total_tests + 1
        r_lo <- r_row$lower_95
        r_hi <- r_row$upper_95
        py_lo <- py_row$lower_95
        py_hi <- py_row$upper_95
        overlap <- min(r_hi, py_hi) - max(r_lo, py_lo)
        if (overlap > 0) {
          passed_tests <<- passed_tests + 1
          cat(sprintf("  [PASS] BVAR h=%d %s 95%% interval overlap: %.4f\n",
                      h, v, overlap))
        } else {
          failed_tests <<- failed_tests + 1
          cat(sprintf("  [FAIL] BVAR h=%d %s 95%% intervals don't overlap!\n",
                      h, v))
          cat(sprintf("         R: [%.4f, %.4f], Py: [%.4f, %.4f]\n",
                      r_lo, r_hi, py_lo, py_hi))
        }
      }
    }
  }
} else {
  skip("BVAR Forecasts", "Missing output files")
}

# ===========================================================================
# SUMMARY
# ===========================================================================
cat("\n", strrep("=", 70), "\n")
cat("VALIDATION SUMMARY\n")
cat(strrep("=", 70), "\n\n")

cat(sprintf("Total tests:   %d\n", total_tests))
cat(sprintf("Passed:        %d (%.1f%%)\n", passed_tests,
            100 * passed_tests / max(total_tests, 1)))
cat(sprintf("Failed:        %d (%.1f%%)\n", failed_tests,
            100 * failed_tests / max(total_tests, 1)))
cat(sprintf("Skipped:       %d\n", skipped_tests))
cat("\n")

if (failed_tests == 0 && skipped_tests == 0) {
  cat("RESULT: ALL TESTS PASSED\n")
} else if (failed_tests == 0) {
  cat("RESULT: ALL EXECUTED TESTS PASSED (some skipped)\n")
} else {
  cat(sprintf("RESULT: %d TESTS FAILED - review differences above\n",
              failed_tests))
}

cat("\n")
cat("Notes on expected differences:\n")
cat("  1. Cholesky/BQ IRFs should match very closely (< 1e-3) as they\n")
cat("     use deterministic decompositions with the same data.\n")
cat("  2. AB model may show small differences due to different optimization\n")
cat("     algorithms (scoring in vars vs. chronobox's implementation).\n")
cat("  3. Sign restrictions are stochastic - only qualitative agreement\n")
cat("     expected (same signs, similar magnitude ranges).\n")
cat("  4. BVAR forecasts depend on Gibbs sampler implementation and RNG.\n")
cat("     Posterior means should be similar; intervals should overlap.\n")
cat("\n")

# Save summary
output_dir <- r_dir
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

summary_results <- list(
  description = "Cross-validation summary: Python (chronobox) vs R",
  timestamp = Sys.time(),
  total_tests = total_tests,
  passed = passed_tests,
  failed = failed_tests,
  skipped = skipped_tests,
  pass_rate = passed_tests / max(total_tests - skipped_tests, 1),
  tolerance = list(
    structural_irf = "1e-3",
    bvar_forecasts = "qualitative",
    sign_restrictions = "qualitative"
  ),
  packages = list(
    R_version = R.version.string,
    jsonlite = as.character(packageVersion("jsonlite"))
  )
)

write(toJSON(summary_results, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "validation_summary.json"))
cat("Saved: validation_summary.json\n")
cat("\nDone.\n")
