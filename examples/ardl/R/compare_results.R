################################################################################
# compare_results.R
#
# Compare ARDL/ECM results: Python (chronobox) vs R (ARDL + dynamac)
#
# Reads JSON outputs from both implementations and reports differences.
# Tolerance thresholds:
#   - Coefficients: < 1e-4
#   - F-statistic:  < 0.05
#   - Speed of adjustment: < 1e-4
#
# Required packages: jsonlite
################################################################################

cat("=== Package Versions ===\n")
cat("R version:", paste(R.version$major, R.version$minor, sep = "."), "\n")
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org")
}
cat("jsonlite:", as.character(packageVersion("jsonlite")), "\n")

library(jsonlite)

set.seed(42)

# -- Tolerance thresholds ------------------------------------------------------
TOL_COEF   <- 1e-4    # for coefficients
TOL_FSTAT  <- 0.05    # for F-statistic
TOL_SOA    <- 1e-4    # for speed of adjustment

# -- Load results --------------------------------------------------------------
script_dir <- dirname(sys.frame(1)$ofile %||% ".")
out_dir    <- file.path(script_dir, "..", "outputs")
if (!dir.exists(out_dir)) out_dir <- "examples/ardl/outputs"

# Python results
py_bounds <- fromJSON(file.path(out_dir, "bounds_test_results.json"))
py_ardl   <- fromJSON(file.path(out_dir, "ardl_coefficients.json"))
py_ecm    <- fromJSON(file.path(out_dir, "ecm_long_run.json"))

# R results
r_bounds <- fromJSON(file.path(out_dir, "R", "bounds_test_results_R.json"))
r_ecm    <- fromJSON(file.path(out_dir, "R", "ecm_results_R.json"))

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("CROSS-VALIDATION: Python (chronobox) vs R (ARDL + dynamac)\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# -- Helper: compare and report ------------------------------------------------
compare_value <- function(name, py_val, r_val, tol, unit = "") {
  diff <- abs(py_val - r_val)
  pass <- diff < tol
  status <- ifelse(pass, "PASS", "FAIL")
  cat(sprintf("  [%s] %-30s  Python: %10.6f  R: %10.6f  Diff: %.6f  (tol: %s%s)\n",
              status, name, py_val, r_val, diff, format(tol, scientific = FALSE), unit))
  return(pass)
}

all_pass <- TRUE

# ==============================================================================
# 1. ARDL Model Selection
# ==============================================================================
cat("\n--- 1. ARDL Model Selection ---\n")
cat("  Python selected: ", py_ardl$model, "\n")
r_order <- r_bounds$auto_ardl$best_order
cat("  R auto_ardl selected: ARDL(",
    paste(r_order, collapse = ", "), ")\n")
cat("  Note: Both use ARDL(1,1,1,1) for comparison below.\n")

# ==============================================================================
# 2. F-statistic (Bounds Test)
# ==============================================================================
cat("\n--- 2. Bounds F-statistic ---\n")
py_fstat <- py_bounds$f_statistic
r_fstat  <- r_bounds$bounds_f_test$f_statistic
p <- compare_value("F-statistic", py_fstat, r_fstat, TOL_FSTAT)
all_pass <- all_pass & p

cat("\n  Python conclusion (5%):", py_bounds$conclusion_5pct, "\n")

# ==============================================================================
# 3. Long-run Coefficients (ARDL)
# ==============================================================================
cat("\n--- 3. Long-run Coefficients (ARDL) ---\n")
py_lr <- py_ardl$long_run_coefficients
r_lr  <- r_bounds$ardl_1111$long_run

for (var in c("x1", "x2", "x3")) {
  p <- compare_value(paste0("LR_", var, " (ARDL)"),
                     py_lr[[var]], r_lr[[var]], TOL_COEF)
  all_pass <- all_pass & p
}

# ==============================================================================
# 4. Long-run Coefficients (ECM / UECM)
# ==============================================================================
cat("\n--- 4. Long-run Coefficients (ECM) ---\n")
py_ecm_lr <- py_ecm$long_run_coefficients
r_uecm_lr <- r_ecm$uecm$long_run

for (var in c("x1", "x2", "x3")) {
  p <- compare_value(paste0("LR_", var, " (ECM)"),
                     py_ecm_lr[[var]], r_uecm_lr[[var]], TOL_COEF)
  all_pass <- all_pass & p
}

# ==============================================================================
# 5. Speed of Adjustment
# ==============================================================================
cat("\n--- 5. Speed of Adjustment ---\n")
py_soa <- py_ecm$speed_of_adjustment$pi_yy
r_soa_uecm <- r_ecm$uecm$speed_of_adjustment
r_soa_dyn  <- r_ecm$dynardl$speed_of_adjustment

p1 <- compare_value("SOA (Python vs R-UECM)", py_soa, r_soa_uecm, TOL_SOA)
p2 <- compare_value("SOA (Python vs R-dynardl)", py_soa, r_soa_dyn, TOL_SOA)
all_pass <- all_pass & p1 & p2

cat("\n  True DGP speed of adjustment:", py_ecm$true_dgp$speed_of_adjustment, "\n")

# ==============================================================================
# 6. R-internal: ARDL vs dynamac consistency
# ==============================================================================
cat("\n--- 6. R Internal: ARDL pkg vs dynamac consistency ---\n")
for (var in c("x1", "x2", "x3")) {
  p <- compare_value(paste0("LR_", var, " (ARDL-UECM vs dynardl)"),
                     r_uecm_lr[[var]], r_ecm$dynardl$long_run[[var]], TOL_COEF)
  all_pass <- all_pass & p
}

p <- compare_value("SOA (ARDL-UECM vs dynardl)",
                   r_soa_uecm, r_soa_dyn, TOL_SOA)
all_pass <- all_pass & p

# ==============================================================================
# 7. Comparison with True DGP
# ==============================================================================
cat("\n--- 7. Proximity to True DGP ---\n")
cat("  True DGP: LR_x1 = 0.6, SOA = -0.25, intercept = 1.5\n\n")

# These are informational, not pass/fail (finite sample bias expected)
cat(sprintf("  %-35s  Estimate: %10.6f  True: %10.6f  Error: %.6f\n",
            "Python LR_x1", py_lr[["x1"]], 0.6, abs(py_lr[["x1"]] - 0.6)))
cat(sprintf("  %-35s  Estimate: %10.6f  True: %10.6f  Error: %.6f\n",
            "R (UECM) LR_x1", r_uecm_lr[["x1"]], 0.6, abs(r_uecm_lr[["x1"]] - 0.6)))
cat(sprintf("  %-35s  Estimate: %10.6f  True: %10.6f  Error: %.6f\n",
            "R (dynardl) LR_x1", r_ecm$dynardl$long_run[["x1"]], 0.6,
            abs(r_ecm$dynardl$long_run[["x1"]] - 0.6)))
cat(sprintf("  %-35s  Estimate: %10.6f  True: %10.6f  Error: %.6f\n",
            "Python SOA", py_soa, -0.25, abs(py_soa - (-0.25))))
cat(sprintf("  %-35s  Estimate: %10.6f  True: %10.6f  Error: %.6f\n",
            "R (UECM) SOA", r_soa_uecm, -0.25, abs(r_soa_uecm - (-0.25))))

# ==============================================================================
# 8. Model fit comparison
# ==============================================================================
cat("\n--- 8. Model Fit (ARDL 1,1,1,1) ---\n")
cat(sprintf("  %-20s  Python: %10.4f  R: %10.4f\n",
            "R-squared", py_ardl$r_squared, r_bounds$ardl_1111$r_squared))
cat(sprintf("  %-20s  Python: %10.4f  R: %10.4f\n",
            "AIC", py_ardl$aic, r_bounds$ardl_1111$aic))
cat(sprintf("  %-20s  Python: %10.4f  R: %10.4f\n",
            "BIC", py_ardl$bic, r_bounds$ardl_1111$bic))
cat("  Note: AIC/BIC may differ by a constant due to likelihood definition.\n")

# ==============================================================================
# SUMMARY
# ==============================================================================
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
if (all_pass) {
  cat("OVERALL RESULT: ALL TESTS PASSED\n")
  cat("chronobox ARDL/ECM results are consistent with R reference.\n")
} else {
  cat("OVERALL RESULT: SOME TESTS FAILED\n")
  cat("Review the FAIL items above. Differences may be due to:\n")
  cat("  - Different likelihood definitions (AIC/BIC constants)\n")
  cat("  - Numerical precision in optimization\n")
  cat("  - Edge effects in lag truncation\n")
}
cat(paste(rep("=", 70), collapse = ""), "\n")

# Save comparison summary
comparison <- list(
  timestamp = Sys.time(),
  tolerance = list(
    coefficients = TOL_COEF,
    f_statistic = TOL_FSTAT,
    speed_of_adjustment = TOL_SOA
  ),
  f_statistic = list(
    python = py_fstat,
    r = r_fstat,
    diff = abs(py_fstat - r_fstat),
    pass = abs(py_fstat - r_fstat) < TOL_FSTAT
  ),
  long_run_ardl = list(
    x1 = list(python = py_lr[["x1"]], r = r_lr[["x1"]],
              diff = abs(py_lr[["x1"]] - r_lr[["x1"]])),
    x2 = list(python = py_lr[["x2"]], r = r_lr[["x2"]],
              diff = abs(py_lr[["x2"]] - r_lr[["x2"]])),
    x3 = list(python = py_lr[["x3"]], r = r_lr[["x3"]],
              diff = abs(py_lr[["x3"]] - r_lr[["x3"]]))
  ),
  speed_of_adjustment = list(
    python = py_soa,
    r_uecm = r_soa_uecm,
    r_dynardl = r_soa_dyn,
    diff_uecm = abs(py_soa - r_soa_uecm),
    diff_dynardl = abs(py_soa - r_soa_dyn)
  ),
  overall_pass = all_pass
)

comp_path <- file.path(out_dir, "R", "comparison_python_vs_R.json")
writeLines(toJSON(comparison, auto_unbox = TRUE, pretty = TRUE), comp_path)
cat("\nComparison saved to:", normalizePath(comp_path, mustWork = FALSE), "\n")

cat("\n=== DONE: compare_results.R ===\n")
