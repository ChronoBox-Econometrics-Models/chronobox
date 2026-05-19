##############################################################################
# compare_results.R
# Compare Python (chronobox) vs R results for advanced models
#
# Purpose: Load output CSVs from both Python and R runs, compute agreement
#          metrics, and produce a summary report.
#
# Required packages:
#   - stats (base R)
#
# Inputs:
#   examples/advanced_models/outputs/          (Python results)
#   examples/advanced_models/outputs/R/        (R results)
#
# Outputs:
#   examples/advanced_models/outputs/R/comparison_report.txt
#   examples/advanced_models/outputs/R/comparison_summary.csv
##############################################################################

set.seed(42)

# --- Paths -------------------------------------------------------------------
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir)) script_dir <- "."

base_dir    <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
py_out_dir  <- file.path(base_dir, "outputs")
r_out_dir   <- file.path(base_dir, "outputs", "R")

report_file <- file.path(r_out_dir, "comparison_report.txt")
summary_file <- file.path(r_out_dir, "comparison_summary.csv")

# --- Helper functions --------------------------------------------------------

#' Compute correlation between two numeric vectors (ignoring NAs)
safe_cor <- function(x, y) {
  ok <- complete.cases(x, y)
  if (sum(ok) < 3) return(NA_real_)
  cor(x[ok], y[ok])
}

#' Compute RMSE between two vectors
rmse <- function(x, y) {
  ok <- complete.cases(x, y)
  if (sum(ok) == 0) return(NA_real_)
  sqrt(mean((x[ok] - y[ok])^2))
}

#' Compute mean absolute error
mae <- function(x, y) {
  ok <- complete.cases(x, y)
  if (sum(ok) == 0) return(NA_real_)
  mean(abs(x[ok] - y[ok]))
}

# --- Start report ------------------------------------------------------------
sink(report_file)
cat("="*70, "\n")
cat("COMPARISON REPORT: Python (chronobox) vs R\n")
cat("Advanced Models Validation\n")
cat("="*70, "\n\n")
cat(sprintf("Generated: %s\n\n", Sys.time()))

summary_rows <- list()

# ============================================================================
# 1. FAVAR Comparison
# ============================================================================
cat("\n", "-"*70, "\n")
cat("1. FAVAR - Factor-Augmented VAR\n")
cat("-"*70, "\n\n")

# --- 1a: Compare factors ---
py_factors_file <- file.path(py_out_dir, "favar_factors.csv")
r_factors_file  <- file.path(r_out_dir, "favar_factors_R.csv")

if (file.exists(py_factors_file) && file.exists(r_factors_file)) {
  py_factors <- read.csv(py_factors_file, stringsAsFactors = FALSE)
  r_factors  <- read.csv(r_factors_file, stringsAsFactors = FALSE)

  cat("Factor comparison:\n")
  cat(sprintf("  Python factors: %d obs x %d factors\n",
              nrow(py_factors), ncol(py_factors) - 1))
  cat(sprintf("  R factors     : %d obs x %d factors\n",
              nrow(r_factors), ncol(r_factors) - 1))

  # PCA factors can differ in sign and ordering. Use absolute correlation
  # to find best-matching factor pairs.
  n_py <- min(ncol(py_factors) - 1, 3)
  n_r  <- min(ncol(r_factors) - 1, 3)
  n_common <- min(nrow(py_factors), nrow(r_factors))

  cat("\n  Cross-correlation matrix (|cor|) between Python and R factors:\n")
  cor_mat <- matrix(NA, n_py, n_r)
  for (i in 1:n_py) {
    for (j in 1:n_r) {
      cor_mat[i, j] <- abs(safe_cor(
        py_factors[1:n_common, i + 1],
        r_factors[1:n_common, j + 1]
      ))
    }
  }
  rownames(cor_mat) <- paste0("Py_F", 1:n_py)
  colnames(cor_mat) <- paste0("R_F", 1:n_r)
  print(round(cor_mat, 4))

  cat("\n  Note: PCA factors have sign/order indeterminacy.\n")
  cat("  High absolute correlations (>0.8) indicate good agreement.\n")

  # Best matches
  for (i in 1:min(n_py, n_r)) {
    best_j <- which.max(cor_mat[i, ])
    cat(sprintf("  Py_F%d best match -> R_F%d (|cor| = %.4f)\n",
                i, best_j, cor_mat[i, best_j]))
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      model = "FAVAR",
      metric = paste0("factor_", i, "_abs_cor"),
      value = round(cor_mat[i, best_j], 4),
      status = ifelse(cor_mat[i, best_j] > 0.5, "PASS", "CHECK"),
      stringsAsFactors = FALSE
    )
  }
} else {
  cat("  SKIPPED: Factor files not found.\n")
  if (!file.exists(py_factors_file)) cat(sprintf("    Missing: %s\n", py_factors_file))
  if (!file.exists(r_factors_file))  cat(sprintf("    Missing: %s\n", r_factors_file))
}

# --- 1b: Compare IRFs ---
py_irf_file <- file.path(py_out_dir, "favar_irf.csv")
r_irf_file  <- file.path(r_out_dir, "favar_irf_R.csv")

if (file.exists(py_irf_file) && file.exists(r_irf_file)) {
  py_irf <- read.csv(py_irf_file, stringsAsFactors = FALSE)
  r_irf  <- read.csv(r_irf_file, stringsAsFactors = FALSE)

  cat("\nIRF comparison:\n")
  cat(sprintf("  Python IRF rows: %d\n", nrow(py_irf)))
  cat(sprintf("  R IRF rows     : %d\n", nrow(r_irf)))
  cat("  Note: IRF structures differ due to different model specs.\n")
  cat("  Qualitative comparison: check that responses have similar\n")
  cat("  signs and decay patterns.\n")

  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    model = "FAVAR", metric = "irf_files_exist",
    value = 1, status = "PASS", stringsAsFactors = FALSE
  )
} else {
  cat("\n  IRF files: not all available for comparison.\n")
}

# ============================================================================
# 2. TVP-VAR Comparison
# ============================================================================
cat("\n\n", "-"*70, "\n")
cat("2. TVP-VAR - Time-Varying Parameter VAR\n")
cat("-"*70, "\n\n")

py_coef_file <- file.path(py_out_dir, "tvp_coefficients.csv")
r_coef_file  <- file.path(r_out_dir, "tvp_coefficients_R.csv")

if (file.exists(py_coef_file) && file.exists(r_coef_file)) {
  py_coef <- read.csv(py_coef_file, stringsAsFactors = FALSE)
  r_coef  <- read.csv(r_coef_file, stringsAsFactors = FALSE)

  cat("Coefficient comparison:\n")
  cat(sprintf("  Python: %d obs x %d cols\n", nrow(py_coef), ncol(py_coef)))
  cat(sprintf("  R     : %d obs x %d cols\n", nrow(r_coef), ncol(r_coef)))

  # Find common coefficient columns
  common_cols <- intersect(names(py_coef), names(r_coef))
  common_cols <- common_cols[common_cols != "date"]

  if (length(common_cols) > 0) {
    cat(sprintf("\n  Common coefficient columns: %d\n", length(common_cols)))

    # Align by date
    py_coef$date <- as.Date(py_coef$date)
    r_coef$date  <- as.Date(r_coef$date)
    common_dates <- intersect(as.character(py_coef$date),
                              as.character(r_coef$date))
    n_common <- length(common_dates)
    cat(sprintf("  Common dates: %d\n", n_common))

    if (n_common > 10) {
      py_sub <- py_coef[as.character(py_coef$date) %in% common_dates, ]
      r_sub  <- r_coef[as.character(r_coef$date) %in% common_dates, ]
      py_sub <- py_sub[order(py_sub$date), ]
      r_sub  <- r_sub[order(r_sub$date), ]

      cat("\n  Per-coefficient correlations (Kalman vs Bayesian):\n")
      for (col in common_cols) {
        rho <- safe_cor(py_sub[[col]], r_sub[[col]])
        rmse_val <- rmse(py_sub[[col]], r_sub[[col]])
        status <- ifelse(!is.na(rho) && abs(rho) > 0.3, "PASS", "CHECK")
        cat(sprintf("    %-35s  cor=%.4f  RMSE=%.4f  [%s]\n",
                    col, ifelse(is.na(rho), NA, rho), rmse_val, status))
        summary_rows[[length(summary_rows) + 1]] <- data.frame(
          model = "TVP-VAR", metric = paste0("coef_cor_", col),
          value = round(ifelse(is.na(rho), 0, rho), 4),
          status = status, stringsAsFactors = FALSE
        )
      }

      cat("\n  Note: Bayesian (bvarsv) vs frequentist (Kalman) estimates\n")
      cat("  may differ substantially in level. Positive correlation\n")
      cat("  indicates agreement in time-varying direction.\n")
    }
  } else {
    cat("  No common coefficient columns found.\n")
    cat(sprintf("  Python cols: %s\n", paste(names(py_coef), collapse = ", ")))
    cat(sprintf("  R cols     : %s\n", paste(names(r_coef), collapse = ", ")))
  }
} else {
  cat("  SKIPPED: Coefficient files not found.\n")
}

# --- TVP-VAR Volatility ---
py_vol_file <- file.path(py_out_dir, "tvp_volatility.csv")
r_vol_file  <- file.path(r_out_dir, "tvp_volatility_R.csv")

if (file.exists(py_vol_file) && file.exists(r_vol_file)) {
  py_vol <- read.csv(py_vol_file, stringsAsFactors = FALSE)
  r_vol  <- read.csv(r_vol_file, stringsAsFactors = FALSE)

  cat("\nVolatility comparison:\n")
  cat(sprintf("  Python: %d obs x %d cols\n", nrow(py_vol), ncol(py_vol)))
  cat(sprintf("  R     : %d obs x %d cols\n", nrow(r_vol), ncol(r_vol)))

  common_vol_cols <- intersect(names(py_vol), names(r_vol))
  common_vol_cols <- common_vol_cols[common_vol_cols != "date"]
  cat(sprintf("  Common volatility columns: %d\n", length(common_vol_cols)))

  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    model = "TVP-VAR", metric = "volatility_files_exist",
    value = 1, status = "PASS", stringsAsFactors = FALSE
  )
} else {
  cat("  Volatility files not all available.\n")
}

# ============================================================================
# 3. GVAR Comparison
# ============================================================================
cat("\n\n", "-"*70, "\n")
cat("3. GVAR - Global VAR\n")
cat("-"*70, "\n\n")

# --- Spillover table ---
py_spillover_file <- file.path(py_out_dir, "spillover_table.json")
r_spillover_file  <- file.path(r_out_dir, "gvar_spillover_R.csv")

if (file.exists(py_spillover_file)) {
  py_spill <- tryCatch({
    jsonlite::fromJSON(py_spillover_file)
  }, error = function(e) {
    # Fallback: parse JSON manually
    lines <- readLines(py_spillover_file)
    json_str <- paste(lines, collapse = "")
    # Simple extraction for comparison
    cat("  Note: jsonlite not available, limited JSON parsing.\n")
    NULL
  })

  if (!is.null(py_spill)) {
    cat("Python spillover table (from JSON):\n")
    cat(sprintf("  Total spillover index: %.2f%%\n", py_spill$total_spillover))
    cat("  FEVD table:\n")
    fevd_df <- as.data.frame(py_spill$fevd_table)
    print(round(fevd_df, 4))
    cat("\n")

    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      model = "GVAR", metric = "py_total_spillover",
      value = round(py_spill$total_spillover, 2),
      status = "INFO", stringsAsFactors = FALSE
    )
  }
}

if (file.exists(r_spillover_file)) {
  r_spill <- read.csv(r_spillover_file, row.names = 1, stringsAsFactors = FALSE)
  cat("R FEVD / spillover table:\n")
  print(round(r_spill, 4))
  cat("\n")

  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    model = "GVAR", metric = "r_spillover_file_exists",
    value = 1, status = "PASS", stringsAsFactors = FALSE
  )
} else {
  cat("  R spillover file not yet generated. Run 03_gvar_validation.R first.\n")
}

# --- Historical decomposition ---
py_hd_file <- file.path(py_out_dir, "hist_decomp.csv")
r_hd_file  <- file.path(r_out_dir, "gvar_hist_decomp_R.csv")

if (file.exists(py_hd_file)) {
  py_hd <- read.csv(py_hd_file, stringsAsFactors = FALSE)
  cat(sprintf("Python HD: %d rows x %d cols\n", nrow(py_hd), ncol(py_hd)))

  # Check which country shocks dominate
  shock_cols <- grep("_shock$", names(py_hd), value = TRUE)
  if (length(shock_cols) > 0) {
    cat("  Mean absolute shock contribution:\n")
    for (sc in shock_cols) {
      cat(sprintf("    %-15s %.4f\n", sc, mean(abs(py_hd[[sc]]), na.rm = TRUE)))
    }
  }
}

if (file.exists(r_hd_file)) {
  r_hd <- read.csv(r_hd_file, row.names = 1, stringsAsFactors = FALSE)
  cat(sprintf("\nR HD: %d rows x %d cols\n", nrow(r_hd), ncol(r_hd)))
  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    model = "GVAR", metric = "hist_decomp_available",
    value = 1, status = "PASS", stringsAsFactors = FALSE
  )
}

# ============================================================================
# Overall Summary
# ============================================================================
cat("\n\n", "="*70, "\n")
cat("OVERALL COMPARISON SUMMARY\n")
cat("="*70, "\n\n")

cat("Methodology differences (expected):\n")
cat("  FAVAR : Python uses PCA on synthetic factors; R uses PCA on macro vars\n")
cat("          Factor sign/order indeterminacy is normal\n")
cat("  TVP-VAR: Python uses Kalman filter/MLE; R uses Bayesian MCMC (bvarsv)\n")
cat("           Point estimates will differ; trends should agree\n")
cat("  GVAR  : Python uses OLS country-by-country; R uses Bayesian (BGVAR)\n")
cat("          Prior shrinkage affects coefficient magnitudes\n\n")

cat("Tolerance guidelines for Bayesian vs frequentist comparison:\n")
cat("  - Do NOT expect numerical equality\n")
cat("  - Correlation > 0.3 for time-varying coefficient paths = acceptable\n")
cat("  - Factor absolute correlation > 0.5 = good agreement\n")
cat("  - Spillover ranking agreement more important than magnitudes\n")
cat("  - Compare posterior means to point estimates as rough guide\n")

sink()
cat(sprintf("Report saved to: %s\n", report_file))

# --- Save summary CSV -------------------------------------------------------
if (length(summary_rows) > 0) {
  summary_df <- do.call(rbind, summary_rows)
  write.csv(summary_df, summary_file, row.names = FALSE)
  cat(sprintf("Summary saved to: %s\n", summary_file))

  cat("\nComparison Summary:\n")
  print(summary_df, row.names = FALSE)
} else {
  cat("\nNo comparison metrics computed. Run individual scripts first.\n")
}

cat("\n=== Comparison Complete ===\n")
