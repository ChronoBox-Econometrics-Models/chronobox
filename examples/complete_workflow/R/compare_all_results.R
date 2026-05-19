################################################################################
# compare_all_results.R
# Comprehensive comparison of Python (chronobox) vs R results for both
# univariate and multivariate workflows.
#
# Reads outputs from:
#   - outputs/         (Python results)
#   - outputs/R/       (R results)
#
# Produces:
#   - outputs/R/comparison_univariate.csv
#   - outputs/R/comparison_multivariate.csv
#   - outputs/R/comparison_summary.csv
#   - Console output with tolerance assessment
#
# Format: metric | Python | R | diferenca_absoluta | within_tolerance
################################################################################

library(jsonlite)

# -- Paths ---------------------------------------------------------------------
get_script_dir <- function() {
  tryCatch({
    return(dirname(sys.frame(1)$ofile))
  }, error = function(e) NULL)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg[1]))))
  }
  return(NULL)
}

script_dir <- get_script_dir()
if (!is.null(script_dir)) {
  base_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
} else {
  base_dir <- normalizePath("examples/complete_workflow", mustWork = FALSE)
}

py_dir <- file.path(base_dir, "outputs")
r_dir  <- file.path(base_dir, "outputs", "R")

cat("=== Cross-Validation: Python (chronobox) vs R ===\n")
cat("Python outputs:", py_dir, "\n")
cat("R outputs:     ", r_dir, "\n\n")

# ==============================================================================
# Tolerance definitions
# ==============================================================================
# Different result types have different expected precision levels.
# These tolerances account for:
#   - Different numerical backends (numpy/scipy vs R)
#   - Different default algorithm settings
#   - MCMC sampling variability for Bayesian methods
tolerances <- list(
  unit_root_statistic = 0.5,     # ADF/KPSS implementations differ slightly
  unit_root_pvalue    = 0.1,     # p-value interpolation differs
  decomposition       = 0.01,    # STL should be very close
  hp_filter           = 0.01,    # HP filter is deterministic
  model_aic           = 10.0,    # AIC scale differs by likelihood constant
  model_bic           = 10.0,    # BIC scale differs similarly
  forecast_rmse       = 5.0,     # Forecast accuracy depends on model params
  forecast_mae        = 5.0,
  forecast_mape       = 2.0,
  forecast_value      = 20.0,    # Point forecasts can differ due to optimization
  irf_value           = 0.05,    # IRF should be close with same lag/method
  fevd_value          = 5.0,     # FEVD in percentage points
  spillover_index     = 5.0,     # Spillover percentage
  bvar_irf            = 0.15,    # BVAR has MCMC variability
  bvar_forecast       = 1.0,     # BVAR forecasts have sampling uncertainty
  cointegration_rank  = 1        # Rank may differ by 1
)

cat("--- Tolerance Definitions ---\n")
for (name in names(tolerances)) {
  cat(sprintf("  %-25s: %.4f\n", name, tolerances[[name]]))
}
cat("\n")

# ==============================================================================
# Helper: build comparison row
# ==============================================================================
make_row <- function(category, metric, py_val, r_val, tol_name) {
  abs_diff <- abs(as.numeric(py_val) - as.numeric(r_val))
  tol <- tolerances[[tol_name]]
  within <- abs_diff <= tol
  data.frame(
    category          = category,
    metric            = metric,
    python            = as.numeric(py_val),
    R                 = as.numeric(r_val),
    diferenca_absoluta = abs_diff,
    tolerance         = tol,
    tolerance_type    = tol_name,
    within_tolerance  = within,
    stringsAsFactors  = FALSE
  )
}

# ==============================================================================
# 1. Univariate comparison
# ==============================================================================
cat("=== UNIVARIATE COMPARISON ===\n\n")

uni_rows <- list()

# --- Unit root tests ---
py_tests <- fromJSON(file.path(py_dir, "univariate_tests.json"))
r_tests  <- fromJSON(file.path(r_dir, "univariate_tests.json"))

test_names <- c("adf_raw", "adf_log", "kpss_raw", "kpss_log",
                "adf_diff_log", "adf_diff_seasonal_log")

for (tn in test_names) {
  if (!is.null(py_tests$tests[[tn]]$statistic) &&
      !is.null(r_tests$tests[[tn]]$statistic)) {
    uni_rows <- c(uni_rows, list(make_row(
      "Unit Root", paste0(tn, "_statistic"),
      py_tests$tests[[tn]]$statistic,
      r_tests$tests[[tn]]$statistic,
      "unit_root_statistic"
    )))
  }
  if (!is.null(py_tests$tests[[tn]]$pvalue) &&
      !is.null(r_tests$tests[[tn]]$pvalue)) {
    uni_rows <- c(uni_rows, list(make_row(
      "Unit Root", paste0(tn, "_pvalue"),
      py_tests$tests[[tn]]$pvalue,
      r_tests$tests[[tn]]$pvalue,
      "unit_root_pvalue"
    )))
  }
}

# --- Zivot-Andrews ---
if (!is.null(py_tests$tests$zivot_andrews_raw$statistic) &&
    !is.null(r_tests$tests$zivot_andrews_raw$statistic)) {
  uni_rows <- c(uni_rows, list(make_row(
    "Unit Root", "zivot_andrews_statistic",
    py_tests$tests$zivot_andrews_raw$statistic,
    r_tests$tests$zivot_andrews_raw$statistic,
    "unit_root_statistic"
  )))
}

# --- Models comparison ---
py_models <- read.csv(file.path(py_dir, "univariate_models_comparison.csv"),
                      stringsAsFactors = FALSE)
r_models  <- read.csv(file.path(r_dir, "univariate_models_comparison.csv"),
                      stringsAsFactors = FALSE)

# Match ETS model
py_ets <- py_models[grep("ETS", py_models$Model), ]
r_ets  <- r_models[grep("ETS", r_models$Model), ]
if (nrow(py_ets) > 0 && nrow(r_ets) > 0) {
  uni_rows <- c(uni_rows, list(make_row(
    "Models", "ETS_AIC", py_ets$AIC[1], r_ets$AIC[1], "model_aic")))
  uni_rows <- c(uni_rows, list(make_row(
    "Models", "ETS_BIC", py_ets$BIC[1], r_ets$BIC[1], "model_bic")))
  uni_rows <- c(uni_rows, list(make_row(
    "Models", "ETS_RMSE", py_ets$RMSE[1], r_ets$RMSE[1], "forecast_rmse")))
  uni_rows <- c(uni_rows, list(make_row(
    "Models", "ETS_MAE", py_ets$MAE[1], r_ets$MAE[1], "forecast_mae")))
}

# Match Holt-Winters
py_hw <- py_models[grep("Holt", py_models$Model), ]
r_hw  <- r_models[grep("Holt", r_models$Model), ]
if (nrow(py_hw) > 0 && nrow(r_hw) > 0) {
  uni_rows <- c(uni_rows, list(make_row(
    "Models", "HW_RMSE", py_hw$RMSE[1], r_hw$RMSE[1], "forecast_rmse")))
  uni_rows <- c(uni_rows, list(make_row(
    "Models", "HW_MAE", py_hw$MAE[1], r_hw$MAE[1], "forecast_mae")))
}

# Match Theta
py_theta <- py_models[grep("Theta", py_models$Model), ]
r_theta  <- r_models[grep("Theta", r_models$Model), ]
if (nrow(py_theta) > 0 && nrow(r_theta) > 0) {
  uni_rows <- c(uni_rows, list(make_row(
    "Models", "Theta_RMSE", py_theta$RMSE[1], r_theta$RMSE[1], "forecast_rmse")))
  uni_rows <- c(uni_rows, list(make_row(
    "Models", "Theta_MAE", py_theta$MAE[1], r_theta$MAE[1], "forecast_mae")))
}

# --- Forecasts (ETS) ---
py_fc <- read.csv(file.path(py_dir, "univariate_forecasts.csv"),
                  stringsAsFactors = FALSE)
r_fc  <- read.csv(file.path(r_dir, "univariate_forecasts.csv"),
                  stringsAsFactors = FALSE)

# Compare first 12 forecast values
n_compare <- min(12, nrow(py_fc), nrow(r_fc))
for (i in 1:n_compare) {
  uni_rows <- c(uni_rows, list(make_row(
    "Forecasts", paste0("ETS_forecast_h", i),
    py_fc$forecast[i], r_fc$forecast[i], "forecast_value"
  )))
}

uni_comparison <- do.call(rbind, uni_rows)

cat("Univariate comparison results:\n")
cat(sprintf("  Total comparisons: %d\n", nrow(uni_comparison)))
cat(sprintf("  Within tolerance:  %d (%.1f%%)\n",
            sum(uni_comparison$within_tolerance),
            100 * mean(uni_comparison$within_tolerance)))
cat(sprintf("  Outside tolerance: %d\n",
            sum(!uni_comparison$within_tolerance)))

if (any(!uni_comparison$within_tolerance)) {
  cat("\n  Metrics outside tolerance:\n")
  outside <- uni_comparison[!uni_comparison$within_tolerance, ]
  for (i in 1:nrow(outside)) {
    cat(sprintf("    %s/%s: Python=%.4f, R=%.4f, diff=%.4f (tol=%.4f)\n",
                outside$category[i], outside$metric[i],
                outside$python[i], outside$R[i],
                outside$diferenca_absoluta[i], outside$tolerance[i]))
  }
}

write.csv(uni_comparison,
          file.path(r_dir, "comparison_univariate.csv"),
          row.names = FALSE)
cat("\nSaved: comparison_univariate.csv\n")

# ==============================================================================
# 2. Multivariate comparison
# ==============================================================================
cat("\n=== MULTIVARIATE COMPARISON ===\n\n")

multi_rows <- list()

# --- Unit root tests ---
py_mtests <- fromJSON(file.path(py_dir, "multivariate_tests.json"))
r_mtests  <- fromJSON(file.path(r_dir, "multivariate_tests.json"))

for (v in py_mtests$variables) {
  # ADF statistic
  if (!is.null(py_mtests$test_details[[v]]$adf_level$statistic) &&
      !is.null(r_mtests$test_details[[v]]$adf_level$statistic)) {
    multi_rows <- c(multi_rows, list(make_row(
      "Unit Root", paste0(v, "_adf_statistic"),
      py_mtests$test_details[[v]]$adf_level$statistic,
      r_mtests$test_details[[v]]$adf_level$statistic,
      "unit_root_statistic"
    )))
    multi_rows <- c(multi_rows, list(make_row(
      "Unit Root", paste0(v, "_adf_pvalue"),
      py_mtests$test_details[[v]]$adf_level$pvalue,
      r_mtests$test_details[[v]]$adf_level$pvalue,
      "unit_root_pvalue"
    )))
  }
  # KPSS statistic
  if (!is.null(py_mtests$test_details[[v]]$kpss_level$statistic) &&
      !is.null(r_mtests$test_details[[v]]$kpss_level$statistic)) {
    multi_rows <- c(multi_rows, list(make_row(
      "Unit Root", paste0(v, "_kpss_statistic"),
      py_mtests$test_details[[v]]$kpss_level$statistic,
      r_mtests$test_details[[v]]$kpss_level$statistic,
      "unit_root_statistic"
    )))
    multi_rows <- c(multi_rows, list(make_row(
      "Unit Root", paste0(v, "_kpss_pvalue"),
      py_mtests$test_details[[v]]$kpss_level$pvalue,
      r_mtests$test_details[[v]]$kpss_level$pvalue,
      "unit_root_pvalue"
    )))
  }
}

# --- Model selection ---
py_msel <- fromJSON(file.path(py_dir, "multivariate_model_selection.json"))
r_msel  <- fromJSON(file.path(r_dir, "multivariate_model_selection.json"))

# Compare IC values at lag 1
if (!is.null(py_msel$ic_values[["1"]]$aic) &&
    !is.null(r_msel$ic_values[["1"]]$aic)) {
  multi_rows <- c(multi_rows, list(make_row(
    "Model Selection", "VAR1_AIC",
    py_msel$ic_values[["1"]]$aic, r_msel$ic_values[["1"]]$aic, "model_aic")))
  multi_rows <- c(multi_rows, list(make_row(
    "Model Selection", "VAR1_BIC",
    py_msel$ic_values[["1"]]$bic, r_msel$ic_values[["1"]]$bic, "model_bic")))
}

# Cointegration rank
multi_rows <- c(multi_rows, list(make_row(
  "Cointegration", "rank_trace",
  py_msel$cointegration_rank_trace, r_msel$cointegration_rank_trace,
  "cointegration_rank")))

# --- Spillover ---
py_spill <- fromJSON(file.path(py_dir, "multivariate_spillover.json"))
r_spill  <- fromJSON(file.path(r_dir, "multivariate_spillover.json"))

multi_rows <- c(multi_rows, list(make_row(
  "Spillover", "total_spillover_index",
  py_spill$total_spillover, r_spill$total_spillover, "spillover_index")))

for (i in seq_along(py_spill$variables)) {
  v <- py_spill$variables[i]
  multi_rows <- c(multi_rows, list(make_row(
    "Spillover", paste0(v, "_from_others"),
    py_spill$directional_from[i], r_spill$directional_from[i], "spillover_index")))
  multi_rows <- c(multi_rows, list(make_row(
    "Spillover", paste0(v, "_to_others"),
    py_spill$directional_to[i], r_spill$directional_to[i], "spillover_index")))
}

# --- IRF (VAR, first few periods) ---
py_irf <- read.csv(file.path(py_dir, "multivariate_irf.csv"),
                   stringsAsFactors = FALSE)
r_irf  <- read.csv(file.path(r_dir, "multivariate_irf.csv"),
                   stringsAsFactors = FALSE)

# Compare VAR IRF at period 0 (should match exactly with Cholesky)
py_irf_0 <- py_irf[py_irf$period == 0, ]
r_irf_0  <- r_irf[r_irf$period == 0, ]

for (i in 1:nrow(py_irf_0)) {
  resp  <- py_irf_0$response[i]
  shock <- py_irf_0$shock[i]
  r_match <- r_irf_0[r_irf_0$response == resp & r_irf_0$shock == shock, ]
  if (nrow(r_match) > 0) {
    multi_rows <- c(multi_rows, list(make_row(
      "IRF (VAR)", paste0("h0_", shock, "->", resp),
      py_irf_0$var_irf[i], r_match$var_irf[1], "irf_value")))
  }
}

# Compare BVAR IRF at period 0
for (i in 1:nrow(py_irf_0)) {
  resp  <- py_irf_0$response[i]
  shock <- py_irf_0$shock[i]
  r_match <- r_irf_0[r_irf_0$response == resp & r_irf_0$shock == shock, ]
  if (nrow(r_match) > 0 &&
      !is.na(py_irf_0$bvar_irf[i]) && !is.na(r_match$bvar_irf[1])) {
    multi_rows <- c(multi_rows, list(make_row(
      "IRF (BVAR)", paste0("h0_", shock, "->", resp),
      py_irf_0$bvar_irf[i], r_match$bvar_irf[1], "bvar_irf")))
  }
}

# --- Forecasts ---
py_mfc <- read.csv(file.path(py_dir, "multivariate_forecasts.csv"),
                   stringsAsFactors = FALSE)
r_mfc  <- read.csv(file.path(r_dir, "multivariate_forecasts.csv"),
                   stringsAsFactors = FALSE)

# Compare VAR forecasts for first period per variable
for (v in c("gdp", "inflation", "fed_funds", "unemployment")) {
  py_v <- py_mfc[py_mfc$variable == v, ]
  r_v  <- r_mfc[r_mfc$variable == v, ]
  if (nrow(py_v) > 0 && nrow(r_v) > 0) {
    multi_rows <- c(multi_rows, list(make_row(
      "Forecasts (VAR)", paste0(v, "_h1"),
      py_v$var_forecast[1], r_v$var_forecast[1], "forecast_value")))
  }
}

multi_comparison <- do.call(rbind, multi_rows)

cat("Multivariate comparison results:\n")
cat(sprintf("  Total comparisons: %d\n", nrow(multi_comparison)))
cat(sprintf("  Within tolerance:  %d (%.1f%%)\n",
            sum(multi_comparison$within_tolerance),
            100 * mean(multi_comparison$within_tolerance)))
cat(sprintf("  Outside tolerance: %d\n",
            sum(!multi_comparison$within_tolerance)))

if (any(!multi_comparison$within_tolerance)) {
  cat("\n  Metrics outside tolerance:\n")
  outside <- multi_comparison[!multi_comparison$within_tolerance, ]
  for (i in 1:nrow(outside)) {
    cat(sprintf("    %s/%s: Python=%.4f, R=%.4f, diff=%.4f (tol=%.4f)\n",
                outside$category[i], outside$metric[i],
                outside$python[i], outside$R[i],
                outside$diferenca_absoluta[i], outside$tolerance[i]))
  }
}

write.csv(multi_comparison,
          file.path(r_dir, "comparison_multivariate.csv"),
          row.names = FALSE)
cat("\nSaved: comparison_multivariate.csv\n")

# ==============================================================================
# 3. Global summary
# ==============================================================================
cat("\n=== GLOBAL SUMMARY ===\n\n")

all_comparison <- rbind(
  cbind(workflow = "univariate", uni_comparison),
  cbind(workflow = "multivariate", multi_comparison)
)

# Summary by category
summary_by_cat <- aggregate(
  within_tolerance ~ workflow + category,
  data = all_comparison,
  FUN = function(x) {
    paste0(sum(x), "/", length(x), " (",
           round(100 * mean(x), 1), "%)")
  }
)
names(summary_by_cat)[3] <- "pass_rate"
cat("Pass rate by category:\n")
print(summary_by_cat, row.names = FALSE)

# Summary by tolerance type
summary_by_tol <- aggregate(
  within_tolerance ~ tolerance_type,
  data = all_comparison,
  FUN = function(x) {
    paste0(sum(x), "/", length(x), " (",
           round(100 * mean(x), 1), "%)")
  }
)
names(summary_by_tol)[2] <- "pass_rate"
cat("\nPass rate by tolerance type:\n")
print(summary_by_tol, row.names = FALSE)

# Overall
total_pass <- sum(all_comparison$within_tolerance)
total_n    <- nrow(all_comparison)
cat(sprintf("\nOVERALL: %d/%d comparisons within tolerance (%.1f%%)\n",
            total_pass, total_n, 100 * total_pass / total_n))

# Tolerance documentation table
cat("\n--- Tolerance Documentation ---\n")
cat(sprintf("%-30s %-10s %s\n", "Result Type", "Tolerance", "Rationale"))
cat(paste(rep("-", 80), collapse = ""), "\n")
cat(sprintf("%-30s %-10s %s\n",
            "Unit root statistic", "0.5",
            "Different ADF implementations (lag selection, regression)"))
cat(sprintf("%-30s %-10s %s\n",
            "Unit root p-value", "0.1",
            "P-value interpolation differs between tseries and statsmodels"))
cat(sprintf("%-30s %-10s %s\n",
            "Decomposition (STL)", "0.01",
            "Deterministic algorithm, should be nearly identical"))
cat(sprintf("%-30s %-10s %s\n",
            "HP filter", "0.01",
            "Deterministic algorithm, same lambda"))
cat(sprintf("%-30s %-10s %s\n",
            "Model AIC/BIC", "10.0",
            "Likelihood constant may differ; relative ranking matters more"))
cat(sprintf("%-30s %-10s %s\n",
            "Forecast RMSE/MAE", "5.0",
            "Different optimization algorithms for model parameters"))
cat(sprintf("%-30s %-10s %s\n",
            "Forecast MAPE", "2.0",
            "Percentage metric, should be reasonably close"))
cat(sprintf("%-30s %-10s %s\n",
            "Forecast values", "20.0",
            "Point forecasts accumulate differences from model estimation"))
cat(sprintf("%-30s %-10s %s\n",
            "IRF (VAR)", "0.05",
            "Same Cholesky method, small numerical differences"))
cat(sprintf("%-30s %-10s %s\n",
            "FEVD / Spillover", "5.0",
            "Percentage points; differences from VAR estimation"))
cat(sprintf("%-30s %-10s %s\n",
            "BVAR IRF", "0.15",
            "MCMC sampling variability with different implementations"))
cat(sprintf("%-30s %-10s %s\n",
            "BVAR forecast", "1.0",
            "MCMC draws produce inherent sampling variation"))
cat(sprintf("%-30s %-10s %s\n",
            "Cointegration rank", "1",
            "Rank may differ by 1 due to borderline test statistics"))

# Save global summary
write.csv(all_comparison,
          file.path(r_dir, "comparison_summary.csv"),
          row.names = FALSE)
cat("\nSaved: comparison_summary.csv\n")

cat("\n=== Comparison Complete ===\n")
