################################################################################
# 01_univariate_workflow_validation.R
# Complete univariate time series workflow in R for cross-validation with
# chronobox Python library.
#
# Pipeline:
#   1. Load airline passengers dataset
#   2. Unit root tests (ADF, KPSS, Zivot-Andrews) via urca/tseries
#   3. STL decomposition via stats::stl()
#   4. HP filter via mFilter::hpfilter()
#   5. ARIMA / ETS models via forecast package
#   6. Forecast accuracy comparison via forecast::accuracy()
#   7. Save all results to outputs/R/
#
# Seed: set.seed(42) for reproducibility
################################################################################

set.seed(42)

library(urca)
library(tseries)
library(forecast)
library(mFilter)
library(jsonlite)

# -- Paths ---------------------------------------------------------------------
# Detect script directory (works with source() and Rscript)
get_script_dir <- function() {
  # Try sys.frame (source)
  tryCatch({
    return(dirname(sys.frame(1)$ofile))
  }, error = function(e) NULL)
  # Try commandArgs (Rscript)
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

data_dir    <- file.path(base_dir, "data")
output_dir  <- file.path(base_dir, "outputs", "R")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== Univariate Workflow Validation (R) ===\n")
cat("Data dir  :", data_dir, "\n")
cat("Output dir:", output_dir, "\n\n")

# ==============================================================================
# 1. Load data
# ==============================================================================
airline <- read.csv(file.path(data_dir, "airline.csv"), stringsAsFactors = FALSE)
airline$date <- as.Date(airline$date)
n_obs <- nrow(airline)
cat("Loaded airline dataset:", n_obs, "observations\n")

# Create ts object (monthly, starting Jan 1949)
y <- ts(airline$passengers, start = c(1949, 1), frequency = 12)
y_log <- log(y)

# ==============================================================================
# 2. Unit root tests
# ==============================================================================
cat("\n--- Unit Root Tests ---\n")

# ADF on raw series
adf_raw <- adf.test(as.numeric(y), alternative = "stationary")
cat("ADF (raw):     stat =", round(adf_raw$statistic, 6),
    " p-value =", round(adf_raw$p.value, 6), "\n")

# ADF on log series
adf_log <- adf.test(as.numeric(y_log), alternative = "stationary")
cat("ADF (log):     stat =", round(adf_log$statistic, 6),
    " p-value =", round(adf_log$p.value, 6), "\n")

# KPSS on raw series
kpss_raw <- kpss.test(as.numeric(y), null = "Level")
cat("KPSS (raw):    stat =", round(kpss_raw$statistic, 6),
    " p-value =", round(kpss_raw$p.value, 6), "\n")

# KPSS on log series
kpss_log <- kpss.test(as.numeric(y_log), null = "Level")
cat("KPSS (log):    stat =", round(kpss_log$statistic, 6),
    " p-value =", round(kpss_log$p.value, 6), "\n")

# Zivot-Andrews on raw series (urca)
za_raw <- ur.za(as.numeric(y), model = "both", lag = NULL)
cat("Zivot-Andrews (raw): stat =", round(za_raw@teststat, 6), "\n")

# ADF on diff(log)
y_diff_log <- diff(y_log)
adf_diff_log <- adf.test(as.numeric(y_diff_log), alternative = "stationary")
cat("ADF (diff log):       stat =", round(adf_diff_log$statistic, 6),
    " p-value =", round(adf_diff_log$p.value, 6), "\n")

# ADF on seasonal diff of diff(log)
y_diff_seasonal_log <- diff(y_diff_log, lag = 12)
adf_diff_seasonal <- adf.test(as.numeric(y_diff_seasonal_log),
                               alternative = "stationary")
cat("ADF (diff+seas log):  stat =", round(adf_diff_seasonal$statistic, 6),
    " p-value =", round(adf_diff_seasonal$p.value, 6), "\n")

# Save unit root test results
unit_root_results <- list(
  dataset = "airline.csv",
  n_obs = n_obs,
  frequency = "monthly",
  tests = list(
    adf_raw = list(
      statistic = unname(adf_raw$statistic),
      pvalue = adf_raw$p.value,
      reject_h0 = adf_raw$p.value < 0.05,
      conclusion = ifelse(adf_raw$p.value < 0.05, "Stationary", "Non-stationary")
    ),
    adf_log = list(
      statistic = unname(adf_log$statistic),
      pvalue = adf_log$p.value,
      reject_h0 = adf_log$p.value < 0.05,
      conclusion = ifelse(adf_log$p.value < 0.05, "Stationary", "Non-stationary")
    ),
    kpss_raw = list(
      statistic = unname(kpss_raw$statistic),
      pvalue = kpss_raw$p.value,
      reject_h0 = kpss_raw$p.value < 0.05,
      conclusion = ifelse(kpss_raw$p.value < 0.05, "Non-stationary", "Stationary")
    ),
    kpss_log = list(
      statistic = unname(kpss_log$statistic),
      pvalue = kpss_log$p.value,
      reject_h0 = kpss_log$p.value < 0.05,
      conclusion = ifelse(kpss_log$p.value < 0.05, "Non-stationary", "Stationary")
    ),
    zivot_andrews_raw = list(
      statistic = unname(za_raw@teststat)
    ),
    adf_diff_log = list(
      statistic = unname(adf_diff_log$statistic),
      pvalue = adf_diff_log$p.value,
      reject_h0 = adf_diff_log$p.value < 0.05,
      conclusion = ifelse(adf_diff_log$p.value < 0.05, "Stationary", "Non-stationary")
    ),
    adf_diff_seasonal_log = list(
      statistic = unname(adf_diff_seasonal$statistic),
      pvalue = adf_diff_seasonal$p.value,
      reject_h0 = adf_diff_seasonal$p.value < 0.05,
      conclusion = ifelse(adf_diff_seasonal$p.value < 0.05, "Stationary", "Non-stationary")
    )
  ),
  integration_order = list(
    regular = 1,
    seasonal = 1,
    log_transform = TRUE
  )
)

write_json(unit_root_results,
           file.path(output_dir, "univariate_tests.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat("Saved: univariate_tests.json\n")

# ==============================================================================
# 3. STL Decomposition
# ==============================================================================
cat("\n--- STL Decomposition ---\n")
stl_result <- stl(y_log, s.window = "periodic")
stl_df <- data.frame(
  date = airline$date,
  seasonal = as.numeric(stl_result$time.series[, "seasonal"]),
  trend    = as.numeric(stl_result$time.series[, "trend"]),
  remainder = as.numeric(stl_result$time.series[, "remainder"])
)
write.csv(stl_df, file.path(output_dir, "univariate_stl_decomposition.csv"),
          row.names = FALSE)
cat("Saved: univariate_stl_decomposition.csv\n")
cat("  Trend range: [", round(min(stl_df$trend), 4), ",",
    round(max(stl_df$trend), 4), "]\n")

# ==============================================================================
# 4. HP Filter
# ==============================================================================
cat("\n--- HP Filter ---\n")
hp_result <- hpfilter(as.numeric(y_log), freq = 14400)  # freq=14400 for monthly
hp_df <- data.frame(
  date = airline$date,
  original = as.numeric(y_log),
  trend = as.numeric(hp_result$trend),
  cycle = as.numeric(hp_result$cycle)
)
write.csv(hp_df, file.path(output_dir, "univariate_hp_filter.csv"),
          row.names = FALSE)
cat("Saved: univariate_hp_filter.csv\n")
cat("  HP trend range: [", round(min(hp_df$trend), 4), ",",
    round(max(hp_df$trend), 4), "]\n")

# ==============================================================================
# 5. Model Estimation
# ==============================================================================
cat("\n--- Model Estimation ---\n")

# Split: train = first 132 obs, test = last 12 obs
n_train <- n_obs - 12
n_test  <- 12
y_train <- window(y, end = c(1949 + (n_train - 1) %/% 12,
                              ((n_train - 1) %% 12) + 1))
y_test  <- window(y, start = c(1949 + n_train %/% 12,
                                (n_train %% 12) + 1))

cat("Train:", length(y_train), "obs, Test:", length(y_test), "obs\n")

# --- ETS ---
ets_model <- ets(y_train)
cat("ETS model:", ets_model$method, "\n")
cat("  AIC =", round(ets_model$aic, 4), ", BIC =", round(ets_model$bic, 4), "\n")
ets_fc <- forecast(ets_model, h = n_test)
ets_acc <- accuracy(ets_fc, y_test)
cat("  Test RMSE =", round(ets_acc["Test set", "RMSE"], 4),
    ", MAE =", round(ets_acc["Test set", "MAE"], 4),
    ", MAPE =", round(ets_acc["Test set", "MAPE"], 4), "\n")

# --- Holt-Winters (multiplicative) ---
hw_model <- HoltWinters(y_train, seasonal = "multiplicative")
hw_fc <- forecast(hw_model, h = n_test)
hw_acc <- accuracy(hw_fc, y_test)
cat("Holt-Winters (mult):\n")
cat("  Test RMSE =", round(hw_acc["Test set", "RMSE"], 4),
    ", MAE =", round(hw_acc["Test set", "MAE"], 4),
    ", MAPE =", round(hw_acc["Test set", "MAPE"], 4), "\n")

# --- Theta ---
theta_fc <- thetaf(y_train, h = n_test)
theta_acc <- accuracy(theta_fc, y_test)
cat("Theta:\n")
cat("  Test RMSE =", round(theta_acc["Test set", "RMSE"], 4),
    ", MAE =", round(theta_acc["Test set", "MAE"], 4),
    ", MAPE =", round(theta_acc["Test set", "MAPE"], 4), "\n")

# --- Auto-ARIMA ---
auto_arima_model <- auto.arima(y_train)
cat("Auto-ARIMA:", capture.output(auto_arima_model)[2], "\n")
cat("  AIC =", round(auto_arima_model$aic, 4),
    ", BIC =", round(auto_arima_model$bic, 4), "\n")
auto_fc <- forecast(auto_arima_model, h = n_test)
auto_acc <- accuracy(auto_fc, y_test)
cat("  Test RMSE =", round(auto_acc["Test set", "RMSE"], 4),
    ", MAE =", round(auto_acc["Test set", "MAE"], 4),
    ", MAPE =", round(auto_acc["Test set", "MAPE"], 4), "\n")

# --- SARIMA(0,1,1)(0,1,1)[12] ---
sarima_011 <- Arima(y_train, order = c(0, 1, 1),
                    seasonal = list(order = c(0, 1, 1), period = 12))
cat("SARIMA(0,1,1)(0,1,1)[12]:\n")
cat("  AIC =", round(sarima_011$aic, 4), ", BIC =", round(sarima_011$bic, 4), "\n")
sarima_011_fc <- forecast(sarima_011, h = n_test)
sarima_011_acc <- accuracy(sarima_011_fc, y_test)
cat("  Test RMSE =", round(sarima_011_acc["Test set", "RMSE"], 4),
    ", MAE =", round(sarima_011_acc["Test set", "MAE"], 4),
    ", MAPE =", round(sarima_011_acc["Test set", "MAPE"], 4), "\n")

# --- SARIMA(1,1,0)(1,1,0)[12] ---
sarima_110 <- Arima(y_train, order = c(1, 1, 0),
                    seasonal = list(order = c(1, 1, 0), period = 12))
cat("SARIMA(1,1,0)(1,1,0)[12]:\n")
cat("  AIC =", round(sarima_110$aic, 4), ", BIC =", round(sarima_110$bic, 4), "\n")
sarima_110_fc <- forecast(sarima_110, h = n_test)
sarima_110_acc <- accuracy(sarima_110_fc, y_test)
cat("  Test RMSE =", round(sarima_110_acc["Test set", "RMSE"], 4),
    ", MAE =", round(sarima_110_acc["Test set", "MAE"], 4),
    ", MAPE =", round(sarima_110_acc["Test set", "MAPE"], 4), "\n")

# ==============================================================================
# 6. Save models comparison table
# ==============================================================================
models_comparison <- data.frame(
  Model = c(
    ets_model$method,
    "Holt-Winters (mult)",
    "Theta",
    paste0("Auto-ARIMA (", capture.output(auto_arima_model)[2], ")"),
    "SARIMA(0,1,1)(0,1,1)[12]",
    "SARIMA(1,1,0)(1,1,0)[12]"
  ),
  AIC = c(ets_model$aic, NA, NA,
          auto_arima_model$aic, sarima_011$aic, sarima_110$aic),
  BIC = c(ets_model$bic, NA, NA,
          auto_arima_model$bic, sarima_011$bic, sarima_110$bic),
  RMSE = c(ets_acc["Test set", "RMSE"],
           hw_acc["Test set", "RMSE"],
           theta_acc["Test set", "RMSE"],
           auto_acc["Test set", "RMSE"],
           sarima_011_acc["Test set", "RMSE"],
           sarima_110_acc["Test set", "RMSE"]),
  MAE = c(ets_acc["Test set", "MAE"],
          hw_acc["Test set", "MAE"],
          theta_acc["Test set", "MAE"],
          auto_acc["Test set", "MAE"],
          sarima_011_acc["Test set", "MAE"],
          sarima_110_acc["Test set", "MAE"]),
  MAPE = c(ets_acc["Test set", "MAPE"],
           hw_acc["Test set", "MAPE"],
           theta_acc["Test set", "MAPE"],
           auto_acc["Test set", "MAPE"],
           sarima_011_acc["Test set", "MAPE"],
           sarima_110_acc["Test set", "MAPE"]),
  stringsAsFactors = FALSE
)

write.csv(models_comparison,
          file.path(output_dir, "univariate_models_comparison.csv"),
          row.names = FALSE)
cat("\nSaved: univariate_models_comparison.csv\n")

# ==============================================================================
# 7. Save ETS forecasts (24-step ahead on full series for comparison)
# ==============================================================================
cat("\n--- 24-Step-Ahead ETS Forecast (full series) ---\n")
ets_full <- ets(y)
ets_fc_full <- forecast(ets_full, h = 24, level = 95)

fc_dates <- seq(as.Date("1961-01-01"), by = "month", length.out = 24)
forecasts_df <- data.frame(
  date = fc_dates,
  forecast = as.numeric(ets_fc_full$mean),
  lower_95 = as.numeric(ets_fc_full$lower),
  upper_95 = as.numeric(ets_fc_full$upper),
  model = ets_full$method,
  stringsAsFactors = FALSE
)

write.csv(forecasts_df,
          file.path(output_dir, "univariate_forecasts.csv"),
          row.names = FALSE)
cat("Saved: univariate_forecasts.csv\n")
cat("  ETS model (full):", ets_full$method, "\n")
cat("  Forecast range: [", round(min(forecasts_df$forecast), 2), ",",
    round(max(forecasts_df$forecast), 2), "]\n")

# ==============================================================================
# Summary
# ==============================================================================
cat("\n=== Univariate Workflow Complete ===\n")
cat("Output files:\n")
cat("  - univariate_tests.json\n")
cat("  - univariate_stl_decomposition.csv\n")
cat("  - univariate_hp_filter.csv\n")
cat("  - univariate_models_comparison.csv\n")
cat("  - univariate_forecasts.csv\n")
