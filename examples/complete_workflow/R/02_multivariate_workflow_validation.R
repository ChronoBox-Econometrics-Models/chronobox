################################################################################
# 02_multivariate_workflow_validation.R
# Complete multivariate time series workflow in R for cross-validation with
# chronobox Python library.
#
# Pipeline:
#   1. Load US macro quarterly dataset (gdp, inflation, fed_funds, unemployment)
#   2. Unit root tests (ADF, KPSS) via urca/tseries
#   3. Johansen cointegration test via urca::ca.jo()
#   4. VAR model selection and estimation via vars
#   5. SVAR estimation, IRF, FEVD via vars
#   6. Bayesian VAR via BVAR
#   7. Spillover analysis (Diebold-Yilmaz) via FEVD
#   8. Save all results to outputs/R/
#
# Seed: set.seed(42) for reproducibility
################################################################################

set.seed(42)

library(urca)
library(tseries)
library(vars)
library(BVAR)
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

data_dir    <- file.path(base_dir, "data")
output_dir  <- file.path(base_dir, "outputs", "R")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== Multivariate Workflow Validation (R) ===\n")
cat("Data dir  :", data_dir, "\n")
cat("Output dir:", output_dir, "\n\n")

# ==============================================================================
# 1. Load data
# ==============================================================================
macro <- read.csv(file.path(data_dir, "us_macro_quarterly.csv"),
                  stringsAsFactors = FALSE)
macro$date <- as.Date(macro$date)
n_obs <- nrow(macro)
var_names <- c("gdp", "inflation", "fed_funds", "unemployment")
cat("Loaded US macro dataset:", n_obs, "observations,",
    length(var_names), "variables\n")
cat("Variables:", paste(var_names, collapse = ", "), "\n")

# Create multivariate ts object (quarterly starting 1975 Q1)
Y <- ts(macro[, var_names], start = c(1975, 1), frequency = 4)

# ==============================================================================
# 2. Unit root tests
# ==============================================================================
cat("\n--- Unit Root Tests ---\n")

test_details <- list()
integration_orders <- list()

for (v in var_names) {
  x <- as.numeric(Y[, v])

  # ADF test
  adf_res <- adf.test(x, alternative = "stationary")
  adf_stat <- unname(adf_res$statistic)
  adf_pval <- adf_res$p.value

  # KPSS test
  kpss_res <- kpss.test(x, null = "Level")
  kpss_stat <- unname(kpss_res$statistic)
  kpss_pval <- kpss_res$p.value

  # Determine integration order:
  # I(0) if ADF rejects AND KPSS does not reject
  # I(1) if ADF rejects AND KPSS also rejects (conflicting)
  # I(1) if ADF does not reject
  adf_reject  <- adf_pval < 0.05
  kpss_reject <- kpss_pval < 0.05

  if (adf_reject && !kpss_reject) {
    io <- "I(0)"
  } else {
    io <- "I(1)"
  }

  cat(sprintf("  %s: ADF stat=%.4f (p=%.4f), KPSS stat=%.4f (p=%.4f) => %s\n",
              v, adf_stat, adf_pval, kpss_stat, kpss_pval, io))

  test_details[[v]] <- list(
    adf_level = list(
      statistic = adf_stat,
      pvalue = adf_pval,
      reject = adf_reject
    ),
    kpss_level = list(
      statistic = kpss_stat,
      pvalue = kpss_pval,
      reject = kpss_reject
    ),
    integration_order = io
  )
  integration_orders[[v]] <- io
}

# Save test results
multivariate_tests <- list(
  dataset = "us_macro_quarterly.csv",
  n_obs = n_obs,
  n_vars = length(var_names),
  variables = var_names,
  integration_orders = integration_orders,
  test_details = test_details
)

write_json(multivariate_tests,
           file.path(output_dir, "multivariate_tests.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat("Saved: multivariate_tests.json\n")

# ==============================================================================
# 3. Cointegration test (Johansen)
# ==============================================================================
cat("\n--- Johansen Cointegration Test ---\n")

# Use variables that are I(1) for cointegration test
i1_vars <- names(which(integration_orders == "I(1)"))
cat("I(1) variables for cointegration:", paste(i1_vars, collapse = ", "), "\n")

if (length(i1_vars) >= 2) {
  Y_coint <- Y[, i1_vars]
  jo_test <- ca.jo(Y_coint, type = "trace", ecdet = "const", K = 2)
  cat("Johansen trace test results:\n")
  print(summary(jo_test))

  # Extract cointegration rank from trace test
  coint_rank <- sum(jo_test@teststat > jo_test@cval[, "5pct"])
  cat("Cointegration rank (trace, 5%):", coint_rank, "\n")
} else {
  coint_rank <- 0
  cat("Not enough I(1) variables for cointegration test\n")
}

# ==============================================================================
# 4. VAR model selection and estimation
# ==============================================================================
cat("\n--- VAR Model Selection ---\n")

# Lag selection
lag_select <- VARselect(Y, lag.max = 8, type = "const")
cat("Information criteria:\n")
print(lag_select$criteria)

optimal_lag_aic <- lag_select$selection["AIC(n)"]
cat("\nOptimal lag (AIC):", optimal_lag_aic, "\n")

# Use lag 1 for consistency with Python results
selected_lag <- 1
cat("Selected lag:", selected_lag, "\n")

# Estimate VAR
var_model <- VAR(Y, p = selected_lag, type = "const")
cat("VAR(", selected_lag, ") estimated\n")

# Save model selection results
ic_values <- list()
for (p in 1:8) {
  ic_values[[as.character(p)]] <- list(
    aic  = lag_select$criteria["AIC(n)", p],
    bic  = lag_select$criteria["SC(n)", p],
    hqic = lag_select$criteria["HQ(n)", p]
  )
}

model_selection <- list(
  integration_orders = integration_orders,
  cointegration_rank_trace = coint_rank,
  optimal_lag_aic = unname(optimal_lag_aic),
  selected_lag = selected_lag,
  ic_values = ic_values
)

write_json(model_selection,
           file.path(output_dir, "multivariate_model_selection.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat("Saved: multivariate_model_selection.json\n")

# ==============================================================================
# 5. Structural VAR, IRF, FEVD
# ==============================================================================
cat("\n--- Structural Analysis ---\n")

# Orthogonal IRF (Cholesky decomposition, same as Python)
irf_result <- vars::irf(var_model, n.ahead = 10, ortho = TRUE, boot = FALSE)

# Build IRF dataframe
irf_rows <- list()
for (resp in var_names) {
  for (shock in var_names) {
    irf_vals <- irf_result$irf[[shock]][, resp]
    for (h in seq_along(irf_vals)) {
      irf_rows <- c(irf_rows, list(data.frame(
        period   = h - 1,
        response = resp,
        shock    = shock,
        var_irf  = irf_vals[h],
        stringsAsFactors = FALSE
      )))
    }
  }
}
irf_df <- do.call(rbind, irf_rows)

# FEVD
fevd_result <- vars::fevd(var_model, n.ahead = 10)

# Build FEVD summary (at horizon 10)
fevd_table <- matrix(0, nrow = length(var_names), ncol = length(var_names))
rownames(fevd_table) <- var_names
colnames(fevd_table) <- var_names
for (i in seq_along(var_names)) {
  fevd_table[i, ] <- as.numeric(fevd_result[[var_names[i]]][10, ])
}
cat("FEVD table (h=10):\n")
print(round(fevd_table, 4))

# Total spillover index (Diebold-Yilmaz)
n_var <- length(var_names)
total_spillover <- (sum(fevd_table) - sum(diag(fevd_table))) / n_var * 100
cat("Total spillover index:", round(total_spillover, 4), "%\n")

# Directional spillover
directional_from <- numeric(n_var)  # FROM others
directional_to   <- numeric(n_var)  # TO others
for (i in 1:n_var) {
  directional_from[i] <- (sum(fevd_table[i, ]) - fevd_table[i, i]) * 100
  directional_to[i]   <- (sum(fevd_table[, i]) - fevd_table[i, i]) * 100
}
net_spillover <- directional_to - directional_from

spillover_results <- list(
  total_spillover = total_spillover,
  fevd_table = as.list(as.data.frame(t(fevd_table))),
  directional_from = directional_from,
  directional_to   = directional_to,
  net_spillover    = net_spillover,
  variables = var_names,
  var_lags  = selected_lag,
  horizon   = 10
)

write_json(spillover_results,
           file.path(output_dir, "multivariate_spillover.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat("Saved: multivariate_spillover.json\n")

# ==============================================================================
# 6. Bayesian VAR
# ==============================================================================
cat("\n--- Bayesian VAR ---\n")

# Minnesota prior BVAR
bvar_model <- bvar(Y, lags = selected_lag,
                   priors = bv_priors(hyper = "auto"),
                   n_draw = 10000L, n_burn = 5000L, verbose = FALSE)

cat("BVAR estimated with", 10000, "draws,", 5000, "burn-in\n")

# BVAR IRF
bvar_irf <- BVAR::irf(bvar_model, horizon = 11, identification = TRUE)

# Extract median BVAR IRF
# bvar_irf$quants has dimensions [quantile, response, horizon, shock]
# Build BVAR IRF data and merge with VAR IRF
bvar_irf_rows <- list()
for (resp_idx in seq_along(var_names)) {
  for (shock_idx in seq_along(var_names)) {
    for (h in 1:11) {  # horizon 0-10
      bvar_irf_rows <- c(bvar_irf_rows, list(data.frame(
        period   = h - 1,
        response = var_names[resp_idx],
        shock    = var_names[shock_idx],
        bvar_irf = bvar_irf$quants["50%", resp_idx, h, shock_idx],
        stringsAsFactors = FALSE
      )))
    }
  }
}
bvar_irf_df <- do.call(rbind, bvar_irf_rows)

# Merge VAR and BVAR IRFs
irf_combined <- merge(irf_df, bvar_irf_df,
                      by = c("period", "response", "shock"),
                      all = TRUE)
# Sort
irf_combined <- irf_combined[order(irf_combined$period,
                                    irf_combined$response,
                                    irf_combined$shock), ]

write.csv(irf_combined,
          file.path(output_dir, "multivariate_irf.csv"),
          row.names = FALSE)
cat("Saved: multivariate_irf.csv\n")

# ==============================================================================
# 7. Forecasts (VAR + BVAR)
# ==============================================================================
cat("\n--- Forecasts ---\n")

h_forecast <- 8  # 8 quarters = 2 years

# VAR forecast
var_fc <- predict(var_model, n.ahead = h_forecast, ci = 0.95)

# BVAR forecast
bvar_fc <- predict(bvar_model, horizon = h_forecast, conf_bands = c(0.68, 0.95))

# Build forecast dates
last_date <- max(macro$date)
fc_dates <- seq(last_date, by = "quarter", length.out = h_forecast + 1)[-1]

fc_rows <- list()
for (v_idx in seq_along(var_names)) {
  v <- var_names[v_idx]
  var_pred <- var_fc$fcst[[v]]

  # BVAR forecast quantiles: [quantile, horizon, variable]
  # Quantile names: "5%", "32%", "50%", "68%", "95%"
  for (h in 1:h_forecast) {
    fc_rows <- c(fc_rows, list(data.frame(
      date           = fc_dates[h],
      variable       = v,
      var_forecast   = var_pred[h, "fcst"],
      bvar_mean      = bvar_fc$quants["50%", h, v_idx],
      bvar_median    = bvar_fc$quants["50%", h, v_idx],
      bvar_lower_68  = bvar_fc$quants["32%", h, v_idx],
      bvar_upper_68  = bvar_fc$quants["68%", h, v_idx],
      bvar_lower_95  = bvar_fc$quants["5%", h, v_idx],
      bvar_upper_95  = bvar_fc$quants["95%", h, v_idx],
      stringsAsFactors = FALSE
    )))
  }
}
fc_df <- do.call(rbind, fc_rows)

write.csv(fc_df, file.path(output_dir, "multivariate_forecasts.csv"),
          row.names = FALSE)
cat("Saved: multivariate_forecasts.csv\n")

# ==============================================================================
# Summary
# ==============================================================================
cat("\n=== Multivariate Workflow Complete ===\n")
cat("Output files:\n")
cat("  - multivariate_tests.json\n")
cat("  - multivariate_model_selection.json\n")
cat("  - multivariate_spillover.json\n")
cat("  - multivariate_irf.csv\n")
cat("  - multivariate_forecasts.csv\n")
