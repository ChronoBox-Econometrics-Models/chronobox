###############################################################################
# 03_bvar_validation.R
#
# Cross-validation script for Bayesian VAR with Minnesota prior
# against chronobox Python results.
#
# Packages used:
#   - BVAR (>= 1.0.4): bvar() for Bayesian VAR with Minnesota prior
#   - vars (>= 1.5-6): VAR() for reduced-form comparison
#
# Implementation differences:
#   - BVAR::bvar() implements Minnesota prior with hyperparameters:
#     lambda1 (own-lag tightness), lambda2 (cross-variable tightness),
#     lambda3 (lag decay). Uses Gibbs sampling for posterior draws.
#   - chronobox uses its own Minnesota prior implementation. Small differences
#     expected due to: (1) different Gibbs samplers, (2) different
#     parameterization of the Minnesota prior, (3) random seed effects.
#   - bvartools provides a more manual interface but follows similar logic.
#
# Tolerance: Posterior means should agree within ~0.05 for coefficients;
#   forecast intervals compared qualitatively (coverage, width).
#
# Data: examples/svar/data/us_macro_quarterly.csv
###############################################################################

library(BVAR)
library(vars)
library(jsonlite)

set.seed(42)

cat("=======================================================================\n")
cat("BVAR Minnesota Prior Validation (R)\n")
cat("=======================================================================\n\n")

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

y <- as.matrix(df[, c("gdp", "inflation", "fed_funds", "unemployment")])
var_names <- c("gdp", "inflation", "fed_funds", "unemployment")
cat("Observations:", nrow(df), "\n")
cat("Variables:", paste(var_names, collapse = ", "), "\n\n")

# ---------------------------------------------------------------------------
# 2. BVAR with Minnesota prior
# ---------------------------------------------------------------------------
cat("--- Estimating BVAR with Minnesota Prior ---\n")
cat("Hyperparameters: lambda1=0.1, lambda2=0.5, lambda3=1.0\n")
cat("Lags: 4, Draws: 5000 (burn-in: 1000)\n\n")

# Minnesota prior setup
# BVAR uses: lambda1 (tightness), lambda2 (cross-variable decay),
# lambda3 (lag decay exponent)
mn_prior <- bv_minnesota(
  lambda1 = bv_lambda(mode = 0.1, sd = 0.01, min = 0.05, max = 0.5),
  lambda2 = bv_lambda(mode = 0.5, sd = 0.01, min = 0.1, max = 2.0),
  lambda3 = bv_lambda(mode = 1.0, sd = 0.01, min = 0.5, max = 3.0),
  b = "auto"  # Use OLS estimates for prior mean
)

# Run BVAR
bvar_model <- bvar(y, lags = 4, n_draw = 6000, n_burn = 1000,
                   priors = mn_prior, verbose = FALSE)

cat("BVAR estimation complete\n")
cat("Effective draws:", 5000, "\n\n")

# ---------------------------------------------------------------------------
# 3. Extract posterior summaries
# ---------------------------------------------------------------------------
cat("--- Posterior Summaries ---\n")

# Coefficient posterior means
coef_summary <- coef(bvar_model)
cat("Coefficient posterior means (first lag):\n")
print(round(coef_summary, 4))
cat("\n")

# Log marginal likelihood (if available)
cat("Note: BVAR package may not directly report marginal likelihood.\n")
cat("Use coefficient comparisons and forecast intervals instead.\n\n")

# ---------------------------------------------------------------------------
# 4. Forecasts
# ---------------------------------------------------------------------------
cat("--- Forecasting (12 steps ahead) ---\n")
n_forecast <- 12

bvar_pred <- predict(bvar_model, horizon = n_forecast, conf_bands = c(0.16, 0.84))

# Build forecast dataframe
forecast_rows <- list()
for (h in 1:n_forecast) {
  for (v in 1:4) {
    # Extract quantiles from prediction object
    pred_quants <- bvar_pred$quants[, h, v]
    pred_median <- pred_quants["50%"]
    pred_mean <- mean(bvar_pred$fcast[, h, v])

    # Confidence bands
    lower_68 <- pred_quants["16%"]
    upper_68 <- pred_quants["84%"]
    lower_95 <- quantile(bvar_pred$fcast[, h, v], 0.025)
    upper_95 <- quantile(bvar_pred$fcast[, h, v], 0.975)

    forecast_rows[[length(forecast_rows) + 1]] <- data.frame(
      horizon = h,
      variable = var_names[v],
      mean = pred_mean,
      median = pred_median,
      lower_68 = lower_68,
      upper_68 = upper_68,
      lower_95 = lower_95,
      upper_95 = upper_95,
      stringsAsFactors = FALSE
    )
  }
}
forecast_df <- do.call(rbind, forecast_rows)
rownames(forecast_df) <- NULL

cat("Forecast sample (first 8 rows):\n")
print(head(forecast_df, 8))
cat("\n")

# ---------------------------------------------------------------------------
# 5. Posterior diagnostics
# ---------------------------------------------------------------------------
cat("--- Posterior Diagnostics ---\n")

# Check effective sample sizes (basic)
cat("Posterior draw dimensions:\n")
cat("  Coefficient draws shape:", dim(bvar_model$beta)[1], "x",
    dim(bvar_model$beta)[2], "x", dim(bvar_model$beta)[3], "\n")
cat("  Sigma draws shape:", dim(bvar_model$sigma)[1], "x",
    dim(bvar_model$sigma)[2], "x", dim(bvar_model$sigma)[3], "\n\n")

# ---------------------------------------------------------------------------
# 6. IRFs from BVAR
# ---------------------------------------------------------------------------
cat("--- Structural IRFs from BVAR ---\n")
bvar_irf <- irf(bvar_model, horizon = 20, identification = TRUE,
                conf_bands = c(0.16, 0.84))

cat("BVAR IRFs computed (Cholesky identification)\n\n")

# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------
output_dir <- file.path(dirname(data_dir), "outputs", "R")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Save forecasts
write.csv(forecast_df, file.path(output_dir, "bvar_forecasts.csv"),
          row.names = FALSE)
cat("Saved: bvar_forecasts.csv\n")

# Save posterior summary
posterior_results <- list(
  description = "R validation: BVAR posterior estimates with Minnesota prior",
  prior = "minnesota",
  hyperparameters = list(
    lambda_1 = 0.1,
    lambda_2 = 0.5,
    lambda_3 = 1.0
  ),
  variable_names = var_names,
  n_obs = nrow(df),
  n_lags = 4,
  n_draws = 5000,
  packages = list(
    BVAR = as.character(packageVersion("BVAR")),
    vars = as.character(packageVersion("vars"))
  )
)

write(toJSON(posterior_results, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "bvar_posteriors.json"))
cat("Saved: bvar_posteriors.json\n")

# ---------------------------------------------------------------------------
# 8. Compare with Python results
# ---------------------------------------------------------------------------
cat("\n--- Comparison with Python (chronobox) results ---\n")

py_forecast_file <- file.path(dirname(data_dir), "outputs", "bvar_forecasts.csv")
if (file.exists(py_forecast_file)) {
  py_fcast <- read.csv(py_forecast_file)

  cat("Comparing BVAR forecasts (h=1, all variables):\n")
  for (v in var_names) {
    r_row <- forecast_df[forecast_df$horizon == 1 & forecast_df$variable == v, ]
    py_row <- py_fcast[py_fcast$horizon == 1 & py_fcast$variable == v, ]

    if (nrow(r_row) > 0 && nrow(py_row) > 0) {
      diff_mean <- abs(r_row$mean - py_row$mean)
      diff_median <- abs(r_row$median - py_row$median)
      cat(sprintf("  %s: R_mean=%.4f, Py_mean=%.4f, diff=%.4f | ",
                  v, r_row$mean, py_row$mean, diff_mean))
      cat(sprintf("R_med=%.4f, Py_med=%.4f, diff=%.4f\n",
                  r_row$median, py_row$median, diff_median))
    }
  }

  cat("\nComparing forecast interval widths (h=1, 68% bands):\n")
  for (v in var_names) {
    r_row <- forecast_df[forecast_df$horizon == 1 & forecast_df$variable == v, ]
    py_row <- py_fcast[py_fcast$horizon == 1 & py_fcast$variable == v, ]

    if (nrow(r_row) > 0 && nrow(py_row) > 0) {
      r_width <- r_row$upper_68 - r_row$lower_68
      py_width <- py_row$upper_68 - py_row$lower_68
      cat(sprintf("  %s: R_width=%.4f, Py_width=%.4f\n",
                  v, r_width, py_width))
    }
  }
} else {
  cat("Python BVAR forecast file not found. Skipping.\n")
}

py_post_file <- file.path(dirname(data_dir), "outputs", "bvar_posteriors.json")
if (file.exists(py_post_file)) {
  py_post <- fromJSON(py_post_file)

  cat("\nPrior hyperparameter comparison:\n")
  cat(sprintf("  lambda_1: R=%.3f, Py=%.3f\n",
              0.1, py_post$hyperparameters$lambda_1))
  cat(sprintf("  lambda_2: R=%.3f, Py=%.3f\n",
              0.5, py_post$hyperparameters$lambda_2))
  cat(sprintf("  lambda_3: R=%.3f, Py=%.3f\n",
              1.0, py_post$hyperparameters$lambda_3))
} else {
  cat("Python BVAR posteriors file not found. Skipping.\n")
}

cat("\n")
cat("Tolerance: Forecast means within ~0.05; interval widths qualitative\n")
cat("Implementation note: BVAR::bvar() uses Gibbs sampling with Minnesota\n")
cat("prior. Differences from chronobox expected due to:\n")
cat("  1. Different random seeds / Gibbs sampler implementations\n")
cat("  2. Possibly different prior parameterization conventions\n")
cat("  3. Burn-in and thinning differences\n")
cat("Forecasts should be qualitatively similar (same direction, similar\n")
cat("magnitude and interval widths).\n")
cat("\nDone.\n")
