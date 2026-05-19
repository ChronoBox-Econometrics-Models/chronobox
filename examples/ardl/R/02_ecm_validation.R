################################################################################
# 02_ecm_validation.R
#
# ECM Validation - Reference implementation in R
# Uses ARDL::recm(), ARDL::uecm() and dynamac::dynardl() / pssbounds()
#
# Purpose: Cross-validate chronobox ECM/ARDL results with two independent
#          R packages for robustness.
# Dataset: examples/ardl/data/ardl_synthetic.csv (seed=42, 200 quarterly obs)
# Output:  examples/ardl/outputs/R/ecm_results_R.json
#
# Required packages: ARDL (>= 0.2.4), dynamac (>= 0.1.11), jsonlite
# Tolerance: coefficients < 1e-4; F-statistic < 0.05
################################################################################

# -- Package versions ----------------------------------------------------------
cat("=== Package Versions ===\n")
cat("R version:", paste(R.version$major, R.version$minor, sep = "."), "\n")
for (pkg in c("ARDL", "dynamac", "jsonlite")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  cat(pkg, ":", as.character(packageVersion(pkg)), "\n")
}

library(ARDL)
library(dynamac)
library(jsonlite)

set.seed(42)

# -- Load data -----------------------------------------------------------------
script_dir <- dirname(sys.frame(1)$ofile %||% ".")
data_path  <- file.path(script_dir, "..", "data", "ardl_synthetic.csv")
if (!file.exists(data_path)) {
  data_path <- "examples/ardl/data/ardl_synthetic.csv"
}
cat("\nLoading data from:", normalizePath(data_path), "\n")

df <- read.csv(data_path, stringsAsFactors = FALSE)
df$date <- as.Date(df$date)
cat("Observations:", nrow(df), "\n")

# Convert to data.frame for ARDL package
data_df <- data.frame(y = df$y, x1 = df$x1, x2 = df$x2, x3 = df$x3)

# ==============================================================================
# PART A: ARDL Package - recm() and uecm()
# ==============================================================================
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("PART A: ARDL Package - ECM Estimation\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# Fit base ARDL(1,1,1,1) matching chronobox
ardl_model <- ardl(y ~ x1 + x2 + x3, data = data_df, order = c(1, 1, 1, 1))

# -- A1: Unrestricted ECM (UECM) -----------------------------------------------
cat("\n=== Unrestricted ECM (UECM) ===\n")
uecm_model <- uecm(ardl_model)
cat("\n--- UECM Summary ---\n")
print(summary(uecm_model))

uecm_coefs <- coef(uecm_model)
cat("\nUECM Coefficients:\n")
print(uecm_coefs)

# Speed of adjustment = coefficient on lagged dependent variable in levels
# In UECM: dy_t = alpha + pi_yy * y_{t-1} + pi_yx * x_{t-1} + ... + short-run
ect_coef <- uecm_coefs["y:l(y, 1)"]
cat("\nSpeed of adjustment (pi_yy):", ect_coef, "\n")
cat("Half-life (quarters):", log(0.5) / log(1 + ect_coef), "\n")

# Long-run coefficients from UECM: -pi_yx / pi_yy
lr_x1_uecm <- -uecm_coefs["x1:l(x1, 1)"] / ect_coef
lr_x2_uecm <- -uecm_coefs["x2:l(x2, 1)"] / ect_coef
lr_x3_uecm <- -uecm_coefs["x3:l(x3, 1)"] / ect_coef
cat("\nLong-run from UECM:\n")
cat("  x1:", lr_x1_uecm, "\n")
cat("  x2:", lr_x2_uecm, "\n")
cat("  x3:", lr_x3_uecm, "\n")

# -- A2: Restricted ECM (RECM) ------------------------------------------------
cat("\n=== Restricted ECM (RECM) ===\n")
recm_model <- recm(ardl_model, case = 3)
cat("\n--- RECM Summary ---\n")
print(summary(recm_model))

recm_coefs <- coef(recm_model)
cat("\nRECM Coefficients:\n")
print(recm_coefs)

# ECT coefficient in RECM
ect_recm <- recm_coefs["ect"]
cat("\nECT coefficient (RECM):", ect_recm, "\n")

# -- A3: Long-run multipliers via multipliers() --------------------------------
cat("\n=== Long-run Multipliers ===\n")
lr_mult <- multipliers(ardl_model, type = "lr")
cat("\nLong-run multipliers:\n")
print(lr_mult)

# ==============================================================================
# PART B: dynamac Package - dynardl() and pssbounds()
# ==============================================================================
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("PART B: dynamac Package - ECM Estimation\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# -- B1: dynardl() model ------------------------------------------------------
cat("\n=== dynardl() Estimation ===\n")
# dynardl expects: dependent var, list of lags for levels and differences
# ARDL(1,1,1,1): 1 lag of y, 0-1 lags of x's
# levels: y_{t-1}, x1_{t-1}, x2_{t-1}, x3_{t-1}
# diffs:  dx1_t, dx2_t, dx3_t (contemporaneous differences)

dyn_model <- dynardl(y ~ x1 + x2 + x3, data = data_df,
                     lags = list("y" = 1, "x1" = 1, "x2" = 1, "x3" = 1),
                     diffs = c("x1", "x2", "x3"),
                     ec = TRUE, simulate = FALSE)

cat("\n--- dynardl Summary ---\n")
print(summary(dyn_model$model))

dyn_coefs <- coef(dyn_model$model)
cat("\ndynardl Coefficients:\n")
print(dyn_coefs)

# Extract ECT (speed of adjustment) from dynardl
# In dynardl with ec=TRUE, the coefficient on l.1.y is the ECT
ect_dyn <- dyn_coefs["l.1.y"]
cat("\nSpeed of adjustment (dynardl):", ect_dyn, "\n")

# Long-run coefficients: -coef(l.1.xi) / coef(l.1.y)
lr_x1_dyn <- -dyn_coefs["l.1.x1"] / ect_dyn
lr_x2_dyn <- -dyn_coefs["l.1.x2"] / ect_dyn
lr_x3_dyn <- -dyn_coefs["l.1.x3"] / ect_dyn
cat("\nLong-run from dynardl:\n")
cat("  x1:", lr_x1_dyn, "\n")
cat("  x2:", lr_x2_dyn, "\n")
cat("  x3:", lr_x3_dyn, "\n")

# -- B2: pssbounds() ----------------------------------------------------------
cat("\n=== PSS Bounds Test (dynamac) ===\n")

# pssbounds: Pesaran, Shin and Smith bounds test
# obs = number of observations, fstat = F-statistic from the model,
# tstat = t-statistic on the ECT, case = 3 (unrestricted intercept, no trend),
# k = number of regressors
pss_result <- pssbounds(obs = nrow(data_df) - 1,  # effective obs after differencing
                        fstat = summary(dyn_model$model)$fstatistic[1],
                        tstat = summary(dyn_model$model)$coefficients["l.1.y", "t value"],
                        case = 3,
                        k = 3)

cat("\nPSS Bounds test output:\n")
print(pss_result)

# ==============================================================================
# PART C: Diagnostics on UECM residuals
# ==============================================================================
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("PART C: Residual Diagnostics\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

resid_uecm <- residuals(uecm_model)

# Breusch-Godfrey serial correlation test
bg_test <- Box.test(resid_uecm, lag = 4, type = "Ljung-Box")
cat("\nLjung-Box test (lag=4):\n")
cat("  Statistic:", bg_test$statistic, "\n")
cat("  p-value:", bg_test$p.value, "\n")

# Shapiro-Wilk normality
sw_test <- shapiro.test(resid_uecm)
cat("\nShapiro-Wilk test:\n")
cat("  Statistic:", sw_test$statistic, "\n")
cat("  p-value:", sw_test$p.value, "\n")

# Residual statistics
cat("\nResidual statistics:\n")
cat("  Mean:", mean(resid_uecm), "\n")
cat("  Std:", sd(resid_uecm), "\n")
cat("  Min:", min(resid_uecm), "\n")
cat("  Max:", max(resid_uecm), "\n")

# ==============================================================================
# Save results
# ==============================================================================

results <- list(
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  ardl_version = as.character(packageVersion("ARDL")),
  dynamac_version = as.character(packageVersion("dynamac")),
  dataset = "ardl_synthetic.csv",
  seed = 42,
  uecm = list(
    speed_of_adjustment = as.numeric(ect_coef),
    half_life_quarters = as.numeric(log(0.5) / log(1 + ect_coef)),
    long_run = list(
      x1 = as.numeric(lr_x1_uecm),
      x2 = as.numeric(lr_x2_uecm),
      x3 = as.numeric(lr_x3_uecm)
    ),
    r_squared = summary(uecm_model)$r.squared,
    coefficients = as.list(uecm_coefs)
  ),
  recm = list(
    ect_coefficient = as.numeric(ect_recm),
    r_squared = summary(recm_model)$r.squared,
    coefficients = as.list(recm_coefs)
  ),
  long_run_multipliers = as.list(as.data.frame(lr_mult)),
  dynardl = list(
    speed_of_adjustment = as.numeric(ect_dyn),
    long_run = list(
      x1 = as.numeric(lr_x1_dyn),
      x2 = as.numeric(lr_x2_dyn),
      x3 = as.numeric(lr_x3_dyn)
    ),
    coefficients = as.list(dyn_coefs)
  ),
  diagnostics = list(
    ljung_box_lag4 = list(
      statistic = as.numeric(bg_test$statistic),
      p_value = as.numeric(bg_test$p.value)
    ),
    shapiro_wilk = list(
      statistic = as.numeric(sw_test$statistic),
      p_value = as.numeric(sw_test$p.value)
    ),
    residual_mean = mean(resid_uecm),
    residual_std = sd(resid_uecm)
  ),
  true_dgp = list(
    long_run_x1 = 0.6,
    speed_of_adjustment = -0.25,
    intercept = 1.5
  )
)

out_dir <- file.path(script_dir, "..", "outputs", "R")
if (!dir.exists(out_dir)) {
  out_dir <- "examples/ardl/outputs/R"
}
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

out_path <- file.path(out_dir, "ecm_results_R.json")
writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE), out_path)
cat("\nResults saved to:", normalizePath(out_path, mustWork = FALSE), "\n")

cat("\n=== DONE: 02_ecm_validation.R ===\n")
