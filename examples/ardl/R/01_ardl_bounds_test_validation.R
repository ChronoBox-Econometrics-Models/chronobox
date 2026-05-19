################################################################################
# 01_ardl_bounds_test_validation.R
#
# ARDL Bounds Test Validation - Reference implementation in R
# Uses the ARDL package for automatic lag selection and PSS bounds testing.
#
# Purpose: Cross-validate chronobox ARDL bounds test results.
# Dataset: examples/ardl/data/ardl_synthetic.csv (seed=42, 200 quarterly obs)
# Output:  examples/ardl/outputs/R/bounds_test_results_R.json
#
# Required packages: ARDL (>= 0.2.4), jsonlite
# Tolerance: F-statistic < 0.05 difference; coefficients < 1e-4
################################################################################

# -- Package versions ----------------------------------------------------------
cat("=== Package Versions ===\n")
cat("R version:", paste(R.version$major, R.version$minor, sep = "."), "\n")
for (pkg in c("ARDL", "jsonlite")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  cat(pkg, ":", as.character(packageVersion(pkg)), "\n")
}

library(ARDL)
library(jsonlite)

set.seed(42)

# -- Load data -----------------------------------------------------------------
script_dir <- dirname(sys.frame(1)$ofile %||% ".")
data_path  <- file.path(script_dir, "..", "data", "ardl_synthetic.csv")
if (!file.exists(data_path)) {
  # Fallback for interactive use
  data_path <- "examples/ardl/data/ardl_synthetic.csv"
}
cat("\nLoading data from:", normalizePath(data_path), "\n")

df <- read.csv(data_path, stringsAsFactors = FALSE)
df$date <- as.Date(df$date)
cat("Observations:", nrow(df), "\n")
cat("Variables:", paste(names(df), collapse = ", "), "\n")

# Convert to ts object (quarterly frequency starting 1970Q1)
y_ts  <- ts(df$y,  start = c(1970, 1), frequency = 4)
x1_ts <- ts(df$x1, start = c(1970, 1), frequency = 4)
x2_ts <- ts(df$x2, start = c(1970, 1), frequency = 4)
x3_ts <- ts(df$x3, start = c(1970, 1), frequency = 4)

data_ts <- ts.union(y = y_ts, x1 = x1_ts, x2 = x2_ts, x3 = x3_ts)
data_df <- as.data.frame(data_ts)
names(data_df) <- c("y", "x1", "x2", "x3")

# ==============================================================================
# 1. Automatic ARDL order selection
# ==============================================================================
cat("\n=== Automatic ARDL Order Selection (AIC) ===\n")

auto_model <- auto_ardl(y ~ x1 + x2 + x3, data = data_df,
                        max_order = 4, selection = "AIC")

best_model <- auto_model$best_model
best_order <- auto_model$best_order
cat("Best ARDL order:", paste0("ARDL(", paste(best_order, collapse = ", "), ")"), "\n")
cat("AIC:", AIC(best_model), "\n")
cat("BIC:", BIC(best_model), "\n")

# Print model summary
cat("\n--- Model Summary ---\n")
print(summary(best_model))

# Also fit the specific ARDL(1,1,1,1) that chronobox selected
cat("\n=== ARDL(1,1,1,1) - Matching chronobox specification ===\n")
ardl_1111 <- ardl(y ~ x1 + x2 + x3, data = data_df, order = c(1, 1, 1, 1))
cat("AIC:", AIC(ardl_1111), "\n")
cat("BIC:", BIC(ardl_1111), "\n")
print(summary(ardl_1111))

# ==============================================================================
# 2. Bounds F-test (Pesaran, Shin and Smith)
# ==============================================================================
cat("\n=== Bounds F-test ===\n")

# F-test on the best model
f_test_best <- bounds_f_test(best_model, case = 3)
cat("\n-- F-test on best model --\n")
print(f_test_best)

# F-test on ARDL(1,1,1,1)
f_test_1111 <- bounds_f_test(ardl_1111, case = 3)
cat("\n-- F-test on ARDL(1,1,1,1) --\n")
print(f_test_1111)

# ==============================================================================
# 3. Bounds t-test
# ==============================================================================
cat("\n=== Bounds t-test ===\n")

t_test_best <- bounds_t_test(best_model, case = 3)
cat("\n-- t-test on best model --\n")
print(t_test_best)

t_test_1111 <- bounds_t_test(ardl_1111, case = 3)
cat("\n-- t-test on ARDL(1,1,1,1) --\n")
print(t_test_1111)

# ==============================================================================
# 4. Extract results and long-run coefficients
# ==============================================================================

# Long-run coefficients from ARDL(1,1,1,1)
coefs <- coef(ardl_1111)
cat("\n=== Raw Coefficients ===\n")
print(coefs)

# Compute long-run multipliers: beta_j / (1 - phi_1)
phi_1 <- coefs["y:l(y, 1)"]
lr_x1 <- (coefs["x1:l(x1, 0)"] + coefs["x1:l(x1, 1)"]) / (1 - phi_1)
lr_x2 <- (coefs["x2:l(x2, 0)"] + coefs["x2:l(x2, 1)"]) / (1 - phi_1)
lr_x3 <- (coefs["x3:l(x3, 0)"] + coefs["x3:l(x3, 1)"]) / (1 - phi_1)

cat("\n=== Long-run Coefficients ===\n")
cat("x1:", lr_x1, "\n")
cat("x2:", lr_x2, "\n")
cat("x3:", lr_x3, "\n")

# ==============================================================================
# 5. Determine F-test conclusion
# ==============================================================================

# Extract F-statistic value
f_stat_value <- f_test_1111$statistic

# Extract critical values from the F-test
f_tab <- f_test_1111$tab
# The table has I(0) and I(1) bounds at various significance levels

cat("\n=== F-test Decision ===\n")
cat("F-statistic:", f_stat_value, "\n")
cat("Critical value table:\n")
print(f_tab)

# ==============================================================================
# 6. Save results to JSON
# ==============================================================================

results <- list(
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  ardl_version = as.character(packageVersion("ARDL")),
  dataset = "ardl_synthetic.csv",
  seed = 42,
  auto_ardl = list(
    best_order = as.list(best_order),
    aic = AIC(best_model),
    bic = BIC(best_model)
  ),
  ardl_1111 = list(
    order = c(1, 1, 1, 1),
    nobs = nobs(ardl_1111),
    r_squared = summary(ardl_1111)$r.squared,
    adj_r_squared = summary(ardl_1111)$adj.r.squared,
    aic = AIC(ardl_1111),
    bic = BIC(ardl_1111),
    coefficients = as.list(coef(ardl_1111)),
    long_run = list(
      x1 = as.numeric(lr_x1),
      x2 = as.numeric(lr_x2),
      x3 = as.numeric(lr_x3)
    )
  ),
  bounds_f_test = list(
    f_statistic = as.numeric(f_stat_value),
    critical_values = as.list(as.data.frame(f_tab))
  ),
  bounds_t_test = list(
    t_statistic = as.numeric(t_test_1111$statistic)
  )
)

out_dir <- file.path(script_dir, "..", "outputs", "R")
if (!dir.exists(out_dir)) {
  out_dir <- "examples/ardl/outputs/R"
}
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

out_path <- file.path(out_dir, "bounds_test_results_R.json")
writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE), out_path)
cat("\nResults saved to:", normalizePath(out_path, mustWork = FALSE), "\n")

cat("\n=== DONE: 01_ardl_bounds_test_validation.R ===\n")
