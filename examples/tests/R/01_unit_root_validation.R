################################################################################
# 01_unit_root_validation.R
# Cross-validation of ADF and Phillips-Perron unit root tests
# Reference packages: urca (ur.df, ur.pp), tseries (adf.test, pp.test)
#
# Compares results with Python chronobox outputs.
# Tolerance: < 1e-4 for test statistics, < 0.01 for p-values
################################################################################

library(urca)
library(tseries)
library(jsonlite)

set.seed(42)

cat("=" %+% rep("=", 69) %+% "\n")
cat("Unit Root Validation: ADF and Phillips-Perron Tests\n")
cat("=" %+% rep("=", 69) %+% "\n\n")

# Helper for string concatenation
`%+%` <- function(a, b) paste0(a, b)

# --- Data Generation (R's RNG, seed=42) ---

generate_unit_root_process <- function(n = 200, phi = 1.0, order = 1,
                                        drift = 0.0, sigma = 1.0) {
  eps <- rnorm(n, 0, sigma)
  y <- numeric(n)
  y[1] <- eps[1]
  for (t in 2:n) {
    y[t] <- drift + phi * y[t - 1] + eps[t]
  }
  if (order == 2 && phi == 1.0) {
    y <- cumsum(y)
  }
  return(y)
}

generate_trend_stationary <- function(n = 200, trend_coef = 0.05,
                                       sigma = 1.0, phi = 0.5) {
  t_idx <- 0:(n - 1)
  eps1 <- rnorm(n, 0, sigma)
  eps2 <- rnorm(n, 0, sigma)

  # Trend-stationary: y = a + b*t + AR(1) error
  u <- numeric(n)
  u[1] <- eps1[1]
  for (i in 2:n) {
    u[i] <- phi * u[i - 1] + eps1[i]
  }
  ts_series <- 10.0 + trend_coef * t_idx + u

  # Difference-stationary: random walk with drift
  ds_series <- numeric(n)
  ds_series[1] <- 10.0
  for (i in 2:n) {
    ds_series[i] <- ds_series[i - 1] + trend_coef + eps2[i]
  }

  return(data.frame(trend_stationary = ts_series,
                    difference_stationary = ds_series))
}

# --- Generate synthetic data ---
cat("Generating synthetic data with set.seed(42)...\n")

set.seed(42)
i0 <- generate_unit_root_process(n = 200, phi = 0.5, order = 1)
i1 <- generate_unit_root_process(n = 200, phi = 1.0, order = 1)

set.seed(42)
# Re-seed for drift series to get fresh random numbers
set.seed(142)
i1_drift <- generate_unit_root_process(n = 200, phi = 1.0, order = 1, drift = 0.5)

set.seed(242)
i2 <- generate_unit_root_process(n = 200, phi = 1.0, order = 2)

set.seed(42)
ts_df <- generate_trend_stationary(n = 200)

# --- Load real GDP datasets ---
cat("Loading GDP datasets from CSV...\n")
data_dir <- file.path(dirname(getwd()), "tests", "data")
# Try multiple possible paths
if (!file.exists(file.path(data_dir, "us_gdp_quarterly.csv"))) {
  data_dir <- file.path(getwd(), "..", "data")
}
if (!file.exists(file.path(data_dir, "us_gdp_quarterly.csv"))) {
  data_dir <- file.path(getwd(), "examples", "tests", "data")
}
if (!file.exists(file.path(data_dir, "us_gdp_quarterly.csv"))) {
  # Absolute path fallback
  data_dir <- "/home/guhaase/projetos/chronobox/examples/tests/data"
}

us_gdp <- read.csv(file.path(data_dir, "us_gdp_quarterly.csv"))
br_gdp <- read.csv(file.path(data_dir, "brazil_gdp.csv"))

log_gdp_us <- us_gdp$log_gdp
log_gdp_br <- br_gdp$log_gdp
dlog_gdp_us <- diff(log_gdp_us)
dlog_gdp_br <- diff(log_gdp_br)

cat("  US GDP: ", length(log_gdp_us), " obs\n")
cat("  Brazil GDP: ", length(log_gdp_br), " obs\n\n")

# --- Results storage ---
results <- list()

################################################################################
# 1. ADF Tests using urca::ur.df()
################################################################################
cat("--- ADF Tests (urca::ur.df) ---\n\n")

run_adf_urca <- function(y, series_name, type = "drift", lags = NULL,
                          selectlags = "AIC") {
  if (is.null(lags)) {
    lags <- trunc((length(y) - 1)^(1/3))
  }

  test <- ur.df(y, type = type, lags = lags, selectlags = selectlags)
  s <- summary(test)

  # Extract test statistic (tau)
  stat <- s@teststat[1, 1]

  # Extract critical values
  cv <- s@cval[1, ]
  names(cv) <- c("1%", "5%", "10%")

  # Determine rejection at 5%
  reject_5pct <- stat < cv["5%"]

  cat(sprintf("  %s [type=%s]: tau = %.4f, cv(5%%) = %.4f => %s\n",
              series_name, type, stat, cv["5%"],
              ifelse(reject_5pct, "REJECT H0", "FAIL TO REJECT")))

  return(list(
    test = "ADF",
    series = series_name,
    regression = type,
    selectlags = selectlags,
    statistic = as.numeric(stat),
    critical_values = list(
      "1%" = as.numeric(cv["1%"]),
      "5%" = as.numeric(cv["5%"]),
      "10%" = as.numeric(cv["10%"])
    ),
    lags_used = test@lag,
    reject_at_5pct = as.logical(reject_5pct),
    decision = ifelse(reject_5pct, "Rejeita H0 (estacionaria)",
                      "Nao rejeita H0 (raiz unitaria)")
  ))
}

# ADF on synthetic series
adf_results <- list()

# I(0) - drift (constant)
adf_results[[1]] <- run_adf_urca(i0, "I(0) sintetica", type = "drift")

# I(1) - level
adf_results[[2]] <- run_adf_urca(i1, "I(1) sintetica - nivel", type = "drift")

# I(1) - first difference
adf_results[[3]] <- run_adf_urca(diff(i1), "I(1) sintetica - 1a diferenca",
                                  type = "drift")

# I(1) with drift - level (use trend)
adf_results[[4]] <- run_adf_urca(i1_drift, "I(1) com drift - nivel",
                                  type = "trend")

# I(1) with drift - first difference
adf_results[[5]] <- run_adf_urca(diff(i1_drift), "I(1) com drift - 1a diferenca",
                                  type = "drift")

# I(2) - level
adf_results[[6]] <- run_adf_urca(i2, "I(2) sintetica - nivel", type = "drift")

# I(2) - first difference
adf_results[[7]] <- run_adf_urca(diff(i2), "I(2) sintetica - 1a diferenca",
                                  type = "drift")

# I(2) - second difference
adf_results[[8]] <- run_adf_urca(diff(diff(i2)), "I(2) sintetica - 2a diferenca",
                                  type = "drift")

cat("\n")

# ADF on GDP series
cat("--- ADF on GDP Series ---\n\n")
gdp_adf_results <- list()

# US GDP - level (with constant)
gdp_adf_results[[1]] <- run_adf_urca(log_gdp_us, "Log PIB EUA (nivel)",
                                       type = "drift")
# US GDP - level (with trend)
gdp_adf_results[[2]] <- run_adf_urca(log_gdp_us, "Log PIB EUA (nivel)",
                                       type = "trend")
# US GDP - first difference
gdp_adf_results[[3]] <- run_adf_urca(dlog_gdp_us, "Log PIB EUA (1a diferenca)",
                                       type = "drift")

# Brazil GDP - level (with constant)
gdp_adf_results[[4]] <- run_adf_urca(log_gdp_br, "Log PIB Brasil (nivel)",
                                       type = "drift")
# Brazil GDP - level (with trend)
gdp_adf_results[[5]] <- run_adf_urca(log_gdp_br, "Log PIB Brasil (nivel)",
                                       type = "trend")
# Brazil GDP - first difference
gdp_adf_results[[6]] <- run_adf_urca(dlog_gdp_br, "Log PIB Brasil (1a diferenca)",
                                       type = "drift")

cat("\n")

# ADF specification comparison (none, drift, trend) for I(0)
cat("--- ADF Specification Comparison ---\n\n")
spec_results <- list()
for (spec in c("none", "drift", "trend")) {
  spec_results[[length(spec_results) + 1]] <- run_adf_urca(
    i0, paste0("I(0) sintetica [", spec, "]"), type = spec
  )
}
for (spec in c("none", "drift", "trend")) {
  spec_results[[length(spec_results) + 1]] <- run_adf_urca(
    i1, paste0("I(1) sintetica [", spec, "]"), type = spec
  )
}

cat("\n")

################################################################################
# 2. ADF Tests using tseries::adf.test() (alternative)
################################################################################
cat("--- ADF Tests (tseries::adf.test) ---\n\n")

tseries_adf_results <- list()

run_adf_tseries <- function(y, series_name, k = NULL) {
  if (is.null(k)) {
    k <- trunc((length(y) - 1)^(1/3))
  }
  test <- adf.test(y, k = k)
  cat(sprintf("  %s: Dickey-Fuller = %.4f, p-value = %.4f\n",
              series_name, test$statistic, test$p.value))
  return(list(
    test = "ADF (tseries)",
    series = series_name,
    statistic = as.numeric(test$statistic),
    pvalue = test$p.value,
    lags_used = test$parameter,
    reject_at_5pct = test$p.value < 0.05,
    decision = ifelse(test$p.value < 0.05,
                      "Rejeita H0 (estacionaria)",
                      "Nao rejeita H0 (raiz unitaria)")
  ))
}

tseries_adf_results[[1]] <- run_adf_tseries(i0, "I(0) sintetica")
tseries_adf_results[[2]] <- run_adf_tseries(i1, "I(1) sintetica - nivel")
tseries_adf_results[[3]] <- run_adf_tseries(diff(i1),
                                             "I(1) sintetica - 1a diferenca")
tseries_adf_results[[4]] <- run_adf_tseries(log_gdp_us, "Log PIB EUA (nivel)")
tseries_adf_results[[5]] <- run_adf_tseries(dlog_gdp_us,
                                             "Log PIB EUA (1a diferenca)")
tseries_adf_results[[6]] <- run_adf_tseries(log_gdp_br, "Log PIB Brasil (nivel)")
tseries_adf_results[[7]] <- run_adf_tseries(dlog_gdp_br,
                                             "Log PIB Brasil (1a diferenca)")

cat("\n")

################################################################################
# 3. Phillips-Perron Tests using urca::ur.pp()
################################################################################
cat("--- Phillips-Perron Tests (urca::ur.pp) ---\n\n")

pp_results <- list()

run_pp_urca <- function(y, series_name, type = "Z-tau",
                         model = "constant", lags = "short") {
  test <- ur.pp(y, type = type, model = model, lags = lags)
  s <- summary(test)

  stat <- s@teststat[1]
  cv <- s@cval[1, ]
  names(cv) <- c("1%", "5%", "10%")

  reject_5pct <- stat < cv["5%"]

  cat(sprintf("  %s [model=%s]: Z-tau = %.4f, cv(5%%) = %.4f => %s\n",
              series_name, model, stat, cv["5%"],
              ifelse(reject_5pct, "REJECT H0", "FAIL TO REJECT")))

  return(list(
    test = "PP",
    series = series_name,
    regression = model,
    type = type,
    lags = lags,
    statistic = as.numeric(stat),
    critical_values = list(
      "1%" = as.numeric(cv["1%"]),
      "5%" = as.numeric(cv["5%"]),
      "10%" = as.numeric(cv["10%"])
    ),
    reject_at_5pct = as.logical(reject_5pct),
    decision = ifelse(reject_5pct, "Rejeita H0 (estacionaria)",
                      "Nao rejeita H0 (raiz unitaria)")
  ))
}

# PP on synthetic series
pp_results[[1]] <- run_pp_urca(i0, "I(0) sintetica", model = "constant")
pp_results[[2]] <- run_pp_urca(i1, "I(1) sintetica - nivel", model = "constant")
pp_results[[3]] <- run_pp_urca(diff(i1), "I(1) sintetica - 1a diferenca",
                                model = "constant")
pp_results[[4]] <- run_pp_urca(i1_drift, "I(1) com drift - nivel",
                                model = "trend")
pp_results[[5]] <- run_pp_urca(diff(i1_drift), "I(1) com drift - 1a diferenca",
                                model = "constant")
pp_results[[6]] <- run_pp_urca(i2, "I(2) sintetica - nivel", model = "constant")
pp_results[[7]] <- run_pp_urca(diff(i2), "I(2) sintetica - 1a diferenca",
                                model = "constant")
pp_results[[8]] <- run_pp_urca(diff(diff(i2)), "I(2) sintetica - 2a diferenca",
                                model = "constant")

cat("\n")

# PP on GDP series
cat("--- Phillips-Perron on GDP Series ---\n\n")
gdp_pp_results <- list()

gdp_pp_results[[1]] <- run_pp_urca(log_gdp_us, "Log PIB EUA (nivel)",
                                     model = "constant")
gdp_pp_results[[2]] <- run_pp_urca(log_gdp_us, "Log PIB EUA (nivel)",
                                     model = "trend")
gdp_pp_results[[3]] <- run_pp_urca(dlog_gdp_us, "Log PIB EUA (1a diferenca)",
                                     model = "constant")
gdp_pp_results[[4]] <- run_pp_urca(log_gdp_br, "Log PIB Brasil (nivel)",
                                     model = "constant")
gdp_pp_results[[5]] <- run_pp_urca(log_gdp_br, "Log PIB Brasil (nivel)",
                                     model = "trend")
gdp_pp_results[[6]] <- run_pp_urca(dlog_gdp_br, "Log PIB Brasil (1a diferenca)",
                                     model = "constant")

cat("\n")

# PP also using tseries::pp.test
cat("--- Phillips-Perron (tseries::pp.test) ---\n\n")
tseries_pp_results <- list()

run_pp_tseries <- function(y, series_name) {
  test <- pp.test(y)
  cat(sprintf("  %s: Dickey-Fuller Z(alpha) = %.4f, p-value = %.4f\n",
              series_name, test$statistic, test$p.value))
  return(list(
    test = "PP (tseries)",
    series = series_name,
    statistic = as.numeric(test$statistic),
    pvalue = test$p.value,
    reject_at_5pct = test$p.value < 0.05
  ))
}

tseries_pp_results[[1]] <- run_pp_tseries(log_gdp_us, "Log PIB EUA (nivel)")
tseries_pp_results[[2]] <- run_pp_tseries(dlog_gdp_us, "Log PIB EUA (1a diferenca)")
tseries_pp_results[[3]] <- run_pp_tseries(log_gdp_br, "Log PIB Brasil (nivel)")
tseries_pp_results[[4]] <- run_pp_tseries(dlog_gdp_br, "Log PIB Brasil (1a diferenca)")

cat("\n")

################################################################################
# 4. Trend-stationary vs Difference-stationary
################################################################################
cat("--- Trend-stationary vs Difference-stationary ---\n\n")

trend_results <- list()
trend_results[[1]] <- run_adf_urca(ts_df$trend_stationary,
                                    "Trend-stationary", type = "trend")
trend_results[[2]] <- run_adf_urca(ts_df$difference_stationary,
                                    "Difference-stationary", type = "trend")
trend_results[[3]] <- run_pp_urca(ts_df$trend_stationary,
                                   "Trend-stationary", model = "trend")
trend_results[[4]] <- run_pp_urca(ts_df$difference_stationary,
                                   "Difference-stationary", model = "trend")

cat("\n")

################################################################################
# 5. Save results
################################################################################
cat("Saving results...\n")

output <- list(
  metadata = list(
    script = "01_unit_root_validation.R",
    tests = c("ADF", "PP"),
    seed = 42,
    packages = c("urca", "tseries"),
    tolerance = list(
      statistic = 1e-4,
      pvalue = 0.01
    ),
    note = paste("Synthetic data uses R's RNG (set.seed(42)), so exact values",
                 "differ from Python. GDP data loaded from CSV is identical.",
                 "Compare qualitative decisions and GDP statistics.")
  ),
  adf_results = adf_results,
  gdp_adf_results = gdp_adf_results,
  specification_comparison = spec_results,
  tseries_adf_results = tseries_adf_results,
  pp_results = pp_results,
  gdp_pp_results = gdp_pp_results,
  tseries_pp_results = tseries_pp_results,
  trend_vs_difference = trend_results
)

# Determine output directory
out_dir <- file.path(data_dir, "..", "outputs", "R")
if (!dir.exists(out_dir)) {
  out_dir <- "/home/guhaase/projetos/chronobox/examples/tests/outputs/R"
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write(toJSON(output, auto_unbox = TRUE, pretty = TRUE),
      file.path(out_dir, "unit_root_results.json"))

cat("Results saved to outputs/R/unit_root_results.json\n")

################################################################################
# 6. Summary Table
################################################################################
cat("\n")
cat("=" %+% rep("=", 69) %+% "\n")
cat("SUMMARY: Unit Root Test Results\n")
cat("=" %+% rep("=", 69) %+% "\n\n")

cat(sprintf("%-35s | %-12s | %-12s | %-12s\n",
            "Series", "ADF (urca)", "PP (urca)", "ADF (tseries)"))
cat(rep("-", 80), sep = "")
cat("\n")

# Print GDP results summary
gdp_series <- c("Log PIB EUA (nivel)", "Log PIB EUA (1a diferenca)",
                 "Log PIB Brasil (nivel)", "Log PIB Brasil (1a diferenca)")

for (s in gdp_series) {
  adf_dec <- ""
  pp_dec <- ""
  ts_dec <- ""

  for (r in gdp_adf_results) {
    if (r$series == s && r$regression == "drift") {
      adf_dec <- ifelse(r$reject_at_5pct, "REJECT", "FAIL")
    }
  }
  for (r in gdp_pp_results) {
    if (r$series == s && r$regression == "constant") {
      pp_dec <- ifelse(r$reject_at_5pct, "REJECT", "FAIL")
    }
  }
  for (r in tseries_adf_results) {
    if (r$series == s) {
      ts_dec <- ifelse(r$reject_at_5pct, "REJECT", "FAIL")
    }
  }

  cat(sprintf("%-35s | %-12s | %-12s | %-12s\n", s, adf_dec, pp_dec, ts_dec))
}

cat("\nDone.\n")
