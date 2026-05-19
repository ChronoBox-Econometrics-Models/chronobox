################################################################################
# 02_kpss_ers_za_validation.R
# Cross-validation of KPSS, ERS (DF-GLS) and Zivot-Andrews tests
# Reference packages: urca (ur.kpss, ur.ers, ur.za), tseries (kpss.test)
#
# Compares results with Python chronobox outputs.
# Tolerance: < 1e-4 for test statistics, < 0.01 for p-values
################################################################################

library(urca)
library(tseries)
library(jsonlite)

set.seed(42)

cat("======================================================================\n")
cat("KPSS, ERS (DF-GLS) and Zivot-Andrews Validation\n")
cat("======================================================================\n\n")

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

generate_structural_break <- function(n = 200, break_point = 0.5,
                                       shift = 3.0, sigma = 1.0,
                                       trend_before = 0.0, trend_after = 0.0) {
  eps <- rnorm(n, 0, sigma)
  bp <- as.integer(n * break_point)
  t_idx <- 0:(n - 1)

  mean_val <- numeric(n)
  mean_val[1:bp] <- trend_before * t_idx[1:bp]
  mean_val[(bp + 1):n] <- trend_before * bp + shift +
    trend_after * (t_idx[(bp + 1):n] - bp)

  y <- mean_val + eps
  return(list(y = y, break_index = bp))
}

generate_trend_stationary <- function(n = 200, trend_coef = 0.05,
                                       sigma = 1.0, phi = 0.5) {
  t_idx <- 0:(n - 1)
  eps1 <- rnorm(n, 0, sigma)
  eps2 <- rnorm(n, 0, sigma)

  u <- numeric(n)
  u[1] <- eps1[1]
  for (i in 2:n) {
    u[i] <- phi * u[i - 1] + eps1[i]
  }
  ts_series <- 10.0 + trend_coef * t_idx + u

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
i0 <- generate_unit_root_process(n = 200, phi = 0.5)
i1 <- generate_unit_root_process(n = 200, phi = 1.0)

set.seed(342)
near_ur <- generate_unit_root_process(n = 200, phi = 0.95)

set.seed(442)
sb <- generate_structural_break(n = 200, break_point = 0.5, shift = 3.0)
structural_break <- sb$y

set.seed(42)
ts_df <- generate_trend_stationary(n = 200)

# --- Load real GDP datasets ---
cat("Loading GDP datasets from CSV...\n")
data_dir <- "/home/guhaase/projetos/chronobox/examples/tests/data"
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
# 1. KPSS Tests using urca::ur.kpss()
################################################################################
cat("--- KPSS Tests (urca::ur.kpss) ---\n\n")

kpss_results <- list()

run_kpss_urca <- function(y, series_name, type = "mu", lags = "short") {
  test <- ur.kpss(y, type = type, lags = lags)
  s <- summary(test)

  stat <- s@teststat[1]
  cv <- s@cval[1, ]

  # KPSS: H0 is stationarity, so reject if stat > cv
  reject_5pct <- stat > cv["5pct"]

  cat(sprintf("  %s [type=%s]: stat = %.4f, cv(5%%) = %.4f => %s\n",
              series_name, type, stat, cv["5pct"],
              ifelse(reject_5pct, "REJECT H0 (unit root evidence)",
                     "FAIL TO REJECT (stationary)")))

  return(list(
    test = "KPSS",
    series = series_name,
    regression = type,
    lags = lags,
    statistic = as.numeric(stat),
    critical_values = list(
      "1%" = as.numeric(cv["1pct"]),
      "5%" = as.numeric(cv["5pct"]),
      "10%" = as.numeric(cv["10pct"])
    ),
    reject_at_5pct = as.logical(reject_5pct),
    decision = ifelse(reject_5pct,
                      "Rejeita H0 (evidencia de raiz unitaria)",
                      "Nao rejeita H0 (serie e estacionaria)")
  ))
}

# KPSS on synthetic series
kpss_results[[1]] <- run_kpss_urca(i0, "I(0) sintetica", type = "mu")
kpss_results[[2]] <- run_kpss_urca(i1, "I(1) sintetica", type = "mu")
kpss_results[[3]] <- run_kpss_urca(near_ur, "Near UR (phi=0.95)", type = "mu")
kpss_results[[4]] <- run_kpss_urca(structural_break, "Com quebra estrutural",
                                    type = "mu")

# KPSS level vs trend
kpss_results[[5]] <- run_kpss_urca(i0, "I(0) (level)", type = "mu")
kpss_results[[6]] <- run_kpss_urca(i0, "I(0) (trend)", type = "tau")
kpss_results[[7]] <- run_kpss_urca(ts_df$trend_stationary,
                                    "Trend-stationary (level)", type = "mu")
kpss_results[[8]] <- run_kpss_urca(ts_df$trend_stationary,
                                    "Trend-stationary (trend)", type = "tau")

cat("\n")

# KPSS on GDP
cat("--- KPSS on GDP Series ---\n\n")
gdp_kpss_results <- list()
gdp_kpss_results[[1]] <- run_kpss_urca(log_gdp_us, "Log PIB EUA (nivel)",
                                         type = "mu")
gdp_kpss_results[[2]] <- run_kpss_urca(dlog_gdp_us, "Log PIB EUA (1a diff)",
                                         type = "mu")
gdp_kpss_results[[3]] <- run_kpss_urca(log_gdp_br, "Log PIB Brasil (nivel)",
                                         type = "mu")
gdp_kpss_results[[4]] <- run_kpss_urca(dlog_gdp_br, "Log PIB Brasil (1a diff)",
                                         type = "mu")

cat("\n")

# KPSS via tseries::kpss.test
cat("--- KPSS (tseries::kpss.test) ---\n\n")
tseries_kpss <- list()

run_kpss_tseries <- function(y, series_name, null = "Level") {
  test <- kpss.test(y, null = null)
  cat(sprintf("  %s [null=%s]: stat = %.4f, p-value = %.4f\n",
              series_name, null, test$statistic, test$p.value))
  return(list(
    test = "KPSS (tseries)",
    series = series_name,
    null_hypothesis = null,
    statistic = as.numeric(test$statistic),
    pvalue = test$p.value,
    reject_at_5pct = test$p.value < 0.05
  ))
}

tseries_kpss[[1]] <- run_kpss_tseries(log_gdp_us, "Log PIB EUA (nivel)")
tseries_kpss[[2]] <- run_kpss_tseries(dlog_gdp_us, "Log PIB EUA (1a diff)")
tseries_kpss[[3]] <- run_kpss_tseries(log_gdp_br, "Log PIB Brasil (nivel)")
tseries_kpss[[4]] <- run_kpss_tseries(dlog_gdp_br, "Log PIB Brasil (1a diff)")

cat("\n")

################################################################################
# 2. ERS / DF-GLS Tests using urca::ur.ers()
################################################################################
cat("--- ERS / DF-GLS Tests (urca::ur.ers) ---\n\n")

ers_results <- list()

run_ers <- function(y, series_name, type = "DF-GLS", model = "constant",
                     lag.max = NULL) {
  if (is.null(lag.max)) {
    lag.max <- trunc(4 * (length(y) / 100)^(1/4))
  }

  test <- ur.ers(y, type = type, model = model, lag.max = lag.max)
  s <- summary(test)

  stat <- s@teststat[1]
  cv <- s@cval[1, ]

  reject_5pct <- stat < cv["5pct"]

  cat(sprintf("  %s [model=%s]: stat = %.4f, cv(5%%) = %.4f => %s\n",
              series_name, model, stat, cv["5pct"],
              ifelse(reject_5pct, "REJECT H0", "FAIL TO REJECT")))

  return(list(
    test = "ERS",
    series = series_name,
    type = type,
    regression = model,
    statistic = as.numeric(stat),
    critical_values = list(
      "1%" = as.numeric(cv["1pct"]),
      "5%" = as.numeric(cv["5pct"]),
      "10%" = as.numeric(cv["10pct"])
    ),
    reject_at_5pct = as.logical(reject_5pct),
    decision = ifelse(reject_5pct, "Rejeita H0 (estacionaria)",
                      "Nao rejeita H0 (raiz unitaria)")
  ))
}

# ERS on synthetic series
ers_results[[1]] <- run_ers(i0, "I(0) sintetica", model = "constant")
ers_results[[2]] <- run_ers(i1, "I(1) sintetica", model = "constant")
ers_results[[3]] <- run_ers(near_ur, "Near UR (phi=0.95)", model = "constant")

# ERS with trend
ers_results[[4]] <- run_ers(ts_df$trend_stationary, "Trend-stationary",
                             model = "trend")
ers_results[[5]] <- run_ers(ts_df$difference_stationary, "Difference-stationary",
                             model = "trend")

cat("\n")

# ERS on GDP
cat("--- ERS on GDP Series ---\n\n")
gdp_ers_results <- list()
gdp_ers_results[[1]] <- run_ers(log_gdp_us, "Log PIB EUA (nivel)",
                                  model = "constant")
gdp_ers_results[[2]] <- run_ers(dlog_gdp_us, "Log PIB EUA (1a diff)",
                                  model = "constant")
gdp_ers_results[[3]] <- run_ers(log_gdp_br, "Log PIB Brasil (nivel)",
                                  model = "constant")
gdp_ers_results[[4]] <- run_ers(dlog_gdp_br, "Log PIB Brasil (1a diff)",
                                  model = "constant")

cat("\n")

################################################################################
# 3. Zivot-Andrews Tests using urca::ur.za()
################################################################################
cat("--- Zivot-Andrews Tests (urca::ur.za) ---\n\n")

za_results <- list()

run_za <- function(y, series_name, model = "both", lag = NULL) {
  test <- ur.za(y, model = model, lag = lag)
  s <- summary(test)

  stat <- s@teststat[1]
  cv <- s@cval[1, ]
  break_idx <- test@bpoint

  reject_5pct <- stat < cv["5pct"]

  cat(sprintf("  %s [model=%s]: stat = %.4f, cv(5%%) = %.4f, break at %d => %s\n",
              series_name, model, stat, cv["5pct"], break_idx,
              ifelse(reject_5pct, "REJECT H0", "FAIL TO REJECT")))

  return(list(
    test = "Zivot-Andrews",
    series = series_name,
    model = model,
    statistic = as.numeric(stat),
    critical_values = list(
      "1%" = as.numeric(cv["1pct"]),
      "5%" = as.numeric(cv["5pct"]),
      "10%" = as.numeric(cv["10pct"])
    ),
    break_index = break_idx,
    reject_at_5pct = as.logical(reject_5pct),
    decision = ifelse(reject_5pct,
                      "Rejeita H0 (estacionaria com quebra)",
                      "Nao rejeita H0 (raiz unitaria)")
  ))
}

# ZA on structural break series - all three models
za_results[[1]] <- run_za(structural_break, "Quebra estrutural (model=intercept)",
                           model = "intercept")
za_results[[2]] <- run_za(structural_break, "Quebra estrutural (model=trend)",
                           model = "trend")
za_results[[3]] <- run_za(structural_break, "Quebra estrutural (model=both)",
                           model = "both")

# ZA on stationary / unit root
za_results[[4]] <- run_za(i0, "I(0) sintetica (sem quebra)", model = "both")
za_results[[5]] <- run_za(i1, "I(1) sintetica (sem quebra)", model = "both")

cat("\n")

# ZA on GDP
cat("--- Zivot-Andrews on GDP Series ---\n\n")
gdp_za_results <- list()

gdp_za_results[[1]] <- run_za(log_gdp_us, "Log PIB EUA (nivel)", model = "both")
gdp_za_results[[2]] <- run_za(dlog_gdp_us, "Log PIB EUA (1a diff)", model = "both")
gdp_za_results[[3]] <- run_za(log_gdp_br, "Log PIB Brasil (nivel)",
                                model = "both")
gdp_za_results[[4]] <- run_za(dlog_gdp_br, "Log PIB Brasil (1a diff)",
                                model = "both")

# ZA Brazil detailed - all 3 models
cat("\n--- Zivot-Andrews Brazil GDP (all models) ---\n\n")
za_brazil_detailed <- list()
for (m in c("intercept", "trend", "both")) {
  za_brazil_detailed[[length(za_brazil_detailed) + 1]] <-
    run_za(log_gdp_br, paste0("Log PIB Brasil [", m, "]"), model = m)
}

cat("\n")

################################################################################
# 4. Battery of tests (comprehensive comparison)
################################################################################
cat("--- Battery of Tests ---\n\n")

battery_results <- list()

run_battery <- function(y, series_name) {
  # ADF
  lags <- trunc((length(y) - 1)^(1/3))
  adf <- ur.df(y, type = "drift", lags = lags, selectlags = "AIC")
  adf_s <- summary(adf)
  adf_stat <- adf_s@teststat[1, 1]
  adf_reject <- adf_stat < adf_s@cval[1, "5pct"]

  # KPSS
  kpss <- ur.kpss(y, type = "mu", lags = "short")
  kpss_s <- summary(kpss)
  kpss_stat <- kpss_s@teststat[1]
  kpss_reject <- kpss_stat > kpss_s@cval[1, "5pct"]

  # ERS
  lag_max <- trunc(4 * (length(y) / 100)^(1/4))
  ers <- ur.ers(y, type = "DF-GLS", model = "constant", lag.max = lag_max)
  ers_s <- summary(ers)
  ers_stat <- ers_s@teststat[1]
  ers_reject <- ers_stat < ers_s@cval[1, "5pct"]

  # ZA
  za <- ur.za(y, model = "both")
  za_s <- summary(za)
  za_stat <- za_s@teststat[1]
  za_reject <- za_stat < za_s@cval[1, "5pct"]

  # Conclusion
  n_reject_ur <- sum(c(adf_reject, ers_reject))  # ADF, ERS test H0: unit root
  n_reject_st <- sum(c(kpss_reject))  # KPSS tests H0: stationarity

  if (n_reject_ur >= 1 && !kpss_reject) {
    conclusion <- "ESTACIONARIA"
  } else {
    conclusion <- "RAIZ UNITARIA"
  }

  cat(sprintf("  %s: ADF=%s KPSS=%s ERS=%s ZA=%s => %s\n",
              series_name,
              ifelse(adf_reject, "rej", "fail"),
              ifelse(kpss_reject, "rej", "fail"),
              ifelse(ers_reject, "rej", "fail"),
              ifelse(za_reject, "rej", "fail"),
              conclusion))

  return(list(
    series = series_name,
    ADF = list(statistic = as.numeric(adf_stat), reject_at_5pct = adf_reject),
    KPSS = list(statistic = as.numeric(kpss_stat), reject_at_5pct = kpss_reject),
    ERS = list(statistic = as.numeric(ers_stat), reject_at_5pct = ers_reject),
    ZA = list(statistic = as.numeric(za_stat), reject_at_5pct = za_reject,
              break_index = za@bpoint),
    conclusion = conclusion
  ))
}

battery_results[["I(0) sintetica"]] <- run_battery(i0, "I(0) sintetica")
battery_results[["I(1) sintetica"]] <- run_battery(i1, "I(1) sintetica")
battery_results[["Near UR (phi=0.95)"]] <- run_battery(near_ur,
                                                         "Near UR (phi=0.95)")
battery_results[["Quebra estrutural"]] <- run_battery(structural_break,
                                                       "Quebra estrutural")
battery_results[["Log PIB EUA (nivel)"]] <- run_battery(log_gdp_us,
                                                          "Log PIB EUA (nivel)")
battery_results[["Log PIB EUA (1a diff)"]] <- run_battery(dlog_gdp_us,
                                                            "Log PIB EUA (1a diff)")
battery_results[["Log PIB Brasil (nivel)"]] <- run_battery(log_gdp_br,
                                                             "Log PIB Brasil (nivel)")
battery_results[["Log PIB Brasil (1a diff)"]] <- run_battery(dlog_gdp_br,
                                                               "Log PIB Brasil (1a diff)")

cat("\n")

################################################################################
# 5. Sample size sensitivity (exercise)
################################################################################
cat("--- Sample Size Sensitivity ---\n\n")

sample_size_results <- list()
for (n_size in c(50, 100, 200, 500)) {
  set.seed(42)
  near_ur_n <- generate_unit_root_process(n = n_size, phi = 0.95)

  lags <- trunc((n_size - 1)^(1/3))
  adf <- ur.df(near_ur_n, type = "drift", lags = lags, selectlags = "AIC")
  adf_s <- summary(adf)

  kpss <- ur.kpss(near_ur_n, type = "mu", lags = "short")
  kpss_s <- summary(kpss)

  lag_max <- trunc(4 * (n_size / 100)^(1/4))
  ers <- ur.ers(near_ur_n, type = "DF-GLS", model = "constant",
                lag.max = max(1, lag_max))
  ers_s <- summary(ers)

  cat(sprintf("  n=%d: ADF=%.4f KPSS=%.4f ERS=%.4f\n",
              n_size, adf_s@teststat[1, 1], kpss_s@teststat[1],
              ers_s@teststat[1]))

  sample_size_results[[length(sample_size_results) + 1]] <- list(
    n = n_size,
    ADF_stat = as.numeric(adf_s@teststat[1, 1]),
    ADF_rejeita = adf_s@teststat[1, 1] < adf_s@cval[1, "5pct"],
    KPSS_stat = as.numeric(kpss_s@teststat[1]),
    KPSS_rejeita = kpss_s@teststat[1] > kpss_s@cval[1, "5pct"],
    ERS_stat = as.numeric(ers_s@teststat[1]),
    ERS_rejeita = ers_s@teststat[1] < ers_s@cval[1, "5pct"]
  )
}

cat("\n")

################################################################################
# 6. Save all results
################################################################################
cat("Saving results...\n")

output <- list(
  metadata = list(
    script = "02_kpss_ers_za_validation.R",
    tests = c("KPSS", "ERS", "Zivot-Andrews"),
    seed = 42,
    packages = c("urca", "tseries"),
    tolerance = list(
      statistic = 1e-4,
      pvalue = 0.01
    ),
    note = paste("Synthetic data uses R's RNG. GDP data from CSV matches Python.",
                 "Critical values from urca may differ slightly from Python",
                 "due to table interpolation methods.")
  ),
  kpss_results = kpss_results,
  gdp_kpss_results = gdp_kpss_results,
  tseries_kpss = tseries_kpss,
  ers_results = ers_results,
  gdp_ers_results = gdp_ers_results,
  za_results = za_results,
  gdp_za_results = gdp_za_results,
  za_brazil_detailed = za_brazil_detailed,
  battery_results = battery_results,
  sample_size_sensitivity = sample_size_results
)

out_dir <- "/home/guhaase/projetos/chronobox/examples/tests/outputs/R"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write(toJSON(output, auto_unbox = TRUE, pretty = TRUE),
      file.path(out_dir, "kpss_ers_za_results.json"))

cat("Results saved to outputs/R/kpss_ers_za_results.json\n")
cat("\nDone.\n")
