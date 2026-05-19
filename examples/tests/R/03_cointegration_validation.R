################################################################################
# 03_cointegration_validation.R
# Cross-validation of cointegration tests
# Engle-Granger: lm() + ur.df() on residuals
# Johansen: urca::ca.jo() (trace and max eigenvalue)
# Phillips-Ouliaris: urca::ca.po()
#
# Compares results with Python chronobox outputs.
# Tolerance: < 1e-4 for test statistics, < 0.01 for p-values
################################################################################

library(urca)
library(jsonlite)

set.seed(42)

cat("======================================================================\n")
cat("Cointegration Tests Validation\n")
cat("======================================================================\n\n")

# --- Data Generation ---

generate_unit_root_process <- function(n = 200, phi = 1.0, seed_offset = 0) {
  set.seed(42 + seed_offset)
  eps <- rnorm(n, 0, 1)
  y <- cumsum(eps)
  return(y)
}

generate_cointegrated_pair <- function(n = 200, beta = 1.5,
                                        sigma_eq = 0.5, sigma_x = 1.0) {
  # x is a random walk
  eps_x <- rnorm(n, 0, sigma_x)
  x <- cumsum(eps_x)

  # Equilibrium error is stationary AR(1) with phi=0.3
  eps_eq <- rnorm(n, 0, sigma_eq)
  eq_error <- numeric(n)
  eq_error[1] <- eps_eq[1]
  for (t in 2:n) {
    eq_error[t] <- 0.3 * eq_error[t - 1] + eps_eq[t]
  }

  # y = beta * x + stationary error
  y <- beta * x + eq_error

  return(data.frame(y = y, x = x, equilibrium_error = eq_error))
}

# --- Generate data ---
cat("Generating synthetic data...\n")

# Cointegrated pair
set.seed(42)
coint_pair <- generate_cointegrated_pair(n = 200, beta = 1.5, sigma_eq = 0.5)

# Independent I(1) series
set.seed(1042)
y_indep1 <- cumsum(rnorm(200))
set.seed(2042)
y_indep2 <- cumsum(rnorm(200))

# Trivariate cointegrated system (rank=1): z = 2*x + noise
set.seed(42)
tri_coint <- generate_cointegrated_pair(n = 200, beta = 1.5, sigma_eq = 0.5)
set.seed(542)
eps_z <- rnorm(200, 0, 0.5)
z_coint <- numeric(200)
z_coint[1] <- eps_z[1]
for (t in 2:200) {
  z_coint[t] <- 0.3 * z_coint[t - 1] + eps_z[t]
}
z_coint <- 2.0 * tri_coint$x + z_coint

# 3 series with common trend (rank=2)
set.seed(42)
common_trend <- cumsum(rnorm(200))
set.seed(642)
s1 <- common_trend + rnorm(200, 0, 0.5)
set.seed(742)
s2 <- 1.5 * common_trend + rnorm(200, 0, 0.5)
set.seed(842)
s3 <- 0.8 * common_trend + rnorm(200, 0, 0.5)

# --- Load real GDP datasets ---
cat("Loading GDP datasets from CSV...\n")
data_dir <- "/home/guhaase/projetos/chronobox/examples/tests/data"
us_gdp <- read.csv(file.path(data_dir, "us_gdp_quarterly.csv"))
br_gdp <- read.csv(file.path(data_dir, "brazil_gdp.csv"))

log_gdp_us <- us_gdp$log_gdp
log_gdp_br <- br_gdp$log_gdp

# Align GDP series (Brazil has 120 obs, US has 200)
# Use last 120 obs of US to match Brazil
n_br <- length(log_gdp_br)
log_gdp_us_aligned <- tail(log_gdp_us, n_br)

cat("  US GDP aligned: ", length(log_gdp_us_aligned), " obs\n")
cat("  Brazil GDP: ", length(log_gdp_br), " obs\n\n")

################################################################################
# 1. Integration check (prerequisite for cointegration)
################################################################################
cat("--- Integration Check ---\n\n")

integration_check <- list()

check_integration <- function(y, series_name) {
  # Level
  adf_level <- ur.df(y, type = "drift", lags = trunc((length(y) - 1)^(1/3)),
                      selectlags = "AIC")
  s_level <- summary(adf_level)
  stat_level <- s_level@teststat[1, 1]
  cv_level <- s_level@cval[1, "5pct"]

  # First difference
  dy <- diff(y)
  adf_diff <- ur.df(dy, type = "drift", lags = trunc((length(dy) - 1)^(1/3)),
                     selectlags = "AIC")
  s_diff <- summary(adf_diff)
  stat_diff <- s_diff@teststat[1, 1]
  cv_diff <- s_diff@cval[1, "5pct"]

  order <- ifelse(stat_level > cv_level && stat_diff < cv_diff, "I(1)", "unknown")

  cat(sprintf("  %s: level=%.4f (cv=%.4f), diff=%.4f (cv=%.4f) => %s\n",
              series_name, stat_level, cv_level, stat_diff, cv_diff, order))

  return(list(
    series = series_name,
    adf_level_stat = as.numeric(stat_level),
    adf_diff_stat = as.numeric(stat_diff),
    order = order
  ))
}

integration_check[[1]] <- check_integration(coint_pair$y, "y (cointegrado)")
integration_check[[2]] <- check_integration(coint_pair$x, "x (cointegrado)")
integration_check[[3]] <- check_integration(y_indep1, "y_indep1")
integration_check[[4]] <- check_integration(y_indep2, "y_indep2")
integration_check[[5]] <- check_integration(log_gdp_us, "Log PIB EUA")
integration_check[[6]] <- check_integration(log_gdp_br, "Log PIB Brasil")

cat("\n")

################################################################################
# 2. Engle-Granger cointegration test
################################################################################
cat("--- Engle-Granger Cointegration Test ---\n\n")

eg_results <- list()

run_engle_granger <- function(y, x, pair_name, trend = "c") {
  # Step 1: Cointegrating regression
  if (trend == "c") {
    model <- lm(y ~ x)
  } else if (trend == "n") {
    model <- lm(y ~ x - 1)
  } else if (trend == "ct") {
    t_idx <- 1:length(y)
    model <- lm(y ~ x + t_idx)
  }

  # Step 2: Test residuals for unit root
  resids <- residuals(model)
  lags <- trunc((length(resids) - 1)^(1/3))
  adf_resid <- ur.df(resids, type = "none", lags = lags, selectlags = "AIC")
  s <- summary(adf_resid)

  stat <- s@teststat[1, 1]

  # Engle-Granger critical values (2 variables, from MacKinnon 1996/2010)
  # These differ from standard ADF critical values
  n <- length(resids)
  cv <- c("1%" = -3.96, "5%" = -3.37, "10%" = -3.07)

  reject_5pct <- stat < cv["5%"]

  coef_beta <- coef(model)[2]

  cat(sprintf("  %s [trend=%s]: tau = %.4f, cv(5%%) = %.4f, beta = %.4f => %s\n",
              pair_name, trend, stat, cv["5%"], coef_beta,
              ifelse(reject_5pct, "COINTEGRATED", "NOT COINTEGRATED")))

  return(list(
    test = "Engle-Granger",
    pair = pair_name,
    trend = trend,
    statistic = as.numeric(stat),
    critical_values = as.list(cv),
    lags_used = adf_resid@lag,
    reject_at_5pct = as.logical(reject_5pct),
    decision = ifelse(reject_5pct, "Cointegradas", "Nao cointegradas"),
    cointegrating_vector = as.numeric(coef_beta)
  ))
}

# Cointegrated pair
eg_results[[1]] <- run_engle_granger(coint_pair$y, coint_pair$x,
                                      "Par cointegrado (beta=1.5)", trend = "c")

# Independent series
eg_results[[2]] <- run_engle_granger(y_indep1, y_indep2,
                                      "Series independentes", trend = "c")

# Different trend specifications
eg_results[[3]] <- run_engle_granger(coint_pair$y, coint_pair$x,
                                      "Cointegrado (trend=n)", trend = "n")
eg_results[[4]] <- run_engle_granger(coint_pair$y, coint_pair$x,
                                      "Cointegrado (trend=c)", trend = "c")
eg_results[[5]] <- run_engle_granger(coint_pair$y, coint_pair$x,
                                      "Cointegrado (trend=ct)", trend = "ct")

# GDP pair
eg_results[[6]] <- run_engle_granger(log_gdp_br, log_gdp_us_aligned,
                                      "PIB Brasil ~ PIB EUA", trend = "c")

cat("\n")

################################################################################
# 3. Phillips-Ouliaris test (urca::ca.po)
################################################################################
cat("--- Phillips-Ouliaris Test (urca::ca.po) ---\n\n")

po_results <- list()

run_po <- function(y, x, pair_name, demean = "constant", type = "Pu") {
  z <- cbind(y, x)

  test <- ca.po(z, demean = demean, type = type)
  s <- summary(test)

  stat <- s@teststat[1]
  cv <- s@cval[1, ]

  reject_5pct <- stat < cv["5pct"]

  cat(sprintf("  %s [demean=%s, type=%s]: stat = %.4f, cv(5%%) = %.4f => %s\n",
              pair_name, demean, type, stat, cv["5pct"],
              ifelse(reject_5pct, "COINTEGRATED", "NOT COINTEGRATED")))

  return(list(
    test = "Phillips-Ouliaris",
    pair = pair_name,
    demean = demean,
    type = type,
    statistic = as.numeric(stat),
    critical_values = list(
      "1%" = as.numeric(cv["1pct"]),
      "5%" = as.numeric(cv["5pct"]),
      "10%" = as.numeric(cv["10pct"])
    ),
    reject_at_5pct = as.logical(reject_5pct),
    decision = ifelse(reject_5pct, "Cointegradas", "Nao cointegradas")
  ))
}

# Cointegrated pair
po_results[[1]] <- run_po(coint_pair$y, coint_pair$x,
                           "Par cointegrado", type = "Pu")

# Independent series
po_results[[2]] <- run_po(y_indep1, y_indep2,
                           "Series independentes", type = "Pu")

# GDP pair
po_results[[3]] <- run_po(log_gdp_br, log_gdp_us_aligned,
                           "PIB Brasil ~ PIB EUA", type = "Pu")

# Also try Pz type
po_results[[4]] <- run_po(coint_pair$y, coint_pair$x,
                           "Par cointegrado (Pz)", type = "Pz")

cat("\n")

################################################################################
# 4. Johansen Cointegration Test (urca::ca.jo)
################################################################################
cat("--- Johansen Cointegration Test (urca::ca.jo) ---\n\n")

johansen_results <- list()

run_johansen <- function(z, system_name, K = 2, type = "trace",
                          ecdet = "const", spec = "longrun") {
  test <- ca.jo(z, type = type, ecdet = ecdet, K = K, spec = spec)
  s <- summary(test)

  # Extract trace/max eigenvalue statistics
  test_stats <- s@teststat
  crit_vals <- s@cval

  # Determine rank
  rank <- 0
  for (i in length(test_stats):1) {
    if (test_stats[i] > crit_vals[i, "5pct"]) {
      rank <- length(test_stats) - i + 1
      break
    }
  }

  # Actually, ca.jo reports in reverse order (r=0 first)
  # Need to count how many hypotheses are rejected
  rank <- 0
  for (i in 1:length(test_stats)) {
    if (test_stats[i] > crit_vals[i, "5pct"]) {
      rank <- rank + 1
    } else {
      break
    }
  }

  eigenvalues <- test@lambda

  cat(sprintf("  %s [K=%d, ecdet=%s, type=%s]: rank = %d\n",
              system_name, K, ecdet, type, rank))
  for (i in 1:length(test_stats)) {
    cat(sprintf("    r <= %d: stat = %.4f, cv(5%%) = %.4f %s\n",
                ncol(z) - i, test_stats[i], crit_vals[i, "5pct"],
                ifelse(test_stats[i] > crit_vals[i, "5pct"], "***", "")))
  }

  # Eigenvectors
  evecs <- test@V

  return(list(
    test = "Johansen",
    system = system_name,
    type = type,
    ecdet = ecdet,
    K = K,
    n_variables = ncol(z),
    eigenvalues = as.numeric(eigenvalues),
    test_stats = as.numeric(test_stats),
    crit_values_5pct = as.numeric(crit_vals[, "5pct"]),
    crit_values_1pct = as.numeric(crit_vals[, "1pct"]),
    crit_values_10pct = as.numeric(crit_vals[, "10pct"]),
    rank = rank,
    eigenvectors = lapply(1:ncol(evecs), function(j) as.numeric(evecs[, j])),
    nobs = nrow(z) - K + 1
  ))
}

# Bivariate: cointegrated pair
z_coint_bi <- cbind(coint_pair$y, coint_pair$x)

johansen_results[[1]] <- run_johansen(z_coint_bi, "Par cointegrado",
                                       K = 2, type = "trace", ecdet = "const")
johansen_results[[2]] <- run_johansen(z_coint_bi, "Par cointegrado (maxeig)",
                                       K = 2, type = "eigen", ecdet = "const")

cat("\n")

# Independent series
z_indep <- cbind(y_indep1, y_indep2)
johansen_results[[3]] <- run_johansen(z_indep, "Series independentes",
                                       K = 2, type = "trace", ecdet = "const")

cat("\n")

# Different deterministic specs
johansen_results[[4]] <- run_johansen(z_coint_bi, "Cointegrado (ecdet=none)",
                                       K = 2, type = "trace", ecdet = "none")
johansen_results[[5]] <- run_johansen(z_coint_bi, "Cointegrado (ecdet=const)",
                                       K = 2, type = "trace", ecdet = "const")
johansen_results[[6]] <- run_johansen(z_coint_bi, "Cointegrado (ecdet=trend)",
                                       K = 2, type = "trace", ecdet = "trend")

cat("\n")

# GDP pair
z_gdp <- cbind(log_gdp_br, log_gdp_us_aligned)
johansen_results[[7]] <- run_johansen(z_gdp, "PIB Brasil x EUA",
                                       K = 4, type = "trace", ecdet = "const")

cat("\n")

# Trivariate system (rank=1)
z_tri_1 <- cbind(tri_coint$y, tri_coint$x, z_coint)
johansen_results[[8]] <- run_johansen(z_tri_1,
                                       "Sistema trivariado (rank esperado=1)",
                                       K = 2, type = "trace", ecdet = "const")

cat("\n")

# 3 series with common trend (rank=2)
z_tri_2 <- cbind(s1, s2, s3)
johansen_results[[9]] <- run_johansen(z_tri_2,
                                       "3 series com trend comum (rank esperado=2)",
                                       K = 2, type = "trace", ecdet = "const")

cat("\n")

################################################################################
# 5. Save results
################################################################################
cat("Saving results...\n")

output <- list(
  metadata = list(
    script = "03_cointegration_validation.R",
    tests = c("Engle-Granger", "Phillips-Ouliaris", "Johansen"),
    seed = 42,
    packages = c("urca"),
    tolerance = list(
      statistic = 1e-4,
      pvalue = 0.01
    ),
    note = paste("Engle-Granger uses MacKinnon critical values (approximated).",
                 "Johansen uses Osterwald-Lenum (1992) tables via urca.",
                 "GDP data loaded from CSV for exact comparison.")
  ),
  integration_check = integration_check,
  engle_granger_results = eg_results,
  phillips_ouliaris_results = po_results,
  johansen_results = johansen_results
)

out_dir <- "/home/guhaase/projetos/chronobox/examples/tests/outputs/R"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write(toJSON(output, auto_unbox = TRUE, pretty = TRUE),
      file.path(out_dir, "cointegration_results.json"))

cat("Results saved to outputs/R/cointegration_results.json\n")

################################################################################
# 6. Summary
################################################################################
cat("\n")
cat("======================================================================\n")
cat("SUMMARY: Cointegration Test Results\n")
cat("======================================================================\n\n")

cat(sprintf("%-35s | %-10s | %-10s | %-10s\n",
            "Par/Sistema", "E-G", "P-O", "Johansen"))
cat(rep("-", 75), sep = "")
cat("\n")

pairs <- c("Par cointegrado (beta=1.5)", "Series independentes",
            "PIB Brasil ~ PIB EUA")
for (p in pairs) {
  eg_dec <- ""
  po_dec <- ""
  jo_dec <- ""

  for (r in eg_results) {
    if (r$pair == p) {
      eg_dec <- ifelse(r$reject_at_5pct, "Coint.", "Nao coint.")
      break
    }
  }
  for (r in po_results) {
    if (r$pair == p || grepl(gsub("\\(.*", "", p), r$pair)) {
      po_dec <- ifelse(r$reject_at_5pct, "Coint.", "Nao coint.")
      break
    }
  }
  for (r in johansen_results) {
    if (grepl(gsub("\\(.*", "", p), r$system)) {
      jo_dec <- paste0("rank=", r$rank)
      break
    }
  }

  cat(sprintf("%-35s | %-10s | %-10s | %-10s\n", p, eg_dec, po_dec, jo_dec))
}

cat("\nDone.\n")
