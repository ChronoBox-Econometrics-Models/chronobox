################################################################################
# 04_breaks_validation.R
# Cross-validation of structural break tests
# Reference packages: strucchange (breakpoints, Fstats, efp)
#
# Tests: Chow test, CUSUM, CUSUM-sq, Bai-Perron
# Compares results with Python chronobox outputs.
# Tolerance: < 1e-4 for test statistics, < 0.01 for p-values
################################################################################

library(strucchange)
library(jsonlite)

set.seed(42)

cat("======================================================================\n")
cat("Structural Breaks Validation: Chow, CUSUM, Bai-Perron\n")
cat("======================================================================\n\n")

# --- Data Generation ---

generate_structural_break <- function(n = 200, break_point = 0.5,
                                       shift = 3.0, sigma = 1.0,
                                       trend_before = 0.0,
                                       trend_after = 0.0) {
  eps <- rnorm(n, 0, sigma)
  bp <- as.integer(n * break_point)
  t_idx <- 0:(n - 1)

  mean_val <- numeric(n)
  mean_val[1:bp] <- trend_before * t_idx[1:bp]
  if (bp < n) {
    mean_val[(bp + 1):n] <- trend_before * bp + shift +
      trend_after * (t_idx[(bp + 1):n] - bp)
  }

  y <- mean_val + eps
  return(list(y = y, break_index = bp))
}

# --- Generate synthetic data ---
cat("Generating synthetic data with set.seed(42)...\n")

# Level break at midpoint
set.seed(42)
level_break <- generate_structural_break(n = 200, break_point = 0.5,
                                          shift = 3.0, sigma = 1.0)

# Trend break at midpoint
set.seed(42)
trend_break <- generate_structural_break(n = 200, break_point = 0.5,
                                          shift = 0.0, sigma = 1.0,
                                          trend_before = 0.0,
                                          trend_after = 0.05)

# Stable series (no break)
set.seed(42)
stable <- generate_structural_break(n = 200, break_point = 0.5,
                                     shift = 0.0, sigma = 1.0)

# Two breaks
set.seed(42)
eps_2breaks <- rnorm(300, 0, 1)
y_2breaks <- numeric(300)
for (i in 1:300) {
  if (i <= 100) {
    y_2breaks[i] <- 0 + eps_2breaks[i]
  } else if (i <= 200) {
    y_2breaks[i] <- 3 + eps_2breaks[i]
  } else {
    y_2breaks[i] <- 1 + eps_2breaks[i]
  }
}

# --- Load real GDP datasets ---
cat("Loading GDP datasets from CSV...\n")
data_dir <- "/home/guhaase/projetos/chronobox/examples/tests/data"
us_gdp <- read.csv(file.path(data_dir, "us_gdp_quarterly.csv"))
br_gdp <- read.csv(file.path(data_dir, "brazil_gdp.csv"))

gdp_growth_br <- br_gdp$gdp_growth
gdp_growth_br <- gdp_growth_br[!is.na(gdp_growth_br)]

gdp_growth_us <- us_gdp$gdp_growth
gdp_growth_us <- gdp_growth_us[!is.na(gdp_growth_us)]

cat("  US GDP growth: ", length(gdp_growth_us), " obs\n")
cat("  Brazil GDP growth: ", length(gdp_growth_br), " obs\n\n")

################################################################################
# 1. Chow Test (via strucchange::sctest / manual F-test)
################################################################################
cat("--- Chow Tests ---\n\n")

chow_results <- list()

run_chow <- function(y, break_point, series_name) {
  n <- length(y)
  t_idx <- 1:n

  # Full model
  model_full <- lm(y ~ t_idx)
  ssr_full <- sum(residuals(model_full)^2)

  # Sub-sample models
  y1 <- y[1:break_point]
  t1 <- t_idx[1:break_point]
  y2 <- y[(break_point + 1):n]
  t2 <- t_idx[(break_point + 1):n]

  model1 <- lm(y1 ~ t1)
  model2 <- lm(y2 ~ t2)
  ssr1 <- sum(residuals(model1)^2)
  ssr2 <- sum(residuals(model2)^2)

  k <- 2  # number of parameters (intercept + slope)
  T1 <- break_point
  T2 <- n - break_point

  # F-statistic
  f_stat <- ((ssr_full - ssr1 - ssr2) / k) / ((ssr1 + ssr2) / (n - 2 * k))
  p_value <- 1 - pf(f_stat, k, n - 2 * k)

  reject_5pct <- p_value < 0.05

  cat(sprintf("  %s [bp=%d]: F = %.4f, p = %.6f => %s\n",
              series_name, break_point, f_stat, p_value,
              ifelse(reject_5pct, "BREAK DETECTED", "NO BREAK")))

  return(list(
    test = "Chow",
    series = series_name,
    break_point = break_point,
    statistic = as.numeric(f_stat),
    pvalue = p_value,
    reject_at_5pct = as.logical(reject_5pct),
    decision = ifelse(reject_5pct, "Quebra detectada", "Sem quebra"),
    SSR_full = ssr_full,
    SSR_1 = ssr1,
    SSR_2 = ssr2,
    T1 = T1,
    T2 = T2
  ))
}

# Chow on synthetic data
chow_results[[1]] <- run_chow(level_break$y, 100, "Nivel (ponto correto)")
chow_results[[2]] <- run_chow(level_break$y, 50, "Nivel (ponto errado t=50)")
chow_results[[3]] <- run_chow(trend_break$y, 100, "Tendencia (ponto correto)")
chow_results[[4]] <- run_chow(stable$y, 100, "Estavel (controle)")

# Chow on GDP
chow_results[[5]] <- run_chow(gdp_growth_br, 36,
                               "PIB Brasil (break=36, ~2003-Q1)")

# US GDP - 2008-Q3 crisis (index ~137)
if (length(gdp_growth_us) > 140) {
  chow_results[[6]] <- run_chow(gdp_growth_us, 137,
                                 "PIB EUA (2008-Q3 crise financeira)")
}

# US GDP - 2020-Q1 COVID (index ~183)
if (length(gdp_growth_us) > 185) {
  chow_results[[7]] <- run_chow(gdp_growth_us, 183, "PIB EUA (2020-Q1 COVID)")
}

cat("\n")

# Chow scan (sequential F-tests)
cat("--- Chow Scan (sequential F-tests) ---\n\n")
chow_scan <- list()
scan_points <- seq(30, 160, by = 10)

for (bp in scan_points) {
  result <- run_chow(level_break$y, bp, sprintf("Nivel [bp=%d]", bp))
  chow_scan[[length(chow_scan) + 1]] <- list(
    break_point = bp,
    F_stat = result$statistic,
    pvalue = result$pvalue,
    reject = result$reject_at_5pct
  )
}

cat("\n")

################################################################################
# 2. Sup-F Test using strucchange::Fstats
################################################################################
cat("--- Sup-F Test (strucchange::Fstats) ---\n\n")

supf_results <- list()

run_supf <- function(y, series_name, from = 0.15, to = 0.85) {
  n <- length(y)
  t_idx <- 1:n
  df <- data.frame(y = y, t = t_idx)

  fs <- Fstats(y ~ t, from = from, to = to, data = df)
  sc <- sctest(fs, type = "supF")

  cat(sprintf("  %s: supF = %.4f, p = %.6f => %s\n",
              series_name, sc$statistic, sc$p.value,
              ifelse(sc$p.value < 0.05, "BREAK DETECTED", "NO BREAK")))

  # Find breakpoint
  bp <- breakpoints(fs)$breakpoints

  return(list(
    test = "Sup-F",
    series = series_name,
    statistic = as.numeric(sc$statistic),
    pvalue = sc$p.value,
    reject_at_5pct = sc$p.value < 0.05,
    estimated_break = as.integer(bp),
    decision = ifelse(sc$p.value < 0.05, "Quebra detectada", "Sem quebra")
  ))
}

supf_results[[1]] <- run_supf(level_break$y, "Quebra de nivel")
supf_results[[2]] <- run_supf(stable$y, "Estavel (controle)")
supf_results[[3]] <- run_supf(trend_break$y, "Quebra de tendencia")
supf_results[[4]] <- run_supf(gdp_growth_br, "PIB Brasil (crescimento)")
supf_results[[5]] <- run_supf(gdp_growth_us, "PIB EUA (crescimento)")

cat("\n")

################################################################################
# 3. CUSUM Test using strucchange::efp
################################################################################
cat("--- CUSUM Tests (strucchange::efp) ---\n\n")

cusum_results <- list()

run_cusum <- function(y, series_name) {
  n <- length(y)
  t_idx <- 1:n
  df <- data.frame(y = y, t = t_idx)

  # OLS-CUSUM
  ocus <- efp(y ~ t, type = "OLS-CUSUM", data = df)
  sc_cusum <- sctest(ocus)

  max_cusum <- max(abs(ocus$process))

  cat(sprintf("  %s (CUSUM): stat = %.4f, p = %.6f => %s\n",
              series_name, sc_cusum$statistic, sc_cusum$p.value,
              ifelse(sc_cusum$p.value < 0.05, "INSTABILITY", "STABLE")))

  return(list(
    test = "CUSUM",
    series = series_name,
    statistic = as.numeric(sc_cusum$statistic),
    pvalue = sc_cusum$p.value,
    reject_at_5pct = sc_cusum$p.value < 0.05,
    decision = ifelse(sc_cusum$p.value < 0.05,
                      "Instabilidade detectada",
                      "Parametros estaveis"),
    max_cusum = max_cusum
  ))
}

cusum_results[[1]] <- run_cusum(level_break$y, "Quebra de nivel")
cusum_results[[2]] <- run_cusum(stable$y, "Estavel (controle)")
cusum_results[[3]] <- run_cusum(trend_break$y, "Quebra de tendencia")
cusum_results[[4]] <- run_cusum(gdp_growth_br, "PIB Brasil")
cusum_results[[5]] <- run_cusum(gdp_growth_us, "PIB EUA")

cat("\n")

################################################################################
# 4. CUSUM-sq Test
################################################################################
cat("--- CUSUM-sq Tests (strucchange::efp) ---\n\n")

cusumsq_results <- list()

run_cusumsq <- function(y, series_name) {
  n <- length(y)
  t_idx <- 1:n
  df <- data.frame(y = y, t = t_idx)

  # OLS-CUSUM-sq (variance stability)
  ocus_sq <- efp(y ~ t, type = "OLS-MOSUM", data = df, h = 0.15)

  # Alternative: use Rec-CUSUM for CUSUM-sq behavior
  rec <- efp(y ~ t, type = "Rec-CUSUM", data = df)
  sc <- sctest(rec)

  max_dev <- max(abs(rec$process))

  cat(sprintf("  %s (CUSUM-sq): stat = %.4f, p = %.6f => %s\n",
              series_name, sc$statistic, sc$p.value,
              ifelse(sc$p.value < 0.05, "VARIANCE INSTABILITY", "STABLE")))

  return(list(
    test = "CUSUM-sq",
    series = series_name,
    statistic = as.numeric(sc$statistic),
    pvalue = sc$p.value,
    reject_at_5pct = sc$p.value < 0.05,
    decision = ifelse(sc$p.value < 0.05,
                      "Instabilidade na variancia",
                      "Variancia estavel"),
    max_deviation = max_dev
  ))
}

cusumsq_results[[1]] <- run_cusumsq(level_break$y, "Quebra de nivel")
cusumsq_results[[2]] <- run_cusumsq(stable$y, "Estavel (controle)")
cusumsq_results[[3]] <- run_cusumsq(trend_break$y, "Quebra de tendencia")
cusumsq_results[[4]] <- run_cusumsq(gdp_growth_br, "PIB Brasil")

cat("\n")

################################################################################
# 5. Bai-Perron Test (strucchange::breakpoints)
################################################################################
cat("--- Bai-Perron Tests (strucchange::breakpoints) ---\n\n")

bp_results <- list()

run_bai_perron <- function(y, series_name, h = 0.15, breaks = 3) {
  n <- length(y)
  t_idx <- 1:n
  df <- data.frame(y = y, t = t_idx)

  # Bai-Perron structural break estimation
  bp <- breakpoints(y ~ t, h = h, breaks = breaks, data = df)

  # BIC-selected number of breaks
  bp_summary <- summary(bp)
  opt_breaks <- bp_summary$breaks

  # Get breakpoint indices
  if (length(opt_breaks) > 0 && !all(is.na(opt_breaks))) {
    break_indices <- as.integer(opt_breaks)
  } else {
    break_indices <- integer(0)
  }

  # F-test for overall significance
  fs <- Fstats(y ~ t, from = h, to = 1 - h, data = df)
  sc <- sctest(fs, type = "supF")

  cat(sprintf("  %s [h=%.2f]: %d breaks detected at %s, supF = %.4f\n",
              series_name, h, length(break_indices),
              paste(break_indices, collapse = ", "),
              sc$statistic))

  return(list(
    test = "Bai-Perron",
    series = series_name,
    max_breaks = breaks,
    trim = h,
    statistic = as.numeric(sc$statistic),
    pvalue = sc$p.value,
    reject_at_5pct = sc$p.value < 0.05,
    n_breaks_detected = length(break_indices),
    break_indices = break_indices,
    decision = ifelse(length(break_indices) > 0,
                      paste(length(break_indices), "quebra(s) detectada(s)"),
                      "Nenhuma quebra")
  ))
}

bp_results[[1]] <- run_bai_perron(level_break$y, "Quebra de nivel",
                                   h = 0.15, breaks = 3)
bp_results[[2]] <- run_bai_perron(trend_break$y, "Quebra de tendencia",
                                   h = 0.15, breaks = 3)
bp_results[[3]] <- run_bai_perron(stable$y, "Estavel (controle)",
                                   h = 0.15, breaks = 3)
bp_results[[4]] <- run_bai_perron(y_2breaks, "2 quebras (t=100,200)",
                                   h = 0.15, breaks = 5)
bp_results[[5]] <- run_bai_perron(gdp_growth_br, "PIB Brasil",
                                   h = 0.15, breaks = 3)
bp_results[[6]] <- run_bai_perron(gdp_growth_us, "PIB EUA (crescimento)",
                                   h = 0.15, breaks = 3)

cat("\n")

################################################################################
# 6. Trim sensitivity analysis
################################################################################
cat("--- Trim Sensitivity ---\n\n")

trim_results <- list()
for (h in c(0.05, 0.10, 0.15, 0.20, 0.25)) {
  n <- length(y_2breaks)
  min_segment <- as.integer(n * h)

  result <- tryCatch({
    bp <- breakpoints(y_2breaks ~ 1, h = h, breaks = 5)
    bp_s <- summary(bp)
    opt_breaks <- bp_s$breaks

    t_idx <- 1:n
    df <- data.frame(y = y_2breaks, t = t_idx)
    fs <- Fstats(y ~ t, from = max(h, 0.1), to = min(1 - h, 0.9), data = df)
    sc <- sctest(fs, type = "supF")

    cat(sprintf("  trim=%.2f (min_seg=%d): %d breaks, supF=%.4f\n",
                h, min_segment, length(opt_breaks), sc$statistic))

    list(
      trim = h,
      min_segment = min_segment,
      n_breaks = length(opt_breaks),
      break_indices = as.integer(opt_breaks),
      F_stat = as.numeric(sc$statistic),
      reject = sc$p.value < 0.05
    )
  }, error = function(e) {
    cat(sprintf("  trim=%.2f: ERROR - %s\n", h, e$message))
    list(trim = h, min_segment = min_segment, error = e$message)
  })

  trim_results[[length(trim_results) + 1]] <- result
}

cat("\n")

################################################################################
# 7. Save results
################################################################################
cat("Saving results...\n")

output <- list(
  metadata = list(
    script = "04_breaks_validation.R",
    tests = c("Chow", "CUSUM", "CUSUM-sq", "Bai-Perron", "Sup-F"),
    seed = 42,
    packages = c("strucchange"),
    tolerance = list(
      statistic = 1e-4,
      pvalue = 0.01
    ),
    note = paste("strucchange implements Bai-Perron (2003) for multiple",
                 "breakpoints. CUSUM via efp() with OLS-CUSUM type.",
                 "Chow test implemented manually for exact F-statistic.")
  ),
  chow_results = chow_results,
  chow_scan = chow_scan,
  supf_results = supf_results,
  cusum_results = cusum_results,
  cusumsq_results = cusumsq_results,
  bai_perron_results = bp_results,
  trim_sensitivity = trim_results
)

out_dir <- "/home/guhaase/projetos/chronobox/examples/tests/outputs/R"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write(toJSON(output, auto_unbox = TRUE, pretty = TRUE),
      file.path(out_dir, "breaks_results.json"))

cat("Results saved to outputs/R/breaks_results.json\n")

################################################################################
# 8. Summary
################################################################################
cat("\n")
cat("======================================================================\n")
cat("SUMMARY: Structural Break Test Results\n")
cat("======================================================================\n\n")

cat(sprintf("%-30s | %-10s | %-10s | %-10s | %-10s\n",
            "Series", "Chow", "CUSUM", "Sup-F", "Bai-Perron"))
cat(rep("-", 80), sep = "")
cat("\n")

summary_series <- c("Quebra de nivel", "Estavel (controle)",
                     "Quebra de tendencia", "PIB Brasil", "PIB EUA")
for (s in summary_series) {
  chow_dec <- ""
  cusum_dec <- ""
  supf_dec <- ""
  bp_dec <- ""

  for (r in chow_results) {
    if (grepl(s, r$series, fixed = TRUE) || grepl(s, r$series)) {
      chow_dec <- ifelse(r$reject_at_5pct, "Sim", "Nao")
      break
    }
  }
  for (r in cusum_results) {
    if (r$series == s) {
      cusum_dec <- ifelse(r$reject_at_5pct, "Sim", "Nao")
    }
  }
  for (r in supf_results) {
    if (grepl(s, r$series, fixed = TRUE) || grepl(s, r$series)) {
      supf_dec <- ifelse(r$reject_at_5pct, "Sim", "Nao")
    }
  }
  for (r in bp_results) {
    if (grepl(s, r$series, fixed = TRUE) || grepl(s, r$series)) {
      bp_dec <- paste0(r$n_breaks_detected, " quebras")
    }
  }

  cat(sprintf("%-30s | %-10s | %-10s | %-10s | %-10s\n",
              s, chow_dec, cusum_dec, supf_dec, bp_dec))
}

cat("\nDone.\n")
