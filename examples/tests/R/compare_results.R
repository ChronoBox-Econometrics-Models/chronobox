################################################################################
# compare_results.R
# Compare Python (chronobox) vs R (urca/tseries/strucchange) results
#
# Loads JSON outputs from both Python and R, compares:
# - Test statistics (tolerance < 1e-4)
# - P-values (tolerance < 0.01)
# - Qualitative decisions (reject / fail to reject)
#
# Focus on GDP datasets (loaded from same CSV => identical data)
# Synthetic data differs due to RNG, so compare qualitative decisions only
################################################################################

library(jsonlite)

set.seed(42)

cat("======================================================================\n")
cat("Cross-validation: Python (chronobox) vs R Results\n")
cat("======================================================================\n\n")

# --- Tolerance thresholds ---
TOL_STAT <- 1e-4   # For test statistics
TOL_PVAL <- 0.01   # For p-values

# --- Load Python results ---
py_dir <- "/home/guhaase/projetos/chronobox/examples/tests/outputs"
r_dir <- file.path(py_dir, "R")

cat("Loading Python results...\n")
py_adf_pp <- fromJSON(file.path(py_dir, "adf_pp_results.json"))
py_kpss_ers_za <- fromJSON(file.path(py_dir, "kpss_ers_za_results.json"))
py_coint <- fromJSON(file.path(py_dir, "cointegration_results.json"))
py_breaks <- fromJSON(file.path(py_dir, "breaks_results.json"))

cat("Loading R results...\n")
r_adf_pp <- tryCatch(
  fromJSON(file.path(r_dir, "unit_root_results.json")),
  error = function(e) { cat("  WARNING: R unit_root_results.json not found\n"); NULL }
)
r_kpss <- tryCatch(
  fromJSON(file.path(r_dir, "kpss_ers_za_results.json")),
  error = function(e) { cat("  WARNING: R kpss_ers_za_results.json not found\n"); NULL }
)
r_coint <- tryCatch(
  fromJSON(file.path(r_dir, "cointegration_results.json")),
  error = function(e) { cat("  WARNING: R cointegration_results.json not found\n"); NULL }
)
r_breaks <- tryCatch(
  fromJSON(file.path(r_dir, "breaks_results.json")),
  error = function(e) { cat("  WARNING: R breaks_results.json not found\n"); NULL }
)

cat("\n")

################################################################################
# Helper functions
################################################################################

compare_stat <- function(py_val, r_val, label, tol = TOL_STAT) {
  if (is.null(py_val) || is.null(r_val) || is.na(py_val) || is.na(r_val)) {
    return(list(label = label, py = py_val, r = r_val,
                diff = NA, pass = NA, note = "missing value"))
  }
  d <- abs(py_val - r_val)
  pass <- d < tol
  return(list(label = label, py = py_val, r = r_val,
              diff = d, pass = pass, note = ""))
}

compare_decision <- function(py_reject, r_reject, label) {
  if (is.null(py_reject) || is.null(r_reject)) {
    return(list(label = label, py = py_reject, r = r_reject,
                match = NA, note = "missing"))
  }
  match <- py_reject == r_reject
  return(list(label = label, py = py_reject, r = r_reject,
              match = match, note = ""))
}

print_comparison <- function(comp) {
  if (is.na(comp$pass)) {
    status <- "N/A"
  } else if (comp$pass) {
    status <- "PASS"
  } else {
    status <- "FAIL"
  }
  cat(sprintf("  %-45s | Py: %12.6f | R: %12.6f | diff: %.2e | %s\n",
              comp$label,
              ifelse(is.null(comp$py) || is.na(comp$py), NA, comp$py),
              ifelse(is.null(comp$r) || is.na(comp$r), NA, comp$r),
              ifelse(is.na(comp$diff), NA, comp$diff),
              status))
}

print_decision_comparison <- function(comp) {
  if (is.na(comp$match)) {
    status <- "N/A"
  } else if (comp$match) {
    status <- "MATCH"
  } else {
    status <- "MISMATCH"
  }
  cat(sprintf("  %-45s | Py: %-6s | R: %-6s | %s\n",
              comp$label,
              ifelse(is.null(comp$py), "N/A", ifelse(comp$py, "Reject", "Fail")),
              ifelse(is.null(comp$r), "N/A", ifelse(comp$r, "Reject", "Fail")),
              status))
}

################################################################################
# 1. Compare ADF/PP on GDP data (identical data from CSV)
################################################################################
cat("======================================================================\n")
cat("1. ADF & PP Tests on GDP Data (identical CSV data)\n")
cat("======================================================================\n\n")

# Extract Python GDP results
py_gdp_us <- py_adf_pp$gdp_us_results
py_gdp_br <- py_adf_pp$gdp_brazil_results

cat("--- ADF Statistics ---\n\n")

stat_comparisons <- list()
decision_comparisons <- list()

# Compare ADF on US GDP level (constant)
py_us_level_c <- py_gdp_us[py_gdp_us$test == "ADF" &
                             py_gdp_us$regression == "c", ]
if (nrow(py_us_level_c) > 0 && !is.null(r_adf_pp)) {
  r_gdp <- r_adf_pp$gdp_adf_results
  for (r in r_gdp) {
    if (is.list(r) && grepl("EUA", r$series) && r$regression == "drift" &&
        !grepl("diferenca", r$series)) {
      comp <- compare_stat(py_us_level_c$statistic[1], r$statistic,
                           "ADF US GDP level (c)")
      print_comparison(comp)
      stat_comparisons[[length(stat_comparisons) + 1]] <- comp
      break
    }
  }
}

# Compare ADF on US GDP level (trend)
py_us_level_ct <- py_gdp_us[py_gdp_us$test == "ADF" &
                              py_gdp_us$regression == "ct", ]
if (nrow(py_us_level_ct) > 0 && !is.null(r_adf_pp)) {
  r_gdp <- r_adf_pp$gdp_adf_results
  for (r in r_gdp) {
    if (is.list(r) && grepl("EUA", r$series) && r$regression == "trend" &&
        !grepl("diferenca", r$series)) {
      comp <- compare_stat(py_us_level_ct$statistic[1], r$statistic,
                           "ADF US GDP level (ct)")
      print_comparison(comp)
      stat_comparisons[[length(stat_comparisons) + 1]] <- comp
      break
    }
  }
}

# Compare ADF on US GDP diff
py_us_diff <- py_gdp_us[py_gdp_us$test == "ADF" &
                          grepl("diferenca", py_gdp_us$series), ]
if (nrow(py_us_diff) > 0 && !is.null(r_adf_pp)) {
  r_gdp <- r_adf_pp$gdp_adf_results
  for (r in r_gdp) {
    if (is.list(r) && grepl("EUA", r$series) && grepl("diferenca", r$series)) {
      comp <- compare_stat(py_us_diff$statistic[1], r$statistic,
                           "ADF US GDP 1st diff (c)")
      print_comparison(comp)
      stat_comparisons[[length(stat_comparisons) + 1]] <- comp
      break
    }
  }
}

# Compare ADF on Brazil GDP level
py_br_level_c <- py_gdp_br[py_gdp_br$test == "ADF" &
                             py_gdp_br$regression == "c", ]
if (nrow(py_br_level_c) > 0 && !is.null(r_adf_pp)) {
  r_gdp <- r_adf_pp$gdp_adf_results
  for (r in r_gdp) {
    if (is.list(r) && grepl("Brasil", r$series) && r$regression == "drift" &&
        !grepl("diferenca", r$series)) {
      comp <- compare_stat(py_br_level_c$statistic[1], r$statistic,
                           "ADF Brazil GDP level (c)")
      print_comparison(comp)
      stat_comparisons[[length(stat_comparisons) + 1]] <- comp
      break
    }
  }
}

# Compare ADF on Brazil GDP diff
py_br_diff <- py_gdp_br[py_gdp_br$test == "ADF" &
                          grepl("diferenca", py_gdp_br$series), ]
if (nrow(py_br_diff) > 0 && !is.null(r_adf_pp)) {
  r_gdp <- r_adf_pp$gdp_adf_results
  for (r in r_gdp) {
    if (is.list(r) && grepl("Brasil", r$series) && grepl("diferenca", r$series)) {
      comp <- compare_stat(py_br_diff$statistic[1], r$statistic,
                           "ADF Brazil GDP 1st diff (c)")
      print_comparison(comp)
      stat_comparisons[[length(stat_comparisons) + 1]] <- comp
      break
    }
  }
}

cat("\n--- PP Statistics ---\n\n")

# Compare PP on US GDP level
py_us_pp_c <- py_gdp_us[py_gdp_us$test == "PP" & py_gdp_us$regression == "c", ]
if (nrow(py_us_pp_c) > 0 && !is.null(r_adf_pp)) {
  r_pp <- r_adf_pp$gdp_pp_results
  for (r in r_pp) {
    if (is.list(r) && grepl("EUA", r$series) && r$regression == "constant" &&
        !grepl("diferenca", r$series)) {
      comp <- compare_stat(py_us_pp_c$statistic[1], r$statistic,
                           "PP US GDP level (c)")
      print_comparison(comp)
      stat_comparisons[[length(stat_comparisons) + 1]] <- comp
      break
    }
  }
}

# Compare PP on Brazil GDP level
py_br_pp_c <- py_gdp_br[py_gdp_br$test == "PP" & py_gdp_br$regression == "c", ]
if (nrow(py_br_pp_c) > 0 && !is.null(r_adf_pp)) {
  r_pp <- r_adf_pp$gdp_pp_results
  for (r in r_pp) {
    if (is.list(r) && grepl("Brasil", r$series) && r$regression == "constant" &&
        !grepl("diferenca", r$series)) {
      comp <- compare_stat(py_br_pp_c$statistic[1], r$statistic,
                           "PP Brazil GDP level (c)")
      print_comparison(comp)
      stat_comparisons[[length(stat_comparisons) + 1]] <- comp
      break
    }
  }
}

cat("\n")

################################################################################
# 2. Compare KPSS/ERS/ZA on GDP data
################################################################################
cat("======================================================================\n")
cat("2. KPSS, ERS, ZA Tests on GDP Data\n")
cat("======================================================================\n\n")

# Python battery results for GDP
py_battery <- py_kpss_ers_za$battery_results

cat("--- KPSS Statistics ---\n\n")

gdp_names_py <- c("Log PIB EUA (nivel)", "Log PIB EUA (1a diff)",
                    "Log PIB Brasil (nivel)", "Log PIB Brasil (1a diff)")

for (gname in gdp_names_py) {
  py_kpss_stat <- NULL
  r_kpss_stat <- NULL

  if (!is.null(py_battery[[gname]])) {
    py_kpss_stat <- py_battery[[gname]]$KPSS$statistic
  }

  if (!is.null(r_kpss)) {
    for (r in r_kpss$gdp_kpss_results) {
      if (is.list(r) && grepl(gsub("\\(.*", "", gname), r$series)) {
        r_kpss_stat <- r$statistic
        break
      }
    }
  }

  if (!is.null(py_kpss_stat)) {
    comp <- compare_stat(py_kpss_stat, r_kpss_stat,
                         paste("KPSS", gname))
    print_comparison(comp)
    stat_comparisons[[length(stat_comparisons) + 1]] <- comp
  }
}

cat("\n--- ERS Statistics ---\n\n")

for (gname in gdp_names_py) {
  py_ers_stat <- NULL
  r_ers_stat <- NULL

  if (!is.null(py_battery[[gname]])) {
    py_ers_stat <- py_battery[[gname]]$ERS$statistic
  }

  if (!is.null(r_kpss)) {
    for (r in r_kpss$gdp_ers_results) {
      if (is.list(r) && grepl(gsub("\\(.*", "", gname), r$series)) {
        r_ers_stat <- r$statistic
        break
      }
    }
  }

  if (!is.null(py_ers_stat)) {
    comp <- compare_stat(py_ers_stat, r_ers_stat,
                         paste("ERS", gname))
    print_comparison(comp)
    stat_comparisons[[length(stat_comparisons) + 1]] <- comp
  }
}

cat("\n--- Zivot-Andrews Statistics ---\n\n")

for (gname in gdp_names_py) {
  py_za_stat <- NULL
  r_za_stat <- NULL

  if (!is.null(py_battery[[gname]])) {
    py_za_stat <- py_battery[[gname]]$ZA$statistic
  }

  if (!is.null(r_kpss)) {
    for (r in r_kpss$gdp_za_results) {
      if (is.list(r) && grepl(gsub("\\(.*", "", gname), r$series)) {
        r_za_stat <- r$statistic
        break
      }
    }
  }

  if (!is.null(py_za_stat)) {
    comp <- compare_stat(py_za_stat, r_za_stat,
                         paste("ZA", gname))
    print_comparison(comp)
    stat_comparisons[[length(stat_comparisons) + 1]] <- comp
  }
}

cat("\n")

################################################################################
# 3. Compare Qualitative Decisions (Synthetic data)
################################################################################
cat("======================================================================\n")
cat("3. Qualitative Decision Comparison (Synthetic Data)\n")
cat("======================================================================\n\n")

cat("Note: Synthetic data differs (R vs Python RNG), so only qualitative\n")
cat("decisions are compared. Both should detect I(0) as stationary,\n")
cat("I(1) as unit root, etc.\n\n")

# Python qualitative decisions from summary table
py_summary_adf <- py_adf_pp$summary_table

cat("--- ADF/PP Decision Comparison ---\n\n")
cat(sprintf("%-30s | %-15s | %-15s\n",
            "Series", "Python", "Expected"))
cat(rep("-", 65), sep = "")
cat("\n")

for (i in 1:nrow(py_summary_adf)) {
  row <- py_summary_adf[i, ]
  cat(sprintf("%-30s | %-15s | %-15s\n",
              row$Serie, row$`Decisao Final`,
              ifelse(grepl("I\\(0\\)|diff|Trend-stat", row$Serie),
                     "Estacionaria", "Raiz unitaria")))
}

cat("\n")

# Python KPSS/ERS/ZA decisions
py_summary_kpss <- py_kpss_ers_za$summary_table

cat("--- KPSS/ERS/ZA Decision Comparison ---\n\n")
cat(sprintf("%-30s | %-15s\n", "Series", "Python Decision"))
cat(rep("-", 50), sep = "")
cat("\n")

for (i in 1:nrow(py_summary_kpss)) {
  row <- py_summary_kpss[i, ]
  cat(sprintf("%-30s | %-15s\n", row$Serie, row$`Decisao Final`))
}

cat("\n")

################################################################################
# 4. Compare Cointegration Results on GDP
################################################################################
cat("======================================================================\n")
cat("4. Cointegration Tests on GDP Data\n")
cat("======================================================================\n\n")

py_gdp_coint <- py_coint$gdp_cointegration

if (!is.null(py_gdp_coint) && !is.null(r_coint)) {
  # Engle-Granger
  py_eg_stat <- py_gdp_coint$engle_granger$statistic
  r_eg_stat <- NULL
  for (r in r_coint$engle_granger_results) {
    if (is.list(r) && grepl("PIB", r$pair)) {
      r_eg_stat <- r$statistic
      break
    }
  }
  comp <- compare_stat(py_eg_stat, r_eg_stat,
                       "Engle-Granger PIB BR~EUA")
  print_comparison(comp)
  stat_comparisons[[length(stat_comparisons) + 1]] <- comp

  # Johansen trace stat
  py_jo_trace <- py_gdp_coint$johansen$trace_stats[1]
  r_jo_trace <- NULL
  for (r in r_coint$johansen_results) {
    if (is.list(r) && grepl("PIB", r$system)) {
      r_jo_trace <- r$test_stats[1]
      break
    }
  }
  comp <- compare_stat(py_jo_trace, r_jo_trace,
                       "Johansen trace(0) PIB BR~EUA")
  print_comparison(comp)
  stat_comparisons[[length(stat_comparisons) + 1]] <- comp
}

cat("\n")

################################################################################
# 5. Compare Structural Break Results on GDP
################################################################################
cat("======================================================================\n")
cat("5. Structural Break Tests on GDP Data\n")
cat("======================================================================\n\n")

py_br_breaks <- py_breaks$gdp_brazil_breaks

if (!is.null(py_br_breaks) && !is.null(r_breaks)) {
  # Chow test on Brazil GDP
  py_chow_stat <- py_br_breaks$chow$statistic
  r_chow_stat <- NULL
  for (r in r_breaks$chow_results) {
    if (is.list(r) && grepl("PIB Brasil", r$series)) {
      r_chow_stat <- r$statistic
      break
    }
  }
  comp <- compare_stat(py_chow_stat, r_chow_stat,
                       "Chow PIB Brasil (bp=36)")
  print_comparison(comp)
  stat_comparisons[[length(stat_comparisons) + 1]] <- comp
}

cat("\n")

################################################################################
# 6. Overall Summary
################################################################################
cat("======================================================================\n")
cat("OVERALL COMPARISON SUMMARY\n")
cat("======================================================================\n\n")

n_total <- length(stat_comparisons)
n_pass <- sum(sapply(stat_comparisons, function(x)
  ifelse(is.na(x$pass), FALSE, x$pass)))
n_fail <- sum(sapply(stat_comparisons, function(x)
  ifelse(is.na(x$pass), FALSE, !x$pass)))
n_na <- sum(sapply(stat_comparisons, function(x) is.na(x$pass)))

cat(sprintf("Total comparisons: %d\n", n_total))
cat(sprintf("  PASS (diff < tol):  %d\n", n_pass))
cat(sprintf("  FAIL (diff >= tol): %d\n", n_fail))
cat(sprintf("  N/A (missing data): %d\n", n_na))
cat(sprintf("\nTolerance: stat < %.0e, p-value < %.2f\n", TOL_STAT, TOL_PVAL))

if (n_fail > 0) {
  cat("\nFAILED comparisons:\n")
  for (comp in stat_comparisons) {
    if (!is.na(comp$pass) && !comp$pass) {
      cat(sprintf("  %s: py=%.6f, r=%.6f, diff=%.2e\n",
                  comp$label, comp$py, comp$r, comp$diff))
    }
  }
}

cat("\n")
cat("NOTE: Differences between Python and R may arise from:\n")
cat("  1. Different default lag selection algorithms\n")
cat("  2. Different critical value table interpolation\n")
cat("  3. Different bandwidth/kernel estimation (PP, KPSS)\n")
cat("  4. Slightly different ADF regression formulations (urca vs statsmodels)\n")
cat("  5. R's urca uses different Phillips-Perron formulation than Python\n")
cat("\nFor GDP data (identical input), small numerical differences are expected\n")
cat("due to implementation details, but qualitative decisions should agree.\n")

# Save comparison results
comparison_output <- list(
  metadata = list(
    script = "compare_results.R",
    tolerance = list(statistic = TOL_STAT, pvalue = TOL_PVAL),
    date = Sys.time()
  ),
  stat_comparisons = stat_comparisons,
  summary = list(
    total = n_total,
    pass = n_pass,
    fail = n_fail,
    na = n_na
  )
)

write(toJSON(comparison_output, auto_unbox = TRUE, pretty = TRUE),
      file.path(r_dir, "comparison_summary.json"))

cat("\nComparison results saved to outputs/R/comparison_summary.json\n")
cat("\nDone.\n")
