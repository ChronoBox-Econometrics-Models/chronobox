##############################################################################
# compare_results.R
#
# Compara resultados dos filtros estimados pelo chronobox (Python) com os
# resultados de referencia do R (mFilter, neverhpfilter, vars).
#
# Tolerancias esperadas:
#   HP filter:   < 1e-6 (solucao exata, mesmo sistema linear tridiagonal)
#   BK filter:   < 1e-4 (diferenças de truncamento nas bordas)
#   CF filter:   < 1e-4 (diferenças no metodo de estimacao espectral)
#   Hamilton:    < 1e-6 (regressao OLS, mesma formulacao)
#   BN:          < 1e-3 (depende da implementacao do AR e da soma infinita)
#   Spillover:   < 1e-2 (diferenças de FEVD, normalizacao, etc.)
#
# Inputs:
#   Python: examples/filters/outputs/filter_cycles.csv
#           examples/filters/outputs/hamilton_bn_results.csv
#           examples/filters/outputs/spillover_table.json
#           examples/filters/outputs/rolling_spillover.csv
#   R:      examples/filters/outputs/R/hp_bk_cf_cycles.csv
#           examples/filters/outputs/R/hamilton_bn_results.csv
#           examples/filters/outputs/R/spillover_summary.csv
#           examples/filters/outputs/R/rolling_spillover.csv
#
# Output: examples/filters/outputs/R/comparison_report.csv
##############################################################################

set.seed(42)

# --- Caminhos ---
base_dir <- file.path(dirname(sys.frame(1)$ofile), "..")
py_output_dir <- file.path(base_dir, "outputs")
r_output_dir <- file.path(base_dir, "outputs", "R")

cat("=============================================================\n")
cat("  Comparacao Python (chronobox) vs R (mFilter/neverhpfilter)\n")
cat("=============================================================\n\n")

# Acumulador de resultados da comparacao
comparison <- data.frame(
  filter = character(),
  country = character(),
  metric = character(),
  python_value = numeric(),
  r_value = numeric(),
  abs_diff = numeric(),
  tolerance = numeric(),
  pass = logical(),
  stringsAsFactors = FALSE
)

add_row <- function(filter, country, metric, py_val, r_val, tol) {
  diff_val <- abs(py_val - r_val)
  new_row <- data.frame(
    filter = filter,
    country = country,
    metric = metric,
    python_value = py_val,
    r_value = r_val,
    abs_diff = diff_val,
    tolerance = tol,
    pass = diff_val < tol,
    stringsAsFactors = FALSE
  )
  return(new_row)
}

# ===================================================================
# 1. Comparacao HP / BK / CF
# ===================================================================
cat("--- 1. HP / BK / CF filters ---\n")

py_cycles_file <- file.path(py_output_dir, "filter_cycles.csv")
r_cycles_file <- file.path(r_output_dir, "hp_bk_cf_cycles.csv")

if (file.exists(py_cycles_file) && file.exists(r_cycles_file)) {
  py_cycles <- read.csv(py_cycles_file, stringsAsFactors = FALSE)
  r_cycles <- read.csv(r_cycles_file, stringsAsFactors = FALSE)

  for (ctry in unique(py_cycles$country)) {
    py_sub <- py_cycles[py_cycles$country == ctry, ]
    r_sub <- r_cycles[r_cycles$country == ctry, ]

    # Alinhar por data
    common_dates <- intersect(py_sub$date, r_sub$date)
    if (length(common_dates) == 0) {
      cat(sprintf("  %s: nenhuma data em comum, pulando\n", ctry))
      next
    }

    py_aligned <- py_sub[py_sub$date %in% common_dates, ]
    r_aligned <- r_sub[r_sub$date %in% common_dates, ]
    py_aligned <- py_aligned[order(py_aligned$date), ]
    r_aligned <- r_aligned[order(r_aligned$date), ]

    n_obs <- nrow(py_aligned)
    cat(sprintf("  %s: %d observacoes em comum\n", ctry, n_obs))

    # --- HP cycle ---
    # Tolerancia: < 1e-6 (solucao exata do mesmo sistema linear)
    hp_mae <- mean(abs(py_aligned$hp_cycle - r_aligned$hp_cycle), na.rm = TRUE)
    hp_max_diff <- max(abs(py_aligned$hp_cycle - r_aligned$hp_cycle), na.rm = TRUE)
    hp_corr <- cor(py_aligned$hp_cycle, r_aligned$hp_cycle, use = "complete.obs")

    cat(sprintf("    HP: MAE = %.2e, max_diff = %.2e, corr = %.8f\n",
                hp_mae, hp_max_diff, hp_corr))

    comparison <- rbind(comparison,
      add_row("HP", ctry, "MAE", hp_mae, 0, 1e-6),
      add_row("HP", ctry, "max_abs_diff", hp_max_diff, 0, 1e-6),
      add_row("HP", ctry, "correlation", hp_corr, 1.0, 1e-6)
    )

    # --- BK cycle ---
    # Tolerancia: < 1e-4 (truncamento nas bordas difere entre implementacoes)
    valid_bk <- !is.na(py_aligned$bk_cycle) & !is.na(r_aligned$bk_cycle)
    if (sum(valid_bk) > 0) {
      bk_mae <- mean(abs(py_aligned$bk_cycle[valid_bk] - r_aligned$bk_cycle[valid_bk]))
      bk_max_diff <- max(abs(py_aligned$bk_cycle[valid_bk] - r_aligned$bk_cycle[valid_bk]))
      bk_corr <- cor(py_aligned$bk_cycle[valid_bk], r_aligned$bk_cycle[valid_bk])

      cat(sprintf("    BK: MAE = %.2e, max_diff = %.2e, corr = %.8f (n_valid = %d)\n",
                  bk_mae, bk_max_diff, bk_corr, sum(valid_bk)))

      comparison <- rbind(comparison,
        add_row("BK", ctry, "MAE", bk_mae, 0, 1e-4),
        add_row("BK", ctry, "max_abs_diff", bk_max_diff, 0, 1e-4),
        add_row("BK", ctry, "correlation", bk_corr, 1.0, 1e-4)
      )
    } else {
      cat(sprintf("    BK: sem observacoes validas para comparacao\n"))
    }

    # --- CF cycle ---
    # Tolerancia: < 1e-4 (estimacao espectral pode diferir)
    valid_cf <- !is.na(py_aligned$cf_cycle) & !is.na(r_aligned$cf_cycle)
    if (sum(valid_cf) > 0) {
      cf_mae <- mean(abs(py_aligned$cf_cycle[valid_cf] - r_aligned$cf_cycle[valid_cf]))
      cf_max_diff <- max(abs(py_aligned$cf_cycle[valid_cf] - r_aligned$cf_cycle[valid_cf]))
      cf_corr <- cor(py_aligned$cf_cycle[valid_cf], r_aligned$cf_cycle[valid_cf])

      cat(sprintf("    CF: MAE = %.2e, max_diff = %.2e, corr = %.8f (n_valid = %d)\n",
                  cf_mae, cf_max_diff, cf_corr, sum(valid_cf)))

      comparison <- rbind(comparison,
        add_row("CF", ctry, "MAE", cf_mae, 0, 1e-4),
        add_row("CF", ctry, "max_abs_diff", cf_max_diff, 0, 1e-4),
        add_row("CF", ctry, "correlation", cf_corr, 1.0, 1e-4)
      )
    } else {
      cat(sprintf("    CF: sem observacoes validas para comparacao\n"))
    }
  }
} else {
  cat("  AVISO: arquivos de ciclos nao encontrados, pulando comparacao HP/BK/CF\n")
  if (!file.exists(py_cycles_file)) cat(sprintf("    Faltando: %s\n", py_cycles_file))
  if (!file.exists(r_cycles_file)) cat(sprintf("    Faltando: %s\n", r_cycles_file))
}

# ===================================================================
# 2. Comparacao Hamilton filter
# ===================================================================
cat("\n--- 2. Hamilton filter ---\n")

py_hamilton_file <- file.path(py_output_dir, "hamilton_bn_results.csv")
r_hamilton_file <- file.path(r_output_dir, "hamilton_bn_results.csv")

if (file.exists(py_hamilton_file) && file.exists(r_hamilton_file)) {
  py_ham <- read.csv(py_hamilton_file, stringsAsFactors = FALSE)
  r_ham <- read.csv(r_hamilton_file, stringsAsFactors = FALSE)

  # Comparar por metodo e pais
  for (method in unique(py_ham$method)) {
    for (ctry in unique(py_ham$country)) {
      py_sub <- py_ham[py_ham$method == method & py_ham$country == ctry, ]
      r_sub <- r_ham[r_ham$method == method & r_ham$country == ctry, ]

      if (nrow(py_sub) == 0 || nrow(r_sub) == 0) next

      common_dates <- intersect(py_sub$date, r_sub$date)
      if (length(common_dates) == 0) next

      py_aligned <- py_sub[py_sub$date %in% common_dates, ]
      r_aligned <- r_sub[r_sub$date %in% common_dates, ]
      py_aligned <- py_aligned[order(py_aligned$date), ]
      r_aligned <- r_aligned[order(r_aligned$date), ]

      valid <- !is.na(py_aligned$cycle) & !is.na(r_aligned$cycle)
      if (sum(valid) == 0) next

      cycle_mae <- mean(abs(py_aligned$cycle[valid] - r_aligned$cycle[valid]))
      cycle_max <- max(abs(py_aligned$cycle[valid] - r_aligned$cycle[valid]))
      cycle_corr <- cor(py_aligned$cycle[valid], r_aligned$cycle[valid])

      # Hamilton: tolerancia depende se eh hamilton ou BN
      if (grepl("hamilton", method)) {
        tol <- 1e-6
      } else {
        tol <- 1e-3  # BN tem mais variacao entre implementacoes
      }

      cat(sprintf("  %s (%s): MAE = %.2e, max_diff = %.2e, corr = %.8f (n = %d)\n",
                  method, ctry, cycle_mae, cycle_max, cycle_corr, sum(valid)))

      comparison <- rbind(comparison,
        add_row(method, ctry, "MAE", cycle_mae, 0, tol),
        add_row(method, ctry, "max_abs_diff", cycle_max, 0, tol),
        add_row(method, ctry, "correlation", cycle_corr, 1.0, tol)
      )
    }
  }
} else {
  cat("  AVISO: arquivos Hamilton/BN nao encontrados, pulando comparacao\n")
  if (!file.exists(py_hamilton_file)) cat(sprintf("    Faltando: %s\n", py_hamilton_file))
  if (!file.exists(r_hamilton_file)) cat(sprintf("    Faltando: %s\n", r_hamilton_file))
}

# ===================================================================
# 3. Comparacao Spillover
# ===================================================================
cat("\n--- 3. Spillover index ---\n")

py_spillover_file <- file.path(py_output_dir, "spillover_table.json")
r_spillover_file <- file.path(r_output_dir, "spillover_summary.csv")

if (file.exists(py_spillover_file) && file.exists(r_spillover_file)) {
  # Ler Python spillover (JSON)
  if (require("jsonlite", quietly = TRUE)) {
    py_sp <- fromJSON(py_spillover_file)

    # Extrair total spillover do Python
    py_total <- py_sp$total_spillover
    if (is.null(py_total)) {
      # Tentar estrutura alternativa
      py_total <- py_sp$spillover_table$total_spillover
    }

    # Ler R spillover
    r_sp <- read.csv(r_spillover_file, stringsAsFactors = FALSE)

    if (!is.null(py_total)) {
      cat(sprintf("  Python total spillover: %.4f\n", py_total))

      # Total spillover do R: soma dos from (ou to) ja contem o total
      # from e to somam ao total_spillover
      r_total_from <- sum(r_sp$from)
      r_total_to <- sum(r_sp$to)
      cat(sprintf("  R total spillover (sum from): %.4f\n", r_total_from))
      cat(sprintf("  R total spillover (sum to):   %.4f\n", r_total_to))

      # Nota: spillover entre implementacoes pode diferir mais
      # pois depende da normalizacao da FEVD (Cholesky vs generalizada)
      diff_total <- abs(py_total - r_total_from)
      cat(sprintf("  Diferenca absoluta: %.4f\n", diff_total))

      comparison <- rbind(comparison,
        add_row("Spillover", "Synthetic", "total_spillover_diff",
                py_total, r_total_from, 1e-2)
      )
    }
  } else {
    cat("  AVISO: pacote jsonlite nao disponivel, pulando comparacao spillover\n")
  }
} else {
  cat("  AVISO: arquivos spillover nao encontrados, pulando comparacao\n")
  if (!file.exists(py_spillover_file)) cat(sprintf("    Faltando: %s\n", py_spillover_file))
  if (!file.exists(r_spillover_file)) cat(sprintf("    Faltando: %s\n", r_spillover_file))
}

# ===================================================================
# 4. Comparacao Rolling Spillover
# ===================================================================
cat("\n--- 4. Rolling spillover ---\n")

py_rolling_file <- file.path(py_output_dir, "rolling_spillover.csv")
r_rolling_file <- file.path(r_output_dir, "rolling_spillover.csv")

if (file.exists(py_rolling_file) && file.exists(r_rolling_file)) {
  py_roll <- read.csv(py_rolling_file, stringsAsFactors = FALSE)
  r_roll <- read.csv(r_rolling_file, stringsAsFactors = FALSE)

  common_dates <- intersect(py_roll$date, r_roll$date)
  if (length(common_dates) > 0) {
    py_aligned <- py_roll[py_roll$date %in% common_dates, ]
    r_aligned <- r_roll[r_roll$date %in% common_dates, ]
    py_aligned <- py_aligned[order(py_aligned$date), ]
    r_aligned <- r_aligned[order(r_aligned$date), ]

    valid <- !is.na(py_aligned$total_spillover) & !is.na(r_aligned$total_spillover)
    if (sum(valid) > 0) {
      roll_mae <- mean(abs(py_aligned$total_spillover[valid] -
                           r_aligned$total_spillover[valid]))
      roll_corr <- cor(py_aligned$total_spillover[valid],
                       r_aligned$total_spillover[valid])

      cat(sprintf("  Rolling spillover: MAE = %.4f, corr = %.4f (n = %d)\n",
                  roll_mae, roll_corr, sum(valid)))

      comparison <- rbind(comparison,
        add_row("Rolling_Spillover", "Synthetic", "MAE", roll_mae, 0, 5.0),
        add_row("Rolling_Spillover", "Synthetic", "correlation", roll_corr, 1.0, 0.5)
      )
    }
  } else {
    cat("  Nenhuma data em comum para rolling spillover\n")
  }
} else {
  cat("  AVISO: arquivos rolling spillover nao encontrados\n")
  if (!file.exists(py_rolling_file)) cat(sprintf("    Faltando: %s\n", py_rolling_file))
  if (!file.exists(r_rolling_file)) cat(sprintf("    Faltando: %s\n", r_rolling_file))
}

# ===================================================================
# Relatorio final
# ===================================================================
cat("\n=============================================================\n")
cat("  RELATORIO DE COMPARACAO\n")
cat("=============================================================\n\n")

if (nrow(comparison) > 0) {
  n_tests <- nrow(comparison)
  n_pass <- sum(comparison$pass, na.rm = TRUE)
  n_fail <- sum(!comparison$pass, na.rm = TRUE)
  n_na <- sum(is.na(comparison$pass))

  cat(sprintf("Total de testes: %d\n", n_tests))
  cat(sprintf("  PASS: %d\n", n_pass))
  cat(sprintf("  FAIL: %d\n", n_fail))
  if (n_na > 0) cat(sprintf("  NA:   %d\n", n_na))

  if (n_fail > 0) {
    cat("\nTestes que FALHARAM:\n")
    fails <- comparison[!comparison$pass & !is.na(comparison$pass), ]
    for (i in 1:nrow(fails)) {
      cat(sprintf("  %s (%s) %s: diff = %.2e > tol = %.2e\n",
                  fails$filter[i], fails$country[i], fails$metric[i],
                  fails$abs_diff[i], fails$tolerance[i]))
    }
  }

  cat("\nNotas sobre tolerancias:\n")
  cat("  HP filter:   < 1e-6 (solucao exata, sistema linear tridiagonal)\n")
  cat("  BK filter:   < 1e-4 (truncamento simetrico nas bordas)\n")
  cat("  CF filter:   < 1e-4 (estimacao espectral, metodo pode variar)\n")
  cat("  Hamilton:    < 1e-6 (regressao OLS, formulacao identica)\n")
  cat("  BN:          < 1e-3 (AR companion form, soma infinita truncada)\n")
  cat("  Spillover:   < 1e-2 (FEVD depende de normalizacao Cholesky vs GVD)\n")
  cat("  Rolling:     estrutural (RNGs diferentes entre Python e R)\n")

  # Salvar relatorio
  report_file <- file.path(r_output_dir, "comparison_report.csv")
  write.csv(comparison, report_file, row.names = FALSE)
  cat(sprintf("\nRelatorio salvo em: %s\n", report_file))
} else {
  cat("Nenhum teste de comparacao foi executado.\n")
  cat("Verifique se os scripts Python e R foram executados primeiro.\n")
}

cat("\n=== compare_results.R concluido ===\n")
