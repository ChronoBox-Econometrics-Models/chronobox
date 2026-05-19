# =============================================================================
# compare_results.R
# Comparacao sistematica de resultados Python (chronobox) vs R
# Gera relatorio de validacao cruzada com metricas e tolerancias
# =============================================================================

set.seed(42)

# --- Configuracao de caminhos ---
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
  script_dir <- "."
}
base_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
output_dir_python <- file.path(base_dir, "outputs")
output_dir_r <- file.path(base_dir, "outputs", "R")

# Verificar que os resultados existem
cat("=============================================================================\n")
cat("  RELATORIO DE VALIDACAO CRUZADA: Python (chronobox) vs R\n")
cat("=============================================================================\n\n")

# =============================================================================
# Funcoes auxiliares
# =============================================================================

# Calcula metricas de comparacao entre dois vetores
compare_vectors <- function(r_vec, py_vec, name = "") {
  valid <- !is.na(r_vec) & !is.na(py_vec)
  if (sum(valid) == 0) {
    cat(sprintf("  %s: sem dados validos para comparacao\n", name))
    return(NULL)
  }
  r <- r_vec[valid]
  p <- py_vec[valid]

  mae <- mean(abs(r - p))
  max_diff <- max(abs(r - p))
  rmse <- sqrt(mean((r - p)^2))
  correlation <- cor(r, p)
  n <- sum(valid)

  cat(sprintf("  %s (n=%d):\n", name, n))
  cat(sprintf("    MAE:         %.6e\n", mae))
  cat(sprintf("    Max diff:    %.6e\n", max_diff))
  cat(sprintf("    RMSE:        %.6e\n", rmse))
  cat(sprintf("    Correlacao:  %.8f\n", correlation))

  return(list(mae = mae, max_diff = max_diff, rmse = rmse,
              correlation = correlation, n = n))
}

# Verifica se uma tolerancia e atendida
check_tolerance <- function(metrics, tol, name) {
  if (is.null(metrics)) {
    cat(sprintf("  [SKIP] %s: sem metricas\n", name))
    return(NA)
  }
  passed <- metrics$max_diff < tol
  status <- ifelse(passed, "PASS", "FAIL")
  cat(sprintf("  [%s] %s: max_diff = %.6e (tol = %.6e)\n",
              status, name, metrics$max_diff, tol))
  return(passed)
}

# =============================================================================
# 1. Decomposicao Classica (airline.csv)
# =============================================================================
cat("--- 1. Decomposicao Classica (airline.csv) ---\n\n")

py_classical <- file.path(output_dir_python, "classical_components.csv")
r_classical <- file.path(output_dir_r, "classical_components.csv")

results_classical <- list()

if (file.exists(py_classical) && file.exists(r_classical)) {
  py_df <- read.csv(py_classical, stringsAsFactors = FALSE)
  r_df <- read.csv(r_classical, stringsAsFactors = FALSE)

  cat("Aditiva:\n")
  results_classical$seasonal_add <- compare_vectors(
    r_df$seasonal_additive, as.numeric(py_df$seasonal_additive),
    "Sazonal aditivo")
  results_classical$trend_add <- compare_vectors(
    r_df$trend_additive, as.numeric(py_df$trend_additive),
    "Trend aditivo")
  results_classical$resid_add <- compare_vectors(
    r_df$residual_additive, as.numeric(py_df$residual_additive),
    "Residual aditivo")

  cat("\nMultiplicativa:\n")
  results_classical$seasonal_mult <- compare_vectors(
    r_df$seasonal_multiplicative, as.numeric(py_df$seasonal_multiplicative),
    "Sazonal multiplicativo")
  results_classical$trend_mult <- compare_vectors(
    r_df$trend_multiplicative, as.numeric(py_df$trend_multiplicative),
    "Trend multiplicativo")
  results_classical$resid_mult <- compare_vectors(
    r_df$residual_multiplicative, as.numeric(py_df$residual_multiplicative),
    "Residual multiplicativo")

  # TOLERANCIA para decomposicao classica: < 1e-6
  # Ambas as implementacoes usam o mesmo algoritmo (media movel centrada).
  cat("\nVerificacao de tolerancia (classica: < 1e-6):\n")
  check_tolerance(results_classical$seasonal_add, 1e-6, "Sazonal aditivo")
  check_tolerance(results_classical$trend_add, 1e-6, "Trend aditivo")
  check_tolerance(results_classical$seasonal_mult, 1e-6, "Sazonal multiplicativo")
  check_tolerance(results_classical$trend_mult, 1e-6, "Trend multiplicativo")
} else {
  cat("Arquivos nao encontrados:\n")
  if (!file.exists(py_classical)) cat("  Python:", py_classical, "\n")
  if (!file.exists(r_classical)) cat("  R:", r_classical, "\n")
  cat("Execute os scripts correspondentes primeiro.\n")
}

# =============================================================================
# 2. STL Decomposition (co2.csv)
# =============================================================================
cat("\n--- 2. STL Decomposition (co2.csv) ---\n\n")

py_stl <- file.path(output_dir_python, "stl_components.csv")
r_stl <- file.path(output_dir_r, "stl_components.csv")

results_stl <- list()

if (file.exists(py_stl) && file.exists(r_stl)) {
  py_df <- read.csv(py_stl, stringsAsFactors = FALSE)
  r_df <- read.csv(r_stl, stringsAsFactors = FALSE)

  for (sw in c("s7", "s15", "s35")) {
    cat(sprintf("STL s.window = %s:\n", sw))

    trend_col <- paste0("trend_", sw)
    seasonal_col <- paste0("seasonal_", sw)
    resid_col <- paste0("residual_", sw)

    if (trend_col %in% names(py_df) && trend_col %in% names(r_df)) {
      results_stl[[paste0("trend_", sw)]] <- compare_vectors(
        r_df[[trend_col]], as.numeric(py_df[[trend_col]]),
        paste0("Trend ", sw))
      results_stl[[paste0("seasonal_", sw)]] <- compare_vectors(
        r_df[[seasonal_col]], as.numeric(py_df[[seasonal_col]]),
        paste0("Sazonal ", sw))
      results_stl[[paste0("resid_", sw)]] <- compare_vectors(
        r_df[[resid_col]], as.numeric(py_df[[resid_col]]),
        paste0("Residual ", sw))
    } else {
      cat(sprintf("  Colunas %s nao encontradas em ambos os arquivos\n", sw))
    }
    cat("\n")
  }

  # TOLERANCIA para STL: componentes devem ter correlacao > 0.999
  # Diferencas absolutas podem ser maiores que classica devido a LOESS.
  cat("Verificacao de tolerancia (STL: correlacao > 0.999):\n")
  for (key in names(results_stl)) {
    m <- results_stl[[key]]
    if (!is.null(m)) {
      status <- ifelse(m$correlation > 0.999, "PASS", "FAIL")
      cat(sprintf("  [%s] %s: cor = %.8f\n", status, key, m$correlation))
    }
  }
} else {
  cat("Arquivos nao encontrados:\n")
  if (!file.exists(py_stl)) cat("  Python:", py_stl, "\n")
  if (!file.exists(r_stl)) cat("  R:", r_stl, "\n")
}

# =============================================================================
# 3. X-13 / STL Proxy (brazil_ipca.csv)
# =============================================================================
cat("\n--- 3. X-13 / STL Proxy (brazil_ipca.csv) ---\n\n")

py_x13 <- file.path(output_dir_python, "x13_adjusted.csv")
r_x13 <- file.path(output_dir_r, "x13_adjusted.csv")

results_x13 <- list()

if (file.exists(py_x13) && file.exists(r_x13)) {
  py_df <- read.csv(py_x13, stringsAsFactors = FALSE)
  r_df <- read.csv(r_x13, stringsAsFactors = FALSE)

  # Comparar series dessazonalizadas
  for (col in c("sa_stl_s7", "sa_stl_s13_robust", "sa_stl_s25")) {
    if (col %in% names(py_df) && col %in% names(r_df)) {
      results_x13[[col]] <- compare_vectors(
        r_df[[col]], as.numeric(py_df[[col]]), col)
    }
  }

  # Comparar trends
  for (col in c("trend_stl_s7", "trend_stl_s13_robust", "trend_stl_s25")) {
    if (col %in% names(py_df) && col %in% names(r_df)) {
      results_x13[[col]] <- compare_vectors(
        r_df[[col]], as.numeric(py_df[[col]]), col)
    }
  }

  cat("\nVerificacao de tolerancia (X-13 proxy: correlacao > 0.99):\n")
  for (key in names(results_x13)) {
    m <- results_x13[[key]]
    if (!is.null(m)) {
      status <- ifelse(m$correlation > 0.99, "PASS", "FAIL")
      cat(sprintf("  [%s] %s: cor = %.8f\n", status, key, m$correlation))
    }
  }
} else {
  cat("Arquivos nao encontrados:\n")
  if (!file.exists(py_x13)) cat("  Python:", py_x13, "\n")
  if (!file.exists(r_x13)) cat("  R:", r_x13, "\n")
}

# =============================================================================
# Resumo Final
# =============================================================================
cat("\n=============================================================================\n")
cat("  RESUMO DA VALIDACAO\n")
cat("=============================================================================\n\n")

cat("Tolerancias documentadas:\n")
cat("  - Decomposicao Classica: max_diff < 1e-6 (algoritmo deterministico)\n")
cat("  - STL Decomposition: correlacao > 0.999 (LOESS pode divergir levemente)\n")
cat("  - X-13 proxy (STL robusto): correlacao > 0.99\n\n")

cat("Notas:\n")
cat("  - Diferencas na classica limitam-se a precisao de ponto flutuante\n")
cat("  - STL: R usa stats::stl() (Fortran), Python usa statsmodels (Cython)\n")
cat("    Implementacoes equivalentes mas nao identicas bit-a-bit\n")
cat("  - X-13: comparacao real requer seasonal::seas() com x13ashtml\n")
cat("    STL robusto serve como proxy para validacao dos componentes\n\n")

# Salvar resumo como JSON
summary_json <- list(
  metadata = list(
    description = "Validacao cruzada R vs Python - decomposicao de series temporais",
    generated_by = "compare_results.R",
    seed = 42,
    tolerance_classical = 1e-6,
    tolerance_stl_correlation = 0.999,
    tolerance_x13_proxy_correlation = 0.99
  ),
  classical = lapply(results_classical, function(m) {
    if (is.null(m)) return(NULL)
    list(mae = m$mae, max_diff = m$max_diff, rmse = m$rmse,
         correlation = m$correlation, n = m$n)
  }),
  stl = lapply(results_stl, function(m) {
    if (is.null(m)) return(NULL)
    list(mae = m$mae, max_diff = m$max_diff, rmse = m$rmse,
         correlation = m$correlation, n = m$n)
  }),
  x13_proxy = lapply(results_x13, function(m) {
    if (is.null(m)) return(NULL)
    list(mae = m$mae, max_diff = m$max_diff, rmse = m$rmse,
         correlation = m$correlation, n = m$n)
  })
)

# Salvar como JSON
json_path <- file.path(output_dir_r, "validation_summary.json")
if (requireNamespace("jsonlite", quietly = TRUE)) {
  jsonlite::write_json(summary_json, json_path, pretty = TRUE, auto_unbox = TRUE)
  cat("Resumo JSON salvo em:", json_path, "\n")
} else {
  # Fallback: salvar como texto estruturado
  sink(json_path)
  cat("# Validation Summary (texto - instale jsonlite para JSON real)\n")
  print(summary_json)
  sink()
  cat("Resumo salvo em:", json_path, "(instale jsonlite para formato JSON)\n")
}

cat("\n=== compare_results.R concluido ===\n")
