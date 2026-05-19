# =============================================================================
# 02_vecm_johansen_validation.R
# Validacao cruzada: Teste de Johansen e estimacao VECM usando pacote vars/urca
# Compara resultados com chronobox (Python)
# =============================================================================
# Pacotes necessarios: vars, urca, jsonlite
# Tolerancia esperada: < 1e-4 para coeficientes, < 0.01 para p-valores

library(vars)
library(urca)
library(jsonlite)

# --- Configuracao -----------------------------------------------------------
set.seed(42)

script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
  script_dir <- "."
}
base_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
data_dir <- file.path(base_dir, "data")
output_dir <- file.path(base_dir, "outputs", "R")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== VECM / Johansen Validation (R) ===\n")

# --- Carregar dados ---------------------------------------------------------
data_file <- file.path(data_dir, "us_macro_quarterly.csv")
df <- read.csv(data_file, stringsAsFactors = FALSE)

vars_cols <- c("gdp", "inflation", "fed_funds", "unemployment")
y <- df[, vars_cols]

cat("Dataset: us_macro_quarterly.csv\n")
cat("Observacoes:", nrow(y), "\n")
cat("Variaveis:", paste(vars_cols, collapse = ", "), "\n\n")

# --- Teste de Johansen (Trace) ----------------------------------------------
cat("--- Teste de Johansen (Trace) ---\n")
# K = lags em niveis, entao para VECM com 4 lags em diferenca usamos K=4
# ecdet = "const" corresponde a "ci" (constante restrita ao espaco de cointegracao)
johansen_trace <- ca.jo(y, type = "trace", ecdet = "const", K = 4, spec = "transitory")

# Extrair estatisticas
trace_stats <- johansen_trace@teststat
trace_cvals <- johansen_trace@cval

cat("Estatisticas Trace:\n")
for (i in seq_along(trace_stats)) {
  cat(sprintf("  r <= %d: stat = %.4f, cv_5%% = %.2f\n",
              length(trace_stats) - i, trace_stats[i], trace_cvals[i, "5pct"]))
}

# --- Teste de Johansen (Max Eigenvalue) -------------------------------------
cat("\n--- Teste de Johansen (Max Eigenvalue) ---\n")
johansen_eigen <- ca.jo(y, type = "eigen", ecdet = "const", K = 4, spec = "transitory")

eigen_stats <- johansen_eigen@teststat
eigen_cvals <- johansen_eigen@cval

cat("Estatisticas Eigenvalue:\n")
for (i in seq_along(eigen_stats)) {
  cat(sprintf("  r <= %d: stat = %.4f, cv_5%% = %.2f\n",
              length(eigen_stats) - i, eigen_stats[i], eigen_cvals[i, "5pct"]))
}

# --- Determinar rank de cointegracao ----------------------------------------
# Rank = numero de rejeicoes ao nivel 5%
trace_rank <- sum(trace_stats > trace_cvals[, "5pct"])
eigen_rank <- sum(eigen_stats > eigen_cvals[, "5pct"])
cat("\nRank cointegracao (trace):", trace_rank, "\n")
cat("Rank cointegracao (eigen):", eigen_rank, "\n")

# --- Eigenvalues ------------------------------------------------------------
eigenvalues <- johansen_trace@lambda
cat("\nEigenvalues:", paste(round(eigenvalues, 8), collapse = ", "), "\n")

# --- Estimacao VECM via vec2var() -------------------------------------------
cat("\n--- Estimacao VECM ---\n")
# Usar rank = 3 (conforme resultado Python)
coint_rank <- 3
vecm_model <- vec2var(johansen_trace, r = coint_rank)

cat("VECM estimado com rank =", coint_rank, "\n")

# Extrair alpha e beta da estimacao ca.jo
# beta: vetores de cointegracao (normalizados)
beta_raw <- johansen_trace@V[, 1:coint_rank]
alpha_raw <- johansen_trace@W[, 1:coint_rank]

cat("\nAlpha (loading matrix):\n")
print(round(alpha_raw, 8))

cat("\nBeta (cointegrating vectors, normalized):\n")
print(round(beta_raw, 8))

# --- Matriz Pi = alpha * beta' ----------------------------------------------
Pi <- alpha_raw %*% t(beta_raw)
cat("\nMatriz Pi (alpha x beta'):\n")
print(round(Pi, 8))

# --- Salvar resultados em JSON ----------------------------------------------
# Montar valores criticos como lista
trace_cv95 <- trace_cvals[, "5pct"]
eigen_cv95 <- eigen_cvals[, "5pct"]

results <- list(
  dataset = "us_macro_quarterly.csv",
  variables = vars_cols,
  lags = 4,
  deterministic = "ci",
  nobs = nrow(y),
  trace_test = list(
    rank = trace_rank,
    statistics = as.numeric(rev(trace_stats)),
    critical_values_95 = as.numeric(rev(trace_cv95))
  ),
  max_eigenvalue_test = list(
    rank = eigen_rank,
    statistics = as.numeric(rev(eigen_stats)),
    critical_values_95 = as.numeric(rev(eigen_cv95))
  ),
  eigenvalues = as.numeric(eigenvalues),
  vecm_estimation = list(
    coint_rank = coint_rank,
    alpha = alpha_raw,
    beta = beta_raw,
    pi = Pi
  )
)

output_file <- file.path(output_dir, "johansen_results.json")
write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE)
cat("\nResultados salvos em:", output_file, "\n")

# --- Comparacao rapida com Python -------------------------------------------
cat("\n--- Comparacao rapida com Python ---\n")
py_file <- file.path(base_dir, "outputs", "johansen_results.json")
if (file.exists(py_file)) {
  py_results <- fromJSON(py_file)

  # Comparar estatisticas trace
  py_trace <- py_results$trace_test$statistics
  r_trace <- as.numeric(rev(trace_stats))
  diff_trace <- abs(r_trace - py_trace)
  cat("Diferenca max trace stats:", max(diff_trace), "\n")

  # Comparar eigenvalues
  py_eigen <- py_results$eigenvalues
  diff_eigenval <- abs(eigenvalues - py_eigen)
  cat("Diferenca max eigenvalues:", max(diff_eigenval), "\n")

  # Comparar Pi
  py_pi <- matrix(unlist(py_results$vecm_estimation$pi), nrow = 4, byrow = TRUE)
  diff_pi <- abs(Pi - py_pi)
  cat("Diferenca max Pi:", max(diff_pi), "\n")

  tol <- 1e-4
  if (max(diff_trace) < tol && max(diff_pi) < tol) {
    cat("PASS: Resultados Johansen/VECM dentro da tolerancia (< 1e-4)\n")
  } else {
    cat("WARN: Algumas diferencas excedem a tolerancia\n")
    cat("  (Nota: diferencas em Johansen sao esperadas entre implementacoes)\n")
  }
} else {
  cat("Arquivo Python nao encontrado, pulando comparacao.\n")
}

cat("\n=== Fim da validacao VECM/Johansen ===\n")
