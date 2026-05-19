# =============================================================================
# 03_irf_fevd_validation.R
# Validacao cruzada: IRF e FEVD usando pacote vars
# Compara resultados com chronobox (Python)
# =============================================================================
# Pacotes necessarios: vars, jsonlite
# Tolerancia esperada: < 1e-4 para coeficientes

library(vars)
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

cat("=== IRF & FEVD Validation (R) ===\n")

# --- Carregar dados ---------------------------------------------------------
data_file <- file.path(data_dir, "us_macro_quarterly.csv")
df <- read.csv(data_file, stringsAsFactors = FALSE)

vars_cols <- c("gdp", "inflation", "fed_funds", "unemployment")
y <- df[, vars_cols]

cat("Dataset: us_macro_quarterly.csv\n")
cat("Observacoes:", nrow(y), "\n\n")

# --- Estimar VAR(1) ---------------------------------------------------------
var_model <- VAR(y, p = 1, type = "const")
cat("Modelo VAR(1) estimado\n\n")

# --- IRF Ortogonalizada (Cholesky) ------------------------------------------
cat("--- IRF Ortogonalizada (Cholesky) ---\n")
n_ahead <- 20

# Calcular IRF ortogonalizada com bootstrap para intervalos de confianca
irf_orth <- irf(var_model, impulse = vars_cols, response = vars_cols,
                n.ahead = n_ahead, ortho = TRUE,
                boot = TRUE, ci = 0.95, runs = 500, seed = 42)

# Extrair valores e salvar em formato CSV
irf_rows <- list()
for (imp in vars_cols) {
  for (resp in vars_cols) {
    irf_vals <- irf_orth$irf[[imp]][, resp]
    lower_vals <- irf_orth$Lower[[imp]][, resp]
    upper_vals <- irf_orth$Upper[[imp]][, resp]

    for (h in 0:n_ahead) {
      irf_rows[[length(irf_rows) + 1]] <- data.frame(
        horizon = h,
        impulse = imp,
        response = resp,
        irf = irf_vals[h + 1],
        lower = lower_vals[h + 1],
        upper = upper_vals[h + 1],
        stringsAsFactors = FALSE
      )
    }
  }
}

irf_df <- do.call(rbind, irf_rows)
rownames(irf_df) <- NULL

# Mostrar primeiras linhas
cat("Primeiras respostas (impulso = gdp):\n")
subset_gdp <- irf_df[irf_df$impulse == "gdp" & irf_df$horizon <= 5, ]
print(subset_gdp[, c("horizon", "response", "irf")], row.names = FALSE)

# Salvar IRF
irf_file <- file.path(output_dir, "irf_results.csv")
write.csv(irf_df, irf_file, row.names = FALSE)
cat("\nIRF salva em:", irf_file, "\n\n")

# --- FEVD -------------------------------------------------------------------
cat("--- FEVD (Forecast Error Variance Decomposition) ---\n")

# FEVD com bootstrap
fevd_result <- fevd(var_model, n.ahead = n_ahead)

# Extrair valores em formato longo
fevd_rows <- list()
for (resp in vars_cols) {
  fevd_mat <- fevd_result[[resp]]  # Matriz (n_ahead x n_vars)

  for (h in 1:n_ahead) {
    for (shock in vars_cols) {
      fevd_rows[[length(fevd_rows) + 1]] <- data.frame(
        horizon = h - 1,
        response = resp,
        shock = shock,
        fevd = fevd_mat[h, shock],
        stringsAsFactors = FALSE
      )
    }
  }
}

fevd_df <- do.call(rbind, fevd_rows)
rownames(fevd_df) <- NULL

# Mostrar primeiras linhas
cat("FEVD para gdp (horizonte 0-5):\n")
subset_fevd <- fevd_df[fevd_df$response == "gdp" & fevd_df$horizon <= 5, ]
print(subset_fevd, row.names = FALSE)

# Salvar FEVD
fevd_file <- file.path(output_dir, "fevd_results.csv")
write.csv(fevd_df, fevd_file, row.names = FALSE)
cat("\nFEVD salva em:", fevd_file, "\n\n")

# --- Comparacao rapida com Python -------------------------------------------
cat("--- Comparacao rapida com Python ---\n")

# Comparar IRF
py_irf_file <- file.path(base_dir, "outputs", "irf_results.csv")
if (file.exists(py_irf_file)) {
  py_irf <- read.csv(py_irf_file, stringsAsFactors = FALSE)

  # Comparar valores de IRF (ponto a ponto)
  merged_irf <- merge(irf_df, py_irf,
                       by = c("horizon", "impulse", "response"),
                       suffixes = c("_r", "_py"))
  diff_irf <- abs(merged_irf$irf_r - merged_irf$irf_py)
  cat("IRF - Diferenca max:", max(diff_irf), "\n")
  cat("IRF - Diferenca media:", mean(diff_irf), "\n")

  tol <- 1e-4
  if (max(diff_irf) < tol) {
    cat("PASS: IRF dentro da tolerancia (< 1e-4)\n")
  } else {
    cat("WARN: Algumas diferencas IRF excedem a tolerancia\n")
    # Mostrar maiores diferencas
    big_diff <- merged_irf[diff_irf > tol, c("horizon", "impulse", "response", "irf_r", "irf_py")]
    if (nrow(big_diff) > 0) {
      cat("Maiores diferencas (top 5):\n")
      big_diff$diff <- abs(big_diff$irf_r - big_diff$irf_py)
      print(head(big_diff[order(-big_diff$diff), ], 5), row.names = FALSE)
    }
  }
} else {
  cat("Arquivo Python IRF nao encontrado.\n")
}

# Comparar FEVD
py_fevd_file <- file.path(base_dir, "outputs", "fevd_results.csv")
if (file.exists(py_fevd_file)) {
  py_fevd <- read.csv(py_fevd_file, stringsAsFactors = FALSE)

  merged_fevd <- merge(fevd_df, py_fevd,
                        by = c("horizon", "response", "shock"),
                        suffixes = c("_r", "_py"))
  diff_fevd <- abs(merged_fevd$fevd_r - merged_fevd$fevd_py)
  cat("\nFEVD - Diferenca max:", max(diff_fevd), "\n")
  cat("FEVD - Diferenca media:", mean(diff_fevd), "\n")

  if (max(diff_fevd) < tol) {
    cat("PASS: FEVD dentro da tolerancia (< 1e-4)\n")
  } else {
    cat("WARN: Algumas diferencas FEVD excedem a tolerancia\n")
  }
} else {
  cat("Arquivo Python FEVD nao encontrado.\n")
}

cat("\n=== Fim da validacao IRF & FEVD ===\n")
