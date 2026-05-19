# =============================================================================
# 01_var_validation.R
# Validacao cruzada: VAR estimation usando pacote vars
# Compara resultados com chronobox (Python)
# =============================================================================
# Pacotes necessarios: vars
# Tolerancia esperada: < 1e-4 para coeficientes VAR

library(vars)
library(jsonlite)

# --- Configuracao -----------------------------------------------------------
set.seed(42)

# Diretorios
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
  script_dir <- "."
}
base_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
data_dir <- file.path(base_dir, "data")
output_dir <- file.path(base_dir, "outputs", "R")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== VAR Validation (R) ===\n")
cat("Data dir:", data_dir, "\n")
cat("Output dir:", output_dir, "\n\n")

# --- Carregar dados ---------------------------------------------------------
data_file <- file.path(data_dir, "us_macro_quarterly.csv")
df <- read.csv(data_file, stringsAsFactors = FALSE)

# Converter para serie temporal (colunas numericas apenas)
vars_cols <- c("gdp", "inflation", "fed_funds", "unemployment")
y <- df[, vars_cols]

cat("Dataset: us_macro_quarterly.csv\n")
cat("Observacoes:", nrow(y), "\n")
cat("Variaveis:", paste(vars_cols, collapse = ", "), "\n\n")

# --- Selecao de ordem (VARselect) -------------------------------------------
cat("--- Selecao de Ordem (VARselect) ---\n")
lag_max <- 12
var_select <- VARselect(y, lag.max = lag_max, type = "const")

# Extrair criterios de informacao
ic_table <- data.frame(
  lag = 1:lag_max,
  aic = var_select$criteria["AIC(n)", ],
  bic = var_select$criteria["SC(n)", ],  # SC = BIC no vars
  hqic = var_select$criteria["HQ(n)", ],
  fpe = var_select$criteria["FPE(n)", ]
)
rownames(ic_table) <- NULL

cat("Ordem selecionada por cada criterio:\n")
cat("  AIC:", var_select$selection["AIC(n)"], "\n")
cat("  BIC:", var_select$selection["SC(n)"], "\n")
cat("  HQ:",  var_select$selection["HQ(n)"], "\n")
cat("  FPE:", var_select$selection["FPE(n)"], "\n\n")

# Salvar criterios de informacao
write.csv(ic_table, file.path(output_dir, "var_ic.csv"), row.names = FALSE)
cat("Criterios salvos em: outputs/R/var_ic.csv\n\n")

# --- Estimacao VAR(1) -------------------------------------------------------
# Usamos lag = 1 para comparacao direta com Python (que usou VAR(1))
cat("--- Estimacao VAR(1) ---\n")
var_model <- VAR(y, p = 1, type = "const")

cat("Modelo: VAR(1) com constante\n")
cat("N. obs:", var_model$obs, "\n")
cat("N. equacoes:", var_model$K, "\n\n")

# --- Extrair coeficientes ---------------------------------------------------
# Coeficientes A_1 (matriz de transicao)
coef_list <- list()
intercept <- numeric(length(vars_cols))
A1 <- matrix(0, nrow = length(vars_cols), ncol = length(vars_cols),
             dimnames = list(vars_cols, vars_cols))

for (i in seq_along(vars_cols)) {
  eq_name <- vars_cols[i]
  eq_coefs <- coef(var_model)[[eq_name]]

  # Intercepto
  intercept[i] <- eq_coefs["const", "Estimate"]

  # Coeficientes lag 1
  for (j in seq_along(vars_cols)) {
    lag_name <- paste0(vars_cols[j], ".l1")
    A1[i, j] <- eq_coefs[lag_name, "Estimate"]
  }
}

cat("Interceptos:\n")
for (i in seq_along(vars_cols)) {
  cat(sprintf("  %s: %.10f\n", vars_cols[i], intercept[i]))
}

cat("\nMatriz A_1 (coeficientes lag 1):\n")
print(round(A1, 8))

# --- Estabilidade -----------------------------------------------------------
roots_mod <- roots(var_model)
is_stable <- all(roots_mod < 1)
cat("\nRaizes (modulo):", paste(round(roots_mod, 6), collapse = ", "), "\n")
cat("Estavel:", is_stable, "\n")

# --- Covariancia residual ---------------------------------------------------
resid_cov <- summary(var_model)$covres
cat("\nCovariancia residual:\n")
print(round(resid_cov, 10))

# --- Salvar resultados em JSON ----------------------------------------------
# Montar estrutura compativel com Python
A1_list <- list()
for (i in seq_along(vars_cols)) {
  row_list <- list()
  for (j in seq_along(vars_cols)) {
    row_list[[vars_cols[j]]] <- A1[i, j]
  }
  A1_list[[vars_cols[i]]] <- row_list
}

results <- list(
  model = "VAR(1)",
  dataset = "us_macro_quarterly.csv",
  variables = vars_cols,
  lag_order = 1,
  trend = "c",
  nobs = var_model$obs,
  neqs = var_model$K,
  is_stable = is_stable,
  max_eigenvalue_modulus = max(roots_mod),
  intercept = intercept,
  coefficients = list(A_1 = A1_list),
  residual_covariance = as.list(as.data.frame(resid_cov))
)

output_file <- file.path(output_dir, "var_coefficients.json")
write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE)
cat("\nResultados salvos em:", output_file, "\n")

# --- Comparacao rapida com Python -------------------------------------------
cat("\n--- Comparacao rapida com Python ---\n")
py_file <- file.path(base_dir, "outputs", "var_coefficients.json")
if (file.exists(py_file)) {
  py_results <- fromJSON(py_file)

  # Comparar interceptos
  py_intercept <- py_results$intercept
  diff_intercept <- abs(intercept - py_intercept)
  cat("Diferenca max intercepto:", max(diff_intercept), "\n")

  # Comparar A_1
  py_A1 <- matrix(0, nrow = 4, ncol = 4)
  for (i in seq_along(vars_cols)) {
    for (j in seq_along(vars_cols)) {
      py_A1[i, j] <- py_results$coefficients$A_1[[vars_cols[i]]][[vars_cols[j]]]
    }
  }
  diff_A1 <- abs(A1 - py_A1)
  cat("Diferenca max A_1:", max(diff_A1), "\n")

  tol <- 1e-4
  if (max(diff_A1) < tol && max(diff_intercept) < tol) {
    cat("PASS: Coeficientes dentro da tolerancia (< 1e-4)\n")
  } else {
    cat("WARN: Algumas diferencas excedem a tolerancia\n")
  }
} else {
  cat("Arquivo Python nao encontrado, pulando comparacao.\n")
}

cat("\n=== Fim da validacao VAR ===\n")
