# =============================================================================
# 04_granger_validation.R
# Validacao cruzada: Testes de causalidade de Granger usando pacote vars
# Compara resultados com chronobox (Python)
# =============================================================================
# Pacotes necessarios: vars, jsonlite
# Tolerancia esperada: < 0.01 para p-valores

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

cat("=== Granger Causality Validation (R) ===\n")

# --- Carregar dados ---------------------------------------------------------
data_file <- file.path(data_dir, "us_macro_quarterly.csv")
df <- read.csv(data_file, stringsAsFactors = FALSE)

vars_cols <- c("gdp", "inflation", "fed_funds", "unemployment")
y <- df[, vars_cols]

cat("Dataset: us_macro_quarterly.csv\n")
cat("Observacoes:", nrow(y), "\n\n")

# --- Estimar VAR(4) para testes de Granger ----------------------------------
# Usamos lag = 4 conforme Python
var_model <- VAR(y, p = 4, type = "const")
cat("Modelo VAR(4) estimado para testes de Granger\n\n")

# --- Testes de Granger (pairwise) -------------------------------------------
cat("--- Testes de Causalidade de Granger ---\n")
cat("H0: X nao Granger-causa Y\n")
cat("Nivel de significancia: 5%\n\n")

# Tabela de resultados pairwise
pairwise_tests <- list()
pvalue_matrix <- matrix(NA, nrow = length(vars_cols), ncol = length(vars_cols),
                        dimnames = list(vars_cols, vars_cols))

for (causing in vars_cols) {
  for (caused in vars_cols) {
    if (causing == caused) next

    # vars::causality() testa se "cause" Granger-causa as demais variaveis
    # Para teste pairwise, precisamos de um VAR bivariado ou usar o teste
    # no contexto multivariado
    # causality() no vars testa se a variavel "cause" Granger-causa "caused"
    # no contexto do VAR completo
    granger_test <- causality(var_model, cause = causing)

    # O teste F de Granger
    fstat <- granger_test$Granger$statistic
    pvalue <- granger_test$Granger$p.value

    pairwise_tests[[length(pairwise_tests) + 1]] <- list(
      causing = causing,
      caused = "all_others",
      fstat = as.numeric(fstat),
      pvalue = as.numeric(pvalue),
      reject_5pct = as.numeric(pvalue) < 0.05
    )

    cat(sprintf("  %s -> others: F = %.4f, p = %.6f %s\n",
                causing,
                as.numeric(fstat),
                as.numeric(pvalue),
                ifelse(as.numeric(pvalue) < 0.05, "*", "")))

    # Nota: vars::causality() testa se "cause" Granger-causa TODAS as outras
    # variaveis conjuntamente, nao pairwise individual
    break  # So precisamos testar uma vez por variavel "cause"
  }
}

# Para testes pairwise individuais, usamos VARs bivariados
cat("\n--- Testes Pairwise (VARs bivariados com 4 lags) ---\n")
pairwise_results <- list()

for (causing in vars_cols) {
  for (caused in vars_cols) {
    if (causing == caused) next

    # Ajustar VAR bivariado
    y_pair <- y[, c(caused, causing)]
    var_pair <- VAR(y_pair, p = 4, type = "const")

    # Teste de Granger
    granger <- causality(var_pair, cause = causing)
    fstat <- as.numeric(granger$Granger$statistic)
    pvalue <- as.numeric(granger$Granger$p.value)

    pvalue_matrix[causing, caused] <- pvalue

    pairwise_results[[length(pairwise_results) + 1]] <- list(
      causing = causing,
      caused = caused,
      fstat = fstat,
      pvalue = pvalue,
      reject_5pct = pvalue < 0.05
    )

    cat(sprintf("  %s -> %s: F = %.4f, p = %.6f %s\n",
                causing, caused, fstat, pvalue,
                ifelse(pvalue < 0.05, "*", "")))
  }
}

# --- Resumo -----------------------------------------------------------------
cat("\n--- Matriz de P-valores ---\n")
print(round(pvalue_matrix, 6))

n_significant <- sum(pvalue_matrix < 0.05, na.rm = TRUE)
cat("\nRelacoes significativas a 5%:", n_significant, "de",
    sum(!is.na(pvalue_matrix)), "\n")

# --- Salvar resultados ------------------------------------------------------
# Converter pvalue_matrix para lista de listas (compativel com Python)
pval_list <- list()
for (i in vars_cols) {
  row_list <- list()
  for (j in vars_cols) {
    if (i == j) {
      row_list[[j]] <- NULL
    } else {
      row_list[[j]] <- pvalue_matrix[i, j]
    }
  }
  pval_list[[i]] <- row_list
}

results <- list(
  dataset = "us_macro_quarterly.csv",
  variables = vars_cols,
  var_lags = 4,
  significance_level = 0.05,
  pvalue_matrix = pval_list,
  pairwise_tests = pairwise_results,
  summary = list(
    n_significant_5pct = n_significant,
    n_total_tests = sum(!is.na(pvalue_matrix)),
    fed_funds_to_gdp_pvalue = pvalue_matrix["fed_funds", "gdp"],
    fed_funds_to_inflation_pvalue = pvalue_matrix["fed_funds", "inflation"],
    fed_funds_to_unemployment_pvalue = pvalue_matrix["fed_funds", "unemployment"]
  )
)

output_file <- file.path(output_dir, "granger_results.json")
write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE)
cat("\nResultados salvos em:", output_file, "\n")

# --- Comparacao rapida com Python -------------------------------------------
cat("\n--- Comparacao rapida com Python ---\n")
py_file <- file.path(base_dir, "outputs", "granger_results.json")
if (file.exists(py_file)) {
  py_results <- fromJSON(py_file)

  # Comparar p-valores pairwise
  cat("Comparacao de p-valores (Python vs R):\n")
  tol_pvalue <- 0.01

  max_diff <- 0
  for (causing in vars_cols) {
    for (caused in vars_cols) {
      if (causing == caused) next

      py_pval <- py_results$pvalue_matrix[[causing]][[caused]]
      r_pval <- pvalue_matrix[causing, caused]

      if (!is.null(py_pval) && !is.na(r_pval)) {
        diff <- abs(r_pval - py_pval)
        max_diff <- max(max_diff, diff)

        if (diff > tol_pvalue) {
          cat(sprintf("  DIFF %s->%s: R=%.6f, Py=%.6f, diff=%.6f\n",
                      causing, caused, r_pval, py_pval, diff))
        }
      }
    }
  }

  cat("Diferenca max p-valores:", max_diff, "\n")
  if (max_diff < tol_pvalue) {
    cat("PASS: P-valores dentro da tolerancia (< 0.01)\n")
  } else {
    cat("WARN: Algumas diferencas excedem a tolerancia\n")
    cat("  (Nota: testes Granger em VAR bivariado vs multivariado podem diferir)\n")
  }
} else {
  cat("Arquivo Python nao encontrado, pulando comparacao.\n")
}

cat("\n=== Fim da validacao Granger ===\n")
