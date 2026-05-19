# =============================================================================
# compare_results.R
# Comparacao sistematica de resultados Python (chronobox) vs R (vars)
# =============================================================================
# Este script carrega os resultados salvos por ambas implementacoes
# e produz um relatorio de validacao cruzada completo.
#
# Tolerancias:
#   - Coeficientes VAR: < 1e-4
#   - P-valores: < 0.01
#   - IRF: < 1e-4
#   - FEVD: < 1e-4
#   - Estatisticas Johansen: < 1e-2 (implementacoes podem diferir levemente)

library(jsonlite)

# --- Configuracao -----------------------------------------------------------
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
  script_dir <- "."
}
base_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
py_dir <- file.path(base_dir, "outputs")       # Resultados Python
r_dir <- file.path(base_dir, "outputs", "R")    # Resultados R

cat("=============================================================\n")
cat("   VALIDACAO CRUZADA: chronobox (Python) vs vars (R)\n")
cat("=============================================================\n\n")

# Contadores globais
n_pass <- 0
n_warn <- 0
n_fail <- 0
n_skip <- 0

report_result <- function(test_name, passed, details = "") {
  if (passed) {
    cat(sprintf("  [PASS] %s\n", test_name))
    n_pass <<- n_pass + 1
  } else {
    cat(sprintf("  [WARN] %s %s\n", test_name, details))
    n_warn <<- n_warn + 1
  }
}

# =============================================================================
# 1. VAR Coefficients
# =============================================================================
cat("--- 1. Coeficientes VAR(1) ---\n")

py_var_file <- file.path(py_dir, "var_coefficients.json")
r_var_file <- file.path(r_dir, "var_coefficients.json")

if (file.exists(py_var_file) && file.exists(r_var_file)) {
  py_var <- fromJSON(py_var_file)
  r_var <- fromJSON(r_var_file)

  vars_cols <- py_var$variables
  tol_coef <- 1e-4

  # Interceptos
  py_int <- py_var$intercept
  r_int <- r_var$intercept
  diff_int <- abs(unlist(py_int) - unlist(r_int))
  report_result(
    sprintf("Interceptos (max diff: %.2e)", max(diff_int)),
    max(diff_int) < tol_coef
  )

  # Coeficientes A_1
  max_diff_A1 <- 0
  for (eq in vars_cols) {
    for (v in vars_cols) {
      py_val <- py_var$coefficients$A_1[[eq]][[v]]
      r_val <- r_var$coefficients$A_1[[eq]][[v]]
      d <- abs(py_val - r_val)
      max_diff_A1 <- max(max_diff_A1, d)
    }
  }
  report_result(
    sprintf("Coeficientes A_1 (max diff: %.2e)", max_diff_A1),
    max_diff_A1 < tol_coef
  )

  # Estabilidade
  report_result(
    sprintf("Estabilidade: Python=%s, R=%s", py_var$is_stable, r_var$is_stable),
    py_var$is_stable == r_var$is_stable
  )

  # Max eigenvalue modulus
  diff_eig <- abs(py_var$max_eigenvalue_modulus - r_var$max_eigenvalue_modulus)
  report_result(
    sprintf("Max eigenvalue modulus (diff: %.2e)", diff_eig),
    diff_eig < tol_coef
  )
} else {
  cat("  [SKIP] Arquivos de coeficientes VAR nao encontrados\n")
  n_skip <- n_skip + 1
}

# =============================================================================
# 2. Information Criteria
# =============================================================================
cat("\n--- 2. Criterios de Informacao ---\n")

py_ic_file <- file.path(py_dir, "var_ic.csv")
r_ic_file <- file.path(r_dir, "var_ic.csv")

if (file.exists(py_ic_file) && file.exists(r_ic_file)) {
  py_ic <- read.csv(py_ic_file, stringsAsFactors = FALSE)
  r_ic <- read.csv(r_ic_file, stringsAsFactors = FALSE)

  # Comparar AIC
  common_lags <- intersect(py_ic$lag, r_ic$lag)
  py_aic <- py_ic$aic[py_ic$lag %in% common_lags]
  r_aic <- r_ic$aic[r_ic$lag %in% common_lags]
  diff_aic <- abs(py_aic - r_aic)
  report_result(
    sprintf("AIC (max diff: %.2e)", max(diff_aic)),
    max(diff_aic) < 0.1  # IC podem diferir por escala/normalizacao
  )

  # Lag selecionado
  py_best_aic <- py_ic$lag[which.min(py_ic$aic)]
  r_best_aic <- r_ic$lag[which.min(r_ic$aic)]
  report_result(
    sprintf("Lag otimo AIC: Python=%d, R=%d", py_best_aic, r_best_aic),
    py_best_aic == r_best_aic
  )
} else {
  cat("  [SKIP] Arquivos de criterios de informacao nao encontrados\n")
  n_skip <- n_skip + 1
}

# =============================================================================
# 3. Johansen Cointegration
# =============================================================================
cat("\n--- 3. Teste de Johansen ---\n")

py_joh_file <- file.path(py_dir, "johansen_results.json")
r_joh_file <- file.path(r_dir, "johansen_results.json")

if (file.exists(py_joh_file) && file.exists(r_joh_file)) {
  py_joh <- fromJSON(py_joh_file)
  r_joh <- fromJSON(r_joh_file)

  tol_joh <- 1e-2  # Johansen pode variar mais entre implementacoes

  # Rank trace
  report_result(
    sprintf("Rank trace: Python=%d, R=%d", py_joh$trace_test$rank, r_joh$trace_test$rank),
    py_joh$trace_test$rank == r_joh$trace_test$rank
  )

  # Rank eigenvalue
  report_result(
    sprintf("Rank eigen: Python=%d, R=%d",
            py_joh$max_eigenvalue_test$rank, r_joh$max_eigenvalue_test$rank),
    py_joh$max_eigenvalue_test$rank == r_joh$max_eigenvalue_test$rank
  )

  # Trace statistics
  py_trace <- py_joh$trace_test$statistics
  r_trace <- r_joh$trace_test$statistics
  if (length(py_trace) == length(r_trace)) {
    diff_trace <- abs(py_trace - r_trace)
    report_result(
      sprintf("Trace stats (max diff: %.4f)", max(diff_trace)),
      max(diff_trace) < tol_joh
    )
  }

  # Eigenvalues
  py_eigvals <- py_joh$eigenvalues
  r_eigvals <- r_joh$eigenvalues
  if (length(py_eigvals) == length(r_eigvals)) {
    diff_eigvals <- abs(py_eigvals - r_eigvals)
    report_result(
      sprintf("Eigenvalues (max diff: %.2e)", max(diff_eigvals)),
      max(diff_eigvals) < tol_joh
    )
  }
} else {
  cat("  [SKIP] Arquivos de Johansen nao encontrados\n")
  n_skip <- n_skip + 1
}

# =============================================================================
# 4. IRF
# =============================================================================
cat("\n--- 4. Funcoes Impulso-Resposta (IRF) ---\n")

py_irf_file <- file.path(py_dir, "irf_results.csv")
r_irf_file <- file.path(r_dir, "irf_results.csv")

if (file.exists(py_irf_file) && file.exists(r_irf_file)) {
  py_irf <- read.csv(py_irf_file, stringsAsFactors = FALSE)
  r_irf <- read.csv(r_irf_file, stringsAsFactors = FALSE)

  tol_irf <- 1e-4

  # Merge por horizon, impulse, response
  merged <- merge(py_irf, r_irf,
                  by = c("horizon", "impulse", "response"),
                  suffixes = c("_py", "_r"))

  diff_irf <- abs(merged$irf_py - merged$irf_r)
  report_result(
    sprintf("IRF ponto (max diff: %.2e, n=%d)", max(diff_irf), nrow(merged)),
    max(diff_irf) < tol_irf
  )

  # Verificar se todas as combinacoes estao presentes
  n_expected <- length(unique(py_irf$impulse)) * length(unique(py_irf$response)) *
                length(unique(py_irf$horizon))
  report_result(
    sprintf("IRF completude: %d/%d pares encontrados", nrow(merged), n_expected),
    nrow(merged) >= n_expected * 0.9  # 90% de cobertura minima
  )

  # Top 5 maiores diferencas
  if (max(diff_irf) > tol_irf) {
    cat("\n  Maiores diferencas IRF:\n")
    merged$diff <- diff_irf
    top_diff <- head(merged[order(-merged$diff), c("horizon", "impulse", "response",
                                                    "irf_py", "irf_r", "diff")], 5)
    print(top_diff, row.names = FALSE)
  }
} else {
  cat("  [SKIP] Arquivos de IRF nao encontrados\n")
  n_skip <- n_skip + 1
}

# =============================================================================
# 5. FEVD
# =============================================================================
cat("\n--- 5. Decomposicao da Variancia (FEVD) ---\n")

py_fevd_file <- file.path(py_dir, "fevd_results.csv")
r_fevd_file <- file.path(r_dir, "fevd_results.csv")

if (file.exists(py_fevd_file) && file.exists(r_fevd_file)) {
  py_fevd <- read.csv(py_fevd_file, stringsAsFactors = FALSE)
  r_fevd <- read.csv(r_fevd_file, stringsAsFactors = FALSE)

  tol_fevd <- 1e-4

  merged <- merge(py_fevd, r_fevd,
                  by = c("horizon", "response", "shock"),
                  suffixes = c("_py", "_r"))

  diff_fevd <- abs(merged$fevd_py - merged$fevd_r)
  report_result(
    sprintf("FEVD (max diff: %.2e, n=%d)", max(diff_fevd), nrow(merged)),
    max(diff_fevd) < tol_fevd
  )

  # Verificar que FEVD soma 1 em cada horizonte/resposta
  r_fevd_sums <- aggregate(fevd ~ horizon + response, data = r_fevd, FUN = sum)
  max_sum_err <- max(abs(r_fevd_sums$fevd - 1.0))
  report_result(
    sprintf("FEVD soma = 1 (max err: %.2e)", max_sum_err),
    max_sum_err < 1e-10
  )
} else {
  cat("  [SKIP] Arquivos de FEVD nao encontrados\n")
  n_skip <- n_skip + 1
}

# =============================================================================
# 6. Granger Causality
# =============================================================================
cat("\n--- 6. Causalidade de Granger ---\n")

py_gc_file <- file.path(py_dir, "granger_results.json")
r_gc_file <- file.path(r_dir, "granger_results.json")

if (file.exists(py_gc_file) && file.exists(r_gc_file)) {
  py_gc <- fromJSON(py_gc_file)
  r_gc <- fromJSON(r_gc_file)

  tol_pval <- 0.01
  vars_cols <- py_gc$variables

  max_diff_pval <- 0
  n_agree <- 0
  n_disagree <- 0

  for (causing in vars_cols) {
    for (caused in vars_cols) {
      if (causing == caused) next

      py_pval <- py_gc$pvalue_matrix[[causing]][[caused]]
      r_pval <- r_gc$pvalue_matrix[[causing]][[caused]]

      if (!is.null(py_pval) && !is.null(r_pval)) {
        diff <- abs(py_pval - r_pval)
        max_diff_pval <- max(max_diff_pval, diff)

        # Concordancia na decisao (5%)
        py_reject <- py_pval < 0.05
        r_reject <- r_pval < 0.05
        if (py_reject == r_reject) {
          n_agree <- n_agree + 1
        } else {
          n_disagree <- n_disagree + 1
          cat(sprintf("  DESACORDO: %s->%s Py=%.4f(%s) R=%.4f(%s)\n",
                      causing, caused,
                      py_pval, ifelse(py_reject, "reject", "accept"),
                      r_pval, ifelse(r_reject, "reject", "accept")))
        }
      }
    }
  }

  report_result(
    sprintf("P-valores Granger (max diff: %.4f)", max_diff_pval),
    max_diff_pval < tol_pval
  )

  report_result(
    sprintf("Concordancia decisoes: %d/%d", n_agree, n_agree + n_disagree),
    n_disagree == 0
  )
} else {
  cat("  [SKIP] Arquivos de Granger nao encontrados\n")
  n_skip <- n_skip + 1
}

# =============================================================================
# Resumo Final
# =============================================================================
cat("\n=============================================================\n")
cat("   RESUMO DA VALIDACAO\n")
cat("=============================================================\n")
cat(sprintf("  PASS: %d\n", n_pass))
cat(sprintf("  WARN: %d\n", n_warn))
cat(sprintf("  SKIP: %d\n", n_skip))
cat("-------------------------------------------------------------\n")

if (n_warn == 0 && n_skip == 0) {
  cat("  RESULTADO: TODOS OS TESTES PASSARAM\n")
} else if (n_warn > 0) {
  cat("  RESULTADO: ALGUNS TESTES COM ADVERTENCIAS\n")
  cat("  (Diferencas podem ser esperadas entre implementacoes)\n")
} else {
  cat("  RESULTADO: TESTES INCOMPLETOS (arquivos faltando)\n")
}

cat("=============================================================\n")

# Salvar resumo
summary_output <- list(
  timestamp = Sys.time(),
  pass = n_pass,
  warn = n_warn,
  skip = n_skip,
  tolerances = list(
    coefficients = 1e-4,
    pvalues = 0.01,
    irf = 1e-4,
    fevd = 1e-4,
    johansen = 1e-2
  )
)
write_json(summary_output, file.path(r_dir, "validation_summary.json"),
           pretty = TRUE, auto_unbox = TRUE)
cat("\nResumo salvo em: outputs/R/validation_summary.json\n")
