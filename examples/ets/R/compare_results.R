# =============================================================================
# compare_results.R
# Compara outputs Python (chronobox) vs R (forecast)
# Carrega CSVs salvos por ambos e calcula diferencas
# =============================================================================

library(forecast)

cat("=== Comparacao Python (chronobox) vs R (forecast) ===\n\n")

# --- Configuracao ---
output_dir <- file.path(dirname(getwd()), "outputs", "R")
python_dir <- file.path(dirname(getwd()), "outputs", "Python")
data_dir <- file.path(dirname(getwd()), "data")

# Tolerancia para comparacao de parametros de suavizacao
TOLERANCE <- 1e-3

# Funcao auxiliar para verificar tolerancia
check_tolerance <- function(name, r_val, py_val, tol = TOLERANCE) {
  diff <- abs(r_val - py_val)
  status <- ifelse(diff < tol, "OK", "FAIL")
  cat(sprintf("  %-20s R=%.6f  Py=%.6f  diff=%.2e  [%s]\n",
              name, r_val, py_val, diff, status))
  return(diff < tol)
}

# =============================================================================
# 1. Carregar resultados R
# =============================================================================
cat("--- Carregando resultados R ---\n")

r_files <- list(
  ets_coef = "ets_coefficients.csv",
  ets_metrics = "ets_metrics.csv",
  ets_forecasts = "ets_forecasts.csv",
  hw_base = "hw_base_coefficients.csv",
  hw_forecast = "hw_forecast_coefficients.csv",
  hw_metrics = "hw_metrics.csv",
  hw_forecasts = "hw_forecasts.csv",
  auto_coef = "auto_ets_coefficients.csv",
  auto_ranking = "auto_ets_ranking.csv",
  theta_fcast = "theta_forecasts.csv",
  oos_metrics = "auto_theta_oos_metrics.csv"
)

r_data <- list()
for (name in names(r_files)) {
  fpath <- file.path(output_dir, r_files[[name]])
  if (file.exists(fpath)) {
    r_data[[name]] <- read.csv(fpath)
    cat(sprintf("  [OK] %s\n", r_files[[name]]))
  } else {
    cat(sprintf("  [MISSING] %s\n", r_files[[name]]))
  }
}
cat("\n")

# =============================================================================
# 2. Carregar resultados Python (se disponiveis)
# =============================================================================
cat("--- Carregando resultados Python ---\n")

py_files <- list.files(python_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(py_files) == 0) {
  cat("  Nenhum resultado Python encontrado em:\n")
  cat(sprintf("    %s\n", python_dir))
  cat("\n  Para gerar os resultados Python, execute os notebooks:\n")
  cat("    examples/ets/solutions/01_ets_introduction_solution.ipynb\n")
  cat("    examples/ets/solutions/02_holt_winters_solution.ipynb\n")
  cat("    examples/ets/solutions/03_auto_ets_theta_solution.ipynb\n\n")
  cat("  Enquanto isso, apresentando apenas os resultados R...\n\n")
} else {
  cat(sprintf("  Encontrados %d arquivos:\n", length(py_files)))
  for (f in py_files) cat(sprintf("    %s\n", basename(f)))
  cat("\n")
}

# =============================================================================
# 3. Resumo dos resultados R - ETS
# =============================================================================
cat("=== Resumo ETS (R) ===\n\n")

if (!is.null(r_data$ets_coef)) {
  df <- r_data$ets_coef
  cat("  Coeficientes de suavizacao:\n")
  cat(sprintf("  %-6s  alpha     beta      gamma     phi       AICc\n", "Model"))
  cat("  ", strrep("-", 65), "\n")
  for (i in 1:nrow(df)) {
    cat(sprintf("  %-6s  %.4f    %-9s %-9s %-9s %.2f\n",
                df$model[i],
                df$alpha[i],
                ifelse(is.na(df$beta[i]), "-", sprintf("%.4f", df$beta[i])),
                ifelse(is.na(df$gamma[i]), "-", sprintf("%.4f", df$gamma[i])),
                ifelse(is.na(df$phi[i]), "-", sprintf("%.4f", df$phi[i])),
                df$AICc[i]))
  }
  cat("\n")
}

if (!is.null(r_data$ets_metrics)) {
  df <- r_data$ets_metrics
  cat("  Metricas de erro (in-sample):\n")
  cat(sprintf("  %-6s  RMSE      MAE       MAPE\n", "Model"))
  cat("  ", strrep("-", 40), "\n")
  for (i in 1:nrow(df)) {
    cat(sprintf("  %-6s  %.4f    %.4f    %.4f%%\n",
                df$model[i], df$RMSE[i], df$MAE[i], df$MAPE[i]))
  }
  cat("\n")
}

# =============================================================================
# 4. Resumo dos resultados R - Holt-Winters
# =============================================================================
cat("=== Resumo Holt-Winters (R) ===\n\n")

if (!is.null(r_data$hw_metrics)) {
  df <- r_data$hw_metrics
  cat("  Metricas:\n")
  cat(sprintf("  %-25s  RMSE      MAE       MAPE\n", "Method"))
  cat("  ", strrep("-", 55), "\n")
  for (i in 1:nrow(df)) {
    cat(sprintf("  %-25s  %.4f    %.4f    %.4f%%\n",
                df$method[i], df$RMSE[i], df$MAE[i], df$MAPE[i]))
  }
  cat("\n")
}

# =============================================================================
# 5. Resumo Auto ETS e Theta
# =============================================================================
cat("=== Resumo Auto ETS & Theta (R) ===\n\n")

if (!is.null(r_data$auto_coef)) {
  df <- r_data$auto_coef
  cat("  Selecao automatica de modelo:\n")
  for (i in 1:nrow(df)) {
    cat(sprintf("    %s -> %s (AICc=%.2f)\n",
                df$dataset[i], df$model_selected[i], df$AICc[i]))
  }
  cat("\n")
}

if (!is.null(r_data$auto_ranking)) {
  df <- r_data$auto_ranking
  cat("  Top 5 modelos por AICc (airline):\n")
  for (i in 1:min(5, nrow(df))) {
    cat(sprintf("    %d. %s (AICc=%.2f)\n", i, df$model[i], df$AICc[i]))
  }
  cat("\n")
}

if (!is.null(r_data$oos_metrics)) {
  df <- r_data$oos_metrics
  cat("  Metricas out-of-sample (h=24):\n")
  cat(sprintf("  %-12s  RMSE      MAE       MAPE\n", "Method"))
  cat("  ", strrep("-", 45), "\n")
  for (i in 1:nrow(df)) {
    cat(sprintf("  %-12s  %.4f    %.4f    %.4f%%\n",
                df$method[i], df$RMSE[i], df$MAE[i], df$MAPE[i]))
  }
  cat("\n")
}

# =============================================================================
# 6. Comparacao direta com Python (se resultados disponiveis)
# =============================================================================
if (length(py_files) > 0) {
  cat("=== Comparacao direta Python vs R ===\n\n")

  # Tentar carregar coeficientes Python
  py_coef_file <- file.path(python_dir, "ets_coefficients.csv")
  if (file.exists(py_coef_file)) {
    py_coef <- read.csv(py_coef_file)
    r_coef <- r_data$ets_coef

    cat("--- Parametros de suavizacao ---\n")
    cat(sprintf("  Tolerancia: %.0e\n\n", TOLERANCE))

    n_pass <- 0
    n_fail <- 0
    n_total <- 0

    # Comparar modelos comuns
    common_models <- intersect(r_coef$model, py_coef$model)
    for (m in common_models) {
      cat(sprintf("  Modelo: %s\n", m))
      r_row <- r_coef[r_coef$model == m, ]
      py_row <- py_coef[py_coef$model == m, ]

      for (param in c("alpha", "beta", "gamma", "phi")) {
        r_val <- r_row[[param]]
        py_val <- py_row[[param]]
        if (!is.na(r_val) && !is.na(py_val)) {
          n_total <- n_total + 1
          if (check_tolerance(param, r_val, py_val)) {
            n_pass <- n_pass + 1
          } else {
            n_fail <- n_fail + 1
          }
        }
      }
      cat("\n")
    }

    cat(sprintf("  Resultado: %d/%d parametros dentro da tolerancia (%.0e)\n",
                n_pass, n_total, TOLERANCE))
    cat(sprintf("  Status: %s\n\n",
                ifelse(n_fail == 0, "TODOS APROVADOS", paste(n_fail, "FALHARAM"))))
  }

  # Comparar forecasts
  py_fcast_file <- file.path(python_dir, "ets_forecasts.csv")
  if (file.exists(py_fcast_file)) {
    py_fcast <- read.csv(py_fcast_file)
    r_fcast <- r_data$ets_forecasts

    cat("--- Forecasts (h=24) ---\n")
    common_cols <- intersect(names(r_fcast), names(py_fcast))
    common_cols <- setdiff(common_cols, "h")

    for (col in common_cols) {
      r_vals <- r_fcast[[col]]
      py_vals <- py_fcast[[col]]
      max_diff <- max(abs(r_vals - py_vals))
      mean_diff <- mean(abs(r_vals - py_vals))
      cat(sprintf("  %-6s  max_diff=%.4f  mean_diff=%.4f\n", col, max_diff, mean_diff))
    }
    cat("\n")
  }

  # Comparar metricas
  py_metrics_file <- file.path(python_dir, "ets_metrics.csv")
  if (file.exists(py_metrics_file)) {
    py_metrics <- read.csv(py_metrics_file)
    r_metrics <- r_data$ets_metrics

    cat("--- Metricas de erro ---\n")
    common_models <- intersect(r_metrics$model, py_metrics$model)
    for (m in common_models) {
      cat(sprintf("  %s:\n", m))
      r_row <- r_metrics[r_metrics$model == m, ]
      py_row <- py_metrics[py_metrics$model == m, ]
      for (metric in c("RMSE", "MAE", "MAPE")) {
        r_val <- r_row[[metric]]
        py_val <- py_row[[metric]]
        diff <- abs(r_val - py_val)
        cat(sprintf("    %-5s  R=%.4f  Py=%.4f  diff=%.4f\n", metric, r_val, py_val, diff))
      }
    }
    cat("\n")
  }
} else {
  cat("=== Comparacao Python vs R ===\n")
  cat("  Resultados Python nao disponiveis.\n")
  cat("  Execute os notebooks Python primeiro para gerar outputs em:\n")
  cat(sprintf("    %s\n\n", python_dir))
}

# =============================================================================
# 7. Recomputar modelos para verificacao rapida
# =============================================================================
cat("=== Verificacao rapida (recomputando) ===\n\n")

# Recarregar dados e ajustar modelos chave
airline <- read.csv(file.path(data_dir, "airline.csv"))
airline_ts <- ts(airline$passengers, start = c(1949, 1), frequency = 12)

cat("Modelos ajustados para comparacao:\n\n")

models_to_check <- list(
  list(name = "ETS(A,N,N)", spec = "ANN", damped = FALSE),
  list(name = "ETS(A,A,N)", spec = "AAN", damped = FALSE),
  list(name = "ETS(A,A,A)", spec = "AAA", damped = FALSE),
  list(name = "ETS(M,A,M)", spec = "MAM", damped = FALSE),
  list(name = "ETS(M,Ad,M)", spec = "MAM", damped = TRUE)
)

cat(sprintf("  %-15s  %-7s  %-7s  %-7s  %-7s  %-10s\n",
            "Modelo", "alpha", "beta", "gamma", "phi", "AICc"))
cat("  ", strrep("-", 60), "\n")

for (m in models_to_check) {
  fit <- ets(airline_ts, model = m$spec, damped = m$damped)
  acc <- accuracy(fit)
  cat(sprintf("  %-15s  %.4f   %-7s  %-7s  %-7s  %.2f\n",
              m$name,
              fit$par["alpha"],
              ifelse("beta" %in% names(fit$par), sprintf("%.4f", fit$par["beta"]), "-"),
              ifelse("gamma" %in% names(fit$par), sprintf("%.4f", fit$par["gamma"]), "-"),
              ifelse("phi" %in% names(fit$par), sprintf("%.4f", fit$par["phi"]), "-"),
              fit$aicc))
}

cat("\n")

# Auto ETS
fit_auto <- ets(airline_ts)
cat(sprintf("Auto ETS selecionou: %s\n", fit_auto$method))

# Theta
theta <- thetaf(airline_ts, h = 24)
cat(sprintf("Theta h=1: %.2f | h=24: %.2f\n", theta$mean[1], theta$mean[24]))

cat("\n=== Comparacao concluida! ===\n")
cat(sprintf("Tolerancia para parametros de suavizacao: < %.0e\n", TOLERANCE))
cat("Nota: Diferencas maiores em alpha/beta/gamma podem ocorrer\n")
cat("      quando otimizadores convergem para solucoes equivalentes\n")
cat("      com parametros ligeiramente diferentes.\n")
