# =============================================================================
# 03_auto_ets_theta_validation.R
# Validacao cruzada: ets() automatico e thetaf()
# Compara selecao automatica e metodo Theta com chronobox
# =============================================================================

library(forecast)

cat("=== Auto ETS & Theta Validation Script (R) ===\n\n")

# --- Configuracao ---
set.seed(42)
output_dir <- file.path(dirname(getwd()), "outputs", "R")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

data_dir <- file.path(dirname(getwd()), "data")

# --- Carregar dados ---
cat("Carregando datasets...\n")

airline <- read.csv(file.path(data_dir, "airline.csv"))
airline_ts <- ts(airline$passengers, start = c(1949, 1), frequency = 12)

synthetic <- read.csv(file.path(data_dir, "ets_synthetic.csv"))
synthetic_ts <- ts(synthetic$value, start = c(2009, 1), frequency = 12)

cat(sprintf("  Airline: %d observacoes\n", length(airline_ts)))
cat(sprintf("  Synthetic: %d observacoes\n\n", length(synthetic_ts)))

# --- Train/Test split (ultimas 24 observacoes como teste) ---
n_airline <- length(airline_ts)
n_test <- 24
airline_train <- window(airline_ts, end = c(1949 + (n_airline - n_test - 1) %/% 12,
                                             ((n_airline - n_test - 1) %% 12) + 1))
airline_test <- window(airline_ts, start = c(1949 + (n_airline - n_test) %/% 12,
                                              ((n_airline - n_test) %% 12) + 1))

# Abordagem mais simples para split
airline_train <- ts(airline$passengers[1:(n_airline - n_test)],
                    start = c(1949, 1), frequency = 12)
airline_test <- ts(airline$passengers[(n_airline - n_test + 1):n_airline],
                   start = c(1949 + (n_airline - n_test) %/% 12,
                             ((n_airline - n_test) %% 12) + 1),
                   frequency = 12)

cat(sprintf("  Train: %d obs | Test: %d obs\n\n", length(airline_train), length(airline_test)))

# =============================================================================
# 1. Auto ETS - Selecao automatica de modelo
# =============================================================================
cat("=== Auto ETS (selecao automatica) ===\n\n")

# --- Com AICc (padrao) ---
cat("--- ets() automatico - criterio AICc (airline completo) ---\n")
fit_auto_aicc <- ets(airline_ts)
cat(sprintf("  Modelo selecionado: %s\n", fit_auto_aicc$method))
cat(sprintf("  alpha = %.6f\n", fit_auto_aicc$par["alpha"]))
if ("beta" %in% names(fit_auto_aicc$par))
  cat(sprintf("  beta  = %.6f\n", fit_auto_aicc$par["beta"]))
if ("gamma" %in% names(fit_auto_aicc$par))
  cat(sprintf("  gamma = %.6f\n", fit_auto_aicc$par["gamma"]))
if ("phi" %in% names(fit_auto_aicc$par))
  cat(sprintf("  phi   = %.6f\n", fit_auto_aicc$par["phi"]))
cat(sprintf("  AIC   = %.4f\n", fit_auto_aicc$aic))
cat(sprintf("  BIC   = %.4f\n", fit_auto_aicc$bic))
cat(sprintf("  AICc  = %.4f\n", fit_auto_aicc$aicc))
cat(sprintf("  loglik= %.4f\n", fit_auto_aicc$loglik))

acc_auto <- accuracy(fit_auto_aicc)
cat(sprintf("  RMSE  = %.4f\n", acc_auto[, "RMSE"]))
cat(sprintf("  MAE   = %.4f\n", acc_auto[, "MAE"]))
cat(sprintf("  MAPE  = %.4f%%\n\n", acc_auto[, "MAPE"]))

# --- Comparar criterios: AIC vs BIC vs AICc ---
cat("--- Comparacao de criterios de selecao ---\n")

# AIC
fit_auto_aic <- ets(airline_ts, ic = "aic")
cat(sprintf("  AIC  -> modelo: %s (AIC=%.2f)\n", fit_auto_aic$method, fit_auto_aic$aic))

# BIC
fit_auto_bic <- ets(airline_ts, ic = "bic")
cat(sprintf("  BIC  -> modelo: %s (BIC=%.2f)\n", fit_auto_bic$method, fit_auto_bic$bic))

# AICc (padrao)
cat(sprintf("  AICc -> modelo: %s (AICc=%.2f)\n\n", fit_auto_aicc$method, fit_auto_aicc$aicc))

# --- Auto ETS no train set com forecast ---
cat("--- ets() automatico no train set (airline) ---\n")
fit_train <- ets(airline_train)
cat(sprintf("  Modelo selecionado: %s\n", fit_train$method))
fcast_train <- forecast(fit_train, h = n_test)

# Metricas out-of-sample
acc_oos <- accuracy(fcast_train, airline_test)
cat("  Metricas in-sample (train):\n")
cat(sprintf("    RMSE = %.4f\n", acc_oos["Training set", "RMSE"]))
cat(sprintf("    MAE  = %.4f\n", acc_oos["Training set", "MAE"]))
cat(sprintf("    MAPE = %.4f%%\n", acc_oos["Training set", "MAPE"]))
cat("  Metricas out-of-sample (test):\n")
cat(sprintf("    RMSE = %.4f\n", acc_oos["Test set", "RMSE"]))
cat(sprintf("    MAE  = %.4f\n", acc_oos["Test set", "MAE"]))
cat(sprintf("    MAPE = %.4f%%\n\n", acc_oos["Test set", "MAPE"]))

# =============================================================================
# 2. Auto ETS no dataset sintetico
# =============================================================================
cat("=== Auto ETS no dataset sintetico ===\n\n")

fit_syn_auto <- ets(synthetic_ts)
cat(sprintf("  Modelo selecionado: %s\n", fit_syn_auto$method))
cat(sprintf("  alpha = %.6f\n", fit_syn_auto$par["alpha"]))
if ("beta" %in% names(fit_syn_auto$par))
  cat(sprintf("  beta  = %.6f\n", fit_syn_auto$par["beta"]))
if ("gamma" %in% names(fit_syn_auto$par))
  cat(sprintf("  gamma = %.6f\n", fit_syn_auto$par["gamma"]))
if ("phi" %in% names(fit_syn_auto$par))
  cat(sprintf("  phi   = %.6f\n", fit_syn_auto$par["phi"]))
cat(sprintf("  AIC   = %.4f\n", fit_syn_auto$aic))
cat(sprintf("  AICc  = %.4f\n", fit_syn_auto$aicc))
cat(sprintf("  BIC   = %.4f\n\n", fit_syn_auto$bic))

acc_syn <- accuracy(fit_syn_auto)
cat(sprintf("  RMSE  = %.4f\n", acc_syn[, "RMSE"]))
cat(sprintf("  MAE   = %.4f\n", acc_syn[, "MAE"]))
cat(sprintf("  MAPE  = %.4f%%\n\n", acc_syn[, "MAPE"]))

# Nota: o dataset sintetico foi gerado como ETS(M,A,M)
cat("  Nota: dataset sintetico foi gerado como ETS(M,A,M)\n")
cat(sprintf("  Auto ETS recuperou: %s\n\n", fit_syn_auto$method))

# =============================================================================
# 3. Metodo Theta
# =============================================================================
cat("=== Metodo Theta ===\n\n")

# --- Theta no airline completo ---
cat("--- thetaf() (airline completo, h=24) ---\n")
theta_full <- thetaf(airline_ts, h = 24)
cat(sprintf("  Forecast h=1: %.4f\n", theta_full$mean[1]))
cat(sprintf("  Forecast h=12: %.4f\n", theta_full$mean[12]))
cat(sprintf("  Forecast h=24: %.4f\n\n", theta_full$mean[24]))

# --- Theta no train set com avaliacao out-of-sample ---
cat("--- thetaf() no train set (h=%d) ---\n", n_test)
theta_train <- thetaf(airline_train, h = n_test)

# Metricas out-of-sample
theta_errors <- as.numeric(airline_test) - as.numeric(theta_train$mean)
theta_rmse <- sqrt(mean(theta_errors^2))
theta_mae <- mean(abs(theta_errors))
theta_mape <- mean(abs(theta_errors / as.numeric(airline_test))) * 100

cat(sprintf("  Metricas out-of-sample:\n"))
cat(sprintf("    RMSE = %.4f\n", theta_rmse))
cat(sprintf("    MAE  = %.4f\n", theta_mae))
cat(sprintf("    MAPE = %.4f%%\n\n", theta_mape))

# --- Theta no sintetico ---
cat("--- thetaf() (sintetico, h=24) ---\n")
theta_syn <- thetaf(synthetic_ts, h = 24)
cat(sprintf("  Forecast h=1: %.4f\n", theta_syn$mean[1]))
cat(sprintf("  Forecast h=12: %.4f\n", theta_syn$mean[12]))
cat(sprintf("  Forecast h=24: %.4f\n\n", theta_syn$mean[24]))

# =============================================================================
# 4. Ranking de modelos - Top 10 por AICc
# =============================================================================
cat("=== Ranking de modelos ETS por AICc (airline) ===\n\n")

# Ajustar diversos modelos ETS e comparar
model_specs <- c("ANN", "AAN", "ANA", "AAA",
                 "MNN", "MAN", "MNA", "MAA",
                 "MNM", "MAM", "MMM", "MMN")

results <- data.frame(
  model = character(),
  damped = logical(),
  AIC = numeric(),
  BIC = numeric(),
  AICc = numeric(),
  stringsAsFactors = FALSE
)

for (spec in model_specs) {
  # Sem damping
  tryCatch({
    fit <- ets(airline_ts, model = spec, damped = FALSE)
    results <- rbind(results, data.frame(
      model = fit$method, damped = FALSE,
      AIC = fit$aic, BIC = fit$bic, AICc = fit$aicc
    ))
  }, error = function(e) {})

  # Com damping (so se tem trend)
  if (substr(spec, 2, 2) %in% c("A", "M")) {
    tryCatch({
      fit_d <- ets(airline_ts, model = spec, damped = TRUE)
      results <- rbind(results, data.frame(
        model = fit_d$method, damped = TRUE,
        AIC = fit_d$aic, BIC = fit_d$bic, AICc = fit_d$aicc
      ))
    }, error = function(e) {})
  }
}

results <- results[order(results$AICc), ]
cat("  Top 10 modelos por AICc:\n")
for (i in 1:min(10, nrow(results))) {
  cat(sprintf("    %2d. %-15s AICc=%.2f  AIC=%.2f  BIC=%.2f\n",
              i, results$model[i], results$AICc[i], results$AIC[i], results$BIC[i]))
}
cat("\n")

# =============================================================================
# 5. Forecast de longo prazo (h=36) - ETS vs Theta
# =============================================================================
cat("=== Comparacao ETS vs Theta (h=36) ===\n\n")

fcast_auto_36 <- forecast(fit_auto_aicc, h = 36)
theta_36 <- thetaf(airline_ts, h = 36)

cat("  h=12:\n")
cat(sprintf("    Auto ETS: %.2f\n", fcast_auto_36$mean[12]))
cat(sprintf("    Theta:    %.2f\n", theta_36$mean[12]))
cat("  h=24:\n")
cat(sprintf("    Auto ETS: %.2f\n", fcast_auto_36$mean[24]))
cat(sprintf("    Theta:    %.2f\n", theta_36$mean[24]))
cat("  h=36:\n")
cat(sprintf("    Auto ETS: %.2f\n", fcast_auto_36$mean[36]))
cat(sprintf("    Theta:    %.2f\n\n", theta_36$mean[36]))

# =============================================================================
# 6. Salvar resultados
# =============================================================================
cat("Salvando resultados...\n")

# Auto ETS coeficientes
auto_coef <- data.frame(
  dataset = c("airline", "synthetic"),
  model_selected = c(fit_auto_aicc$method, fit_syn_auto$method),
  alpha = c(fit_auto_aicc$par["alpha"], fit_syn_auto$par["alpha"]),
  beta = c(ifelse("beta" %in% names(fit_auto_aicc$par), fit_auto_aicc$par["beta"], NA),
           ifelse("beta" %in% names(fit_syn_auto$par), fit_syn_auto$par["beta"], NA)),
  gamma = c(ifelse("gamma" %in% names(fit_auto_aicc$par), fit_auto_aicc$par["gamma"], NA),
            ifelse("gamma" %in% names(fit_syn_auto$par), fit_syn_auto$par["gamma"], NA)),
  phi = c(ifelse("phi" %in% names(fit_auto_aicc$par), fit_auto_aicc$par["phi"], NA),
          ifelse("phi" %in% names(fit_syn_auto$par), fit_syn_auto$par["phi"], NA)),
  AIC = c(fit_auto_aicc$aic, fit_syn_auto$aic),
  BIC = c(fit_auto_aicc$bic, fit_syn_auto$bic),
  AICc = c(fit_auto_aicc$aicc, fit_syn_auto$aicc)
)
write.csv(auto_coef, file.path(output_dir, "auto_ets_coefficients.csv"), row.names = FALSE)
cat("  -> auto_ets_coefficients.csv\n")

# Comparacao de criterios de selecao
ic_compare <- data.frame(
  criterion = c("AIC", "BIC", "AICc"),
  model_selected = c(fit_auto_aic$method, fit_auto_bic$method, fit_auto_aicc$method),
  value = c(fit_auto_aic$aic, fit_auto_bic$bic, fit_auto_aicc$aicc)
)
write.csv(ic_compare, file.path(output_dir, "auto_ets_ic_comparison.csv"), row.names = FALSE)
cat("  -> auto_ets_ic_comparison.csv\n")

# Ranking completo
write.csv(results, file.path(output_dir, "auto_ets_ranking.csv"), row.names = FALSE)
cat("  -> auto_ets_ranking.csv\n")

# Theta forecasts
theta_df <- data.frame(
  h = 1:24,
  theta_airline = as.numeric(theta_full$mean),
  theta_synthetic = as.numeric(theta_syn$mean),
  auto_ets_airline = as.numeric(forecast(fit_auto_aicc, h = 24)$mean)
)
write.csv(theta_df, file.path(output_dir, "theta_forecasts.csv"), row.names = FALSE)
cat("  -> theta_forecasts.csv\n")

# Out-of-sample metricas
oos_metrics <- data.frame(
  method = c("auto_ets", "theta"),
  dataset = rep("airline", 2),
  RMSE = c(acc_oos["Test set", "RMSE"], theta_rmse),
  MAE = c(acc_oos["Test set", "MAE"], theta_mae),
  MAPE = c(acc_oos["Test set", "MAPE"], theta_mape)
)
write.csv(oos_metrics, file.path(output_dir, "auto_theta_oos_metrics.csv"), row.names = FALSE)
cat("  -> auto_theta_oos_metrics.csv\n")

# Forecasts h=36
fcast_36_df <- data.frame(
  h = 1:36,
  auto_ets = as.numeric(fcast_auto_36$mean),
  theta = as.numeric(theta_36$mean)
)
write.csv(fcast_36_df, file.path(output_dir, "auto_theta_forecasts_h36.csv"), row.names = FALSE)
cat("  -> auto_theta_forecasts_h36.csv\n")

cat("\n=== Concluido! ===\n")
cat("Tolerancia esperada: < 1e-3 para parametros de suavizacao\n")
cat("Nota: thetaf() implementa o metodo Theta de Assimakopoulos & Nikolopoulos (2000)\n")
