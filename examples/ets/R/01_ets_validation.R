# =============================================================================
# 01_ets_validation.R
# Validacao cruzada: forecast::ets() com modelos especificos
# Compara parametros de suavizacao com chronobox
# =============================================================================

library(forecast)

cat("=== ETS Validation Script (R) ===\n\n")

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

# =============================================================================
# 1. ETS(A,N,N) - Simple Exponential Smoothing
# =============================================================================
cat("--- Modelo ETS(A,N,N) - SES ---\n")

fit_ann <- ets(airline_ts, model = "ANN")
cat(sprintf("  alpha = %.6f\n", fit_ann$par["alpha"]))
cat(sprintf("  l0    = %.4f\n", fit_ann$initstate[1]))
cat(sprintf("  AIC   = %.4f\n", fit_ann$aic))
cat(sprintf("  BIC   = %.4f\n", fit_ann$bic))
cat(sprintf("  AICc  = %.4f\n", fit_ann$aicc))
cat(sprintf("  sigma2= %.6f\n", fit_ann$sigma2))
cat(sprintf("  loglik= %.4f\n\n", fit_ann$loglik))

# Metricas de erro
acc_ann <- accuracy(fit_ann)
cat("  Metricas in-sample:\n")
cat(sprintf("    RMSE = %.4f\n", acc_ann[, "RMSE"]))
cat(sprintf("    MAE  = %.4f\n", acc_ann[, "MAE"]))
cat(sprintf("    MAPE = %.4f%%\n\n", acc_ann[, "MAPE"]))

# Forecast h=24
fcast_ann <- forecast(fit_ann, h = 24)

# =============================================================================
# 2. ETS(A,A,N) - Holt's Linear Trend
# =============================================================================
cat("--- Modelo ETS(A,A,N) - Holt Linear ---\n")

fit_aan <- ets(airline_ts, model = "AAN")
cat(sprintf("  alpha = %.6f\n", fit_aan$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", fit_aan$par["beta"]))
cat(sprintf("  l0    = %.4f\n", fit_aan$initstate[1]))
cat(sprintf("  b0    = %.4f\n", fit_aan$initstate[2]))
cat(sprintf("  AIC   = %.4f\n", fit_aan$aic))
cat(sprintf("  BIC   = %.4f\n", fit_aan$bic))
cat(sprintf("  AICc  = %.4f\n", fit_aan$aicc))
cat(sprintf("  sigma2= %.6f\n", fit_aan$sigma2))
cat(sprintf("  loglik= %.4f\n\n", fit_aan$loglik))

acc_aan <- accuracy(fit_aan)
cat("  Metricas in-sample:\n")
cat(sprintf("    RMSE = %.4f\n", acc_aan[, "RMSE"]))
cat(sprintf("    MAE  = %.4f\n", acc_aan[, "MAE"]))
cat(sprintf("    MAPE = %.4f%%\n\n", acc_aan[, "MAPE"]))

fcast_aan <- forecast(fit_aan, h = 24)

# =============================================================================
# 3. ETS(A,A,A) - Holt-Winters Additive
# =============================================================================
cat("--- Modelo ETS(A,A,A) - HW Aditivo ---\n")

fit_aaa <- ets(airline_ts, model = "AAA")
cat(sprintf("  alpha = %.6f\n", fit_aaa$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", fit_aaa$par["beta"]))
cat(sprintf("  gamma = %.6f\n", fit_aaa$par["gamma"]))
cat(sprintf("  l0    = %.4f\n", fit_aaa$initstate[1]))
cat(sprintf("  b0    = %.4f\n", fit_aaa$initstate[2]))
cat(sprintf("  AIC   = %.4f\n", fit_aaa$aic))
cat(sprintf("  BIC   = %.4f\n", fit_aaa$bic))
cat(sprintf("  AICc  = %.4f\n", fit_aaa$aicc))
cat(sprintf("  sigma2= %.6f\n", fit_aaa$sigma2))
cat(sprintf("  loglik= %.4f\n\n", fit_aaa$loglik))

acc_aaa <- accuracy(fit_aaa)
cat("  Metricas in-sample:\n")
cat(sprintf("    RMSE = %.4f\n", acc_aaa[, "RMSE"]))
cat(sprintf("    MAE  = %.4f\n", acc_aaa[, "MAE"]))
cat(sprintf("    MAPE = %.4f%%\n\n", acc_aaa[, "MAPE"]))

# Seasonal components
cat("  Componentes sazonais iniciais (s1..s12):\n")
seasonal_init <- fit_aaa$initstate[3:14]
for (i in 1:12) {
  cat(sprintf("    s%02d = %.4f\n", i, seasonal_init[i]))
}
cat("\n")

fcast_aaa <- forecast(fit_aaa, h = 24)

# =============================================================================
# 4. ETS(M,A,M) - Multiplicative Error & Seasonality
# =============================================================================
cat("--- Modelo ETS(M,A,M) - Multiplicativo ---\n")

fit_mam <- ets(airline_ts, model = "MAM")
cat(sprintf("  alpha = %.6f\n", fit_mam$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", fit_mam$par["beta"]))
cat(sprintf("  gamma = %.6f\n", fit_mam$par["gamma"]))
cat(sprintf("  l0    = %.4f\n", fit_mam$initstate[1]))
cat(sprintf("  b0    = %.4f\n", fit_mam$initstate[2]))
cat(sprintf("  AIC   = %.4f\n", fit_mam$aic))
cat(sprintf("  BIC   = %.4f\n", fit_mam$bic))
cat(sprintf("  AICc  = %.4f\n", fit_mam$aicc))
cat(sprintf("  sigma2= %.6f\n", fit_mam$sigma2))
cat(sprintf("  loglik= %.4f\n\n", fit_mam$loglik))

acc_mam <- accuracy(fit_mam)
cat("  Metricas in-sample:\n")
cat(sprintf("    RMSE = %.4f\n", acc_mam[, "RMSE"]))
cat(sprintf("    MAE  = %.4f\n", acc_mam[, "MAE"]))
cat(sprintf("    MAPE = %.4f%%\n\n", acc_mam[, "MAPE"]))

fcast_mam <- forecast(fit_mam, h = 24)

# =============================================================================
# 5. ETS(M,Ad,M) - Damped Multiplicative
# =============================================================================
cat("--- Modelo ETS(M,Ad,M) - Damped Multiplicativo ---\n")

fit_madm <- ets(airline_ts, model = "MAM", damped = TRUE)
cat(sprintf("  alpha = %.6f\n", fit_madm$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", fit_madm$par["beta"]))
cat(sprintf("  gamma = %.6f\n", fit_madm$par["gamma"]))
cat(sprintf("  phi   = %.6f\n", fit_madm$par["phi"]))
cat(sprintf("  l0    = %.4f\n", fit_madm$initstate[1]))
cat(sprintf("  b0    = %.4f\n", fit_madm$initstate[2]))
cat(sprintf("  AIC   = %.4f\n", fit_madm$aic))
cat(sprintf("  BIC   = %.4f\n", fit_madm$bic))
cat(sprintf("  AICc  = %.4f\n", fit_madm$aicc))
cat(sprintf("  sigma2= %.6f\n", fit_madm$sigma2))
cat(sprintf("  loglik= %.4f\n\n", fit_madm$loglik))

acc_madm <- accuracy(fit_madm)
cat("  Metricas in-sample:\n")
cat(sprintf("    RMSE = %.4f\n", acc_madm[, "RMSE"]))
cat(sprintf("    MAE  = %.4f\n", acc_madm[, "MAE"]))
cat(sprintf("    MAPE = %.4f%%\n\n", acc_madm[, "MAPE"]))

fcast_madm <- forecast(fit_madm, h = 24)

# =============================================================================
# 6. Modelos no dataset sintetico (gerado como ETS(M,A,M))
# =============================================================================
cat("--- Modelos no dataset sintetico ---\n\n")

fit_syn_mam <- ets(synthetic_ts, model = "MAM")
cat("ETS(M,A,M) no sintetico:\n")
cat(sprintf("  alpha = %.6f\n", fit_syn_mam$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", fit_syn_mam$par["beta"]))
cat(sprintf("  gamma = %.6f\n", fit_syn_mam$par["gamma"]))
cat(sprintf("  AIC   = %.4f\n", fit_syn_mam$aic))
cat(sprintf("  AICc  = %.4f\n", fit_syn_mam$aicc))

acc_syn <- accuracy(fit_syn_mam)
cat(sprintf("  RMSE  = %.4f\n", acc_syn[, "RMSE"]))
cat(sprintf("  MAE   = %.4f\n", acc_syn[, "MAE"]))
cat(sprintf("  MAPE  = %.4f%%\n\n", acc_syn[, "MAPE"]))

# =============================================================================
# 7. Salvar resultados em CSV
# =============================================================================
cat("Salvando resultados...\n")

# Coeficientes de todos os modelos
coef_df <- data.frame(
  model = c("ANN", "AAN", "AAA", "MAM", "MAdM"),
  dataset = rep("airline", 5),
  alpha = c(fit_ann$par["alpha"], fit_aan$par["alpha"], fit_aaa$par["alpha"],
            fit_mam$par["alpha"], fit_madm$par["alpha"]),
  beta = c(NA, fit_aan$par["beta"], fit_aaa$par["beta"],
           fit_mam$par["beta"], fit_madm$par["beta"]),
  gamma = c(NA, NA, fit_aaa$par["gamma"],
            fit_mam$par["gamma"], fit_madm$par["gamma"]),
  phi = c(NA, NA, NA, NA, fit_madm$par["phi"]),
  AIC = c(fit_ann$aic, fit_aan$aic, fit_aaa$aic, fit_mam$aic, fit_madm$aic),
  BIC = c(fit_ann$bic, fit_aan$bic, fit_aaa$bic, fit_mam$bic, fit_madm$bic),
  AICc = c(fit_ann$aicc, fit_aan$aicc, fit_aaa$aicc, fit_mam$aicc, fit_madm$aicc),
  loglik = c(fit_ann$loglik, fit_aan$loglik, fit_aaa$loglik, fit_mam$loglik, fit_madm$loglik),
  sigma2 = c(fit_ann$sigma2, fit_aan$sigma2, fit_aaa$sigma2, fit_mam$sigma2, fit_madm$sigma2)
)
write.csv(coef_df, file.path(output_dir, "ets_coefficients.csv"), row.names = FALSE)
cat("  -> ets_coefficients.csv\n")

# Metricas de erro
metrics_df <- data.frame(
  model = c("ANN", "AAN", "AAA", "MAM", "MAdM"),
  dataset = rep("airline", 5),
  RMSE = c(acc_ann[, "RMSE"], acc_aan[, "RMSE"], acc_aaa[, "RMSE"],
           acc_mam[, "RMSE"], acc_madm[, "RMSE"]),
  MAE = c(acc_ann[, "MAE"], acc_aan[, "MAE"], acc_aaa[, "MAE"],
          acc_mam[, "MAE"], acc_madm[, "MAE"]),
  MAPE = c(acc_ann[, "MAPE"], acc_aan[, "MAPE"], acc_aaa[, "MAPE"],
           acc_mam[, "MAPE"], acc_madm[, "MAPE"])
)
write.csv(metrics_df, file.path(output_dir, "ets_metrics.csv"), row.names = FALSE)
cat("  -> ets_metrics.csv\n")

# Forecasts
fcast_df <- data.frame(
  h = 1:24,
  ANN = as.numeric(fcast_ann$mean),
  AAN = as.numeric(fcast_aan$mean),
  AAA = as.numeric(fcast_aaa$mean),
  MAM = as.numeric(fcast_mam$mean),
  MAdM = as.numeric(fcast_madm$mean)
)
write.csv(fcast_df, file.path(output_dir, "ets_forecasts.csv"), row.names = FALSE)
cat("  -> ets_forecasts.csv\n")

# Fitted values
fitted_df <- data.frame(
  date = airline$date,
  actual = as.numeric(airline_ts),
  ANN = as.numeric(fitted(fit_ann)),
  AAN = as.numeric(fitted(fit_aan)),
  AAA = as.numeric(fitted(fit_aaa)),
  MAM = as.numeric(fitted(fit_mam)),
  MAdM = as.numeric(fitted(fit_madm))
)
write.csv(fitted_df, file.path(output_dir, "ets_fitted_values.csv"), row.names = FALSE)
cat("  -> ets_fitted_values.csv\n")

# Sintetico
syn_coef <- data.frame(
  model = "MAM",
  dataset = "synthetic",
  alpha = fit_syn_mam$par["alpha"],
  beta = fit_syn_mam$par["beta"],
  gamma = fit_syn_mam$par["gamma"],
  phi = NA,
  AIC = fit_syn_mam$aic,
  BIC = fit_syn_mam$bic,
  AICc = fit_syn_mam$aicc,
  loglik = fit_syn_mam$loglik,
  sigma2 = fit_syn_mam$sigma2
)
write.csv(syn_coef, file.path(output_dir, "ets_synthetic_coefficients.csv"), row.names = FALSE)
cat("  -> ets_synthetic_coefficients.csv\n")

cat("\n=== Concluido! ===\n")
cat("Tolerancia esperada: < 1e-3 para parametros de suavizacao\n")
cat("Nota: Diferencas maiores podem ocorrer devido a otimizacao numerica\n")
