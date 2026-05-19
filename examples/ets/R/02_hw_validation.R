# =============================================================================
# 02_hw_validation.R
# Validacao cruzada: forecast::hw() e HoltWinters()
# Compara modelos aditivo, multiplicativo e damped com chronobox
# =============================================================================

library(forecast)

cat("=== Holt-Winters Validation Script (R) ===\n\n")

# --- Configuracao ---
set.seed(42)
output_dir <- file.path(dirname(getwd()), "outputs", "R")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

data_dir <- file.path(dirname(getwd()), "data")

# --- Carregar dados ---
cat("Carregando datasets...\n")

airline <- read.csv(file.path(data_dir, "airline.csv"))
airline_ts <- ts(airline$passengers, start = c(1949, 1), frequency = 12)

ipca <- read.csv(file.path(data_dir, "brazil_ipca.csv"))
ipca_ts <- ts(ipca$ipca, start = c(2004, 1), frequency = 12)

cat(sprintf("  Airline: %d observacoes\n", length(airline_ts)))
cat(sprintf("  IPCA: %d observacoes\n\n", length(ipca_ts)))

# =============================================================================
# 1. HoltWinters() - Funcao base do R
# =============================================================================
cat("=== HoltWinters() do stats ===\n\n")

# --- Aditivo ---
cat("--- HoltWinters Aditivo (airline) ---\n")
hw_add <- HoltWinters(airline_ts, seasonal = "additive")
cat(sprintf("  alpha = %.6f\n", hw_add$alpha))
cat(sprintf("  beta  = %.6f\n", hw_add$beta))
cat(sprintf("  gamma = %.6f\n", hw_add$gamma))
cat("  Coeficientes sazonais:\n")
for (i in 1:12) {
  cat(sprintf("    s%02d = %.4f\n", i, hw_add$coefficients[paste0("s", i)]))
}

# Metricas in-sample (HoltWinters fitted comeca apos m periodos)
hw_add_fitted <- fitted(hw_add)[, "xhat"]
hw_add_actual <- airline_ts[-(1:12)]
hw_add_resid <- hw_add_actual - hw_add_fitted
cat(sprintf("\n  RMSE = %.4f\n", sqrt(mean(hw_add_resid^2))))
cat(sprintf("  MAE  = %.4f\n", mean(abs(hw_add_resid))))
cat(sprintf("  MAPE = %.4f%%\n\n", mean(abs(hw_add_resid / hw_add_actual)) * 100))

# Forecast h=24 e h=36
fcast_hw_add_24 <- predict(hw_add, n.ahead = 24)
fcast_hw_add_36 <- predict(hw_add, n.ahead = 36)

# --- Multiplicativo ---
cat("--- HoltWinters Multiplicativo (airline) ---\n")
hw_mul <- HoltWinters(airline_ts, seasonal = "multiplicative")
cat(sprintf("  alpha = %.6f\n", hw_mul$alpha))
cat(sprintf("  beta  = %.6f\n", hw_mul$beta))
cat(sprintf("  gamma = %.6f\n", hw_mul$gamma))
cat("  Coeficientes sazonais:\n")
for (i in 1:12) {
  cat(sprintf("    s%02d = %.4f\n", i, hw_mul$coefficients[paste0("s", i)]))
}

hw_mul_fitted <- fitted(hw_mul)[, "xhat"]
hw_mul_actual <- airline_ts[-(1:12)]
hw_mul_resid <- hw_mul_actual - hw_mul_fitted
cat(sprintf("\n  RMSE = %.4f\n", sqrt(mean(hw_mul_resid^2))))
cat(sprintf("  MAE  = %.4f\n", mean(abs(hw_mul_resid))))
cat(sprintf("  MAPE = %.4f%%\n\n", mean(abs(hw_mul_resid / hw_mul_actual)) * 100))

fcast_hw_mul_24 <- predict(hw_mul, n.ahead = 24)
fcast_hw_mul_36 <- predict(hw_mul, n.ahead = 36)

# =============================================================================
# 2. forecast::hw() - Wrapper com ETS
# =============================================================================
cat("=== forecast::hw() ===\n\n")

# --- Aditivo ---
cat("--- hw() Aditivo (airline) ---\n")
hw_f_add <- hw(airline_ts, seasonal = "additive", h = 24)
cat(sprintf("  alpha = %.6f\n", hw_f_add$model$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", hw_f_add$model$par["beta"]))
cat(sprintf("  gamma = %.6f\n", hw_f_add$model$par["gamma"]))
cat(sprintf("  AIC   = %.4f\n", hw_f_add$model$aic))
cat(sprintf("  BIC   = %.4f\n", hw_f_add$model$bic))
cat(sprintf("  AICc  = %.4f\n", hw_f_add$model$aicc))

acc_hw_add <- accuracy(hw_f_add)
cat(sprintf("  RMSE  = %.4f\n", acc_hw_add[, "RMSE"]))
cat(sprintf("  MAE   = %.4f\n", acc_hw_add[, "MAE"]))
cat(sprintf("  MAPE  = %.4f%%\n\n", acc_hw_add[, "MAPE"]))

# --- Multiplicativo ---
cat("--- hw() Multiplicativo (airline) ---\n")
hw_f_mul <- hw(airline_ts, seasonal = "multiplicative", h = 24)
cat(sprintf("  alpha = %.6f\n", hw_f_mul$model$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", hw_f_mul$model$par["beta"]))
cat(sprintf("  gamma = %.6f\n", hw_f_mul$model$par["gamma"]))
cat(sprintf("  AIC   = %.4f\n", hw_f_mul$model$aic))
cat(sprintf("  BIC   = %.4f\n", hw_f_mul$model$bic))
cat(sprintf("  AICc  = %.4f\n", hw_f_mul$model$aicc))

acc_hw_mul <- accuracy(hw_f_mul)
cat(sprintf("  RMSE  = %.4f\n", acc_hw_mul[, "RMSE"]))
cat(sprintf("  MAE   = %.4f\n", acc_hw_mul[, "MAE"]))
cat(sprintf("  MAPE  = %.4f%%\n\n", acc_hw_mul[, "MAPE"]))

# --- Damped Aditivo ---
cat("--- hw() Damped Aditivo (airline) ---\n")
hw_f_damp <- hw(airline_ts, seasonal = "additive", damped = TRUE, h = 24)
cat(sprintf("  alpha = %.6f\n", hw_f_damp$model$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", hw_f_damp$model$par["beta"]))
cat(sprintf("  gamma = %.6f\n", hw_f_damp$model$par["gamma"]))
cat(sprintf("  phi   = %.6f\n", hw_f_damp$model$par["phi"]))
cat(sprintf("  AIC   = %.4f\n", hw_f_damp$model$aic))
cat(sprintf("  BIC   = %.4f\n", hw_f_damp$model$bic))
cat(sprintf("  AICc  = %.4f\n", hw_f_damp$model$aicc))

acc_hw_damp <- accuracy(hw_f_damp)
cat(sprintf("  RMSE  = %.4f\n", acc_hw_damp[, "RMSE"]))
cat(sprintf("  MAE   = %.4f\n", acc_hw_damp[, "MAE"]))
cat(sprintf("  MAPE  = %.4f%%\n\n", acc_hw_damp[, "MAPE"]))

# =============================================================================
# 3. ETS(A,A,A) e ETS(A,Ad,A) - forecast::ets()
# =============================================================================
cat("=== ETS com trend linear e damped ===\n\n")

# Nao-damped
cat("--- ETS(A,A,A) Non-damped (airline) ---\n")
fit_aaa <- ets(airline_ts, model = "AAA", damped = FALSE)
cat(sprintf("  alpha = %.6f\n", fit_aaa$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", fit_aaa$par["beta"]))
cat(sprintf("  gamma = %.6f\n", fit_aaa$par["gamma"]))
cat(sprintf("  AIC   = %.4f\n", fit_aaa$aic))
cat(sprintf("  AICc  = %.4f\n", fit_aaa$aicc))

acc_aaa <- accuracy(fit_aaa)
cat(sprintf("  RMSE  = %.4f\n", acc_aaa[, "RMSE"]))
cat(sprintf("  MAE   = %.4f\n", acc_aaa[, "MAE"]))
cat(sprintf("  MAPE  = %.4f%%\n\n", acc_aaa[, "MAPE"]))

fcast_aaa_12 <- forecast(fit_aaa, h = 12)
fcast_aaa_36 <- forecast(fit_aaa, h = 36)

# Damped
cat("--- ETS(A,Ad,A) Damped (airline) ---\n")
fit_aada <- ets(airline_ts, model = "AAA", damped = TRUE)
cat(sprintf("  alpha = %.6f\n", fit_aada$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", fit_aada$par["beta"]))
cat(sprintf("  gamma = %.6f\n", fit_aada$par["gamma"]))
cat(sprintf("  phi   = %.6f\n", fit_aada$par["phi"]))
cat(sprintf("  AIC   = %.4f\n", fit_aada$aic))
cat(sprintf("  AICc  = %.4f\n", fit_aada$aicc))

acc_aada <- accuracy(fit_aada)
cat(sprintf("  RMSE  = %.4f\n", acc_aada[, "RMSE"]))
cat(sprintf("  MAE   = %.4f\n", acc_aada[, "MAE"]))
cat(sprintf("  MAPE  = %.4f%%\n\n", acc_aada[, "MAPE"]))

fcast_aada_12 <- forecast(fit_aada, h = 12)
fcast_aada_36 <- forecast(fit_aada, h = 36)

# =============================================================================
# 4. HoltWinters no IPCA
# =============================================================================
cat("=== Holt-Winters no IPCA ===\n\n")

cat("--- hw() Aditivo (IPCA) ---\n")
hw_ipca_add <- hw(ipca_ts, seasonal = "additive", h = 12)
cat(sprintf("  alpha = %.6f\n", hw_ipca_add$model$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", hw_ipca_add$model$par["beta"]))
cat(sprintf("  gamma = %.6f\n", hw_ipca_add$model$par["gamma"]))

acc_ipca_add <- accuracy(hw_ipca_add)
cat(sprintf("  RMSE  = %.4f\n", acc_ipca_add[, "RMSE"]))
cat(sprintf("  MAE   = %.4f\n", acc_ipca_add[, "MAE"]))
cat(sprintf("  MAPE  = %.4f%%\n\n", acc_ipca_add[, "MAPE"]))

cat("--- hw() Multiplicativo (IPCA) ---\n")
hw_ipca_mul <- hw(ipca_ts, seasonal = "multiplicative", h = 12)
cat(sprintf("  alpha = %.6f\n", hw_ipca_mul$model$par["alpha"]))
cat(sprintf("  beta  = %.6f\n", hw_ipca_mul$model$par["beta"]))
cat(sprintf("  gamma = %.6f\n", hw_ipca_mul$model$par["gamma"]))

acc_ipca_mul <- accuracy(hw_ipca_mul)
cat(sprintf("  RMSE  = %.4f\n", acc_ipca_mul[, "RMSE"]))
cat(sprintf("  MAE   = %.4f\n", acc_ipca_mul[, "MAE"]))
cat(sprintf("  MAPE  = %.4f%%\n\n", acc_ipca_mul[, "MAPE"]))

# =============================================================================
# 5. Salvar resultados
# =============================================================================
cat("Salvando resultados...\n")

# Coeficientes HoltWinters base R
hw_base_df <- data.frame(
  method = c("HoltWinters_add", "HoltWinters_mul"),
  dataset = rep("airline", 2),
  alpha = c(hw_add$alpha, hw_mul$alpha),
  beta = c(hw_add$beta, hw_mul$beta),
  gamma = c(hw_add$gamma, hw_mul$gamma),
  phi = c(NA, NA)
)
write.csv(hw_base_df, file.path(output_dir, "hw_base_coefficients.csv"), row.names = FALSE)
cat("  -> hw_base_coefficients.csv\n")

# Coeficientes forecast::hw()
hw_forecast_df <- data.frame(
  method = c("hw_additive", "hw_multiplicative", "hw_damped_additive"),
  dataset = rep("airline", 3),
  alpha = c(hw_f_add$model$par["alpha"], hw_f_mul$model$par["alpha"],
            hw_f_damp$model$par["alpha"]),
  beta = c(hw_f_add$model$par["beta"], hw_f_mul$model$par["beta"],
           hw_f_damp$model$par["beta"]),
  gamma = c(hw_f_add$model$par["gamma"], hw_f_mul$model$par["gamma"],
            hw_f_damp$model$par["gamma"]),
  phi = c(NA, NA, hw_f_damp$model$par["phi"]),
  AIC = c(hw_f_add$model$aic, hw_f_mul$model$aic, hw_f_damp$model$aic),
  BIC = c(hw_f_add$model$bic, hw_f_mul$model$bic, hw_f_damp$model$bic),
  AICc = c(hw_f_add$model$aicc, hw_f_mul$model$aicc, hw_f_damp$model$aicc)
)
write.csv(hw_forecast_df, file.path(output_dir, "hw_forecast_coefficients.csv"), row.names = FALSE)
cat("  -> hw_forecast_coefficients.csv\n")

# Metricas hw()
hw_metrics_df <- data.frame(
  method = c("hw_additive", "hw_multiplicative", "hw_damped_additive",
             "ETS_AAA", "ETS_AAdA"),
  dataset = rep("airline", 5),
  RMSE = c(acc_hw_add[, "RMSE"], acc_hw_mul[, "RMSE"], acc_hw_damp[, "RMSE"],
           acc_aaa[, "RMSE"], acc_aada[, "RMSE"]),
  MAE = c(acc_hw_add[, "MAE"], acc_hw_mul[, "MAE"], acc_hw_damp[, "MAE"],
          acc_aaa[, "MAE"], acc_aada[, "MAE"]),
  MAPE = c(acc_hw_add[, "MAPE"], acc_hw_mul[, "MAPE"], acc_hw_damp[, "MAPE"],
           acc_aaa[, "MAPE"], acc_aada[, "MAPE"])
)
write.csv(hw_metrics_df, file.path(output_dir, "hw_metrics.csv"), row.names = FALSE)
cat("  -> hw_metrics.csv\n")

# Forecasts h=24 de hw()
hw_fcast_df <- data.frame(
  h = 1:24,
  hw_additive = as.numeric(hw_f_add$mean),
  hw_multiplicative = as.numeric(hw_f_mul$mean),
  hw_damped_additive = as.numeric(hw_f_damp$mean)
)
write.csv(hw_fcast_df, file.path(output_dir, "hw_forecasts.csv"), row.names = FALSE)
cat("  -> hw_forecasts.csv\n")

# Forecasts ETS h=12 e h=36
ets_fcast_df <- data.frame(
  h = 1:36,
  ETS_AAA = as.numeric(fcast_aaa_36$mean),
  ETS_AAdA = as.numeric(fcast_aada_36$mean)
)
write.csv(ets_fcast_df, file.path(output_dir, "hw_ets_forecasts_h36.csv"), row.names = FALSE)
cat("  -> hw_ets_forecasts_h36.csv\n")

# Fitted values
hw_fitted_df <- data.frame(
  date = airline$date,
  actual = as.numeric(airline_ts),
  hw_add_fitted = c(rep(NA, 12), as.numeric(hw_add_fitted)),
  hw_mul_fitted = c(rep(NA, 12), as.numeric(hw_mul_fitted)),
  ets_aaa_fitted = as.numeric(fitted(fit_aaa)),
  ets_aada_fitted = as.numeric(fitted(fit_aada))
)
write.csv(hw_fitted_df, file.path(output_dir, "hw_fitted_values.csv"), row.names = FALSE)
cat("  -> hw_fitted_values.csv\n")

# IPCA
ipca_metrics_df <- data.frame(
  method = c("hw_additive", "hw_multiplicative"),
  dataset = rep("ipca", 2),
  alpha = c(hw_ipca_add$model$par["alpha"], hw_ipca_mul$model$par["alpha"]),
  beta = c(hw_ipca_add$model$par["beta"], hw_ipca_mul$model$par["beta"]),
  gamma = c(hw_ipca_add$model$par["gamma"], hw_ipca_mul$model$par["gamma"]),
  RMSE = c(acc_ipca_add[, "RMSE"], acc_ipca_mul[, "RMSE"]),
  MAE = c(acc_ipca_add[, "MAE"], acc_ipca_mul[, "MAE"]),
  MAPE = c(acc_ipca_add[, "MAPE"], acc_ipca_mul[, "MAPE"])
)
write.csv(ipca_metrics_df, file.path(output_dir, "hw_ipca_results.csv"), row.names = FALSE)
cat("  -> hw_ipca_results.csv\n")

cat("\n=== Concluido! ===\n")
cat("Tolerancia esperada: < 1e-3 para parametros de suavizacao\n")
cat("Nota: HoltWinters() do base R usa otimizacao diferente de forecast::hw()\n")
