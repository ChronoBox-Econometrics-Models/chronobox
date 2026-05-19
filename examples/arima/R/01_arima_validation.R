# ==============================================================================
# 01_arima_validation.R
# Validacao cruzada: ARIMA com forecast::Arima()
#
# Objetivo: Reproduzir os modelos ARIMA ajustados pelo chronobox (Python)
#           usando o pacote forecast do R como referencia.
#
# Datasets: airline.csv (log-passengers), nile.csv (flow)
# Modelos:  ARIMA(1,1,1) e ARIMA(2,1,2) para airline
#           ARIMA(1,1,0), ARIMA(0,1,1), ARIMA(1,1,1), ARIMA(2,1,1) para nile
#
# Dependencias: forecast, tseries
# ==============================================================================

# --- Configuracao inicial -----------------------------------------------------
set.seed(42)

library(forecast)
library(tseries)

# Caminhos relativos (assumindo execucao a partir de examples/arima/)
data_dir    <- file.path("..", "data")
output_dir  <- file.path("..", "outputs", "R")

# Criar diretorio de saida se nao existir
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# --- Funcao auxiliar para extrair coeficientes --------------------------------
# Extrai coeficientes, sigma2, AIC, BIC e log-likelihood de um modelo Arima
extract_results <- function(fit, dataset_name, model_name) {
  coefs <- coef(fit)
  data.frame(
    dataset   = dataset_name,
    model     = model_name,
    param     = names(coefs),
    value     = as.numeric(coefs),
    sigma2    = fit$sigma2,
    aic       = fit$aic,
    bic       = BIC(fit),
    loglik    = fit$loglik,
    stringsAsFactors = FALSE
  )
}

# ==============================================================================
# SECAO 1: Carregar dados
# ==============================================================================

cat("=== Carregando datasets ===\n")

# Airline passengers (aplicamos log para estabilizar variancia, como no Python)
airline_raw <- read.csv(file.path(data_dir, "airline.csv"))
airline_ts  <- ts(log(airline_raw$passengers), start = c(1949, 1), frequency = 12)
cat(sprintf("airline.csv: %d observacoes (log-transformado)\n", length(airline_ts)))

# Nile river flow
nile_raw <- read.csv(file.path(data_dir, "nile.csv"))
nile_ts  <- ts(nile_raw$flow, start = 1871, frequency = 1)
cat(sprintf("nile.csv: %d observacoes\n", length(nile_ts)))

# ==============================================================================
# SECAO 2: Ajustar modelos ARIMA para Nile
# ==============================================================================

cat("\n=== Modelos ARIMA para Nile ===\n")

# Lista para armazenar resultados
all_coefs <- list()

# ARIMA(1,1,0)
fit_nile_110 <- Arima(nile_ts, order = c(1, 1, 0), method = "css-ml")
cat(sprintf("Nile ARIMA(1,1,0): AIC = %.4f\n", fit_nile_110$aic))
all_coefs[["nile_110"]] <- extract_results(fit_nile_110, "nile", "ARIMA(1,1,0)")

# ARIMA(0,1,1)
fit_nile_011 <- Arima(nile_ts, order = c(0, 1, 1), method = "css-ml")
cat(sprintf("Nile ARIMA(0,1,1): AIC = %.4f\n", fit_nile_011$aic))
all_coefs[["nile_011"]] <- extract_results(fit_nile_011, "nile", "ARIMA(0,1,1)")

# ARIMA(1,1,1) - modelo principal para comparacao
fit_nile_111 <- Arima(nile_ts, order = c(1, 1, 1), method = "css-ml")
cat(sprintf("Nile ARIMA(1,1,1): AIC = %.4f\n", fit_nile_111$aic))
all_coefs[["nile_111"]] <- extract_results(fit_nile_111, "nile", "ARIMA(1,1,1)")

# ARIMA(2,1,1)
fit_nile_211 <- Arima(nile_ts, order = c(2, 1, 1), method = "css-ml")
cat(sprintf("Nile ARIMA(2,1,1): AIC = %.4f\n", fit_nile_211$aic))
all_coefs[["nile_211"]] <- extract_results(fit_nile_211, "nile", "ARIMA(2,1,1)")

# ==============================================================================
# SECAO 3: Ajustar modelos ARIMA para Airline (log-passengers)
# ==============================================================================

cat("\n=== Modelos ARIMA para Airline (log) ===\n")

# ARIMA(1,1,0)
fit_air_110 <- Arima(airline_ts, order = c(1, 1, 0), method = "css-ml")
cat(sprintf("Airline ARIMA(1,1,0): AIC = %.4f\n", fit_air_110$aic))
all_coefs[["airline_110"]] <- extract_results(fit_air_110, "airline", "ARIMA(1,1,0)")

# ARIMA(0,1,1)
fit_air_011 <- Arima(airline_ts, order = c(0, 1, 1), method = "css-ml")
cat(sprintf("Airline ARIMA(0,1,1): AIC = %.4f\n", fit_air_011$aic))
all_coefs[["airline_011"]] <- extract_results(fit_air_011, "airline", "ARIMA(0,1,1)")

# ARIMA(1,1,1)
fit_air_111 <- Arima(airline_ts, order = c(1, 1, 1), method = "css-ml")
cat(sprintf("Airline ARIMA(1,1,1): AIC = %.4f\n", fit_air_111$aic))
all_coefs[["airline_111"]] <- extract_results(fit_air_111, "airline", "ARIMA(1,1,1)")

# ARIMA(2,1,1)
fit_air_211 <- Arima(airline_ts, order = c(2, 1, 1), method = "css-ml")
cat(sprintf("Airline ARIMA(2,1,1): AIC = %.4f\n", fit_air_211$aic))
all_coefs[["airline_211"]] <- extract_results(fit_air_211, "airline", "ARIMA(2,1,1)")

# ARIMA(2,1,2)
fit_air_212 <- Arima(airline_ts, order = c(2, 1, 2), method = "css-ml")
cat(sprintf("Airline ARIMA(2,1,2): AIC = %.4f\n", fit_air_212$aic))
all_coefs[["airline_212"]] <- extract_results(fit_air_212, "airline", "ARIMA(2,1,2)")

# ==============================================================================
# SECAO 4: Diagnostico de residuos
# ==============================================================================

cat("\n=== Diagnostico de residuos ===\n")

# Teste Ljung-Box para os modelos principais
models_diag <- list(
  "Nile ARIMA(1,1,1)"    = fit_nile_111,
  "Airline ARIMA(1,1,1)"  = fit_air_111,
  "Airline ARIMA(2,1,2)"  = fit_air_212
)

for (name in names(models_diag)) {
  fit <- models_diag[[name]]
  lb  <- Box.test(residuals(fit), lag = 10, type = "Ljung-Box")
  cat(sprintf("%s - Ljung-Box p-value: %.4f %s\n",
              name, lb$p.value,
              ifelse(lb$p.value > 0.05, "(residuos brancos)", "(correlacao detectada)")))
}

# ==============================================================================
# SECAO 5: Previsoes
# ==============================================================================

cat("\n=== Previsoes ===\n")

# Funcao para gerar dataframe de previsoes
make_forecast_df <- function(fit, dataset_name, model_name, h) {
  fc <- forecast(fit, h = h, level = 95)
  data.frame(
    dataset  = dataset_name,
    model    = model_name,
    step     = 1:h,
    forecast = as.numeric(fc$mean),
    lower_95 = as.numeric(fc$lower),
    upper_95 = as.numeric(fc$upper),
    stringsAsFactors = FALSE
  )
}

# Previsoes 10 passos para Nile ARIMA(1,1,1)
fc_nile <- make_forecast_df(fit_nile_111, "nile", "ARIMA(1,1,1)", 10)

# Previsoes 12 passos para Airline ARIMA(1,1,1) e ARIMA(2,1,2)
fc_air_111 <- make_forecast_df(fit_air_111, "airline", "ARIMA(1,1,1)", 12)
fc_air_212 <- make_forecast_df(fit_air_212, "airline", "ARIMA(2,1,2)", 12)

all_forecasts <- rbind(fc_nile, fc_air_111, fc_air_212)

cat(sprintf("Previsoes geradas: %d linhas\n", nrow(all_forecasts)))

# ==============================================================================
# SECAO 6: Salvar resultados
# ==============================================================================

cat("\n=== Salvando resultados ===\n")

# Combinar todos os coeficientes
coefs_df <- do.call(rbind, all_coefs)
rownames(coefs_df) <- NULL

# Salvar coeficientes
write.csv(coefs_df, file.path(output_dir, "arima_coefficients_R.csv"),
          row.names = FALSE)
cat("Salvo: arima_coefficients_R.csv\n")

# Salvar previsoes
write.csv(all_forecasts, file.path(output_dir, "arima_forecasts_R.csv"),
          row.names = FALSE)
cat("Salvo: arima_forecasts_R.csv\n")

# ==============================================================================
# SECAO 7: Resumo comparativo
# ==============================================================================

cat("\n=== Resumo dos coeficientes (R) ===\n")
cat("\nNile ARIMA(1,1,1):\n")
print(coef(fit_nile_111))
cat(sprintf("  sigma2: %.6f, AIC: %.4f\n", fit_nile_111$sigma2, fit_nile_111$aic))

cat("\nAirline ARIMA(1,1,1):\n")
print(coef(fit_air_111))
cat(sprintf("  sigma2: %.6f, AIC: %.4f\n", fit_air_111$sigma2, fit_air_111$aic))

cat("\nAirline ARIMA(2,1,2):\n")
print(coef(fit_air_212))
cat(sprintf("  sigma2: %.6f, AIC: %.4f\n", fit_air_212$sigma2, fit_air_212$aic))

cat("\n=== Script 01 concluido com sucesso ===\n")
