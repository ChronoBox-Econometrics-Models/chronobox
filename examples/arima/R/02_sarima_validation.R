# ==============================================================================
# 02_sarima_validation.R
# Validacao cruzada: SARIMA com forecast::Arima()
#
# Objetivo: Reproduzir os modelos SARIMA ajustados pelo chronobox (Python)
#           usando forecast::Arima() com componentes sazonais.
#
# Datasets: airline.csv, brazil_ipca.csv
# Modelos:  SARIMA(0,1,1)(0,1,1)[12] e variantes
#
# Dependencias: forecast
# ==============================================================================

# --- Configuracao inicial -----------------------------------------------------
set.seed(42)

library(forecast)

# Caminhos relativos (assumindo execucao a partir de examples/arima/)
data_dir    <- file.path("..", "data")
output_dir  <- file.path("..", "outputs", "R")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# --- Funcao auxiliar ----------------------------------------------------------
extract_sarima <- function(fit, dataset_name, model_name) {
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

# Airline passengers (log-transformado)
airline_raw <- read.csv(file.path(data_dir, "airline.csv"))
airline_ts  <- ts(log(airline_raw$passengers), start = c(1949, 1), frequency = 12)
cat(sprintf("airline.csv: %d obs (log)\n", length(airline_ts)))

# IPCA
ipca_raw <- read.csv(file.path(data_dir, "brazil_ipca.csv"))
ipca_ts  <- ts(ipca_raw$ipca, start = c(2004, 1), frequency = 12)
cat(sprintf("brazil_ipca.csv: %d obs\n", length(ipca_ts)))

# ==============================================================================
# SECAO 2: SARIMA para Airline
# ==============================================================================

cat("\n=== SARIMA para Airline (log) ===\n")

all_results <- list()

# ARIMA(1,1,1) sem sazonalidade (baseline para comparacao de AIC)
fit_air_noseas <- Arima(airline_ts, order = c(1, 1, 1), method = "css-ml")
cat(sprintf("Airline ARIMA(1,1,1):              AIC = %.4f\n", fit_air_noseas$aic))

# SARIMA(0,1,1)(0,1,1)[12] - modelo classico de Box-Jenkins para airline
fit_air_s011 <- Arima(airline_ts,
                      order    = c(0, 1, 1),
                      seasonal = list(order = c(0, 1, 1), period = 12),
                      method   = "css-ml")
cat(sprintf("Airline SARIMA(0,1,1)(0,1,1)[12]:  AIC = %.4f\n", fit_air_s011$aic))
all_results[["airline_s011"]] <- extract_sarima(fit_air_s011, "airline",
                                                "SARIMA(0,1,1)(0,1,1)[12]")

# Melhoria de AIC com sazonalidade
cat(sprintf("  Ganho de AIC: %.2f (sazonal vs nao-sazonal)\n",
            fit_air_noseas$aic - fit_air_s011$aic))

# Diagnostico
cat("\nDiagnostico de residuos - SARIMA(0,1,1)(0,1,1)[12]:\n")
lb <- Box.test(residuals(fit_air_s011), lag = 24, type = "Ljung-Box")
cat(sprintf("  Ljung-Box(24) p-value: %.4f\n", lb$p.value))

# ==============================================================================
# SECAO 3: SARIMA para IPCA
# ==============================================================================

cat("\n=== SARIMA para IPCA ===\n")

# SARIMA(1,1,0)(1,1,0)[12]
fit_ipca_s1 <- Arima(ipca_ts,
                     order    = c(1, 1, 0),
                     seasonal = list(order = c(1, 1, 0), period = 12),
                     method   = "css-ml")
cat(sprintf("IPCA SARIMA(1,1,0)(1,1,0)[12]: AIC = %.4f\n", fit_ipca_s1$aic))
all_results[["ipca_s110"]] <- extract_sarima(fit_ipca_s1, "ipca",
                                             "SARIMA(1,1,0)(1,1,0)[12]")

# SARIMA(0,1,1)(0,1,1)[12]
fit_ipca_s2 <- Arima(ipca_ts,
                     order    = c(0, 1, 1),
                     seasonal = list(order = c(0, 1, 1), period = 12),
                     method   = "css-ml")
cat(sprintf("IPCA SARIMA(0,1,1)(0,1,1)[12]: AIC = %.4f\n", fit_ipca_s2$aic))
all_results[["ipca_s011"]] <- extract_sarima(fit_ipca_s2, "ipca",
                                             "SARIMA(0,1,1)(0,1,1)[12]")

# SARIMA(1,1,1)(0,1,1)[12]
fit_ipca_s3 <- Arima(ipca_ts,
                     order    = c(1, 1, 1),
                     seasonal = list(order = c(0, 1, 1), period = 12),
                     method   = "css-ml")
cat(sprintf("IPCA SARIMA(1,1,1)(0,1,1)[12]: AIC = %.4f\n", fit_ipca_s3$aic))
all_results[["ipca_s111"]] <- extract_sarima(fit_ipca_s3, "ipca",
                                             "SARIMA(1,1,1)(0,1,1)[12]")

# SARIMA(1,1,0)(0,1,1)[12]
fit_ipca_s4 <- Arima(ipca_ts,
                     order    = c(1, 1, 0),
                     seasonal = list(order = c(0, 1, 1), period = 12),
                     method   = "css-ml")
cat(sprintf("IPCA SARIMA(1,1,0)(0,1,1)[12]: AIC = %.4f\n", fit_ipca_s4$aic))
all_results[["ipca_s100_011"]] <- extract_sarima(fit_ipca_s4, "ipca",
                                                 "SARIMA(1,1,0)(0,1,1)[12]")

# SARIMA(0,1,1)(1,1,1)[12]
fit_ipca_s5 <- Arima(ipca_ts,
                     order    = c(0, 1, 1),
                     seasonal = list(order = c(1, 1, 1), period = 12),
                     method   = "css-ml")
cat(sprintf("IPCA SARIMA(0,1,1)(1,1,1)[12]: AIC = %.4f\n", fit_ipca_s5$aic))
all_results[["ipca_s011_111"]] <- extract_sarima(fit_ipca_s5, "ipca",
                                                 "SARIMA(0,1,1)(1,1,1)[12]")

# ==============================================================================
# SECAO 4: Airline sem transformacao log (para documentar heterocedasticidade)
# ==============================================================================

cat("\n=== Airline sem log (heterocedasticidade) ===\n")

airline_nolog <- ts(airline_raw$passengers, start = c(1949, 1), frequency = 12)
fit_air_nolog <- Arima(airline_nolog,
                       order    = c(0, 1, 1),
                       seasonal = list(order = c(0, 1, 1), period = 12),
                       method   = "css-ml")
cat(sprintf("Airline (sem log) SARIMA(0,1,1)(0,1,1)[12]: AIC = %.4f\n",
            fit_air_nolog$aic))
cat("  Nota: Residuos apresentam heterocedasticidade sem log.\n")

# ==============================================================================
# SECAO 5: Previsao IPCA 12 meses (melhor modelo sazonal)
# ==============================================================================

cat("\n=== Previsao IPCA 12 meses - SARIMA(0,1,1)(0,1,1)[12] ===\n")

fc_ipca <- forecast(fit_ipca_s2, h = 12, level = 95)
cat("Previsoes pontuais:\n")
print(round(as.numeric(fc_ipca$mean), 6))

# ==============================================================================
# SECAO 6: Salvar resultados
# ==============================================================================

cat("\n=== Salvando resultados ===\n")

# Combinar todos os coeficientes SARIMA
sarima_df <- do.call(rbind, all_results)
rownames(sarima_df) <- NULL

write.csv(sarima_df, file.path(output_dir, "sarima_results_R.csv"),
          row.names = FALSE)
cat("Salvo: sarima_results_R.csv\n")

# ==============================================================================
# SECAO 7: Tabela comparativa de AIC
# ==============================================================================

cat("\n=== Comparacao de AIC ===\n")
aic_table <- data.frame(
  dataset = c("airline", "airline", "ipca", "ipca", "ipca", "ipca", "ipca"),
  model   = c("ARIMA(1,1,1)", "SARIMA(0,1,1)(0,1,1)[12]",
              "SARIMA(1,1,0)(1,1,0)[12]", "SARIMA(0,1,1)(0,1,1)[12]",
              "SARIMA(1,1,1)(0,1,1)[12]", "SARIMA(1,1,0)(0,1,1)[12]",
              "SARIMA(0,1,1)(1,1,1)[12]"),
  aic     = c(fit_air_noseas$aic, fit_air_s011$aic,
              fit_ipca_s1$aic, fit_ipca_s2$aic, fit_ipca_s3$aic,
              fit_ipca_s4$aic, fit_ipca_s5$aic),
  stringsAsFactors = FALSE
)
print(aic_table)

cat("\n=== Script 02 concluido com sucesso ===\n")
