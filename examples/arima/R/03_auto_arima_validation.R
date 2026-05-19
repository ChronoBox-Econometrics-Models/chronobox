# ==============================================================================
# 03_auto_arima_validation.R
# Validacao cruzada: auto.arima() vs chronobox auto_arima
#
# Objetivo: Usar forecast::auto.arima() em todos os datasets e comparar
#           a ordem selecionada com os resultados do chronobox.
#
# Datasets: airline.csv, nile.csv, brazil_ipca.csv
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

# ==============================================================================
# SECAO 1: Carregar dados
# ==============================================================================

cat("=== Carregando datasets ===\n")

# Airline (log)
airline_raw <- read.csv(file.path(data_dir, "airline.csv"))
airline_ts  <- ts(log(airline_raw$passengers), start = c(1949, 1), frequency = 12)

# Nile
nile_raw <- read.csv(file.path(data_dir, "nile.csv"))
nile_ts  <- ts(nile_raw$flow, start = 1871, frequency = 1)

# IPCA
ipca_raw <- read.csv(file.path(data_dir, "brazil_ipca.csv"))
ipca_ts  <- ts(ipca_raw$ipca, start = c(2004, 1), frequency = 12)

cat(sprintf("Datasets carregados: airline(%d), nile(%d), ipca(%d)\n",
            length(airline_ts), length(nile_ts), length(ipca_ts)))

# ==============================================================================
# SECAO 2: auto.arima() para Nile
# ==============================================================================

cat("\n=== auto.arima() para Nile ===\n")

# Rodar auto.arima com criterio AIC (padrao)
auto_nile <- auto.arima(nile_ts, ic = "aic", stepwise = FALSE, approximation = FALSE)
cat("Modelo selecionado (AIC):\n")
print(auto_nile)

# Tambem testar com BIC
auto_nile_bic <- auto.arima(nile_ts, ic = "bic", stepwise = FALSE, approximation = FALSE)
cat("\nModelo selecionado (BIC):\n")
print(auto_nile_bic)

# ==============================================================================
# SECAO 3: auto.arima() para Airline
# ==============================================================================

cat("\n=== auto.arima() para Airline (log) ===\n")

auto_airline <- auto.arima(airline_ts, ic = "aic", stepwise = FALSE, approximation = FALSE)
cat("Modelo selecionado (AIC):\n")
print(auto_airline)

# ==============================================================================
# SECAO 4: auto.arima() para IPCA
# ==============================================================================

cat("\n=== auto.arima() para IPCA ===\n")

auto_ipca <- auto.arima(ipca_ts, ic = "aic", stepwise = FALSE, approximation = FALSE)
cat("Modelo selecionado (AIC):\n")
print(auto_ipca)

# ==============================================================================
# SECAO 5: Funcao para formatar nome do modelo
# ==============================================================================

# Extrai a string do modelo no formato ARIMA(p,d,q) ou ARIMA(p,d,q)(P,D,Q)[s]
format_model_name <- function(fit) {
  ord <- arimaorder(fit)
  p <- ord["p"]; d <- ord["d"]; q <- ord["q"]

  if ("P" %in% names(ord) && (ord["P"] > 0 || ord["D"] > 0 || ord["Q"] > 0)) {
    P <- ord["P"]; D <- ord["D"]; Q <- ord["Q"]
    s <- fit$arma[5]  # periodo sazonal
    return(sprintf("ARIMA(%d,%d,%d)(%d,%d,%d)[%d]", p, d, q, P, D, Q, s))
  } else {
    return(sprintf("ARIMA(%d,%d,%d)", p, d, q))
  }
}

# ==============================================================================
# SECAO 6: Montar tabela de comparacao
# ==============================================================================

cat("\n=== Tabela de comparacao ===\n")

# Resultados do R
results_r <- data.frame(
  dataset = c("nile", "nile", "airline", "ipca"),
  method_r = c("auto.arima(aic)", "auto.arima(bic)", "auto.arima(aic)", "auto.arima(aic)"),
  model_r  = c(format_model_name(auto_nile),
               format_model_name(auto_nile_bic),
               format_model_name(auto_airline),
               format_model_name(auto_ipca)),
  aic_r    = c(auto_nile$aic, auto_nile_bic$aic, auto_airline$aic, auto_ipca$aic),
  bic_r    = c(BIC(auto_nile), BIC(auto_nile_bic), BIC(auto_airline), BIC(auto_ipca)),
  aicc_r   = c(auto_nile$aicc, auto_nile_bic$aicc, auto_airline$aicc, auto_ipca$aicc),
  loglik_r = c(auto_nile$loglik, auto_nile_bic$loglik, auto_airline$loglik, auto_ipca$loglik),
  n_params_r = c(length(coef(auto_nile)) + 1,   # +1 para sigma2
                 length(coef(auto_nile_bic)) + 1,
                 length(coef(auto_airline)) + 1,
                 length(coef(auto_ipca)) + 1),
  stringsAsFactors = FALSE
)

# Resultados do Python (lidos do CSV de comparacao)
python_csv <- file.path("..", "outputs", "auto_arima_comparison.csv")
if (file.exists(python_csv)) {
  python_results <- read.csv(python_csv, stringsAsFactors = FALSE)
  cat("Resultados Python carregados para comparacao.\n")
  cat("\nPython auto_arima:\n")
  print(python_results)
} else {
  cat("AVISO: auto_arima_comparison.csv nao encontrado.\n")
}

cat("\nR auto.arima:\n")
print(results_r)

# ==============================================================================
# SECAO 7: Salvar resultados
# ==============================================================================

cat("\n=== Salvando resultados ===\n")

write.csv(results_r, file.path(output_dir, "auto_arima_comparison_R.csv"),
          row.names = FALSE)
cat("Salvo: auto_arima_comparison_R.csv\n")

# ==============================================================================
# SECAO 8: Notas sobre diferencas esperadas
# ==============================================================================

cat("\n=== Notas sobre diferencas Python vs R ===\n")
cat("1. auto.arima() do R e auto_arima do chronobox podem selecionar\n")
cat("   modelos diferentes devido a diferencas no espaco de busca.\n")
cat("2. O R usa stepwise=TRUE por padrao; aqui usamos stepwise=FALSE\n")
cat("   para busca exaustiva (mais comparavel ao grid search do Python).\n")
cat("3. Diferencas em AIC/BIC sao esperadas se os modelos diferirem.\n")
cat("4. Quando o mesmo modelo e selecionado, coeficientes devem\n")
cat("   concordar com tolerancia < 1e-4.\n")

cat("\n=== Script 03 concluido com sucesso ===\n")
