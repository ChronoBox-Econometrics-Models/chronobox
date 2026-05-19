# ==============================================================================
# 04_arfima_validation.R
# Validacao cruzada: ARFIMA com fracdiff::fracdiff()
#
# Objetivo: Estimar o parametro d fracionario usando o pacote fracdiff
#           e comparar com os resultados do chronobox (Python).
#
# Datasets: nile.csv, brazil_ipca.csv
# Metodos:  fracdiff() para estimacao ML de d
#           fdGPH() e fdSperio() para estimacao semi-parametrica
#
# Dependencias: fracdiff, forecast
# ==============================================================================

# --- Configuracao inicial -----------------------------------------------------
set.seed(42)

library(fracdiff)
library(forecast)

# Caminhos relativos (assumindo execucao a partir de examples/arima/)
data_dir    <- file.path("..", "data")
output_dir  <- file.path("..", "outputs", "R")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ==============================================================================
# SECAO 1: Carregar dados
# ==============================================================================

cat("=== Carregando datasets ===\n")

# Nile
nile_raw <- read.csv(file.path(data_dir, "nile.csv"))
nile_ts  <- ts(nile_raw$flow, start = 1871, frequency = 1)
cat(sprintf("nile.csv: %d obs\n", length(nile_ts)))

# IPCA
ipca_raw <- read.csv(file.path(data_dir, "brazil_ipca.csv"))
ipca_ts  <- ts(ipca_raw$ipca, start = c(2004, 1), frequency = 12)
cat(sprintf("brazil_ipca.csv: %d obs\n", length(ipca_ts)))

# ==============================================================================
# SECAO 2: Estimacao de d para Nile
# ==============================================================================

cat("\n=== Estimacao de d fracionario para Nile ===\n")

# GPH estimator (Geweke-Porter-Hudak)
# Usa floor(n^0.5) frequencias por padrao
nile_gph <- fdGPH(nile_ts)
cat(sprintf("GPH:           d = %.6f (se = %.6f)\n", nile_gph$d, nile_gph$sd.as))

# Periodograma suavizado (Sperio)
nile_sperio <- fdSperio(nile_ts)
cat(sprintf("Sperio:        d = %.6f (se = %.6f)\n", nile_sperio$d, nile_sperio$sd.as))

# fracdiff ML estimation - ARFIMA(0,d,0)
nile_fd00 <- fracdiff(nile_ts, nar = 0, nma = 0)
cat(sprintf("ARFIMA(0,d,0): d = %.6f (se = %.6f)\n", nile_fd00$d, nile_fd00$stderror.dpq[1]))

# ARFIMA(1,d,0)
nile_fd10 <- fracdiff(nile_ts, nar = 1, nma = 0)
cat(sprintf("ARFIMA(1,d,0): d = %.6f, ar1 = %.6f\n", nile_fd10$d, nile_fd10$ar))

# ARFIMA(0,d,1)
nile_fd01 <- fracdiff(nile_ts, nar = 0, nma = 1)
cat(sprintf("ARFIMA(0,d,1): d = %.6f, ma1 = %.6f\n", nile_fd01$d, nile_fd01$ma))

# ARFIMA(1,d,1)
nile_fd11 <- fracdiff(nile_ts, nar = 1, nma = 1)
cat(sprintf("ARFIMA(1,d,1): d = %.6f, ar1 = %.6f, ma1 = %.6f\n",
            nile_fd11$d, nile_fd11$ar, nile_fd11$ma))

# ==============================================================================
# SECAO 3: Estimacao de d para IPCA
# ==============================================================================

cat("\n=== Estimacao de d fracionario para IPCA ===\n")

# GPH
ipca_gph <- fdGPH(ipca_ts)
cat(sprintf("GPH:           d = %.6f (se = %.6f)\n", ipca_gph$d, ipca_gph$sd.as))

# Sperio
ipca_sperio <- fdSperio(ipca_ts)
cat(sprintf("Sperio:        d = %.6f (se = %.6f)\n", ipca_sperio$d, ipca_sperio$sd.as))

# ARFIMA(0,d,0)
ipca_fd00 <- fracdiff(ipca_ts, nar = 0, nma = 0)
cat(sprintf("ARFIMA(0,d,0): d = %.6f\n", ipca_fd00$d))

# ARFIMA(1,d,0)
ipca_fd10 <- fracdiff(ipca_ts, nar = 1, nma = 0)
cat(sprintf("ARFIMA(1,d,0): d = %.6f, ar1 = %.6f\n", ipca_fd10$d, ipca_fd10$ar))

# ARFIMA(0,d,1)
ipca_fd01 <- fracdiff(ipca_ts, nar = 0, nma = 1)
cat(sprintf("ARFIMA(0,d,1): d = %.6f, ma1 = %.6f\n", ipca_fd01$d, ipca_fd01$ma))

# ARFIMA(1,d,1)
ipca_fd11 <- fracdiff(ipca_ts, nar = 1, nma = 1)
cat(sprintf("ARFIMA(1,d,1): d = %.6f, ar1 = %.6f, ma1 = %.6f\n",
            ipca_fd11$d, ipca_fd11$ar, ipca_fd11$ma))

# ==============================================================================
# SECAO 4: Montar tabela de resultados
# ==============================================================================

cat("\n=== Montando tabela de resultados ===\n")

# Funcao para extrair AIC de fracdiff (nao tem metodo AIC nativo)
# Usa a formula: AIC = -2*loglik + 2*k
fracdiff_aic <- function(fit) {
  k <- 1 + length(fit$ar) + length(fit$ma)  # d + AR + MA params
  -2 * fit$log.likelihood + 2 * k
}

results <- data.frame(
  dataset = c(
    # Nile - estimadores semi-parametricos
    "nile", "nile",
    # Nile - modelos ARFIMA
    "nile", "nile", "nile", "nile",
    # IPCA - estimadores semi-parametricos
    "ipca", "ipca",
    # IPCA - modelos ARFIMA
    "ipca", "ipca", "ipca", "ipca"
  ),
  method = c(
    "GPH", "Sperio",
    "ARFIMA(0,d,0)", "ARFIMA(1,d,0)", "ARFIMA(0,d,1)", "ARFIMA(1,d,1)",
    "GPH", "Sperio",
    "ARFIMA(0,d,0)", "ARFIMA(1,d,0)", "ARFIMA(0,d,1)", "ARFIMA(1,d,1)"
  ),
  d = c(
    nile_gph$d, nile_sperio$d,
    nile_fd00$d, nile_fd10$d, nile_fd01$d, nile_fd11$d,
    ipca_gph$d, ipca_sperio$d,
    ipca_fd00$d, ipca_fd10$d, ipca_fd01$d, ipca_fd11$d
  ),
  d_se = c(
    nile_gph$sd.as, nile_sperio$sd.as,
    nile_fd00$stderror.dpq[1], nile_fd10$stderror.dpq[1],
    nile_fd01$stderror.dpq[1], nile_fd11$stderror.dpq[1],
    ipca_gph$sd.as, ipca_sperio$sd.as,
    ipca_fd00$stderror.dpq[1], ipca_fd10$stderror.dpq[1],
    ipca_fd01$stderror.dpq[1], ipca_fd11$stderror.dpq[1]
  ),
  ar1 = c(
    NA, NA,
    NA, nile_fd10$ar, NA, nile_fd11$ar,
    NA, NA,
    NA, ipca_fd10$ar, NA, ipca_fd11$ar
  ),
  ma1 = c(
    NA, NA,
    NA, NA, nile_fd01$ma, nile_fd11$ma,
    NA, NA,
    NA, NA, ipca_fd01$ma, ipca_fd11$ma
  ),
  aic = c(
    NA, NA,
    fracdiff_aic(nile_fd00), fracdiff_aic(nile_fd10),
    fracdiff_aic(nile_fd01), fracdiff_aic(nile_fd11),
    NA, NA,
    fracdiff_aic(ipca_fd00), fracdiff_aic(ipca_fd10),
    fracdiff_aic(ipca_fd01), fracdiff_aic(ipca_fd11)
  ),
  loglik = c(
    NA, NA,
    nile_fd00$log.likelihood, nile_fd10$log.likelihood,
    nile_fd01$log.likelihood, nile_fd11$log.likelihood,
    NA, NA,
    ipca_fd00$log.likelihood, ipca_fd10$log.likelihood,
    ipca_fd01$log.likelihood, ipca_fd11$log.likelihood
  ),
  stringsAsFactors = FALSE
)

print(results)

# ==============================================================================
# SECAO 5: Salvar resultados
# ==============================================================================

cat("\n=== Salvando resultados ===\n")

write.csv(results, file.path(output_dir, "arfima_results_R.csv"),
          row.names = FALSE)
cat("Salvo: arfima_results_R.csv\n")

# ==============================================================================
# SECAO 6: Notas sobre diferencas Python vs R
# ==============================================================================

cat("\n=== Notas sobre diferencas Python vs R ===\n")
cat("1. O estimador GPH do fracdiff usa floor(n^0.5) frequencias,\n")
cat("   enquanto o chronobox pode usar bandwidth diferente.\n")
cat("2. Local Whittle (Python) vs Sperio (R) sao estimadores\n")
cat("   semi-parametricos diferentes - comparacao e aproximada.\n")
cat("3. fracdiff::fracdiff() usa MLE aproximado (Haslett-Raftery),\n")
cat("   que pode diferir do metodo CSS-ML do statsmodels.\n")
cat("4. Tolerancia esperada para d: < 0.05 (semi-parametrico),\n")
cat("   < 0.01 (parametrico com mesmo metodo).\n")

cat("\n=== Script 04 concluido com sucesso ===\n")
