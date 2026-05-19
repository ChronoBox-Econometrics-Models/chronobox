# =============================================================================
# 03_x13_validation.R
# Validacao cruzada: X-13 ARIMA-SEATS via seasonal::seas()
# Se o pacote seasonal nao estiver disponivel, usa STL robusto como proxy
# =============================================================================

set.seed(42)

# --- Configuracao de caminhos ---
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
  script_dir <- "."
}
base_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
data_dir <- file.path(base_dir, "data")
output_dir <- file.path(base_dir, "outputs", "R")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# --- Carregar dados brazil_ipca.csv ---
ipca_path <- file.path(data_dir, "brazil_ipca.csv")
cat("Carregando dados de:", ipca_path, "\n")

ipca_df <- read.csv(ipca_path, stringsAsFactors = FALSE)
ipca_df$date <- as.Date(ipca_df$date)

start_year <- as.numeric(format(ipca_df$date[1], "%Y"))
start_month <- as.numeric(format(ipca_df$date[1], "%m"))
ipca_ts <- ts(ipca_df$ipca, start = c(start_year, start_month), frequency = 12)

cat("Serie IPCA: n =", length(ipca_ts),
    ", inicio =", start(ipca_ts)[1], "/", start(ipca_ts)[2],
    ", fim =", end(ipca_ts)[1], "/", end(ipca_ts)[2], "\n")

# --- Carregar dados airline.csv ---
airline_path <- file.path(data_dir, "airline.csv")
airline_df <- read.csv(airline_path, stringsAsFactors = FALSE)
airline_df$date <- as.Date(airline_df$date)
airline_ts <- ts(airline_df$passengers, start = c(1949, 1), frequency = 12)

# =============================================================================
# Tentar X-13 ARIMA-SEATS via seasonal::seas()
# =============================================================================

use_x13 <- FALSE
x13_results <- list()

cat("\n--- Tentando X-13 ARIMA-SEATS ---\n")

if (requireNamespace("seasonal", quietly = TRUE)) {
  library(seasonal)
  tryCatch({
    # X-13 no airline dataset
    seas_airline <- seas(airline_ts)
    cat("X-13 executado com sucesso no dataset airline!\n")
    use_x13 <- TRUE

    # Extrair serie dessazonalizada
    sa_airline <- final(seas_airline)
    trend_airline <- trend(seas_airline)
    seasonal_airline <- seasonal(seas_airline)

    cat("Airline - range sazonal X-13:", diff(range(seasonal_airline)), "\n")
    cat("Airline - desvio padrao SA:", sd(as.numeric(sa_airline)), "\n")

    # X-13 no IPCA dataset
    seas_ipca <- seas(ipca_ts)
    sa_ipca <- final(seas_ipca)
    trend_ipca <- trend(seas_ipca)
    seasonal_ipca <- seasonal(seas_ipca)
    cat("IPCA - range sazonal X-13:", diff(range(seasonal_ipca)), "\n")

    x13_results <- list(
      airline = list(sa = sa_airline, trend = trend_airline, seasonal = seasonal_airline),
      ipca = list(sa = sa_ipca, trend = trend_ipca, seasonal = seasonal_ipca)
    )

  }, error = function(e) {
    cat("Erro ao executar X-13:", conditionMessage(e), "\n")
    cat("X-13 requer o binario x13ashtml instalado no sistema.\n")
  })
} else {
  cat("Pacote 'seasonal' nao instalado.\n")
  cat("Para instalar: install.packages('seasonal')\n")
  cat("O pacote requer o binario X-13ARIMA-SEATS do US Census Bureau.\n")
}

# =============================================================================
# Alternativa: STL robusto como proxy para ajuste sazonal
# =============================================================================
cat("\n--- STL robusto como proxy/alternativa ao X-13 ---\n")

# STL robusto no airline (log-transformado para lidar com sazonalidade multiplicativa)
airline_log_ts <- log(airline_ts)
stl_airline <- stl(airline_log_ts, s.window = 13, robust = TRUE)
sa_stl_airline <- exp(airline_log_ts - stl_airline$time.series[, "seasonal"])
trend_stl_airline <- exp(stl_airline$time.series[, "trend"])
seasonal_stl_airline <- stl_airline$time.series[, "seasonal"]

cat("Airline STL robusto (log) - range sazonal:", diff(range(seasonal_stl_airline)), "\n")

# STL robusto no IPCA com diferentes s.window
stl_ipca_s7 <- stl(ipca_ts, s.window = 7, robust = TRUE)
stl_ipca_s13 <- stl(ipca_ts, s.window = 13, robust = TRUE)
stl_ipca_s25 <- stl(ipca_ts, s.window = 25, robust = TRUE)

sa_stl_ipca_s7 <- ipca_ts - stl_ipca_s7$time.series[, "seasonal"]
sa_stl_ipca_s13 <- ipca_ts - stl_ipca_s13$time.series[, "seasonal"]
sa_stl_ipca_s25 <- ipca_ts - stl_ipca_s25$time.series[, "seasonal"]

cat("IPCA STL s=7 - desvio padrao residual:", sd(stl_ipca_s7$time.series[, "remainder"]), "\n")
cat("IPCA STL s=13 - desvio padrao residual:", sd(stl_ipca_s13$time.series[, "remainder"]), "\n")
cat("IPCA STL s=25 - desvio padrao residual:", sd(stl_ipca_s25$time.series[, "remainder"]), "\n")

# =============================================================================
# Comparacao X-13 vs STL (se X-13 disponivel)
# =============================================================================
if (use_x13) {
  cat("\n--- Comparacao X-13 vs STL ---\n")

  # Airline
  cor_sa <- cor(as.numeric(x13_results$airline$sa), as.numeric(sa_stl_airline))
  cat("Airline: correlacao SA (X-13 vs STL):", cor_sa, "\n")

  diff_sa <- as.numeric(x13_results$airline$sa) - as.numeric(sa_stl_airline)
  cat("Airline: MAE entre X-13 e STL:", mean(abs(diff_sa)), "\n")

  # IPCA
  cor_sa_ipca <- cor(as.numeric(x13_results$ipca$sa), as.numeric(sa_stl_ipca_s13))
  cat("IPCA: correlacao SA (X-13 vs STL s=13):", cor_sa_ipca, "\n")
} else {
  cat("\n--- X-13 nao disponivel, usando apenas STL robusto ---\n")
  cat("NOTA: O chronobox usa STL robusto como proxy para X-13.\n")
  cat("Para validacao completa com X-13, instale o pacote seasonal:\n")
  cat("  install.packages('seasonal')\n")
  cat("  # E garanta que x13ashtml esta no PATH\n")
}

# =============================================================================
# Salvar resultados
# =============================================================================
cat("\n--- Salvando resultados ---\n")

# Resultados IPCA (serie principal para X-13/proxy)
results_ipca <- data.frame(
  date = ipca_df$date,
  observed = as.numeric(ipca_ts),
  sa_stl_s7 = as.numeric(sa_stl_ipca_s7),
  sa_stl_s13_robust = as.numeric(sa_stl_ipca_s13),
  sa_stl_s25 = as.numeric(sa_stl_ipca_s25),
  trend_stl_s7 = as.numeric(stl_ipca_s7$time.series[, "trend"]),
  trend_stl_s13_robust = as.numeric(stl_ipca_s13$time.series[, "trend"]),
  trend_stl_s25 = as.numeric(stl_ipca_s25$time.series[, "trend"]),
  seasonal_stl_s13_robust = as.numeric(stl_ipca_s13$time.series[, "seasonal"])
)

# Adicionar colunas X-13 se disponivel
if (use_x13) {
  results_ipca$sa_x13 <- as.numeric(x13_results$ipca$sa)
  results_ipca$trend_x13 <- as.numeric(x13_results$ipca$trend)
  results_ipca$seasonal_x13 <- as.numeric(x13_results$ipca$seasonal)
}

output_path_ipca <- file.path(output_dir, "x13_adjusted.csv")
write.csv(results_ipca, output_path_ipca, row.names = FALSE)
cat("Resultados IPCA salvos em:", output_path_ipca, "\n")

# Resultados Airline
results_airline <- data.frame(
  date = airline_df$date,
  observed = as.numeric(airline_ts),
  sa_stl_robust = as.numeric(sa_stl_airline),
  trend_stl_robust = as.numeric(trend_stl_airline),
  seasonal_stl_robust_log = as.numeric(seasonal_stl_airline)
)

if (use_x13) {
  results_airline$sa_x13 <- as.numeric(x13_results$airline$sa)
  results_airline$trend_x13 <- as.numeric(x13_results$airline$trend)
  results_airline$seasonal_x13 <- as.numeric(x13_results$airline$seasonal)
}

output_path_airline <- file.path(output_dir, "x13_airline.csv")
write.csv(results_airline, output_path_airline, row.names = FALSE)
cat("Resultados Airline salvos em:", output_path_airline, "\n")

# =============================================================================
# Comparacao com resultados Python
# =============================================================================
cat("\n--- Comparacao com Python ---\n")

python_path <- file.path(base_dir, "outputs", "x13_adjusted.csv")
if (file.exists(python_path)) {
  python_df <- read.csv(python_path, stringsAsFactors = FALSE)

  # Comparar series dessazonalizadas STL s=13 robusto
  if ("sa_stl_s13_robust" %in% names(python_df)) {
    diff_sa <- abs(as.numeric(results_ipca$sa_stl_s13_robust) -
                   as.numeric(python_df$sa_stl_s13_robust))
    cat("IPCA SA STL s=13 robusto - diff max:", max(diff_sa, na.rm = TRUE), "\n")
    cat("IPCA SA STL s=13 robusto - diff media:", mean(diff_sa, na.rm = TRUE), "\n")
  }

  # TOLERANCIA DOCUMENTADA:
  # STL robusto como proxy para X-13 pode ter diferencas significativas
  # em relacao ao X-13 real (metodos fundamentalmente diferentes).
  # Entre R e Python para o mesmo STL: tolerancia < 0.01.
  # Entre X-13 e STL proxy: diferencas podem ser maiores, mas alta correlacao
  # (tipicamente > 0.95) e esperada.
  cat("\nTOLERANCIA: STL R vs Python < 0.01\n")
  cat("X-13 vs STL proxy: correlacao > 0.95 esperada\n")
} else {
  cat("Arquivo Python nao encontrado:", python_path, "\n")
}

cat("\n=== 03_x13_validation.R concluido ===\n")
