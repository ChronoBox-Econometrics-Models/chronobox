# =============================================================================
# 02_stl_validation.R
# Validacao cruzada: STL Decomposition com diferentes parametros
# Usa stats::stl() e stlplus (se disponivel) como referencia
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

# --- Carregar dados co2.csv ---
co2_path <- file.path(data_dir, "co2.csv")
cat("Carregando dados de:", co2_path, "\n")

co2_df <- read.csv(co2_path, stringsAsFactors = FALSE)
co2_df$date <- as.Date(co2_df$date)

# Criar objeto ts mensal
start_year <- as.numeric(format(co2_df$date[1], "%Y"))
start_month <- as.numeric(format(co2_df$date[1], "%m"))
co2_ts <- ts(co2_df$co2_ppm, start = c(start_year, start_month), frequency = 12)

cat("Serie CO2: n =", length(co2_ts),
    ", inicio =", start(co2_ts)[1], "/", start(co2_ts)[2],
    ", fim =", end(co2_ts)[1], "/", end(co2_ts)[2], "\n")

# =============================================================================
# STL com diferentes parametros s.window
# =============================================================================

# Configuracao 1: s.window = 7
cat("\n--- STL com s.window = 7 ---\n")
stl_s7 <- stl(co2_ts, s.window = 7)
cat("Componentes: seasonal, trend, remainder\n")
cat("Range sazonal:", diff(range(stl_s7$time.series[, "seasonal"])), "\n")
cat("Range trend:", diff(range(stl_s7$time.series[, "trend"])), "\n")
cat("Desvio padrao residual:", sd(stl_s7$time.series[, "remainder"]), "\n")

# Configuracao 2: s.window = 15
cat("\n--- STL com s.window = 15 ---\n")
stl_s15 <- stl(co2_ts, s.window = 15)
cat("Range sazonal:", diff(range(stl_s15$time.series[, "seasonal"])), "\n")
cat("Range trend:", diff(range(stl_s15$time.series[, "trend"])), "\n")
cat("Desvio padrao residual:", sd(stl_s15$time.series[, "remainder"]), "\n")

# Configuracao 3: s.window = 35
cat("\n--- STL com s.window = 35 ---\n")
stl_s35 <- stl(co2_ts, s.window = 35)
cat("Range sazonal:", diff(range(stl_s35$time.series[, "seasonal"])), "\n")
cat("Range trend:", diff(range(stl_s35$time.series[, "trend"])), "\n")
cat("Desvio padrao residual:", sd(stl_s35$time.series[, "remainder"]), "\n")

# Configuracao 4: s.window = "periodic" (sazonal fixo)
cat("\n--- STL com s.window = 'periodic' ---\n")
stl_periodic <- stl(co2_ts, s.window = "periodic")
cat("Range sazonal:", diff(range(stl_periodic$time.series[, "seasonal"])), "\n")
cat("Range trend:", diff(range(stl_periodic$time.series[, "trend"])), "\n")
cat("Desvio padrao residual:", sd(stl_periodic$time.series[, "remainder"]), "\n")

# Verificar propriedade fundamental: observed = trend + seasonal + remainder
# (STL e sempre aditivo)
for (name_stl in c("s7", "s15", "s35", "periodic")) {
  stl_obj <- get(paste0("stl_", name_stl))
  recon <- stl_obj$time.series[, "trend"] +
           stl_obj$time.series[, "seasonal"] +
           stl_obj$time.series[, "remainder"]
  max_err <- max(abs(as.numeric(co2_ts) - as.numeric(recon)))
  cat(sprintf("STL %s - erro reconstrucao max: %.2e\n", name_stl, max_err))
}

# =============================================================================
# STL com t.window customizado
# =============================================================================
cat("\n--- STL com s.window=13, t.window=23 ---\n")
stl_custom <- stl(co2_ts, s.window = 13, t.window = 23)
cat("Range sazonal:", diff(range(stl_custom$time.series[, "seasonal"])), "\n")
cat("Range trend:", diff(range(stl_custom$time.series[, "trend"])), "\n")
cat("Desvio padrao residual:", sd(stl_custom$time.series[, "remainder"]), "\n")

# =============================================================================
# STL robusto (outer loops)
# =============================================================================
cat("\n--- STL robusto (s.window=13, robust=TRUE) ---\n")
stl_robust <- stl(co2_ts, s.window = 13, robust = TRUE)
cat("Range sazonal:", diff(range(stl_robust$time.series[, "seasonal"])), "\n")
cat("Range trend:", diff(range(stl_robust$time.series[, "trend"])), "\n")
cat("Desvio padrao residual:", sd(stl_robust$time.series[, "remainder"]), "\n")

# =============================================================================
# stlplus (se disponivel)
# =============================================================================
cat("\n--- stlplus ---\n")
if (requireNamespace("stlplus", quietly = TRUE)) {
  library(stlplus)
  stlplus_fit <- stlplus(co2_ts, s.window = 13)
  cat("stlplus com s.window=13 executado com sucesso\n")
  cat("Range sazonal:", diff(range(stlplus_fit$data$seasonal)), "\n")
  cat("Range trend:", diff(range(stlplus_fit$data$trend)), "\n")
  cat("Desvio padrao residual:", sd(stlplus_fit$data$remainder), "\n")
} else {
  cat("Pacote stlplus nao disponivel. Instale com: install.packages('stlplus')\n")
  cat("Usando apenas stats::stl() para validacao.\n")
}

# =============================================================================
# Salvar resultados em CSV
# =============================================================================
cat("\n--- Salvando resultados ---\n")

results <- data.frame(
  date = co2_df$date,
  observed = as.numeric(co2_ts),
  trend_s7 = as.numeric(stl_s7$time.series[, "trend"]),
  seasonal_s7 = as.numeric(stl_s7$time.series[, "seasonal"]),
  residual_s7 = as.numeric(stl_s7$time.series[, "remainder"]),
  trend_s15 = as.numeric(stl_s15$time.series[, "trend"]),
  seasonal_s15 = as.numeric(stl_s15$time.series[, "seasonal"]),
  residual_s15 = as.numeric(stl_s15$time.series[, "remainder"]),
  trend_s35 = as.numeric(stl_s35$time.series[, "trend"]),
  seasonal_s35 = as.numeric(stl_s35$time.series[, "seasonal"]),
  residual_s35 = as.numeric(stl_s35$time.series[, "remainder"]),
  trend_periodic = as.numeric(stl_periodic$time.series[, "trend"]),
  seasonal_periodic = as.numeric(stl_periodic$time.series[, "seasonal"]),
  residual_periodic = as.numeric(stl_periodic$time.series[, "remainder"]),
  trend_robust = as.numeric(stl_robust$time.series[, "trend"]),
  seasonal_robust = as.numeric(stl_robust$time.series[, "seasonal"]),
  residual_robust = as.numeric(stl_robust$time.series[, "remainder"])
)

output_path <- file.path(output_dir, "stl_components.csv")
write.csv(results, output_path, row.names = FALSE)
cat("Resultados salvos em:", output_path, "\n")

# =============================================================================
# Comparacao com resultados Python (chronobox)
# =============================================================================
cat("\n--- Comparacao com Python ---\n")

python_path <- file.path(base_dir, "outputs", "stl_components.csv")
if (file.exists(python_path)) {
  python_df <- read.csv(python_path, stringsAsFactors = FALSE)

  # Comparar STL s.window=7
  if ("trend_s7" %in% names(python_df)) {
    diff_trend <- abs(results$trend_s7 - python_df$trend_s7)
    diff_seasonal <- abs(results$seasonal_s7 - python_df$seasonal_s7)
    cat("STL s=7 trend - diff max:", max(diff_trend, na.rm = TRUE), "\n")
    cat("STL s=7 seasonal - diff max:", max(diff_seasonal, na.rm = TRUE), "\n")
  }

  # Comparar STL s.window=15
  if ("trend_s15" %in% names(python_df)) {
    diff_trend <- abs(results$trend_s15 - python_df$trend_s15)
    diff_seasonal <- abs(results$seasonal_s15 - python_df$seasonal_s15)
    cat("STL s=15 trend - diff max:", max(diff_trend, na.rm = TRUE), "\n")
    cat("STL s=15 seasonal - diff max:", max(diff_seasonal, na.rm = TRUE), "\n")
  }

  # Comparar STL s.window=35
  if ("trend_s35" %in% names(python_df)) {
    diff_trend <- abs(results$trend_s35 - python_df$trend_s35)
    diff_seasonal <- abs(results$seasonal_s35 - python_df$seasonal_s35)
    cat("STL s=35 trend - diff max:", max(diff_trend, na.rm = TRUE), "\n")
    cat("STL s=35 seasonal - diff max:", max(diff_seasonal, na.rm = TRUE), "\n")
  }

  # TOLERANCIA DOCUMENTADA:
  # STL implementacoes podem diferir levemente entre R e Python (statsmodels)
  # devido a detalhes de convergencia do LOESS. Tolerancia esperada: < 0.01
  # para a maioria dos casos, com possibilidade de diferencas maiores nas
  # bordas da serie. Para mesmos parametros, correlacao > 0.999.
  cat("\nTOLERANCIA: Componentes STL podem diferir em ate ~0.01\n")
  cat("Diferencas maiores nas bordas sao esperadas (efeito de borda LOESS).\n")
} else {
  cat("Arquivo Python nao encontrado em:", python_path, "\n")
  cat("Execute os notebooks Python primeiro para gerar os resultados.\n")
}

cat("\n=== 02_stl_validation.R concluido ===\n")
