# =============================================================================
# 01_classical_validation.R
# Validacao cruzada: Decomposicao Classica (aditiva e multiplicativa)
# Usa stats::decompose() como referencia para comparar com chronobox
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

# --- Carregar dados airline.csv ---
airline_path <- file.path(data_dir, "airline.csv")
cat("Carregando dados de:", airline_path, "\n")

airline_df <- read.csv(airline_path, stringsAsFactors = FALSE)
airline_df$date <- as.Date(airline_df$date)

# Criar objeto ts (serie mensal, inicio jan/1949)
airline_ts <- ts(airline_df$passengers, start = c(1949, 1), frequency = 12)

cat("Serie airline: n =", length(airline_ts),
    ", inicio =", start(airline_ts)[1], "/", start(airline_ts)[2],
    ", fim =", end(airline_ts)[1], "/", end(airline_ts)[2], "\n")

# =============================================================================
# Decomposicao Aditiva
# =============================================================================
cat("\n--- Decomposicao Classica Aditiva ---\n")

decomp_add <- decompose(airline_ts, type = "additive")

# Extrair componentes
trend_add <- as.numeric(decomp_add$trend)
seasonal_add <- as.numeric(decomp_add$seasonal)
random_add <- as.numeric(decomp_add$random)
observed <- as.numeric(airline_ts)

# Verificar propriedade fundamental: observed = trend + seasonal + residual
# (exceto onde trend/residual sao NA nas bordas)
valid_idx <- which(!is.na(trend_add) & !is.na(random_add))
reconstruction_add <- trend_add[valid_idx] + seasonal_add[valid_idx] + random_add[valid_idx]
max_error_add <- max(abs(observed[valid_idx] - reconstruction_add))
cat("Erro maximo de reconstrucao (aditiva):", max_error_add, "\n")
cat("Tolerancia < 1e-6:", max_error_add < 1e-6, "\n")
# TOLERANCIA: A decomposicao classica e exata (media movel + subtracao),
# entao o erro de reconstrucao deve ser < 1e-10 (erro de ponto flutuante).

cat("NAs no trend (bordas):", sum(is.na(trend_add)), "\n")
cat("Range sazonal:", range(seasonal_add), "\n")
cat("Desvio padrao residual:", sd(random_add, na.rm = TRUE), "\n")

# =============================================================================
# Decomposicao Multiplicativa
# =============================================================================
cat("\n--- Decomposicao Classica Multiplicativa ---\n")

decomp_mult <- decompose(airline_ts, type = "multiplicative")

trend_mult <- as.numeric(decomp_mult$trend)
seasonal_mult <- as.numeric(decomp_mult$seasonal)
random_mult <- as.numeric(decomp_mult$random)

# Verificar: observed = trend * seasonal * residual
valid_idx_m <- which(!is.na(trend_mult) & !is.na(random_mult))
reconstruction_mult <- trend_mult[valid_idx_m] * seasonal_mult[valid_idx_m] * random_mult[valid_idx_m]
max_error_mult <- max(abs(observed[valid_idx_m] - reconstruction_mult))
cat("Erro maximo de reconstrucao (multiplicativa):", max_error_mult, "\n")
cat("Tolerancia < 1e-6:", max_error_mult < 1e-6, "\n")
# TOLERANCIA: Decomposicao multiplicativa classica tambem e exata,
# com erros limitados a precisao de ponto flutuante (< 1e-10).

cat("NAs no trend (bordas):", sum(is.na(trend_mult)), "\n")
cat("Range sazonal (fatores):", range(seasonal_mult), "\n")
cat("Desvio padrao residual:", sd(random_mult, na.rm = TRUE), "\n")

# =============================================================================
# Salvar resultados em CSV
# =============================================================================

# Montar data.frame com todos os componentes
results <- data.frame(
  date = airline_df$date,
  observed = observed,
  trend_additive = trend_add,
  seasonal_additive = seasonal_add,
  residual_additive = random_add,
  trend_multiplicative = trend_mult,
  seasonal_multiplicative = seasonal_mult,
  residual_multiplicative = random_mult
)

output_path <- file.path(output_dir, "classical_components.csv")
write.csv(results, output_path, row.names = FALSE, na = "")
cat("\nResultados salvos em:", output_path, "\n")

# =============================================================================
# Comparacao com resultados Python (chronobox)
# =============================================================================
cat("\n--- Comparacao com Python ---\n")

python_path <- file.path(base_dir, "outputs", "classical_components.csv")
if (file.exists(python_path)) {
  python_df <- read.csv(python_path, stringsAsFactors = FALSE)

  # Comparar componentes sazonais (disponiveis para todas as observacoes)
  # Aditivo
  diff_seasonal_add <- abs(results$seasonal_additive - python_df$seasonal_additive)
  cat("Sazonal aditivo - diff max:", max(diff_seasonal_add, na.rm = TRUE), "\n")
  cat("Sazonal aditivo - diff media:", mean(diff_seasonal_add, na.rm = TRUE), "\n")

  # Multiplicativo
  diff_seasonal_mult <- abs(results$seasonal_multiplicative - python_df$seasonal_multiplicative)
  cat("Sazonal multiplicativo - diff max:", max(diff_seasonal_mult, na.rm = TRUE), "\n")
  cat("Sazonal multiplicativo - diff media:", mean(diff_seasonal_mult, na.rm = TRUE), "\n")

  # Trend (onde disponivel)
  valid_trend <- which(!is.na(results$trend_additive) & python_df$trend_additive != "")
  if (length(valid_trend) > 0) {
    py_trend <- as.numeric(python_df$trend_additive[valid_trend])
    r_trend <- results$trend_additive[valid_trend]
    diff_trend_add <- abs(r_trend - py_trend)
    cat("Trend aditivo - diff max:", max(diff_trend_add, na.rm = TRUE), "\n")
    cat("Trend aditivo - diff media:", mean(diff_trend_add, na.rm = TRUE), "\n")
  }

  # TOLERANCIA DOCUMENTADA:
  # Para decomposicao classica, tanto R quanto Python usam o mesmo algoritmo
  # (media movel centrada + subtracao/divisao), portanto as diferencas devem
  # ser < 1e-6 (limitadas apenas por precisao de ponto flutuante).
  cat("\nTOLERANCIA: Componentes classicos devem concordar com < 1e-6\n")
  cat("Isso porque ambos implementam o mesmo algoritmo deterministico.\n")
} else {
  cat("Arquivo Python nao encontrado em:", python_path, "\n")
  cat("Execute os notebooks Python primeiro para gerar os resultados.\n")
}

cat("\n=== 01_classical_validation.R concluido ===\n")
