##############################################################################
# 01_hp_bk_cf_validation.R
#
# Validacao cruzada dos filtros HP, Baxter-King e Christiano-Fitzgerald
# usando o pacote mFilter como referencia.
#
# Parametros:
#   HP filter:  lambda = 1600 (padrao para dados trimestrais)
#   BK filter:  pl = 6, pu = 32 (bandas de frequencia para ciclos de negocios)
#               K = 12 (truncation lag, simétrico, default mFilter)
#   CF filter:  pl = 6, pu = 32 (mesmas bandas de frequencia)
#
# Tolerancias esperadas vs chronobox:
#   HP:  < 1e-6 (solucao exata, mesmo sistema linear)
#   BK:  < 1e-4 (diferenças por truncamento nas bordas)
#   CF:  < 1e-4 (diferenças por metodo de estimacao espectral)
#
# Datasets: examples/filters/data/us_gdp_quarterly.csv
#           examples/filters/data/brazil_gdp.csv
##############################################################################

# Pacote necessario
if (!require("mFilter", quietly = TRUE)) {
  install.packages("mFilter", repos = "https://cloud.r-project.org")
  library(mFilter)
}

set.seed(42)

# --- Caminhos ---
base_dir <- file.path(dirname(sys.frame(1)$ofile), "..")
data_dir <- file.path(base_dir, "data")
output_dir <- file.path(base_dir, "outputs", "R")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Validacao HP / BK / CF com mFilter ===\n\n")

# --- Funcao auxiliar para processar um dataset ---
process_country <- function(csv_file, country_label) {
  cat(sprintf("--- Processando: %s (%s) ---\n", country_label, csv_file))

  df <- read.csv(csv_file, stringsAsFactors = FALSE)
  dates <- as.Date(df$date)
  y <- ts(df$gdp_log, frequency = 4, start = c(as.numeric(format(dates[1], "%Y")),
                                                  as.numeric(format(dates[1], "%m")) %/% 3 + 1))

  # -------------------------------------------------------
  # Hodrick-Prescott filter (lambda = 1600)
  # -------------------------------------------------------
  hp <- hpfilter(y, freq = 1600, type = "lambda")
  hp_trend <- as.numeric(hp$trend)
  hp_cycle <- as.numeric(hp$cycle)

  cat(sprintf("  HP: cycle std = %.8f, range = [%.8f, %.8f]\n",
              sd(hp_cycle), min(hp_cycle), max(hp_cycle)))

  # -------------------------------------------------------
  # Baxter-King filter (pl=6, pu=32, K=12)
  # Nota: BK é um filtro band-pass simetrico; as primeiras e ultimas K
  # observacoes sao NA (perda de dados nas bordas).
  # -------------------------------------------------------
  bk <- bkfilter(y, pl = 6, pu = 32, nfix = 12)
  bk_cycle <- as.numeric(bk$cycle)

  valid_bk <- !is.na(bk_cycle)
  cat(sprintf("  BK: cycle std = %.8f (obs validas: %d/%d)\n",
              sd(bk_cycle, na.rm = TRUE), sum(valid_bk), length(bk_cycle)))

  # -------------------------------------------------------
  # Christiano-Fitzgerald filter (pl=6, pu=32)
  # CF usa toda a amostra (sem perda de bordas).
  # -------------------------------------------------------
  cf <- cffilter(y, pl = 6, pu = 32)
  cf_cycle <- as.numeric(cf$cycle)

  cat(sprintf("  CF: cycle std = %.8f\n", sd(cf_cycle)))

  # --- Correlacoes entre filtros ---
  # Para correlacao com BK, usar apenas observacoes validas
  valid_idx <- which(valid_bk)
  cor_hp_bk <- cor(hp_cycle[valid_idx], bk_cycle[valid_idx])
  cor_hp_cf <- cor(hp_cycle, cf_cycle)
  cor_bk_cf <- cor(bk_cycle[valid_idx], cf_cycle[valid_idx])

  cat(sprintf("  Correlacoes: HP-BK = %.6f, HP-CF = %.6f, BK-CF = %.6f\n\n",
              cor_hp_bk, cor_hp_cf, cor_bk_cf))

  # --- Montar data.frame de resultados ---
  result <- data.frame(
    date = as.character(dates),
    country = country_label,
    hp_cycle = hp_cycle,
    hp_trend = hp_trend,
    cf_cycle = cf_cycle,
    bk_cycle = bk_cycle,
    stringsAsFactors = FALSE
  )

  return(result)
}

# --- Processar ambos os paises ---
us_file <- file.path(data_dir, "us_gdp_quarterly.csv")
br_file <- file.path(data_dir, "brazil_gdp.csv")

results_us <- process_country(us_file, "US")
results_br <- process_country(br_file, "BR")

# --- Combinar e salvar ---
all_results <- rbind(results_us, results_br)

output_file <- file.path(output_dir, "hp_bk_cf_cycles.csv")
write.csv(all_results, output_file, row.names = FALSE)
cat(sprintf("Resultados salvos em: %s\n", output_file))

# --- Salvar estatísticas resumidas em JSON ---
stats_us <- list(
  n_obs = nrow(results_us),
  hp_cycle_std = sd(results_us$hp_cycle),
  bk_cycle_std = sd(results_us$bk_cycle, na.rm = TRUE),
  cf_cycle_std = sd(results_us$cf_cycle),
  hp_cycle_min = min(results_us$hp_cycle),
  hp_cycle_max = max(results_us$hp_cycle)
)

stats_br <- list(
  n_obs = nrow(results_br),
  hp_cycle_std = sd(results_br$hp_cycle),
  bk_cycle_std = sd(results_br$bk_cycle, na.rm = TRUE),
  cf_cycle_std = sd(results_br$cf_cycle),
  hp_cycle_min = min(results_br$hp_cycle),
  hp_cycle_max = max(results_br$hp_cycle)
)

# Salvar summary como CSV simples
summary_df <- data.frame(
  country = c("US", "BR"),
  n_obs = c(stats_us$n_obs, stats_br$n_obs),
  hp_std = c(stats_us$hp_cycle_std, stats_br$hp_cycle_std),
  bk_std = c(stats_us$bk_cycle_std, stats_br$bk_cycle_std),
  cf_std = c(stats_us$cf_cycle_std, stats_br$cf_cycle_std),
  hp_min = c(stats_us$hp_cycle_min, stats_br$hp_cycle_min),
  hp_max = c(stats_us$hp_cycle_max, stats_br$hp_cycle_max)
)

summary_file <- file.path(output_dir, "hp_bk_cf_summary.csv")
write.csv(summary_df, summary_file, row.names = FALSE)
cat(sprintf("Resumo salvo em: %s\n", summary_file))

cat("\n=== 01_hp_bk_cf_validation.R concluido com sucesso ===\n")
