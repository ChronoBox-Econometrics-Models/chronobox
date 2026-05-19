##############################################################################
# 03_spillover_validation.R
#
# Validacao cruzada do Diebold-Yilmaz Spillover Index usando o pacote vars
# para estimacao VAR e calculo manual do FEVD (Forecast Error Variance
# Decomposition), que eh a base do spillover index.
#
# Metodologia:
#   1. Estimar VAR(p) com p lags
#   2. Calcular FEVD para horizonte H
#   3. Total spillover = (soma off-diagonal da FEVD) / k * 100
#   4. Directional FROM/TO e net spillover
#
# O pacote frequencyConnectedness pode ser usado como alternativa,
# mas aqui usamos vars + calculo manual para transparencia.
#
# Parametros:
#   VAR lags: p = 2
#   Horizonte FEVD: H = 10
#   Dados: serie sintetica multivariada (4 variaveis)
#          + PIB EUA vs PIB Brasil
#
# Dataset sintetico: reproduz generate_multivariate_cycle() do Python
##############################################################################

# Pacotes necessarios
if (!require("vars", quietly = TRUE)) {
  install.packages("vars", repos = "https://cloud.r-project.org")
  library(vars)
}

set.seed(42)

# --- Caminhos ---
base_dir <- file.path(dirname(sys.frame(1)$ofile), "..")
data_dir <- file.path(base_dir, "data")
output_dir <- file.path(base_dir, "outputs", "R")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Validacao Spillover Index (Diebold-Yilmaz) ===\n\n")

# -------------------------------------------------------------------
# Gerar dados sinteticos multivariados (reproduzindo o Python)
#
# Python usa np.random.default_rng(42) que eh PCG64.
# R nao tem PCG64 nativo, entao geramos dados similares
# com a mesma estrutura mas usando set.seed(42) do R.
# A comparacao sera estrutural (mesma magnitude), nao numerica exata.
# -------------------------------------------------------------------
generate_multivariate <- function(n = 200, k = 4,
                                  common_weight = 0.6, seed = 42) {
  set.seed(seed)
  t_seq <- 1:n

  # Fator comum: sen(2*pi*t/32) + 0.5*sen(2*pi*t/16) + random walk
  common <- sin(2 * pi * t_seq / 32) + 0.5 * sin(2 * pi * t_seq / 16)
  common <- common + cumsum(rnorm(n, 0, 0.05))

  # Gerar k series
  data <- matrix(0, nrow = n, ncol = k)
  for (j in 1:k) {
    idio <- cumsum(rnorm(n, 0, 0.1))
    idio <- idio + 0.8 * sin(2 * pi * t_seq / (20 + 5 * j))
    series <- common_weight * common + (1 - common_weight) * idio
    trend_j <- 100 + 0.3 * t_seq
    data[, j] <- trend_j + series + rnorm(n, 0, 0.2)
  }

  colnames(data) <- paste0("var_", 1:k)

  # Datas trimestrais a partir de 1970
  dates <- seq(as.Date("1970-01-01"), by = "quarter", length.out = n)

  return(list(data = data, dates = dates))
}

# -------------------------------------------------------------------
# Calcular Spillover Table a partir de FEVD
# -------------------------------------------------------------------
compute_spillover <- function(var_data, p = 2, H = 10) {
  k <- ncol(var_data)
  var_names <- colnames(var_data)

  # Estimar VAR(p)
  var_model <- VAR(var_data, p = p, type = "const")

  # FEVD com horizonte H
  fevd_obj <- fevd(var_model, n.ahead = H)

  # Extrair a matriz de FEVD no horizonte H
  # fevd_obj eh uma lista com k elementos, cada um eh uma matriz (H x k)
  # A ultima linha (horizonte H) contem as proporcoes acumuladas
  fevd_matrix <- matrix(0, k, k)
  for (i in 1:k) {
    fevd_matrix[i, ] <- fevd_obj[[i]][H, ]
  }

  # Normalizar para que as linhas somem 1 (ja deve ser assim)
  row_sums <- rowSums(fevd_matrix)
  fevd_norm <- fevd_matrix / row_sums

  # Total spillover = soma off-diagonal / k * 100
  total_spillover <- (sum(fevd_norm) - sum(diag(fevd_norm))) / k * 100

  # Directional FROM (quanto cada variavel recebe das outras)
  dir_from <- (rowSums(fevd_norm) - diag(fevd_norm)) / k * 100

  # Directional TO (quanto cada variavel transmite para as outras)
  dir_to <- (colSums(fevd_norm) - diag(fevd_norm)) / k * 100

  # Net spillover
  net <- dir_to - dir_from

  # Pairwise spillover
  pairwise <- (t(fevd_norm) - fevd_norm) / k * 100

  return(list(
    fevd_matrix = fevd_norm,
    total_spillover = total_spillover,
    directional_from = dir_from,
    directional_to = dir_to,
    net_spillover = net,
    pairwise = pairwise,
    var_names = var_names,
    p = p,
    H = H,
    k = k
  ))
}

# -------------------------------------------------------------------
# Rolling spillover
# -------------------------------------------------------------------
compute_rolling_spillover <- function(var_data, dates, p = 2, H = 10,
                                      window = 100) {
  n <- nrow(var_data)
  k <- ncol(var_data)
  n_windows <- n - window + 1

  results <- data.frame(
    date = dates[(window):n],
    total_spillover = numeric(n_windows)
  )

  # Adicionar colunas para from/to/net por variavel
  for (j in 1:k) {
    results[[paste0("from_var_", j)]] <- numeric(n_windows)
    results[[paste0("to_var_", j)]] <- numeric(n_windows)
    results[[paste0("net_var_", j)]] <- numeric(n_windows)
  }

  for (i in 1:n_windows) {
    window_data <- var_data[i:(i + window - 1), ]
    tryCatch({
      sp <- compute_spillover(window_data, p = p, H = H)
      results$total_spillover[i] <- sp$total_spillover
      for (j in 1:k) {
        results[[paste0("from_var_", j)]][i] <- sp$directional_from[j]
        results[[paste0("to_var_", j)]][i] <- sp$directional_to[j]
        results[[paste0("net_var_", j)]][i] <- sp$net_spillover[j]
      }
    }, error = function(e) {
      results$total_spillover[i] <<- NA
    })
  }

  return(results)
}

# ===================================================================
# 1. Dados sinteticos multivariados (4 variaveis)
# ===================================================================
cat("--- 1. Serie sintetica multivariada (4 variaveis) ---\n")
mv <- generate_multivariate(n = 200, k = 4, common_weight = 0.6, seed = 42)

# Usar primeiras diferencas (stationaridade) para o VAR
mv_diff <- diff(mv$data)
colnames(mv_diff) <- paste0("var_", 1:4)

sp_synthetic <- compute_spillover(mv_diff, p = 2, H = 10)

cat(sprintf("  Total spillover: %.4f\n", sp_synthetic$total_spillover))
cat("  Directional FROM:\n")
for (j in 1:4) {
  cat(sprintf("    Var %d: %.4f\n", j, sp_synthetic$directional_from[j]))
}
cat("  Directional TO:\n")
for (j in 1:4) {
  cat(sprintf("    Var %d: %.4f\n", j, sp_synthetic$directional_to[j]))
}
cat("  Net spillover:\n")
for (j in 1:4) {
  cat(sprintf("    Var %d: %.4f\n", j, sp_synthetic$net_spillover[j]))
}

# ===================================================================
# 2. PIB EUA vs PIB Brasil
# ===================================================================
cat("\n--- 2. PIB EUA vs PIB Brasil ---\n")
us_df <- read.csv(file.path(data_dir, "us_gdp_quarterly.csv"), stringsAsFactors = FALSE)
br_df <- read.csv(file.path(data_dir, "brazil_gdp.csv"), stringsAsFactors = FALSE)

# Usar periodo comum (Brasil comeca em 2000, EUA em 1970)
# Brasil: 2000-Q1 a 2029-Q4 (120 obs)
# EUA: 1970-Q1 a 2019-Q4 (200 obs)
# Periodo comum: 2000-Q1 a 2019-Q4 (80 obs)
us_dates <- as.Date(us_df$date)
br_dates <- as.Date(br_df$date)

common_start <- max(min(us_dates), min(br_dates))
common_end <- min(max(us_dates), max(br_dates))

us_mask <- us_dates >= common_start & us_dates <= common_end
br_mask <- br_dates >= common_start & br_dates <= common_end

gdp_combined <- cbind(
  pib_eua = us_df$gdp_log[us_mask],
  pib_brasil = br_df$gdp_log[br_mask]
)

# Primeiras diferencas
n_common <- min(sum(us_mask), sum(br_mask))
gdp_combined <- gdp_combined[1:n_common, ]
gdp_diff <- diff(gdp_combined)

sp_gdp <- compute_spillover(gdp_diff, p = 2, H = 10)

cat(sprintf("  Total spillover: %.4f\n", sp_gdp$total_spillover))
cat(sprintf("  FROM PIB EUA: %.4f, FROM PIB Brasil: %.4f\n",
            sp_gdp$directional_from[1], sp_gdp$directional_from[2]))
cat(sprintf("  TO PIB EUA: %.4f, TO PIB Brasil: %.4f\n",
            sp_gdp$directional_to[1], sp_gdp$directional_to[2]))
cat(sprintf("  Net PIB EUA: %.4f, Net PIB Brasil: %.4f\n",
            sp_gdp$net_spillover[1], sp_gdp$net_spillover[2]))

# ===================================================================
# 3. Sensibilidade: variando horizonte e lags
# ===================================================================
cat("\n--- 3. Analise de sensibilidade ---\n")

# Por horizonte
cat("  Sensibilidade por horizonte (H):\n")
sensitivity_H <- data.frame(H = integer(), total_spillover = numeric())
for (H in c(2, 5, 10, 20, 50)) {
  sp_h <- compute_spillover(mv_diff, p = 2, H = H)
  sensitivity_H <- rbind(sensitivity_H, data.frame(H = H, total_spillover = sp_h$total_spillover))
  cat(sprintf("    H=%2d: total_spillover = %.4f\n", H, sp_h$total_spillover))
}

# Por lags
cat("  Sensibilidade por lags (p):\n")
sensitivity_p <- data.frame(p = integer(), total_spillover = numeric())
for (p_val in c(1, 2, 3, 4, 6, 8)) {
  sp_p <- compute_spillover(mv_diff, p = p_val, H = 10)
  sensitivity_p <- rbind(sensitivity_p, data.frame(p = p_val, total_spillover = sp_p$total_spillover))
  cat(sprintf("    p=%d: total_spillover = %.4f\n", p_val, sp_p$total_spillover))
}

# ===================================================================
# 4. Rolling spillover (janela = 100 trimestres)
# ===================================================================
cat("\n--- 4. Rolling spillover (window=100) ---\n")
rolling <- compute_rolling_spillover(mv_diff, mv$dates[-1], p = 2, H = 10, window = 100)
cat(sprintf("  %d janelas calculadas\n", nrow(rolling)))
cat(sprintf("  Total spillover: media = %.4f, min = %.4f, max = %.4f\n",
            mean(rolling$total_spillover, na.rm = TRUE),
            min(rolling$total_spillover, na.rm = TRUE),
            max(rolling$total_spillover, na.rm = TRUE)))

# ===================================================================
# Salvar resultados
# ===================================================================

# FEVD table (sintetico)
fevd_file <- file.path(output_dir, "spillover_fevd.csv")
fevd_df <- as.data.frame(sp_synthetic$fevd_matrix)
colnames(fevd_df) <- paste0("var_", 1:4)
rownames(fevd_df) <- paste0("var_", 1:4)
write.csv(fevd_df, fevd_file)
cat(sprintf("\nFEVD salvo em: %s\n", fevd_file))

# Spillover summary
summary_df <- data.frame(
  variable = paste0("var_", 1:4),
  from = sp_synthetic$directional_from,
  to = sp_synthetic$directional_to,
  net = sp_synthetic$net_spillover
)
summary_file <- file.path(output_dir, "spillover_summary.csv")
write.csv(summary_df, summary_file, row.names = FALSE)
cat(sprintf("Spillover summary salvo em: %s\n", summary_file))

# Rolling spillover
rolling_file <- file.path(output_dir, "rolling_spillover.csv")
write.csv(rolling, rolling_file, row.names = FALSE)
cat(sprintf("Rolling spillover salvo em: %s\n", rolling_file))

# Sensitivity
sensitivity_file <- file.path(output_dir, "spillover_sensitivity.csv")
sens_df <- rbind(
  data.frame(param = "horizon", value = sensitivity_H$H,
             total_spillover = sensitivity_H$total_spillover),
  data.frame(param = "lags", value = sensitivity_p$p,
             total_spillover = sensitivity_p$total_spillover)
)
write.csv(sens_df, sensitivity_file, row.names = FALSE)
cat(sprintf("Sensitivity salvo em: %s\n", sensitivity_file))

# GDP spillover
gdp_summary <- data.frame(
  variable = c("PIB_EUA", "PIB_Brasil"),
  from = sp_gdp$directional_from,
  to = sp_gdp$directional_to,
  net = sp_gdp$net_spillover
)
gdp_file <- file.path(output_dir, "spillover_gdp.csv")
write.csv(gdp_summary, gdp_file, row.names = FALSE)
cat(sprintf("GDP spillover salvo em: %s\n", gdp_file))

cat("\n=== 03_spillover_validation.R concluido com sucesso ===\n")
