##############################################################################
# 02_hamilton_bn_validation.R
#
# Validacao cruzada do filtro de Hamilton e decomposicao Beveridge-Nelson
# usando neverhpfilter::yth_filter() como referencia para Hamilton, e
# implementacao manual de BN decomposition.
#
# Hamilton filter:
#   - Regressao de y(t+h) sobre y(t), y(t-1), ..., y(t-p+1)
#   - h = 8 (padrao para dados trimestrais, 2 anos a frente)
#   - p = 4 (4 lags na regressao auxiliar)
#   - Ciclo = residuo da regressao
#
# Beveridge-Nelson decomposition:
#   - Baseada em modelo AR(p)
#   - Tendencia = limite da previsao de longo prazo
#   - Ciclo = y(t) - tendencia(t)
#   - Testamos AR(1), AR(2), AR(4), AR(8)
#
# Datasets: examples/filters/data/us_gdp_quarterly.csv
#           examples/filters/data/brazil_gdp.csv
##############################################################################

# Pacotes necessarios
if (!require("neverhpfilter", quietly = TRUE)) {
  install.packages("neverhpfilter", repos = "https://cloud.r-project.org")
  library(neverhpfilter)
}
if (!require("xts", quietly = TRUE)) {
  install.packages("xts", repos = "https://cloud.r-project.org")
  library(xts)
}

set.seed(42)

# --- Caminhos ---
base_dir <- file.path(dirname(sys.frame(1)$ofile), "..")
data_dir <- file.path(base_dir, "data")
output_dir <- file.path(base_dir, "outputs", "R")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Validacao Hamilton Filter e BN Decomposition ===\n\n")

# -------------------------------------------------------------------
# Beveridge-Nelson decomposition manual via AR(p)
#
# Para um AR(p): y_t = c + phi_1*y_{t-1} + ... + phi_p*y_{t-p} + e_t
# A tendencia BN eh: tau_t = y_t + sum_{j=1}^{inf} E[Delta y_{t+j} | I_t]
# Na pratica, usamos a representacao companion para calcular
# a soma das previsoes futuras do crescimento.
# -------------------------------------------------------------------
bn_decomposition <- function(y, ar_order = 4) {
  n <- length(y)
  dy <- diff(y)  # primeira diferenca

  # Ajustar AR(p) na primeira diferenca
  if (ar_order >= length(dy)) {
    stop("ar_order muito grande para a serie")
  }

  # Construir matriz de regressao
  X <- matrix(NA, nrow = length(dy) - ar_order, ncol = ar_order)
  for (j in 1:ar_order) {
    X[, j] <- dy[(ar_order - j + 1):(length(dy) - j)]
  }
  Y <- dy[(ar_order + 1):length(dy)]

  # OLS
  fit <- lm(Y ~ X)
  phi <- coef(fit)[-1]  # coeficientes AR (sem intercepto)
  mu <- coef(fit)[1]    # intercepto (media do crescimento)

  # Companion matrix para AR(p)
  if (ar_order == 1) {
    A <- matrix(phi, 1, 1)
  } else {
    A <- matrix(0, ar_order, ar_order)
    A[1, ] <- phi
    A[2:ar_order, 1:(ar_order - 1)] <- diag(ar_order - 1)
  }

  # Vetor seletor (pega primeiro elemento do estado)
  e1 <- rep(0, ar_order)
  e1[1] <- 1

  # Soma da funcao impulso-resposta: sum_{j=1}^{inf} A^j
  # = A * (I - A)^{-1}  (se estavel)
  I_mat <- diag(ar_order)

  # Verificar estabilidade
  eigenvalues <- eigen(A)$values
  if (any(abs(eigenvalues) >= 1)) {
    warning("AR model is not stationary, BN decomposition may be unreliable")
  }

  # C_bn = e1' * A * (I - A)^{-1} * e1
  # Mas precisamos da soma completa para o vetor de estado
  inv_I_minus_A <- solve(I_mat - A)
  C_bn_matrix <- A %*% inv_I_minus_A

  # Calcular tendencia BN para cada t
  trend <- rep(NA, n)
  cycle <- rep(NA, n)

  # Para t >= ar_order+1 (precisa dos lags)
  for (t in (ar_order + 2):n) {
    # Estado: [dy_t, dy_{t-1}, ..., dy_{t-p+1}]
    state <- rep(0, ar_order)
    for (j in 1:ar_order) {
      idx <- t - j
      if (idx >= 1 && idx < n) {
        state[j] <- dy[idx]
      }
    }

    # Tendencia BN = y_t + mu/(1-sum(phi)) + e1' * C_bn * state
    long_run_drift <- mu / (1 - sum(phi))
    correction <- as.numeric(t(e1) %*% C_bn_matrix %*% state)
    trend[t] <- y[t] + long_run_drift + correction
    cycle[t] <- y[t] - trend[t]
  }

  return(list(trend = trend, cycle = cycle, ar_order = ar_order, phi = phi, mu = mu))
}

# --- Funcao para processar um pais ---
process_country <- function(csv_file, country_label) {
  cat(sprintf("--- Processando: %s ---\n", country_label))

  df <- read.csv(csv_file, stringsAsFactors = FALSE)
  dates <- as.Date(df$date)
  y_raw <- df$gdp_log

  # -------------------------------------------------------
  # Hamilton Filter via neverhpfilter
  # yth_filter espera um objeto xts
  # h = 8 (8 trimestres = 2 anos a frente)
  # -------------------------------------------------------
  y_xts <- xts(y_raw, order.by = dates)

  hamilton_results <- list()

  for (h_val in c(4, 8, 12, 16)) {
    cat(sprintf("  Hamilton h=%d...\n", h_val))
    tryCatch({
      ham <- yth_filter(y_xts, h = h_val, p = 4)
      # yth_filter retorna colunas: y, trend, cycle (ou similar)
      ham_df <- as.data.frame(ham)
      ham_dates <- index(ham)

      # Extrair ciclo (residuo) - ultima coluna geralmente eh o ciclo
      col_names <- colnames(ham_df)
      cycle_col <- grep("cycle|residual", col_names, ignore.case = TRUE, value = TRUE)
      if (length(cycle_col) == 0) {
        # Se nao tem coluna cycle, o ciclo eh a diferenca y - trend
        cycle_vals <- ham_df[, ncol(ham_df)]
      } else {
        cycle_vals <- ham_df[, cycle_col[1]]
      }

      trend_vals <- ham_df[, grep("trend|yth", col_names, ignore.case = TRUE)[1]]

      valid <- !is.na(cycle_vals)
      cat(sprintf("    cycle std = %.8f (obs validas: %d/%d)\n",
                  sd(cycle_vals[valid]), sum(valid), length(cycle_vals)))

      hamilton_results[[paste0("hamilton_h", h_val)]] <- data.frame(
        date = as.character(ham_dates),
        country = country_label,
        method = paste0("hamilton_h", h_val),
        trend = as.numeric(trend_vals),
        cycle = as.numeric(cycle_vals),
        stringsAsFactors = FALSE
      )
    }, error = function(e) {
      cat(sprintf("    ERRO: %s\n", e$message))
    })
  }

  # -------------------------------------------------------
  # Beveridge-Nelson decomposition
  # Testar com diferentes ordens AR
  # -------------------------------------------------------
  bn_results <- list()

  for (ar_p in c(1, 2, 4, 8)) {
    cat(sprintf("  BN decomposition AR(%d)...\n", ar_p))
    tryCatch({
      bn <- bn_decomposition(y_raw, ar_order = ar_p)
      valid <- !is.na(bn$cycle)
      cat(sprintf("    cycle std = %.8f (obs validas: %d/%d)\n",
                  sd(bn$cycle, na.rm = TRUE), sum(valid), length(bn$cycle)))

      bn_results[[paste0("bn_ar", ar_p)]] <- data.frame(
        date = as.character(dates),
        country = country_label,
        method = paste0("bn_ar", ar_p),
        trend = bn$trend,
        cycle = bn$cycle,
        stringsAsFactors = FALSE
      )
    }, error = function(e) {
      cat(sprintf("    ERRO: %s\n", e$message))
    })
  }

  cat("\n")
  return(list(hamilton = hamilton_results, bn = bn_results))
}

# --- Processar ambos os paises ---
us_file <- file.path(data_dir, "us_gdp_quarterly.csv")
br_file <- file.path(data_dir, "brazil_gdp.csv")

results_us <- process_country(us_file, "US")
results_br <- process_country(br_file, "BR")

# --- Combinar todos os resultados ---
all_dfs <- c(results_us$hamilton, results_us$bn,
             results_br$hamilton, results_br$bn)

all_results <- do.call(rbind, all_dfs)

# --- Salvar ---
output_file <- file.path(output_dir, "hamilton_bn_results.csv")
write.csv(all_results, output_file, row.names = FALSE)
cat(sprintf("Resultados salvos em: %s\n", output_file))

# --- Salvar resumo ---
summary_rows <- list()
for (name in names(all_dfs)) {
  df_tmp <- all_dfs[[name]]
  valid <- !is.na(df_tmp$cycle)
  summary_rows[[name]] <- data.frame(
    method = name,
    country = df_tmp$country[1],
    n_valid = sum(valid),
    n_total = nrow(df_tmp),
    cycle_mean = mean(df_tmp$cycle, na.rm = TRUE),
    cycle_std = sd(df_tmp$cycle, na.rm = TRUE),
    cycle_min = min(df_tmp$cycle, na.rm = TRUE),
    cycle_max = max(df_tmp$cycle, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

summary_df <- do.call(rbind, summary_rows)
summary_file <- file.path(output_dir, "hamilton_bn_summary.csv")
write.csv(summary_df, summary_file, row.names = FALSE)
cat(sprintf("Resumo salvo em: %s\n", summary_file))

cat("\n=== 02_hamilton_bn_validation.R concluido com sucesso ===\n")
