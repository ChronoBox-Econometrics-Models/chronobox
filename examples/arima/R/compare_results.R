# ==============================================================================
# compare_results.R
# Comparacao automatica de resultados Python (chronobox) vs R (forecast)
#
# Objetivo: Ler outputs de ambos os ambientes e calcular diferencas
#           absolutas e relativas para validacao cruzada.
#
# Inputs:  examples/arima/outputs/*.json, *.csv (Python)
#          examples/arima/outputs/R/*.csv (R)
#
# Output:  Tabela de comparacao no console + CSV sumario
#
# Dependencias: jsonlite
# ==============================================================================

set.seed(42)

library(jsonlite)

# Caminhos relativos (assumindo execucao a partir de examples/arima/)
python_dir <- file.path("..", "outputs")
r_dir      <- file.path("..", "outputs", "R")

cat("================================================================\n")
cat("  COMPARACAO DE RESULTADOS: chronobox (Python) vs forecast (R)\n")
cat("================================================================\n\n")

# ==============================================================================
# SECAO 1: Comparacao de coeficientes ARIMA
# ==============================================================================

cat("=== 1. Coeficientes ARIMA ===\n\n")

# Carregar resultados Python (JSON)
py_coefs <- fromJSON(file.path(python_dir, "arima_coefficients.json"))

# Carregar resultados R (CSV)
r_coefs <- read.csv(file.path(r_dir, "arima_coefficients_R.csv"),
                    stringsAsFactors = FALSE)

# Mapear nomes de parametros entre Python e R
# Python: ar.L1, ar.L2, ma.L1, ma.L2
# R:      ar1, ar2, ma1, ma2 (nomes do forecast::Arima)
param_map <- c("ar1" = "ar.L1", "ar2" = "ar.L2",
               "ma1" = "ma.L1", "ma2" = "ma.L2")

# Funcao para comparar um modelo
compare_model <- function(py_model, r_subset, model_key) {
  comparisons <- list()

  for (i in seq_len(nrow(r_subset))) {
    r_param <- r_subset$param[i]
    r_value <- r_subset$value[i]

    # Mapear nome R -> Python
    py_param <- ifelse(r_param %in% names(param_map), param_map[r_param], r_param)

    if (py_param %in% names(py_model)) {
      py_value <- py_model[[py_param]]
      abs_diff <- abs(py_value - r_value)
      rel_diff <- ifelse(abs(py_value) > 1e-10, abs_diff / abs(py_value), NA)

      comparisons[[length(comparisons) + 1]] <- data.frame(
        model     = model_key,
        param     = py_param,
        python    = py_value,
        r         = r_value,
        abs_diff  = abs_diff,
        rel_diff  = rel_diff,
        stringsAsFactors = FALSE
      )
    }
  }

  # Comparar sigma2
  if ("sigma2" %in% names(py_model)) {
    py_s2 <- py_model[["sigma2"]]
    r_s2  <- r_subset$sigma2[1]
    abs_d <- abs(py_s2 - r_s2)
    rel_d <- ifelse(abs(py_s2) > 1e-10, abs_d / abs(py_s2), NA)
    comparisons[[length(comparisons) + 1]] <- data.frame(
      model = model_key, param = "sigma2",
      python = py_s2, r = r_s2,
      abs_diff = abs_d, rel_diff = rel_d,
      stringsAsFactors = FALSE
    )
  }

  # Comparar AIC
  if ("aic" %in% names(py_model)) {
    py_aic <- py_model[["aic"]]
    r_aic  <- r_subset$aic[1]
    abs_d  <- abs(py_aic - r_aic)
    rel_d  <- ifelse(abs(py_aic) > 1e-10, abs_d / abs(py_aic), NA)
    comparisons[[length(comparisons) + 1]] <- data.frame(
      model = model_key, param = "aic",
      python = py_aic, r = r_aic,
      abs_diff = abs_d, rel_diff = rel_d,
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, comparisons)
}

# Iterar sobre modelos que existem em ambos
all_comparisons <- list()

# Mapeamento de chaves Python -> filtros R
model_keys <- list(
  "nile_ARIMA(1,1,0)"    = list(dataset = "nile", model = "ARIMA(1,1,0)"),
  "nile_ARIMA(0,1,1)"    = list(dataset = "nile", model = "ARIMA(0,1,1)"),
  "nile_ARIMA(1,1,1)"    = list(dataset = "nile", model = "ARIMA(1,1,1)"),
  "nile_ARIMA(2,1,1)"    = list(dataset = "nile", model = "ARIMA(2,1,1)"),
  "airline_ARIMA(1,1,0)"  = list(dataset = "airline", model = "ARIMA(1,1,0)"),
  "airline_ARIMA(0,1,1)"  = list(dataset = "airline", model = "ARIMA(0,1,1)"),
  "airline_ARIMA(1,1,1)"  = list(dataset = "airline", model = "ARIMA(1,1,1)"),
  "airline_ARIMA(2,1,1)"  = list(dataset = "airline", model = "ARIMA(2,1,1)"),
  "airline_ARIMA(2,1,2)"  = list(dataset = "airline", model = "ARIMA(2,1,2)")
)

for (py_key in names(model_keys)) {
  if (py_key %in% names(py_coefs)) {
    info <- model_keys[[py_key]]
    r_sub <- r_coefs[r_coefs$dataset == info$dataset & r_coefs$model == info$model, ]

    if (nrow(r_sub) > 0) {
      comp <- compare_model(py_coefs[[py_key]], r_sub, py_key)
      if (!is.null(comp) && nrow(comp) > 0) {
        all_comparisons[[length(all_comparisons) + 1]] <- comp
      }
    }
  }
}

coef_comparison <- do.call(rbind, all_comparisons)
rownames(coef_comparison) <- NULL

cat("Comparacao de coeficientes (Python vs R):\n")
print(coef_comparison, digits = 6)

# Verificar tolerancia
max_coef_diff <- max(coef_comparison$abs_diff[coef_comparison$param != "aic" &
                                                coef_comparison$param != "sigma2"],
                     na.rm = TRUE)
cat(sprintf("\nMaxima diferenca absoluta em coeficientes: %.6e\n", max_coef_diff))
cat(sprintf("Dentro da tolerancia 1e-4? %s\n",
            ifelse(max_coef_diff < 1e-4, "SIM", "NAO (verificar)")))

# ==============================================================================
# SECAO 2: Comparacao de previsoes
# ==============================================================================

cat("\n=== 2. Previsoes ===\n\n")

# Python forecasts
py_fc <- read.csv(file.path(python_dir, "arima_forecasts.csv"),
                  stringsAsFactors = FALSE)

# R forecasts
r_fc <- read.csv(file.path(r_dir, "arima_forecasts_R.csv"),
                 stringsAsFactors = FALSE)

# Merge por dataset + model + step
fc_merged <- merge(py_fc, r_fc,
                   by = c("dataset", "model", "step"),
                   suffixes = c("_py", "_r"))

fc_merged$abs_diff_forecast <- abs(fc_merged$forecast_py - fc_merged$forecast_r)
fc_merged$abs_diff_lower    <- abs(fc_merged$lower_95_py - fc_merged$lower_95_r)
fc_merged$abs_diff_upper    <- abs(fc_merged$upper_95_py - fc_merged$upper_95_r)

cat("Diferencas em previsoes (resumo por modelo):\n")
for (mod in unique(fc_merged$model)) {
  sub <- fc_merged[fc_merged$model == mod, ]
  cat(sprintf("\n  %s (dataset: %s):\n", mod, sub$dataset[1]))
  cat(sprintf("    Forecast: max abs diff = %.6e, mean = %.6e\n",
              max(sub$abs_diff_forecast), mean(sub$abs_diff_forecast)))
  cat(sprintf("    Lower 95: max abs diff = %.6e\n", max(sub$abs_diff_lower)))
  cat(sprintf("    Upper 95: max abs diff = %.6e\n", max(sub$abs_diff_upper)))
}

# ==============================================================================
# SECAO 3: Comparacao SARIMA
# ==============================================================================

cat("\n\n=== 3. Resultados SARIMA ===\n\n")

# Python SARIMA
py_sarima <- fromJSON(file.path(python_dir, "sarima_results.json"))

# R SARIMA
r_sarima <- read.csv(file.path(r_dir, "sarima_results_R.csv"),
                     stringsAsFactors = FALSE)

# Comparar AIC do modelo airline SARIMA(0,1,1)(0,1,1)[12]
py_air_sarima <- py_sarima[["airline_SARIMA(0,1,1)(0,1,1)[12]"]]
r_air_sarima  <- r_sarima[r_sarima$dataset == "airline" &
                           r_sarima$model == "SARIMA(0,1,1)(0,1,1)[12]", ]

if (!is.null(py_air_sarima) && nrow(r_air_sarima) > 0) {
  cat("Airline SARIMA(0,1,1)(0,1,1)[12]:\n")
  cat(sprintf("  AIC Python: %.4f, AIC R: %.4f, diff: %.4e\n",
              py_air_sarima$aic, r_air_sarima$aic[1],
              abs(py_air_sarima$aic - r_air_sarima$aic[1])))
}

# IPCA SARIMA(0,1,1)(0,1,1)[12]
py_ipca_sarima <- py_sarima[["ipca_SARIMA(0,1,1)(0,1,1)[12]"]]
r_ipca_sarima  <- r_sarima[r_sarima$dataset == "ipca" &
                            r_sarima$model == "SARIMA(0,1,1)(0,1,1)[12]", ]

if (!is.null(py_ipca_sarima) && nrow(r_ipca_sarima) > 0) {
  cat("\nIPCA SARIMA(0,1,1)(0,1,1)[12]:\n")
  cat(sprintf("  AIC Python: %.4f, AIC R: %.4f, diff: %.4e\n",
              py_ipca_sarima$aic, r_ipca_sarima$aic[1],
              abs(py_ipca_sarima$aic - r_ipca_sarima$aic[1])))
}

# ==============================================================================
# SECAO 4: Comparacao ARFIMA (d fracionario)
# ==============================================================================

cat("\n=== 4. Parametro d fracionario (ARFIMA) ===\n\n")

# Python ARFIMA
py_arfima <- fromJSON(file.path(python_dir, "arfima_results.json"))

# R ARFIMA
r_arfima <- read.csv(file.path(r_dir, "arfima_results_R.csv"),
                     stringsAsFactors = FALSE)

# Comparar estimadores semi-parametricos
cat("Estimadores semi-parametricos de d:\n")
cat("\nNile:\n")

# GPH
py_nile_gph <- py_arfima$d_estimation$nile$gph$d
r_nile_gph  <- r_arfima$d[r_arfima$dataset == "nile" & r_arfima$method == "GPH"]
if (length(r_nile_gph) > 0) {
  cat(sprintf("  GPH: Python = %.6f, R = %.6f, diff = %.6e\n",
              py_nile_gph, r_nile_gph, abs(py_nile_gph - r_nile_gph)))
}

# Local Whittle (Python) vs Sperio (R) - metodos diferentes
py_nile_lw <- py_arfima$d_estimation$nile$local_whittle$d
r_nile_sp  <- r_arfima$d[r_arfima$dataset == "nile" & r_arfima$method == "Sperio"]
if (length(r_nile_sp) > 0) {
  cat(sprintf("  Local Whittle (Py) vs Sperio (R): %.6f vs %.6f, diff = %.6e\n",
              py_nile_lw, r_nile_sp, abs(py_nile_lw - r_nile_sp)))
  cat("  (Nota: metodos diferentes - comparacao aproximada)\n")
}

cat("\nIPCA:\n")
py_ipca_gph <- py_arfima$d_estimation$ipca$gph$d
r_ipca_gph  <- r_arfima$d[r_arfima$dataset == "ipca" & r_arfima$method == "GPH"]
if (length(r_ipca_gph) > 0) {
  cat(sprintf("  GPH: Python = %.6f, R = %.6f, diff = %.6e\n",
              py_ipca_gph, r_ipca_gph, abs(py_ipca_gph - r_ipca_gph)))
}

# ==============================================================================
# SECAO 5: Sumario geral
# ==============================================================================

cat("\n================================================================\n")
cat("  SUMARIO DA VALIDACAO CRUZADA\n")
cat("================================================================\n\n")

# Montar sumario
summary_df <- data.frame(
  categoria = c("Coeficientes ARIMA", "Previsoes ARIMA",
                 "AIC ARIMA", "Sigma2 ARIMA"),
  max_abs_diff = c(
    max_coef_diff,
    max(fc_merged$abs_diff_forecast, na.rm = TRUE),
    max(coef_comparison$abs_diff[coef_comparison$param == "aic"], na.rm = TRUE),
    max(coef_comparison$abs_diff[coef_comparison$param == "sigma2"], na.rm = TRUE)
  ),
  tolerancia = c(1e-4, 1e-3, 1.0, 1.0),
  stringsAsFactors = FALSE
)

summary_df$status <- ifelse(summary_df$max_abs_diff <= summary_df$tolerancia,
                            "OK", "VERIFICAR")

cat("Resultado por categoria:\n")
print(summary_df, digits = 6)

# Salvar sumario
write.csv(summary_df, file.path(r_dir, "validation_summary.csv"),
          row.names = FALSE)
cat("\nSalvo: validation_summary.csv\n")

# Salvar comparacao detalhada de coeficientes
write.csv(coef_comparison, file.path(r_dir, "coef_comparison_detail.csv"),
          row.names = FALSE)
cat("Salvo: coef_comparison_detail.csv\n")

cat("\n================================================================\n")
cat("  Validacao cruzada concluida.\n")
cat("  Tolerancia para coeficientes: < 1e-4\n")
cat("  Diferencas maiores sao esperadas para:\n")
cat("  - Modelos com parametrizacao diferente (Python CSS-ML vs R ML)\n")
cat("  - Estimadores semi-parametricos (GPH bandwidth pode diferir)\n")
cat("  - auto.arima: diferentes espacos de busca\n")
cat("================================================================\n")
