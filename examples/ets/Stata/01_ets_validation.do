* =============================================================================
* 01_ets_validation.do
* Validacao cruzada: Suavizacao Exponencial Simples e Dupla (Holt)
*
* Usa tssmooth exponential (SES) e tssmooth dexponential (Holt)
* para comparar com resultados do chronobox (Python) e forecast (R).
*
* LIMITACOES DO STATA PARA ETS:
*   - tssmooth so implementa suavizacao exponencial classica, nao o
*     framework ETS completo (error-trend-season taxonomy).
*   - Nao ha selecao automatica de modelo (auto-ETS).
*   - tssmooth exponential equivale a ETS(A,N,N) ou SES.
*   - tssmooth dexponential equivale a suavizacao de Holt (trend linear),
*     mas nao oferece damped trend nativamente.
*   - Nao ha equivalente direto de modelos multiplicativos (M,*,*).
*   - Criterios de informacao (AIC/AICc/BIC) nao sao reportados.
*
* Datasets: airline.csv, ets_synthetic.csv
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de diretorios ---
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

* Criar diretorio de saida se nao existir
capture mkdir "`output_dir'"

display _newline
display "============================================="
display " ETS Validation - Suavizacao Simples e Dupla"
display "============================================="
display _newline

* =============================================================================
* 1. AIRLINE DATASET - Suavizacao Exponencial Simples (SES)
* =============================================================================

display "--- Carregando airline.csv ---"
import delimited "`data_dir'/airline.csv", clear

* Criar variavel temporal mensal
generate month_id = _n
tsset month_id

* Renomear para facilitar
rename passengers y

display "Observacoes: " _N
summarize y, detail

* --- SES com alpha otimizado ---
display _newline
display "=== SES (alpha otimizado) ==="
tssmooth exponential ses_opt = y
* tssmooth exponential otimiza alpha automaticamente

* Capturar alpha estimado
display "Alpha estimado (otimizado): " r(alpha)
local alpha_ses_opt = r(alpha)

* Calcular residuos e metricas
generate resid_ses_opt = y - ses_opt
quietly summarize resid_ses_opt
local mean_resid = r(mean)
generate resid_sq = resid_ses_opt^2
quietly summarize resid_sq
local mse_ses_opt = r(mean)
local rmse_ses_opt = sqrt(`mse_ses_opt')
generate resid_abs = abs(resid_ses_opt)
quietly summarize resid_abs
local mae_ses_opt = r(mean)
generate pct_err = abs(resid_ses_opt / y) * 100
quietly summarize pct_err
local mape_ses_opt = r(mean)
drop resid_sq resid_abs pct_err

display "RMSE: " %9.4f `rmse_ses_opt'
display "MAE:  " %9.4f `mae_ses_opt'
display "MAPE: " %9.4f `mape_ses_opt' "%"

* --- SES com alpha fixo = 0.2 ---
display _newline
display "=== SES (alpha = 0.2) ==="
tssmooth exponential ses_02 = y, parms(0.2)

generate resid_ses_02 = y - ses_02
generate resid_sq = resid_ses_02^2
quietly summarize resid_sq
local rmse_ses_02 = sqrt(r(mean))
generate resid_abs = abs(resid_ses_02)
quietly summarize resid_abs
local mae_ses_02 = r(mean)
generate pct_err = abs(resid_ses_02 / y) * 100
quietly summarize pct_err
local mape_ses_02 = r(mean)
drop resid_sq resid_abs pct_err

display "RMSE: " %9.4f `rmse_ses_02'
display "MAE:  " %9.4f `mae_ses_02'
display "MAPE: " %9.4f `mape_ses_02' "%"

* --- SES com alpha fixo = 0.5 ---
display _newline
display "=== SES (alpha = 0.5) ==="
tssmooth exponential ses_05 = y, parms(0.5)

generate resid_ses_05 = y - ses_05
generate resid_sq = resid_ses_05^2
quietly summarize resid_sq
local rmse_ses_05 = sqrt(r(mean))
generate resid_abs = abs(resid_ses_05)
quietly summarize resid_abs
local mae_ses_05 = r(mean)
generate pct_err = abs(resid_ses_05 / y) * 100
quietly summarize pct_err
local mape_ses_05 = r(mean)
drop resid_sq resid_abs pct_err

display "RMSE: " %9.4f `rmse_ses_05'
display "MAE:  " %9.4f `mae_ses_05'
display "MAPE: " %9.4f `mape_ses_05' "%"

* =============================================================================
* 2. AIRLINE DATASET - Suavizacao Dupla (Holt / dexponential)
* =============================================================================

display _newline
display "=== Suavizacao Dupla de Holt (dexponential, otimizado) ==="
* tssmooth dexponential ajusta tanto alpha quanto beta
tssmooth dexponential holt_opt = y

local alpha_holt = r(alpha)
local beta_holt = r(beta)
display "Alpha: " %9.4f `alpha_holt'
display "Beta:  " %9.4f `beta_holt'

* Metricas
generate resid_holt = y - holt_opt
generate resid_sq = resid_holt^2
quietly summarize resid_sq
local rmse_holt = sqrt(r(mean))
generate resid_abs = abs(resid_holt)
quietly summarize resid_abs
local mae_holt = r(mean)
generate pct_err = abs(resid_holt / y) * 100
quietly summarize pct_err
local mape_holt = r(mean)
drop resid_sq resid_abs pct_err

display "RMSE: " %9.4f `rmse_holt'
display "MAE:  " %9.4f `mae_holt'
display "MAPE: " %9.4f `mape_holt' "%"

* --- Holt com parametros fixos ---
display _newline
display "=== Suavizacao Dupla (alpha=0.3, beta=0.1) ==="
tssmooth dexponential holt_fixed = y, parms(0.3 0.1)

generate resid_holt_f = y - holt_fixed
generate resid_sq = resid_holt_f^2
quietly summarize resid_sq
local rmse_holt_f = sqrt(r(mean))
generate resid_abs = abs(resid_holt_f)
quietly summarize resid_abs
local mae_holt_f = r(mean)
generate pct_err = abs(resid_holt_f / y) * 100
quietly summarize pct_err
local mape_holt_f = r(mean)
drop resid_sq resid_abs pct_err

display "RMSE: " %9.4f `rmse_holt_f'
display "MAE:  " %9.4f `mae_holt_f'
display "MAPE: " %9.4f `mape_holt_f' "%"

* =============================================================================
* 3. ETS SYNTHETIC DATASET
* =============================================================================

display _newline
display "============================================="
display " ETS Synthetic Dataset"
display "============================================="

import delimited "`data_dir'/ets_synthetic.csv", clear

generate month_id = _n
tsset month_id
rename value y

* SES otimizado
tssmooth exponential ses_synth = y
local alpha_synth = r(alpha)

generate resid_synth = y - ses_synth
generate resid_sq = resid_synth^2
quietly summarize resid_sq
local rmse_synth_ses = sqrt(r(mean))

display "SES alpha: " %9.4f `alpha_synth'
display "SES RMSE:  " %9.4f `rmse_synth_ses'

* Holt otimizado
tssmooth dexponential holt_synth = y
local alpha_synth_h = r(alpha)
local beta_synth_h = r(beta)

generate resid_synth_h = y - holt_synth
generate resid_sq2 = resid_synth_h^2
quietly summarize resid_sq2
local rmse_synth_holt = sqrt(r(mean))

display "Holt alpha: " %9.4f `alpha_synth_h'
display "Holt beta:  " %9.4f `beta_synth_h'
display "Holt RMSE:  " %9.4f `rmse_synth_holt'

* =============================================================================
* 4. Exportar resultados
* =============================================================================

display _newline
display "--- Exportando resultados ---"

* Salvar coeficientes
clear
set obs 4
generate str20 model = ""
generate double alpha = .
generate double beta = .
generate double rmse = .
generate double mae = .
generate double mape = .

replace model = "SES_opt" in 1
replace alpha = `alpha_ses_opt' in 1
replace rmse = `rmse_ses_opt' in 1
replace mae = `mae_ses_opt' in 1
replace mape = `mape_ses_opt' in 1

replace model = "SES_0.2" in 2
replace alpha = 0.2 in 2
replace rmse = `rmse_ses_02' in 2
replace mae = `mae_ses_02' in 2
replace mape = `mape_ses_02' in 2

replace model = "Holt_opt" in 3
replace alpha = `alpha_holt' in 3
replace beta = `beta_holt' in 3
replace rmse = `rmse_holt' in 3
replace mae = `mae_holt' in 3
replace mape = `mape_holt' in 3

replace model = "Holt_fixed" in 4
replace alpha = 0.3 in 4
replace beta = 0.1 in 4
replace rmse = `rmse_holt_f' in 4
replace mae = `mae_holt_f' in 4
replace mape = `mape_holt_f' in 4

export delimited "`output_dir'/ets_basic_results.csv", replace
display "Salvo: ets_basic_results.csv"

display _newline
display "============================================="
display " 01_ets_validation.do concluido!"
display "============================================="
display _newline
display "NOTA: tssmooth exponential = SES (equivale a ETS(A,N,N))"
display "      tssmooth dexponential = Holt (equivale a trend linear)"
display "      Stata NAO reporta AIC/AICc/BIC para tssmooth."
display "      Modelos multiplicativos NAO sao suportados via tssmooth."
