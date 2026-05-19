* ==============================================================================
* 01_arima_validation.do
* Validacao cruzada: ARIMA com comando arima do Stata
*
* Objetivo: Reproduzir os modelos ARIMA ajustados pelo chronobox (Python)
*           e pelo forecast (R) usando o Stata como terceira referencia.
*
* Datasets: airline.csv (log-passengers), nile.csv (flow)
* Modelos:  ARIMA(1,1,1) e ARIMA(2,1,2) para airline
*           ARIMA(1,1,0), ARIMA(0,1,1), ARIMA(1,1,1), ARIMA(2,1,1) para nile
*
* Dependencias: Stata 14+ (comando arima)
* ==============================================================================

clear all
set more off
set seed 42

* --- Caminhos -----------------------------------------------------------------
* Assumindo execucao a partir de examples/arima/Stata/
local data_dir   "../data"
local output_dir "../outputs/Stata"

* Criar diretorio de saida
capture mkdir "`output_dir'"

* ==============================================================================
* SECAO 1: Carregar e preparar dados - NILE
* ==============================================================================

display _newline(2)
display "=== Carregando dataset Nile ==="

import delimited "`data_dir'/nile.csv", clear varnames(1)

* Converter data string para formato Stata
generate year = real(substr(date, 1, 4))

* Declarar serie temporal com tsset
tsset year

* Verificar dados carregados
display "Nile: " _N " observacoes"
summarize flow

* ==============================================================================
* SECAO 2: Ajustar modelos ARIMA para Nile
* ==============================================================================

display _newline(2)
display "=== Modelos ARIMA para Nile ==="

* --- ARIMA(1,1,0) -------------------------------------------------------------
display _newline(1)
display "--- Nile ARIMA(1,1,0) ---"
arima flow, arima(1,1,0)
estat ic
* Salvar coeficientes
matrix b_nile_110 = e(b)
scalar aic_nile_110 = -2*e(ll) + 2*e(k)
scalar bic_nile_110 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_nile_110
display "BIC = " bic_nile_110

* Diagnostico de residuos
predict double resid_nile_110, residuals
corrgram resid_nile_110, lags(20)
drop resid_nile_110

* --- ARIMA(0,1,1) -------------------------------------------------------------
display _newline(1)
display "--- Nile ARIMA(0,1,1) ---"
arima flow, arima(0,1,1)
estat ic
matrix b_nile_011 = e(b)
scalar aic_nile_011 = -2*e(ll) + 2*e(k)
scalar bic_nile_011 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_nile_011
display "BIC = " bic_nile_011

predict double resid_nile_011, residuals
corrgram resid_nile_011, lags(20)
drop resid_nile_011

* --- ARIMA(1,1,1) - modelo principal ------------------------------------------
display _newline(1)
display "--- Nile ARIMA(1,1,1) ---"
arima flow, arima(1,1,1)
estat ic
matrix b_nile_111 = e(b)
scalar aic_nile_111 = -2*e(ll) + 2*e(k)
scalar bic_nile_111 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_nile_111
display "BIC = " bic_nile_111

* Diagnostico detalhado de residuos
predict double resid_nile_111, residuals
corrgram resid_nile_111, lags(20)

* Previsao 10 passos a frente
* (Stata predict gera previsao in-sample; para out-of-sample, expandimos)
predict double yhat_nile_111, y
display "Previsao in-sample gerada para Nile ARIMA(1,1,1)"
drop resid_nile_111 yhat_nile_111

* --- ARIMA(2,1,1) -------------------------------------------------------------
display _newline(1)
display "--- Nile ARIMA(2,1,1) ---"
arima flow, arima(2,1,1)
estat ic
matrix b_nile_211 = e(b)
scalar aic_nile_211 = -2*e(ll) + 2*e(k)
scalar bic_nile_211 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_nile_211
display "BIC = " bic_nile_211

predict double resid_nile_211, residuals
corrgram resid_nile_211, lags(20)
drop resid_nile_211

* ==============================================================================
* SECAO 3: Exportar resultados Nile
* ==============================================================================

display _newline(2)
display "=== Exportando resultados Nile ==="

* Criar dataset de coeficientes Nile
preserve
clear
set obs 10
generate str20 dataset = "nile"
generate str20 model = ""
generate str10 param = ""
generate double value = .
generate double aic = .
generate double bic = .

* ARIMA(1,1,0): ar1
local row = 1
replace model = "ARIMA(1,1,0)" in `row'
replace param = "ar.L1" in `row'
replace value = b_nile_110[1,1] in `row'
replace aic = aic_nile_110 in `row'
replace bic = bic_nile_110 in `row'

* ARIMA(0,1,1): ma1
local row = 2
replace model = "ARIMA(0,1,1)" in `row'
replace param = "ma.L1" in `row'
replace value = b_nile_011[1,1] in `row'
replace aic = aic_nile_011 in `row'
replace bic = bic_nile_011 in `row'

* ARIMA(1,1,1): ar1, ma1
local row = 3
replace model = "ARIMA(1,1,1)" in `row'
replace param = "ar.L1" in `row'
replace value = b_nile_111[1,1] in `row'
replace aic = aic_nile_111 in `row'
replace bic = bic_nile_111 in `row'

local row = 4
replace model = "ARIMA(1,1,1)" in `row'
replace param = "ma.L1" in `row'
replace value = b_nile_111[1,2] in `row'
replace aic = aic_nile_111 in `row'
replace bic = bic_nile_111 in `row'

* ARIMA(2,1,1): ar1, ar2, ma1
local row = 5
replace model = "ARIMA(2,1,1)" in `row'
replace param = "ar.L1" in `row'
replace value = b_nile_211[1,1] in `row'
replace aic = aic_nile_211 in `row'
replace bic = bic_nile_211 in `row'

local row = 6
replace model = "ARIMA(2,1,1)" in `row'
replace param = "ar.L2" in `row'
replace value = b_nile_211[1,2] in `row'
replace aic = aic_nile_211 in `row'
replace bic = bic_nile_211 in `row'

local row = 7
replace model = "ARIMA(2,1,1)" in `row'
replace param = "ma.L1" in `row'
replace value = b_nile_211[1,3] in `row'
replace aic = aic_nile_211 in `row'
replace bic = bic_nile_211 in `row'

* Remover linhas vazias
drop if model == ""

export delimited using "`output_dir'/arima_nile_coefficients.csv", replace
display "Salvo: arima_nile_coefficients.csv"
restore

* ==============================================================================
* SECAO 4: Carregar e preparar dados - AIRLINE
* ==============================================================================

display _newline(2)
display "=== Carregando dataset Airline ==="

import delimited "`data_dir'/airline.csv", clear varnames(1)

* Criar variavel temporal mensal
generate mdate = monthly(substr(date, 1, 7), "YM")
format mdate %tm
tsset mdate

* Aplicar log (como no Python e R)
generate double log_passengers = ln(passengers)

display "Airline: " _N " observacoes (log-transformado)"
summarize log_passengers

* ==============================================================================
* SECAO 5: Ajustar modelos ARIMA para Airline (log)
* ==============================================================================

display _newline(2)
display "=== Modelos ARIMA para Airline (log) ==="

* --- ARIMA(1,1,0) -------------------------------------------------------------
display _newline(1)
display "--- Airline ARIMA(1,1,0) ---"
arima log_passengers, arima(1,1,0)
estat ic
matrix b_air_110 = e(b)
scalar aic_air_110 = -2*e(ll) + 2*e(k)
scalar bic_air_110 = -2*e(ll) + e(k)*ln(e(N))

* --- ARIMA(0,1,1) -------------------------------------------------------------
display _newline(1)
display "--- Airline ARIMA(0,1,1) ---"
arima log_passengers, arima(0,1,1)
estat ic
matrix b_air_011 = e(b)
scalar aic_air_011 = -2*e(ll) + 2*e(k)
scalar bic_air_011 = -2*e(ll) + e(k)*ln(e(N))

* --- ARIMA(1,1,1) - modelo principal ------------------------------------------
display _newline(1)
display "--- Airline ARIMA(1,1,1) ---"
arima log_passengers, arima(1,1,1)
estat ic
matrix b_air_111 = e(b)
scalar aic_air_111 = -2*e(ll) + 2*e(k)
scalar bic_air_111 = -2*e(ll) + e(k)*ln(e(N))

* Diagnostico de residuos
predict double resid_air_111, residuals
corrgram resid_air_111, lags(24)
drop resid_air_111

* --- ARIMA(2,1,1) -------------------------------------------------------------
display _newline(1)
display "--- Airline ARIMA(2,1,1) ---"
arima log_passengers, arima(2,1,1)
estat ic
matrix b_air_211 = e(b)
scalar aic_air_211 = -2*e(ll) + 2*e(k)
scalar bic_air_211 = -2*e(ll) + e(k)*ln(e(N))

* --- ARIMA(2,1,2) - modelo completo ------------------------------------------
display _newline(1)
display "--- Airline ARIMA(2,1,2) ---"
arima log_passengers, arima(2,1,2)
estat ic
matrix b_air_212 = e(b)
scalar aic_air_212 = -2*e(ll) + 2*e(k)
scalar bic_air_212 = -2*e(ll) + e(k)*ln(e(N))

* Diagnostico de residuos
predict double resid_air_212, residuals
corrgram resid_air_212, lags(24)

* Previsao in-sample
predict double yhat_air_212, y
display "Previsao in-sample gerada para Airline ARIMA(2,1,2)"
drop resid_air_212 yhat_air_212

* ==============================================================================
* SECAO 6: Exportar resultados Airline
* ==============================================================================

display _newline(2)
display "=== Exportando resultados Airline ==="

preserve
clear
set obs 15
generate str20 dataset = "airline"
generate str20 model = ""
generate str10 param = ""
generate double value = .
generate double aic = .
generate double bic = .

* ARIMA(1,1,0): ar1
local row = 1
replace model = "ARIMA(1,1,0)" in `row'
replace param = "ar.L1" in `row'
replace value = b_air_110[1,1] in `row'
replace aic = aic_air_110 in `row'
replace bic = bic_air_110 in `row'

* ARIMA(0,1,1): ma1
local row = 2
replace model = "ARIMA(0,1,1)" in `row'
replace param = "ma.L1" in `row'
replace value = b_air_011[1,1] in `row'
replace aic = aic_air_011 in `row'
replace bic = bic_air_011 in `row'

* ARIMA(1,1,1): ar1, ma1
local row = 3
replace model = "ARIMA(1,1,1)" in `row'
replace param = "ar.L1" in `row'
replace value = b_air_111[1,1] in `row'
replace aic = aic_air_111 in `row'
replace bic = bic_air_111 in `row'

local row = 4
replace model = "ARIMA(1,1,1)" in `row'
replace param = "ma.L1" in `row'
replace value = b_air_111[1,2] in `row'
replace aic = aic_air_111 in `row'
replace bic = bic_air_111 in `row'

* ARIMA(2,1,1): ar1, ar2, ma1
local row = 5
replace model = "ARIMA(2,1,1)" in `row'
replace param = "ar.L1" in `row'
replace value = b_air_211[1,1] in `row'
replace aic = aic_air_211 in `row'
replace bic = bic_air_211 in `row'

local row = 6
replace model = "ARIMA(2,1,1)" in `row'
replace param = "ar.L2" in `row'
replace value = b_air_211[1,2] in `row'
replace aic = aic_air_211 in `row'
replace bic = bic_air_211 in `row'

local row = 7
replace model = "ARIMA(2,1,1)" in `row'
replace param = "ma.L1" in `row'
replace value = b_air_211[1,3] in `row'
replace aic = aic_air_211 in `row'
replace bic = bic_air_211 in `row'

* ARIMA(2,1,2): ar1, ar2, ma1, ma2
local row = 8
replace model = "ARIMA(2,1,2)" in `row'
replace param = "ar.L1" in `row'
replace value = b_air_212[1,1] in `row'
replace aic = aic_air_212 in `row'
replace bic = bic_air_212 in `row'

local row = 9
replace model = "ARIMA(2,1,2)" in `row'
replace param = "ar.L2" in `row'
replace value = b_air_212[1,2] in `row'
replace aic = aic_air_212 in `row'
replace bic = bic_air_212 in `row'

local row = 10
replace model = "ARIMA(2,1,2)" in `row'
replace param = "ma.L1" in `row'
replace value = b_air_212[1,3] in `row'
replace aic = aic_air_212 in `row'
replace bic = bic_air_212 in `row'

local row = 11
replace model = "ARIMA(2,1,2)" in `row'
replace param = "ma.L2" in `row'
replace value = b_air_212[1,4] in `row'
replace aic = aic_air_212 in `row'
replace bic = bic_air_212 in `row'

* Remover linhas vazias
drop if model == ""

export delimited using "`output_dir'/arima_airline_coefficients.csv", replace
display "Salvo: arima_airline_coefficients.csv"
restore

* ==============================================================================
* SECAO 7: Previsao out-of-sample para Airline ARIMA(1,1,1)
* ==============================================================================

display _newline(2)
display "=== Previsao out-of-sample Airline ARIMA(1,1,1) ==="

* Re-estimar o modelo
arima log_passengers, arima(1,1,1)

* Expandir dataset para previsao (12 meses a frente)
local last_obs = _N
local last_mdate = mdate[`last_obs']
set obs `=`last_obs' + 12'

* Preencher datas futuras
forvalues i = 1/12 {
    local new_obs = `last_obs' + `i'
    replace mdate = `last_mdate' + `i' in `new_obs'
}

tsset mdate
predict double forecast_air_111, y dynamic(`=`last_mdate'+1')

* Exportar previsoes
preserve
keep if mdate > `last_mdate'
generate step = _n
keep step forecast_air_111
rename forecast_air_111 forecast
export delimited using "`output_dir'/arima_forecasts_airline.csv", replace
display "Salvo: arima_forecasts_airline.csv"
restore

* Limpar observacoes extras
drop if mdate > `last_mdate'
tsset mdate

* ==============================================================================
* SECAO 8: Resumo comparativo
* ==============================================================================

display _newline(2)
display "=== Resumo dos modelos ==="
display _newline(1)

display "Nile:"
display "  ARIMA(1,1,0) AIC = " aic_nile_110 " BIC = " bic_nile_110
display "  ARIMA(0,1,1) AIC = " aic_nile_011 " BIC = " bic_nile_011
display "  ARIMA(1,1,1) AIC = " aic_nile_111 " BIC = " bic_nile_111
display "  ARIMA(2,1,1) AIC = " aic_nile_211 " BIC = " bic_nile_211

display _newline(1)
display "Airline (log):"
display "  ARIMA(1,1,0) AIC = " aic_air_110 " BIC = " bic_air_110
display "  ARIMA(0,1,1) AIC = " aic_air_011 " BIC = " bic_air_011
display "  ARIMA(1,1,1) AIC = " aic_air_111 " BIC = " bic_air_111
display "  ARIMA(2,1,1) AIC = " aic_air_211 " BIC = " bic_air_211
display "  ARIMA(2,1,2) AIC = " aic_air_212 " BIC = " bic_air_212

display _newline(2)
display "=== Script 01_arima_validation.do concluido com sucesso ==="
