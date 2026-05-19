* ==============================================================================
* 02_sarima_validation.do
* Validacao cruzada: SARIMA com comando arima do Stata
*
* Objetivo: Reproduzir os modelos SARIMA ajustados pelo chronobox (Python)
*           e pelo forecast (R) usando o Stata como terceira referencia.
*
* Datasets: airline.csv (log-passengers), brazil_ipca.csv
* Modelos:  SARIMA(0,1,1)(0,1,1)[12] e variantes
*
* Nota: No Stata, a sintaxe SARIMA e:
*   arima y, arima(p,d,q) sarima(P,D,Q,s)
*   onde s e o periodo sazonal (12 para dados mensais)
*
* Dependencias: Stata 14+
* ==============================================================================

clear all
set more off
set seed 42

* --- Caminhos -----------------------------------------------------------------
local data_dir   "../data"
local output_dir "../outputs/Stata"

capture mkdir "`output_dir'"

* ==============================================================================
* SECAO 1: Carregar e preparar dados - AIRLINE
* ==============================================================================

display _newline(2)
display "=== Carregando dataset Airline ==="

import delimited "`data_dir'/airline.csv", clear varnames(1)

* Criar variavel temporal mensal
generate mdate = monthly(substr(date, 1, 7), "YM")
format mdate %tm
tsset mdate

* Aplicar log para estabilizar variancia
generate double log_passengers = ln(passengers)

display "Airline: " _N " observacoes (log-transformado)"
summarize log_passengers

* ==============================================================================
* SECAO 2: Baseline ARIMA(1,1,1) sem sazonalidade
* ==============================================================================

display _newline(2)
display "=== Baseline: ARIMA(1,1,1) sem sazonalidade ==="

arima log_passengers, arima(1,1,1)
estat ic
scalar aic_air_noseas = -2*e(ll) + 2*e(k)
scalar bic_air_noseas = -2*e(ll) + e(k)*ln(e(N))
display "AIC (sem sazonal) = " aic_air_noseas

* ==============================================================================
* SECAO 3: SARIMA para Airline
* ==============================================================================

display _newline(2)
display "=== SARIMA para Airline (log) ==="

* --- SARIMA(0,1,1)(0,1,1)[12] - modelo classico Box-Jenkins -----------------
display _newline(1)
display "--- Airline SARIMA(0,1,1)(0,1,1)[12] ---"
arima log_passengers, arima(0,1,1) sarima(0,1,1,12)
estat ic
matrix b_air_s011 = e(b)
scalar aic_air_s011 = -2*e(ll) + 2*e(k)
scalar bic_air_s011 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_air_s011
display "BIC = " bic_air_s011

* Ganho de AIC com sazonalidade
display "Ganho de AIC (sazonal vs nao-sazonal): " aic_air_noseas - aic_air_s011

* Diagnostico de residuos
predict double resid_s011, residuals
corrgram resid_s011, lags(24)
drop resid_s011

* --- SARIMA(1,1,0)(1,1,0)[12] ------------------------------------------------
display _newline(1)
display "--- Airline SARIMA(1,1,0)(1,1,0)[12] ---"
arima log_passengers, arima(1,1,0) sarima(1,1,0,12)
estat ic
matrix b_air_s110 = e(b)
scalar aic_air_s110 = -2*e(ll) + 2*e(k)
scalar bic_air_s110 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_air_s110

* --- SARIMA(1,1,1)(0,1,1)[12] ------------------------------------------------
display _newline(1)
display "--- Airline SARIMA(1,1,1)(0,1,1)[12] ---"
arima log_passengers, arima(1,1,1) sarima(0,1,1,12)
estat ic
matrix b_air_s111 = e(b)
scalar aic_air_s111 = -2*e(ll) + 2*e(k)
scalar bic_air_s111 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_air_s111

* ==============================================================================
* SECAO 4: Exportar resultados SARIMA Airline
* ==============================================================================

display _newline(2)
display "=== Exportando resultados SARIMA Airline ==="

preserve
clear
set obs 10
generate str20 dataset = "airline"
generate str40 model = ""
generate str15 param = ""
generate double value = .
generate double aic = .
generate double bic = .

* SARIMA(0,1,1)(0,1,1)[12]: ma1, sma1
local row = 1
replace model = "SARIMA(0,1,1)(0,1,1)[12]" in `row'
replace param = "ma.L1" in `row'
replace value = b_air_s011[1,1] in `row'
replace aic = aic_air_s011 in `row'
replace bic = bic_air_s011 in `row'

local row = 2
replace model = "SARIMA(0,1,1)(0,1,1)[12]" in `row'
replace param = "sma.L1" in `row'
replace value = b_air_s011[1,2] in `row'
replace aic = aic_air_s011 in `row'
replace bic = bic_air_s011 in `row'

* SARIMA(1,1,0)(1,1,0)[12]: ar1, sar1
local row = 3
replace model = "SARIMA(1,1,0)(1,1,0)[12]" in `row'
replace param = "ar.L1" in `row'
replace value = b_air_s110[1,1] in `row'
replace aic = aic_air_s110 in `row'
replace bic = bic_air_s110 in `row'

local row = 4
replace model = "SARIMA(1,1,0)(1,1,0)[12]" in `row'
replace param = "sar.L1" in `row'
replace value = b_air_s110[1,2] in `row'
replace aic = aic_air_s110 in `row'
replace bic = bic_air_s110 in `row'

* SARIMA(1,1,1)(0,1,1)[12]: ar1, ma1, sma1
local row = 5
replace model = "SARIMA(1,1,1)(0,1,1)[12]" in `row'
replace param = "ar.L1" in `row'
replace value = b_air_s111[1,1] in `row'
replace aic = aic_air_s111 in `row'
replace bic = bic_air_s111 in `row'

local row = 6
replace model = "SARIMA(1,1,1)(0,1,1)[12]" in `row'
replace param = "ma.L1" in `row'
replace value = b_air_s111[1,2] in `row'
replace aic = aic_air_s111 in `row'
replace bic = bic_air_s111 in `row'

local row = 7
replace model = "SARIMA(1,1,1)(0,1,1)[12]" in `row'
replace param = "sma.L1" in `row'
replace value = b_air_s111[1,3] in `row'
replace aic = aic_air_s111 in `row'
replace bic = bic_air_s111 in `row'

drop if model == ""
export delimited using "`output_dir'/sarima_airline_results.csv", replace
display "Salvo: sarima_airline_results.csv"
restore

* ==============================================================================
* SECAO 5: Carregar e preparar dados - IPCA
* ==============================================================================

display _newline(2)
display "=== Carregando dataset IPCA ==="

import delimited "`data_dir'/brazil_ipca.csv", clear varnames(1)

* Criar variavel temporal mensal
generate mdate = monthly(substr(date, 1, 7), "YM")
format mdate %tm
tsset mdate

display "IPCA: " _N " observacoes"
summarize ipca

* ==============================================================================
* SECAO 6: SARIMA para IPCA
* ==============================================================================

display _newline(2)
display "=== SARIMA para IPCA ==="

* --- SARIMA(1,1,0)(1,1,0)[12] ------------------------------------------------
display _newline(1)
display "--- IPCA SARIMA(1,1,0)(1,1,0)[12] ---"
arima ipca, arima(1,1,0) sarima(1,1,0,12)
estat ic
matrix b_ipca_s1 = e(b)
scalar aic_ipca_s1 = -2*e(ll) + 2*e(k)
scalar bic_ipca_s1 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_ipca_s1

* --- SARIMA(0,1,1)(0,1,1)[12] ------------------------------------------------
display _newline(1)
display "--- IPCA SARIMA(0,1,1)(0,1,1)[12] ---"
arima ipca, arima(0,1,1) sarima(0,1,1,12)
estat ic
matrix b_ipca_s2 = e(b)
scalar aic_ipca_s2 = -2*e(ll) + 2*e(k)
scalar bic_ipca_s2 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_ipca_s2

* --- SARIMA(1,1,1)(0,1,1)[12] ------------------------------------------------
display _newline(1)
display "--- IPCA SARIMA(1,1,1)(0,1,1)[12] ---"
arima ipca, arima(1,1,1) sarima(0,1,1,12)
estat ic
matrix b_ipca_s3 = e(b)
scalar aic_ipca_s3 = -2*e(ll) + 2*e(k)
scalar bic_ipca_s3 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_ipca_s3

* --- SARIMA(1,1,0)(0,1,1)[12] ------------------------------------------------
display _newline(1)
display "--- IPCA SARIMA(1,1,0)(0,1,1)[12] ---"
arima ipca, arima(1,1,0) sarima(0,1,1,12)
estat ic
matrix b_ipca_s4 = e(b)
scalar aic_ipca_s4 = -2*e(ll) + 2*e(k)
scalar bic_ipca_s4 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_ipca_s4

* --- SARIMA(0,1,1)(1,1,1)[12] ------------------------------------------------
display _newline(1)
display "--- IPCA SARIMA(0,1,1)(1,1,1)[12] ---"
arima ipca, arima(0,1,1) sarima(1,1,1,12)
estat ic
matrix b_ipca_s5 = e(b)
scalar aic_ipca_s5 = -2*e(ll) + 2*e(k)
scalar bic_ipca_s5 = -2*e(ll) + e(k)*ln(e(N))
display "AIC = " aic_ipca_s5

* ==============================================================================
* SECAO 7: Exportar resultados SARIMA IPCA
* ==============================================================================

display _newline(2)
display "=== Exportando resultados SARIMA IPCA ==="

preserve
clear
set obs 15
generate str20 dataset = "ipca"
generate str40 model = ""
generate str15 param = ""
generate double value = .
generate double aic = .
generate double bic = .

* SARIMA(1,1,0)(1,1,0)[12]: ar1, sar1
local row = 1
replace model = "SARIMA(1,1,0)(1,1,0)[12]" in `row'
replace param = "ar.L1" in `row'
replace value = b_ipca_s1[1,1] in `row'
replace aic = aic_ipca_s1 in `row'
replace bic = bic_ipca_s1 in `row'

local row = 2
replace model = "SARIMA(1,1,0)(1,1,0)[12]" in `row'
replace param = "sar.L1" in `row'
replace value = b_ipca_s1[1,2] in `row'
replace aic = aic_ipca_s1 in `row'
replace bic = bic_ipca_s1 in `row'

* SARIMA(0,1,1)(0,1,1)[12]: ma1, sma1
local row = 3
replace model = "SARIMA(0,1,1)(0,1,1)[12]" in `row'
replace param = "ma.L1" in `row'
replace value = b_ipca_s2[1,1] in `row'
replace aic = aic_ipca_s2 in `row'
replace bic = bic_ipca_s2 in `row'

local row = 4
replace model = "SARIMA(0,1,1)(0,1,1)[12]" in `row'
replace param = "sma.L1" in `row'
replace value = b_ipca_s2[1,2] in `row'
replace aic = aic_ipca_s2 in `row'
replace bic = bic_ipca_s2 in `row'

* SARIMA(1,1,1)(0,1,1)[12]: ar1, ma1, sma1
local row = 5
replace model = "SARIMA(1,1,1)(0,1,1)[12]" in `row'
replace param = "ar.L1" in `row'
replace value = b_ipca_s3[1,1] in `row'
replace aic = aic_ipca_s3 in `row'
replace bic = bic_ipca_s3 in `row'

local row = 6
replace model = "SARIMA(1,1,1)(0,1,1)[12]" in `row'
replace param = "ma.L1" in `row'
replace value = b_ipca_s3[1,2] in `row'
replace aic = aic_ipca_s3 in `row'
replace bic = bic_ipca_s3 in `row'

local row = 7
replace model = "SARIMA(1,1,1)(0,1,1)[12]" in `row'
replace param = "sma.L1" in `row'
replace value = b_ipca_s3[1,3] in `row'
replace aic = aic_ipca_s3 in `row'
replace bic = bic_ipca_s3 in `row'

* SARIMA(1,1,0)(0,1,1)[12]: ar1, sma1
local row = 8
replace model = "SARIMA(1,1,0)(0,1,1)[12]" in `row'
replace param = "ar.L1" in `row'
replace value = b_ipca_s4[1,1] in `row'
replace aic = aic_ipca_s4 in `row'
replace bic = bic_ipca_s4 in `row'

local row = 9
replace model = "SARIMA(1,1,0)(0,1,1)[12]" in `row'
replace param = "sma.L1" in `row'
replace value = b_ipca_s4[1,2] in `row'
replace aic = aic_ipca_s4 in `row'
replace bic = bic_ipca_s4 in `row'

* SARIMA(0,1,1)(1,1,1)[12]: ma1, sar1, sma1
local row = 10
replace model = "SARIMA(0,1,1)(1,1,1)[12]" in `row'
replace param = "ma.L1" in `row'
replace value = b_ipca_s5[1,1] in `row'
replace aic = aic_ipca_s5 in `row'
replace bic = bic_ipca_s5 in `row'

local row = 11
replace model = "SARIMA(0,1,1)(1,1,1)[12]" in `row'
replace param = "sar.L1" in `row'
replace value = b_ipca_s5[1,2] in `row'
replace aic = aic_ipca_s5 in `row'
replace bic = bic_ipca_s5 in `row'

local row = 12
replace model = "SARIMA(0,1,1)(1,1,1)[12]" in `row'
replace param = "sma.L1" in `row'
replace value = b_ipca_s5[1,3] in `row'
replace aic = aic_ipca_s5 in `row'
replace bic = bic_ipca_s5 in `row'

drop if model == ""
export delimited using "`output_dir'/sarima_ipca_results.csv", replace
display "Salvo: sarima_ipca_results.csv"
restore

* ==============================================================================
* SECAO 8: Tabela comparativa de AIC
* ==============================================================================

display _newline(2)
display "=== Comparacao de AIC SARIMA ==="

display _newline(1)
display "Airline:"
display "  ARIMA(1,1,1) [sem sazonal]:      AIC = " aic_air_noseas
display "  SARIMA(0,1,1)(0,1,1)[12]:        AIC = " aic_air_s011
display "  SARIMA(1,1,0)(1,1,0)[12]:        AIC = " aic_air_s110
display "  SARIMA(1,1,1)(0,1,1)[12]:        AIC = " aic_air_s111

display _newline(1)
display "IPCA:"
display "  SARIMA(1,1,0)(1,1,0)[12]:        AIC = " aic_ipca_s1
display "  SARIMA(0,1,1)(0,1,1)[12]:        AIC = " aic_ipca_s2
display "  SARIMA(1,1,1)(0,1,1)[12]:        AIC = " aic_ipca_s3
display "  SARIMA(1,1,0)(0,1,1)[12]:        AIC = " aic_ipca_s4
display "  SARIMA(0,1,1)(1,1,1)[12]:        AIC = " aic_ipca_s5

display _newline(2)
display "=== Script 02_sarima_validation.do concluido com sucesso ==="
