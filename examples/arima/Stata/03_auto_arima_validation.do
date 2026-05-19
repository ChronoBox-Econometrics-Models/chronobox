* ==============================================================================
* 03_auto_arima_validation.do
* Validacao cruzada: Selecao automatica de ARIMA via grid search
*
* Objetivo: Iterar sobre combinacoes de (p,d,q) usando forvalues loops,
*           calcular AIC/BIC para cada modelo e selecionar o melhor.
*           Equivalente ao auto.arima do R e pm.auto_arima do Python.
*
* Datasets: airline.csv (log-passengers), nile.csv (flow)
*
* Nota: Stata nao possui auto_arima nativo, entao fazemos grid search
*       manual sobre o espaco de parametros.
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
* SECAO 1: Funcao de grid search ARIMA para Nile
* ==============================================================================

display _newline(2)
display "================================================================"
display "  AUTO ARIMA via Grid Search - Nile"
display "================================================================"

import delimited "`data_dir'/nile.csv", clear varnames(1)

* Preparar serie temporal
generate year = real(substr(date, 1, 4))
tsset year

display "Nile: " _N " observacoes"

* --- Grid search sobre (p, d, q) ---------------------------------------------
* Espaco de busca: p in {0,1,2,3}, d in {0,1,2}, q in {0,1,2,3}
* Limitamos a ordens razoaveis para eficiencia

* Criar dataset temporario para armazenar resultados
tempfile nile_data
save `nile_data'

* Inicializar dataset de resultados
clear
set obs 1
generate int p = .
generate int d = .
generate int q = .
generate double aic = .
generate double bic = .
generate double loglik = .
generate int converged = .
tempfile results_nile
save `results_nile'

* Carregar dados novamente
use `nile_data', clear

* Iterar sobre combinacoes
local best_aic = 1e10
local best_p = 0
local best_d = 0
local best_q = 0
local model_count = 0

display _newline(1)
display "Iniciando grid search (p=0..3, d=0..2, q=0..3)..."
display _newline(1)

forvalues p = 0/3 {
    forvalues d = 0/2 {
        forvalues q = 0/3 {
            * Pular modelo trivial ARIMA(0,0,0)
            if (`p' == 0 & `d' == 0 & `q' == 0) continue

            * Tentar ajustar o modelo
            capture noisily {
                quietly arima flow, arima(`p',`d',`q')
                local conv = e(converged)

                if (`conv' == 1) {
                    local this_aic = -2*e(ll) + 2*e(k)
                    local this_bic = -2*e(ll) + e(k)*ln(e(N))
                    local this_ll  = e(ll)
                    local model_count = `model_count' + 1

                    * Salvar resultado
                    preserve
                    use `results_nile', clear
                    local new_obs = _N + 1
                    set obs `new_obs'
                    replace p = `p' in `new_obs'
                    replace d = `d' in `new_obs'
                    replace q = `q' in `new_obs'
                    replace aic = `this_aic' in `new_obs'
                    replace bic = `this_bic' in `new_obs'
                    replace loglik = `this_ll' in `new_obs'
                    replace converged = 1 in `new_obs'
                    save `results_nile', replace
                    restore

                    * Atualizar melhor modelo
                    if (`this_aic' < `best_aic') {
                        local best_aic = `this_aic'
                        local best_p = `p'
                        local best_d = `d'
                        local best_q = `q'
                    }

                    display "  ARIMA(`p',`d',`q'): AIC = " %10.4f `this_aic' " BIC = " %10.4f `this_bic'
                }
            }
        }
    }
}

display _newline(1)
display "Total de modelos convergidos: `model_count'"
display "Melhor modelo (AIC): ARIMA(`best_p',`best_d',`best_q') com AIC = " %10.4f `best_aic'

* Re-estimar melhor modelo e exibir detalhes
display _newline(1)
display "--- Re-estimando melhor modelo: ARIMA(`best_p',`best_d',`best_q') ---"
arima flow, arima(`best_p',`best_d',`best_q')
estat ic

* Exportar tabela de resultados Nile
preserve
use `results_nile', clear
drop if p == .
sort aic
generate str20 dataset = "nile"
generate str20 model = "ARIMA(" + string(p) + "," + string(d) + "," + string(q) + ")"
order dataset model p d q aic bic loglik converged
export delimited using "`output_dir'/auto_arima_nile.csv", replace
display "Salvo: auto_arima_nile.csv"

* Top 5 modelos
display _newline(1)
display "Top 5 modelos para Nile (por AIC):"
list model aic bic in 1/5, separator(0)
restore

* ==============================================================================
* SECAO 2: Grid search ARIMA para Airline (log)
* ==============================================================================

display _newline(2)
display "================================================================"
display "  AUTO ARIMA via Grid Search - Airline (log)"
display "================================================================"

import delimited "`data_dir'/airline.csv", clear varnames(1)

generate mdate = monthly(substr(date, 1, 7), "YM")
format mdate %tm
tsset mdate
generate double log_passengers = ln(passengers)

display "Airline: " _N " observacoes (log-transformado)"

* Salvar dados
tempfile airline_data
save `airline_data'

* Inicializar dataset de resultados
clear
set obs 1
generate int p = .
generate int d = .
generate int q = .
generate double aic = .
generate double bic = .
generate double loglik = .
generate int converged = .
tempfile results_airline
save `results_airline'

use `airline_data', clear

local best_aic = 1e10
local best_p = 0
local best_d = 0
local best_q = 0
local model_count = 0

display _newline(1)
display "Iniciando grid search (p=0..3, d=0..2, q=0..3)..."
display _newline(1)

forvalues p = 0/3 {
    forvalues d = 0/2 {
        forvalues q = 0/3 {
            if (`p' == 0 & `d' == 0 & `q' == 0) continue

            capture noisily {
                quietly arima log_passengers, arima(`p',`d',`q')
                local conv = e(converged)

                if (`conv' == 1) {
                    local this_aic = -2*e(ll) + 2*e(k)
                    local this_bic = -2*e(ll) + e(k)*ln(e(N))
                    local this_ll  = e(ll)
                    local model_count = `model_count' + 1

                    preserve
                    use `results_airline', clear
                    local new_obs = _N + 1
                    set obs `new_obs'
                    replace p = `p' in `new_obs'
                    replace d = `d' in `new_obs'
                    replace q = `q' in `new_obs'
                    replace aic = `this_aic' in `new_obs'
                    replace bic = `this_bic' in `new_obs'
                    replace loglik = `this_ll' in `new_obs'
                    replace converged = 1 in `new_obs'
                    save `results_airline', replace
                    restore

                    if (`this_aic' < `best_aic') {
                        local best_aic = `this_aic'
                        local best_p = `p'
                        local best_d = `d'
                        local best_q = `q'
                    }

                    display "  ARIMA(`p',`d',`q'): AIC = " %10.4f `this_aic' " BIC = " %10.4f `this_bic'
                }
            }
        }
    }
}

display _newline(1)
display "Total de modelos convergidos: `model_count'"
display "Melhor modelo (AIC): ARIMA(`best_p',`best_d',`best_q') com AIC = " %10.4f `best_aic'

* Re-estimar melhor modelo
display _newline(1)
display "--- Re-estimando melhor modelo: ARIMA(`best_p',`best_d',`best_q') ---"
arima log_passengers, arima(`best_p',`best_d',`best_q')
estat ic

* Exportar tabela de resultados Airline
preserve
use `results_airline', clear
drop if p == .
sort aic
generate str20 dataset = "airline"
generate str20 model = "ARIMA(" + string(p) + "," + string(d) + "," + string(q) + ")"
order dataset model p d q aic bic loglik converged
export delimited using "`output_dir'/auto_arima_airline.csv", replace
display "Salvo: auto_arima_airline.csv"

display _newline(1)
display "Top 5 modelos para Airline (por AIC):"
list model aic bic in 1/5, separator(0)
restore

* ==============================================================================
* SECAO 3: Exportar tabela combinada
* ==============================================================================

display _newline(2)
display "=== Exportando tabela combinada ==="

* Combinar resultados de ambos os datasets
preserve
use `results_nile', clear
drop if p == .
generate str20 dataset = "nile"
generate str20 model = "ARIMA(" + string(p) + "," + string(d) + "," + string(q) + ")"
tempfile combined
save `combined'

use `results_airline', clear
drop if p == .
generate str20 dataset = "airline"
generate str20 model = "ARIMA(" + string(p) + "," + string(d) + "," + string(q) + ")"
append using `combined'

order dataset model p d q aic bic loglik converged
sort dataset aic
export delimited using "`output_dir'/auto_arima_comparison.csv", replace
display "Salvo: auto_arima_comparison.csv"
restore

* ==============================================================================
* SECAO 4: Nota sobre limitacoes
* ==============================================================================

display _newline(2)
display "================================================================"
display "  NOTAS SOBRE A SELECAO AUTOMATICA NO STATA"
display "================================================================"
display _newline(1)
display "1. Stata nao possui um comando auto_arima nativo como o R (forecast)"
display "   ou Python (pmdarima). Este script implementa grid search manual."
display _newline(1)
display "2. O espaco de busca foi limitado a p,q in {0..3} e d in {0..2}."
display "   auto.arima do R usa heuristicas mais sofisticadas (stepwise search)."
display _newline(1)
display "3. Modelos que nao convergem sao descartados silenciosamente."
display "   Diferencas na selecao podem ocorrer por convergencia diferente."
display _newline(1)
display "4. Para SARIMA, seria necessario um loop adicional sobre (P,D,Q,s),"
display "   o que aumenta significativamente o tempo de execucao."

display _newline(2)
display "=== Script 03_auto_arima_validation.do concluido com sucesso ==="
