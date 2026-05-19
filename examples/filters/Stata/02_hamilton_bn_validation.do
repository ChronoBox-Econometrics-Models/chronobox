********************************************************************************
* 02_hamilton_bn_validation.do
*
* Validacao cruzada do filtro de Hamilton e decomposicao Beveridge-Nelson
* usando implementacoes MANUAIS em Stata.
*
* IMPORTANTE: Nem Hamilton filter nem BN decomposition sao comandos nativos
* do Stata. Ambos sao implementados manualmente aqui:
*   - Hamilton filter: regressao de y(t+h) sobre y(t), ..., y(t-p+1)
*   - BN decomposition: modelo AR(p) na primeira diferenca + companion form
*
* Hamilton filter (Hamilton, 2018):
*   - Regressao: y(t+h) = alpha + beta_1*y(t) + ... + beta_p*y(t-p+1) + e(t+h)
*   - h = 8 (padrao trimestral, 2 anos a frente)
*   - p = 4 (4 lags na regressao auxiliar)
*   - Ciclo = residuo da regressao
*   - Tendencia = valor ajustado (predicted)
*
* Beveridge-Nelson decomposition:
*   - Estimar AR(p) na primeira diferenca de y
*   - Tendencia BN = y(t) + soma das revisoes futuras esperadas
*   - Ciclo BN = y(t) - tendencia(t)
*   - Simplificacao: usando AR(4) com soma truncada
*
* LIMITACOES DO STATA:
*   1. Nao ha comando nativo para Hamilton filter (tsfilter nao inclui)
*   2. Nao ha comando nativo para BN decomposition
*   3. A implementacao manual pode diferir ligeiramente das bibliotecas
*      especializadas (neverhpfilter em R, chronobox em Python)
*   4. A BN decomposition usa soma truncada (100 periodos) em vez da
*      formula exata via inversa da companion matrix
*
* Datasets: examples/filters/data/us_gdp_quarterly.csv
*           examples/filters/data/brazil_gdp.csv
*
* Saida:    examples/filters/outputs/Stata/hamilton_bn_us.csv
*           examples/filters/outputs/Stata/hamilton_bn_br.csv
*           examples/filters/outputs/Stata/hamilton_bn_summary.csv
********************************************************************************

clear all
set more off
set seed 42

local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"
capture mkdir "`output_dir'"

display _newline
display "=== Validacao Hamilton Filter e BN Decomposition (Stata) ==="
display _newline

* ======================================================================
* Programa: Hamilton filter
*
* Implementacao manual via regress.
* Para cada horizonte h, regride y em t+h sobre y(t), y(t-1), ..., y(t-p+1).
* O ciclo eh o residuo, a tendencia eh o valor ajustado.
* ======================================================================

capture program drop hamilton_filter
program define hamilton_filter
    syntax varname, h(integer) p(integer) cycle(name) trend(name)

    * Variavel dependente: y deslocada h periodos para frente
    * Equivalente: regressar y sobre L(h).y, L(h+1).y, ..., L(h+p-1).y
    * Ou seja: y_t = alpha + sum(beta_j * L(h+j-1).y_t) + e_t
    * O ciclo eh e_t, a tendencia eh y_hat

    local lags ""
    forvalues j = 0/`=`p'-1' {
        local lag_val = `h' + `j'
        local lags "`lags' L`lag_val'.`varlist'"
    }

    * Regressao OLS
    quietly regress `varlist' `lags'

    * Ciclo = residuo
    quietly predict double `cycle', residuals
    * Tendencia = valor ajustado
    quietly predict double `trend', xb

    * Reportar
    summarize `cycle', detail
    display "  Hamilton h=`h' p=`p': cycle std = " %12.8f r(sd) ///
            " (obs validas: " r(N) "/" _N ")"
end

* ======================================================================
* Programa: BN decomposition (simplificada)
*
* Implementacao manual via AR(p) na primeira diferenca.
* Usa soma truncada em 100 periodos para aproximar a soma infinita.
*
* Metodologia:
*   1. dy = primeira diferenca de y
*   2. Estimar AR(p): dy_t = c + phi_1*dy_{t-1} + ... + phi_p*dy_{t-p} + e_t
*   3. Para cada t, calcular a soma das previsoes futuras de dy
*   4. Tendencia BN = y_t + soma das previsoes
*   5. Ciclo BN = y_t - tendencia_t = -soma das previsoes
*
* NOTA: Esta eh uma simplificacao. A implementacao exata usaria a forma
* companion e a inversa (I-A)^{-1}. Aqui iteramos diretamente.
* ======================================================================

capture program drop bn_decomposition
program define bn_decomposition
    syntax varname, ar_order(integer) cycle(name) trend(name)

    * Primeira diferenca
    tempvar dy
    generate double `dy' = D.`varlist'

    * Construir lags para AR(p)
    local lags ""
    forvalues j = 1/`ar_order' {
        local lags "`lags' L`j'.`dy'"
    }

    * Estimar AR(p) na primeira diferenca
    quietly regress `dy' `lags'

    * Extrair coeficientes
    local mu = _b[_cons]
    forvalues j = 1/`ar_order' {
        local phi`j' = _b[L`j'.`dy']
    }

    * Soma dos coeficientes AR (para drift de longo prazo)
    local sum_phi = 0
    forvalues j = 1/`ar_order' {
        local sum_phi = `sum_phi' + `phi`j''
    }

    display "  BN AR(`ar_order'): mu = " %10.6f `mu' ///
            ", sum(phi) = " %10.6f `sum_phi'

    * Drift de longo prazo: mu / (1 - sum(phi))
    local long_run_drift = `mu' / (1 - `sum_phi')

    * Calcular BN trend via soma truncada de previsoes (100 passos)
    local max_horizon = 100

    * Gerar variaveis de ciclo e tendencia (inicialmente missing)
    generate double `cycle' = .
    generate double `trend' = .

    * Para cada observacao (a partir de ar_order+2 para ter lags disponiveis)
    local start_obs = `ar_order' + 2
    local n_obs = _N

    quietly {
        forvalues t = `start_obs'/`n_obs' {
            * Coletar estado atual: [dy_t, dy_{t-1}, ..., dy_{t-p+1}]
            * Precisamos dos ultimos ar_order valores de dy
            local state_ok = 1
            forvalues j = 1/`ar_order' {
                local idx = `t' - `j'
                if `idx' < 1 {
                    local state_ok = 0
                }
            }

            if `state_ok' == 0 {
                continue
            }

            * Inicializar estado com valores observados de dy
            forvalues j = 1/`ar_order' {
                local s`j' = `dy'[`t' - `j' + 1]
                if missing(`s`j'') {
                    local state_ok = 0
                }
            }

            if `state_ok' == 0 {
                continue
            }

            * Iterar previsoes futuras e acumular soma
            local cum_sum = 0

            forvalues step = 1/`max_horizon' {
                * Previsao: dy_{t+step} = mu + sum(phi_j * s_j)
                local dy_pred = `mu'
                forvalues j = 1/`ar_order' {
                    local dy_pred = `dy_pred' + `phi`j'' * `s`j''
                }

                local cum_sum = `cum_sum' + `dy_pred'

                * Atualizar estado (shift)
                if `ar_order' > 1 {
                    forvalues j = `ar_order'(-1)2 {
                        local jm1 = `j' - 1
                        local s`j' = `s`jm1''
                    }
                }
                local s1 = `dy_pred'
            }

            * Tendencia BN = y_t + soma acumulada das previsoes
            replace `trend' = `varlist'[`t'] + `cum_sum' in `t'
            replace `cycle' = `varlist'[`t'] - `trend'[`t'] in `t'
        }
    }

    * Reportar
    summarize `cycle', detail
    display "  BN AR(`ar_order'): cycle std = " %12.8f r(sd) ///
            " (obs validas: " r(N) "/" _N ")"
end


* ======================================================================
* 1. Processar EUA
* ======================================================================
display "--- Processando: US ---"

import delimited "`data_dir'/us_gdp_quarterly.csv", clear

generate year = real(substr(date, 1, 4))
generate month = real(substr(date, 6, 2))
generate qtr = ceil(month / 3)
generate tq = yq(year, qtr)
format tq %tq
tsset tq
rename gdp_log y

* --- Hamilton filter com varios horizontes ---
foreach h_val in 4 8 12 16 {
    display _newline
    display "  Hamilton h=`h_val', p=4..."
    hamilton_filter y, h(`h_val') p(4) cycle(ham_cycle_h`h_val') trend(ham_trend_h`h_val')
}

* --- BN decomposition com varias ordens AR ---
foreach ar_p in 1 2 4 8 {
    display _newline
    display "  BN decomposition AR(`ar_p')..."
    bn_decomposition y, ar_order(`ar_p') cycle(bn_cycle_ar`ar_p') trend(bn_trend_ar`ar_p')
}

* --- Exportar resultados US ---
* Hamilton h=8 (padrao) e BN AR(4) (padrao) como colunas principais
generate country = "US"

export delimited date country ///
    ham_cycle_h4 ham_trend_h4 ///
    ham_cycle_h8 ham_trend_h8 ///
    ham_cycle_h12 ham_trend_h12 ///
    ham_cycle_h16 ham_trend_h16 ///
    bn_cycle_ar1 bn_trend_ar1 ///
    bn_cycle_ar2 bn_trend_ar2 ///
    bn_cycle_ar4 bn_trend_ar4 ///
    bn_cycle_ar8 bn_trend_ar8 ///
    using "`output_dir'/hamilton_bn_us.csv", replace

display _newline
display "  Resultados US salvos em: `output_dir'/hamilton_bn_us.csv"

* ======================================================================
* 2. Processar Brasil
* ======================================================================
display _newline
display "--- Processando: BR ---"

import delimited "`data_dir'/brazil_gdp.csv", clear

generate year = real(substr(date, 1, 4))
generate month = real(substr(date, 6, 2))
generate qtr = ceil(month / 3)
generate tq = yq(year, qtr)
format tq %tq
tsset tq
rename gdp_log y

foreach h_val in 4 8 12 16 {
    display _newline
    display "  Hamilton h=`h_val', p=4..."
    hamilton_filter y, h(`h_val') p(4) cycle(ham_cycle_h`h_val') trend(ham_trend_h`h_val')
}

foreach ar_p in 1 2 4 8 {
    display _newline
    display "  BN decomposition AR(`ar_p')..."
    bn_decomposition y, ar_order(`ar_p') cycle(bn_cycle_ar`ar_p') trend(bn_trend_ar`ar_p')
}

generate country = "BR"

export delimited date country ///
    ham_cycle_h4 ham_trend_h4 ///
    ham_cycle_h8 ham_trend_h8 ///
    ham_cycle_h12 ham_trend_h12 ///
    ham_cycle_h16 ham_trend_h16 ///
    bn_cycle_ar1 bn_trend_ar1 ///
    bn_cycle_ar2 bn_trend_ar2 ///
    bn_cycle_ar4 bn_trend_ar4 ///
    bn_cycle_ar8 bn_trend_ar8 ///
    using "`output_dir'/hamilton_bn_br.csv", replace

display _newline
display "  Resultados BR salvos em: `output_dir'/hamilton_bn_br.csv"

* ======================================================================
* 3. Resumo
* ======================================================================
display _newline
display "=== Limitacoes do Stata para Hamilton e BN ==="
display ""
display "1. Hamilton filter NAO e nativo do Stata."
display "   - Implementado manualmente via regress com lags."
display "   - A formulacao e identica a Hamilton (2018):"
display "     y(t) = alpha + beta_1*L(h).y + ... + beta_p*L(h+p-1).y + e(t)"
display "   - Ciclo = residuo, tendencia = valor ajustado."
display "   - Para referencia em R: neverhpfilter::yth_filter()"
display ""
display "2. BN decomposition NAO e nativa do Stata."
display "   - Implementada via AR(p) na primeira diferenca."
display "   - Usa soma truncada (100 passos) em vez da formula exata"
display "     via companion matrix: A*(I-A)^{-1}."
display "   - Diferenças vs R/Python esperadas na 3a casa decimal (~1e-3)."
display ""
display "3. Pacotes Stata de terceiros que poderiam ser usados:"
display "   - 'hamilton13' (ssc install hamilton13) — apenas para h=8,p=4"
display "   - Nenhum pacote padrao para BN decomposition"
display ""

display "=== 02_hamilton_bn_validation.do concluido com sucesso ==="
