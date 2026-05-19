* =============================================================================
* 03_x13_validation.do
* Validacao cruzada: X-13 ARIMA-SEATS
*
* SOBRE O X-13 NO STATA:
* O Stata nao possui implementacao nativa do X-13 ARIMA-SEATS.
* O pacote externo "x13as" (por Kit Baum) permite chamar o binario
* X-13ARIMA-SEATS do US Census Bureau a partir do Stata.
*
* Instalacao do x13as:
*   ssc install x13as
*   * Requer tambem o binario x13ashtml no PATH do sistema
*   * Download: https://www.census.gov/data/software/x13as.html
*
* Alternativas se x13as nao estiver disponivel:
* 1. tsfilter hp para extrair tendencia
* 2. Dummies sazonais para dessazonalizar
* 3. Decomposicao classica manual (ver 01_classical_validation.do)
*
* LIMITACOES:
* - x13as depende de binario externo (x13ashtml)
* - Nem todas as opcoes do X-13 estao expostas no wrapper Stata
* - Para uso completo do X-13, considerar R (seasonal::seas()) ou
*   Python (statsmodels.tsa.x13)
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de caminhos ---
local base_dir "."
if "`c(filename)'" != "" {
    local script_dir = subinstr("`c(filename)'", "/03_x13_validation.do", "", 1)
    local base_dir = subinstr("`script_dir'", "/Stata", "", 1)
}

local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

capture mkdir "`output_dir'"

* =============================================================================
* Carregar dados brazil_ipca.csv
* =============================================================================
display _n "--- Carregando dados brazil_ipca.csv ---"

import delimited "`data_dir'/brazil_ipca.csv", clear

generate date_stata = date(date, "YMD")
format date_stata %td

generate ym = mofd(date_stata)
format ym %tm

tsset ym

display "Serie IPCA: N = " _N
summarize ipca, detail

* =============================================================================
* Tentar X-13 via x13as
* =============================================================================
display _n "--- Tentando X-13 ARIMA-SEATS via x13as ---"

capture which x13as
local x13_available = (_rc == 0)

if `x13_available' {
    display "Pacote x13as encontrado!"

    * Executar X-13 na serie IPCA
    capture noisily x13as ipca, transform(log) ///
        arima((0 1 1)(0 1 1)) ///
        save(d11 d12 d13)

    if _rc == 0 {
        display "X-13 executado com sucesso!"

        * d11 = serie dessazonalizada
        * d12 = componente sazonal
        * d13 = tendencia-ciclo
        rename ipca_d11 sa_x13
        rename ipca_d12 seasonal_x13
        rename ipca_d13 trend_x13

        summarize sa_x13
        display "Serie dessazonalizada X-13 - media: " r(mean)
        summarize seasonal_x13
        display "Componente sazonal X-13 - range: " r(min) " a " r(max)
        summarize trend_x13
        display "Tendencia X-13 - range: " r(min) " a " r(max)
    }
    else {
        display "Erro ao executar x13as. Verifique se x13ashtml esta no PATH."
        local x13_available = 0
    }
}
else {
    display "Pacote x13as NAO encontrado."
    display "Para instalar: ssc install x13as"
    display "Tambem requer o binario x13ashtml do US Census Bureau."
    display ""
    display "Usando alternativas (tsfilter + dummies) como proxy."
}

* =============================================================================
* Alternativa: Dessazonalizacao via HP + dummies sazonais
* =============================================================================
display _n "--- Alternativa: HP + dummies sazonais ---"

* Filtro HP para tendencia
tsfilter hp ipca_cycle_hp = ipca, smooth(14400) trend(ipca_trend_hp)

display "Trend HP:"
summarize ipca_trend_hp
display "Range: " r(min) " a " r(max)

* Componente sazonal via dummies
generate month = month(date_stata)
generate detrended = ipca - ipca_trend_hp

* Dummies mensais
tabulate month, generate(m_)

* Regressao para estimar sazonalidade
regress detrended m_1-m_12, noconstant
predict seasonal_hp, xb

* Serie dessazonalizada
generate sa_hp = ipca - seasonal_hp

* Residual
generate residual_hp = ipca - ipca_trend_hp - seasonal_hp

display "Serie dessazonalizada (HP proxy):"
summarize sa_hp
display "Componente sazonal (HP proxy):"
summarize seasonal_hp
display "Range sazonal: " r(min) " a " r(max)
display "Residual:"
summarize residual_hp
display "Desvio padrao: " r(sd)

* =============================================================================
* Carregar e processar airline.csv para comparacao adicional
* =============================================================================
display _n "--- Processando airline.csv ---"

preserve
import delimited "`data_dir'/airline.csv", clear

generate date_stata = date(date, "YMD")
format date_stata %td
generate ym = mofd(date_stata)
format ym %tm
tsset ym

* HP filter no airline
tsfilter hp air_cycle = passengers, smooth(14400) trend(air_trend)

* Sazonalidade via dummies
generate month = month(date_stata)
generate air_detrended = passengers - air_trend
tabulate month, generate(am_)
regress air_detrended am_1-am_12, noconstant
predict air_seasonal, xb
generate air_sa = passengers - air_seasonal

display "Airline - tendencia HP:"
summarize air_trend
display "Airline - componente sazonal:"
summarize air_seasonal

* Exportar airline
keep date passengers air_trend air_seasonal air_sa
rename passengers observed
rename air_trend trend_hp
rename air_seasonal seasonal_proxy
rename air_sa sa_proxy

export delimited using "`output_dir'/x13_airline.csv", replace
display "Airline salvo em: `output_dir'/x13_airline.csv"
restore

* =============================================================================
* Exportar resultados IPCA
* =============================================================================
display _n "--- Exportando resultados IPCA ---"

* Preparar dataset para exportacao
if `x13_available' {
    keep date ipca ipca_trend_hp seasonal_hp sa_hp residual_hp ///
         sa_x13 seasonal_x13 trend_x13
    rename ipca observed
}
else {
    keep date ipca ipca_trend_hp seasonal_hp sa_hp residual_hp
    rename ipca observed
}

rename ipca_trend_hp trend_hp
rename seasonal_hp seasonal_proxy
rename sa_hp sa_proxy
rename residual_hp residual_proxy

export delimited using "`output_dir'/x13_adjusted.csv", replace
display "Resultados salvos em: `output_dir'/x13_adjusted.csv"

* =============================================================================
* Comparacao com resultados Python/R
* =============================================================================
display _n "--- Comparacao com Python/R ---"

display "LIMITACOES DO STATA PARA X-13:"
display ""
display "1. x13as requer binario externo (x13ashtml) - nem sempre disponivel"
display "2. Wrapper Stata expoe subconjunto das opcoes do X-13"
display "3. Alternativa HP+dummies NAO e equivalente ao X-13:"
display "   - X-13 usa ARIMA para extrapolar nas bordas"
display "   - X-13 usa filtros Henderson para tendencia"
display "   - X-13 detecta outliers e efeitos de calendario"
display "   - HP+dummies assume sazonalidade fixa"
display ""
display "4. Para validacao cruzada rigorosa com X-13, usar R (seasonal::seas())"
display "   que tem wrapper completo e bem mantido."

capture confirm file "`base_dir'/outputs/x13_adjusted.csv"
if _rc == 0 {
    preserve
    import delimited "`base_dir'/outputs/x13_adjusted.csv", clear

    capture confirm variable sa_stl_s13_robust
    if _rc == 0 {
        rename sa_stl_s13_robust py_sa_stl
        keep date py_sa_stl
        tempfile python_x13
        save `python_x13'
        restore

        merge 1:1 date using `python_x13', nogenerate

        * Correlacao entre HP proxy e STL proxy do Python
        correlate sa_proxy py_sa_stl
        display "Correlacao SA (Stata HP proxy vs Python STL): " r(rho)

        display _n "TOLERANCIA: Correlacao > 0.95 esperada entre metodos"
        display "proxy diferentes (HP vs STL). Valores exatos diferem"
        display "pois sao metodos fundamentalmente distintos."
    }
    else {
        restore
        display "Colunas esperadas nao encontradas no arquivo Python."
    }
}
else {
    display "Arquivo Python nao encontrado."
    display "Execute os notebooks Python primeiro."
}

display _n "=== 03_x13_validation.do concluido ==="
