* =============================================================================
* 01_classical_validation.do
* Validacao cruzada: Decomposicao Classica (aditiva e multiplicativa)
* Implementacao manual com moving averages via tssmooth ma
*
* Stata NAO possui um comando decompose() nativo como R.
* Este script implementa a decomposicao classica manualmente:
*   1. Trend: media movel centrada (2x12 para dados mensais)
*   2. Sazonal: media dos desvios/razoes por mes
*   3. Residual: observado - trend - sazonal (aditivo)
*              ou observado / (trend * sazonal) (multiplicativo)
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de caminhos ---
* Ajuste o caminho base conforme necessario
local base_dir "."
if "`c(filename)'" != "" {
    local script_dir = subinstr("`c(filename)'", "/01_classical_validation.do", "", 1)
    local base_dir = subinstr("`script_dir'", "/Stata", "", 1)
}

local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

capture mkdir "`output_dir'"

* =============================================================================
* Carregar dados airline.csv
* =============================================================================
display _n "--- Carregando dados airline.csv ---"

import delimited "`data_dir'/airline.csv", clear

* Converter data para formato Stata
generate date_stata = date(date, "YMD")
format date_stata %td

* Criar variavel de tempo mensal para tsset
generate ym = mofd(date_stata)
format ym %tm

tsset ym

display "Serie airline: N = " _N
summarize passengers, detail

* =============================================================================
* DECOMPOSICAO ADITIVA - Implementacao manual
* =============================================================================
display _n "--- Decomposicao Classica Aditiva ---"

* Passo 1: Trend via media movel centrada 2x12
* Para dados mensais com periodo 12, usamos MA(12) centrada
* Isto equivale a uma MA(12) seguida de MA(2) para centralizar
* tssmooth ma calcula a media movel

* Media movel simples de 12 periodos
tssmooth ma trend_ma12 = passengers, window(6 1 5)

* Media movel centrada 2x12 (media de duas MA(12) adjacentes)
* Em Stata, uma MA centrada de ordem par requer um passo extra.
* Usamos window(5 1 6) e fazemos a media com a anterior.
tssmooth ma trend_ma12b = passengers, window(5 1 6)
generate trend_add = (trend_ma12 + trend_ma12b) / 2

* Passo 2: Detrend (remover tendencia)
generate detrended_add = passengers - trend_add

* Passo 3: Componente sazonal - media por mes
generate month = month(date_stata)

* Calcular media sazonal por mes (excluindo observacoes sem trend)
bysort month: egen seasonal_raw_add = mean(detrended_add) if !missing(trend_add)

* Propagar o valor sazonal para todas as observacoes do mesmo mes
bysort month: egen seasonal_temp = mean(seasonal_raw_add)

* Centralizar para soma zero (garantir que componente sazonal some zero)
summarize seasonal_temp, meanonly
generate seasonal_add = seasonal_temp - r(mean)

* Passo 4: Residual
generate residual_add = passengers - trend_add - seasonal_add

* Verificar reconstrucao aditiva
generate recon_add = trend_add + seasonal_add + residual_add
generate diff_recon_add = abs(passengers - recon_add) if !missing(trend_add)
summarize diff_recon_add
display "Erro maximo reconstrucao (aditiva): " r(max)

* Estatisticas do componente sazonal
summarize seasonal_add
display "Range sazonal aditivo: " r(min) " a " r(max)

* Estatisticas do residual
summarize residual_add if !missing(residual_add)
display "Desvio padrao residual (aditivo): " r(sd)

* =============================================================================
* DECOMPOSICAO MULTIPLICATIVA - Implementacao manual
* =============================================================================
display _n "--- Decomposicao Classica Multiplicativa ---"

* Passo 1: Trend - mesmo que aditivo (media movel centrada)
* trend_add ja calculado acima - reutilizar
generate trend_mult = trend_add

* Passo 2: Detrend multiplicativo (razao)
generate detrended_mult = passengers / trend_mult if !missing(trend_mult)

* Passo 3: Componente sazonal multiplicativo - media por mes
bysort month: egen seasonal_raw_mult = mean(detrended_mult) if !missing(trend_mult)
bysort month: egen seasonal_mult_temp = mean(seasonal_raw_mult)

* Normalizar para media = 1 (em vez de soma zero como no aditivo)
summarize seasonal_mult_temp, meanonly
generate seasonal_mult = seasonal_mult_temp / r(mean)

* Passo 4: Residual multiplicativo
generate residual_mult = passengers / (trend_mult * seasonal_mult) if !missing(trend_mult)

* Verificar reconstrucao multiplicativa
generate recon_mult = trend_mult * seasonal_mult * residual_mult
generate diff_recon_mult = abs(passengers - recon_mult) if !missing(trend_mult)
summarize diff_recon_mult
display "Erro maximo reconstrucao (multiplicativa): " r(max)

* Estatisticas do componente sazonal multiplicativo
summarize seasonal_mult
display "Range sazonal multiplicativo: " r(min) " a " r(max)

* Estatisticas do residual multiplicativo
summarize residual_mult if !missing(residual_mult)
display "Desvio padrao residual (multiplicativo): " r(sd)

* =============================================================================
* Exportar resultados
* =============================================================================
display _n "--- Exportando resultados ---"

* Manter apenas as variaveis relevantes
keep date passengers trend_add seasonal_add residual_add ///
     trend_mult seasonal_mult residual_mult

* Renomear para consistencia com R e Python
rename passengers observed
rename trend_add trend_additive
rename seasonal_add seasonal_additive
rename residual_add residual_additive
rename trend_mult trend_multiplicative
rename seasonal_mult seasonal_multiplicative
rename residual_mult residual_multiplicative

export delimited using "`output_dir'/classical_components.csv", replace
display "Resultados salvos em: `output_dir'/classical_components.csv"

* =============================================================================
* Comparacao com resultados Python/R
* =============================================================================
display _n "--- Comparacao com Python ---"

capture confirm file "`base_dir'/outputs/classical_components.csv"
if _rc == 0 {
    preserve
    import delimited "`base_dir'/outputs/classical_components.csv", clear
    rename seasonal_additive py_seasonal_add
    rename seasonal_multiplicative py_seasonal_mult
    rename trend_additive py_trend_add
    rename trend_multiplicative py_trend_mult
    tempfile python_results
    save `python_results'
    restore

    merge 1:1 date using `python_results', nogenerate

    * Comparar componente sazonal aditivo
    generate diff_seasonal_add = abs(seasonal_additive - py_seasonal_add)
    summarize diff_seasonal_add
    display "Sazonal aditivo - diff max: " r(max)
    display "Sazonal aditivo - diff media: " r(mean)

    * Comparar componente sazonal multiplicativo
    generate diff_seasonal_mult = abs(seasonal_multiplicative - py_seasonal_mult)
    summarize diff_seasonal_mult
    display "Sazonal multiplicativo - diff max: " r(max)
    display "Sazonal multiplicativo - diff media: " r(mean)

    * TOLERANCIA DOCUMENTADA:
    * A decomposicao classica manual em Stata pode ter diferencas maiores
    * em relacao a R/Python devido a:
    * 1. tssmooth ma usa implementacao propria de media movel
    * 2. Tratamento de bordas pode diferir
    * 3. Centralizacao do componente sazonal pode ter pequenas diferencas
    * Tolerancia esperada: < 0.5 para componentes sazonais
    * Trend deve ser muito proximo (< 0.01) onde ambos estao definidos
    display _n "TOLERANCIA: Componentes classicos Stata vs Python podem diferir"
    display "devido a implementacao manual. Correlacao > 0.99 esperada."
}
else {
    display "Arquivo Python nao encontrado."
    display "Execute os notebooks Python primeiro."
}

display _n "=== 01_classical_validation.do concluido ==="
