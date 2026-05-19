* =============================================================================
* 02_stl_validation.do
* Validacao cruzada: STL Decomposition - alternativa via tsfilter
*
* LIMITACAO IMPORTANTE:
* Stata NAO possui implementacao nativa de STL (Seasonal and Trend
* decomposition using Loess). O STL foi proposto por Cleveland et al. (1990)
* e usa regressao local (LOESS) iterativa para decompor a serie.
*
* Alternativas disponiveis no Stata:
* 1. tsfilter hp  - Filtro Hodrick-Prescott (extrai tendencia)
* 2. tsfilter bk  - Filtro Baxter-King (band-pass)
* 3. tsfilter cf  - Filtro Christiano-Fitzgerald (band-pass)
* 4. tssmooth ma  - Medias moveis (suavizacao simples)
*
* Este script usa tsfilter como alternativa para extrair componentes
* de tendencia e ciclo, documentando as diferencas em relacao ao STL.
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de caminhos ---
local base_dir "."
if "`c(filename)'" != "" {
    local script_dir = subinstr("`c(filename)'", "/02_stl_validation.do", "", 1)
    local base_dir = subinstr("`script_dir'", "/Stata", "", 1)
}

local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

capture mkdir "`output_dir'"

* =============================================================================
* Carregar dados co2.csv
* =============================================================================
display _n "--- Carregando dados co2.csv ---"

import delimited "`data_dir'/co2.csv", clear

* Converter data para formato Stata
generate date_stata = date(date, "YMD")
format date_stata %td

generate ym = mofd(date_stata)
format ym %tm

tsset ym

display "Serie CO2: N = " _N
summarize co2_ppm, detail

* =============================================================================
* LIMITACOES DO STATA PARA DECOMPOSICAO STL
* =============================================================================
* O STL (Seasonal-Trend Decomposition using LOESS) requer:
*   - Regressao local (LOESS) iterativa
*   - Separacao explicita de seasonal + trend + remainder
*   - Parametros s.window e t.window para controle
*
* Stata nao tem LOESS iterativo para decomposicao. As alternativas sao:
*
* 1. tsfilter hp: Filtro HP separa trend vs ciclo, mas NAO separa
*    componente sazonal. Util para series dessazonalizadas.
*
* 2. tsfilter bk/cf: Filtros band-pass extraem componente ciclico
*    em faixa de frequencia especifica, mas tambem nao isolam sazonalidade.
*
* 3. Abordagem manual: media movel para trend + dummies sazonais para
*    componente sazonal (equivale a decomposicao classica, nao STL).
*
* CONCLUSAO: Resultados do Stata NAO serao diretamente comparaveis ao STL
* de R/Python. Usamos tsfilter como referencia para a componente de
* tendencia apenas.
* =============================================================================

* =============================================================================
* Filtro Hodrick-Prescott (HP)
* =============================================================================
display _n "--- Filtro Hodrick-Prescott ---"

* Lambda = 14400 para dados mensais (padrao)
tsfilter hp co2_cycle_hp = co2_ppm, smooth(14400) trend(co2_trend_hp)

display "HP Filter (lambda=14400):"
summarize co2_trend_hp
display "Range trend HP: " r(min) " a " r(max)
summarize co2_cycle_hp
display "Desvio padrao ciclo HP: " r(sd)

* Lambda = 129600 (mais suave, comum para mensal)
tsfilter hp co2_cycle_hp2 = co2_ppm, smooth(129600) trend(co2_trend_hp2)

display _n "HP Filter (lambda=129600):"
summarize co2_trend_hp2
display "Range trend HP2: " r(min) " a " r(max)
summarize co2_cycle_hp2
display "Desvio padrao ciclo HP2: " r(sd)

* Verificar reconstrucao: observado = trend + ciclo
generate recon_hp = co2_trend_hp + co2_cycle_hp
generate diff_hp = abs(co2_ppm - recon_hp)
summarize diff_hp
display "Erro reconstrucao HP: " r(max)

* =============================================================================
* Filtro Baxter-King (BK)
* =============================================================================
display _n "--- Filtro Baxter-King ---"

* Parametros: minperiod=18, maxperiod=96 (ciclos de negocios para mensal)
* O filtro BK perde observacoes nas bordas (order = k observacoes de cada lado)
tsfilter bk co2_cycle_bk = co2_ppm, minperiod(18) maxperiod(96)

display "BK Filter (18-96 meses):"
summarize co2_cycle_bk
display "Desvio padrao ciclo BK: " r(sd)
display "N validos BK: " r(N)

* Nota: tsfilter bk nao retorna o trend diretamente
* Trend = observado - ciclo (aproximacao)
generate co2_trend_bk = co2_ppm - co2_cycle_bk
summarize co2_trend_bk
display "Range trend BK (aprox): " r(min) " a " r(max)

* =============================================================================
* Filtro Christiano-Fitzgerald (CF)
* =============================================================================
display _n "--- Filtro Christiano-Fitzgerald ---"

* O CF e mais flexivel que o BK e nao perde observacoes nas bordas
tsfilter cf co2_cycle_cf = co2_ppm, minperiod(18) maxperiod(96)

display "CF Filter (18-96 meses):"
summarize co2_cycle_cf
display "Desvio padrao ciclo CF: " r(sd)
display "N validos CF: " r(N)

generate co2_trend_cf = co2_ppm - co2_cycle_cf
summarize co2_trend_cf
display "Range trend CF (aprox): " r(min) " a " r(max)

* =============================================================================
* Extrair componente sazonal via dummies mensais (proxy)
* =============================================================================
display _n "--- Componente sazonal via dummies mensais ---"

* Usar trend HP como base e extrair sazonalidade dos residuos
generate detrended = co2_ppm - co2_trend_hp

* Criar dummies mensais
generate month = month(date_stata)
tabulate month, generate(m_)

* Regressao dos residuos nas dummies mensais (sem constante)
regress detrended m_1-m_12, noconstant

* Prever componente sazonal
predict seasonal_proxy, xb

* Residual
generate residual_proxy = co2_ppm - co2_trend_hp - seasonal_proxy

display "Componente sazonal (proxy via dummies):"
summarize seasonal_proxy
display "Range sazonal: " r(min) " a " r(max)
summarize residual_proxy
display "Desvio padrao residual: " r(sd)

* =============================================================================
* Exportar resultados
* =============================================================================
display _n "--- Exportando resultados ---"

keep date co2_ppm co2_trend_hp co2_cycle_hp co2_trend_hp2 co2_cycle_hp2 ///
     co2_cycle_bk co2_trend_bk co2_cycle_cf co2_trend_cf ///
     seasonal_proxy residual_proxy

rename co2_ppm observed

export delimited using "`output_dir'/stl_components.csv", replace
display "Resultados salvos em: `output_dir'/stl_components.csv"

* =============================================================================
* Comparacao com resultados Python/R (STL)
* =============================================================================
display _n "--- Comparacao com Python (STL) ---"

display _n "NOTA IMPORTANTE:"
display "Os resultados do Stata (tsfilter HP/BK/CF) NAO sao diretamente"
display "comparaveis ao STL de R/Python. As diferencas fundamentais sao:"
display ""
display "1. STL usa LOESS iterativo; tsfilter hp usa penalizacao L2"
display "2. STL separa seasonal+trend+remainder; HP separa trend+ciclo"
display "3. BK/CF sao filtros band-pass, nao decomposicoes completas"
display "4. A componente sazonal via dummies e fixa (nao varia no tempo)"
display "   enquanto STL permite sazonalidade variante no tempo"
display ""
display "Para comparacao valida de tendencia, use apenas a correlacao"
display "entre trend_hp e trend do STL (ambos capturam tendencia de longo prazo)."

capture confirm file "`base_dir'/outputs/stl_components.csv"
if _rc == 0 {
    preserve
    import delimited "`base_dir'/outputs/stl_components.csv", clear

    * Verificar se coluna trend_s7 existe
    capture confirm variable trend_s7
    if _rc == 0 {
        rename trend_s7 py_trend_s7
        rename trend_s15 py_trend_s15
        keep date py_trend_s7 py_trend_s15
        tempfile python_stl
        save `python_stl'
        restore

        merge 1:1 date using `python_stl', nogenerate

        * Correlacao entre trend HP e trend STL
        correlate co2_trend_hp py_trend_s7
        display "Correlacao trend HP vs STL s=7: " r(rho)

        correlate co2_trend_hp py_trend_s15
        display "Correlacao trend HP vs STL s=15: " r(rho)

        display _n "TOLERANCIA: Correlacao trend HP vs STL > 0.99 esperada"
        display "(ambos capturam tendencia de longo prazo, diferem no detalhe)"
    }
    else {
        restore
        display "Colunas STL nao encontradas no arquivo Python."
    }
}
else {
    display "Arquivo Python STL nao encontrado."
    display "Execute os notebooks Python primeiro."
}

display _n "=== 02_stl_validation.do concluido ==="
