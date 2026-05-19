********************************************************************************
* 01_hp_bk_cf_validation.do
*
* Validacao cruzada dos filtros HP, Baxter-King e Christiano-Fitzgerald
* usando comandos nativos do Stata: tsfilter hp, tsfilter bk, tsfilter cf.
*
* Parametros:
*   HP filter:  smooth(1600) — padrao para dados trimestrais
*   BK filter:  minperiod(6) maxperiod(32) — bandas de ciclo de negocios
*   CF filter:  minperiod(6) maxperiod(32) — mesmas bandas de frequencia
*
* Tolerancias esperadas vs chronobox (Python):
*   HP:  < 1e-6 (solucao exata, mesmo sistema linear)
*   BK:  < 1e-4 (diferenças de truncamento nas bordas)
*   CF:  < 1e-4 (diferenças na estimacao espectral)
*
* Datasets: examples/filters/data/us_gdp_quarterly.csv
*           examples/filters/data/brazil_gdp.csv
*
* Saida:    examples/filters/outputs/Stata/hp_bk_cf_us.csv
*           examples/filters/outputs/Stata/hp_bk_cf_br.csv
*           examples/filters/outputs/Stata/hp_bk_cf_summary.csv
********************************************************************************

clear all
set more off
set seed 42

* --- Caminhos relativos (assumindo execucao a partir de examples/filters/Stata/) ---
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

* Criar diretorio de saida (silencioso se ja existe)
capture mkdir "`output_dir'"

display _newline
display "=== Validacao HP / BK / CF com tsfilter (Stata) ==="
display _newline

********************************************************************************
* MACRO: processar um pais
* Argumentos: csv_file, country_label, output_suffix
********************************************************************************

* ======================================================================
* 1. Processar EUA
* ======================================================================
display "--- Processando: US (us_gdp_quarterly.csv) ---"

import delimited "`data_dir'/us_gdp_quarterly.csv", clear

* Criar variavel de tempo trimestral a partir da coluna date
* Formato esperado: YYYY-MM-DD
generate year = real(substr(date, 1, 4))
generate month = real(substr(date, 6, 2))
generate qtr = ceil(month / 3)
generate tq = yq(year, qtr)
format tq %tq

* Declarar serie temporal
tsset tq

* Renomear para facilitar
rename gdp_log y

* -------------------------------------------------------
* Hodrick-Prescott filter (lambda = 1600)
* tsfilter hp cria variaveis de ciclo e tendencia
* -------------------------------------------------------
tsfilter hp hp_cycle = y, smooth(1600) trend(hp_trend)

summarize hp_cycle, detail
local hp_sd = r(sd)
local hp_min = r(min)
local hp_max = r(max)
display "  HP: cycle std = " %12.8f `hp_sd' ///
        ", range = [" %12.8f `hp_min' ", " %12.8f `hp_max' "]"

* -------------------------------------------------------
* Baxter-King filter (pl=6, pu=32)
* Nota: BK e um filtro band-pass simetrico; observacoes nas
* bordas sao perdidas (set to missing).
* -------------------------------------------------------
tsfilter bk bk_cycle = y, minperiod(6) maxperiod(32)

summarize bk_cycle, detail
local bk_sd = r(sd)
local bk_n = r(N)
local bk_total = _N
display "  BK: cycle std = " %12.8f `bk_sd' ///
        " (obs validas: `bk_n'/`bk_total')"

* -------------------------------------------------------
* Christiano-Fitzgerald filter (pl=6, pu=32)
* CF usa toda a amostra (sem perda de bordas).
* -------------------------------------------------------
tsfilter cf cf_cycle = y, minperiod(6) maxperiod(32)

summarize cf_cycle, detail
local cf_sd = r(sd)
display "  CF: cycle std = " %12.8f `cf_sd'

* --- Correlacoes entre filtros ---
* HP vs BK (apenas observacoes validas de BK)
correlate hp_cycle bk_cycle
local cor_hp_bk = r(rho)

* HP vs CF
correlate hp_cycle cf_cycle
local cor_hp_cf = r(rho)

* BK vs CF (apenas observacoes validas de BK)
correlate bk_cycle cf_cycle
local cor_bk_cf = r(rho)

display "  Correlacoes: HP-BK = " %8.6f `cor_hp_bk' ///
        ", HP-CF = " %8.6f `cor_hp_cf' ///
        ", BK-CF = " %8.6f `cor_bk_cf'

* --- Preparar exportacao ---
generate country = "US"

* Exportar resultados US
export delimited date country hp_cycle hp_trend cf_cycle bk_cycle ///
    using "`output_dir'/hp_bk_cf_us.csv", replace
display "  Resultados US salvos em: `output_dir'/hp_bk_cf_us.csv"

* Salvar estatisticas US em locals
local us_n = _N
local us_hp_sd = `hp_sd'
local us_bk_sd = `bk_sd'
local us_cf_sd = `cf_sd'
local us_hp_min = `hp_min'
local us_hp_max = `hp_max'

* ======================================================================
* 2. Processar Brasil
* ======================================================================
display _newline
display "--- Processando: BR (brazil_gdp.csv) ---"

import delimited "`data_dir'/brazil_gdp.csv", clear

generate year = real(substr(date, 1, 4))
generate month = real(substr(date, 6, 2))
generate qtr = ceil(month / 3)
generate tq = yq(year, qtr)
format tq %tq

tsset tq
rename gdp_log y

* HP filter
tsfilter hp hp_cycle = y, smooth(1600) trend(hp_trend)

summarize hp_cycle, detail
local hp_sd = r(sd)
local hp_min = r(min)
local hp_max = r(max)
display "  HP: cycle std = " %12.8f `hp_sd' ///
        ", range = [" %12.8f `hp_min' ", " %12.8f `hp_max' "]"

* BK filter
tsfilter bk bk_cycle = y, minperiod(6) maxperiod(32)

summarize bk_cycle, detail
local bk_sd = r(sd)
local bk_n = r(N)
local bk_total = _N
display "  BK: cycle std = " %12.8f `bk_sd' ///
        " (obs validas: `bk_n'/`bk_total')"

* CF filter
tsfilter cf cf_cycle = y, minperiod(6) maxperiod(32)

summarize cf_cycle, detail
local cf_sd = r(sd)
display "  CF: cycle std = " %12.8f `cf_sd'

* Correlacoes
correlate hp_cycle bk_cycle
local cor_hp_bk = r(rho)
correlate hp_cycle cf_cycle
local cor_hp_cf = r(rho)
correlate bk_cycle cf_cycle
local cor_bk_cf = r(rho)

display "  Correlacoes: HP-BK = " %8.6f `cor_hp_bk' ///
        ", HP-CF = " %8.6f `cor_hp_cf' ///
        ", BK-CF = " %8.6f `cor_bk_cf'

generate country = "BR"

export delimited date country hp_cycle hp_trend cf_cycle bk_cycle ///
    using "`output_dir'/hp_bk_cf_br.csv", replace
display "  Resultados BR salvos em: `output_dir'/hp_bk_cf_br.csv"

local br_n = _N
local br_hp_sd = `hp_sd'
local br_bk_sd = `bk_sd'
local br_cf_sd = `cf_sd'
local br_hp_min = `hp_min'
local br_hp_max = `hp_max'

* ======================================================================
* 3. Salvar resumo comparativo
* ======================================================================
display _newline
display "--- Resumo ---"

clear
set obs 2

generate country = ""
replace country = "US" in 1
replace country = "BR" in 2

generate n_obs = .
replace n_obs = `us_n' in 1
replace n_obs = `br_n' in 2

generate hp_std = .
replace hp_std = `us_hp_sd' in 1
replace hp_std = `br_hp_sd' in 2

generate bk_std = .
replace bk_std = `us_bk_sd' in 1
replace bk_std = `br_bk_sd' in 2

generate cf_std = .
replace cf_std = `us_cf_sd' in 1
replace cf_std = `br_cf_sd' in 2

generate hp_min = .
replace hp_min = `us_hp_min' in 1
replace hp_min = `br_hp_min' in 2

generate hp_max = .
replace hp_max = `us_hp_max' in 1
replace hp_max = `br_hp_max' in 2

export delimited using "`output_dir'/hp_bk_cf_summary.csv", replace
display "Resumo salvo em: `output_dir'/hp_bk_cf_summary.csv"

* ======================================================================
* Notas sobre validacao cruzada
* ======================================================================
display _newline
display "=== Notas ==="
display "HP filter com smooth(1600) deve produzir resultados identicos"
display "ao R (mFilter::hpfilter) e Python (chronobox) ate ~1e-6."
display "A solucao e exata: mesmo sistema linear tridiagonal."
display ""
display "BK e CF podem diferir nas bordas (~1e-4) dependendo da"
display "implementacao do truncamento e estimacao espectral."

display _newline
display "=== 01_hp_bk_cf_validation.do concluido com sucesso ==="
