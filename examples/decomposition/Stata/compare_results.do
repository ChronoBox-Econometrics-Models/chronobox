* =============================================================================
* compare_results.do
* Comparacao sistematica de resultados Python (chronobox) vs R vs Stata
* Gera relatorio de validacao cruzada com metricas e tolerancias
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de caminhos ---
local base_dir "."
if "`c(filename)'" != "" {
    local script_dir = subinstr("`c(filename)'", "/compare_results.do", "", 1)
    local base_dir = subinstr("`script_dir'", "/Stata", "", 1)
}

local output_python "`base_dir'/outputs"
local output_r "`base_dir'/outputs/R"
local output_stata "`base_dir'/outputs/Stata"

display "============================================================================="
display "  RELATORIO DE VALIDACAO CRUZADA: Python vs R vs Stata"
display "  Decomposicao de Series Temporais"
display "============================================================================="
display ""

* =============================================================================
* Programa auxiliar: calcula metricas de comparacao entre dois vetores
* =============================================================================
capture program drop compare_vectors
program define compare_vectors
    args var1 var2 label

    * Calcular diferencas
    tempvar diff absdiff
    quietly generate `diff' = `var1' - `var2'
    quietly generate `absdiff' = abs(`diff')

    * MAE
    quietly summarize `absdiff'
    local mae = r(mean)
    local maxdiff = r(max)
    local n_valid = r(N)

    * RMSE
    tempvar sq
    quietly generate `sq' = `diff'^2
    quietly summarize `sq'
    local rmse = sqrt(r(mean))

    * Correlacao
    quietly correlate `var1' `var2'
    local corr = r(rho)

    display "  `label' (n=`n_valid'):"
    display "    MAE:         " %12.6e `mae'
    display "    Max diff:    " %12.6e `maxdiff'
    display "    RMSE:        " %12.6e `rmse'
    display "    Correlacao:  " %10.8f `corr'

    * Cleanup
    quietly drop `diff' `absdiff' `sq'
end

* =============================================================================
* 1. Decomposicao Classica (airline.csv)
* =============================================================================
display _n "--- 1. Decomposicao Classica (airline.csv) ---" _n

* Verificar existencia dos arquivos
local has_python = 0
local has_r = 0
local has_stata = 0

capture confirm file "`output_python'/classical_components.csv"
if _rc == 0 local has_python = 1

capture confirm file "`output_r'/classical_components.csv"
if _rc == 0 local has_r = 1

capture confirm file "`output_stata'/classical_components.csv"
if _rc == 0 local has_stata = 1

display "Arquivos disponiveis:"
display "  Python: `has_python'"
display "  R:      `has_r'"
display "  Stata:  `has_stata'"

if `has_python' & `has_stata' {
    display _n "Comparacao Python vs Stata (decomposicao classica):" _n

    * Carregar Stata results
    import delimited "`output_stata'/classical_components.csv", clear
    rename seasonal_additive stata_seasonal_add
    rename seasonal_multiplicative stata_seasonal_mult
    rename trend_additive stata_trend_add
    rename trend_multiplicative stata_trend_mult
    keep date stata_*
    tempfile stata_classical
    save `stata_classical'

    * Carregar Python results
    import delimited "`output_python'/classical_components.csv", clear
    rename seasonal_additive py_seasonal_add
    rename seasonal_multiplicative py_seasonal_mult
    rename trend_additive py_trend_add
    rename trend_multiplicative py_trend_mult
    keep date py_*
    tempfile python_classical
    save `python_classical'

    * Merge
    use `stata_classical', clear
    merge 1:1 date using `python_classical', nogenerate

    display "Aditiva:"
    capture noisily compare_vectors stata_seasonal_add py_seasonal_add "Sazonal aditivo"
    capture noisily compare_vectors stata_trend_add py_trend_add "Trend aditivo"

    display _n "Multiplicativa:"
    capture noisily compare_vectors stata_seasonal_mult py_seasonal_mult "Sazonal multiplicativo"
    capture noisily compare_vectors stata_trend_mult py_trend_mult "Trend multiplicativo"

    display _n "Verificacao de tolerancia (classica Stata vs Python):"
    display "  NOTA: Stata usa implementacao manual com tssmooth ma."
    display "  Diferencas maiores que R vs Python sao esperadas."
    display "  Tolerancia: correlacao > 0.99 para todos os componentes."
}
else {
    display "Arquivos insuficientes para comparacao classica."
    display "Execute os scripts de cada plataforma primeiro."
}

* =============================================================================
* 2. STL / Filtros de Tendencia (co2.csv)
* =============================================================================
display _n _n "--- 2. Filtros de Tendencia (co2.csv) ---" _n

local has_python_stl = 0
local has_stata_stl = 0

capture confirm file "`output_python'/stl_components.csv"
if _rc == 0 local has_python_stl = 1

capture confirm file "`output_stata'/stl_components.csv"
if _rc == 0 local has_stata_stl = 1

display "Arquivos disponiveis:"
display "  Python (STL):      `has_python_stl'"
display "  Stata (tsfilter):  `has_stata_stl'"

if `has_python_stl' & `has_stata_stl' {
    display _n "Comparacao Python STL vs Stata tsfilter (tendencia):" _n

    * Carregar Stata
    import delimited "`output_stata'/stl_components.csv", clear
    rename co2_trend_hp stata_trend_hp
    rename co2_trend_hp2 stata_trend_hp2
    keep date stata_trend_hp stata_trend_hp2
    tempfile stata_stl
    save `stata_stl'

    * Carregar Python
    import delimited "`output_python'/stl_components.csv", clear
    capture confirm variable trend_s7
    if _rc == 0 {
        rename trend_s7 py_trend_s7
        rename trend_s15 py_trend_s15
        keep date py_trend_s7 py_trend_s15
        tempfile python_stl
        save `python_stl'

        use `stata_stl', clear
        merge 1:1 date using `python_stl', nogenerate

        display "Trend HP (Stata) vs Trend STL s=7 (Python):"
        capture noisily compare_vectors stata_trend_hp py_trend_s7 "HP vs STL s=7"

        display _n "Trend HP (Stata) vs Trend STL s=15 (Python):"
        capture noisily compare_vectors stata_trend_hp py_trend_s15 "HP vs STL s=15"

        display _n "NOTA: HP e STL usam metodos diferentes para trend."
        display "HP minimiza penalizacao L2; STL usa LOESS iterativo."
        display "Correlacao alta (> 0.99) esperada, valores absolutos diferem."
    }
    else {
        display "Colunas STL nao encontradas no arquivo Python."
    }
}
else {
    display "Arquivos insuficientes para comparacao STL/filtros."
}

* =============================================================================
* 3. X-13 / Proxy (brazil_ipca.csv)
* =============================================================================
display _n _n "--- 3. X-13 / Proxy (brazil_ipca.csv) ---" _n

local has_python_x13 = 0
local has_stata_x13 = 0

capture confirm file "`output_python'/x13_adjusted.csv"
if _rc == 0 local has_python_x13 = 1

capture confirm file "`output_stata'/x13_adjusted.csv"
if _rc == 0 local has_stata_x13 = 1

display "Arquivos disponiveis:"
display "  Python (STL proxy): `has_python_x13'"
display "  Stata (HP proxy):   `has_stata_x13'"

if `has_python_x13' & `has_stata_x13' {
    display _n "Comparacao Python STL proxy vs Stata HP proxy:" _n

    * Carregar Stata
    import delimited "`output_stata'/x13_adjusted.csv", clear
    rename sa_proxy stata_sa
    rename trend_hp stata_trend
    keep date stata_sa stata_trend
    tempfile stata_x13
    save `stata_x13'

    * Carregar Python
    import delimited "`output_python'/x13_adjusted.csv", clear
    capture confirm variable sa_stl_s13_robust
    if _rc == 0 {
        rename sa_stl_s13_robust py_sa
        rename trend_stl_s13_robust py_trend
        keep date py_sa py_trend
        tempfile python_x13
        save `python_x13'

        use `stata_x13', clear
        merge 1:1 date using `python_x13', nogenerate

        display "SA (Stata HP vs Python STL s=13 robusto):"
        capture noisily compare_vectors stata_sa py_sa "SA Stata vs Python"

        display _n "Trend (Stata HP vs Python STL s=13 robusto):"
        capture noisily compare_vectors stata_trend py_trend "Trend Stata vs Python"
    }
    else {
        display "Colunas esperadas nao encontradas no arquivo Python."
    }
}
else {
    display "Arquivos insuficientes para comparacao X-13/proxy."
}

* =============================================================================
* Resumo Final
* =============================================================================
display _n "============================================================================="
display "  RESUMO DA VALIDACAO - STATA"
display "============================================================================="
display ""
display "Metodos utilizados no Stata:"
display "  - Decomposicao classica: implementacao manual com tssmooth ma"
display "  - Filtros de tendencia: tsfilter hp, tsfilter bk, tsfilter cf"
display "  - Sazonalidade: dummies mensais via regressao"
display "  - X-13: pacote externo x13as (se disponivel)"
display ""
display "Tolerancias documentadas:"
display "  - Classica Stata vs Python/R: correlacao > 0.99"
display "    (implementacao manual pode diferir do stats::decompose)"
display "  - Trend HP vs STL: correlacao > 0.99"
display "    (metodos diferentes, mas ambos capturam tendencia longo prazo)"
display "  - SA proxy (HP+dummies) vs STL robusto: correlacao > 0.95"
display "    (metodos fundamentalmente distintos)"
display ""
display "LIMITACOES GERAIS DO STATA PARA DECOMPOSICAO:"
display "  1. Sem STL nativo - LOESS iterativo nao disponivel"
display "  2. X-13 via x13as requer binario externo"
display "  3. tsfilter nao separa componente sazonal explicitamente"
display "  4. Sazonalidade via dummies e fixa (nao varia no tempo)"
display "  5. Sem decomposicao multiplicativa nativa (tsfilter e aditivo)"
display "  6. Filtro BK perde observacoes nas bordas"
display ""
display "RECOMENDACAO:"
display "  Para decomposicao completa (trend+seasonal+remainder), usar R ou Python."
display "  Stata e adequado para filtragem de tendencia (HP) e dessazonalizacao"
display "  simples via dummies, mas nao substitui STL ou X-13 completo."
display ""
display "=== compare_results.do concluido ==="
