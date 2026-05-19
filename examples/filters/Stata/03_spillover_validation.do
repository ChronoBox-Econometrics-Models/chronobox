********************************************************************************
* 03_spillover_validation.do
*
* Validacao cruzada do Diebold-Yilmaz Spillover Index usando VAR + FEVD
* manual em Stata.
*
* IMPORTANTE: O spillover index de Diebold-Yilmaz NAO e um comando nativo
* do Stata. A implementacao aqui usa:
*   1. var — estimacao VAR nativa do Stata
*   2. fcast compute / irf create — para FEVD
*   3. Calculo manual do spillover a partir da tabela FEVD
*
* Metodologia:
*   - Estimar VAR(p) com p=2 lags em k=4 variaveis
*   - Calcular FEVD (Forecast Error Variance Decomposition) com horizonte H=10
*   - Normalizar para que cada linha some 100%
*   - Total spillover = soma off-diagonal / k * 100
*   - FROM_i = soma da linha i excluindo diagonal / k * 100
*   - TO_i = soma da coluna i excluindo diagonal / k * 100
*   - NET_i = TO_i - FROM_i
*
* LIMITACOES DO STATA:
*   1. Spillover index nao e comando nativo
*   2. FEVD do Stata usa decomposicao de Cholesky (ordering-dependent)
*      enquanto Diebold-Yilmaz (2012) usa Generalized FEVD (KPPS)
*   3. Para GVD (generalized): precisaria do pacote 'spillover' (ssc)
*      ou implementacao manual mais complexa
*   4. Resultados diferirao do Python/R se usarem GVD vs Cholesky
*
* Dataset: serie sintetica multivariada gerada pelo Python
*          + PIB EUA vs PIB Brasil
*
* Saida: examples/filters/outputs/Stata/spillover_fevd.csv
*        examples/filters/outputs/Stata/spillover_summary.csv
*        examples/filters/outputs/Stata/spillover_gdp.csv
********************************************************************************

clear all
set more off
set seed 42

local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"
capture mkdir "`output_dir'"

display _newline
display "=== Validacao Spillover Index (Diebold-Yilmaz) via VAR + FEVD ==="
display _newline

* ======================================================================
* 1. Gerar dados sinteticos multivariados (4 variaveis)
*
* Reproduzindo a estrutura do gerador Python:
*   - Fator comum: sin(2*pi*t/32) + 0.5*sin(2*pi*t/16) + random walk
*   - k=4 series com peso 0.6 no fator comum + 0.4 idiossincrático
*   - Tendencia linear: 100 + 0.3*t
*
* NOTA: RNG do Stata difere do NumPy, entao os valores exatos serao
* diferentes. A comparacao e estrutural (mesma magnitude e propriedades).
* ======================================================================
display "--- 1. Gerando serie sintetica multivariada (4 variaveis) ---"

local n_obs = 200
local k_vars = 4
local common_weight = 0.6

set obs `n_obs'

* Indice temporal
generate t = _n

* Fator comum: ciclos + random walk
generate double common = sin(2 * _pi * t / 32) + 0.5 * sin(2 * _pi * t / 16)
generate double rw_shock = rnormal(0, 0.05)
generate double rw = sum(rw_shock)
replace common = common + rw

* Gerar 4 series
forvalues j = 1/`k_vars' {
    * Componente idiossincrático: random walk + ciclo proprio
    generate double idio_shock_`j' = rnormal(0, 0.1)
    generate double idio_`j' = sum(idio_shock_`j')
    local period_j = 20 + 5 * `j'
    replace idio_`j' = idio_`j' + 0.8 * sin(2 * _pi * t / `period_j')

    * Serie final: ponderacao comum + idio + tendencia + ruido
    generate double var_`j' = `common_weight' * common ///
        + (1 - `common_weight') * idio_`j' ///
        + 100 + 0.3 * t ///
        + rnormal(0, 0.2)
}

* Datas trimestrais a partir de 1970-Q1
generate tq = yq(1970, 1) + t - 1
format tq %tq
tsset tq

* Primeiras diferencas (para estacionariedade)
forvalues j = 1/`k_vars' {
    generate double dvar_`j' = D.var_`j'
}

* ======================================================================
* 2. Estimar VAR(2) nas primeiras diferencas
* ======================================================================
display _newline
display "--- 2. Estimando VAR(2) ---"

var dvar_1 dvar_2 dvar_3 dvar_4, lags(1/2)

* ======================================================================
* 3. Calcular FEVD com horizonte H=10
*
* irf create gera a FEVD baseada em Cholesky.
* Precisamos extrair os valores do horizonte 10 para montar a tabela.
* ======================================================================
display _newline
display "--- 3. Calculando FEVD (Cholesky, H=10) ---"

* Criar IRF/FEVD
irf create spillover_irf, set(spillover_irf, replace) step(10)

* Extrair FEVD no horizonte 10
* A FEVD em Stata e acessada via irf table ou irf graph
* Vamos extrair os valores programaticamente

* Montar matriz FEVD 4x4 no horizonte 10
* fevd(i,j) = proporcao da variancia do erro de previsao de i explicada por j
matrix fevd_mat = J(4, 4, 0)

* Para cada variavel resposta (i) e cada impulso (j)
local var_names "dvar_1 dvar_2 dvar_3 dvar_4"

local i = 0
foreach resp of local var_names {
    local i = `i' + 1
    local j = 0
    foreach imp of local var_names {
        local j = `j' + 1

        * Acessar FEVD: irf table fevd, irf(spillover_irf) impulse(imp) response(resp)
        quietly irf table fevd, irf(spillover_irf) impulse(`imp') response(`resp')

        * O ultimo valor (step 10) esta na matrix r(table)
        * r(table) tem colunas: step, fevd, lower, upper
        matrix temp_tab = r(table)
        local nrows = rowsof(temp_tab)
        matrix fevd_mat[`i', `j'] = temp_tab[`nrows', 1]
    }
}

* Exibir matriz FEVD
display _newline
display "  Matriz FEVD (Cholesky) no horizonte H=10:"
matrix list fevd_mat, format(%8.4f)

* ======================================================================
* 4. Calcular spillover indices
* ======================================================================
display _newline
display "--- 4. Calculando spillover indices ---"

* Normalizar linhas para somar 1
matrix fevd_norm = J(4, 4, 0)
forvalues i = 1/4 {
    local row_sum = 0
    forvalues j = 1/4 {
        local row_sum = `row_sum' + fevd_mat[`i', `j']
    }
    forvalues j = 1/4 {
        matrix fevd_norm[`i', `j'] = fevd_mat[`i', `j'] / `row_sum'
    }
}

* Total spillover = (soma total - soma diagonal) / k * 100
local total_sum = 0
local diag_sum = 0
forvalues i = 1/4 {
    forvalues j = 1/4 {
        local total_sum = `total_sum' + fevd_norm[`i', `j']
    }
    local diag_sum = `diag_sum' + fevd_norm[`i', `i']
}
local total_spillover = (`total_sum' - `diag_sum') / `k_vars' * 100

display "  Total spillover: " %8.4f `total_spillover'

* Directional FROM e TO
forvalues i = 1/4 {
    * FROM_i = (soma linha i - diagonal) / k * 100
    local from_`i' = 0
    forvalues j = 1/4 {
        local from_`i' = `from_`i'' + fevd_norm[`i', `j']
    }
    local from_`i' = (`from_`i'' - fevd_norm[`i', `i']) / `k_vars' * 100

    * TO_i = (soma coluna i - diagonal) / k * 100
    local to_`i' = 0
    forvalues j = 1/4 {
        local to_`i' = `to_`i'' + fevd_norm[`j', `i']
    }
    local to_`i' = (`to_`i'' - fevd_norm[`i', `i']) / `k_vars' * 100

    * NET_i = TO_i - FROM_i
    local net_`i' = `to_`i'' - `from_`i''

    display "  Var `i': FROM = " %8.4f `from_`i'' ///
            "  TO = " %8.4f `to_`i'' ///
            "  NET = " %8.4f `net_`i''
}

* ======================================================================
* 5. Exportar FEVD matrix
* ======================================================================
display _newline
display "--- 5. Exportando resultados ---"

* FEVD normalizada como CSV
preserve
clear
set obs 4
generate variable = ""
replace variable = "var_1" in 1
replace variable = "var_2" in 2
replace variable = "var_3" in 3
replace variable = "var_4" in 4

forvalues j = 1/4 {
    generate double var_`j' = .
    forvalues i = 1/4 {
        replace var_`j' = fevd_norm[`i', `j'] in `i'
    }
}

export delimited using "`output_dir'/spillover_fevd.csv", replace
display "  FEVD salva em: `output_dir'/spillover_fevd.csv"
restore

* Spillover summary
preserve
clear
set obs 4
generate variable = ""
generate double from_spillover = .
generate double to_spillover = .
generate double net_spillover = .

forvalues i = 1/4 {
    replace variable = "var_`i'" in `i'
    replace from_spillover = `from_`i'' in `i'
    replace to_spillover = `to_`i'' in `i'
    replace net_spillover = `net_`i'' in `i'
}

export delimited using "`output_dir'/spillover_summary.csv", replace
display "  Summary salvo em: `output_dir'/spillover_summary.csv"
restore

* ======================================================================
* 6. PIB EUA vs PIB Brasil
* ======================================================================
display _newline
display "--- 6. PIB EUA vs PIB Brasil ---"

* Carregar e preparar dados dos EUA
import delimited "`data_dir'/us_gdp_quarterly.csv", clear
generate year = real(substr(date, 1, 4))
generate month = real(substr(date, 6, 2))
generate qtr = ceil(month / 3)
generate tq_us = yq(year, qtr)
rename gdp_log gdp_us
keep tq_us gdp_us date
rename date date_us
rename tq_us tq
tempfile us_data
save `us_data'

* Carregar dados do Brasil
import delimited "`data_dir'/brazil_gdp.csv", clear
generate year = real(substr(date, 1, 4))
generate month = real(substr(date, 6, 2))
generate qtr = ceil(month / 3)
generate tq = yq(year, qtr)
rename gdp_log gdp_br
keep tq gdp_br

* Merge pelo trimestre
merge 1:1 tq using `us_data', keep(match) nogenerate

format tq %tq
tsset tq

* Primeiras diferencas
generate double dgdp_us = D.gdp_us
generate double dgdp_br = D.gdp_br

* Estimar VAR(2)
var dgdp_us dgdp_br, lags(1/2)

* FEVD
irf create gdp_irf, set(gdp_irf, replace) step(10)

* Extrair FEVD 2x2
matrix fevd_gdp = J(2, 2, 0)

local gdp_vars "dgdp_us dgdp_br"
local i = 0
foreach resp of local gdp_vars {
    local i = `i' + 1
    local j = 0
    foreach imp of local gdp_vars {
        local j = `j' + 1
        quietly irf table fevd, irf(gdp_irf) impulse(`imp') response(`resp')
        matrix temp_tab = r(table)
        local nrows = rowsof(temp_tab)
        matrix fevd_gdp[`i', `j'] = temp_tab[`nrows', 1]
    }
}

display _newline
display "  FEVD PIB EUA vs PIB Brasil (Cholesky, H=10):"
matrix list fevd_gdp, format(%8.4f)

* Normalizar e calcular spillover
matrix fevd_gdp_norm = J(2, 2, 0)
forvalues i = 1/2 {
    local row_sum = 0
    forvalues j = 1/2 {
        local row_sum = `row_sum' + fevd_gdp[`i', `j']
    }
    forvalues j = 1/2 {
        matrix fevd_gdp_norm[`i', `j'] = fevd_gdp[`i', `j'] / `row_sum'
    }
}

local gdp_total = 0
local gdp_diag = 0
forvalues i = 1/2 {
    forvalues j = 1/2 {
        local gdp_total = `gdp_total' + fevd_gdp_norm[`i', `j']
    }
    local gdp_diag = `gdp_diag' + fevd_gdp_norm[`i', `i']
}
local gdp_spillover = (`gdp_total' - `gdp_diag') / 2 * 100

local from_us = (fevd_gdp_norm[1,1] + fevd_gdp_norm[1,2] - fevd_gdp_norm[1,1]) / 2 * 100
local from_br = (fevd_gdp_norm[2,1] + fevd_gdp_norm[2,2] - fevd_gdp_norm[2,2]) / 2 * 100
local to_us = (fevd_gdp_norm[1,1] + fevd_gdp_norm[2,1] - fevd_gdp_norm[1,1]) / 2 * 100
local to_br = (fevd_gdp_norm[1,2] + fevd_gdp_norm[2,2] - fevd_gdp_norm[2,2]) / 2 * 100
local net_us = `to_us' - `from_us'
local net_br = `to_br' - `from_br'

display "  Total spillover: " %8.4f `gdp_spillover'
display "  FROM PIB EUA: " %8.4f `from_us' "  FROM PIB Brasil: " %8.4f `from_br'
display "  TO PIB EUA: " %8.4f `to_us' "  TO PIB Brasil: " %8.4f `to_br'
display "  Net PIB EUA: " %8.4f `net_us' "  Net PIB Brasil: " %8.4f `net_br'

* Exportar
preserve
clear
set obs 2
generate variable = ""
replace variable = "PIB_EUA" in 1
replace variable = "PIB_Brasil" in 2

generate double from_spillover = .
replace from_spillover = `from_us' in 1
replace from_spillover = `from_br' in 2

generate double to_spillover = .
replace to_spillover = `to_us' in 1
replace to_spillover = `to_br' in 2

generate double net_spillover = .
replace net_spillover = `net_us' in 1
replace net_spillover = `net_br' in 2

export delimited using "`output_dir'/spillover_gdp.csv", replace
display "  GDP spillover salvo em: `output_dir'/spillover_gdp.csv"
restore

* ======================================================================
* Notas sobre limitacoes
* ======================================================================
display _newline
display "=== Limitacoes do Stata para Spillover Index ==="
display ""
display "1. Spillover index NAO e comando nativo do Stata."
display "   Implementado manualmente via var + irf create + fevd."
display ""
display "2. FEVD do Stata usa decomposicao de Cholesky, que depende"
display "   da ordenacao das variaveis. Diebold-Yilmaz (2012) propoe"
display "   Generalized VD (KPPS, 1998) que e ordering-invariant."
display ""
display "3. Diferenças esperadas vs Python/R:"
display "   - Se ambos usam Cholesky: diferenças ~1e-6 (numericas)"
display "   - Se Python/R usam GVD: diferenças podem ser substanciais"
display "   - Rolling spillover nao implementado aqui (complexidade)"
display ""
display "4. Para GVD em Stata, considerar:"
display "   - ssc install spillover (se disponivel)"
display "   - Implementacao manual via mata (algebra matricial)"
display ""

display "=== 03_spillover_validation.do concluido com sucesso ==="
