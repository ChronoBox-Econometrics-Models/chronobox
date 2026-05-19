* =============================================================================
* 01_var_validation.do
* Validacao cruzada: VAR estimation usando Stata
* Compara resultados com chronobox (Python)
* =============================================================================
* Comandos: varsoc, var, varlmar
* Dataset: us_macro_quarterly.csv
* Tolerancia esperada: < 1e-4 para coeficientes VAR
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de diretorios ----------------------------------------------
* Ajuste o caminho base conforme necessario
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

* Criar diretorio de saida
capture mkdir "`output_dir'"

display "=== VAR Validation (Stata) ==="
display "Data dir: `data_dir'"
display "Output dir: `output_dir'"
display ""

* --- Carregar dados ----------------------------------------------------------
* Importar CSV com dados macroeconomicos trimestrais dos EUA
import delimited "`data_dir'/us_macro_quarterly.csv", clear

display "Dataset: us_macro_quarterly.csv"
display "Observacoes: " _N
display "Variaveis: gdp, inflation, fed_funds, unemployment"
display ""

* --- Configurar serie temporal -----------------------------------------------
* Gerar variavel de tempo sequencial (trimestral)
gen t = _n
tsset t, quarterly

* Verificar a estrutura
describe gdp inflation fed_funds unemployment
summarize gdp inflation fed_funds unemployment

* --- Selecao de ordem (varsoc) -----------------------------------------------
display ""
display "--- Selecao de Ordem (varsoc) ---"

* varsoc calcula criterios de informacao para diferentes lags
varsoc gdp inflation fed_funds unemployment, maxlag(12)

* Salvar criterios de informacao em CSV
* Reestimar para capturar a matriz de resultados
quietly varsoc gdp inflation fed_funds unemployment, maxlag(12)

* Extrair criterios da matriz r(stats)
matrix ic_stats = r(stats)

* Salvar tabela de criterios de informacao
preserve
clear
svmat ic_stats, names(col)

* Renomear colunas (varsoc retorna: lag, LL, LR, df, p, FPE, AIC, HQIC, SBIC)
* A ordem exata depende da versao do Stata, vamos usar os nomes genericos
gen lag = _n
order lag

* Exportar criterios de informacao
export delimited using "`output_dir'/var_ic.csv", replace
display "Criterios de informacao salvos em: outputs/Stata/var_ic.csv"
restore

* --- Estimacao VAR(1) --------------------------------------------------------
display ""
display "--- Estimacao VAR(1) ---"

* Estimar VAR(1) com constante (padrao do Stata)
var gdp inflation fed_funds unemployment, lags(1)

display "Modelo: VAR(1) com constante"
display "N. obs: " e(N)
display "N. equacoes: " e(neqs)
display ""

* --- Extrair e exibir coeficientes -------------------------------------------
display "--- Coeficientes ---"

* Matriz de coeficientes
matrix coefs = e(b)
matrix var_coefs = e(b)

* Exibir coeficientes por equacao
display "Coeficientes da equacao gdp:"
display "  gdp.L1 = " _b[gdp:L.gdp]
display "  inflation.L1 = " _b[gdp:L.inflation]
display "  fed_funds.L1 = " _b[gdp:L.fed_funds]
display "  unemployment.L1 = " _b[gdp:L.unemployment]
display "  _cons = " _b[gdp:_cons]

display ""
display "Coeficientes da equacao inflation:"
display "  gdp.L1 = " _b[inflation:L.gdp]
display "  inflation.L1 = " _b[inflation:L.inflation]
display "  fed_funds.L1 = " _b[inflation:L.fed_funds]
display "  unemployment.L1 = " _b[inflation:L.unemployment]
display "  _cons = " _b[inflation:_cons]

display ""
display "Coeficientes da equacao fed_funds:"
display "  gdp.L1 = " _b[fed_funds:L.gdp]
display "  inflation.L1 = " _b[fed_funds:L.inflation]
display "  fed_funds.L1 = " _b[fed_funds:L.fed_funds]
display "  unemployment.L1 = " _b[fed_funds:L.unemployment]
display "  _cons = " _b[fed_funds:_cons]

display ""
display "Coeficientes da equacao unemployment:"
display "  gdp.L1 = " _b[unemployment:L.gdp]
display "  inflation.L1 = " _b[unemployment:L.inflation]
display "  fed_funds.L1 = " _b[unemployment:L.fed_funds]
display "  unemployment.L1 = " _b[unemployment:L.unemployment]
display "  _cons = " _b[unemployment:_cons]

* --- Salvar coeficientes em CSV ----------------------------------------------
* Montar matriz A_1 (4x4) + interceptos
preserve
clear
set obs 4

gen str20 equation = ""
gen double intercept = .
gen double gdp_l1 = .
gen double inflation_l1 = .
gen double fed_funds_l1 = .
gen double unemployment_l1 = .

* Equacao gdp
replace equation = "gdp" in 1
replace intercept = _b[gdp:_cons] in 1
replace gdp_l1 = _b[gdp:L.gdp] in 1
replace inflation_l1 = _b[gdp:L.inflation] in 1
replace fed_funds_l1 = _b[gdp:L.fed_funds] in 1
replace unemployment_l1 = _b[gdp:L.unemployment] in 1

* Equacao inflation
replace equation = "inflation" in 2
replace intercept = _b[inflation:_cons] in 2
replace gdp_l1 = _b[inflation:L.gdp] in 2
replace inflation_l1 = _b[inflation:L.inflation] in 2
replace fed_funds_l1 = _b[inflation:L.fed_funds] in 2
replace unemployment_l1 = _b[inflation:L.unemployment] in 2

* Equacao fed_funds
replace equation = "fed_funds" in 3
replace intercept = _b[fed_funds:_cons] in 3
replace gdp_l1 = _b[fed_funds:L.gdp] in 3
replace inflation_l1 = _b[fed_funds:L.inflation] in 3
replace fed_funds_l1 = _b[fed_funds:L.fed_funds] in 3
replace unemployment_l1 = _b[fed_funds:L.unemployment] in 3

* Equacao unemployment
replace equation = "unemployment" in 4
replace intercept = _b[unemployment:_cons] in 4
replace gdp_l1 = _b[unemployment:L.gdp] in 4
replace inflation_l1 = _b[unemployment:L.inflation] in 4
replace fed_funds_l1 = _b[unemployment:L.fed_funds] in 4
replace unemployment_l1 = _b[unemployment:L.unemployment] in 4

export delimited using "`output_dir'/var_coefficients.csv", replace
display ""
display "Coeficientes salvos em: outputs/Stata/var_coefficients.csv"
restore

* --- Estabilidade do VAR -----------------------------------------------------
display ""
display "--- Estabilidade ---"
varstable

* Salvar eigenvalues da estabilidade
varstable, estimates

* --- Teste de autocorrelacao residual (varlmar) ------------------------------
display ""
display "--- Teste de Autocorrelacao Residual (LM) ---"
varlmar, mlag(8)

* Salvar resultados do teste LM
preserve
clear
set obs 8
gen int lag = _n
gen double chi2 = .
gen double df = .
gen double pvalue = .

* Nota: Os valores exatos dependem da execucao; aqui documentamos a estrutura
* Os resultados sao exibidos no output do Stata
* Para exportacao automatica, reestimar e capturar r()
restore

* --- Covariancia residual ----------------------------------------------------
display ""
display "--- Covariancia Residual ---"
estat residuals

* Exibir matriz de covariancia
matrix sigma = e(Sigma)
matrix list sigma, format(%12.8f)

display ""
display "=== Fim da validacao VAR (Stata) ==="
