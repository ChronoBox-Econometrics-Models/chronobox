* =============================================================================
* 03_bvar_validation.do
* Validacao cruzada: Bayesian VAR (BVAR) usando Stata
* Compara resultados com chronobox (Python) e R (BVAR package)
* =============================================================================
* Dataset: us_macro_quarterly.csv
* Variaveis: gdp, inflation, fed_funds, unemployment
* =============================================================================
*
* NOTA IMPORTANTE: BVAR NAO E NATIVO NO STATA
*
* O Stata nao possui comando built-in para Bayesian VAR com prior Minnesota.
* Este script documenta as alternativas disponiveis e fornece uma aproximacao
* usando as ferramentas nativas do Stata.
*
* Opcoes disponiveis:
*
* 1. bayes: var (Stata 17+)
*    - O Stata 17 introduziu o prefixo `bayes:` que permite estimacao
*      bayesiana de modelos VAR. No entanto, a prior Minnesota padrao
*      (Litterman, 1986) nao e diretamente suportada como opcao pre-
*      -configurada. E possivel especificar priors customizadas.
*
* 2. Pacotes externos do SSC:
*    - bvar12 (Dieppe, Legrand, van Roye): BVAR toolbox para Stata
*    - bvartools: similar ao pacote R
*    - Estes pacotes podem nao estar atualizados ou disponíveis
*
* 3. Implementacao manual via Mata:
*    - E possivel implementar o Gibbs sampler para BVAR com prior
*      Minnesota usando a linguagem Mata do Stata
*    - Requer conhecimento avancado de programacao em Stata
*
* ABORDAGEM DESTE SCRIPT:
* Como alternativa, demonstramos:
*   a) VAR classico (OLS) como baseline
*   b) bayes: var com priors diffuse (se Stata 17+ disponivel)
*   c) Comparacao qualitativa com resultados Python/R
* =============================================================================

clear all
set more off
set seed 42

* --- Configuracao de diretorios ----------------------------------------------
local base_dir ".."
local data_dir "`base_dir'/data"
local output_dir "`base_dir'/outputs/Stata"

capture mkdir "`output_dir'"

display "======================================================================="
display "BVAR Validation (Stata)"
display "======================================================================="
display ""
display "NOTA: BVAR com prior Minnesota nao e nativo no Stata."
display "Este script demonstra alternativas e documenta limitacoes."
display ""

* =============================================================================
* PARTE 1: Carregar dados
* =============================================================================

display "--- Carregar Dados ---"

import delimited "`data_dir'/us_macro_quarterly.csv", clear

display "Dataset: us_macro_quarterly.csv"
display "Observacoes: " _N
display ""

gen t = _n
tsset t, quarterly

summarize gdp inflation fed_funds unemployment

* =============================================================================
* PARTE 2: VAR classico (baseline para comparacao)
* =============================================================================

display ""
display "--- VAR(4) Classico (OLS) como Baseline ---"

var gdp inflation fed_funds unemployment, lags(1/4)

display "VAR(4) classico estimado"
display "N. obs: " e(N)
display ""

* Salvar coeficientes OLS para comparacao com BVAR
matrix ols_coefs = e(b)
matrix sigma_ols = e(Sigma)

display "--- Sigma_u (OLS) ---"
matrix list sigma_ols, format(%12.8f)

* Previsao OLS (baseline)
display ""
display "--- Previsoes OLS (12 passos a frente) ---"

* Usar fcast para gerar previsoes
fcast compute ols_fc_, step(12) dynamic(.)

* Exibir previsoes
list ols_fc_gdp ols_fc_inflation ols_fc_fed_funds ols_fc_unemployment ///
    if ols_fc_gdp != . , noobs

* Salvar previsoes OLS
preserve
keep if ols_fc_gdp != .
gen horizon = _n
keep horizon ols_fc_gdp ols_fc_inflation ols_fc_fed_funds ols_fc_unemployment
rename ols_fc_gdp gdp_forecast
rename ols_fc_inflation inflation_forecast
rename ols_fc_fed_funds fed_funds_forecast
rename ols_fc_unemployment unemployment_forecast
export delimited using "`output_dir'/var_ols_forecasts.csv", replace
display "Salvo: var_ols_forecasts.csv"
restore

* Limpar previsoes
drop ols_fc_*

* =============================================================================
* PARTE 3: Tentativa de bayes: var (Stata 17+)
* =============================================================================

display ""
display "--- Tentativa: bayes: var (requer Stata 17+) ---"
display ""

* Verificar se bayes esta disponivel
capture which bayes
if _rc == 0 {
    display "Comando bayes disponivel. Tentando bayes: var ..."
    display ""

    * bayes: var com priors diffuse (nao e Minnesota, mas e Bayesiano)
    * A prior Minnesota precisaria ser especificada manualmente
    capture noisily bayes, rseed(42) mcmcsize(5000) burnin(1000): ///
        var gdp inflation fed_funds unemployment, lags(1/4)

    if _rc == 0 {
        display "bayes: var estimado com sucesso"
        display ""

        * Salvar resultados bayesianos
        * bayesstats summary
        capture noisily bayesstats summary

        * Previsoes bayesianas
        capture noisily bayespredict fc_, rseed(42) saving("`output_dir'/bvar_draws", replace)
        if _rc == 0 {
            display "Previsoes bayesianas geradas"
        }
        else {
            display "Nota: bayespredict nao disponivel ou falhou"
        }
    }
    else {
        display "bayes: var falhou (possivelmente versao do Stata insuficiente)"
        display "Continuando com alternativas..."
    }
}
else {
    display "Comando bayes nao disponivel nesta versao do Stata."
    display "Requer Stata 17 ou superior."
    display "Continuando com VAR classico como referencia."
}

* =============================================================================
* PARTE 4: Aproximacao manual via ridge-like shrinkage
* =============================================================================

display ""
display "--- Aproximacao: Shrinkage Manual (Ridge-like) ---"
display ""
display "Como proxy para o efeito da prior Minnesota, aplicamos shrinkage"
display "manual nos coeficientes OLS em direcao ao prior (random walk)."
display ""

* Extrair coeficientes OLS em formato de matriz
* Para cada equacao, o prior Minnesota shrinks:
*   - Own lag 1 -> 1 (random walk)
*   - Cross lags -> 0
*   - Higher lags -> 0 (com decay)

* Lambda (tightness) = 0.1 (mesmo usado em R e Python)
local lambda = 0.1

display "Shrinkage parameter (lambda): `lambda'"
display ""
display "Nota: Esta e uma APROXIMACAO simples. O BVAR verdadeiro usa"
display "Gibbs sampling com prior Minnesota completo (Litterman, 1986)."
display "Para resultados exatos, usar R (BVAR) ou Python (chronobox)."
display ""

* Coeficientes shrunk = lambda * OLS + (1-lambda) * prior
* Prior Minnesota: coef(own lag 1) = 1, todos outros = 0
* Esta e uma simplificacao grosseira do prior Minnesota real

display "Coeficientes OLS vs Prior Minnesota (lag 1, equacao GDP):"
display "  gdp.L1 (OLS):    " _b[gdp:L.gdp]
display "  gdp.L1 (prior):  1.0 (random walk)"
display "  gdp.L1 (shrunk): " `lambda' * 1.0 + (1-`lambda') * _b[gdp:L.gdp]
display ""
display "  inflation.L1 (OLS):    " _b[gdp:L.inflation]
display "  inflation.L1 (prior):  0.0 (cross-variable)"
display "  inflation.L1 (shrunk): " (1-`lambda') * _b[gdp:L.inflation]

* =============================================================================
* PARTE 5: Salvar documentacao de limitacoes
* =============================================================================

display ""
display "--- Exportar Documentacao ---"

preserve
clear
set obs 10
gen str60 topic = ""
gen str200 description = ""

replace topic = "bvar_availability" in 1
replace description = "BVAR with Minnesota prior is NOT natively available in Stata" in 1

replace topic = "bayes_prefix" in 2
replace description = "Stata 17+ has bayes: var but without pre-configured Minnesota prior" in 2

replace topic = "ssc_packages" in 3
replace description = "bvar12 (Dieppe et al.) available on SSC but may be outdated" in 3

replace topic = "mata_implementation" in 4
replace description = "Manual Gibbs sampler can be implemented in Mata" in 4

replace topic = "minnesota_prior" in 5
replace description = "lambda1=0.1, lambda2=0.5, lambda3=1.0 used in Python/R" in 5

replace topic = "n_draws_python_r" in 6
replace description = "Python/R use 5000 posterior draws with 1000 burn-in" in 6

replace topic = "recommendation" in 7
replace description = "Use R (BVAR package) or Python (chronobox) for BVAR with Minnesota" in 7

replace topic = "ols_baseline" in 8
replace description = "OLS VAR(4) coefficients serve as comparison baseline" in 8

replace topic = "shrinkage_note" in 9
replace description = "Ridge-like shrinkage is a rough approximation, not true BVAR" in 9

replace topic = "cross_validation" in 10
replace description = "Compare OLS forecasts with BVAR posterior means from Python/R" in 10

export delimited using "`output_dir'/bvar_limitations.csv", replace
display "Salvo: bvar_limitations.csv"
restore

* =============================================================================
* PARTE 6: Comparacao com resultados Python/R
* =============================================================================

display ""
display "--- Comparacao com Python/R (se arquivos disponíveis) ---"

* Tentar carregar previsoes do Python
capture confirm file "`base_dir'/outputs/bvar_forecasts.csv"
if _rc == 0 {
    display "Arquivo Python encontrado: outputs/bvar_forecasts.csv"

    preserve
    import delimited "`base_dir'/outputs/bvar_forecasts.csv", clear

    display "Previsoes BVAR Python (primeiras 4 linhas):"
    list in 1/4, noobs
    restore
}
else {
    display "Arquivo Python (bvar_forecasts.csv) nao encontrado. Pulando."
}

* Tentar carregar previsoes do R
capture confirm file "`base_dir'/outputs/R/bvar_forecasts.csv"
if _rc == 0 {
    display "Arquivo R encontrado: outputs/R/bvar_forecasts.csv"

    preserve
    import delimited "`base_dir'/outputs/R/bvar_forecasts.csv", clear

    display "Previsoes BVAR R (primeiras 4 linhas):"
    list in 1/4, noobs
    restore
}
else {
    display "Arquivo R (bvar_forecasts.csv) nao encontrado. Pulando."
}

* =============================================================================
* PARTE 7: Resumo e limitacoes
* =============================================================================

display ""
display "======================================================================="
display "RESUMO: LIMITACOES DO STATA PARA BVAR"
display "======================================================================="
display ""
display "1. BVAR com prior Minnesota:"
display "   - NAO nativo no Stata (nenhuma versao)"
display "   - bayes: var (Stata 17+) permite priors customizadas,"
display "     mas requer especificacao manual da prior Minnesota"
display ""
display "2. Pacotes externos:"
display "   - bvar12 (SSC): implementa BVAR, mas pode estar desatualizado"
display "   - Instalacao: ssc install bvar12"
display "   - Documentacao limitada comparada com R/Python"
display ""
display "3. Alternativas recomendadas:"
display "   - R: pacote BVAR (bvar() com bv_minnesota())"
display "   - Python: chronobox (BVAR com prior Minnesota)"
display "   - Ambos oferecem Gibbs sampling completo e diagnosticos"
display ""
display "4. O que o Stata FAZ bem para VAR/SVAR:"
display "   - VAR classico (var)"
display "   - SVAR com restricoes Cholesky e AB (svar)"
display "   - SVAR com restricoes de longo prazo (svar + lrestrictions)"
display "   - IRFs, FEVDs, testes de diagnostico (irf, varlmar, varstable)"
display ""
display "=== Fim da validacao BVAR (Stata) ==="
