* ==============================================================================
* 04_arfima_validation.do
* Validacao cruzada: ARFIMA (Autoregressive Fractionally Integrated MA)
*
* Objetivo: Estimar modelos ARFIMA no Stata para validacao cruzada
*           dos resultados do chronobox (Python) e fracdiff (R).
*
* Datasets: nile.csv (flow), brazil_ipca.csv (ipca)
*
* NOTA IMPORTANTE SOBRE LIMITACOES DO STATA:
* -------------------------------------------
* - O comando `arfima` foi introduzido no Stata 14.
* - Versoes anteriores NAO possuem ARFIMA nativo.
* - Se o comando `arfima` nao estiver disponivel, este script
*   documentara a limitacao e salvara um arquivo de referencia.
* - A estimacao de d fracionario no Stata usa ML (Maximum Likelihood),
*   enquanto Python usa GPH e Local Whittle (semi-parametricos).
*   Diferencas nos estimadores sao esperadas.
*
* Dependencias: Stata 14+ (para comando arfima nativo)
* ==============================================================================

clear all
set more off
set seed 42

* --- Caminhos -----------------------------------------------------------------
local data_dir   "../data"
local output_dir "../outputs/Stata"

capture mkdir "`output_dir'"

* ==============================================================================
* SECAO 1: Verificar disponibilidade do comando arfima
* ==============================================================================

display _newline(2)
display "================================================================"
display "  ARFIMA - Verificacao de disponibilidade"
display "================================================================"

* Testar se o comando arfima esta disponivel
capture which arfima
local arfima_available = (_rc == 0)

if (`arfima_available') {
    display "Comando arfima encontrado. Prosseguindo com estimacao."
}
else {
    display "AVISO: Comando arfima NAO disponivel nesta versao do Stata."
    display "       O comando arfima requer Stata 14 ou superior."
    display "       Este script documentara as limitacoes e exportara referencia."
}

* ==============================================================================
* SECAO 2: ARFIMA para Nile (se disponivel)
* ==============================================================================

display _newline(2)
display "=== ARFIMA para Nile ==="

import delimited "`data_dir'/nile.csv", clear varnames(1)

generate year = real(substr(date, 1, 4))
tsset year

display "Nile: " _N " observacoes"

if (`arfima_available') {
    * --- ARFIMA(1,d,1) com d fracionario --------------------------------------
    display _newline(1)
    display "--- Nile ARFIMA(1,d,1) ---"
    capture noisily {
        arfima flow, ar(1) ma(1)
        matrix b_nile_arfima = e(b)
        scalar d_nile = b_nile_arfima[1, colnumb(b_nile_arfima, "ARFIMA:d")]
        scalar aic_nile_arfima = -2*e(ll) + 2*e(k)
        scalar bic_nile_arfima = -2*e(ll) + e(k)*ln(e(N))
        display "d estimado (Nile) = " d_nile
        display "AIC = " aic_nile_arfima
        display "BIC = " bic_nile_arfima
        matrix list b_nile_arfima
        local nile_arfima_ok = 1
    }
    if (_rc != 0) {
        display "AVISO: ARFIMA(1,d,1) nao convergiu para Nile."
        local nile_arfima_ok = 0
    }

    * --- ARFIMA(0,d,0) - apenas d fracionario ---------------------------------
    display _newline(1)
    display "--- Nile ARFIMA(0,d,0) ---"
    capture noisily {
        arfima flow
        matrix b_nile_frac = e(b)
        scalar d_nile_only = b_nile_frac[1, colnumb(b_nile_frac, "ARFIMA:d")]
        scalar aic_nile_frac = -2*e(ll) + 2*e(k)
        display "d estimado (apenas fracionario) = " d_nile_only
        display "AIC = " aic_nile_frac
        local nile_frac_ok = 1
    }
    if (_rc != 0) {
        display "AVISO: ARFIMA(0,d,0) nao convergiu para Nile."
        local nile_frac_ok = 0
    }
}
else {
    display "Pulando estimacao ARFIMA para Nile (comando nao disponivel)."
    local nile_arfima_ok = 0
    local nile_frac_ok = 0
}

* ==============================================================================
* SECAO 3: ARFIMA para IPCA (se disponivel)
* ==============================================================================

display _newline(2)
display "=== ARFIMA para IPCA ==="

import delimited "`data_dir'/brazil_ipca.csv", clear varnames(1)

generate mdate = monthly(substr(date, 1, 7), "YM")
format mdate %tm
tsset mdate

display "IPCA: " _N " observacoes"

if (`arfima_available') {
    * --- ARFIMA(1,d,1) --------------------------------------------------------
    display _newline(1)
    display "--- IPCA ARFIMA(1,d,1) ---"
    capture noisily {
        arfima ipca, ar(1) ma(1)
        matrix b_ipca_arfima = e(b)
        scalar d_ipca = b_ipca_arfima[1, colnumb(b_ipca_arfima, "ARFIMA:d")]
        scalar aic_ipca_arfima = -2*e(ll) + 2*e(k)
        scalar bic_ipca_arfima = -2*e(ll) + e(k)*ln(e(N))
        display "d estimado (IPCA) = " d_ipca
        display "AIC = " aic_ipca_arfima
        display "BIC = " bic_ipca_arfima
        matrix list b_ipca_arfima
        local ipca_arfima_ok = 1
    }
    if (_rc != 0) {
        display "AVISO: ARFIMA(1,d,1) nao convergiu para IPCA."
        local ipca_arfima_ok = 0
    }

    * --- ARFIMA(0,d,0) -------------------------------------------------------
    display _newline(1)
    display "--- IPCA ARFIMA(0,d,0) ---"
    capture noisily {
        arfima ipca
        matrix b_ipca_frac = e(b)
        scalar d_ipca_only = b_ipca_frac[1, colnumb(b_ipca_frac, "ARFIMA:d")]
        scalar aic_ipca_frac = -2*e(ll) + 2*e(k)
        display "d estimado (apenas fracionario) = " d_ipca_only
        display "AIC = " aic_ipca_frac
        local ipca_frac_ok = 1
    }
    if (_rc != 0) {
        display "AVISO: ARFIMA(0,d,0) nao convergiu para IPCA."
        local ipca_frac_ok = 0
    }
}
else {
    display "Pulando estimacao ARFIMA para IPCA (comando nao disponivel)."
    local ipca_arfima_ok = 0
    local ipca_frac_ok = 0
}

* ==============================================================================
* SECAO 4: Exportar resultados (se disponiveis)
* ==============================================================================

display _newline(2)
display "=== Exportando resultados ARFIMA ==="

if (`arfima_available') {
    preserve
    clear
    set obs 6
    generate str20 dataset = ""
    generate str20 model = ""
    generate str10 param = ""
    generate double value = .
    generate double aic = .
    generate double bic = .

    local row = 0

    * Nile ARFIMA(1,d,1)
    if (`nile_arfima_ok') {
        local ++row
        replace dataset = "nile" in `row'
        replace model = "ARFIMA(1,d,1)" in `row'
        replace param = "d" in `row'
        replace value = d_nile in `row'
        replace aic = aic_nile_arfima in `row'
        replace bic = bic_nile_arfima in `row'
    }

    * Nile ARFIMA(0,d,0)
    if (`nile_frac_ok') {
        local ++row
        replace dataset = "nile" in `row'
        replace model = "ARFIMA(0,d,0)" in `row'
        replace param = "d" in `row'
        replace value = d_nile_only in `row'
        replace aic = aic_nile_frac in `row'
    }

    * IPCA ARFIMA(1,d,1)
    if (`ipca_arfima_ok') {
        local ++row
        replace dataset = "ipca" in `row'
        replace model = "ARFIMA(1,d,1)" in `row'
        replace param = "d" in `row'
        replace value = d_ipca in `row'
        replace aic = aic_ipca_arfima in `row'
        replace bic = bic_ipca_arfima in `row'
    }

    * IPCA ARFIMA(0,d,0)
    if (`ipca_frac_ok') {
        local ++row
        replace dataset = "ipca" in `row'
        replace model = "ARFIMA(0,d,0)" in `row'
        replace param = "d" in `row'
        replace value = d_ipca_only in `row'
        replace aic = aic_ipca_frac in `row'
    }

    drop if dataset == ""
    export delimited using "`output_dir'/arfima_results.csv", replace
    display "Salvo: arfima_results.csv"
    restore
}
else {
    * Exportar arquivo de referencia documentando limitacao
    preserve
    clear
    set obs 1
    generate str80 nota = "ARFIMA nao disponivel - requer Stata 14+"
    generate str80 alternativa = "Usar Python (statsmodels) ou R (fracdiff, arfima)"
    generate str80 metodo_python = "GPH e Local Whittle (semi-parametricos)"
    generate str80 metodo_r = "fracdiff::fracdiff (ML), fdGPH, fdSperio"
    generate str80 metodo_stata = "arfima (ML, Stata 14+)"
    export delimited using "`output_dir'/arfima_results.csv", replace
    display "Salvo: arfima_results.csv (referencia de limitacao)"
    restore
}

* ==============================================================================
* SECAO 5: Documentacao de limitacoes e diferencas metodologicas
* ==============================================================================

display _newline(2)
display "================================================================"
display "  DOCUMENTACAO: ARFIMA no Stata vs Python vs R"
display "================================================================"
display _newline(1)

display "1. DISPONIBILIDADE:"
display "   - Python: statsmodels.tsa.arima.ARIMA com order=(p,d,q)"
display "     onde d pode ser fracionario. Tambem estimadores GPH e"
display "     Local Whittle semi-parametricos."
display "   - R: pacote fracdiff para estimacao ML de d fracionario."
display "     Funcoes fdGPH e fdSperio para estimadores semi-parametricos."
display "   - Stata 14+: comando arfima com estimacao ML."
display "     Versoes anteriores NAO possuem ARFIMA nativo."
display _newline(1)

display "2. METODOS DE ESTIMACAO:"
display "   - Stata arfima: Maximum Likelihood (ML) exata."
display "   - Python GPH: semi-parametrico, baseado no log-periodograma."
display "   - Python Local Whittle: semi-parametrico, no dominio da frequencia."
display "   - R fracdiff: ML aproximada (Haslett & Raftery)."
display "   - R fdGPH: equivalente ao GPH do Python."
display _newline(1)

display "3. DIFERENCAS ESPERADAS:"
display "   - Estimadores ML (Stata, R fracdiff) vs semi-parametricos (GPH,"
display "     Local Whittle) podem diferir significativamente."
display "   - Dentro de ML: diferencas < 0.01 em d sao tipicas."
display "   - GPH vs Local Whittle: diferencas dependem da bandwidth."
display "   - Comparacao direta so e valida entre metodos equivalentes."
display _newline(1)

display "4. RECOMENDACAO PARA VALIDACAO CRUZADA:"
display "   - Comparar Stata arfima (ML) com R fracdiff (ML)."
display "   - Comparar Python GPH com R fdGPH."
display "   - NAO comparar diretamente ML vs semi-parametrico."

display _newline(2)
display "=== Script 04_arfima_validation.do concluido com sucesso ==="
