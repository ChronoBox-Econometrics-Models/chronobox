// MathJax configuration for ChronoBox documentation

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams',
    tagSide: 'right',
    tagIndent: '0.8em',
    multlineWidth: '85%',
    // Common time series econometrics notation
    macros: {
      E: "\\mathbb{E}",
      Var: "\\mathrm{Var}",
      Cov: "\\mathrm{Cov}",
      Corr: "\\mathrm{Corr}",
      plim: "\\operatorname{plim}",
      eps: "\\varepsilon",
      argmin: "\\operatorname{argmin}",
      argmax: "\\operatorname{argmax}",
      // Time series operators
      Lag: "\\mathcal{L}",
      diff: "\\Delta"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  loader: {
    load: ['[tex]/ams']
  },
  svg: {
    fontCache: 'global',
    displayAlign: 'left',
    displayIndent: '2em'
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
