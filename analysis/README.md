# GetDist analysis and publication figures

The untouched original `GraficasTesisFinal.py` produces weighted posterior summaries, triangle plots, MCMC convergence and consistency checks, model evidence and Bayes factors, plus CSV and LaTeX tables.

```bash
pip install numpy pandas matplotlib getdist pyyaml
python analysis/GraficasTesisFinal.py --poly-zip chains_poly.zip --mcmc-zip chains_mcmc.zip --lcdm-zip chainslcdm.zip --out publication_analysis
```

Supply the original chain ZIPs alongside the repository. The complete script has not been executed in this publication step.
