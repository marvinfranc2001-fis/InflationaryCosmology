# Definitive machine-learning analysis report

## Methodological rules

- Physical model classification uses only the SR1, SR2, and LambdaCDM+r PolyChord posterior samples.
- Model-classification accuracy quantifies posterior separability; it is not Bayesian evidence and is not used for model selection.
- MCMC-versus-PolyChord classification is performed separately inside SR1 and SR2. Balanced accuracy near 0.5 means the sampler cannot be inferred reliably from the posterior parameters.
- The primary model classifier excludes alpha_s because alpha_s=0 in SR1 and is model-derived in SR2; the classifier including running is explicitly treated as a model-definition diagnostic.
- Chi-square emulators are trained independently for each PolyChord model and exact likelihood implementation. Absolute chi-square values from native and clik implementations are never combined.
- Neural emulators are local approximations inside the posterior-supported parameter region and must not be extrapolated globally.
- Final cosmological model preference remains based on the PolyChord Bayesian evidence, not on any ML classifier or embedding.

## Classification summary

- **models_PolyChord_without_running**: repeated-CV balanced accuracy 0.8479 ± 0.0131; chance level 0.3333; random-label p-value 0.009901.
- **models_PolyChord_with_model_imposed_running**: repeated-CV balanced accuracy 0.9922 ± 0.0022; chance level 0.3333; random-label p-value 0.009901.
- **sampler_consistency_SR1**: repeated-CV balanced accuracy 0.5029 ± 0.0035; chance level 0.5000; random-label p-value 0.1188.
- **sampler_consistency_SR2**: repeated-CV balanced accuracy 0.5052 ± 0.0022; chance level 0.5000; random-label p-value 0.0297.

## Ablation summary

- **primordial_with_running**: balanced accuracy 0.9965 ± 0.0012.
- **primordial_without_running**: balanced accuracy 0.9948 ± 0.0017.
- **all_physical_with_running**: balanced accuracy 0.9922 ± 0.0022.
- **physical_plus_nuisance**: balanced accuracy 0.9920 ± 0.0021.
- **physical_without_running**: balanced accuracy 0.8479 ± 0.0131.
- **late_cosmology**: balanced accuracy 0.3862 ± 0.0101.
- **nuisance_only**: balanced accuracy 0.3594 ± 0.0144.

## Internal emulator summary

- **SR1_PolyChord**: total chi-square MAE=0.9443, RMSE=1.3741, R2=0.9021.
- **SR2_PolyChord**: total chi-square MAE=1.0464, RMSE=1.4777, R2=0.8761.
- **LCDM_PolyChord**: total chi-square MAE=1.6672, RMSE=2.1718, R2=0.7767.

## Thesis-ready figure hierarchy

1. Primary ML result: model-classification confusion matrix without running.
2. Sampler consistency: SR1 and SR2 confusion matrices, preferably as a two-panel figure.
3. Physics interpretation: repeated-CV ablation study and global SHAP/permutation importance.
4. Exploratory geometry: PCA and UMAP; state clearly that coordinates have no direct physical units.
5. Emulator validation: external predicted-versus-Cobaya chi-square and the consolidated MAE/RMSE/R2/coverage comparison.
