# Machine learning | posterior analysis

`Graf_ml_definitivo.zip` includes the definitive exported outputs across seven groups:

| Directory | Methods |
|---|---|
| 01_classification | Three-model posterior classification and MCMC/PolyChord consistency |
| 02_ablation | Parameter ablation and relevance |
| 03_explainability | Feature importance and interpretability |
| 04_embeddings | PCA and UMAP |
| 05_chi2_emulators | Local neural χ² ensembles, candidates |
| 06_external_validation | Direct Cobaya/CLASS candidate checks |
| 07_validation_comparison | Metrics across three model variants |

Selected light summaries are under `outputs/`. The complete archive has an indexed file list in `data/manifests/Graf_ml_definitivo.csv` but its large binary classifiers/checkpoints remain in the original ZIP.

Original Python source: [`validar_emulador_cobaya.py`](validar_emulador_cobaya.py) (external checks). This script imports **`tesis_chain_utils.py`**, which has not been uploaded. Also missing from the attachments is the ML pipeline source that generated all the archived plots; these should be recovered separately without inventing their contents.

**Classification accuracy is not Bayesian evidence.** Local emulators are supplementary computational diagnostics and must be validated against the same likelihood implementation that generated their training targets.
