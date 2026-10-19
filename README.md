# Study of the Starobinsky Inflationary Model Using the Cobaya Code

**Thesis / Tesis de Física — Marvin Francisco Jiménez Peñafiel**  
Yachay Tech — School of Physical Sciences and Nanotechnology (2026).  
Advisor: **Clara Inés Rojas Cely**.

> **Research archive in progress.** This repository preserves genuine exploratory files already uploaded and adds documentation, mathematical tests and clearly labelled **reconstructed examples**. The final SR1/SR2/ΛCDM+r YAML configurations, actual final Slurm wrappers, ML scripts and definitive production chains have **not yet been imported** from HPC. Do not mistake the examples for production configurations.

## Overview

Numerical comparison of the Starobinsky model in two slow-roll implementations (**SR1** and **SR2**) against a phenomenological flat **ΛCDM + r** reference. **Cobaya** orchestrated inference, **CLASS/classy** calculated CMB observables and **PolyChord** performed nested sampling (Bayesian evidence). MCMC served as a posterior consistency check. Likelihoods: Planck 2018 and BICEP/Keck 2018. Complementary ML analyzed posterior classification, parameter importance, PCA/UMAP, and local neural emulators of χ².

### Results reported in the final thesis (not produced by this repository's examples)

| Model | ln Z | Estimated uncertainty |
| --- | ---: | ---: |
| ΛCDM + r | -797.3157 | 0.2742 |
| Starobinsky SR1 | -795.3275 | 0.2655 |
| Starobinsky SR2 | -794.4122 | 0.2636 |

The final comparison used **PolyChord `nlive=300`** for all three models. The difference SR2–SR1 was **0.9154 ± 0.3741** (weak preference, **not decisive**). The adopted MCMC diagnostic was **R−1 < 0.01**. These values were copied from tables 5.2–5.3 of the thesis, not recomputed here.

## Where to start

| Material | Location | Status |
| --- | --- | --- |
| Original exploratory Cobaya theory | [`effective.py`](effective.py) | Original file already in repository; **SR2 formulas differ from the final thesis's equations** |
| Original early PolyChord YAML | [`polychord_starobinsky_Nfree.yaml`](polychord_starobinsky_Nfree.yaml) | **Exploratory**: nlive=100; no BICEP/Keck or low-ℓ EE |
| Original HPC setup notebook | [`INSTALL_cluster_cobaya.ipynb`](INSTALL_cluster_cobaya.ipynb) | Saved installation notes (commands must be checked against the cluster) |
| Original exploratory raw outputs | [`chains/`](chains) | Historical output, not definitive inference |
| Installation and troubleshooting | [`docs/instalacion-cluster.md`](docs/instalacion-cluster.md) | Annotated workflow |
| Connecting, SLURM, monitoring, resume | [`docs/comandos-cluster.md`](docs/comandos-cluster.md) | Historical instructions and safe examples |
| Final cosmological methodology | [`docs/metodologia.md`](docs/metodologia.md) | Transcribed from thesis |
| Audit of original code vs thesis | [`docs/auditoria-codigo.md`](docs/auditoria-codigo.md) | **Read before running** |
| Example YAMLs | [`configs_reconstruidas/`](configs_reconstruidas) | Illustrative, **not reproduction-ready** |
| Independent equations / tests | [`modelos/`](modelos), [`tests/`](tests) | No Cobaya dependencies |
| HPC import script | [`scripts/importar_originales_hpc.sh`](scripts/importar_originales_hpc.sh) | Copy originals on the HPC, then review before versioning |
| Numerical results summary | [`resultados/README.md`](resultados/README.md) | Thesis values only |
| ML source inventory | [`machine_learning/README.md`](machine_learning/README.md) | Files to recover |

## Quick local checks

```bash
python -m unittest discover -s tests -v
python scripts/comprobar_reconstruccion.py  # optional PyYAML
```

These checks test the **transcribed mathematics and YAML syntax only**. They do **not** initialize CLASS, validate the likelihoods or reproduce the scientific results.

## Complete the archive with the original HPC source

After cloning this repository **on your authorized HPC account**:

```bash
bash scripts/importar_originales_hpc.sh
# Optional: import selected output summaries to a Git-ignored local folder
bash scripts/importar_originales_hpc.sh --resultados
```

The script copies local source code into **`originales_hpc/`** (Git-ignored pending inspection). Review its output, strip credentials and non-shareable data, compare versioned files with the final methods, and selectively stage suitable originals with `git add -f`. It never accesses a different user's home directory.

## Citation and reproducibility notes

Jiménez Peñafiel, M. F. (2026). *Study of the Starobinsky Inflationary Model Using the Cobaya Code*. Thesis, Yachay Tech. Cite the relevant Cobaya, CLASS, PolyChord and Planck/BICEP–Keck publications as well. Data products may require third-party licenses; do not mirror proprietary likelihood datasets.

**Repository currently private.** Do not commit cluster credentials, SSH keys, huge sampling chains, or environment directories.
