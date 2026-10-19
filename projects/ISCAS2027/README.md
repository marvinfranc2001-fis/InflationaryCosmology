# ISCAS 2027 — Research extension
### Working manuscripts, benchmarks and prospective experiments

> **In progress · Private research material.** This project is an extension to the [Starobinsky inflation thesis](../../README.md). It is intentionally separated from the final 2026 PolyChord and MCMC results. Filenames alone do not establish manuscript content, scientific conclusions, conference submission or acceptance.

## Project organization

| Section | What it will contain | Original files |
|---|---|---|
| **Manuscripts** | Original versions, kept separately | `ISCAS_2027_Manuscrito_Borrador.pdf`, `ISCAS_2027_Manuscrito_Redaccion_Revisada.pdf`, `JiménezRojas.pdf` |
| **LaTeX source** | Editable revised manuscript plus the full original source ZIP | `ISCAS_2027_Manuscrito_Redaccion_Revisada_LaTeX/`, `ISCAS_2027_Proyecto_LaTeX.zip` |
| **Benchmarks / experiments** | Eleven Python scripts, copied without changes | `experiments/scripts/iscas_*.py` |
| **SR1 experiment bundle** | Packaged follow-up experiment | `ISCAS_2027_Siguiente_Experimento_SR1.zip` |
| **Supporting materials** | Original research materials bundle | `ISCAS_2027_materials.tar.gz` |
| **Lab notes** | Follow-up experimental instructions | `LEEME_Siguiente_Experimento_ISCAS.txt` |
| **Integrity / provenance** | SHA-256 hashes recorded at the point of copying | `provenance/FILES_SHA256.txt` |

### Experiment script index

The filenames suggest these topic areas; **they are not descriptions verified against the code yet**.

| Benchmark & diagnostic | Candidate evaluation & impact |
|---|---|
| `iscas_class_same_node_benchmark.py` | `iscas_prospective_candidates.py` |
| `iscas_ensemble_benchmark.py` | `iscas_prospective_exact.py` |
| `iscas_frozen_gate_holdout.py` | `iscas_prospective_triage.py` |
| `iscas_ood_gate_audit.py` | `iscas_posterior_impact.py` |
| `iscas_selective_routing_analysis.py` | `iscas_sr1_tail_sensitivity.py` |
| `iscas_make_paper_results.py` | |

## Upload from your Linux workstation

Your original source folder is:

`~/Downloads/ISCAS2027/`

Inside your already cloned repository, run:

```bash
cd ~/Marv/Investigacion/ThesisInflation/InflationaryCosmology
git pull --ff-only
bash scripts/importar_iscas2027.sh "$HOME/Downloads/ISCAS2027"
git diff --cached --stat
git lfs ls-files
```

The importer requires confirmation by typing **PREPARAR** and stops after staging. Review the file list before running:

```bash
git commit -m "Add ISCAS2027 extension sources via Git LFS"
git push origin main
```

PDF, ZIP and tar.gz files in this project are tracked with **Git LFS**. Plain-text scripts, notes and LaTeX sources remain in ordinary Git. The importer does not modify the files in `~/Downloads/ISCAS2027/` and does not alter the definitive thesis results.

### Checks and scientific caution

- Keep `ISCAS_2027_Manuscrito_Borrador.pdf` distinct from `ISCAS_2027_Manuscrito_Redaccion_Revisada.pdf`; do not silently replace either version.
- Examine the contents of the archives, any third-party papers/materials and the draft manuscript's confidentiality before sharing the repository publicly. Its current GitHub visibility is private.
- This section should later include verified setup, scripts' command-line interfaces, hardware/runtime environment, experiment inputs and outputs.
- See [the file inventory](SOURCE_INVENTORY.md). **Original content has not yet been transferred to GitHub** as of the preparation of this folder; the author's local import and push are required.
