# ISCAS 2027 original files — expected import inventory

This is a **filename-level inventory** provided by the author from the local Linux directory `~/Downloads/ISCAS2027/`. The actual manuscript, scripts and archive contents have not been accessed from this chat.

## Named sources

**Manuscripts (3 PDFs):** `ISCAS_2027_Manuscrito_Borrador.pdf`, `ISCAS_2027_Manuscrito_Redaccion_Revisada.pdf`, `JiménezRojas.pdf`.

**LaTeX (2 entries):** `ISCAS_2027_Manuscrito_Redaccion_Revisada_LaTeX/` (directory), `ISCAS_2027_Proyecto_LaTeX.zip`.

**Experimental bundles / notes (3 entries):** `ISCAS_2027_Siguiente_Experimento_SR1.zip`, `ISCAS_2027_materials.tar.gz`, `LEEME_Siguiente_Experimento_ISCAS.txt`.

**Python scripts (11):**
- `iscas_class_same_node_benchmark.py`
- `iscas_ensemble_benchmark.py`
- `iscas_frozen_gate_holdout.py`
- `iscas_make_paper_results.py`
- `iscas_ood_gate_audit.py`
- `iscas_posterior_impact.py`
- `iscas_prospective_candidates.py`
- `iscas_prospective_exact.py`
- `iscas_prospective_triage.py`
- `iscas_selective_routing_analysis.py`
- `iscas_sr1_tail_sensitivity.py`

**Total:** 18 individual files plus one LaTeX directory of unknown size and file count.

## Publication method

1. Use [`scripts/importar_iscas2027.sh`](../../scripts/importar_iscas2027.sh) from the root of your thesis Git clone.
2. Confirm the 18 named files and LaTeX folder all exist, and inspect them for sensitive and copyrighted material.
3. Let the script copy them into `projects/ISCAS2027/` and create `provenance/FILES_SHA256.txt`.
4. Review staged file names, LFS pointers and ZIP sizes. The script does **not** commit/push.
5. Commit and push from the user's authenticated Linux Git terminal.
6. Verify the resulting GitHub files and their SHA-256 manifest; retrieve heavy objects with `git lfs pull`.

No claims about the science, manuscript acceptance or benchmarks will be inferred from the filenames. The ISCAS2027 work is not part of the finalized 2026 evidence comparison.
