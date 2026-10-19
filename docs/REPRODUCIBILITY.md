# Reproducibility and data management

## Included

- The five exact production `.input.yaml` snapshots from definitive outputs, together with `.updated.yaml` metadata, as read-only provenance.
- Original user scripts `analysis/GraficasTesisFinal.py` and `machine_learning/validar_emulador_cobaya.py`.
- PolyChord `.logZ` and `.stats` and MCMC `.checkpoint`/`.progress` outputs, plus checksums and indexes of all uploaded archives.
- Small original ML summaries. Full ML models and figures are preserved in `Graf_ml_definitivo.zip`.

## Large source archives (keep originals)

- `chains_poly.zip` (~370 MiB): SR1 and SR2 nested sampling, original weights, full sampler checkpoint files.
- `chains_mcmc.zip` (~137 MiB): 32-chain SR1/SR2 MCMC results.
- `chainslcdm.zip` (~61 MiB): ΛCDM+r PolyChord.
- `Graf_ml_definitivo.zip` (~205 MiB): figures, trained models, emulators and validation datasets.
- `Yamls.zip`: six original configuration notes.

These archives are now **tracked in Git LFS under `data/archives/`**, in commit [`957863f`](https://github.com/marvinfranc2001-fis/InflationaryCosmology/commit/957863f81a746f397dd3a6ce081a26dcc3538189), rather than ordinary Git. The recorded SHA-256 digests in `data/manifests/archives_sha256.json` agree with all five Git LFS pointers. Their original data can be downloaded after cloning via `git lfs pull`. Individual `.resume` and ML model files remain inside these ZIPs instead of being separately tracked.

### Verify SHA256

```bash
git lfs pull
python3 scripts/verificar_archivos.py .
cat data/manifests/archives_sha256.json
```

### Run GetDist postprocessing

```bash
python -m pip install numpy pandas matplotlib getdist pyyaml
python analysis/GraficasTesisFinal.py --poly-zip data/archives/chains_poly.zip --mcmc-zip data/archives/chains_mcmc.zip --lcdm-zip data/archives/chainslcdm.zip --out publication_analysis
```

### External ML validation

The original `machine_learning/validar_emulador_cobaya.py` imports `tesis_chain_utils.py`, which the user did not upload here. It also requires trained candidate CSVs and Cobaya, CLASS and the appropriate licensed likelihoods. Thus the original source is archived but not yet standalone runnable.

### Versioned binary files (completed)

All five original ZIPs were transferred successfully through Git LFS. The `.gitattributes` rule applies to `data/archives/*.zip`; after cloning, use `git lfs pull` and `git lfs ls-files`. Do not re-add unpacked restart files into ordinary Git.

Do not commit SSH keys, passwords, licensed Planck/BK data or HPC account tokens. The published results in this repo were read from supplied outputs, **not recalculated with fresh nested sampling**.
