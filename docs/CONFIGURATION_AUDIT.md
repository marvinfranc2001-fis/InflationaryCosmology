# Configuration audit — definitive versus exploratory

This distinction is mandatory for interpreting the published results. The exact `.input.yaml` source snapshots included in the final chain ZIPs have been extracted without modifying their content.

| Model | Family | Definitive source | Key method |
|---|---|---|---|
| SR1 | PolyChord | `configs/production/polychord/SR1.input.yaml` | nlive=300, 4 CMB likelihoods |
| SR2 | PolyChord | `configs/production/polychord/SR2.input.yaml` | nlive=300, 4 CMB likelihoods |
| ΛCDM + r | PolyChord | `configs/production/polychord/LCDM_r.input.yaml` | nlive=300, 4 CMB likelihoods |
| SR1 | MCMC | `configs/production/mcmc/SR1.input.yaml` | native Plik-lite; R−1 stop=0.01 |
| SR2 | MCMC | `configs/production/mcmc/SR2.input.yaml` | native Plik-lite; R−1 stop=0.01 |

## Theory model verification

Final YAMLs implement the primordial observables as deterministic functions of `N_star`:

- SR1: `n_s = 1-2/N-4.5/N²`; `r = 12/N²`; `alpha_s = 0`.
- SR2: `n_s = 1-2/N-1.5/N²`; `r = (12/N²)(1-2/N)`; `alpha_s = -2/N²-3/N³`.

The original top-level `effective.py` contains a different, earlier second-order approximation. It does **not** define the final model if Cobaya uses only the explicit YAML `lambda N_star` relations. Preserve it for provenance; do not silently replace it.

## Raw archival notes from Yamls.zip

`configs/notes_originales/` preserves six user-supplied `.txt` notes, including incidental terminal prompts. `configs/transcripciones/` holds the same YAML configuration text after removing prompt lines. **All final scientific comparison values should be associated with the exact `.input.yaml` files embedded in the chain ZIPs.**

`mcmcLCDM.txt` is **exploratory**: different priors on `n_s` (0.90–1.05) and `r` (0–0.2), use of `planck_2018_lensing.native`, and a different output path. Never mix it into the final homogeneous PolyChord evidence comparison, which **did not** include an independent Planck lensing reconstruction likelihood.

## Portable execution

Some original MCMC YAML files refer to absolute HPC paths rooted in another user home. These paths document provenance, not recommended runtime settings. Copy the YAML to your own workspace and modify paths only in a **separate named portability patch**; do not overwrite the original.

For PolyChord versus MCMC, `TTTEEE_lite` and `TTTEEE_lite_native` share physical Plik-lite data but may use different absolute likelihood normalizations; compare posterior locations without naively subtracting absolute raw χ² across implementations.
