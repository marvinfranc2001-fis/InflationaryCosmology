# Metodología cosmológica documentada

Fuente: manuscrito de tesis *Study of the Starobinsky Inflationary Model Using the Cobaya Code*, capítulo 4.

## Modelos

Se analizaron **ΛCDM + r** (referencia fenomenológica, con `n_s` y `r_0.05` muestreados independientemente), **SR1** y **SR2**, con `N_star` muestreado y las observables inflacionarias derivadas del mismo. Pivote `k*=0.05 Mpc^-1`.

**SR1**: `n_s = 1 - 2/N_star - 9/(2*N_star**2)`; `r_0.05 = 12/N_star**2`; `alpha_s = 0`.

**SR2**: `n_s = 1 - 2/N_star - 3/(2*N_star**2)`; `r_0.05 = 12/N_star**2 * (1 - 2/N_star)`; `alpha_s = -2/N_star**2 - 3/N_star**3`.

Espectro escalar: `ln P_R(k) = ln A_s + (n_s-1) ln(k/k*) + 0.5 alpha_s [ln(k/k*)]^2`.

## Priors de los parámetros físicos

| Parámetro | Uniforme |
| --- | --- |
| `N_star` (sólo SR1/SR2) | [50, 60] |
| `A_s` | [1.8e-9, 3.0e-9] |
| `H0` (km/s/Mpc) | [60, 75] |
| `omega_b` | [0.020, 0.024] |
| `omega_cdm` | [0.10, 0.14] |
| `tau_reio` | [0.04, 0.08] |
| `n_s` (sólo ΛCDM+r) | [0.94, 0.99] |
| `r_0.05` (sólo ΛCDM+r) | [0, 0.08] |

Los nuisance de Planck y BK18 se marginalizaron y sus priors permanecieron iguales entre modelos. **Deben leerse de los YAML definitivos; no reconstruirlos arbitrariamente.**

## Likelihoods

La comparación final PolyChord `nlive=300` utilizó:
- `planck_2018_lowl.TT`
- `planck_2018_lowl.EE`
- `planck_2018_highl_plik.TTTEEE_lite`
- `bicep_keck_2018`

La rama MCMC de contraste usó `planck_2018_highl_plik.TTTEEE_lite_native` (diferente de la variante PolyChord). No sumar/comparar directamente absolutos χ² de ambas configuraciones. El likelihood de reconstrucción de lente gravitacional de Planck no fue incluido independientemente en la comparación principal.

## Muestreo, cómputo y validación

El flujo fue `parámetros → teoría primordial → CLASS → espectros CMB → likelihoods → MCMC / PolyChord`. MCMC: Metropolis–Hastings adaptativo, criterio `R−1 < 0.01` y exclusión del 30% inicial en comparaciones posteriores. PolyChord: estimaciones `ln Z ± sigma_lnZ`, puntos vivos 300 en las tres ejecuciones definitivas (se excluyó la corrida exploratoria de referencia con 1000). Cálculo en infraestructura HPC de CEDIA, SLURM, MPI, partición CPU, un nodo; `OMP_NUM_THREADS=1`.

Análisis ML complementario: clasificación posterior, ablación, SHAP, PCA, UMAP, ensambles de emuladores locales χ² y validación externa con Cobaya y CLASS. Ni los clasificadores ni los emuladores sustituyeron la evidencia bayesiana directa.
