# Results | Final inference

Numerical evidence values were **extracted from user-provided original `.logZ` outputs**:

| Model | ln Z | uncertainty |
|---|---:|---:|
| ΛCDM + r | −797.3157492846282 | 0.274179560762351 |
| SR1 | −795.3275075271462 | 0.265472584645632 |
| SR2 | −794.4121528421132 | 0.263563683747322 |

The `.stats` nested-sampling evidence estimates differ slightly from the `.logZ` output estimates; do **not** mix estimators when forming Bayes factors. The thesis's main table uses the `.logZ` values.

MCMC checkpoints:

- SR1: `converged: true`, `Rminus1_last: 0.008343404399456297`, `mpi_size: 32`.
- SR2: `converged: true`, `Rminus1_last: 0.008235957187523893`, `mpi_size: 32`.

They are historical convergence indicators, not independent convergence verification. The MCMC outputs are for posterior consistency rather than Bayesian evidence. See `data/summaries` for individual metadata and `data/manifests` for full source indexes.
