# Original thesis archives · Git LFS

The five unmodified original ZIP archives were uploaded successfully in commit [`957863f`](https://github.com/marvinfranc2001-fis/InflationaryCosmology/commit/957863f81a746f397dd3a6ce081a26dcc3538189).

| ZIP | Bytes | SHA-256 |
|---|---:|---|
| `Yamls.zip` | 4,672 | `c98a213317665665703d74fb799e6ddc1ebad8071050e5fb0fc37b5a466b8789` |
| `chains_poly.zip` | 387,480,823 | `97c91bc82f9c527378a6a67271072d7654870f46168acc573d98c6593bdc8040` |
| `chains_mcmc.zip` | 143,684,644 | `9fcd6b7c33f412e7c599671c3aaec40c27ac86333b3d1bbf821149e6022d8597` |
| `chainslcdm.zip` | 63,548,377 | `d137a24ccb0d05d86668376eb32af44c171ed8b9d09baf542607f0f13c2c3a37` |
| `Graf_ml_definitivo.zip` | 214,998,882 | `759a48c4dd11d4eaf898c56c286f09e27a1f3aed64006d2248804daed2113d21` |

GitHub's Git tree contains lightweight **Git LFS pointers** rather than the large binary ZIPs. All five pointers have been checked against the source SHA-256 manifest, and the terminal reported that Git LFS uploaded all five objects. A fresh independent `git lfs pull` download is recommended as a further accessibility test.

```bash
git lfs install
git lfs pull
python3 scripts/verificar_archivos.py .
```

The repository remains private. Do not distribute proprietary observation likelihood files, HPC secrets, or unlicensed data if changing visibility.
