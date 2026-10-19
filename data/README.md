# Input archives and provenance

`manifests/` contains SHA-256 hashes for every uploaded archive and per-entry lists of path/size/CRC32. `summaries/` contains original small run metadata (`.logZ`, `.stats`, `.checkpoint`, `.progress`, etc.)

Raw chains, PolyChord restart snapshots, training datasets and large ML model checkpoints remain in the original ZIPs, not ordinary Git. Use LFS or a research data repository for authorized hosting. Never upload secrets or separately licensed Planck/BICEP-Keck likelihoods.
