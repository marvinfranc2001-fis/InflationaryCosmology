#!/usr/bin/env bash
set -euo pipefail
mkdir -p versiones_entorno
date -Is > versiones_entorno/fecha.txt
python --version > versiones_entorno/python.txt 2>&1 || true
python -m pip freeze > versiones_entorno/pip-freeze.txt 2>&1 || true
conda env export --no-builds > versiones_entorno/conda-env.yml 2>&1 || true
for c in mpirun mpicc enroot; do command -v "$c" >> versiones_entorno/rutas.txt || true; done
echo "Revisar versiones_entorno/ antes de publicar."
