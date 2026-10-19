#!/usr/bin/env bash
# Ejecutar SOLO desde la raíz de la copia local Git en la cuenta HPC autorizada.
set -euo pipefail
SRC="${TESIS_HPC_SOURCE:-$HOME/Yaml}"
if [[ ! -d "$SRC" || ! -f README.md ]]; then
  echo "Ubícate en el repositorio y comprueba que exista $SRC" >&2; exit 1
fi
DST="$PWD/originales_hpc"
mkdir -p "$DST"
COUNT=0
while IFS= read -r -d '' f; do
  rel="${f#"$SRC"/}"
  mkdir -p "$DST/$(dirname "$rel")"
  cp -p -- "$f" "$DST/$rel"
  COUNT=$((COUNT+1))
done < <(find "$SRC" -type f \( -name '*.yaml' -o -name '*.yml' -o -name '*.slurm' -o -name '*.py' -o -name '*.sh' -o -name '*.ipynb' -o -name '*.toml' -o -name '*.cfg' \) ! -path '*/.git/*' ! -path '*/__pycache__/*' -print0)
find "$DST" -type f -print0 | sort -z | xargs -0 -r sha256sum > MANIFEST_ORIGINALES_SHA256.txt
printf 'Archivos originales copiados localmente: %s\n' "$COUNT"
printf '%s\n' "No se han añadido a Git. Revisar secretos/licencias antes de versionar."
if [[ "${1:-}" == "--resultados" ]]; then
  mkdir -p resultados_locales
  while IFS= read -r -d '' f; do
    rel="${f#"$SRC"/}"
    mkdir -p "resultados_locales/$(dirname "$rel")"
    cp -p -- "$f" "resultados_locales/$rel"
  done < <(find "$SRC" -type f \( -name '*.stats' -o -name '*.paramnames' -o -name '*_equal_weights.txt' \) -print0)
fi
