#!/usr/bin/env bash
# Coloque todos los ZIP ORIGINALES en la carpeta indicada. Revise primero la documentación.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SRC="${1:-}"
MODE="${2:---prepare}"
if [[ -z "$SRC" || ! -d "$SRC" ]]; then
  echo 'Uso: bash scripts/publicar_archivos_git_lfs.sh /ruta/a/mis/ZIPs [--prepare|--push]' >&2
  exit 2
fi
if [[ "$MODE" != '--prepare' && "$MODE" != '--push' ]]; then
  echo 'Modo inválido; usar --prepare o --push' >&2; exit 2
fi
for exe in git python3; do command -v "$exe" >/dev/null || { echo "Falta $exe" >&2; exit 1; }; done
if ! git lfs version >/dev/null 2>&1; then
  echo 'Falta Git LFS. Instálalo primero: https://git-lfs.com/' >&2; exit 1
fi
if [[ ! -d .git ]]; then
  echo 'Debes ejecutar desde un clon local real del repositorio (no un ZIP descargado).' >&2; exit 1
fi
remote="$(git remote get-url origin 2>/dev/null || true)"
case "$remote" in
  *github.com/marvinfranc2001-fis/InflationaryCosmology* ) ;;
  *) echo "Repo origin inesperado: $remote" >&2; exit 1 ;;
esac
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'El árbol de trabajo contiene cambios. Guarda tus cambios antes de continuar.' >&2; exit 1
fi
SRC="$(cd "$SRC" && pwd)"
# Importante: verificar ANTES de copiar, añadir, commitear o publicar.
python3 scripts/verificar_archivos.py "$SRC"

mkdir -p data/archives
git lfs install --local
git lfs track 'data/archives/*.zip'
for file in Yamls.zip chains_poly.zip chains_mcmc.zip chainslcdm.zip Graf_ml_definitivo.zip; do
  cp -p "$SRC/$file" "data/archives/$file"
done
python3 scripts/verificar_archivos.py "$ROOT"
git add .gitattributes data/archives/
git lfs ls-files

echo
echo 'Archivos preparados, con hashes SHA-256 comprobados y añadidos al índice mediante LFS.'
echo 'Revisa: git status --short && git diff --cached --stat'
if [[ "$MODE" == '--prepare' ]]; then
  echo 'No se realizó ningún commit ni push. Para publicar, ejecuta:'
  echo 'git commit -m "Archive complete thesis data via Git LFS" && git push origin main'
  exit 0
fi

if [[ "$(git branch --show-current)" != main ]]; then
  echo 'Para publicación automática usa una copia en la rama main; los archivos han quedado preparados.' >&2
  exit 1
fi
printf '¿Publicar estos ZIP mediante Git LFS en origin/main? Escribe PUBLICAR: '
read -r response
if [[ "$response" != 'PUBLICAR' ]]; then
  echo 'No se hizo commit/push; puedes revisar o descartar los cambios.'
  exit 0
fi
git commit -m 'Archive complete thesis data via Git LFS'
git push origin main
printf 'Publicación terminada. Comprueba en GitHub la presencia de los cinco punteros LFS.\n'
