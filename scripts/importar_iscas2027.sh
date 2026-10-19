#!/usr/bin/env bash
# Copia el inventario ISCAS2027 al repositorio: no hace commit ni push.
# Uso: bash scripts/importar_iscas2027.sh "$HOME/Downloads/ISCAS2027"
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:-$HOME/Downloads/ISCAS2027}"
PROJECT="projects/ISCAS2027"
cd "$ROOT"

[[ -d .git && "$(git branch --show-current)" == main ]] || {
  echo "ERROR: Se requiere un clon Git en la rama main" >&2; exit 1;
}
case "$(git remote get-url origin)" in
  *github.com/marvinfranc2001-fis/InflationaryCosmology*) ;;
  *) echo "ERROR: Este no es el repositorio de la tesis" >&2; exit 1 ;;
esac
git diff --quiet && git diff --cached --quiet || {
  echo "ERROR: Hay otros cambios versionados o staged pendientes" >&2; exit 1;
}
git lfs version >/dev/null 2>&1 || {
  echo "ERROR: Instala git-lfs antes de continuar" >&2; exit 1;
}
[[ -d "$SOURCE" ]] || { echo "ERROR: Carpeta inexistente: $SOURCE" >&2; exit 1; }
SOURCE="$(cd "$SOURCE" && pwd -P)"
[[ "$SOURCE" != "$ROOT" && "$SOURCE" != "$ROOT/"* ]] || {
  echo "ERROR: El origen debe estar fuera del repositorio" >&2; exit 1;
}

declare -a MAP=(
  "ISCAS_2027_Manuscrito_Borrador.pdf|manuscripts/ISCAS_2027_Manuscrito_Borrador.pdf"
  "ISCAS_2027_Manuscrito_Redaccion_Revisada.pdf|manuscripts/ISCAS_2027_Manuscrito_Redaccion_Revisada.pdf"
  "JiménezRojas.pdf|manuscripts/JiménezRojas.pdf"
  "ISCAS_2027_Proyecto_LaTeX.zip|latex/ISCAS_2027_Proyecto_LaTeX.zip"
  "ISCAS_2027_Siguiente_Experimento_SR1.zip|experiments/bundles/ISCAS_2027_Siguiente_Experimento_SR1.zip"
  "ISCAS_2027_materials.tar.gz|materials/ISCAS_2027_materials.tar.gz"
  "LEEME_Siguiente_Experimento_ISCAS.txt|notes/LEEME_Siguiente_Experimento_ISCAS.txt"
  "iscas_class_same_node_benchmark.py|experiments/scripts/iscas_class_same_node_benchmark.py"
  "iscas_ensemble_benchmark.py|experiments/scripts/iscas_ensemble_benchmark.py"
  "iscas_frozen_gate_holdout.py|experiments/scripts/iscas_frozen_gate_holdout.py"
  "iscas_make_paper_results.py|experiments/scripts/iscas_make_paper_results.py"
  "iscas_ood_gate_audit.py|experiments/scripts/iscas_ood_gate_audit.py"
  "iscas_posterior_impact.py|experiments/scripts/iscas_posterior_impact.py"
  "iscas_prospective_candidates.py|experiments/scripts/iscas_prospective_candidates.py"
  "iscas_prospective_exact.py|experiments/scripts/iscas_prospective_exact.py"
  "iscas_prospective_triage.py|experiments/scripts/iscas_prospective_triage.py"
  "iscas_selective_routing_analysis.py|experiments/scripts/iscas_selective_routing_analysis.py"
  "iscas_sr1_tail_sensitivity.py|experiments/scripts/iscas_sr1_tail_sensitivity.py"
)
LATEX="ISCAS_2027_Manuscrito_Redaccion_Revisada_LaTeX"

for item in "${MAP[@]}"; do
  source_file="${item%%|*}"
  [[ -f "$SOURCE/$source_file" && ! -L "$SOURCE/$source_file" ]] || {
    echo "ERROR: Falta archivo o es enlace simbólico: $source_file" >&2; exit 1;
  }
done
[[ -d "$SOURCE/$LATEX" && ! -L "$SOURCE/$LATEX" ]] || {
  echo "ERROR: Falta directorio LaTeX $LATEX" >&2; exit 1;
}
if find "$SOURCE/$LATEX" -type l -print -quit | grep -q .; then
  echo "ERROR: Hay enlaces simbólicos en la carpeta LaTeX; revisarlos primero" >&2
  exit 1
fi
if [[ -e "$PROJECT/latex/$LATEX" ]]; then
  echo "ERROR: Ya existe el destino LaTeX; no sobrescribiré archivos" >&2
  exit 1
fi
echo "Origen: $SOURCE"
echo "Revisa confidencialidad del manuscrito, licencias y archivos sensibles antes de publicar."
read -r -p "Para COPIAR y preparar (sin publicar), escribe PREPARAR: " answer
[[ "$answer" == "PREPARAR" ]] || { echo "Cancelado"; exit 0; }

for item in "${MAP[@]}"; do
  source_file="${item%%|*}"
  dest_rel="${item#*|}"
  dest="$PROJECT/$dest_rel"
  mkdir -p -- "$(dirname "$dest")"
  if [[ -e "$dest" ]]; then
    cmp -s -- "$SOURCE/$source_file" "$dest" || {
      echo "ERROR: Archivo existente diferente, no sobrescrito: $dest" >&2; exit 1;
    }
  else
    cp -p -- "$SOURCE/$source_file" "$dest"
  fi
  echo "OK: $dest"
done
mkdir -p "$PROJECT/latex"
cp -a -- "$SOURCE/$LATEX" "$PROJECT/latex/$LATEX"
git lfs install --local

mkdir -p "$PROJECT/provenance"
(
 for item in "${MAP[@]}"; do
   sha256sum -- "$PROJECT/${item#*|}"
 done
 find "$PROJECT/latex/$LATEX" -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
) > "$PROJECT/provenance/FILES_SHA256.txt"

git add -- "$PROJECT"
echo ""
echo "Archivos preparados, pero NO publicados:"
git status --short
echo "LFS:"
git lfs ls-files
echo ""
echo "Antes de publicar revisa: git diff --cached --stat"
echo 'Después: git commit -m "Add ISCAS2027 extension sources via Git LFS"'
echo '         git push origin main'
