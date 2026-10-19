# Almacenar el archivo completo mediante Git LFS

## Estado real

Los cinco ZIP enviados con la tesis fueron examinados e inventariados. En GitHub ya están sus manifiestos y hashes SHA-256, YAML de producción, diagnósticos de evidencia, resumen de ML, scripts y vistas previas. **Estado actualizado (2026-10-18): los cinco ZIP originales ya se publicaron mediante Git LFS**, usando el equipo Linux del autor. El commit original de carga es [`957863f`](https://github.com/marvinfranc2001-fis/InflationaryCosmology/commit/957863f81a746f397dd3a6ce081a26dcc3538189). El cliente Git LFS confirmó transferencia `100% (5/5), 810 MB`, y los cinco punteros verificados en la rama `main` contienen los mismos SHA-256 y tamaños que los ZIP suministrados. El plugin de ChatGPT por sí solo no puede transferir objetos LFS. Se recomienda probar la descarga desde un segundo clon con `git lfs pull`.

| Archivo | Tamaño (bytes) | Contenido principal |
|---|---:|---|
| `Yamls.zip` | 4 672 | Apuntes originales de configuraciones |
| `chains_poly.zip` | 387 480 823 | PolyChord SR1/SR2, muestras, checkpoints y restart |
| `chains_mcmc.zip` | 143 684 644 | MCMC de SR1/SR2 |
| `chainslcdm.zip` | 63 548 377 | PolyChord de ΛCDM+r |
| `Graf_ml_definitivo.zip` | 214 998 882 | Figuras originales, ensambles de ML y resultados |

**Conserva los cinco ZIP sin modificarlos.** GitHub LFS puede generar cargos/cuotas según la cuenta. Comprueba las condiciones de almacenamiento y transferencia de GitHub antes de subirlos; el total son unos **810 MB**. Si no quieres usar LFS, una alternativa es un depósito científico con DOI y enlazarlo desde `data/README.md`.

## Descargar y verificar los originales después de la publicación

```bash
# Desde un clon con Git LFS instalado:
git lfs pull
python3 scripts/verificar_archivos.py .
```

## Registro de la carga original desde Linux (ya completada)

Instala `git`, `git-lfs` y Python 3. Después de asegurarte de tener autorización para publicar todos los archivos incluidos:

```bash
# Clona el repositorio PRIVATE con tu autenticación habitual de GitHub.
git clone https://github.com/marvinfranc2001-fis/InflationaryCosmology.git
cd InflationaryCosmology

# Comprobación sin cambios: carpeta donde están los cinco ZIP originales
python3 scripts/verificar_archivos.py /ruta/real/a/los/cinco/ZIPs

# Preparar datos en Git LFS. Verifica hashes y muestra qué se añadirá.
bash scripts/publicar_archivos_git_lfs.sh /ruta/real/a/los/cinco/ZIPs --prepare

git status --short
git lfs ls-files

# Cuando todo esté correcto:
git commit -m "Archive complete thesis data via Git LFS"
git push origin main
```

Como alternativa, inicia con un clon limpio y utiliza el modo interactivo `--push`, que exige escribir `PUBLICAR` antes de crear commit/push:

```bash
bash scripts/publicar_archivos_git_lfs.sh /ruta/real/a/los/cinco/ZIPs --push
```

El script rechaza un origen de Git distinto, un árbol de trabajo sucio, archivos faltantes o un SHA-256 incorrecto. El modo por defecto `--prepare` **no publica**. Para reintentar después de `--prepare`, haz `git commit` y `git push` manualmente; no ejecutes otra vez `--push` sobre un árbol ya modificado.

## Comprobar después del push

```bash
git lfs ls-files
git lfs status
python3 scripts/verificar_archivos.py .
```

Los `.resume` individuales superan en algunos casos los 100 MB, pero van dentro de `chains_poly.zip` y `chainslcdm.zip`, que se almacenan mediante Git LFS. **No descomprimas y añadas todos los objetos binarios en Git normal.** Git LFS conserva la referencia ligera en el commit y el contenido real en el almacenamiento LFS de GitHub. Un tercero con acceso al repositorio necesita `git lfs pull` tras clonar.

## Seguridad y reproducibilidad

Los ZIP de esta entrega contienen resultados científicos y modelos, no el código/licencias de los datasets observacionales. Debes revisar los ZIP antes de hacer público el repositorio. Se recomienda mantenerlo privado mientras completas la revisión, especialmente por rutas históricas o metadatos internos de HPC. Los archivos `.input.yaml` conservan rutas absolutas originales para fines de trazabilidad; no debe interpretarse que funcionen sin adaptar el entorno.
