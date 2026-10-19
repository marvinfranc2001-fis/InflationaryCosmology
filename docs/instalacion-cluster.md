# Instalación y reproducción del entorno HPC (guía documentada)

Para conocer la secuencia original de comandos consulta primero el notebook real [`INSTALL_cluster_cobaya.ipynb`](../INSTALL_cluster_cobaya.ipynb). Esta página organiza sus pasos y separa las hipótesis de instalación del entorno realmente verificado.

## Entorno histórico

En octubre 2026 se verificó que, **dentro de Enroot en compute**, se podía ejecutar:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate inflacion
python -c "import cobaya; print('Cobaya OK')"
python -c "import classy; print('CLASS OK')"
python -c "import pypolychord; print('PolyChord OK')"
```

El contenedor usado fue `condaforge+miniforge3`, entorno Conda `inflacion`. La instalación local en `login1` fallaba por `GLIBC_2.28 not found`. El entorno exportado exacto, incluidas revisiones, **sigue pendiente**.

## Lista de pasos de instalación (desde el notebook original)

1. Usar la cuenta autorizada en CEDIA, reservar recursos de cómputo y arrancar Enroot; `enroot list` permite comprobar qué contenedores existen.
2. Activar Conda dentro del contenedor; para una instalación nueva separada utilizar un nombre temporal, no sobrescribir `inflacion`.
3. Instalar Python 3.10 y dependencias científicas como `numpy`, `scipy`, `cobaya` y compiladores C/C++/Fortran.
4. Configurar MPI y `mpi4py` con `mpicc` y `mpif90`. No mezclar MPI de implementaciones distintas.
5. Instalar o compilar `classy` (CLASS) compatible con Cobaya.
6. Compilar PolyChordLite con `make pypolychord MPI=1` en una versión académicamente autorizada; comprobar `import pypolychord`.
7. Instalar las likelihoods requeridas (`planck_2018_lowl.TT`, `planck_2018_lowl.EE`, `planck_2018_highl_plik.TTTEEE_lite`, `bicep_keck_2018`) vía `cobaya-install` y proveedores autorizados; algunas requieren datos externos bajo licencias.
8. Poner los módulos originales `effective.py` en `PYTHONPATH`, utilizar `OMP_NUM_THREADS=1`, iniciar cada YAML exacto con `cobaya-run <ruta> --test` antes de ejecutar producción.

### Compilación de referencia para PolyChordLite

```bash
# Ya en un nodo/Enroot con entorno activado, y repo PolyChordLite existente:
export CC=mpicc CXX=mpicxx FC=mpif90 F77=mpif90 F90=mpif90
cd ~/cobaya_packages/code/PolyChordLite
make clean
make pypolychord MPI=1
python -m pip install . --no-build-isolation
```

Verificar previamente compiladores, versión de MPI y licencia. Comandos de referencia, no instalador probado ahora.

## Captura de versiones antes de reproducir

```bash
bash scripts/capturar_versiones_hpc.sh
```

Esto guarda versiones bajo `versiones_entorno/` (ignorado por Git hasta revisión). `conda env export` en 2026 no reemplaza el lock original si éste no existe; registrar fecha y hash para distinguir una reconstrucción.

## Ejecución

Para el orden diario y seguimiento de SLURM ver [`docs/comandos-cluster.md`](comandos-cluster.md). Para ejecutar producción necesitas **YAML auténticos**, el mismo código Python y los mismos datos/priors: `configs_reconstruidas/` **no** satisface ese requisito.
