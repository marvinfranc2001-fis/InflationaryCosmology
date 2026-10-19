# Comandos operativos del clúster (SLURM, Enroot, Conda, Cobaya)

Este es un **registro de procedimientos históricamente usados**, no una sesión SSH activa. Ajustar el host SSH, particiones, CPUs, memory, políticas de MPI y rutas al ambiente autorizado.

## Orden de acceso: SSH → asignación SLURM → nodo → Enroot → Conda

```bash
# Desde tu equipo: sustituir por tu alias de acceso autorizado
ssh USUARIO@HOST_CLUSTER

# En login1: revisar recursos y solicitar una sesión
sinfo
salloc -p cpu -n 2 -c 1 --mem=16G --time=04:00:00

# La asignación indica el nodo que corresponde. Ejemplo histórico:
ssh compute-0-2

# En compute-0-2; contenedor usado en una sesión verificada:
enroot list
enroot start 'condaforge+miniforge3' /bin/bash

# Dentro del contenedor:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate inflacion
export OMP_NUM_THREADS=1
cd ~/Yaml
python -c 'import cobaya, classy, pypolychord; print("OK")'
```

**Importante:** no intentar entrar a un nodo arbitrario sin reserva; algunas instalaciones colocan la shell de `salloc` ya en compute, otras requieren `srun --pty bash`. Obedecer la configuración del centro. No correr inferencia cosmológica ni compilaciones pesadas en `login1`.

## Envío y monitorización de trabajos batch

```bash
cd ~/Yaml/final
sbatch run_polychord_SR1.slurm
squeue -u "$USER"
squeue -j JOBID -o "%.18i %.10P %.25j %.10u %.2t %.12M %.12l %.4D %R"
scontrol show job JOBID
tail -n 100 SR1_300_JOBID.out
tail -n 100 SR1_300_JOBID.err
```

Nombres de logs y Job IDs son ilustrativos: el wrapper controla el prefijo real. Para observar evidencias PolyChord examinar `ndead`, `nposterior`, `log(Z)`, `log(Z_1)` y si el proceso terminó antes de informar el resultado definitivo.

## Cobaya: pruebas, reanudación y errores

```bash
cobaya-run archivo-original.yaml --test
cobaya-run archivo-original.yaml
cobaya-run archivo-original.yaml --resume
```

`--resume` sólo cuando existe estado válido de la misma configuración/salida. No aplicar `--force` sobre resultados que quieras conservar. En varias corridas el wrapper SLURM manejaba copias `/scratch`→home con `rsync`; verificar que el proceso finalizó y que la copia fue consistente antes de declarar los resultados disponibles.

## Error de GLIBC que apareció en login

```text
/lib64/libc.so.6: version 'GLIBC_2.28' not found
```

La versión de Python dentro de una distribución Conda más reciente no podía correr directamente en el nodo login antiguo. **No sustituir bibliotecas GLIBC del sistema**. Volver al flujo Enroot/nodo compatible y activar el entorno allí.

## Salud del entorno

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate inflacion
which python
python -c 'import cobaya; print("Cobaya OK")'
python -c 'import classy; print("CLASS OK")'
python -c 'import pypolychord; print("PolyChord OK")'
command -v mpirun
```
