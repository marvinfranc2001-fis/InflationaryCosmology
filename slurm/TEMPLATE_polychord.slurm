#!/usr/bin/env bash
# PLANTILLA DIDÁCTICA, NO EL WRAPPER ORIGINAL; no ejecuta inferencia.
#SBATCH --job-name=starobinsky_demo
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=starobinsky_demo_%j.out
#SBATCH --error=starobinsky_demo_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
echo "Asignado a $(hostname) job ${SLURM_JOB_ID:-desconocido}"
echo "Sustituir por wrapper ORIGINAL, con Enroot, MPI, scratch, rsync, resume."
exit 2
