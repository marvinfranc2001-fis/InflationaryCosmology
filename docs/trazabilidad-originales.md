# Trazabilidad y catálogo de archivos originales

## Archivos presentes ya en GitHub

- `effective.py`: código Python de Starobinsky **histórico** (incluye SR1, SR2, SR2 reheating; SR2 difiere de la tesis final).
- `polychord_starobinsky_Nfree.yaml`: PolyChord **exploratorio** (nlive=100).
- `chains/`: salida exploratoria Nfree; incluye `.input.yaml`, `.updated.yaml`, `.stats`, `.resume`, `*_dead*.txt`.
- `INSTALL_cluster_cobaya.ipynb`: guía de instalación original cargada por el usuario.
- `Starobinsky_Cobaya_Tesis_Repositorio_Previo_GitHub.zip`: paquete de documentación reconstruido, no archivo directo del clúster.

## Inventario histórico — pendiente de extraer directamente del HPC

**Cuenta propia, `~/Yaml/`:** `effective.py`, `mcmc_starobinsky_effective_SR1.yaml`, `mcmc_starobinsky_effective_SR2.yaml`, `mcmc_starobinsky_fenomenologico.yaml`, `polychord_starobinsky_SR1.yaml`, `polychord_starobinsky_SR2.yaml`, `polychord_starobinsky_N50.yaml`, `polychord_starobinsky_N60.yaml`, `polychord_starobinsky_Nfree.yaml`, `polychord_starobinsky_SR2_reh.yaml`, `comprobar_dependencia_starobinsky.py`, `run_polychord_SR1_test30.slurm`, `run_polychord_SR1_test30_resume.slurm`.

**Cuenta propia, `~/Yaml/final/`:** `polychord_starobinsky_SR1.yaml`, `run_polychord_SR1.slurm`, `run_polychord_SR1_resume.slurm`, `chains_SR1_300/`, logs `SR1_300_15453.out` y `SR1_300_resume_16729.out`.

**Cuenta propia, `~/Yaml/final/Datos/`:** `ml_final_tesis.py`, `pytorch_tesis.py`, `tesis_chain_utils.py`, `validar_emulador_cobaya.py`, `grafin_publicable_getdist_verde.py`, `codigos_tesis_corregidos/`, `ml_final/`, `pytorch_resultados_tesis_nuevos/`, `graficos_finales/`, `Resultados/`.

**Cuenta de colaboradora**: se documentaron trabajos SR2 MPI, como `run_polychord_SR2_MPI_seguro.slurm`, `chains_SR2_300_MPI`, directorio `~/final`. El importador **NO accede ni solicita** recursos de otras cuentas; integrar sólo materiales cedidos expresamente por su propietario.

## Proceso controlado

1. Clonar el repositorio privado dentro del HPC, con permiso de acceso, y ejecutar `bash scripts/importar_originales_hpc.sh`.
2. El script solo copia los archivos fuente legibles de `~/Yaml/` a `originales_hpc/` en la copia local; no modifica los originales.
3. Revisar archivos (rutas, información sensible, derechos, versiones), confirmar SHA256 y ejecutar pruebas.
4. Agregar selectivamente archivos autorizados tras revisión; `originales_hpc/` está ignorado por defecto para impedir publicación accidental.
5. Resultados grandes/externos requieren repositorio o almacenamiento de datos separado con un manifiesto de hashes.

La existencia de un nombre de archivo en esta lista **no prueba** que su contenido se haya recuperado o verificado. 
