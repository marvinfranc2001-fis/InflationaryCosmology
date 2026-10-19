# Auditoría: código original exploratorio versus tesis final

**Fecha de revisión:** 2026-10-18. **No se ha modificado ningún archivo original.**

## Hallazgos observables en el repositorio

1. `effective.py` es un archivo real con las clases `EffectiveStarobinsky_SR1`, `EffectiveStarobinsky_SR2` y `EffectiveStarobinsky_SR2_reh`. En SR1 usa `epsilon=3/(4N²)` y `eta=-1/N`: su expresión resultante `n_s=1-2/N-9/(2N²)` y `r=12/N²` coincide algebraicamente con SR1 de la tesis.
2. **Atención en SR2:** el archivo existente usa la expresión `n_s=1-6*epsilon+2*eta-24*epsilon**2+16*epsilon*eta-2*xi2`, así como `r=16*epsilon*(1+C*(epsilon-eta))` con `C=-0.73`. La tesis final, en cambio, reporta `n_s=1-2/N-3/(2N²)`, `r=12/N²*(1-2/N)` y `alpha_s=-2/N²-3/N³`. **No son ecuaciones equivalentes en general**. Es imprescindible localizar la implementación exacta usada en las corridas de junio 2026; no concluir que el `effective.py` actual reconstruye sus evidencias.
3. `polychord_starobinsky_Nfree.yaml` configura **`nlive=100`** y sólo `planck_2018_lowl.TT` y `planck_2018_highl_plik.TTTEEE_lite`. No contiene `planck_2018_lowl.EE` ni `bicep_keck_2018`. No es comparable directamente con las evidencias definitivas.
4. El YAML de exploración usa la clase SR1 y contiene una lista de parámetros derivados (`n_s`, `r`, `alpha_s`) cuyo enlace real a CLASS debe verificarse mediante inicialización de Cobaya; una apariencia válida en YAML no demuestra que CLASS haya usado las cantidades derivadas deseadas.
5. `INSTALL_cluster_cobaya.ipynb` preserva comandos de instalación, incluidos `enroot start 'condaforge+miniforge3' /bin/bash`, Conda y compilación PolyChord. Es documentación histórica, no un entorno exportado bloqueado por versiones.
6. El `chains/` incluido contiene archivos de prueba **Nfree**. No interpretarlos como muestras definitivas SR1/SR2 y no compararlos con las tablas 5.2–5.3 de la tesis.

## Datos de producción a recuperar

- YAML exactos finales SR1, SR2 y ΛCDM + r, incluyendo los nuisance y la lista completa de likelihoods.
- Versiones de `effective.py` o sus módulos personalizados realmente empleados en cada corrida definitiva.
- `run_polychord_SR1.slurm`, `run_polychord_SR1_resume.slurm`, `run_polychord_SR2_MPI_seguro.slurm`, MCMC y rutas/variables de ejecución.
- Logs finales con Job IDs, rutas y evidencias, así como scripts de postprocesamiento ML.
- Export de versiones del contenedor y entorno; checksums de originales.

## Reglas de interpretación

Las expresiones de la tesis se incluyen en `modelos/starobinsky.py` **exclusivamente como material matemático de referencia**, no como sustituto compatible con Cobaya. Los YAML ilustrativos se agrupan en `configs_reconstruidas/` y usan fórmulas claras, pero se debe ejecutar `cobaya-run --test` con los originales exactos y su software antes de publicar una reproducción. Nunca modificar silenciosamente los archivos históricos.
