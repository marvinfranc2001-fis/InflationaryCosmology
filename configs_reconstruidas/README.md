# Ejemplos reconstruidos — NO configuraciones finales

Estos archivos YAML se derivan de ecuaciones y priors reportados en el manuscrito. **No son copias de las corridas definitivas.** En especial, no incorporan todas las opciones de likelihoods/nuisance, rutas, versiones, interfaces del módulo `effective.py`, ajustes MPI y muestreador de producción.

- `polychord_SR1.example.yaml`, `polychord_SR2.example.yaml`, `polychord_LCDM_r.example.yaml`: esquema ilustrativo de nested sampling con `nlive=300`.
- `mcmc_SR1.example.yaml` y `mcmc_SR2.example.yaml`: esquema de Metropolis con la variante `TTTEEE_lite_native`.
- **No usar estas plantillas para calcular ni publicar evidencia científica.**
- La clase Cobaya original debe inspeccionarse y compararse con las ecuaciones de la tesis final; revisar `docs/auditoria-codigo.md`.
