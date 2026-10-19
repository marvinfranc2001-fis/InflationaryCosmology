# Machine learning — inventario de la tesis

La tesis estudió clasificación de posteriores (regresión logística, random forest), ablación, permutation tests, SHAP, PCA y UMAP; emuladores neuronales locales de χ² con PyTorch y validación externa Cobaya/CLASS, además de autoencoders. Estos métodos **no reemplazaron** las evidencias calculadas por PolyChord.

Los códigos originales deben recuperarse de `~/Yaml/final/Datos/`:
- `ml_final_tesis.py`
- `pytorch_tesis.py`
- `tesis_chain_utils.py`
- `validar_emulador_cobaya.py`
- `grafin_publicable_getdist_verde.py`
- `codigos_tesis_corregidos/`, `ml_final/`, `pytorch_resultados_tesis_nuevos/`

El importador HPC copia fuentes, pero no añade automáticamente binarios de entrenamiento ni datos grandes. Una vez recuperados, registrar semillas, particiones, arquitectura, hiperparámetros y métricas de validación.
