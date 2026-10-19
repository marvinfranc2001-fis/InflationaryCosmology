# Resultados científicos — lectura histórica

Transcripción desde el manuscrito final, tablas 5.2 y 5.3 (no corrida en el presente repositorio).

| Modelo | ln Z | error ln Z |
| --- | ---: | ---: |
| Reference ΛCDM + r | -797.3157 | 0.2742 |
| Starobinsky SR1 | -795.3275 | 0.2655 |
| Starobinsky SR2 | -794.4122 | 0.2636 |

| Comparación | Δ ln Z | Error combinado | Bayes factor |
| --- | ---: | ---: | ---: |
| SR1 / ΛCDM + r | 1.9882 | 0.3816 | 7.30 |
| SR2 / ΛCDM + r | 2.9036 | 0.3803 | 18.24 |
| SR2 / SR1 | 0.9154 | 0.3741 | 2.50 |

- Las conclusiones dependen del conjunto específico de datos, priors y nuisance.
- El resultado SR2 frente a SR1 se interpreta como apoyo **débil**, no prueba concluyente.
- Ambas corridas MCMC superaron el criterio adoptado `R−1<0.01`; el máximo desplazamiento de medias MCMC versus PolyChord fue `0.0763 sigma` (para `ln(10^10 A_s)` en SR2).
- La running `alpha_s` de SR2 fue derivada de `N_star`, **no medida independientemente**.
- Los clasificadores y emuladores ML son complementarios y no sustituyen el cálculo exacto de evidencia.

## Archivos pendientes para replicar estas cifras

Las cadenas **finales** `nlive=300` y sus estadísticas deben copiarse desde los directorios de salida autorizados del clúster; los archivos Nfree incluidos en `chains/` representan una **prueba inicial** con otros likelihoods y `nlive=100`. No convertir su `log(Z)` en un valor de las tablas.
