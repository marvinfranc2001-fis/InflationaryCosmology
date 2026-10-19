"""Ecuaciones reportadas en la tesis final; no reemplaza effective.py en Cobaya."""
from dataclasses import dataclass
from math import exp, log

K_PIVOT = 0.05

@dataclass(frozen=True)
class Observables:
    n_s: float
    r_005: float
    alpha_s: float

def sr1(N: float) -> Observables:
    if N <= 0:
        raise ValueError("N debe ser positivo")
    return Observables(1 - 2/N - 9/(2*N*N), 12/(N*N), 0.0)

def sr2(N: float) -> Observables:
    if N <= 0:
        raise ValueError("N debe ser positivo")
    return Observables(1 - 2/N - 3/(2*N*N), (12/(N*N))*(1-2/N), -2/(N*N)-3/(N**3))

def primordial_scalar_power(k:float, A_s:float, n_s:float, alpha_s:float=0.0) -> float:
    if k <= 0 or A_s <= 0:
        raise ValueError("k y A_s deben ser positivos")
    x = log(k/K_PIVOT)
    return exp(log(A_s)+(n_s-1)*x+0.5*alpha_s*x*x)
