#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Valida candidatos de active learning con evaluaciones reales Cobaya/CLASS.

Compatible con las salidas de ``ml_final_tesis_corregido.py`` para:
  --model sr1
  --model sr2
  --model lcdm

Ejemplo:
  python validar_emulador_cobaya_corregido.py \
      --model lcdm \
      --packages-path ~/cobaya_packages

El script selecciona automáticamente el YAML PolyChord y el CSV top10 del
modelo. También se pueden indicar explícitamente con --yaml y --candidates.

Dependencias:
  pip install numpy pandas matplotlib pyyaml cobaya
Además deben estar instalados CLASS y las likelihoods de Planck/BK18 usadas
por el YAML.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tesis_chain_utils import (
    GREEN,
    canonical_column,
    prepare_roots,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validación Cobaya de un emulador local")
    p.add_argument("--model", choices=["sr1", "sr2", "lcdm"], required=True)
    p.add_argument("--yaml", type=Path, default=None)
    p.add_argument("--candidates", type=Path, default=None)
    p.add_argument("--poly-zip", type=Path, default=Path("chains_poly.zip"))
    p.add_argument("--mcmc-zip", type=Path, default=Path("chains_mcmc.zip"))
    p.add_argument("--lcdm-zip", type=Path, default=Path("chainslcdm.zip"))
    p.add_argument("--base", type=Path, default=Path.cwd())
    p.add_argument("--cache-dir", type=Path, default=Path(".tesis_chain_cache"))
    p.add_argument("--ml-out", type=Path, default=Path("ml_final_resultados_tesis_nuevos"))
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--packages-path", type=Path, default=None)
    p.add_argument("--max-candidates", type=int, default=10)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


ARGS = parse_args()

try:
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file
except ImportError as exc:
    raise SystemExit(
        "Cobaya no está instalado en este entorno. Activa tu venv de cosmología "
        "y ejecuta: python -m pip install cobaya\n"
        f"Error original: {exc}"
    )

MODEL_TO_DATASET = {
    "sr1": "SR1_PolyChord",
    "sr2": "SR2_PolyChord",
    "lcdm": "LCDM_PolyChord",
}
DATASET = MODEL_TO_DATASET[ARGS.model]
OUT = (ARGS.outdir or Path(f"validation_emulator_cobaya_{ARGS.model}")).expanduser()
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11.5,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "figure.dpi": 130,
    "savefig.dpi": 500,
    "axes.edgecolor": GREEN["text"],
    "axes.labelcolor": GREEN["text"],
    "xtick.color": GREEN["text"],
    "ytick.color": GREEN["text"],
    "text.color": GREEN["text"],
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=500, bbox_inches="tight")
    plt.close(fig)


def auto_paths() -> tuple[Path, Path]:
    roots = prepare_roots(
        poly_zip=ARGS.poly_zip,
        mcmc_zip=ARGS.mcmc_zip,
        lcdm_zip=ARGS.lcdm_zip,
        base=ARGS.base,
        cache_dir=ARGS.cache_dir,
    )
    yaml_path = Path(str(roots[DATASET]) + ".input.yaml")
    candidate_path = (
        ARGS.ml_out
        / "05_chi2_emulators"
        / DATASET
        / "top10_candidates_for_validation.csv"
    )
    return yaml_path, candidate_path


auto_yaml, auto_candidates = auto_paths()
yaml_path = (ARGS.yaml or auto_yaml).expanduser()
candidates_path = (ARGS.candidates or auto_candidates).expanduser()

if not yaml_path.exists():
    raise FileNotFoundError(f"No existe el YAML: {yaml_path}")
if not candidates_path.exists():
    raise FileNotFoundError(
        f"No existe el CSV de candidatos: {candidates_path}\n"
        "Ejecuta primero ml_final_tesis_corregido.py o usa --candidates."
    )
if ARGS.max_candidates < 1:
    raise ValueError("--max-candidates debe ser positivo")

print("Modelo:", ARGS.model)
print("YAML:", yaml_path)
print("Candidatos:", candidates_path)
print("Salida:", OUT)

# Se usa input.yaml, no updated.yaml, para evitar rutas absolutas antiguas del cluster.
info = yaml_load_file(str(yaml_path))
info.pop("sampler", None)
info.pop("output", None)
info.pop("force", None)

packages_path = str(ARGS.packages_path.expanduser()) if ARGS.packages_path else None
model = get_model(
    info,
    stop_at_error=True,
    debug=ARGS.debug,
    packages_path=packages_path,
)

sampled_params = list(model.parameterization.sampled_params())
print("\nParámetros muestreados requeridos por Cobaya:")
for param in sampled_params:
    print("  ", param)

candidates = pd.read_csv(candidates_path)
if "dataset" in candidates.columns:
    declared = set(candidates["dataset"].dropna().astype(str))
    if declared and declared != {DATASET}:
        raise ValueError(
            f"El CSV declara dataset={declared}, pero --model {ARGS.model} requiere {DATASET}."
        )

# Conversiones permitidas y transparentes.
if "A_s" not in candidates.columns and "logA" in candidates.columns:
    candidates["A_s"] = np.exp(candidates["logA"].astype(float)) / 1.0e10

if "acquisition_score" in candidates.columns:
    candidates = candidates.sort_values("acquisition_score")
candidates = candidates.head(ARGS.max_candidates).reset_index(drop=True)

missing = [p for p in sampled_params if p not in candidates.columns]
if missing:
    raise ValueError(
        "Faltan parámetros que Cobaya necesita en el CSV:\n  - "
        + "\n  - ".join(missing)
        + "\nUsa el top10 generado para el mismo modelo por ml_final_tesis_corregido.py."
    )

rows: list[dict] = []
for i, row in candidates.iterrows():
    point = {p: float(row[p]) for p in sampled_params}
    print("\n" + "=" * 70)
    print(f"Evaluando candidato {i + 1}/{len(candidates)}")

    result_row: dict = {
        "candidate_id": i + 1,
        "model": ARGS.model,
        "dataset": DATASET,
        "predicted_chi2": float(row.get("predicted_chi2", np.nan)),
        "predicted_delta_chi2": float(row.get("predicted_delta_chi2", np.nan)),
        "emulator_uncertainty": float(row.get("emulator_uncertainty", np.nan)),
        "acquisition_score": float(row.get("acquisition_score", np.nan)),
    }
    result_row.update(point)

    try:
        posterior = model.logposterior(
            point,
            as_dict=True,
            make_finite=False,
            return_derived=True,
            cached=False,
        )
        loglikes = posterior.get("loglikes", {})
        logpriors = posterior.get("logpriors", {})
        derived = posterior.get("derived", {})

        if not loglikes:
            result_row.update({
                "status": "OUTSIDE_PRIOR_OR_INVALID",
                "error_message": "Cobaya no evaluó likelihoods; revise priors/parámetros.",
            })
            rows.append(result_row)
            print("Punto fuera del prior o inválido")
            continue

        chi2_components = {
            "chi2__" + canonical_column(name): -2.0 * float(value)
            for name, value in loglikes.items()
        }
        chi2_real = float(sum(chi2_components.values()))
        predicted = result_row["predicted_chi2"]
        residual = chi2_real - predicted if np.isfinite(predicted) else np.nan
        uncertainty = result_row["emulator_uncertainty"]

        result_row.update({
            "status": "OK",
            "real_chi2": chi2_real,
            "real_logpost": float(posterior.get("logpost", np.nan)),
            "real_logprior": float(sum(logpriors.values())) if isinstance(logpriors, dict) else np.nan,
            "residual_real_minus_pred": residual,
            "abs_residual": abs(residual) if np.isfinite(residual) else np.nan,
            "standardized_residual": (
                residual / uncertainty
                if np.isfinite(residual) and np.isfinite(uncertainty) and uncertainty > 0
                else np.nan
            ),
        })
        result_row.update(chi2_components)

        if isinstance(derived, dict):
            for key, value in derived.items():
                try:
                    result_row["derived__" + key] = float(value)
                except (TypeError, ValueError):
                    pass

        print(f"chi2 real      = {chi2_real:.8f}")
        print(f"chi2 predicho  = {predicted:.8f}")
        print(f"residual       = {residual:.8f}")

    except Exception as exc:
        result_row.update({
            "status": "ERROR",
            "error_message": str(exc),
        })
        print("ERROR:", exc)

    rows.append(result_row)

results = pd.DataFrame(rows)
results.to_csv(OUT / "cobaya_validation_results.csv", index=False)
ok = results[results["status"] == "OK"].copy()

if len(ok) > 0:
    finite = ok[
        np.isfinite(ok["predicted_chi2"])
        & np.isfinite(ok["real_chi2"])
    ].copy()
    if len(finite) > 0:
        residual = finite["real_chi2"].to_numpy() - finite["predicted_chi2"].to_numpy()
        denom = np.sum((finite["real_chi2"] - finite["real_chi2"].mean()) ** 2)
        r2 = 1.0 - np.sum(residual**2) / denom if denom > 0 else np.nan
        summary = {
            "model": ARGS.model,
            "dataset": DATASET,
            "N_requested": len(candidates),
            "N_OK": len(finite),
            "MAE_chi2": float(np.mean(np.abs(residual))),
            "RMSE_chi2": float(np.sqrt(np.mean(residual**2))),
            "bias_real_minus_pred": float(np.mean(residual)),
            "R2": float(r2),
        }
        if "emulator_uncertainty" in finite:
            unc = finite["emulator_uncertainty"].to_numpy(float)
            valid_unc = np.isfinite(unc) & (unc > 0)
            if valid_unc.any():
                summary["coverage_abs_residual_le_1sigma"] = float(
                    np.mean(np.abs(residual[valid_unc]) <= unc[valid_unc])
                )
                summary["mean_abs_standardized_residual"] = float(
                    np.mean(np.abs(residual[valid_unc] / unc[valid_unc]))
                )
        pd.DataFrame([summary]).to_csv(OUT / "validation_summary_metrics.csv", index=False)

        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        ax.scatter(finite["predicted_chi2"], finite["real_chi2"],
                   s=55, color=GREEN["medium"], alpha=0.85)
        lo = min(finite["predicted_chi2"].min(), finite["real_chi2"].min())
        hi = max(finite["predicted_chi2"].max(), finite["real_chi2"].max())
        ax.plot([lo, hi], [lo, hi], color=GREEN["dark"], lw=2)
        ax.set_xlabel(r"Emulador: $\chi^2_{\rm pred}$")
        ax.set_ylabel(r"Cobaya/CLASS: $\chi^2_{\rm real}$")
        ax.set_title(f"Validación externa — {DATASET}")
        ax.grid(alpha=0.25, color=GREEN["grid"])
        fig.tight_layout()
        save_fig(fig, "predicted_vs_real_chi2")

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.axhline(0.0, color=GREEN["dark"], lw=2)
        ax.scatter(finite["candidate_id"], residual,
                   s=55, color=GREEN["medium"], alpha=0.85)
        if "emulator_uncertainty" in finite:
            unc = finite["emulator_uncertainty"].to_numpy(float)
            if np.isfinite(unc).any():
                ax.errorbar(finite["candidate_id"], residual, yerr=unc,
                            fmt="none", ecolor=GREEN["sage"], capsize=3, alpha=0.8)
        ax.set_xlabel("ID del candidato")
        ax.set_ylabel(r"$\chi^2_{\rm real}-\chi^2_{\rm pred}$")
        ax.set_title("Residuos de validación")
        ax.grid(alpha=0.25, color=GREEN["grid"])
        fig.tight_layout()
        save_fig(fig, "validation_residuals")

        print("\nRESUMEN")
        for key, value in summary.items():
            print(f"{key}: {value}")

# Cierra recursos externos si la versión de Cobaya lo implementa.
close_method = getattr(model, "close", None)
if callable(close_method):
    close_method()

(OUT / "environment.json").write_text(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "model": ARGS.model,
    "dataset": DATASET,
    "yaml": str(yaml_path.resolve()),
    "candidates": str(candidates_path.resolve()),
    "packages_path": packages_path,
    "command": " ".join(sys.argv),
}, indent=2), encoding="utf-8")

(OUT / "NOTA_METODOLOGICA.txt").write_text(
    "La validación debe usar la misma implementación de likelihood que la cadena "
    "con la que se entrenó el emulador. Los emuladores PolyChord usan el YAML "
    "TTTEEE_lite (clik). Aunque TTTEEE_lite_native produce posteriores compatibles, "
    "su chi2 absoluto puede diferir por una constante, por lo que no debe compararse "
    "directamente con la predicción raw del emulador clik.\n",
    encoding="utf-8",
)

print("\nGuardado en:", OUT.resolve())