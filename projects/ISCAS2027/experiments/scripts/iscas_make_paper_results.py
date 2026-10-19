#!/usr/bin/env python3
"""Consolidate existing ISCAS 2027 ensemble, exact CLASS and routing outputs.

Does NOT run CLASS or train anything. Read-only analysis of existing benchmark
outputs, and creates a fresh results directory with paper tables and figures.

Example:
  python iscas_make_paper_results.py \
      --benchmarks ~/Yaml/final/Datos/ISCAS_2027/benchmarks \
      --out ~/Yaml/final/Datos/ISCAS_2027/paper_results

Scientific caveats: exact-vs-NN speedup is a measured single-evaluation ratio,
not a posterior-sampling speedup. External validation set is deliberately
stratified (acquisition/uncertainty/low_chi2/random); retrospective CLASS
fallback assumes the stored exact values and is not prospective validation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATASETS = ["SR1_PolyChord", "SR2_PolyChord", "LCDM_PolyChord"]
LABEL = {"SR1_PolyChord": "SR1", "SR2_PolyChord": "SR2", "LCDM_PolyChord": "Lambda-CDM"}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def ordered_files(base, pattern):
    return sorted((p for p in base.glob(pattern) if p.is_file()),
                  key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def same_set(a, b):
    if not isinstance(a, list) or not isinstance(b, list):
        return None
    return sorted(a) == sorted(b)


def corr(a, b):
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else float("nan")


def require_columns(df, columns, file):
    miss = set(columns) - set(df.columns)
    if miss:
        raise ValueError(f"Columnas ausentes {sorted(miss)} en {file}")


def read_matching_class(benchmarks, dataset, expected_host):
    # Only accept a successful 10+-point run on the same HPC node.
    candidates = []
    for path in ordered_files(benchmarks, "class_exact_*/summary.csv"):
        tbl = pd.read_csv(path)
        require_columns(tbl, ["dataset", "n", "all_match_reference", "median_exact_seconds_per_point"], path)
        match = tbl.loc[tbl.dataset.eq(dataset)]
        if match.empty:
            continue
        row = match.iloc[0]
        success = str(row["all_match_reference"]).strip().lower() in ("true", "1")
        if not success or int(row["n"]) < 10:
            continue
        environment = read_json(path.parent / "environment.json")
        if environment.get("host") != expected_host:
            continue
        candidates.append((path, row, environment))
    if not candidates:
        raise RuntimeError(f"No hay benchmark exacto compatible (>=10 puntos, match=True, host={expected_host}) para {dataset}")
    return candidates[0]


def plot_figures(out, table, points, routing):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("AVISO: Matplotlib no está instalado. CSV generados; se omiten figuras.")
        return []
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.labelsize": 10, "savefig.dpi": 300,
                         "figure.dpi": 110})
    images = []
    ordered = table.set_index("dataset").loc[DATASETS]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.1, 3.8))
    ax.bar(x - .2, ordered["class_median_s"], width=.38, label="Cobaya–CLASS")
    ax.bar(x + .2, ordered["ensemble_batch1_latency_s"], width=.38, label="Five-network ensemble")
    ax.set_yscale("log")
    ax.set_xticks(x, [LABEL[d] for d in DATASETS])
    ax.set_ylabel("Time per $\\chi^2$ evaluation (s, log scale)")
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_eval_runtime.png", bbox_inches="tight")
    fig.savefig(out / "fig_eval_runtime.pdf", bbox_inches="tight")
    plt.close(fig)
    images += ["fig_eval_runtime.png", "fig_eval_runtime.pdf"]

    fig, ax = plt.subplots(figsize=(6.1, 3.8))
    w = .25
    ax.bar(x - w, ordered["MAE_chi2"], width=w, label="NN (all selected candidates)")
    if routing is not None:
        rt = routing.set_index("dataset").loc[DATASETS]
        ax.bar(x, rt["ideal_offline_hybrid_mae"], width=w,
               label="Retrospective ideal fallback")
    random_mae = [points.loc[(points.dataset == d) & (points.selection_source == "random"),
                             "absolute_error"].mean() for d in DATASETS]
    ax.bar(x + w, random_mae, width=w, label="NN (random stratum only)")
    ax.set_xticks(x, [LABEL[d] for d in DATASETS])
    ax.set_ylabel("Mean absolute $\\chi^2$ error")
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, 1.20), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_error_comparison.png", bbox_inches="tight")
    fig.savefig(out / "fig_error_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    images += ["fig_error_comparison.png", "fig_error_comparison.pdf"]

    colors = {"random": "C0", "uncertainty": "C1", "acquisition": "C2", "low_chi2": "C3"}
    for dataset in DATASETS:
        subset = points[points.dataset.eq(dataset)]
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for source in ["random", "uncertainty", "acquisition", "low_chi2"]:
            s = subset[subset.selection_source.eq(source)]
            if s.empty:
                continue
            ax.scatter(s["ensemble_sigma"], s["absolute_error"], s=25, alpha=.75,
                       label=source, color=colors[source], edgecolors="none")
        ax.set_xlabel("Ensemble standard deviation in $\\chi^2$")
        ax.set_ylabel("Absolute $\\chi^2$ error")
        ax.set_title(LABEL[dataset], fontsize=11)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        stub = "fig_uncertainty_error_" + dataset.replace("_PolyChord", "")
        fig.savefig(out / (stub + ".png"), bbox_inches="tight")
        fig.savefig(out / (stub + ".pdf"), bbox_inches="tight")
        plt.close(fig)
        images += [stub + ".png", stub + ".pdf"]
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    base = args.benchmarks.expanduser().resolve()
    if not base.is_dir():
        parser.error(f"No existe {base}")
    ensemble_files = ordered_files(base, "ensemble_*/ensemble_benchmark.csv")
    if not ensemble_files:
        parser.error("Falta ensemble_*/ensemble_benchmark.csv")
    ensemble_csv = ensemble_files[0]
    ensemble_dir = ensemble_csv.parent
    points_csv = ensemble_dir / "ensemble_external_replay_points.csv"
    metrics_csv = ensemble_dir / "ensemble_external_replay_metrics.csv"
    for path in [points_csv, metrics_csv]:
        if not path.is_file():
            raise FileNotFoundError(path)
    env_nn = read_json(ensemble_dir / "environment.json")
    host = str(env_nn.get("host", ""))
    if not host:
        raise ValueError("Falta 'host' en environment.json del benchmark neuronal")
    bmk = pd.read_csv(ensemble_csv)
    metrics = pd.read_csv(metrics_csv)
    points = pd.read_csv(points_csv)
    require_columns(bmk, ["dataset", "batch_size", "latency_ms_per_point_amortized", "throughput_points_per_second"], ensemble_csv)
    require_columns(metrics, ["dataset", "MAE_chi2", "RMSE_chi2", "R2_chi2", "coverage_1sigma"], metrics_csv)
    require_columns(points, ["dataset", "selection_source", "absolute_error", "ensemble_sigma"], points_csv)

    source_files = [ensemble_csv, points_csv, metrics_csv, ensemble_dir / "environment.json"]
    class_info = {}
    rows = []
    for d in DATASETS:
        curr = bmk[bmk.dataset.eq(d)]
        if set(curr.batch_size.astype(int)) != {1, 16, 64, 256, 1024}:
            raise ValueError(f"Faltan lotes esperados en {d}")
        bm1 = curr[curr.batch_size.eq(1)].iloc[0]
        bm1024 = curr[curr.batch_size.eq(1024)].iloc[0]
        mm = metrics[metrics.dataset.eq(d)]
        if len(mm) != 1:
            raise ValueError(f"Métricas de {d} duplicadas o ausentes")
        mm = mm.iloc[0]
        path, exact, env_exact = read_matching_class(base, d, host)
        class_info[d] = {"file": str(path), "host": env_exact.get("host"),
                         "cpu_affinity": env_exact.get("cpu_affinity"),
                         "slurm_job_id": env_exact.get("slurm_job_id"),
                         "thread_vars": env_exact.get("thread_vars"),
                         "yaml_paths": env_exact.get("yaml_paths"),
                         "yaml_sha256": env_exact.get("yaml_sha256")}
        source_files.extend([path, path.parent / "environment.json"])
        ratio = float(exact["median_exact_seconds_per_point"]) / (float(bm1["latency_ms_per_point_amortized"]) / 1000)
        affinity_match = same_set(env_nn.get("cpu_affinity"), env_exact.get("cpu_affinity"))
        threads_ok = int(bm1.get("torch_threads", -1)) == 1 and all(
            str(env_exact.get("thread_vars", {}).get(t, "")) == "1"
            for t in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
        rows.append({
            "dataset": d,
            "n_exact": int(exact["n"]),
            "class_median_s": float(exact["median_exact_seconds_per_point"]),
            "class_mean_s": float(exact["mean_exact_seconds_per_point"]),
            "max_abs_class_delta_chi2": float(exact.get("max_abs_delta_chi2", float("nan"))),
            "ensemble_batch1_latency_s": float(bm1["latency_ms_per_point_amortized"]) / 1000,
            "ensemble_batch1_latency_ms": float(bm1["latency_ms_per_point_amortized"]),
            "ensemble_batch1024_throughput_s": float(bm1024["throughput_points_per_second"]),
            "class_to_ensemble_single_eval_ratio": ratio,
            "same_node": True,
            "same_cpu_affinity": affinity_match,
            "single_thread_settings": threads_ok,
            "MAE_chi2": float(mm["MAE_chi2"]),
            "RMSE_chi2": float(mm["RMSE_chi2"]),
            "R2_chi2": float(mm["R2_chi2"]),
            "coverage_1sigma": float(mm["coverage_1sigma"]),
            "coverage_2sigma": float(mm["coverage_2sigma"]),
            "n_external_valid": int(mm["n_valid"]),
        })
    table = pd.DataFrame(rows)
    source_rows = []
    for d in DATASETS:
        subset = points[points.dataset.eq(d)]
        sigma = subset.ensemble_sigma.to_numpy(dtype=float)
        error = subset.absolute_error.to_numpy(dtype=float)
        source_rows.append({"dataset": d, "selection_source": "ALL_MIXED",
                            "n": len(subset), "mae_chi2": error.mean(),
                            "spearman_sigma_abs_error": corr(sigma, error),
                            "mean_sigma": sigma.mean()})
        for source, group in subset.groupby("selection_source"):
            sigma = group.ensemble_sigma.to_numpy(dtype=float)
            error = group.absolute_error.to_numpy(dtype=float)
            source_rows.append({"dataset": d, "selection_source": source,
                                "n": len(group), "mae_chi2": error.mean(),
                                "spearman_sigma_abs_error": corr(sigma, error),
                                "mean_sigma": sigma.mean()})
    source_table = pd.DataFrame(source_rows)
    selective_files = ordered_files(base, "selective_*/selective_cv_summary.csv")
    routing = None
    if selective_files:
        routing = pd.read_csv(selective_files[0])
        require_columns(routing, ["dataset", "fallback_fraction_oof", "ideal_offline_hybrid_mae", "original_mae"], selective_files[0])
        if set(routing.dataset) != set(DATASETS):
            raise ValueError("Datos incompletos en los resultados selectivos")
        source_files.append(selective_files[0])

    out = args.out.expanduser().resolve() / ("iscas_summary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=False)
    table.to_csv(out / "table_1_performance_and_accuracy.csv", index=False)
    source_table.to_csv(out / "table_2_selection_source_diagnostics.csv", index=False)
    if routing is not None:
        routing.to_csv(out / "table_3_retrospective_routing.csv", index=False)
    generated = plot_figures(out, table, points, routing)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_ensemble_run": str(ensemble_dir),
        "ensemble_environment": env_nn,
        "class_runs": class_info,
        "source_selective_analysis": str(selective_files[0]) if selective_files else None,
        "source_sha256": {str(p): sha256(p) for p in source_files if p.is_file()},
        "figures": generated,
        "limitations": [
            "Speedup ratios concern an individual chi2 function call, not an entire MCMC/PolyChord run.",
            "Neural and CLASS timings were collected with different Conda environments/processes; inspect CPU affinity and threading.",
            "Mixed 100-point external validation per model is purposively stratified; pooled metrics are not population-generalization estimates.",
            "Retrospective routing assumes exact substitution from saved Cobaya values and has not been prospectively evaluated.",
            "Random stratum has only 25 points/model; high statistical uncertainty.",
        ],
    }
    (out / "provenance_and_caveats.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
    (out / "README.txt").write_text(
        "ISCAS 2027 — summary of observed experiments\n\n"
        "table_1: median exact CLASS vs CPU ensemble per-point latency + accuracy.\n"
        "table_2: source-stratified diagnostics, essential for selection-bias audit.\n"
        "table_3 (if available): ideal retrospective selective routing; NOT tested online.\n"
        "fig_eval_runtime: logarithmic per-call runtime plot.\n"
        "fig_error_comparison: pooled, random and retrospective error.\n"
        "fig_uncertainty_error: dispersion vs external absolute error.\n"
        "Read provenance_and_caveats.json before publishing speedup claims.\n")
    print("\n=== SCIENTIFIC PERFORMANCE TABLE ===")
    print(table[["dataset", "class_median_s", "ensemble_batch1_latency_ms",
                 "class_to_ensemble_single_eval_ratio", "same_cpu_affinity", "single_thread_settings",
                 "MAE_chi2", "R2_chi2"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== SOURCE-STRATIFIED DIAGNOSTICS ===")
    print(source_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if (table.same_cpu_affinity == False).any() or table.same_cpu_affinity.isna().any():
        print("\nWARNING: CPU affinity differs or cannot be verified across processes. Audit before publishing speedup ratios.")
    if not table.single_thread_settings.all():
        print("WARNING: Single-thread settings differ or were not captured.")
    print("\nSaved to:", out)
    print("No Cobaya/CLASS calls made. No new network training performed.")


if __name__ == "__main__":
    main()
