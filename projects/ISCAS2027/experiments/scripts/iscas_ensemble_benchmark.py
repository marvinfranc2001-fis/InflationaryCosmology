#!/usr/bin/env python3
"""Reproducible CPU benchmark and external-validation replay for the saved
Starobinsky neural-chi2 ensembles (SR1, SR2, LCDM PolyChord).

Usage (inside the named Slurm compute allocation, env iscas_repro):
    python iscas_ensemble_benchmark.py \
        --root ~/Yaml/final/Datos/ISCAS_2027/import_local/DataProcessing/Graf_ml_definitivo \
        --out ~/Yaml/final/Datos/ISCAS_2027/benchmarks

Inputs (trusted user-produced checkpoints/scalers):
    ROOT/05_chi2_emulators/{DATASET}/emulator_member_XX.pt, *scaler.joblib, metadata.json
    ROOT/06_external_validation/{DATASET}/cobaya_validation_results.csv
The script DOES NOT call Cobaya/CLASS or retrain anything. It reports inference
and selected-candidate validation metrics; those candidates are NOT IID posterior
samples. Historical Cobaya runtimes from another machine are NOT used for speedup.
"""
import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn

DATASETS = ("SR1_PolyChord", "SR2_PolyChord", "LCDM_PolyChord")
BATCH_SIZES = (1, 16, 64, 256, 1024)


class Chi2Emulator(nn.Module):
    """Architecture from starobinsky_ml_pipeline_c_v2.py, not the older 192-128-80 net."""
    def __init__(self, n_inputs: int, n_outputs: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128), nn.SiLU(), nn.BatchNorm1d(128), nn.Dropout(0.05),
            nn.Linear(128, 96), nn.SiLU(), nn.BatchNorm1d(96), nn.Dropout(0.05),
            nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        return self.network(x)


def predict_raw(ensemble, xs, ys, raw_features):
    """Full inference: input scaling, 5 forwards, output scaling, ensemble sigma."""
    scaled = xs.transform(raw_features)
    tensor = torch.from_numpy(np.asarray(scaled, dtype=np.float32))
    with torch.inference_mode():
        standardized = np.stack([m(tensor).numpy() for m in ensemble], axis=0)
    raw = np.stack([ys.inverse_transform(p) for p in standardized], axis=0)
    return raw.mean(axis=0), raw.std(axis=0, ddof=1)


def infer_timed(ensemble, xs, ys, raw_features, batch_size):
    # All preprocessing, per-member forward passes, and postprocessing timed.
    for begin in range(0, len(raw_features), batch_size):
        predict_raw(ensemble, xs, ys, raw_features[begin:begin + batch_size])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Parent output folder; a fresh timestamped child is created")
    parser.add_argument("--n", type=int, default=3000,
                        help="Predictions per timing repetition (default 3000)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.n < 1024 or args.repeats < 1 or args.threads < 1:
        parser.error("Use --n >= 1024, --repeats >= 1, --threads >= 1")
    root = args.root.expanduser().resolve()
    for part in ("05_chi2_emulators", "06_external_validation"):
        if not (root / part).is_dir():
            raise FileNotFoundError(f"Missing directory: {root / part}")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.out.expanduser() / f"ensemble_{stamp}_{platform.node()}"
    output.mkdir(parents=True, exist_ok=False)
    spec = {
        "utc_timestamp": stamp,
        "root": str(root),
        "output": str(output),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "threads_torch": torch.get_num_threads(),
        "threads_interop": torch.get_num_interop_threads(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "n_timing_points": args.n,
        "repeats": args.repeats,
        "batch_sizes": BATCH_SIZES,
        "timing_scope": "x_scaler + all five neural models + y_scaler + ensemble mean/std; initialized checkpoints excluded",
        "input_distribution": "Repeated real cosmological parameter vectors from external-validation candidates",
        "warning": "External-validation candidates were selected by mixed acquisition/uncertainty/low_chi2/random strategies and are not iid. No same-node CLASS speedup is computed.",
    }
    (output / "environment.json").write_text(json.dumps(spec, indent=2, default=str))

    benchmark_rows, metric_rows, replay_rows = [], [], []
    for dataset in DATASETS:
        folder = root / "05_chi2_emulators" / dataset
        csvpath = root / "06_external_validation" / dataset / "cobaya_validation_results.csv"
        metadata = json.loads((folder / "metadata.json").read_text())
        features, targets = metadata["features"], metadata["targets"]
        if "chi2" not in targets:
            raise ValueError(f"Missing chi2 target for {dataset}")
        target_idx = targets.index("chi2")
        xs = joblib.load(folder / "x_scaler.joblib")  # Load only trusted local artifacts.
        ys = joblib.load(folder / "y_scaler.joblib")
        ensemble = []
        for k in range(1, 6):
            pth = folder / f"emulator_member_{k:02d}.pt"
            model = Chi2Emulator(len(features), len(targets))
            checkpoint = torch.load(pth, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint, strict=True)
            model.eval()
            ensemble.append(model)

        df = pd.read_csv(csvpath)
        if not set(features).issubset(df.columns):
            raise ValueError(f"Missing feature columns: {set(features) - set(df.columns)}")
        if not {"predicted_chi2", "real_chi2", "status"}.issubset(df.columns):
            raise ValueError(f"Missing validation columns in {csvpath}")
        ok = df.loc[df["status"].eq("OK")].copy()
        raw_x = ok[features].to_numpy(dtype=np.float64)
        valid = np.isfinite(raw_x).all(axis=1) & np.isfinite(ok["real_chi2"].to_numpy())
        ok = ok.loc[valid].copy()
        raw_x = raw_x[valid]
        if len(ok) < 2:
            raise ValueError(f"Insufficient valid points for {dataset}")

        pred, uncertainty = predict_raw(ensemble, xs, ys, raw_x)
        yhat = pred[:, target_idx].astype(float)
        sigma = uncertainty[:, target_idx].astype(float)
        exact = ok["real_chi2"].to_numpy(dtype=float)
        saved = ok["predicted_chi2"].to_numpy(dtype=float)
        residual = exact - yhat
        abs_err = np.abs(residual)
        r2_denom = np.square(exact - exact.mean()).sum()
        max_replay_delta = float(np.max(np.abs(yhat - saved)))
        metric_rows.append({
            "dataset": dataset, "n_valid": len(ok),
            "MAE_chi2": float(abs_err.mean()),
            "RMSE_chi2": float(np.sqrt(np.mean(residual ** 2))),
            "R2_chi2": float(1 - np.square(residual).sum() / r2_denom) if r2_denom else np.nan,
            "bias_real_minus_predicted": float(residual.mean()),
            "coverage_1sigma": float(np.mean(abs_err <= sigma)),
            "coverage_2sigma": float(np.mean(abs_err <= 2 * sigma)),
            "mean_ensemble_sigma": float(sigma.mean()),
            "max_abs_replay_delta_vs_stored_prediction": max_replay_delta,
        })
        for ix, (_, row) in enumerate(ok.iterrows()):
            replay_rows.append({
                "dataset": dataset,
                "candidate_id": row.get("candidate_id", ix + 1),
                "selection_source": row.get("selection_source", ""),
                "real_chi2": exact[ix], "saved_predicted_chi2": saved[ix],
                "replayed_predicted_chi2": yhat[ix],
                "ensemble_sigma": sigma[ix],
                "absolute_error": abs_err[ix],
            })

        # Reuse physically meaningful input vectors, repeat to achieve timing n.
        repeated_x = np.tile(raw_x, (int(np.ceil(args.n / len(raw_x))), 1))[:args.n]
        predict_raw(ensemble, xs, ys, repeated_x[:min(32, len(repeated_x))])
        print(f"\n{dataset}: {len(ok)} external points, replay max |delta|={max_replay_delta:.6g} chi2")
        for batch in BATCH_SIZES:
            timings = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                infer_timed(ensemble, xs, ys, repeated_x, batch)
                timings.append(time.perf_counter() - start)
            med = float(np.median(timings))
            benchmark_rows.append({
                "dataset": dataset, "batch_size": batch,
                "n_points": args.n, "repeats": args.repeats,
                "median_seconds_total": med,
                "latency_ms_per_point_amortized": med * 1000 / args.n,
                "throughput_points_per_second": args.n / med,
                "min_seconds_total": min(timings),
                "max_seconds_total": max(timings),
                "n_ensemble_members": 5,
                "torch_threads": args.threads,
                "host": platform.node(),
            })
            print(f"  batch={batch:4d}: {med:.4f}s/{args.n}, {args.n / med:,.0f} points/s")

    pd.DataFrame(benchmark_rows).to_csv(output / "ensemble_benchmark.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "ensemble_external_replay_metrics.csv", index=False)
    pd.DataFrame(replay_rows).to_csv(output / "ensemble_external_replay_points.csv", index=False)
    print("\n=== METRICS (RECOMPUTED ON SAVED EXACT REFERENCE) ===")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"\nSaved output directory: {output}")
    print("NOTE: Cobaya/CLASS not run; do NOT calculate cross-machine speedup.")


if __name__ == "__main__":
    main()
