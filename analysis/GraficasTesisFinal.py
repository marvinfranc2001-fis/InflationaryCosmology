#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-ready analysis of Starobinsky SR1/SR2 and ΛCDM+r with GetDist.

Expected inputs (ZIP archives or already-extracted directories):
    chains_poly.zip   : PolyChord chains for SR1 and SR2
    chains_mcmc.zip   : MCMC chains for SR1 and SR2
    chainslcdm.zip    : PolyChord chain for ΛCDM+r

The script produces:
  * direct MCMC-versus-PolyChord comparisons for SR1 and SR2;
  * GetDist triangle plots for inflationary and late-time parameters;
  * publication-style one-dimensional posterior densities;
  * an n_s-r plane with ΛCDM+r credible regions and Starobinsky predictions;
  * MCMC convergence diagnostics (R-1 and acceptance rate);
  * quantitative consistency diagnostics between samplers;
  * Bayesian evidences and Bayes factors from PolyChord;
  * CSV and LaTeX tables plus an automatically generated Markdown report;
  * likelihood and prior compatibility checks from the YAML files.

All numerical summaries use the complete post-burn-in samples. To avoid slow
kernel-density estimates, GetDist figures use a reproducible random subsample
of at most --max-plot-points samples per chain.

Quick start (place the three ZIP files in the working directory):
    python starobinsky_publication_analysis_en.py

Explicit usage:
    python starobinsky_publication_analysis_en.py \
      --poly-zip chains_poly.zip \
      --mcmc-zip chains_mcmc.zip \
      --lcdm-zip chainslcdm.zip \
      --out publication_analysis

Dependencies:
    pip install numpy pandas matplotlib getdist pyyaml
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import yaml
from getdist import MCSamples, plots


# -----------------------------------------------------------------------------
# PUBLICATION STYLE: DISTINCT GREEN, BLACK, AND GREY TONES
# -----------------------------------------------------------------------------

PALETTE = {
    # Reference model: neutral tones keep it visually separate from SR1/SR2.
    "lcdm": "#3F3F3F",        # charcoal
    "lcdm_fill": "#D8D8D8",   # light grey

    # SR1: forest-green family.
    "sr1_poly": "#0B3D2E",    # dark forest green
    "sr1_mcmc": "#4F9D69",    # medium leaf green

    # SR2: teal-green family, distinguishable from SR1 in print and on screen.
    "sr2_poly": "#00796B",    # dark teal green
    "sr2_mcmc": "#79CDBA",    # light teal green

    "accent": "#111111",      # black for reference lines and thresholds
    "grid": "#D7D7D7",        # neutral light-grey grid
    "text": "#111111",        # near-black text
    "muted": "#6A6A6A",
    "background": "#FFFFFF",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 11.5,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.linewidth": 1.0,
    "axes.edgecolor": PALETTE["text"],
    "axes.labelcolor": PALETTE["text"],
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
    "text.color": PALETTE["text"],
    "figure.facecolor": PALETTE["background"],
    "axes.facecolor": PALETTE["background"],
    "legend.frameon": False,
    "figure.dpi": 140,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

PARAM_LABELS = {
    "N_star": r"N_\star",
    "n_s": r"n_s",
    "r": r"r_{0.05}",
    "alpha_s": r"\alpha_s",
    "A_s": r"A_s",
    "logA": r"\ln(10^{10}A_s)",
    "H0": r"H_0\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]",
    "omega_b": r"\omega_b",
    "omega_cdm": r"\omega_{\mathrm{cdm}}",
    "tau_reio": r"\tau_{\mathrm{reio}}",
}

# Units and definitions are kept separate from compact axis labels so that
# triangle plots remain readable while exported tables remain unambiguous.
PARAM_UNITS = {
    "N_star": "dimensionless",
    "n_s": "dimensionless",
    "r": "dimensionless",
    "alpha_s": "dimensionless",
    "A_s": "dimensionless",
    "logA": "dimensionless",
    "H0": "km s^-1 Mpc^-1",
    "omega_b": "dimensionless",
    "omega_cdm": "dimensionless",
    "tau_reio": "dimensionless",
}

PARAM_DEFINITIONS = {
    "N_star": "Number of e-folds between pivot-scale horizon exit and the end of inflation",
    "n_s": "Scalar spectral index at k_pivot = 0.05 Mpc^-1",
    "r": "Tensor-to-scalar ratio at k_pivot = 0.05 Mpc^-1",
    "alpha_s": "Running of the scalar spectral index, d n_s / d ln k",
    "A_s": "Amplitude of scalar perturbations at k_pivot = 0.05 Mpc^-1",
    "logA": "Natural logarithm of 10^10 A_s",
    "H0": "Hubble constant",
    "omega_b": "Physical baryon density, Omega_b h^2",
    "omega_cdm": "Physical cold-dark-matter density, Omega_cdm h^2",
    "tau_reio": "Optical depth to reionization",
}

ANALYSIS_PARAMS = [
    "N_star", "n_s", "r", "alpha_s", "logA",
    "H0", "omega_b", "omega_cdm", "tau_reio",
]

MODEL_META = {
    "LCDM+r PolyChord": {
        "short": r"$\Lambda$CDM+$r$",
        "model": "lcdm",
        "sampler": "PolyChord",
        "color": PALETTE["lcdm"],
        "linestyle": "-",
        "label": r"$\Lambda$CDM+$r$ (PolyChord)",
    },
    "SR1 PolyChord": {
        "short": "SR1-PC",
        "model": "sr1",
        "sampler": "PolyChord",
        "color": PALETTE["sr1_poly"],
        "linestyle": "-",
        "label": "SR1 (PolyChord)",
    },
    "SR1 MCMC": {
        "short": "SR1-MCMC",
        "model": "sr1",
        "sampler": "MCMC",
        "color": PALETTE["sr1_mcmc"],
        "linestyle": "--",
        "label": "SR1 (MCMC)",
    },
    "SR2 PolyChord": {
        "short": "SR2-PC",
        "model": "sr2",
        "sampler": "PolyChord",
        "color": PALETTE["sr2_poly"],
        "linestyle": "-",
        "label": "SR2 (PolyChord)",
    },
    "SR2 MCMC": {
        "short": "SR2-MCMC",
        "model": "sr2",
        "sampler": "MCMC",
        "color": PALETTE["sr2_mcmc"],
        "linestyle": "--",
        "label": "SR2 (MCMC)",
    },
}


@dataclass
class ChainData:
    name: str
    root: Path
    frame: pd.DataFrame
    plot_sample: MCSamples
    yaml_path: Path | None


# -----------------------------------------------------------------------------
# COMMAND-LINE ARGUMENTS AND PATHS
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication-ready GetDist analysis of SR1, SR2, and ΛCDM+r",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--poly-zip", type=Path, default=Path("chains_poly.zip"))
    parser.add_argument("--mcmc-zip", type=Path, default=Path("chains_mcmc.zip"))
    parser.add_argument("--lcdm-zip", type=Path, default=Path("chainslcdm.zip"))
    parser.add_argument(
        "--base", type=Path, default=Path.cwd(),
        help="Alternative directory containing already-extracted chains.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("publication_analysis"),
        help="Output directory.",
    )
    parser.add_argument(
        "--burn-mcmc", type=float, default=0.30,
        help="Initial fraction removed independently from EACH MCMC chain (default: 0.30).",
    )
    parser.add_argument(
        "--max-plot-points", type=int, default=5000,
        help="Maximum number of samples per chain used for KDEs and GetDist figures.",
    )
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory used to extract and reuse ZIP contents (default: .getdist_chain_cache next to the output directory).",
    )
    parser.add_argument(
        "--no-png", action="store_true",
        help="Generate PDF files only (PNG files at 600 dpi are produced by default).",
    )
    return parser.parse_args()


def safe_extract_selected(zip_path: Path, destination: Path, kind: str) -> None:
    """Extract only required chains and metadata while preventing zip-slip."""
    if not zip_path.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)

    if kind in {"poly", "lcdm"}:
        keep = re.compile(
            r"(?:\.1\.txt|\.logZ|\.input\.yaml|\.updated\.yaml|\.stats|\.paramnames)$"
        )
    elif kind == "mcmc":
        keep = re.compile(
            r"(?:\.[0-9]+\.txt|\.input\.yaml|\.updated\.yaml|\.progress|\.checkpoint|\.covmat)$"
        )
    else:
        raise ValueError(f"Unrecognized ZIP type: {kind}")

    base_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if keep.search(m.filename)]
        if not members:
            raise RuntimeError(f"No usable chain files were found in {zip_path}")
        for member in members:
            target = (destination / member.filename).resolve()
            if base_resolved not in target.parents and target != base_resolved:
                raise RuntimeError(f"Unsafe path inside ZIP archive: {member.filename}")
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as dest:
                dest.write(source.read())


def find_one(search_roots: Iterable[Path], pattern: str, description: str) -> Path:
    matches: list[Path] = []
    for base in search_roots:
        if base.exists():
            matches.extend(base.rglob(pattern))
    matches = sorted({m.resolve() for m in matches})
    if not matches:
        roots_text = "\n".join(f"  - {p}" for p in search_roots)
        raise FileNotFoundError(
            f"Could not find {description} ({pattern}). Searched in:\n{roots_text}"
        )
    if len(matches) > 1:
        print(f"[WARNING] Multiple matches found for {description}; using {matches[0]}")
    return matches[0]


def strip_chain_suffix(path: Path) -> Path:
    suffix = ".1.txt"
    text = str(path)
    if not text.endswith(suffix):
        raise ValueError(path)
    return Path(text[:-len(suffix)])


def prepare_roots(args: argparse.Namespace) -> dict[str, Path]:
    cache = args.cache_dir or (args.out.parent / ".getdist_chain_cache")
    safe_extract_selected(args.poly_zip, cache / "poly", "poly")
    safe_extract_selected(args.mcmc_zip, cache / "mcmc", "mcmc")
    safe_extract_selected(args.lcdm_zip, cache / "lcdm", "lcdm")

    search_roots = [cache, args.base]
    files = {
        "LCDM+r PolyChord": find_one(
            search_roots, "polychord_starobinsky_lcdm.1.txt", "ΛCDM+r PolyChord"
        ),
        "SR1 PolyChord": find_one(
            search_roots, "polychord_starobinsky_SR1_300.1.txt", "SR1 PolyChord"
        ),
        "SR2 PolyChord": find_one(
            search_roots, "polychord_starobinsky_SR2_300_MPI.1.txt", "SR2 PolyChord"
        ),
        "SR1 MCMC": find_one(
            search_roots, "mcmc_starobinsky_SR1_CORREGIDO.1.txt", "SR1 MCMC"
        ),
        "SR2 MCMC": find_one(
            search_roots, "mcmc_starobinsky_SR2_CORREGIDO.1.txt", "SR2 MCMC"
        ),
    }
    return {name: strip_chain_suffix(path) for name, path in files.items()}


# -----------------------------------------------------------------------------
# CHAIN READING AND DERIVED PARAMETERS
# -----------------------------------------------------------------------------

def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = handle.readline()
    if not first.startswith("#"):
        raise ValueError(f"Cobaya header not found in {path}")
    return first.lstrip("#").split()


def selected_columns(header: list[str]) -> list[str]:
    wanted = [
        "weight", "minuslogpost", "chi2",
        "N_star", "n_s", "r", "alpha_s", "A_s",
        "H0", "omega_b", "omega_cdm", "tau_reio",
    ]
    missing = [c for c in ["weight", "minuslogpost"] if c not in header]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    return [c for c in wanted if c in header]


def read_cobaya_file(path: Path) -> pd.DataFrame:
    header = read_header(path)
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=header,
        usecols=selected_columns(header),
        engine="c",
    )


def add_derived(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    frame = frame.copy()

    if "A_s" in frame and "logA" not in frame:
        positive = frame["A_s"].to_numpy(dtype=float) > 0
        if not np.all(positive):
            raise ValueError("A_s contains non-positive values")
        frame["logA"] = np.log(1.0e10 * frame["A_s"].to_numpy(dtype=float))

    if model == "lcdm":
        return frame

    if "N_star" not in frame:
        raise ValueError(f"The {model} chain does not contain N_star")

    n = frame["N_star"].to_numpy(dtype=float)
    if model == "sr1":
        # Reproduce exactly the parameterization used in the YAML runs.
        if "n_s" not in frame:
            frame["n_s"] = 1.0 - 2.0 / n - 4.5 / n**2
        if "r" not in frame:
            frame["r"] = 12.0 / n**2
        if "alpha_s" not in frame:
            # In SR1, alpha_s was fixed to zero in the sampled runs.
            frame["alpha_s"] = 0.0
    elif model == "sr2":
        if "n_s" not in frame:
            frame["n_s"] = 1.0 - 2.0 / n - 1.5 / n**2
        if "r" not in frame:
            frame["r"] = (12.0 / n**2) * (1.0 - 2.0 / n)
        if "alpha_s" not in frame:
            frame["alpha_s"] = -2.0 / n**2 - 3.0 / n**3
    else:
        raise ValueError(model)
    return frame


def numeric_chain_files(root: Path) -> list[Path]:
    regex = re.compile(re.escape(root.name) + r"\.([0-9]+)\.txt$")
    indexed: list[tuple[int, Path]] = []
    for path in root.parent.glob(root.name + ".*.txt"):
        match = regex.match(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    return [path for _, path in sorted(indexed)]

def load_polychord(root: Path, model: str) -> pd.DataFrame:
    path = Path(str(root) + ".1.txt")
    frame = read_cobaya_file(path)
    frame["chain"] = 1
    return add_derived(frame, model)


def load_mcmc(root: Path, model: str, burn_fraction: float) -> pd.DataFrame:
    files = numeric_chain_files(root)
    if not files:
        raise FileNotFoundError(f"No MCMC chain files were found for {root}")
    frames: list[pd.DataFrame] = []
    for chain_id, path in enumerate(files, start=1):
        frame = read_cobaya_file(path)
        cut = int(math.floor(len(frame) * burn_fraction))
        if cut >= len(frame):
            raise ValueError(f"The burn-in fraction removes all samples from {path}")
        frame = frame.iloc[cut:].copy()
        frame["chain"] = chain_id
        frames.append(frame)
    return add_derived(pd.concat(frames, ignore_index=True), model)


def yaml_for_root(root: Path) -> Path | None:
    candidates = [
        Path(str(root) + ".updated.yaml"),
        Path(str(root) + ".input.yaml"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# -----------------------------------------------------------------------------
# GETDIST: REPRODUCIBLE SUBSAMPLING FOR FIGURES
# -----------------------------------------------------------------------------

def deterministic_subsample(
    frame: pd.DataFrame, max_points: int, rng: np.random.Generator
) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame.copy()

    # For MCMC, preserve each chain proportionally. For PolyChord (a single
    # weighted chain), sample randomly while retaining the original weights.
    if "chain" in frame and frame["chain"].nunique() > 1:
        pieces = []
        counts = frame["chain"].value_counts().sort_index()
        allocation = np.maximum(
            1, np.floor(max_points * counts / counts.sum()).astype(int)
        )
        # Correct rounding so that the requested maximum is not exceeded.
        while allocation.sum() > max_points:
            idx = allocation.idxmax()
            if allocation.loc[idx] > 1:
                allocation.loc[idx] -= 1
            else:
                break
        for chain_id, n_take in allocation.items():
            group = frame[frame["chain"] == chain_id]
            choice = rng.choice(len(group), size=min(n_take, len(group)), replace=False)
            pieces.append(group.iloc[np.sort(choice)])
        return pd.concat(pieces, ignore_index=True)

    choice = rng.choice(len(frame), size=max_points, replace=False)
    return frame.iloc[np.sort(choice)].copy()


def make_getdist_sample(
    name: str,
    frame: pd.DataFrame,
    max_points: int,
    seed: int,
) -> MCSamples:
    meta = MODEL_META[name]
    rng = np.random.default_rng(seed + sum(ord(c) for c in name))
    reduced = deterministic_subsample(frame, max_points, rng)

    names = []
    for param in ANALYSIS_PARAMS:
        if param not in reduced:
            continue
        values = reduced[param].to_numpy(dtype=float)
        # GetDist removes exactly fixed parameters, so they are excluded here.
        if np.nanstd(values) <= max(1e-15, 1e-12 * max(1.0, abs(np.nanmean(values)))):
            continue
        names.append(param)

    samples = reduced[names].to_numpy(dtype=float)
    weights = reduced["weight"].to_numpy(dtype=float)
    loglikes = reduced["minuslogpost"].to_numpy(dtype=float)

    gd = MCSamples(
        samples=samples,
        weights=weights,
        loglikes=loglikes,
        names=names,
        labels=[PARAM_LABELS[p] for p in names],
        label=meta["label"],
        settings={
            "ignore_rows": 0,
            "smooth_scale_1D": 0.38,
            "smooth_scale_2D": 0.48,
            "boundary_correction_order": 1,
            "mult_bias_correction_order": 1,
        },
    )
    if "r" in names:
        gd.setRanges({"r": [0.0, None]})
    return gd


def has_param(sample: MCSamples, param: str) -> bool:
    return sample.getParamNames().parWithName(param) is not None


def new_plotter(width: float = 8.0):
    g = plots.get_subplot_plotter(width_inch=width)
    settings = g.settings
    settings.figure_legend_frame = False
    settings.alpha_filled_add = 0.70
    settings.linewidth = 1.8
    settings.axes_fontsize = 10
    settings.lab_fontsize = 13
    settings.legend_fontsize = 9.5
    settings.num_plot_contours = 2
    settings.solid_contour_palefactor = 0.55
    settings.axis_marker_lw = 1.0
    settings.axis_marker_color = PALETTE["accent"]
    settings.colorbar_label_pad = 8
    return g


def export_getdist(g, out: Path, stem: str, make_png: bool) -> None:
    pdf = out / f"{stem}.pdf"
    g.export(str(pdf))
    if make_png:
        png = out / f"{stem}.png"
        g.export(str(png), dpi=600)
    plt.close(g.fig)


def save_matplotlib(fig, out: Path, stem: str, make_png: bool) -> None:
    fig.savefig(out / f"{stem}.pdf", metadata={
        "Title": stem,
        "Author": "Reproducible GetDist analysis",
        "Subject": "Starobinsky SR1/SR2 and LambdaCDM+r",
    })
    if make_png:
        fig.savefig(out / f"{stem}.png", dpi=600)
    plt.close(fig)


# -----------------------------------------------------------------------------
# STATISTICS FROM ALL POST-BURN-IN SAMPLES
# -----------------------------------------------------------------------------

def weighted_quantile(values, quantiles, weights) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0 or weights.sum() <= 0:
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= weights.sum()
    return np.interp(quantiles, cdf, values)


def summarize_frame(name: str, frame: pd.DataFrame) -> list[dict]:
    """Compute weighted posterior summaries using all retained samples."""
    rows = []
    weights = frame["weight"].to_numpy(dtype=float)
    for param in ANALYSIS_PARAMS:
        if param not in frame:
            continue
        values = frame[param].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
        values = values[valid]
        w = weights[valid]
        if len(values) == 0 or w.sum() <= 0:
            continue
        mean = np.average(values, weights=w)
        variance = np.average((values - mean) ** 2, weights=w)
        q025, q16, q50, q84, q975 = weighted_quantile(
            values, [0.025, 0.16, 0.50, 0.84, 0.975], w
        )
        rows.append({
            "chain": name,
            "model": MODEL_META[name]["model"],
            "sampler": MODEL_META[name]["sampler"],
            "parameter": param,
            "latex_label": PARAM_LABELS[param],
            "unit": PARAM_UNITS[param],
            "definition": PARAM_DEFINITIONS[param],
            "mean": mean,
            "std": math.sqrt(max(variance, 0.0)),
            "q2.5": q025,
            "q16": q16,
            "median": q50,
            "q84": q84,
            "q97.5": q975,
            "err68_minus": q50 - q16,
            "err68_plus": q84 - q50,
            "err95_minus": q50 - q025,
            "err95_plus": q975 - q50,
            "n_rows": len(frame),
            "weight_sum": w.sum(),
        })
    return rows

def histogram_agreement(
    a: np.ndarray, wa: np.ndarray, b: np.ndarray, wb: np.ndarray, bins: int = 100
) -> tuple[float, float]:
    """Return Jensen-Shannon divergence in bits and histogram overlap in [0, 1]."""
    lo = min(
        weighted_quantile(a, [0.0025], wa)[0],
        weighted_quantile(b, [0.0025], wb)[0],
    )
    hi = max(
        weighted_quantile(a, [0.9975], wa)[0],
        weighted_quantile(b, [0.9975], wb)[0],
    )
    if not np.isfinite(lo + hi) or hi <= lo:
        return 0.0, 1.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, weights=wa)
    pb, _ = np.histogram(b, bins=edges, weights=wb)
    pa = pa.astype(float)
    pb = pb.astype(float)
    pa /= pa.sum()
    pb /= pb.sum()
    eps = 1e-300
    m = 0.5 * (pa + pb)
    js_nats = 0.5 * np.sum(pa * np.log((pa + eps) / (m + eps)))
    js_nats += 0.5 * np.sum(pb * np.log((pb + eps) / (m + eps)))
    overlap = np.minimum(pa, pb).sum()
    return js_nats / math.log(2.0), overlap


def sampler_consistency(
    model: str, poly: pd.DataFrame, mcmc: pd.DataFrame
) -> list[dict]:
    """Compare PolyChord and MCMC one-dimensional marginal distributions."""
    rows = []
    common = [p for p in ANALYSIS_PARAMS if p in poly and p in mcmc]
    wp = poly["weight"].to_numpy(dtype=float)
    wm = mcmc["weight"].to_numpy(dtype=float)
    for param in common:
        xp = poly[param].to_numpy(dtype=float)
        xm = mcmc[param].to_numpy(dtype=float)
        mup = np.average(xp, weights=wp)
        mum = np.average(xm, weights=wm)
        sigp = math.sqrt(np.average((xp - mup) ** 2, weights=wp))
        sigm = math.sqrt(np.average((xm - mum) ** 2, weights=wm))
        denom = math.sqrt(sigp**2 + sigm**2)
        z = abs(mup - mum) / denom if denom > 0 else 0.0
        js_bits, overlap = histogram_agreement(xp, wp, xm, wm)
        rows.append({
            "model": model.upper(),
            "parameter": param,
            "unit": PARAM_UNITS[param],
            "mean_polychord": mup,
            "std_polychord": sigp,
            "mean_mcmc": mum,
            "std_mcmc": sigm,
            "mean_difference": mum - mup,
            "combined_sigma_shift": z,
            "js_divergence_bits": js_bits,
            "histogram_overlap": overlap,
        })
    return rows

# -----------------------------------------------------------------------------
# BAYESIAN EVIDENCE AND CONVERGENCE
# -----------------------------------------------------------------------------

def read_logz(root: Path) -> tuple[float, float]:
    path = Path(str(root) + ".logZ")
    if not path.exists():
        # Fall back to the PolyChord .stats file.
        raw = root.parent / f"{root.name}_polychord_raw" / f"{root.name}.stats"
        if not raw.exists():
            raise FileNotFoundError(f"Neither .logZ nor .stats was found for {root}")
        text = raw.read_text(errors="ignore")
        match = re.search(
            r"log\(Z\)\s*=\s*([+\-0-9.Ee]+)\s*\+/-\s*([+\-0-9.Ee]+)", text
        )
        if not match:
            raise ValueError(f"Could not parse logZ from {raw}")
        return float(match.group(1)), float(match.group(2))

    text = path.read_text(errors="ignore")
    z = re.search(r"logZ:\s*([+\-0-9.Ee]+)", text)
    s = re.search(r"logZstd:\s*([+\-0-9.Ee]+)", text)
    if not z or not s:
        raise ValueError(f"Unrecognized evidence-file format: {path}")
    return float(z.group(1)), float(s.group(1))


def evidence_table(roots: dict[str, Path]) -> pd.DataFrame:
    """Build absolute-evidence and pairwise Bayes-factor summaries."""
    base_names = ["LCDM+r PolyChord", "SR1 PolyChord", "SR2 PolyChord"]
    rows = []
    values = {}
    for name in base_names:
        logz, sigma = read_logz(roots[name])
        values[name] = (logz, sigma)
        rows.append({
            "comparison": name,
            "logZ": logz,
            "logZ_std": sigma,
            "delta_logZ": np.nan,
            "delta_logZ_std": np.nan,
            "Bayes_factor": np.nan,
            "reference": "",
        })

    comparisons = [
        ("SR1 vs LCDM+r", "SR1 PolyChord", "LCDM+r PolyChord"),
        ("SR2 vs LCDM+r", "SR2 PolyChord", "LCDM+r PolyChord"),
        ("SR2 vs SR1", "SR2 PolyChord", "SR1 PolyChord"),
    ]
    for label, numerator, denominator in comparisons:
        za, sa = values[numerator]
        zb, sb = values[denominator]
        delta = za - zb
        sigma = math.sqrt(sa**2 + sb**2)
        rows.append({
            "comparison": label,
            "logZ": np.nan,
            "logZ_std": np.nan,
            "delta_logZ": delta,
            "delta_logZ_std": sigma,
            "Bayes_factor": math.exp(delta),
            "reference": denominator,
        })
    return pd.DataFrame(rows)

def read_progress(root: Path) -> pd.DataFrame | None:
    path = Path(str(root) + ".progress")
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline().lstrip("#").split()
    return pd.read_csv(
        path, sep=r"\s+", comment="#", names=header, engine="python"
    )


def checkpoint_info(root: Path) -> dict:
    path = Path(str(root) + ".checkpoint")
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("sampler", {}).get("mcmc", {}) or {}


# -----------------------------------------------------------------------------
# LIKELIHOOD AND PRIOR COMPATIBILITY
# -----------------------------------------------------------------------------

def load_yaml(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def normalize_likelihood(name: str) -> str:
    # TTTEEE_lite_native is the Python implementation of the same Planck
    # Plik-lite likelihood used by TTTEEE_lite/CLIK.
    return name.replace("TTTEEE_lite_native", "TTTEEE_lite")


def normalized_likelihoods(doc: dict) -> list[str]:
    like = doc.get("likelihood", {}) or {}
    return sorted(normalize_likelihood(str(key)) for key in like.keys())


def sampled_prior_signature(doc: dict) -> dict:
    signature = {}
    params = doc.get("params", {}) or {}
    for name, specification in params.items():
        if not isinstance(specification, dict):
            continue
        if "prior" in specification:
            signature[name] = specification.get("prior")
    return signature


def compatibility_report(chains: dict[str, ChainData], out: Path) -> dict:
    """Check whether sampler and evidence comparisons use compatible inputs."""
    docs = {name: load_yaml(chain.yaml_path) for name, chain in chains.items()}
    likes = {name: normalized_likelihoods(doc) for name, doc in docs.items()}
    priors = {name: sampled_prior_signature(doc) for name, doc in docs.items()}

    checks = {}
    for model, pc_name, mc_name in [
        ("SR1", "SR1 PolyChord", "SR1 MCMC"),
        ("SR2", "SR2 PolyChord", "SR2 MCMC"),
    ]:
        common_prior_names = sorted(set(priors[pc_name]) & set(priors[mc_name]))
        prior_equal = {
            parameter: priors[pc_name].get(parameter) == priors[mc_name].get(parameter)
            for parameter in common_prior_names
        }
        checks[model] = {
            "polychord_likelihoods": likes[pc_name],
            "mcmc_likelihoods": likes[mc_name],
            "likelihoods_equivalent_after_normalization": likes[pc_name] == likes[mc_name],
            "common_priors_equal": all(prior_equal.values()),
            "prior_details": prior_equal,
        }
    poly_likes = {
        name: likes[name]
        for name in ["LCDM+r PolyChord", "SR1 PolyChord", "SR2 PolyChord"]
    }
    checks["polychord_evidence"] = {
        "likelihoods": poly_likes,
        "same_data_combination": len({tuple(v) for v in poly_likes.values()}) == 1,
    }

    (out / "likelihood_prior_compatibility.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "LIKELIHOOD AND PRIOR COMPATIBILITY CHECK",
        "=" * 72,
        "TTTEEE_lite_native is normalized to TTTEEE_lite because it is the",
        "native Python implementation of the same Planck Plik-lite likelihood.",
        "",
    ]
    for model in ["SR1", "SR2"]:
        item = checks[model]
        lines.extend([
            f"{model}:",
            f"  Equivalent likelihoods: {item['likelihoods_equivalent_after_normalization']}",
            f"  Equal common priors:     {item['common_priors_equal']}",
            f"  PolyChord: {item['polychord_likelihoods']}",
            f"  MCMC:      {item['mcmc_likelihoods']}",
            "",
        ])
    lines.extend([
        "PolyChord evidence comparison:",
        f"  Same likelihood set: {checks['polychord_evidence']['same_data_combination']}",
    ])
    (out / "likelihood_prior_compatibility.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return checks

# -----------------------------------------------------------------------------
# GETDIST FIGURES AND DIAGNOSTICS
# -----------------------------------------------------------------------------

def style_lists(names: list[str]):
    colors = [MODEL_META[n]["color"] for n in names]
    line_args = [
        {"color": MODEL_META[n]["color"], "ls": MODEL_META[n]["linestyle"], "lw": 2.0}
        for n in names
    ]
    return colors, line_args


def plot_individual_posteriors(
    chains: dict[str, ChainData], out: Path, make_png: bool
) -> None:
    """Plot normalized one-dimensional marginal posterior densities."""
    preferred_order = [
        "LCDM+r PolyChord", "SR1 PolyChord", "SR1 MCMC",
        "SR2 PolyChord", "SR2 MCMC",
    ]
    for param in ANALYSIS_PARAMS:
        names = [n for n in preferred_order if has_param(chains[n].plot_sample, param)]
        if not names:
            continue
        samples = [chains[n].plot_sample for n in names]
        colors, line_args = style_lists(names)
        g = new_plotter(width=6.8)
        g.plot_1d(
            samples,
            param,
            normalized=True,
            colors=colors,
            line_args=line_args,
        )
        g.add_legend(
            [MODEL_META[n]["label"] for n in names],
            legend_loc="upper right",
            colored_text=True,
        )
        ax = g.subplots[0, 0]
        ax.grid(True, color=PALETTE["grid"], alpha=0.60, linewidth=0.7)
        ax.set_ylabel("Normalized posterior density")
        ax.set_axisbelow(True)
        export_getdist(g, out, f"posterior_1d_{param}", make_png)

def triangle_sampler_validation(
    chains: dict[str, ChainData], out: Path, make_png: bool
) -> None:
    configurations = [
        (
            "SR1",
            ["SR1 PolyChord", "SR1 MCMC"],
            ["N_star", "H0", "omega_b", "omega_cdm"],
        ),
        (
            "SR2",
            ["SR2 PolyChord", "SR2 MCMC"],
            ["N_star", "H0", "omega_b", "omega_cdm"],
        ),
    ]
    for model, names, desired in configurations:
        params = [
            p for p in desired
            if all(has_param(chains[n].plot_sample, p) for n in names)
        ]
        samples = [chains[n].plot_sample for n in names]
        colors, line_args = style_lists(names)
        g = new_plotter(width=10.0)
        g.triangle_plot(
            samples,
            params,
            filled=[True, False],
            contour_colors=colors,
            contour_ls=["-", "--"],
            contour_lws=[1.8, 2.0],
            line_args=line_args,
            legend_labels=[MODEL_META[n]["label"] for n in names],
            legend_loc="upper right",
        )
        export_getdist(g, out, f"triangle_sampler_validation_{model}", make_png)


def triangle_all_samplers(
    chains: dict[str, ChainData], out: Path, make_png: bool
) -> None:
    names = ["SR1 PolyChord", "SR1 MCMC", "SR2 PolyChord", "SR2 MCMC"]
    desired = ["N_star", "H0", "omega_b", "omega_cdm"]
    params = [
        p for p in desired if all(has_param(chains[n].plot_sample, p) for n in names)
    ]
    colors, line_args = style_lists(names)
    g = new_plotter(width=10.5)
    g.triangle_plot(
        [chains[n].plot_sample for n in names],
        params,
        filled=[True, False, True, False],
        contour_colors=colors,
        contour_ls=["-", "--", "-", "--"],
        contour_lws=[1.7, 1.9, 1.7, 1.9],
        line_args=line_args,
        legend_labels=[MODEL_META[n]["label"] for n in names],
        legend_ncol=2,
        legend_loc="upper right",
    )
    export_getdist(g, out, "triangle_sampler_comparison_SR1_SR2", make_png)


def triangle_polychord_models(
    chains: dict[str, ChainData], out: Path, make_png: bool
) -> None:
    names = ["LCDM+r PolyChord", "SR1 PolyChord", "SR2 PolyChord"]
    desired = ["n_s", "r", "H0", "omega_cdm"]
    params = [
        p for p in desired if all(has_param(chains[n].plot_sample, p) for n in names)
    ]
    colors, line_args = style_lists(names)

    # Figure 5.3 is inserted into the thesis at roughly text width, so the
    # exported triangle plot is scaled down substantially by LaTeX.  Use a
    # slightly smaller canvas and deliberately larger text so axis labels and
    # numerical tick labels remain readable in the final PDF.
    g = new_plotter(width=8.5)
    g.settings.axes_fontsize = 15.0    # numerical tick labels
    g.settings.lab_fontsize = 18.0     # parameter labels (n_s, r_0.05, H_0, ...)
    g.settings.legend_fontsize = 13.5  # model legend
    g.settings.linewidth = 2.0

    g.triangle_plot(
        [chains[n].plot_sample for n in names],
        params,
        filled=True,
        contour_colors=colors,
        contour_lws=[1.8, 2.0, 2.0],
        line_args=line_args,
        legend_labels=[MODEL_META[n]["label"] for n in names],
        legend_loc="upper right",
    )

    # Explicitly reinforce tick/axis-label sizes after GetDist creates all
    # subplots. This avoids small labels after the figure is scaled in LaTeX.
    for ax in np.asarray(g.subplots, dtype=object).ravel():
        if ax is not None:
            ax.tick_params(axis="both", which="major", labelsize=15)
            ax.xaxis.label.set_size(18)
            ax.yaxis.label.set_size(18)

    export_getdist(g, out, "triangle_polychord_models_LCDM_SR1_SR2", make_png)


def plot_ns_r_plane(
    chains: dict[str, ChainData], out: Path, make_png: bool, seed: int
) -> None:
    """Plot the primordial n_s-r plane at k_pivot = 0.05 Mpc^-1."""
    # GetDist draws the free ΛCDM+r credible regions.
    lcdm = chains["LCDM+r PolyChord"].plot_sample
    g = new_plotter(width=7.4)
    g.plot_2d(
        lcdm,
        "n_s",
        "r",
        filled=True,
        colors=[PALETTE["lcdm"]],
        contour_colors=[PALETTE["lcdm"]],
        add_legend_proxy=False,
    )
    ax = g.subplots[0, 0]

    # Theoretical Starobinsky trajectories over the adopted N_star prior.
    n_grid = np.linspace(50.0, 60.0, 600)
    ns1 = 1.0 - 2.0 / n_grid - 4.5 / n_grid**2
    r1 = 12.0 / n_grid**2
    ns2 = 1.0 - 2.0 / n_grid - 1.5 / n_grid**2
    r2 = (12.0 / n_grid**2) * (1.0 - 2.0 / n_grid)
    ax.plot(ns1, r1, color=PALETTE["sr1_poly"], lw=2.5, label="SR1 prediction")
    ax.plot(ns2, r2, color=PALETTE["sr2_poly"], lw=2.5, label="SR2 prediction")

    rng = np.random.default_rng(seed + 731)
    scatter_specs = [
        ("SR1 PolyChord", "o", 13, 0.24),
        ("SR1 MCMC", "x", 13, 0.30),
        ("SR2 PolyChord", "o", 13, 0.24),
        ("SR2 MCMC", "x", 13, 0.30),
    ]
    for name, marker, size, alpha in scatter_specs:
        frame = chains[name].frame
        n_take = min(1800, len(frame))
        idx = rng.choice(len(frame), size=n_take, replace=False)
        ax.scatter(
            frame.iloc[idx]["n_s"], frame.iloc[idx]["r"],
            s=size, marker=marker, alpha=alpha,
            color=MODEL_META[name]["color"], rasterized=True,
            linewidths=0.8,
            label=MODEL_META[name]["label"],
        )

    for n_value in [50, 55, 60]:
        ns1v = 1.0 - 2.0 / n_value - 4.5 / n_value**2
        r1v = 12.0 / n_value**2
        ns2v = 1.0 - 2.0 / n_value - 1.5 / n_value**2
        r2v = (12.0 / n_value**2) * (1.0 - 2.0 / n_value)
        ax.annotate(
            rf"$N_\star={n_value}$", (ns1v, r1v), xytext=(4, 3),
            textcoords="offset points", fontsize=7.6,
            color=PALETTE["sr1_poly"],
        )
        ax.annotate(
            rf"$N_\star={n_value}$", (ns2v, r2v), xytext=(4, -10),
            textcoords="offset points", fontsize=7.6,
            color=PALETTE["sr2_poly"],
        )

    ax.set_xlim(0.94, 0.985)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$n_s$ at $k_\mathrm{pivot}=0.05\,\mathrm{Mpc}^{-1}$")
    ax.set_ylabel(r"$r_{0.05}$")
    ax.set_title(r"Primordial constraints in the $n_s$--$r$ plane")
    ax.grid(True, color=PALETTE["grid"], alpha=0.60, linewidth=0.7)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    lcdm_label = r"$\Lambda$CDM+$r$: 68% and 95% credible regions"
    handles.insert(
        0,
        Patch(
            facecolor=PALETTE["lcdm_fill"],
            edgecolor=PALETTE["lcdm"],
            label=lcdm_label,
        ),
    )
    labels.insert(0, lcdm_label)

    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(
        unique.values(), unique.keys(), fontsize=8.0,
        loc="upper right", ncol=1,
    )
    export_getdist(g, out, "ns_r_plane_LCDM_SR1_SR2", make_png)

def plot_consistency(consistency: pd.DataFrame, out: Path, make_png: bool) -> None:
    """Plot normalized shifts between MCMC and PolyChord posterior means."""
    params = [
        p for p in ANALYSIS_PARAMS
        if p in consistency["parameter"].unique()
    ]
    y = np.arange(len(params))
    fig, ax = plt.subplots(figsize=(7.4, max(4.2, 0.48 * len(params))))
    width = 0.34
    for offset, model, color in [
        (-width / 2, "SR1", PALETTE["sr1_poly"]),
        (+width / 2, "SR2", PALETTE["sr2_poly"]),
    ]:
        subset = consistency[consistency["model"] == model].set_index("parameter")
        values = [
            subset.loc[p, "combined_sigma_shift"] if p in subset.index else np.nan
            for p in params
        ]
        ax.barh(y + offset, values, height=width, color=color, alpha=0.92, label=model)

    ax.axvline(0.1, color=PALETTE["muted"], lw=1.2, ls=":", label=r"$0.1\sigma$")
    ax.axvline(0.5, color=PALETTE["accent"], lw=1.2, ls="--", label=r"$0.5\sigma$")
    ax.set_yticks(y)
    ax.set_yticklabels([f"${PARAM_LABELS[p]}$" for p in params])
    ax.set_xlabel(
        r"$|\mu_{\rm MCMC}-\mu_{\rm PolyChord}|/"
        r"\sqrt{\sigma^2_{\rm MCMC}+\sigma^2_{\rm PolyChord}}$"
    )
    ax.set_title("Posterior consistency between MCMC and PolyChord")
    ax.grid(axis="x", color=PALETTE["grid"], alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="lower right")
    save_matplotlib(fig, out, "sampler_consistency_MCMC_vs_PolyChord", make_png)

def plot_evidence(evidence: pd.DataFrame, out: Path, make_png: bool) -> None:
    """Plot evidence differences relative to the ΛCDM+r reference model."""
    base = evidence[evidence["logZ"].notna()].copy()
    order = ["LCDM+r PolyChord", "SR1 PolyChord", "SR2 PolyChord"]
    base = base.set_index("comparison").loc[order].reset_index()
    reference = float(
        base.loc[base["comparison"] == "LCDM+r PolyChord", "logZ"].iloc[0]
    )
    ref_sigma = float(
        base.loc[base["comparison"] == "LCDM+r PolyChord", "logZ_std"].iloc[0]
    )
    base["delta"] = base["logZ"] - reference
    base["delta_sigma"] = np.sqrt(base["logZ_std"]**2 + ref_sigma**2)
    base.loc[base["comparison"] == "LCDM+r PolyChord", "delta_sigma"] = 0.0

    colors = [PALETTE["lcdm"], PALETTE["sr1_poly"], PALETTE["sr2_poly"]]
    labels = [r"$\Lambda$CDM+$r$", "SR1", "SR2"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(base))
    bars = ax.bar(x, base["delta"], color=colors, alpha=0.95, width=0.67)
    ax.errorbar(
        x, base["delta"], yerr=base["delta_sigma"], fmt="none",
        ecolor=PALETTE["accent"], capsize=4, lw=1.2,
    )
    ax.axhline(0.0, color=PALETTE["accent"], lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta\ln Z$ relative to $\Lambda$CDM+$r$")
    ax.set_title("Bayesian evidence comparison (PolyChord)")
    ax.grid(axis="y", color=PALETTE["grid"], alpha=0.75)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, base["delta"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.2f}",
            ha="center", va="bottom", fontsize=10,
        )
    save_matplotlib(fig, out, "bayesian_evidence", make_png)

def plot_convergence(
    roots: dict[str, Path], out: Path, make_png: bool
) -> pd.DataFrame:
    """Plot Gelman-Rubin convergence and MCMC acceptance-rate histories."""
    rows = []
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax2 = ax.twinx()
    for name, model_label, color in [
        ("SR1 MCMC", "SR1", PALETTE["sr1_mcmc"]),
        ("SR2 MCMC", "SR2", PALETTE["sr2_mcmc"]),
    ]:
        progress = read_progress(roots[name])
        checkpoint = checkpoint_info(roots[name])
        if progress is None:
            continue
        ax.plot(
            progress["N"], progress["Rminus1"],
            color=color, lw=2.2, label=rf"{model_label}: $R-1$",
        )
        ax2.plot(
            progress["N"], progress["acceptance_rate"],
            color=color, lw=1.4, ls=":", alpha=0.90,
            label=f"{model_label}: acceptance rate",
        )
        rows.append({
            "chain": name,
            "converged": bool(checkpoint.get("converged", False)),
            "final_Rminus1": float(checkpoint.get("Rminus1_last", np.nan)),
            "mpi_size": checkpoint.get("mpi_size", np.nan),
            "final_N": float(progress["N"].iloc[-1]),
            "final_acceptance_rate": float(progress["acceptance_rate"].iloc[-1]),
        })

    ax.axhline(
        0.01, color=PALETTE["accent"], ls="--", lw=1.2,
        label=r"Convergence criterion: $R-1=0.01$",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative MCMC samples")
    ax.set_ylabel(r"$R-1$")
    ax2.set_ylabel("Acceptance rate")
    ax2.set_ylim(0.0, 1.0)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("MCMC convergence diagnostics")
    ax.grid(True, color=PALETTE["grid"], alpha=0.70, which="both")
    ax.set_axisbelow(True)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.4, ncol=2, loc="upper right")
    save_matplotlib(fig, out, "mcmc_convergence", make_png)
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# LATEX TABLES AND AUTOMATED REPORT
# -----------------------------------------------------------------------------

def latex_number(value: float, err_minus: float, err_plus: float) -> str:
    if not np.isfinite(value):
        return "--"
    scale = abs(value)
    if scale != 0 and (scale < 1e-3 or scale >= 1e4):
        exponent = int(math.floor(math.log10(scale)))
        factor = 10.0**exponent
        return (
            rf"$({value/factor:.4g}_{{-{err_minus/factor:.2g}}}"
            rf"^{{+{err_plus/factor:.2g}}})\times10^{{{exponent}}}$"
        )
    return rf"${value:.6g}_{{-{err_minus:.2g}}}^{{+{err_plus:.2g}}}$"


def write_latex_tables(summary: pd.DataFrame, evidence: pd.DataFrame, out: Path) -> None:
    """Export publication-ready posterior and evidence tables."""
    table = summary.copy()
    table["result_68"] = table.apply(
        lambda row: latex_number(
            row["median"], row["err68_minus"], row["err68_plus"]
        ),
        axis=1,
    )
    wide = table.pivot(
        index=["latex_label", "unit"],
        columns="chain",
        values="result_68",
    )
    preferred = [
        "LCDM+r PolyChord", "SR1 PolyChord", "SR1 MCMC",
        "SR2 PolyChord", "SR2 MCMC",
    ]
    wide = wide.reindex(columns=[column for column in preferred if column in wide.columns])
    wide.index.names = ["Parameter", "Unit"]
    wide.to_csv(out / "posterior_constraints_latex_ready.csv")

    latex = wide.to_latex(
        escape=False,
        na_rep="--",
        column_format="ll" + "c" * len(wide.columns),
        caption=(
            "Posterior constraints. The median and central 68\\% credible "
            "interval are reported. All quantities are dimensionless unless "
            "a unit is explicitly shown."
        ),
        label="tab:posterior_constraints",
    )
    (out / "posterior_constraints.tex").write_text(latex, encoding="utf-8")

    evidence.to_latex(
        out / "bayesian_evidence.tex",
        index=False,
        escape=False,
        float_format=lambda value: f"{value:.4f}",
        caption="Bayesian evidences and pairwise model comparisons.",
        label="tab:bayesian_evidence",
    )

def get_summary_row(summary: pd.DataFrame, chain: str, param: str) -> pd.Series | None:
    rows = summary[(summary["chain"] == chain) & (summary["parameter"] == param)]
    return None if rows.empty else rows.iloc[0]

def format_constraint(row: pd.Series | None, digits: int = 5) -> str:
    if row is None:
        return "not available"
    unit = "" if row["unit"] == "dimensionless" else f" {row['unit']}"
    return (
        f"{row['median']:.{digits}g} "
        f"(-{row['err68_minus']:.2g}, +{row['err68_plus']:.2g}){unit}"
    )

def write_report(
    summary: pd.DataFrame,
    consistency: pd.DataFrame,
    evidence: pd.DataFrame,
    convergence: pd.DataFrame,
    compatibility: dict,
    out: Path,
) -> None:
    """Write a concise English report that mirrors the generated outputs."""
    comparisons = evidence[evidence["delta_logZ"].notna()].set_index("comparison")
    max_shift = consistency.loc[consistency["combined_sigma_shift"].idxmax()]
    min_overlap = consistency.loc[consistency["histogram_overlap"].idxmin()]

    sr1_pc_ns = get_summary_row(summary, "SR1 PolyChord", "n_s")
    sr1_pc_r = get_summary_row(summary, "SR1 PolyChord", "r")
    sr2_pc_ns = get_summary_row(summary, "SR2 PolyChord", "n_s")
    sr2_pc_r = get_summary_row(summary, "SR2 PolyChord", "r")
    lcdm_ns = get_summary_row(summary, "LCDM+r PolyChord", "n_s")
    lcdm_r = get_summary_row(summary, "LCDM+r PolyChord", "r")

    lines = [
        "# Reproducible analysis: Starobinsky SR1, SR2, and ΛCDM+r",
        "",
        "## Data and configuration compatibility",
        "",
        (
            "The YAML files use the same physical likelihood combination. "
            "`TTTEEE_lite_native` is normalized to `TTTEEE_lite` because both "
            "represent the Planck Plik-lite likelihood; the low-ℓ and "
            "BICEP/Keck likelihoods also match."
        ),
        (
            "- SR1: equivalent likelihoods = "
            f"{compatibility['SR1']['likelihoods_equivalent_after_normalization']}; "
            "equal common priors = "
            f"{compatibility['SR1']['common_priors_equal']}."
        ),
        (
            "- SR2: equivalent likelihoods = "
            f"{compatibility['SR2']['likelihoods_equivalent_after_normalization']}; "
            "equal common priors = "
            f"{compatibility['SR2']['common_priors_equal']}."
        ),
        "",
        "## Main posterior constraints from PolyChord",
        "",
        f"- ΛCDM+r: n_s = {format_constraint(lcdm_ns)}, r_0.05 = {format_constraint(lcdm_r)}.",
        f"- SR1: n_s = {format_constraint(sr1_pc_ns)}, r_0.05 = {format_constraint(sr1_pc_r)}.",
        f"- SR2: n_s = {format_constraint(sr2_pc_ns)}, r_0.05 = {format_constraint(sr2_pc_r)}.",
        "",
        "## MCMC-PolyChord validation",
        "",
        (
            f"The largest normalized mean shift is "
            f"{max_shift['combined_sigma_shift']:.3f} sigma for "
            f"{max_shift['model']} in {max_shift['parameter']}. "
            f"The smallest histogram overlap is "
            f"{min_overlap['histogram_overlap']:.3f} "
            f"({min_overlap['model']}, {min_overlap['parameter']})."
        ),
        (
            "Together with the GetDist triangle plots, these diagnostics test "
            "whether the posterior inference depends appreciably on the sampler."
        ),
        "",
        "## Bayesian evidence from PolyChord",
        "",
    ]
    for label in ["SR1 vs LCDM+r", "SR2 vs LCDM+r", "SR2 vs SR1"]:
        row = comparisons.loc[label]
        lines.append(
            f"- {label}: delta ln Z = {row['delta_logZ']:.3f} ± "
            f"{row['delta_logZ_std']:.3f}; B = {row['Bayes_factor']:.3g}."
        )

    lines.extend([
        "",
        (
            "Standard MCMC does not directly evaluate the evidence integral. "
            "Model selection is therefore based only on the PolyChord runs, "
            "after verifying compatible likelihoods."
        ),
        "",
        "## MCMC convergence",
        "",
    ])
    for _, row in convergence.iterrows():
        lines.append(
            f"- {row['chain']}: converged={row['converged']}, "
            f"R-1={row['final_Rminus1']:.5f}, "
            f"acceptance rate={100.0 * row['final_acceptance_rate']:.2f}%."
        )

    lines.extend([
        "",
        "## Recommended manuscript figures",
        "",
        (
            "1. Use `triangle_sampler_validation_SR1.pdf` and "
            "`triangle_sampler_validation_SR2.pdf` for sampler validation."
        ),
        (
            "2. Use `triangle_polychord_models_LCDM_SR1_SR2.pdf` and "
            "`ns_r_plane_LCDM_SR1_SR2.pdf` for the physical model comparison."
        ),
        "3. Report Bayesian evidence only for the PolyChord runs.",
        "4. State the per-chain MCMC burn-in and the R-1 < 0.01 criterion.",
        (
            "5. State explicitly that alpha_s was fixed to zero in SR1 and is "
            "derived from N_star in SR2."
        ),
        (
            "6. Define omega_b = Omega_b h^2 and omega_cdm = Omega_cdm h^2, "
            "and state that r_0.05 and n_s are evaluated at "
            "k_pivot = 0.05 Mpc^-1."
        ),
    ])
    (out / "ANALYSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if not (0.0 <= args.burn_mcmc < 1.0):
        raise ValueError("--burn-mcmc must be in [0, 1)")
    if args.max_plot_points < 2000:
        raise ValueError("--max-plot-points must be at least 2000")

    args.out.mkdir(parents=True, exist_ok=True)
    print("[1/7] Locating and extracting chain files...")
    roots = prepare_roots(args)
    (args.out / "roots_used.json").write_text(
        json.dumps({key: str(value) for key, value in roots.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[2/7] Loading complete post-burn-in samples...")
    frames = {
        "LCDM+r PolyChord": load_polychord(roots["LCDM+r PolyChord"], "lcdm"),
        "SR1 PolyChord": load_polychord(roots["SR1 PolyChord"], "sr1"),
        "SR1 MCMC": load_mcmc(roots["SR1 MCMC"], "sr1", args.burn_mcmc),
        "SR2 PolyChord": load_polychord(roots["SR2 PolyChord"], "sr2"),
        "SR2 MCMC": load_mcmc(roots["SR2 MCMC"], "sr2", args.burn_mcmc),
    }
    for name, frame in frames.items():
        print(f"      {name:20s}: {len(frame):,} retained posterior rows")

    print("[3/7] Building GetDist samples for figures...")
    chains: dict[str, ChainData] = {}
    for name, frame in frames.items():
        sample = make_getdist_sample(name, frame, args.max_plot_points, args.seed)
        chains[name] = ChainData(
            name=name,
            root=roots[name],
            frame=frame,
            plot_sample=sample,
            yaml_path=yaml_for_root(roots[name]),
        )

    print("[4/7] Computing posterior tables and compatibility checks...")
    summary_rows = []
    for name, frame in frames.items():
        summary_rows.extend(summarize_frame(name, frame))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out / "posterior_summary.csv", index=False)

    consistency = pd.DataFrame(
        sampler_consistency("sr1", frames["SR1 PolyChord"], frames["SR1 MCMC"])
        + sampler_consistency("sr2", frames["SR2 PolyChord"], frames["SR2 MCMC"])
    )
    consistency.to_csv(args.out / "sampler_consistency.csv", index=False)

    evidence = evidence_table(roots)
    evidence.to_csv(args.out / "bayesian_evidence.csv", index=False)

    compatibility = compatibility_report(chains, args.out)

    print("[5/7] Generating GetDist figures...")
    make_png = not args.no_png
    plot_individual_posteriors(chains, args.out, make_png)
    triangle_sampler_validation(chains, args.out, make_png)
    triangle_all_samplers(chains, args.out, make_png)
    triangle_polychord_models(chains, args.out, make_png)
    plot_ns_r_plane(chains, args.out, make_png, args.seed)

    print("[6/7] Generating evidence and convergence diagnostics...")
    plot_consistency(consistency, args.out, make_png)
    plot_evidence(evidence, args.out, make_png)
    convergence = plot_convergence(roots, args.out, make_png)
    convergence.to_csv(args.out / "mcmc_convergence.csv", index=False)

    print("[7/7] Writing LaTeX tables and the Markdown report...")
    write_latex_tables(summary, evidence, args.out)
    write_report(summary, consistency, evidence, convergence, compatibility, args.out)

    manifest = sorted(
        str(path.relative_to(args.out))
        for path in args.out.rglob("*")
        if path.is_file()
    )
    (args.out / "MANIFEST.txt").write_text("\n".join(manifest), encoding="utf-8")

    print("\nANALYSIS COMPLETE")
    print(f"Results: {args.out.resolve()}")
    print("Main outputs:")
    for filename in [
        "triangle_sampler_validation_SR1.pdf",
        "triangle_sampler_validation_SR2.pdf",
        "triangle_sampler_comparison_SR1_SR2.pdf",
        "triangle_polychord_models_LCDM_SR1_SR2.pdf",
        "ns_r_plane_LCDM_SR1_SR2.pdf",
        "sampler_consistency_MCMC_vs_PolyChord.pdf",
        "bayesian_evidence.pdf",
        "mcmc_convergence.pdf",
        "posterior_summary.csv",
        "sampler_consistency.csv",
        "bayesian_evidence.csv",
        "ANALYSIS_REPORT.md",
    ]:
        print(f"  - {filename}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise

