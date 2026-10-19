#!/usr/bin/env python3
"""Prospective-new-point audit: compare ensemble dispersion with training-support distance.

Loads 3 existing 24-point prospective CLASS tests. Does NOT rerun CLASS, retrain,
produce posterior samples, or claim a deployed hybrid. Supports risk rankings
and fixed-budget, retrospective triage diagnostics (same test data determine
which proportion is flagged, but *no exact errors* are used to score/rank).

Usage (Conda iscas_repro, CEDIA compute node):
    python iscas_ood_gate_audit.py

Assumes original `iscas_posterior_impact.py` exists beside the script.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from iscas_posterior_impact import PREFIXES, parse_chain, sample_files

DATASET_NAMES = {'sr1': 'SR1_PolyChord', 'sr2': 'SR2_PolyChord', 'lcdm': 'LCDM_PolyChord'}
CLASS_DIRS = {'sr1': 'SR1_class_exact', 'sr2': 'SR2_class_exact', 'lcdm': 'LCDM_class_exact'}
BASE_DEFAULT = Path.home() / 'Yaml/final/Datos/ISCAS_2027'


def empirical_pct(values, baseline):
    # Fraction of known baseline values <= new point (no prospective labels).
    b = np.sort(np.asarray(baseline, dtype=float))
    return np.searchsorted(b, np.asarray(values, dtype=float), side='right') / len(b)


def ranks(values):
    # Candidate-pool ordinal ranks, no exact-label information.
    return pd.Series(np.asarray(values, dtype=float)).rank(pct=True, method='average').to_numpy()


def summarize_gate(valid, col, budget):
    valid = valid.copy()
    chosen = valid.sort_values([col, 'candidate_id'], ascending=[False, True]).head(budget)
    flagged = set(chosen['candidate_id'].astype(str))
    severe = valid['severe'].astype(bool)
    selected = valid['candidate_id'].astype(str).isin(flagged)
    caught = int((selected & severe).sum())
    n_severe = int(severe.sum())
    missed = int((~selected & severe).sum())
    unflagged = valid.loc[~selected]
    return {
        'method': col,
        'n_valid': int(len(valid)), 'n_flagged': int(selected.sum()),
        'n_severe': n_severe, 'n_caught': caught, 'n_missed': missed,
        'severe_recall': caught / n_severe if n_severe else float('nan'),
        'flagged_precision': caught / int(selected.sum()) if selected.any() else float('nan'),
        'unflagged_mae': float(unflagged['abs_error'].mean()) if len(unflagged) else float('nan'),
        'unflagged_max_abs_error': float(unflagged['abs_error'].max()) if len(unflagged) else float('nan'),
        'unflagged_min_signed_error': float(unflagged['error_nn_minus_exact'].min()) if len(unflagged) else float('nan'),
        'flagged_candidate_ids': ','.join(valid.loc[selected, 'candidate_id'].astype(str)),
        'missed_severe_ids': ','.join(valid.loc[~selected & severe, 'candidate_id'].astype(str)),
    }, flagged


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--base', type=Path, default=BASE_DEFAULT)
    a.add_argument('--root', type=Path, default=None)
    a.add_argument('--chain-root', type=Path, default=None)
    a.add_argument('--out', type=Path, default=None)
    a.add_argument('--budget', type=float, default=.20,
                   help='Fraction of *valid* prospective points to flag per model (default .20).')
    a.add_argument('--max-ref', type=int, default=4000,
                   help='Max training-chain support points for neighbor search (default 4000).')
    a.add_argument('--ref-mass', type=float, default=.99,
                   help='Mass of exact training posterior retained in support (default .99).')
    args=a.parse_args()
    if not (0 < args.budget < 1 and 0.8 <= args.ref_mass < 1 and args.max_ref >= 50):
        a.error('Require 0<budget<1; .8<=ref-mass<1; max-ref>=50')
    base = args.base.expanduser().resolve()
    root = (args.root or (base/'import_local/DataProcessing/Graf_ml_definitivo')).expanduser().resolve()
    chainroot = (args.chain_root or (Path.home()/'Yaml/final/Datos/Resultados')).expanduser().resolve()
    audit = base/'prospective_audit'
    out = (args.out or (audit/'ood_gate_audit')).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_details, all_metrics, all_rejected = [], [], []

    for model, dataset in DATASET_NAMES.items():
        run_dir = audit/CLASS_DIRS[model]
        exact_path = run_dir/'prospective_exact_results.csv'
        manifest_path = run_dir/'run_info.json'
        if not exact_path.is_file() or not manifest_path.is_file():
            print('SKIP', model, ': missing prospective results or run_info.json')
            continue
        manifest = json.loads(manifest_path.read_text())
        proposed_path = Path(manifest['candidates']).expanduser().resolve()
        if not proposed_path.is_file():
            raise FileNotFoundError('Exact run points to missing candidates CSV: '+str(proposed_path))
        proposed = pd.read_csv(proposed_path)
        exact = pd.read_csv(exact_path)
        if proposed['candidate_id'].duplicated().any() or exact['candidate_id'].duplicated().any():
            raise ValueError('Duplicated candidate_id for '+model)
        merged = proposed.merge(
            exact[['candidate_id', 'status', 'nn_chi2', 'nn_sigma',
                   'error_nn_minus_exact', 'abs_error', 'exact_chi2', 'error_message']],
            on='candidate_id', how='inner', validate='one_to_one', suffixes=('_proposal','_exact'))
        if len(merged) != len(exact):
            raise ValueError('Candidate ID mismatch: '+model)
        # Verify same predictions and same prospect selection used for CLASS.
        for f in ['nn_chi2', 'nn_sigma']:
            lhs = merged[f+'_proposal'].to_numpy(float)
            rhs = merged[f+'_exact'].to_numpy(float)
            if not np.allclose(lhs, rhs, rtol=1e-5, atol=1e-5, equal_nan=True):
                raise ValueError(f'Mismatched {f}: proposed vs evaluated candidate CSV for {model}')
        merged['nn_chi2'] = merged['nn_chi2_proposal']
        merged['nn_sigma'] = merged['nn_sigma_proposal']

        folder = root/'05_chi2_emulators'/dataset
        meta = json.loads((folder/'metadata.json').read_text())
        features = meta['features']
        if not set(features).issubset(merged.columns):
            raise ValueError(f'Missing features {set(features)-set(merged.columns)} for {model}')
        paths = sample_files(chainroot, PREFIXES[dataset])
        if not paths:
            raise FileNotFoundError(f'Cannot locate {PREFIXES[dataset]}.*.txt under {chainroot}')
        chain = pd.concat([parse_chain(p, PREFIXES[dataset]) for p in paths], ignore_index=True)
        use = ['chi2', 'weight']+features
        mask = np.isfinite(chain[use].to_numpy(dtype=float)).all(axis=1) & (chain['weight'].to_numpy(float)>0)
        chain = chain.loc[mask]
        if len(chain)<100:
            raise ValueError('Insufficient training chain samples for '+model)
        chi2 = chain['chi2'].to_numpy(float)
        w = chain['weight'].to_numpy(dtype=float, copy=True)
        w = w / w.sum()
        order = np.argsort(chi2)
        qchi2 = float(np.interp(args.ref_mass, np.cumsum(w[order]), chi2[order]))
        support = chain.loc[chi2<=qchi2]
        if len(support)<100: raise ValueError('Too few posterior support points for '+model)
        rng = np.random.default_rng(20271012)   # fixed seed; does not depend on test labels
        if len(support)>args.max_ref:
            support = support.iloc[np.sort(rng.choice(len(support), args.max_ref, replace=False))]

        # The scaler was fit only on the pre-existing emulator training partition.
        scaler = joblib.load(folder/'x_scaler.joblib')
        X = np.asarray(scaler.transform(support[features].to_numpy(float)), dtype=float)
        Xnew = np.asarray(scaler.transform(merged[features].to_numpy(float)), dtype=float)
        knn = NearestNeighbors(n_neighbors=6, algorithm='auto', n_jobs=1).fit(X)
        ref_d = knn.kneighbors(X, return_distance=True)[0][:,1:6].mean(axis=1)
        point_d = knn.kneighbors(Xnew, n_neighbors=5, return_distance=True)[0].mean(axis=1)
        merged['support_knn_distance'] = point_d
        merged['support_distance_percentile'] = empirical_pct(point_d,ref_d)
        valid = merged.loc[merged['status'].eq('OK')].copy()
        bad = merged.loc[~merged['status'].eq('OK')].copy()
        if len(bad):
            bad.insert(0, 'model', model)
            all_rejected.append(bad)
        if len(valid)<5:
            print('SKIP', model, ': not enough valid points')
            continue
        valid['error_nn_minus_exact'] = pd.to_numeric(valid['error_nn_minus_exact'],errors='coerce')
        valid['abs_error'] = pd.to_numeric(valid['abs_error'],errors='coerce')
        valid['severe'] = valid['error_nn_minus_exact'] < -50.
        if not np.isfinite(valid['abs_error'].to_numpy(float)).all():
            raise ValueError('Non-finite exact error in OK rows for '+model)
        # Candidate-set ranks do not use exact labels, but *do* depend on this
        # targeted, nonrepresentative candidate pool (not calibrated thresholds).
        valid['sigma_rank'] = ranks(valid['nn_sigma'])
        valid['distance_rank'] = ranks(valid['support_knn_distance'])
        valid['combined_score'] = .5*valid['sigma_rank']+.5*valid['distance_rank']
        budget = max(1, math.ceil(args.budget*len(valid)))
        print('\n===', model, '===')
        print('Valid', len(valid),' / proposed',len(merged), '| budget',budget,
              '| severe',int(valid['severe'].sum()))
        print('Train chain rows',len(chain),' support rows',len(support),
              '| weighted chi2 99%-mass threshold', round(qchi2,3))
        flagged_by = {}
        for strategy in ['sigma_rank','distance_rank','combined_score']:
            metric,flagged = summarize_gate(valid,strategy,budget)
            metric['model']=model
            metric['n_proposed']=len(merged)
            metric['n_invalid']=len(bad)
            metric['reference_support_n']=len(support)
            metric['reference_chi2_weighted_quantile']=qchi2
            all_metrics.append(metric)
            flagged_by[strategy]=flagged
            print(f"{strategy:15s} flagged={metric['n_flagged']} "
                  f"caught={metric['n_caught']}/{metric['n_severe']} "
                  f"missed={metric['missed_severe_ids'] or '-'} "
                  f"unflagged_MAE={metric['unflagged_mae']:.3f}")
        for strategy,flagged in flagged_by.items():
            valid['flagged_'+strategy]=valid['candidate_id'].astype(str).isin(flagged)
        print('Severe records:')
        severe_cols=['candidate_id','selection_source','error_nn_minus_exact',
                     'nn_sigma','support_knn_distance','support_distance_percentile',
                     'flagged_sigma_rank','flagged_distance_rank','flagged_combined_score']
        print(valid.loc[valid['severe'],severe_cols].to_string(index=False))
        valid.insert(0,'model',model)
        all_details.append(valid)

    if not all_metrics: raise SystemExit('No completed candidate audits found')
    metrics = pd.DataFrame(all_metrics)
    details = pd.concat(all_details,ignore_index=True)
    metrics.to_csv(out/'risk_gate_summary.csv',index=False)
    details.to_csv(out/'risk_gate_pointwise.csv',index=False)
    if all_rejected:
        pd.concat(all_rejected,ignore_index=True).to_csv(out/'rejected_candidates.csv',index=False)
    (out/'README.txt').write_text(
        'Ood gate audit; prospective coordinates and CLASS labels newly evaluated, but\n'
        'proposal origins overlap with training-chain source. Rank thresholds depend\n'
        'on this deliberately stratified candidate set, although scores/selection\n'
        'never use held-out CLASS labels. Therefore exploratory retrospective\n'
        'comparison only; NOT a trained, independently tested, deployed gate.\n'
        'The training-support distance uses 5-nearest-neighbor standardized-feature\n'
        'distances to original exact posterior chain points covering its first\n'
        f'{args.ref_mass:.0%} cumulative weighted chi2 region, capped at {args.max_ref} points.\n'
        f'Selection budget: ceil({args.budget} * n_valid) per model; invalid\n'
        'candidates excluded and analyzed separately. No CLASS or neural calls run.\n'
    )
    print('\nSAVED:',out)
    print('WARNING: No validated end-to-end hybrid posterior inference here.')


if __name__ == '__main__':
    main()
