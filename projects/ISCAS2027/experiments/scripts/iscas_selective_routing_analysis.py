#!/usr/bin/env python3
"""Exploratory offline selective-inference analysis of saved Cobaya reference rows.

No Cobaya/CLASS calls. Uses a pre-specified rule: send a point to CLASS if its
5-network ensemble dispersion exceeds the 80th percentile measured on the
other 4/5 folds. Never uses the evaluation fold's true chi2 to select a point.

Crucial limitations: candidate set is purposefully selected (25 acquisition,
25 uncertainty, 25 low_chi2, 25 random per model); evaluation is exploratory,
not an independent validation of a newly designed routing policy; 'hybrid'
errors assume an exact CLASS fallback and ignore its computational cost.
"""
import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold


def corr(a, b):
    v = spearmanr(a, b)
    return float(v.statistic) if np.isfinite(v.statistic) else float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--points', type=Path, required=True, help='ensemble_external_replay_points.csv')
    p.add_argument('--out', type=Path, required=True, help='Output parent directory')
    p.add_argument('--threshold-quantile', type=float, default=.80)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not 0 < args.threshold_quantile < 1:
        p.error('--threshold-quantile must lie between 0 and 1')
    df = pd.read_csv(args.points.expanduser())
    required = {'dataset', 'selection_source', 'ensemble_sigma', 'absolute_error'}
    if not required.issubset(df.columns):
        raise ValueError('Missing columns: ' + ', '.join(sorted(required-set(df.columns))))

    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.out.expanduser() / f'selective_{now}_{platform.node()}'
    out.mkdir(parents=True, exist_ok=False)
    summaries, strata, oof = [], [], []
    for name, group in df.groupby('dataset', sort=True):
        group = group.reset_index(drop=True)
        sigma = group['ensemble_sigma'].to_numpy(dtype=float)
        err = group['absolute_error'].to_numpy(dtype=float)
        if not (np.isfinite(sigma).all() and np.isfinite(err).all()):
            raise ValueError(f'Non-finite data in {name}')
        if len(group) < args.folds:
            raise ValueError(f'Fewer observations than folds for {name}')
        if (sigma < 0).any() or (err < 0).any():
            raise ValueError(f'Negative sigma or error in {name}')

        random_mask = group['selection_source'].eq('random').to_numpy()
        flagged = np.zeros(len(group), dtype=bool)
        thresholds = np.zeros(len(group), dtype=float)
        fold_idx = np.full(len(group), -1, dtype=int)
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for fold, (cal_idx, test_idx) in enumerate(kf.split(group), start=1):
            # Only ensemble sigma on the calibration fold determines the gate.
            threshold = float(np.quantile(sigma[cal_idx], args.threshold_quantile))
            flagged[test_idx] = sigma[test_idx] >= threshold
            thresholds[test_idx] = threshold
            fold_idx[test_idx] = fold

        # Offline theoretical replay. Flagged points have zero error ONLY if a
        # full, correctly configured CLASS call replaces their NN prediction.
        ideal_hybrid_errors = np.where(flagged, 0.0, err)
        rmse = lambda x: float(np.sqrt(np.mean(x*x)))
        summaries.append({
            'dataset': name,
            'n': len(group),
            'pooled_spearman_sigma_abs_error': corr(sigma, err),
            'random_subset_n': int(random_mask.sum()),
            'random_subset_spearman': corr(sigma[random_mask], err[random_mask]) if random_mask.sum() > 2 else np.nan,
            'fallback_fraction_oof': float(flagged.mean()),
            'original_mae': float(err.mean()),
            'accepted_mae': float(err[~flagged].mean()) if (~flagged).any() else np.nan,
            'ideal_offline_hybrid_mae': float(ideal_hybrid_errors.mean()),
            'original_rmse': rmse(err),
            'ideal_offline_hybrid_rmse': rmse(ideal_hybrid_errors),
            'offline_mae_reduction_fraction': float(1-ideal_hybrid_errors.mean()/err.mean()) if err.mean() else np.nan,
            'notes': 'Retrospective mixed candidate pool, oracle-perfect CLASS fallback; NOT prospective speedup or unbiased deployment performance',
        })
        for source, sub in group.groupby('selection_source', sort=True):
            idx = sub.index.to_numpy()
            strata.append({
                'dataset': name, 'source': source, 'n': len(idx),
                'spearman_sigma_abs_error': corr(sigma[idx], err[idx]) if len(idx) > 2 else np.nan,
                'mean_abs_error': float(err[idx].mean()),
                'fallback_fraction_oof': float(flagged[idx].mean()),
                'accepted_mae_oof': float(err[idx][~flagged[idx]].mean()) if (~flagged[idx]).any() else np.nan,
            })
        for i, row in group.iterrows():
            oof.append({
                'dataset': name,
                'candidate_id': row.get('candidate_id', i),
                'selection_source': row['selection_source'],
                'ensemble_sigma': float(sigma[i]),
                'absolute_error_nn': float(err[i]),
                'cv_fold': int(fold_idx[i]),
                'threshold_calibrated_on_other_folds': float(thresholds[i]),
                'route_to_exact_CLASS': bool(flagged[i]),
                'ideal_hybrid_absolute_error': float(ideal_hybrid_errors[i]),
            })

    pd.DataFrame(summaries).to_csv(out/'selective_cv_summary.csv', index=False)
    pd.DataFrame(strata).to_csv(out/'selection_source_audit.csv', index=False)
    pd.DataFrame(oof).to_csv(out/'selective_cv_points.csv', index=False)
    (out/'README.json').write_text(json.dumps({
        'timestamp_utc': now, 'source_points': str(args.points.expanduser()),
        'threshold_quantile': args.threshold_quantile,
        'folds': args.folds, 'seed': args.seed,
        'description': 'Out-of-fold sigma threshold only; ideal fallback error assumes exact CLASS at flagged points.',
        'limitations': [
            'The same historical 300 points were inspected before choosing to study selective inference.',
            'Mixed source groups produce substantial distribution and selection effects.',
            'No new exact evaluations, no independent prospective validation, no hybrid wall-clock speedup.',
        ],
    }, indent=2))
    print('=== STRATIFIED ERROR DIAGNOSTICS ===')
    print(pd.DataFrame(strata).to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('\n=== FIVE-FOLD OFFLINE ROUTING EXPERIMENT (20% sigma threshold) ===')
    print(pd.DataFrame(summaries).drop(columns=['notes']).to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('\nSaved to', out)
    print('CAUTION: Retrospective, selection-biased sample. Exact CLASS fallback is assumed, not executed. No speedup claim.')

if __name__ == '__main__':
    main()
