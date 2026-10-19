#!/usr/bin/env python3
"""Frozen-threshold triage for a NEW ISCAS likelihood-surrogate stress test.

SCORE stage (iscas_repro, BEFORE running CLASS):
  - calibrates unlabeled sigma/distance percentiles + cutoffs using the
    PREVIOUS 54 valid new-point CLASS evaluations (a development set);
  - computes training-support 5NN distance of NEW candidates using the
    original exact PolyChord chain and the fitted emulator x_scaler;
  - fixes sigma-only, distance-only and 50:50 combined decisions and saves
    their SHA256-anchored manifest *before* their new CLASS evaluations.

AUDIT stage (iscas_repro, AFTER running CLASS):
  - joins precommitted decisions to held-out exact outcomes by candidate ID;
  - reports severe-error recall and residual MAE; never modifies cutoffs.

The test proposals are chain-derived and deliberately stratified, not an
independent draw from a cosmological posterior or a deployed hybrid sampler.
The calibration scores are from the old targeted test; validation on new
coordinates is held out, not posterior inference. Candidate prior failures
are included in the report and must be handled before any deployment.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DEFAULT = Path.home() / 'Yaml/final/Datos/ISCAS_2027'
MODEL_NAMES = {'sr1':'SR1_PolyChord', 'sr2':'SR2_PolyChord','lcdm':'LCDM_PolyChord'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def empirical_cdf(x, calibration):
    a=np.sort(np.asarray(calibration,dtype=float))
    return np.searchsorted(a,np.asarray(x,dtype=float),side='right')/len(a)


def mk_cutoff(x, q=.8):
    return float(np.quantile(np.asarray(x,float),q))


def support_distances(model, frame, base, root, chainroot, max_ref=4000):
    # Same reference-support definition and RNG as iscas_ood_gate_audit.py.
    from sklearn.neighbors import NearestNeighbors
    import joblib
    from iscas_posterior_impact import PREFIXES,parse_chain,sample_files
    folder = root/'05_chi2_emulators'/MODEL_NAMES[model]
    meta=json.loads((folder/'metadata.json').read_text())
    features=list(meta['features'])
    if not set(features).issubset(frame.columns):
        raise ValueError(f'Missing candidate features: {set(features)-set(frame.columns)}')
    paths=sample_files(chainroot,PREFIXES[MODEL_NAMES[model]])
    if not paths: raise FileNotFoundError('Original exact PolyChord chain files not found')
    chain=pd.concat([parse_chain(p,PREFIXES[MODEL_NAMES[model]]) for p in paths],ignore_index=True)
    keep=['chi2','weight']+features
    valid=np.isfinite(chain[keep].to_numpy(float)).all(axis=1)&(chain['weight'].to_numpy(float)>0)
    chain=chain.loc[valid]
    w=chain['weight'].to_numpy(float,copy=True)
    w=w/w.sum()
    chi=chain['chi2'].to_numpy(float)
    idx=np.argsort(chi)
    qchi=float(np.interp(.99,np.cumsum(w[idx]),chi[idx]))
    support=chain.loc[chi<=qchi]
    if len(support)<100: raise ValueError('Insufficient reference support')
    if len(support)>max_ref:
        rng=np.random.default_rng(20271012)
        subset=np.sort(rng.choice(len(support),max_ref,replace=False))
        support=support.iloc[subset]
    scaler=joblib.load(folder/'x_scaler.joblib')
    a=np.asarray(scaler.transform(support[features].to_numpy(float)),float)
    b=np.asarray(scaler.transform(frame[features].to_numpy(float)),float)
    knn=NearestNeighbors(n_neighbors=5,n_jobs=1).fit(a)
    d=knn.kneighbors(b,return_distance=True)[0].mean(axis=1)
    return d, {'reference_n':len(chain),'support_n':len(support),
               'reference_chi2_99pct_mass_cut':qchi,'max_ref':max_ref,
               'features':features,'x_scaler_sha256':digest(folder/'x_scaler.joblib'),
               'source_chain_files':[str(p) for p in paths]}


def score(a):
    base=a.base.expanduser().resolve()
    root=(a.root or base/'import_local/DataProcessing/Graf_ml_definitivo').expanduser().resolve()
    chainroot=(a.chain_root or Path.home()/'Yaml/final/Datos/Resultados').expanduser().resolve()
    calibr=a.calibration.expanduser().resolve()
    candpath=a.candidates.expanduser().resolve()
    out=a.out.expanduser().resolve()
    if out.exists(): raise FileExistsError('Output folder already exists; choose a new --out: '+str(out))
    calibration=pd.read_csv(calibr)
    old=calibration.loc[calibration['model'].eq(a.model)].copy()
    if len(old)<12: raise ValueError('Calibration data missing or too small for '+a.model)
    for k in ('nn_sigma','support_knn_distance'):
        if not np.isfinite(old[k].to_numpy(float)).all():raise ValueError('Nonfinite calibration scores')
    new=pd.read_csv(candpath)
    if not {'candidate_id','nn_sigma','nn_chi2','selection_source'}.issubset(new.columns):
        raise ValueError('Candidates CSV missing standard columns')
    if new['candidate_id'].duplicated().any():raise ValueError('Duplicated candidate IDs')
    if not np.isfinite(new[['nn_sigma','nn_chi2']].to_numpy(float)).all():
        raise ValueError('Nonfinite NN predictions')
    d,reference=support_distances(a.model,new,base,root,chainroot)
    new['support_knn_distance']=d
    old_sigma=old['nn_sigma'].to_numpy(float)
    old_dist=old['support_knn_distance'].to_numpy(float)
    old_comb=(empirical_cdf(old_sigma,old_sigma)+empirical_cdf(old_dist,old_dist))*.5
    new_comb=(empirical_cdf(new['nn_sigma'],old_sigma)+empirical_cdf(d,old_dist))*.5
    cuts={'sigma':mk_cutoff(old_sigma), 'distance':mk_cutoff(old_dist),
          'combined':mk_cutoff(old_comb)}
    new['sigma_calibration_percentile']=empirical_cdf(new['nn_sigma'],old_sigma)
    new['distance_calibration_percentile']=empirical_cdf(d,old_dist)
    new['combined_frozen_score']=new_comb
    new['gate_sigma']=new['nn_sigma']>cuts['sigma']
    new['gate_distance']=new['support_knn_distance']>cuts['distance']
    new['gate_combined']=new_comb>cuts['combined']
    # Add explicit invariant declaring gate records before exact values exist.
    if 'exact_chi2' in new or 'error_nn_minus_exact' in new:
        raise ValueError('Scoring must happen BEFORE exact labels are merged')
    out.mkdir(parents=True)
    saved=out/'candidates_frozen_gates.csv'
    new.to_csv(saved,index=False,float_format='%.16g')
    manifest={'utc_created':datetime.now(timezone.utc).isoformat(),
              'model':a.model,'stage':'pre-exact scoring','cutoffs':cuts,
              'calibration_n':len(old),'calibration':str(calibr),
              'calibration_sha256':digest(calibr),'input_candidates':str(candpath),
              'input_candidates_sha256':digest(candpath),
              'new_candidates_n':len(new),'frozen_candidates_sha256':digest(saved),
              'score_def':'mean of sigma and distance empirical-CDF percentiles relative to old development data',
              'cutoff_def':'80th percentile of old valid development scores; future flagged fraction NOT fixed',
              'risk_cutoff':-50.0,'reference':reference,
              'notes':['Choose among strategies on development data; do not retune after seeing holdout CLASS results',
                       'Future CLASS status is unknown when gate decisions are made',
                       'Invalid/prior-rejected candidates are NOT removed before scoring',
                       'Chain support reuses original training source; test proposals are chain-derived and targeted',
                       'An exact source prior check remains necessary for any deployment']}
    (out/'frozen_gate_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('PRE-COMMITTED:',saved)
    print('Calibration n:',len(old),' | new candidates:',len(new))
    print('Frozen cutoffs:',cuts)
    print(new[['gate_sigma','gate_distance','gate_combined']].sum().to_string())
    print('IMPORTANT: run CLASS now on candidates_frozen_gates.csv, not on older candidates.csv')


def audit(a):
    from scipy.stats import spearmanr
    out=a.out.expanduser().resolve()
    mfile=out/'frozen_gate_manifest.json'
    if not mfile.is_file():raise FileNotFoundError(mfile)
    meta=json.loads(mfile.read_text())
    p=out/'candidates_frozen_gates.csv'
    if digest(p)!=meta['frozen_candidates_sha256']:
        raise ValueError('Gate decisions have been modified since the frozen manifest was created')
    scored=pd.read_csv(p)
    exact_path=a.exact.expanduser().resolve()
    exact=pd.read_csv(exact_path)
    if not {'candidate_id','status','error_nn_minus_exact','abs_error'}.issubset(exact.columns):
        raise ValueError('Exact output missing expected columns')
    if exact['candidate_id'].duplicated().any():raise ValueError('Duplicate IDs in CLASS output')
    if set(exact['candidate_id'])!=set(scored['candidate_id']):
        raise ValueError('Exact results do not match all prescored IDs (missing or extra rows)')
    merged=scored.merge(exact[['candidate_id','status','error_nn_minus_exact','abs_error','exact_chi2']],
                        on='candidate_id',validate='one_to_one',how='left')
    ok=merged.loc[merged['status'].eq('OK')].copy()
    if not len(ok):raise ValueError('No valid exact observations')
    ok['severe']=pd.to_numeric(ok['error_nn_minus_exact'])<-50
    ok['abs_error']=pd.to_numeric(ok['abs_error'])
    summary=[]
    for strategy in ('sigma','distance','combined'):
        flag=ok['gate_'+strategy].astype(bool)
        severe=ok['severe'].astype(bool)
        un=ok.loc[~flag]
        caught=int((flag&severe).sum())
        total=int(severe.sum())
        summary.append({'model':meta['model'],'strategy':strategy,
            'n_proposed':len(merged),'n_valid':len(ok),'n_invalid':len(merged)-len(ok),
            'flagged_proposed':int(merged['gate_'+strategy].sum()),
            'flagged_valid':int(flag.sum()),'flag_fraction_proposed':float(merged['gate_'+strategy].mean()),
            'flag_fraction_valid':float(flag.mean()),'n_severe':total,
            'severe_caught':caught,'severe_missed':total-caught,
            'severe_recall':caught/total if total else float('nan'),
            'mae_unflagged':float(un['abs_error'].mean()) if len(un) else float('nan'),
            'max_error_unflagged':float(un['abs_error'].max()) if len(un) else float('nan'),
            'signed_worst_unflagged':float(un['error_nn_minus_exact'].min()) if len(un) else float('nan'),
            'missed_ids':','.join(ok.loc[(~flag)&severe,'candidate_id'].astype(str))})
    result=pd.DataFrame(summary)
    dest=out/'holdout_gate_results.csv'
    if dest.is_file():raise FileExistsError('Refusing to overwrite existing audit '+str(dest))
    result.to_csv(dest,index=False)
    merged.to_csv(out/'holdout_pointwise.csv',index=False)
    print('NEW HELD-OUT GATE RESULTS:',meta['model'])
    print(result[['strategy','flagged_proposed','flagged_valid','n_severe','severe_caught',
                  'severe_missed','mae_unflagged','missed_ids']].to_string(index=False))
    print('Invalid/preflight failures:',len(merged)-len(ok),'of',len(merged))
    print('Saved:',dest)
    print('CAUTION: Independent new CLASS checks of chain-derived targeted proposals, NOT independent posterior inference.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    p_score=sub.add_parser('score',help='Freeze thresholds before new CLASS evaluations')
    p_score.add_argument('--model',choices=MODEL_NAMES,required=True)
    p_score.add_argument('--base',type=Path,default=BASE_DEFAULT)
    p_score.add_argument('--root',type=Path)
    p_score.add_argument('--chain-root',type=Path)
    p_score.add_argument('--calibration',type=Path,default=BASE_DEFAULT/'prospective_audit/ood_gate_audit/risk_gate_pointwise.csv')
    p_score.add_argument('--candidates',type=Path,required=True)
    p_score.add_argument('--out',type=Path,required=True)
    p_audit=sub.add_parser('audit',help='Evaluate PRE-frozen decisions on new exact labels')
    p_audit.add_argument('--out',type=Path,required=True)
    p_audit.add_argument('--exact',type=Path,required=True)
    a=p.parse_args()
    (score if a.mode=='score' else audit)(a)

if __name__=='__main__':main()
