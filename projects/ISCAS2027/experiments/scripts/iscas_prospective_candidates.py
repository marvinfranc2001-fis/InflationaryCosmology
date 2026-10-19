#!/usr/bin/env python3
"""Generate new surrogate-selected candidate points for independent exact checks.

This is NOT posterior sampling. It is a prospective, targeted, surrogate-guided
stress test built from jittered PolyChord chain locations. The chain used here
was also the emulator's training source, so proposal *design* is not independent,
but the generated parameter coordinates and future CLASS evaluations are new.

Requirements: Python environment iscas_repro, existing iscas_ensemble_benchmark.py,
iscas_posterior_impact.py, and 05_chi2_emulators/ checkpoints.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from iscas_ensemble_benchmark import Chi2Emulator, predict_raw
from iscas_posterior_impact import PREFIXES, parse_chain, sample_files

# Physical priors from the thesis and original model definitions. Foreground
# parameters use empirical search envelopes; CLASS will check true model priors.
HARD_BOUNDS = {
    'N_star': (50., 60.), 'A_s': (1.8e-9, 3.0e-9),
    'H0': (60., 75.), 'omega_b': (.020, .024),
    'omega_cdm': (.10, .14), 'tau_reio': (.04, .08),
    'n_s': (.94, .99), 'r': (0., .08), 'r0.05': (0., .08),
}

def std_cov(x, center, spread, weights=None):
    z = (x-center)/spread
    if weights is None:
        cov=np.cov(z,rowvar=False)
    else:
        avg=np.average(z,axis=0,weights=weights)
        a=z-avg
        cov=(a * weights[:,None]).T @ a / np.sum(weights)
    cov=.5*(cov+cov.T)+1.e-4*np.eye(cov.shape[0])
    return np.linalg.cholesky(cov)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['sr1','sr2','lcdm'],default='sr1')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--chain-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20271012)
    p.add_argument('--n-pool',type=int,default=12000,
                   help='Total generated proposals, half posterior-near and half wide')
    p.add_argument('--n-per-group',type=int,default=8,
                   help='Eight posterior_near_random, eight wide_random, eight wide_nn_best')
    args=p.parse_args()
    if args.n_pool<120 or args.n_per_group<1: p.error('n-pool >=120 and n-per-group >=1')
    torch.set_num_threads(1)
    rng=np.random.default_rng(args.seed)
    dataset=next(k for k in PREFIXES if k.lower().startswith(args.model))
    chainroot=args.chain_root.expanduser().resolve()
    root=args.root.expanduser().resolve()
    folder=root/'05_chi2_emulators'/dataset
    meta=json.loads((folder/'metadata.json').read_text())
    features=meta['features']; targets=meta['targets']; tix=targets.index('chi2')
    prefix=PREFIXES[dataset]
    paths=sample_files(chainroot,prefix)
    if not paths:
        raise SystemExit(f'Cannot find {prefix}.[N].txt under {chainroot}')
    df=pd.concat([parse_chain(path,prefix) for path in paths],ignore_index=True)
    cols=list(features)+['weight']
    if not set(cols).issubset(df.columns):
        raise RuntimeError(f'Missing chain columns: {set(cols)-set(df.columns)}')
    df=df.loc[np.isfinite(df[cols].to_numpy(float)).all(axis=1) & (df['weight']>0)].copy()
    X=df[features].to_numpy(float)
    weights=df['weight'].to_numpy(float)
    weights=weights/weights.sum()
    # Feature-centered standardized coordinates make tiny As and O(1) nuisance
    # parameters contribute on comparable scales.
    center=np.median(X,axis=0)
    spread=np.maximum(np.std(X,axis=0),1.e-20)
    # Posterior covariance: preserve correlations on well-supported region.
    L_near=std_cov(X,center,spread,weights=weights)
    L_wide=std_cov(X,center,spread)
    # True prior on sampled physical params, exploratory empirical envelopes
    # for nuisance parameters. These are SEARCH bounds, not actual priors.
    lo=[];hi=[]
    for j,feat in enumerate(features):
        if feat in HARD_BOUNDS: a,b=HARD_BOUNDS[feat]
        else:
            q0,q1=np.quantile(X[:,j],[.001,.999])
            margin=.15*(q1-q0)
            a,b=q0-margin,q1+margin
        if not np.isfinite(a+b) or b<=a: raise RuntimeError(f'Invalid bounds for {feat}')
        lo.append(a);hi.append(b)
    lo=np.asarray(lo);hi=np.asarray(hi)
    pool=[];orig=[]
    for group,ns,L,scale in [
        ('posterior_near',args.n_pool//2,L_near,.45),
        ('wide',args.n_pool-args.n_pool//2,L_wide,.55),
    ]:
        # Rejection sampling: preserve bounds without clipping artefacts.
        accepted=[]; attempts=0
        while sum(len(z) for z in accepted)<ns and attempts<100:
            batch=max(1024,min(30000,2*(ns-sum(len(z) for z in accepted))))
            if group=='posterior_near':
                roots=rng.choice(len(X),size=batch,replace=True,p=weights)
            else:
                roots=rng.choice(len(X),size=batch,replace=True)
            scale_i = rng.uniform(.2,1.2,size=(batch,1)) if group=='wide' else np.ones((batch,1))
            proposal=X[roots]+scale * scale_i * (rng.standard_normal((batch,len(features))) @ L.T)*spread
            keep=np.isfinite(proposal).all(axis=1) & (proposal>=lo).all(axis=1) & (proposal<=hi).all(axis=1)
            accepted.append(proposal[keep]);attempts+=1
        tmp=np.concatenate(accepted)[:ns]
        if len(tmp)<ns: raise RuntimeError(f'Only accepted {len(tmp)}/{ns} proposals for {group}')
        pool.append(tmp);orig.extend([group]*ns)
    P=np.concatenate(pool)
    ensemble=[]
    for i in range(1,6):
        net=Chi2Emulator(len(features),len(targets))
        net.load_state_dict(torch.load(folder/f'emulator_member_{i:02d}.pt',map_location='cpu',weights_only=True))
        net.eval();ensemble.append(net)
    xs=joblib.load(folder/'x_scaler.joblib');ys=joblib.load(folder/'y_scaler.joblib')
    pred=[];sigma=[]
    for i in range(0,len(P),512):
        a,b=predict_raw(ensemble,xs,ys,P[i:i+512]);pred.append(a[:,tix]);sigma.append(b[:,tix])
    pred=np.concatenate(pred);sigma=np.concatenate(sigma)
    n=args.n_per_group
    near_ind=np.arange(args.n_pool//2)
    wide_ind=np.arange(args.n_pool//2,len(P))
    near_random=rng.choice(near_ind,size=n,replace=False)
    wide_random=rng.choice(wide_ind,size=n,replace=False)
    used=set(map(int,np.concatenate([near_random,wide_random])))
    best=[]
    # Greedy NN-chi2 selection with basic diversity in standardized space.
    for idx in wide_ind[np.argsort(pred[wide_ind])]:
        idx=int(idx)
        if idx in used: continue
        normalized=(P[idx]-center)/spread
        if any(np.linalg.norm(normalized-(P[j]-center)/spread)<.4 for j in best):
            continue
        best.append(idx)
        if len(best)==n:break
    if len(best)<n:
        best+= [int(j) for j in wide_ind[np.argsort(pred[wide_ind])]
                 if int(j) not in used and int(j) not in best][:n-len(best)]
    selected=list(map(int,near_random))+list(map(int,wide_random))+best
    tags=(['posterior_near_random']*n+['wide_random']*n+['wide_nn_best']*n)
    result=pd.DataFrame(P[selected],columns=features)
    result.insert(0,'candidate_id',[f'{args.model.upper()}_P{i:03d}' for i in range(len(result))])
    result.insert(1,'selection_source',tags)
    result['nn_chi2']=pred[selected]
    result['nn_sigma']=sigma[selected]
    result['proposal_origin']=[orig[i] for i in selected]
    result['proposal_index']=selected
    created=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=args.out.expanduser().resolve()/f'prospective_candidates_{args.model}_{created}'
    out.mkdir(parents=True,exist_ok=False)
    result.to_csv(out/'candidates.csv',index=False,float_format='%.16g')
    summary=result.groupby('selection_source').agg(n=('nn_chi2','size'),
                             min_nn_chi2=('nn_chi2','min'),
                             median_nn_chi2=('nn_chi2','median'),
                             max_sigma=('nn_sigma','max')).reset_index()
    summary.to_csv(out/'proposal_summary.csv',index=False)
    manifest={'model':args.model,'dataset':dataset,'seed':args.seed,'n_pool':len(P),
              'n_selected':len(result),'source_chain_files':[str(t) for t in paths],
              'chain_sha256':[hashlib.sha256(t.read_bytes()).hexdigest() for t in paths],
              'training_independent':False,
              'exact_evaluations_new':True,
              'purpose':'Prospective targeted emulator-guided exact validation, not posterior sampling',
              'selection_rule':'8 posterior-near random + 8 wide random + 8 wide low NN chi2; no exact test labels used',
              'physical_bounds':{x:HARD_BOUNDS[x] for x in features if x in HARD_BOUNDS},
              'nuisance_bounds_empirical':{x:[float(lo[i]),float(hi[i])] for i,x in enumerate(features) if x not in HARD_BOUNDS},
              'known_limitations':['Chain source overlaps training data; proposal generation is not independent',
                                   'Nuisance search envelopes are empirical, not normalized priors',
                                   'Selection is non-iid and targeted; no posterior or evidence inference can be made',
                                   'New CLASS calls must be independently checked and failures retained']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Saved:',out,flush=True)
    print(summary.to_string(index=False),flush=True)
    print('WARNING: These are NOT independent posterior samples; new exact calls are required.',flush=True)

if __name__=='__main__':main()
