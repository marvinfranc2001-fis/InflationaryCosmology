#!/usr/bin/env python3
"""Retrospective SR1 emulator likelihood tail audit on exact PolyChord support.

Reads previously saved pointwise CSV. No CLASS calls. No training.
Important: all threshold filters use EXACT chi2 and therefore are *diagnostic only*,
not deployable neural-only trust gates; these samples overlap training data.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def normalized(a):
    a=np.asarray(a,dtype=np.float64).copy()
    if not np.all(np.isfinite(a)) or np.any(a<0) or a.sum() <= 0:
        raise ValueError("Invalid nonnegative weight vector")
    return a/a.sum()


def ess(w):
    w=normalized(w)
    return 1.0 / np.sum(w*w)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,default=Path.home()/'Yaml/final/Datos/ISCAS_2027')
    args=p.parse_args()
    base=args.base.expanduser()
    files=list((base/'posterior_audit').glob('posterior_impact_*/SR1_PolyChord_pointwise.csv.gz'))
    if not files:
        raise SystemExit('No saved pointwise CSV found in posterior_audit/')
    src=max(files,key=lambda x:x.stat().st_mtime)
    df=pd.read_csv(src)
    required=['chi2_exact','chi2_predicted','weight_exact_normalized','weight_nn_reweighted_normalized','ensemble_sigma']
    absent=[x for x in required if x not in df.columns]
    if absent: raise SystemExit('Missing fields: '+', '.join(absent))
    ex=df['chi2_exact'].to_numpy(dtype=float,copy=True)
    nn=df['chi2_predicted'].to_numpy(dtype=float,copy=True)
    w0=normalized(df['weight_exact_normalized'].to_numpy(dtype=float,copy=True))
    w1=normalized(df['weight_nn_reweighted_normalized'].to_numpy(dtype=float,copy=True))
    residual=nn-ex
    dchi=ex-np.min(ex)
    out=base/'posterior_audit'/'SR1_tail_sensitivity'
    out.mkdir(parents=True,exist_ok=True)
    print('Source:',src)
    print('Rows:',len(df),'exact chi2 min:',round(float(ex.min()),5))
    print('Global MAE unweighted:',round(float(np.mean(abs(residual))),5))
    print('Global MAE weighted by exact posterior:',round(float(np.dot(w0,abs(residual))),5))
    print('Global Kish ESS exact/NN:',round(ess(w0),2),round(ess(w1),2))
    print('\nEXACT chi2 support sensitivity: thresholds are post-hoc, NOT deployable gates')
    rows=[]
    for cut in [2,5,10,20,50,100,200,500,1000,2000,float('inf')]:
        m=dchi<=cut
        if not m.any():continue
        a=float(w0[m].sum())
        b=float(w1[m].sum())
        v0=normalized(w0[m]);v1=normalized(w1[m])
        tv=float(np.abs(v0-v1).sum()/2)
        r={
            'max_delta_exact_chi2':cut, 'n':int(m.sum()),
            'exact_posterior_mass_in_region':a,
            'nn_reweighted_mass_in_region_of_full_chain':b,
            'mae_unweighted_in_region':float(np.mean(np.abs(residual[m]))),
            'mae_exact_weighted_in_region':float(np.dot(v0,np.abs(residual[m]))),
            'ess_exact_conditional':ess(v0),
            'ess_nn_conditional':ess(v1),
            'ess_ratio_conditional':ess(v1)/ess(v0),
            'discrete_TV_conditional':tv}
        rows.append(r)
    report=pd.DataFrame(rows)
    print(report.to_string(index=False,float_format=lambda z:f'{z:.5g}'))
    report.to_csv(out/'exact_chi2_support_sensitivity.csv',index=False)
    print('\nTAIL-ERROR MASS: fraction of exact and neural-reweighted posterior mass')
    tails=[]
    for t in [2,5,10,20,50,100,200,500,1000]:
        m=residual < -t
        tails.append({'nn_underprediction_exceeds_chi2':t,
                      'n':int(m.sum()),
                      'exact_weight_fraction':float(w0[m].sum()),
                      'neural_reweighted_fraction':float(w1[m].sum())})
    tail=pd.DataFrame(tails)
    print(tail.to_string(index=False,float_format=lambda z:f'{z:.5g}'))
    tail.to_csv(out/'underprediction_tail_mass.csv',index=False)
    top=np.argsort(w1)[-10:][::-1]
    top_df=pd.DataFrame({'row':top,'exact_chi2':ex[top], 'nn_chi2':nn[top],
                         'delta_nn_minus_exact':residual[top],
                         'original_exact_posterior_weight':w0[top],
                         'reweighted_nn_weight':w1[top]})
    top_df.to_csv(out/'top10_reweight_dominators.csv',index=False)
    print('\nTop 10 dominant rows saved, with original and reweighted weights.')
    print('Saved:',out)
    print('CAUTION: in-sample retrospective diagnostic; exact-chi2 cuts cannot be used in deployment.')

if __name__=='__main__':main()
