#!/usr/bin/env python3
"""Evaluate NEW emulator-selected candidate points with original Cobaya-CLASS.

Run in Conda `inflacion` on a CEDIA compute node, using the *same definitive*
PolyChord .input.yaml and the exact same likelihood implementation.
Resumable: after each point, append a CSV record and flush. Never overwrite
previous exact evaluations. Report failures, including prior/out-of-domain.
"""
from __future__ import annotations
import argparse,csv,json,os,platform,time
from datetime import datetime,timezone
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

FIELDS=['candidate_id','selection_source','status','exact_chi2','nn_chi2',
        'error_nn_minus_exact','abs_error','nn_sigma','runtime_seconds',
        'logpost','logprior','error_message']

def save_row(dest,record):
    exists=dest.exists()
    with dest.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if not exists:w.writeheader()
        w.writerow({k:record.get(k,'') for k in FIELDS})
        f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True,help='Prospective candidates CSV')
    p.add_argument('--yaml',type=Path,required=True,help='Definitive matching PolyChord input YAML')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n-max',type=int,default=24)
    p.add_argument('--packages-path',type=Path,default=Path.home()/'cobaya_packages')
    a=p.parse_args()
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file
    import numpy as np
    import pandas as pd
    from math import isfinite
    cfile=a.candidates.expanduser().resolve()
    yfile=a.yaml.expanduser().resolve()
    if not cfile.exists() or not yfile.exists(): raise SystemExit('Missing candidates CSV or YAML')
    cand=pd.read_csv(cfile)
    required={'candidate_id','selection_source','nn_chi2','nn_sigma'}
    if not required.issubset(cand.columns): raise SystemExit(f'Missing columns: {required-set(cand.columns)}')
    if cand['candidate_id'].duplicated().any(): raise SystemExit('Duplicate candidate IDs')
    info=yaml_load_file(str(yfile))
    for key in ('sampler','output','force'):info.pop(key,None)
    out=a.out.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    csv_out=out/'prospective_exact_results.csv'
    already=set()
    if csv_out.exists():
        prev=pd.read_csv(csv_out)
        already=set(prev['candidate_id'].astype(str))
        print(f'Resuming; completed rows: {len(already)}',flush=True)
    pkg=a.packages_path.expanduser().resolve()
    if not pkg.is_dir():raise SystemExit(f'Packages path not found: {pkg}')
    print('YAML:',yfile,'\nPackages:',pkg,flush=True)
    t0=time.perf_counter()
    m=get_model(info,stop_at_error=False,debug=False,packages_path=str(pkg))
    print('CLASS initialized after',round(time.perf_counter()-t0,2),'s',flush=True)
    sampled=list(m.parameterization.sampled_params())
    missing=set(sampled)-set(cand.columns)
    if missing:raise SystemExit(f'Sampled params missing from candidates CSV: {sorted(missing)}')
    (out/'run_info.json').write_text(json.dumps({'yaml':str(yfile),'candidates':str(cfile),
        'utc':datetime.now(timezone.utc).isoformat(),'host':platform.node(),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID',''),
        'sampled_params':sampled,'n_max':a.n_max,
        'caveat':'Targeted prospective exact evaluations; not an independent posterior sample'},indent=2))
    for k,(_,item) in enumerate(cand.iloc[:a.n_max].iterrows()):
        cid=str(item['candidate_id'])
        if cid in already:continue
        point={x:float(item[x]) for x in sampled}
        rec={'candidate_id':cid,'selection_source':item['selection_source'],
             'nn_chi2':float(item['nn_chi2']),'nn_sigma':float(item['nn_sigma']),
             'status':'FAILED','exact_chi2':'','error_message':''}
        print(f'[{k+1}/{min(a.n_max,len(cand))}] {cid} {item["selection_source"]} ...',end=' ',flush=True)
        tstart=time.perf_counter()
        try:
            # No cached results; these are new points not contained in the chain.
            post=m.logposterior(point,as_dict=True,make_finite=False,return_derived=True,cached=False)
            likes=post.get('loglikes',{})
            if isinstance(likes,dict):values=np.array(list(likes.values()),dtype=float)
            else:values=np.asarray(likes,dtype=float).ravel()
            # Cobaya exposes `logpriors` (plural), commonly as a dict.
            prior=post.get('logpriors',post.get('logprior',float('nan')))
            priorvals=list(prior.values()) if isinstance(prior,dict) else prior
            priorarr=np.asarray(priorvals,dtype=float).ravel()
            rec['logprior']=float(priorarr.sum()) if priorarr.size else float('nan')
            rec['logpost']=float(post.get('logpost',float('nan')))
            if (values.size<1 or not np.isfinite(values).all()
                    or not np.isfinite(rec['logprior']) or not np.isfinite(rec['logpost'])):
                rec['status']='INVALID_OR_PRIOR_REJECTED'
                rec['error_message']='Loglikes or prior not finite'
            else:
                exact=-2.*float(np.sum(values))
                rec['exact_chi2']=exact
                rec['error_nn_minus_exact']=float(item['nn_chi2'])-exact
                rec['abs_error']=abs(rec['error_nn_minus_exact'])
                rec['status']='OK'
        except Exception as exc:
            rec['status']='ERROR'
            rec['error_message']=str(exc)[:300]
        rec['runtime_seconds']=time.perf_counter()-tstart
        save_row(csv_out,rec)
        already.add(cid)
        print(rec['status'],f'{rec["runtime_seconds"]:.1f}s',
              f'error={rec.get("error_nn_minus_exact", "NA")}',flush=True)
    result=pd.read_csv(csv_out)
    valid=result[result['status']=='OK'].copy()
    if len(valid):
        valid['abs_error']=pd.to_numeric(valid['abs_error'])
        print('\nSUCCESSFUL:',len(valid),'/',len(result))
        print(valid.groupby('selection_source')['abs_error'].agg(['count','mean','median','max']).to_string())
        print('large_nn_underestimation_gt50:',int((valid['error_nn_minus_exact']< -50).sum()))
        print('NN-best points with exact delta chi2 can be checked in:',csv_out)
    print('\nSAVED:',csv_out)

if __name__=='__main__': main()
