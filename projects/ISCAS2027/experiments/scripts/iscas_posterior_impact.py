#!/usr/bin/env python3
"""Exploratory posterior-impact audit by importance reweighting saved PolyChord samples.

Usage (on compute node with iscas_repro):
  python iscas_posterior_impact.py --root PATH/Graf_ml_definitivo \
    --chain-root ~/Yaml/final/Datos/Resultados --out ~/Yaml/final/Datos/ISCAS_2027/posterior_audit

Method: starting from samples already distributed according to the EXACT posterior,
weights q_i are replaced by normalized q_i exp[-(NNchi2_i - exactchi2_i)/2].
This measures sensitivity ON THE ORIGINAL SUPPORT, not independent/emulator-driven
posterior sampling. IMPORTANT: saved chains overlap the emulators' training source:
these results are IN-SAMPLE/exploratory and cannot validate posterior fidelity.
Do not claim a speedup, an evidence estimate, or posterior convergence from this.
"""
import argparse, json, math, re, sys, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.special import logsumexp
from iscas_ensemble_benchmark import Chi2Emulator, predict_raw

PREFIXES = {
    'SR1_PolyChord': 'polychord_starobinsky_SR1_300',
    'SR2_PolyChord': 'polychord_starobinsky_SR2_300_MPI',
    'LCDM_PolyChord': 'polychord_starobinsky_lcdm',
}

def sample_files(parent: Path, prefix: str):
    paths = [p for p in parent.rglob(f'{prefix}.*.txt')
             if re.fullmatch(re.escape(prefix) + r'\.\d+\.txt', p.name)]
    return sorted(paths, key=lambda p: (str(p.parent), int(p.name[len(prefix)+1:-4])))

def parse_chain(path: Path, prefix: str):
    # Cobaya/GetDist format often has a comment header; otherwise use .paramnames.
    comment_lines = []
    with path.open(errors='replace') as f:
        for line in f:
            if line.lstrip().startswith('#'):
                comment_lines.append(line.strip().lstrip('#').strip())
            elif line.strip():
                break
    data = pd.read_csv(path, sep=r'\s+', comment='#', header=None, dtype=float)
    ncol = data.shape[1]
    header = None
    for c in reversed(comment_lines):
        cand = c.split()
        if len(cand) == ncol and cand[0].lower() in ('weight','weights'):
            header = cand
            break
    if header is None:
        pname = path.with_name(prefix + '.paramnames')
        if pname.is_file():
            names=[]
            for line in pname.read_text(errors='replace').splitlines():
                if line.strip() and not line.lstrip().startswith('#'):
                    names.append(line.split()[0].rstrip('*'))
            if len(names)+2 == ncol:
                header = ['weight','minuslogpost'] + names
    if header is None:
        raise ValueError(f'No se pueden identificar {ncol} columnas en {path}. '
                         f'Encabezados: {comment_lines[-2:]}. '
                         f'Busca el archivo {prefix}.paramnames y comparte su contenido.')
    data.columns=header
    return data

def weighted_quantile(x,w,qs):
    order=np.argsort(x)
    xx=x[order]; ww=w[order]
    xp=(np.cumsum(ww)-.5*ww)/np.sum(ww)
    return np.interp(qs,xp,xx)

def stats(x,w):
    mu=float(np.sum(x*w))
    sd=float(np.sqrt(np.sum(w*(x-mu)**2)))
    q=weighted_quantile(x,w,[.16,.5,.84])
    return mu,sd,*[float(z) for z in q]

def histogram_js_overlap(x,w0,w1):
    # Same bins determined exclusively by exact reference posterior (weighted 0.001, 0.999)
    lo,hi=weighted_quantile(x,w0,[.001,.999])
    if not np.isfinite(lo+hi) or hi<=lo:
        return float('nan'),float('nan')
    bins=np.linspace(lo,hi,41)
    p=np.histogram(x,bins=bins,weights=w0)[0].astype(float)
    q=np.histogram(x,bins=bins,weights=w1)[0].astype(float)
    p=p/p.sum();q=q/q.sum()
    m=.5*(p+q)
    def kl(a,b):
        good=a>0
        return np.sum(a[good]*np.log(a[good]/b[good]))
    return float(.5*(kl(p,m)+kl(q,m))),float(np.minimum(p,q).sum())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True,help='Parent with 05_chi2_emulators')
    ap.add_argument('--chain-root',type=Path,required=True,help='Existing exact PolyChord chain directory tree')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',choices=['all','sr1','sr2','lcdm'],default='all')
    ap.add_argument('--batch-size',type=int,default=512)
    ap.add_argument('--max-files',type=int,default=0,help='0=read all .N.txt chain segments')
    args=ap.parse_args()
    torch.set_num_threads(1)
    root=args.root.expanduser().resolve()
    chainroot=args.chain_root.expanduser().resolve()
    if not chainroot.exists():
        raise FileNotFoundError(f'No existe --chain-root: {chainroot}')
    selected=[x for x in PREFIXES if args.model=='all' or x.lower().startswith(args.model)]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=args.out.expanduser() / f'posterior_impact_{stamp}_{platform.node()}'
    out.mkdir(parents=True,exist_ok=False)
    audit=[];rows=[];problems=[]
    for dataset in selected:
        prefix=PREFIXES[dataset]
        files=sample_files(chainroot,prefix)
        if not files:
            problems.append(f'{dataset}: no se hallan {prefix}.[N].txt en {chainroot}')
            print('NO CHAIN:',problems[-1]);continue
        if args.max_files:
            files=files[:args.max_files]
        print(f'\n{dataset}: leyendo {len(files)} archivos ...',flush=True)
        chunks=[parse_chain(p,prefix) for p in files]
        df=pd.concat(chunks,ignore_index=True)
        del chunks
        folder=root/'05_chi2_emulators'/dataset
        meta=json.loads((folder/'metadata.json').read_text())
        features=meta['features']; targets=meta['targets']
        if not set(features).issubset(df.columns):
            missing=set(features)-set(df.columns)
            problems.append(f'{dataset}: faltan features: {sorted(missing)}; columnas: {list(df.columns)}')
            print('MISSING FEATURES:',problems[-1]); continue
        if 'chi2' not in df.columns:
            problems.append(f'{dataset}: falta columna exacta chi2; columnas: {list(df.columns)}')
            print('MISSING EXACT CHI2:',problems[-1]);continue
        if 'weight' not in df.columns:
            problems.append(f'{dataset}: falta columna weight; no se puede preservar posterior de PolyChord')
            print('MISSING WEIGHT:',problems[-1]);continue
        dcols=['weight','chi2']+features
        mask=np.isfinite(df[dcols].to_numpy(float)).all(axis=1) & (df['weight'].to_numpy(float)>0)
        df=df.loc[mask].copy()
        if len(df)<100:
            problems.append(f'{dataset}: menos de 100 muestras validas; encontrado={len(df)}')
            print(problems[-1]);continue
        ensemble=[]
        for i in range(1,6):
            m=Chi2Emulator(len(features),len(targets))
            m.load_state_dict(torch.load(folder/f'emulator_member_{i:02d}.pt',map_location='cpu',weights_only=True))
            m.eval(); ensemble.append(m)
        xs=joblib.load(folder/'x_scaler.joblib');ys=joblib.load(folder/'y_scaler.joblib')
        X=df[features].to_numpy(float)
        pred=[];unc=[]
        for j in range(0,len(df),args.batch_size):
            p,u=predict_raw(ensemble,xs,ys,X[j:j+args.batch_size])
            pred.append(p[:,targets.index('chi2')]);unc.append(u[:,targets.index('chi2')])
        exact=df['chi2'].to_numpy(float)
        est=np.concatenate(pred);sig=np.concatenate(unc)
        delta=est-exact
        w_exact=df['weight'].to_numpy(float)
        # Avoid overflow: log-space reweighting under exact posterior.
        logw=np.log(w_exact)
        w0=np.exp(logw-logsumexp(logw))
        w1=np.exp(logw-.5*delta-logsumexp(logw-.5*delta))
        ess_exact=1/np.sum(w0**2);ess_nn=1/np.sum(w1**2)
        diagnostic={
            'dataset':dataset,'n_samples':int(len(df)),'n_files':int(len(files)),
            'is_training_independent':False,'kind':'Exploratory in-sample support reweighting; NOT a validation',
            'chi2_mae_on_chain':float(np.mean(np.abs(delta))),
            'chi2_rmse_on_chain':float(np.sqrt(np.mean(delta**2))),
            'chi2_residual_bias_pred_minus_exact':float(np.mean(delta)),
            'ess_exact_kish':float(ess_exact),'ess_nn_reweighted_kish':float(ess_nn),
            'ess_ratio':float(ess_nn/ess_exact),
            'max_normalized_weight_nn':float(w1.max()),
            'fraction_weight_nn_top_1pct_points':float(np.sort(w1)[-max(1,len(w1)//100):].sum()),
            'source_note':'Exact weighted PolyChord posterior, not emulator-sampled posterior.',
            'chain_files':';'.join(str(p) for p in files),
        }
        audit.append(diagnostic)
        interesting=['N_star','n_s','r','r0.05','alpha_s','H0','omega_b','omega_cdm','tau_reio','A_s']
        for col in interesting:
            if col not in df.columns:continue
            x=df[col].to_numpy(float)
            if not np.isfinite(x).all():continue
            a0=stats(x,w0);a1=stats(x,w1)
            js,ov=histogram_js_overlap(x,w0,w1)
            rows.append({'dataset':dataset,'parameter':col,
                         'mean_exact':a0[0],'mean_nn_reweight':a1[0],
                         'sd_exact':a0[1],'sd_nn_reweight':a1[1],
                         'median_exact':a0[3],'median_nn_reweight':a1[3],
                         'p16_exact':a0[2],'p84_exact':a0[4],
                         'p16_nn_reweight':a1[2],'p84_nn_reweight':a1[4],
                         'mean_shift_over_combined_sd':abs(a1[0]-a0[0])/np.hypot(a0[1],a1[1]) if np.hypot(a0[1],a1[1])>0 else np.nan,
                         'js_divergence_40bins':js,'histogram_overlap_40bins':ov})
        pd.DataFrame({'weight_exact_normalized':w0,'weight_nn_reweighted_normalized':w1,
                      'chi2_exact':exact,'chi2_predicted':est,'chi2_delta_pred_minus_exact':delta,
                      'ensemble_sigma':sig}).to_csv(out/f'{dataset}_pointwise.csv.gz',index=False,compression='gzip')
        print(f'  samples={len(df)}; MAE(delta chi2)={diagnostic["chi2_mae_on_chain"]:.4f}; '
              f'ESS reweight/exact={diagnostic["ess_ratio"]:.3f}; '
              f'max weight={diagnostic["max_normalized_weight_nn"]:.4f}',flush=True)
    if audit:pd.DataFrame(audit).to_csv(out/'posterior_weight_diagnostics.csv',index=False)
    if rows:pd.DataFrame(rows).to_csv(out/'posterior_parameter_shifts.csv',index=False)
    manifest={'model':args.model,'root':str(root),'chainroot':str(chainroot),
              'sources': 'Existing exact PolyChord chains: same source used for training',
              'limitations':['This audit is IN-SAMPLE; cannot establish generalization or posterior fidelity',
                             'Reweighting cannot discover regions outside the exact chain support',
                             'ESS is Kish weight concentration, not sampler autocorrelation-adjusted ESS',
                             'No independent PolyChord evidence or end-to-end runtime estimated'],
              'problems':problems}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nOutput directory:',out)
    if rows:
        print(pd.DataFrame(rows)[['dataset','parameter','median_exact','median_nn_reweight','mean_shift_over_combined_sd']].to_string(index=False))
    if problems:
        print('\nWARNINGS / BLOCKERS:',*problems,sep='\n- ')
        if not audit:sys.exit(2)
if __name__=='__main__': main()
