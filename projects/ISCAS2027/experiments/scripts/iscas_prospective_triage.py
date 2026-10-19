#!/usr/bin/env python3
"""Summarize 3 model targeted prospective exact tests without rerunning CLASS.

Pure data audit: candidate generation was chain-derived and selection-biased.
No output is a validated posterior, calibration threshold, or hybrid speedup.

Run: conda activate iscas_repro
     python ~/Yaml/final/Datos/ISCAS_2027/iscas_prospective_triage.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path.home() / 'Yaml/final/Datos/ISCAS_2027/prospective_audit'
MODELS = {'SR1':'SR1_class_exact', 'SR2':'SR2_class_exact',
          'LCDM+r':'LCDM_class_exact'}
GROUPS = ['posterior_near_random','wide_nn_best','wide_random']

records = []
rejected = []
details = []

def safe_spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or pd.Series(a).nunique() < 2 or pd.Series(b).nunique() < 2:
        return float('nan')
    return float(spearmanr(a, b).statistic)

for model, directory in MODELS.items():
    p = BASE / directory / 'prospective_exact_results.csv'
    if not p.is_file():
        print('MISSING:', p)
        continue
    df = pd.read_csv(p)
    required = {'candidate_id','selection_source','status','error_nn_minus_exact',
                'abs_error','nn_sigma','nn_chi2'}
    if not required.issubset(df.columns):
        raise SystemExit('Missing columns in '+str(p)+': '+str(required-set(df.columns)))
    df['status'] = df['status'].astype(str)
    df['is_valid'] = df['status'].eq('OK')
    df['error_nn_minus_exact'] = pd.to_numeric(df['error_nn_minus_exact'],errors='coerce')
    df['abs_error'] = pd.to_numeric(df['abs_error'],errors='coerce')
    df['nn_sigma'] = pd.to_numeric(df['nn_sigma'],errors='coerce')
    bad = df.loc[~df.is_valid].copy()
    if len(bad):
        bad.insert(0,'model',model)
        rejected.append(bad)
    v = df.loc[df.is_valid].copy()
    v.insert(0,'model',model)
    details.append(v)
    print('\n==',model,'==')
    print('Valid:',len(v),'/',len(df),' | rejected:',len(bad))
    print('Severe underpredictions (< -50):',int((v.error_nn_minus_exact < -50).sum()))
    print('Spearman(sigma, abs_error), all selected valid:',
          round(safe_spearman(v.nn_sigma,v.abs_error),3))
    wide = v.loc[v.selection_source.eq('wide_random')]
    print('Spearman(sigma, abs_error), wide_random:',
          round(safe_spearman(wide.nn_sigma,wide.abs_error),3))
    for group in GROUPS:
        z = df.loc[df.selection_source.eq(group)]
        zz = z.loc[z.is_valid]
        metric = dict(model=model,selection_source=group,proposed=len(z),valid=len(zz),
                      invalid=len(z)-len(zz),
                      mae=(float(zz.abs_error.mean()) if len(zz) else float('nan')),
                      max_abs_error=(float(zz.abs_error.max()) if len(zz) else float('nan')),
                      severe_underestimation_n=int((zz.error_nn_minus_exact<-50).sum()),
                      sigma_median=(float(zz.nn_sigma.median()) if len(zz) else float('nan')))
        records.append(metric)
        print('%23s n=%d/%d MAE=%8.3f max=%8.3f severe=%d' %
              (group,metric['valid'],metric['proposed'],metric['mae'],
               metric['max_abs_error'],metric['severe_underestimation_n']))
    if len(v)>=5:
        # Ranking analysis only. The threshold is fitted to the same small,
        # stratified audit population, NOT an independently validated gate.
        cutoff=float(v.nn_sigma.quantile(.8))
        flagged=v.nn_sigma>=cutoff
        severe=v.error_nn_minus_exact < -50
        print('RETROSPECTIVE sigma top-20%: flagged',int(flagged.sum()),
              'caught severe',int((flagged&severe).sum()),'of',int(severe.sum()),
              '| cutoff',round(cutoff,4))
    print('Worst underestimates:')
    print(v.nsmallest(min(3,len(v)),'error_nn_minus_exact')[
        ['candidate_id','selection_source','error_nn_minus_exact','nn_sigma']
    ].to_string(index=False))

if not records:
    raise SystemExit('No result files found.')
summary=pd.DataFrame(records)
out=BASE/'triage_summary'
out.mkdir(parents=True,exist_ok=True)
summary.to_csv(out/'prospective_strata_summary.csv',index=False)
if details:
    pd.concat(details,ignore_index=True).to_csv(out/'prospective_valid_records.csv',index=False)
if rejected:
    pd.concat(rejected,ignore_index=True).to_csv(out/'prospective_rejected_records.csv',index=False)
print('\nSaved:',out)
print('CAUTION: all gates/rankings are retrospective; no posterior inference is measured.')
