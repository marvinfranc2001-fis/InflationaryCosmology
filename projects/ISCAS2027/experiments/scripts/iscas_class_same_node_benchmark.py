#!/usr/bin/env python3
"""Small, same-node Cobaya/CLASS timing and reference-replay audit for ISCAS.

Uses *only* previously validated candidate input vectors, with the original
Cobaya `.input.yaml` for each model. Produces fresh exact calculations (does
not overwrite prior results). If the values fail to agree with the stored
references, DO NOT interpret its timings as a valid surrogate speedup.

Run within an allocated CEDIA compute node, in the `inflacion` environment.
E.g.:
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 python iscas_class_same_node_benchmark.py --model sr1 \
   --search-root ~/Yaml/final/Datos \
   --root ~/Yaml/final/Datos/ISCAS_2027/import_local/DataProcessing/Graf_ml_definitivo \
   --out ~/Yaml/final/Datos/ISCAS_2027/benchmarks

The pipeline used Cobaya get_model, logposterior(...,cached=False), and summed
-2*loglikes; this script replicates that operation. The original validation
selected four non-IID strata. Timing should not be extrapolated to deployment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

# Set these before NumPy / Cobaya load.
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

NAMES = {'sr1': 'SR1_PolyChord','sr2':'SR2_PolyChord','lcdm':'LCDM_PolyChord'}


def get_yaml(root: Path, dataset: str, search_root: Path, forced: Path | None) -> Path:
    if forced is not None:
        if not forced.is_file():
            raise FileNotFoundError(f'YAML inexistente: {forced}')
        return forced.resolve()
    env_path = root / '06_external_validation' / dataset / 'validation_environment.json'
    original_name = Path(json.loads(env_path.read_text())['yaml']).name
    matches = sorted(search_root.rglob(original_name))
    if not matches:
        raise FileNotFoundError(
            f'No se encontró el YAML original {original_name} bajo {search_root}. '
            'Hay que transferirlo desde la computadora local o indicar --yaml.')
    if len(matches) > 1:
        raise ValueError(
            f'Hay {len(matches)} copias del YAML {original_name}. '
            'Selecciona la correcta con --yaml:\n' + '\n'.join(map(str, matches)))
    return matches[0].resolve()


def choose_rows(csv_path: Path, n_points: int, seed: int) -> list[dict]:
    import random
    with csv_path.open(newline='') as f:
        valid = [row for row in csv.DictReader(f) if row['status'] == 'OK']
    rnd = random.Random(seed)
    random_source = [r for r in valid if r.get('selection_source') == 'random']
    uncertainty_source = [r for r in valid if r.get('selection_source') == 'uncertainty']
    # Both ordinary candidates and the systematically difficult stratum.
    n_hi = max(1, n_points // 3) if n_points > 1 else 0
    n_rand = n_points - n_hi
    if len(random_source) < n_rand or len(uncertainty_source) < n_hi:
        raise RuntimeError('Insuficientes puntos OK para el muestreo fijado')
    chosen = rnd.sample(random_source, n_rand) + rnd.sample(uncertainty_source, n_hi)
    rnd.shuffle(chosen)
    return chosen


def extract_loglikes(posterior, model):
    import numpy as np
    likes = posterior.get('loglikes', {})
    if isinstance(likes, dict):
        return {str(k): float(v) for k,v in likes.items()}
    array = np.asarray(likes,dtype=float).ravel()
    names = list(getattr(model,'likelihood',{}))
    if len(names) != len(array):
        raise RuntimeError('Nombres y número de likelihoods no coinciden')
    return {str(k):float(v) for k,v in zip(names,array)}


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=['sr1','sr2','lcdm','all'], default='sr1')
    p.add_argument('--root', type=Path, required=True, help='Graf_ml_definitivo con 06_external_validation')
    p.add_argument('--search-root',type=Path,required=True,help='Carpeta para buscar YAML originales')
    p.add_argument('--yaml',type=Path,help='Ruta explícita .input.yaml (solo para un modelo)')
    p.add_argument('--packages-path',type=Path,help='Carpeta existente de paquetes de Cobaya')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int,default=5,help='Número de puntos por modelo (default: 5)')
    p.add_argument('--seed',type=int,default=314159)
    p.add_argument('--tolerance',type=float,default=.10,help='Tolerancia absoluta en chi2 para concordancia')
    args = p.parse_args()
    if args.n < 1 or args.tolerance <= 0:
        p.error('Se requiere --n >= 1 y --tolerance > 0')
    if args.model == 'all' and args.yaml:
        p.error('--yaml solo se admite con un único modelo')
    root = args.root.expanduser().resolve()
    search_root = args.search_root.expanduser().resolve()
    if not (root/'06_external_validation').is_dir() or not search_root.is_dir():
        p.error('Comprueba --root y --search-root')
    datasets = list(NAMES) if args.model == 'all' else [args.model]
    yamls = {}
    for modelname in datasets:
        name = NAMES[modelname]
        path = get_yaml(root,name,search_root,args.yaml.expanduser() if args.yaml else None)
        yamls[modelname] = path
        print(f'YAML {name}: {path}',flush=True)
    pkg = args.packages_path.expanduser().resolve() if args.packages_path else None
    if pkg is not None and not pkg.is_dir():
        raise NotADirectoryError(f'No existe packages-path: {pkg}')
    if pkg is None:
        candidate=Path.home()/'cobaya_packages'
        if candidate.is_dir():
            pkg=candidate
    try:
        from cobaya.model import get_model
        from cobaya.yaml import yaml_load_file
        import cobaya
        import classy
    except ImportError as exc:
        raise SystemExit('Activa el entorno inflacion con cobaya/classy antes: '+str(exc))
    import numpy as np
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=args.out.expanduser().resolve()/f'class_exact_{stamp}_{platform.node()}'
    out.mkdir(parents=True,exist_ok=False)
    env = {'created_utc':stamp,'host':platform.node(),'slurm_job_id':os.environ.get('SLURM_JOB_ID',''),
           'python':platform.python_version(),'numpy':np.__version__,
           'cobaya':getattr(cobaya,'__version__','unknown'),
           'classy_version':getattr(classy,'__version__','unknown'),
           'thread_vars':{key:os.environ[key] for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
           'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
           'packages_path':str(pkg) if pkg else None,
           'yaml_paths':{k:str(v) for k,v in yamls.items()},
           'yaml_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in yamls.items()},
           'n_requested_per_model':args.n,'seed':args.seed,'chi2_acceptance_tolerance':args.tolerance,
           'protocol':'Warm-up 1 previously saved point, then n uncached Cobaya logposterior evaluations; initialization separately timed; input sourced from 06_external_validation',
           'warning':'Compute speedup only with matched chi2, same machine/core settings and matching model/likelihood.'}
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    rows=[]
    summaries=[]
    for key in datasets:
        name=NAMES[key]
        path=yamls[key]
        selected=choose_rows(root/'06_external_validation'/name/'cobaya_validation_results.csv',args.n,args.seed)
        info=yaml_load_file(str(path))
        for k in ('sampler','output','force'):
            info.pop(k,None)
        print(f'\n=== {name} ===',flush=True)
        print('Initializing Cobaya/CLASS; this can take time...',flush=True)
        t0=time.perf_counter()
        m=get_model(info,stop_at_error=True,debug=False,packages_path=str(pkg) if pkg else None)
        init_sec=time.perf_counter()-t0
        sampled=list(m.parameterization.sampled_params())
        for r in selected:
            missing=[x for x in sampled if x not in r or r[x]=='']
            if missing:
                raise ValueError(f'{name}: faltan parámetros muestreados en el CSV: {missing}')
        # Warm-up outside timing.
        first={x:float(selected[0][x]) for x in sampled}
        print(f'Initializing done ({init_sec:.1f}s). Warm-up exact evaluation...',flush=True)
        _=m.logposterior(first,as_dict=True,make_finite=False,return_derived=True,cached=False)
        for r in selected:
            point={x:float(r[x]) for x in sampled}
            print(f'  point {r["candidate_id"]} [{r["selection_source"]}] ... ',end='',flush=True)
            begin=time.perf_counter()
            post=m.logposterior(point,as_dict=True,make_finite=False,return_derived=True,cached=False)
            elapsed=time.perf_counter()-begin
            likes=extract_loglikes(post,m)
            if not likes or any(not np.isfinite(v) for v in likes.values()):
                raise RuntimeError(f'{name} candidate {r["candidate_id"]}: loglikes invalidos')
            exact_new=-2*sum(likes.values())
            original=float(r['real_chi2'])
            delta=exact_new-original
            agrees=abs(delta)<=args.tolerance
            record={'dataset':name,'candidate_id':r['candidate_id'],
                    'selection_source':r['selection_source'],'elapsed_seconds':elapsed,
                    'chi2_now':exact_new,'chi2_original':original,
                    'chi2_delta':delta,'match_reference':agrees,
                    'n_likelihood_components':len(likes)}
            rows.append(record)
            print(f'{elapsed:.3f}s | delta chi2={delta:+.6f} | match={agrees}',flush=True)
            with (out/'exact_points.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(record));writer.writeheader();writer.writerows(rows)
        this=[r for r in rows if r['dataset']==name]
        summary={'dataset':name,'n':len(this),'init_seconds_excluded':init_sec,
                 'median_exact_seconds_per_point':statistics.median(x['elapsed_seconds'] for x in this),
                 'mean_exact_seconds_per_point':statistics.mean(x['elapsed_seconds'] for x in this),
                 'max_abs_delta_chi2':max(abs(x['chi2_delta']) for x in this),
                 'all_match_reference':all(x['match_reference'] for x in this)}
        summaries.append(summary)
        print('Result:',summary,flush=True)
        close=getattr(m,'close',None)
        if callable(close):
            close()
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(f'\nSaved: {out}',flush=True)
    if not all(x['all_match_reference'] for x in summaries):
        print('IMPORTANT: Reference mismatch. DO NOT report CLASS-vs-ensemble speedup.',flush=True)
    else:
        print('Reference replay passed numerical tolerance; same-node timing available.',flush=True)

if __name__=='__main__':
    main()
