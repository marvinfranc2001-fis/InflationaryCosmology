#!/usr/bin/env python3
"""Recalcula estadísticas y vista previa usando las CADENAS PolyChord originales (no Nested Sampling)."""
from __future__ import annotations
import argparse
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHAIN_INFO = (
    ('LCDM+r', 'chainslcdm.zip', 'chainslcdm/polychord_starobinsky_lcdm.1.txt'),
    ('SR1', 'chains_poly.zip', 'chains_poly_SR1_300/polychord_starobinsky_SR1_300.1.txt'),
    ('SR2', 'chains_poly.zip', 'chains_poly_SR2_300/polychord_starobinsky_SR2_300_MPI.1.txt'),
)
VARS = ['N_star','n_s','r','alpha_s','A_s','H0','omega_b','omega_cdm','tau_reio']

def load_chain(zip_path: Path, internal_path: str) -> pd.DataFrame:
    with ZipFile(zip_path) as z:
        with z.open(internal_path) as f:
            hdr = f.readline().decode('utf-8-sig').lstrip('#').split()
        wanted = [name for name in ['weight'] + VARS if name in hdr]
        with z.open(internal_path) as f:
            data = pd.read_csv(f, sep=r'\s+', comment='#', names=hdr, usecols=wanted)
    if data.empty or 'weight' not in data:
        raise ValueError(f'Cadena inválida: {internal_path}')
    return data

def weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float,float,float,float]:
    is_valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    val, wt = values[is_valid], weights[is_valid]
    if val.size == 0 or wt.sum() <= 0:
        raise ValueError('Sin pesos positivos finitos')
    order = np.argsort(val)
    val, wt = val[order], wt[order]
    cdf = (np.cumsum(wt) - wt/2)/wt.sum()
    q16, q50, q84 = np.interp([0.16, 0.5, 0.84], cdf, val)
    mean = np.average(val, weights=wt)
    return float(mean),float(q16),float(q50),float(q84)

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives',type=Path,default=Path.cwd(),help='Directorio de ZIPs originales')
    p.add_argument('--output',type=Path,default=ROOT/'results'/'weighted_constraints_original_polychord.csv')
    p.add_argument('--figure',type=Path,default=ROOT/'figures'/'original_polychord_ns_r.svg')
    args = p.parse_args()
    frames={}
    rows=[]
    for name,arch,internal in CHAIN_INFO:
        source=args.archives/arch
        if not source.exists():
            raise FileNotFoundError(f'No se encontró {source}. Conservar ZIPs sin modificar.')
        frame=load_chain(source,internal)
        weights=frame.weight.to_numpy(dtype=float)
        frames[name]=frame
        for var in VARS:
            if var not in frame:
                continue
            mean,q16,med,q84=weighted_stats(frame[var].to_numpy(dtype=float),weights)
            rows.append(dict(model=name,parameter=var,mean=mean,q16=q16,median=med,q84=q84,nrows=len(frame),weight_sum=weights.sum(),source_zip=arch))
        print(name, len(frame), 'samples (including weighted rows)')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,index=False,float_format='%.12g')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(20260626)
    fig,ax=plt.subplots(figsize=(8.4,5.2))
    for name,frame in frames.items():
        w=frame.weight.to_numpy(dtype=float)
        n=min(1800,len(frame))
        pick=rng.choice(len(frame),size=n,replace=True,p=w/w.sum())
        ax.scatter(frame.n_s.to_numpy()[pick],frame.r.to_numpy()[pick],s=8,alpha=.17,rasterized=True,label=f'{name} (weighted posterior preview)')
    ngrid=np.linspace(50,60,300)
    ax.plot(1-2/ngrid-9/(2*ngrid**2),12/ngrid**2,linewidth=2,label='SR1 analytical trajectory')
    ax.plot(1-2/ngrid-3/(2*ngrid**2),(12/ngrid**2)*(1-2/ngrid),linewidth=2,label='SR2 analytical trajectory')
    ax.set_xlabel(r'$n_s$ at $k_*=0.05\,\mathrm{Mpc}^{-1}$')
    ax.set_ylabel(r'$r_{0.05}$')
    ax.set_title('Primordial parameter space — original PolyChord samples')
    ax.set_xlim(0.935,0.995);ax.set_ylim(0,.085)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8,loc='upper left')
    fig.text(.5,.01,'Weighted random visualization only; use the complete chains for inference and credible regions.',ha='center',fontsize=8)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.figure,bbox_inches='tight')
    plt.close(fig)
    print('Saved',args.output,args.figure)

if __name__=='__main__':
    main()
