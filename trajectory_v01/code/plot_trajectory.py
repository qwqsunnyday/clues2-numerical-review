"""Render one phase's conditional MLE trajectory and neutral sensitivity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/REVIEW_ENV/figure_helpers')
from figure_style import new_figure,save_figure

p=argparse.ArgumentParser()
p.add_argument('--arm',choices=['published','shapeit419'],required=True)
args=p.parse_args()
fig,ax=new_figure(183,100)
rows=[];checks={}
for mode,color,style in [('mle','#3B6FB6','-'),('neutral','#777777','--')]:
    d=ROOT/'receipt_v01/results'/f'{args.arm}_{mode}'
    assert (d/'NUMERICAL.PASS').read_text().strip()=='PASS'
    identity=json.loads((d/'identity.json').read_text())
    validation=json.loads((d/'validation.json').read_text())
    assert validation['PASS'] and validation['branches']==200
    a=np.load(d/'posterior.npz')
    post=a['posterior'];freq=a['frequency'];g=a['generation']
    assert post.shape==(999,1800) and np.array_equal(g,np.arange(1,1000))
    assert np.isfinite(post).all() and np.min(post)>=0 and np.max(np.abs(post.sum(axis=1)-1))<1e-8
    assert np.all(np.diff(freq)>0) and freq[0]==0 and freq[-1]==1
    mean=np.sum(post*freq[None,:],axis=1)
    cum=post.cumsum(axis=1)
    qs=np.array([[freq[np.searchsorted(cum[i],q)] for i in range(len(g))] for q in (.025,.5,.975)])
    remote=list(csv.DictReader((d/'trajectory.tsv').open(),delimiter='\t'))
    expected=np.array([[float(x[k]) for k in ['mean','q025','median','q975']] for x in remote])
    computed=np.column_stack([mean,qs.T])
    err=float(np.max(np.abs(computed-expected)))
    assert err<1e-12
    assert np.all(qs[0]<=qs[1]) and np.all(qs[1]<=qs[2])
    label=f"MLE: s = {identity['s']:.4f}" if mode=='mle' else 'Neutral: s = 0'
    ax.fill_between(g,100*qs[0],100*qs[2],color=color,alpha=.14,lw=0)
    ax.plot(g,100*mean,color=color,ls=style,lw=1.25,label=label)
    for i in range(len(g)):
        rows.append(dict(arm=args.arm,mode=mode,s=identity['s'],generation=int(g[i]),mean=mean[i],
                         q025=qs[0,i],median=qs[1,i],q975=qs[2,i]))
    checks[mode]=dict(independent_summary_error=err,validation=validation,
                     posterior_sha256=hashlib.sha256((d/'posterior.npz').read_bytes()).hexdigest())
ax.scatter([0],[100*24/4096],marker='D',s=16,color='black',zorder=5,label='Present input: 24/4,096')
ax.set_xlim(1000,-20)
upper=max(max(r['q975'] for r in rows)*100,100*24/4096)*1.07
ax.set_ylim(0,upper)
ax.set_xlabel('Generations before present')
ax.set_ylabel('Conditional allele frequency (%)')
ax.legend(loc='upper left')
source=ROOT/'results/source_data'/f'{args.arm}_trajectory_v01.tsv'
source.parent.mkdir(parents=True,exist_ok=True)
with source.open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
base=ROOT/'figures/draft'/f'20260921_clues_fixed_{args.arm}_trajectory_v01'
base.parent.mkdir(parents=True,exist_ok=True)
for ext in ('png','pdf','svg'):
    out=base.with_suffix('.'+ext)
    assert not out.exists()
    save_figure(fig,out)
checks['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
checks['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
checks['scientific_QA']='PASS for conditional visualization; biological selection not accepted'
(ROOT/'results'/f'{args.arm}_plot_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(dict(arm=args.arm,checks=checks)))
