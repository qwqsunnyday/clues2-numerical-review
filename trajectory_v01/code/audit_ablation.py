"""Controlled 2x2 audit of HMM reset and branch aggregation.

The fixed likelihood supplies importance weights L(s)/L(0). Each branch's
forward/backward product must first sum to one at each generation; otherwise
its likelihood is counted twice. Selection uncertainty is NOT marginalized.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',type=int,required=True)
    args=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'HMM computation requires Slurm'
    root=Path(__file__).resolve().parents[1]
    fix=root.parent
    arm=('published','shapeit419')[args.task//2]
    mode=('mle','neutral')[args.task%2]
    out=root/'results_ablation_v01'/arm
    out.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')
    sys.path.insert(0,str(fix/'src'))
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    import scipy
    from scipy.special import logsumexp
    from scipy.stats import norm
    import upstream_io
    import trajectory_hmm as hmm
    import legacy_hmm as legacy
    config=json.loads((fix/'config.json').read_text())
    spec=next(x for x in config['runs'] if x['arm']==arm and x['df']==1800)
    previous=fix/'results_r02'/f'{arm}_df1800'
    summary=json.loads((previous/'summary.json').read_text())
    assert summary['validation']['PASS']
    s=summary['at_best']['s'] if mode=='mle' else 0.
    source=fix/'src/hmm_utils.py'
    expected=source.read_text().replace('cache=True)', 'cache=True, nogil=True)')
    assert (root/'src/trajectory_hmm.py').read_text()==expected
    times_path=fix.parent/'runs'/arm/'private/posterior_times.txt'
    assert sha(times_path)==spec['input_sha256']
    paths=[root/'code/audit_ablation.py',root/'code/ablation.sbatch',root/'src/legacy_hmm.py',root/'src/trajectory_hmm.py',
           fix/'src/upstream_io.py',fix/'src/hmm_utils.py',fix/'src/hmm_full_range.py',fix/'config.json',
           previous/'summary.json',previous/'branch_contributions.tsv']
    identity=dict(run_id='trajectory_controlled_ablation_v01',arm=arm,mode=mode,s=s,df=1800,
                  job=os.environ['SLURM_JOB_ID'],array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),
                  array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),input_sha256=sha(times_path),
                  parameters=config['parameters'],python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                  source_sha256={str(x.relative_to(fix)):sha(x) for x in paths},
                  start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    for path in paths:
        dest=out/'code_snapshot'/path.relative_to(fix)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
    par=config['parameters']
    ns=argparse.Namespace(times=str(times_path),popFreq=par['popFreq'],ancientSamps=None,ancientHaps=None,
        N=par['N_haploid'],coal=None,tCutoff=float(par['tCutoff']),timeBins=None,df=1800,h=par['h'])
    (_,times,epochs,ne,freqs,anc,haps,no_coals,current,lf,l1f,dt,at,h)=upstream_io.load_data(ns)
    ne*=.5
    assert times.shape[2]==200 and not no_coals
    probs=np.linspace(0,1,2000);probs[0]=1e-10;probs[-1]=1-1e-10
    z=norm.ppf(probs);cdf=norm.cdf(z)
    sel=np.full(len(epochs),s)
    transition=hmm._nstep_log_trans_prob(ne[0],s,freqs,z,cdf,cdf,h)
    records=list(csv.DictReader((previous/'branch_contributions.tsv').open(),delimiter='\t'))
    ratios=np.array([float(r['log_ratio']) for r in records if abs(float(r['s'])-s)<1e-13])
    assert len(ratios)==200 and mode=='mle'
    neutral_sel=np.zeros(len(epochs))
    neutral_transition=legacy._nstep_log_trans_prob(ne[0],0.,freqs,z,cdf,cdf,h)
    def backward(i,module,selection,trans):
        return module.backward_algorithm(selection,times[:,:,i],dt,at,epochs,ne,h,freqs,lf,l1f,z,cdf,cdf,
            anc,haps,trans,noCoals=0,precomputematrixboolean=1,currFreq=current)
    def one(i):
        bf=backward(i,hmm,sel,transition)
        bo=backward(i,legacy,sel,transition)
        bn=backward(i,legacy,neutral_sel,neutral_transition)
        a=hmm.forward_algorithm(sel,times[:,:,i],dt,at,epochs,ne,h,freqs,lf,l1f,z,cdf,cdf,anc,haps,noCoals=0)
        jf=a[1:]+bf[:-1];jo=a[1:]+bo[:-1]
        nf=logsumexp(jf,axis=1);no=logsumexp(jo,axis=1)
        llf=float(logsumexp(bf[-2]));llo=float(logsumexp(bo[-2]));ll0=float(logsumexp(bn[-2]))
        fixed_error=float(np.max(np.abs(nf-(llf-np.log(len(freqs))))))
        assert fixed_error<1e-7
        old_ratio=llo-ll0
        # Factor 1: old/reset backward state update. Factor 2: old raw joint
        # aggregation / normalized branch posterior. Everything else fixed.
        arrays=[jo+old_ratio,jo-no[:,None]+old_ratio,jf+ratios[i],jf-nf[:,None]+ratios[i]]
        metrics=dict(branch=i+1,old_log_ratio=old_ratio,fixed_log_ratio=float(ratios[i]),
            old_evidence_spread=float(np.ptp(no)),fixed_evidence_error=fixed_error)
        return arrays,metrics
    names=['old_hmm_old_aggregation','old_hmm_normalized_aggregation','reset_hmm_old_aggregation','reset_hmm_normalized_aggregation']
    mixtures=[np.full((len(epochs)-1,len(freqs)),-np.inf) for _ in names]
    metrics=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,200,4):
            for arrays,record in pool.map(one,range(start,start+4)):
                for j in range(4):mixtures[j]=np.logaddexp(mixtures[j],arrays[j])
                metrics.append(record)
            print(json.dumps(dict(arm=arm,completed=start+4)),flush=True)
    rows=[];curves={}
    for name,mix in zip(names,mixtures):
        post=np.exp(mix-logsumexp(mix,axis=1)[:,None])
        assert np.isfinite(post).all() and np.max(np.abs(post.sum(axis=1)-1))<1e-8
        curves[name]=post@freqs
        cumulative=post.cumsum(axis=1)
        quantiles=[freqs[(cumulative>=q).argmax(axis=1)] for q in (.025,.5,.975)]
        np.savez_compressed(out/(name+'.npz'),frequency=freqs,posterior=post)
        rows.extend(dict(arm=arm,variant=name,s=s,generation=i+1,mean=curves[name][i],q025=quantiles[0][i],median=quantiles[1][i],q975=quantiles[2][i]) for i in range(len(post)))
    existing=np.load(root/'results'/f'{arm}_mle/posterior.npz')['posterior']
    reproduced=np.load(out/(names[-1]+'.npz'))['posterior']
    error=float(np.max(np.abs(existing-reproduced)))
    assert error<1e-8
    with (out/'trajectory.tsv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    (out/'branch_checks.json').write_text(json.dumps(metrics,indent=2)+'\n')
    comparisons={}
    for a,b in [(names[0],names[2]),(names[2],names[3]),(names[0],names[3])]:
        comparisons[a+' -> '+b]=float(np.max(np.abs(curves[a]-curves[b]))*100)
    validation=dict(PASS=True,draws=200,df=1800,s=s,reproduces_current_posterior_max_error=error,
        max_mean_differences_percentage_points=comparisons,
        max_old_evidence_spread=max(x['old_evidence_spread'] for x in metrics),
        max_fixed_evidence_error=max(x['fixed_evidence_error'] for x in metrics),
        limitation='Both phases held at their repaired MLE; not a test of all s or old Gaussian s integration')
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (out/'NUMERICAL.PASS').write_text('PASS\n')

if __name__=='__main__':
    main()
