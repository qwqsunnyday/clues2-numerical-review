"""Conditional branch-mixture smoothing, with explicit per-branch normalization.

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
    out=root/'results'/f'{arm}_{mode}'
    out.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')
    sys.path.insert(0,str(fix/'src'))
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    import scipy
    from scipy.special import logsumexp
    from scipy.stats import norm
    from posterior_paths import sample_paths
    import upstream_io
    import trajectory_hmm as hmm
    import hmm_full_range as oracle
    config=json.loads((fix/'config.json').read_text())
    spec=next(x for x in config['runs'] if x['arm']==arm and x['df']==1800)
    previous=fix/'results_r03'/f'{arm}_df1800'
    summary=json.loads((previous/'summary.json').read_text())
    assert summary['validation']['PASS']
    s=summary['at_best']['s'] if mode=='mle' else 0.
    source=fix/'src/hmm_utils.py'
    expected=source.read_text().replace('cache=True)', 'cache=True, nogil=True)')
    assert (root/'src/trajectory_hmm.py').read_text()==expected
    times_path=(fix/config['input_root_relative']).resolve()/'runs'/arm/'private/posterior_times.txt'
    assert sha(times_path)==spec['input_sha256']
    paths=[root/'code/run_trajectory.py',root/'code/run.sbatch',root/'src/trajectory_hmm.py',root/'src/posterior_paths.py',
           fix/'src/upstream_io.py',fix/'src/hmm_utils.py',fix/'src/hmm_full_range.py',fix/'config.json',
           previous/'summary.json',previous/'branch_contributions.tsv']
    identity=dict(run_id='conditional_trajectory_v02',arm=arm,mode=mode,s=s,df=1800,
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
    assert len(ratios)==200
    weights=np.exp(ratios-logsumexp(ratios))

    seed=20260921+args.task
    rng=np.random.default_rng(seed)
    # Draw a branch once per entire path, not independently each generation.
    selected_branches=rng.choice(200,size=512,p=weights)
    uniforms=rng.random((512,len(epochs)-1))
    path_states=np.full(uniforms.shape,-1,dtype=np.int64)

    def backward(i,module):
        return module.backward_algorithm(sel,times[:,:,i],dt,at,epochs,ne,h,freqs,lf,l1f,z,cdf,cdf,
            anc,haps,transition,noCoals=0,precomputematrixboolean=1,currFreq=current)

    def one(i):
        b=backward(i,hmm)
        a=hmm.forward_algorithm(sel,times[:,:,i],dt,at,epochs,ne,h,freqs,lf,l1f,z,cdf,cdf,anc,haps,noCoals=0)
        joint=a[1:]+b[:-1]
        norm=logsumexp(joint,axis=1)
        # Constant evidence across time independently checks forward/backward
        # indexing, coalescent emissions, and the uniform terminal factor.
        ll=float(logsumexp(b[-2]))
        err=float(np.max(np.abs(norm-(ll-np.log(len(freqs))))))
        assert err<1e-7, (i,err)
        post=np.exp(joint-norm[:,None])
        oracle_error=0.
        if i in (15,150,153):
            ref=a[1:]+backward(i,oracle)[:-1]
            ref=np.exp(ref-logsumexp(ref,axis=1)[:,None])
            oracle_error=float(np.max(np.abs(post-ref)))
            assert oracle_error<1e-8
        slots=np.flatnonzero(selected_branches==i)
        states=sample_paths(b[:-1],transition,uniforms[slots])
        return post,err,oracle_error,ll,slots,states

    mixture=np.zeros((len(epochs)-1,len(freqs)))
    errors=[];oracle_errors=[];scores=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        # Bounded batches avoid keeping 200 full posterior matrices in RAM.
        for start in range(0,200,4):
            for i,(post,err,oe,ll,slots,states) in zip(range(start,start+4),pool.map(one,range(start,start+4))):
                mixture+=weights[i]*post
                path_states[slots]=states
                errors.append(err);oracle_errors.append(oe);scores.append(ll)
            print(json.dumps(dict(completed=start+4,arm=arm,mode=mode,max_evidence_error=max(errors))),flush=True)
    assert np.isfinite(mixture).all() and np.min(mixture)>=0
    mass_error=float(np.max(np.abs(mixture.sum(axis=1)-1)))
    assert mass_error<1e-8
    np.savez_compressed(out/'posterior.npz',frequency=freqs,generation=np.arange(1,len(epochs)),posterior=mixture)
    assert np.min(path_states)>=0 and np.max(path_states)<len(freqs)
    np.savez_compressed(out/'paths.npz',frequency=freqs[path_states],generation=np.arange(1,len(epochs)),
                        state=path_states,branch_1based=selected_branches+1,seed=seed)
    mean=mixture@freqs
    cumulative=mixture.cumsum(axis=1)
    q=[freqs[(cumulative>=v).argmax(axis=1)] for v in (.025,.5,.975)]
    with (out/'trajectory.tsv').open('x',newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n')
        w.writerow(['arm','mode','s','generation','mean','q025','median','q975'])
        w.writerows((arm,mode,s,i+1,mean[i],q[0][i],q[1][i],q[2][i]) for i in range(len(mean)))
    validation=dict(PASS=True,branches=200,ESS=float(1/(weights@weights)),
        normalization_error=mass_error,forward_backward_evidence_error=max(errors),
        full_range_posterior_error=max(oracle_errors),shape=list(mixture.shape),
        path_count=512,path_seed=seed,path_sampler="Backward sampling of complete conditional paths; fixed branch per path",
        selection_uncertainty_included=False,biological_acceptance=False,
        interval='Pointwise equal-tail 95% conditional posterior; fixed s, topology, demography and recoding',
        end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (out/'NUMERICAL.PASS').write_text('PASS\n')


if __name__=='__main__':
    main()
