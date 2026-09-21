"""Run the fixed one-epoch modern-data likelihood on all frozen branch draws.

All draws remain in the estimator. Parallelism preserves their order. New s/CI
outputs are separate from the frozen upstream runs and do not imply calibrated
biological inference. No old Gaussian trajectory integration is called.
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    args.config=args.config.resolve()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Real HMM runs require a Slurm job')
    root=Path(__file__).resolve().parents[1]
    config=json.loads(args.config.read_text())
    spec=config['runs'][args.index]
    arm,df=spec['arm'],spec['df']
    old=root.parent
    output_root=args.output_root.resolve() if args.output_root else root/'results'
    out=output_root/f'{arm}_df{df}'
    out.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    import scipy
    import numba
    from scipy.special import logsumexp
    from scipy.stats import norm
    import upstream_io
    import hmm_utils as fixed
    import hmm_full_range as oracle
    from profile_search import fit_profile

    source_id=json.loads((root/'source_identity.json').read_text())
    for name,digest in source_id['files'].items():
        assert sha(root/'src'/name)==digest
    times_path=old/'runs'/arm/'private/posterior_times.txt'
    assert sha(times_path)==spec['input_sha256']
    metadata=json.loads((old/'public'/f'{arm}.json').read_text())
    assert metadata['conversion']['samples']==200
    assert metadata['AC']==24 and metadata['AN']==4096
    code_paths=list((root/'src').glob('*.py'))+list((root/'code').glob('*.py'))+list((root/'tests').glob('*.py'))+[root/'code/run.sbatch',args.config,root/'source_identity.json']
    identity=dict(run_id='clues_numerical_fix_v01',arm=arm,df=df,job=os.environ['SLURM_JOB_ID'],
                  array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),
                  restart=os.environ.get('SLURM_RESTART_COUNT','0'),argv=sys.argv,cwd=os.getcwd(),
                  start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),python=sys.version,
                  numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,workers=args.workers,
                  input_sha256=sha(times_path),parameters=config['parameters'],
                  source_sha256={str(p.relative_to(root)):sha(p) for p in code_paths},
                  original_derived=metadata['conversion']['original_derived'],
                  converted_derived=metadata['conversion']['converted_derived'],
                  flips=metadata['conversion']['flips'])
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    for p in code_paths:
        dest=out/'code_snapshot'/p.relative_to(root)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)

    scientific=config['parameters']
    namespace=argparse.Namespace(times=str(times_path),popFreq=scientific['popFreq'],
                                 ancientSamps=None,ancientHaps=None,N=scientific['N_haploid'],
                                 coal=None,tCutoff=float(scientific['tCutoff']),timeBins=None,df=df,h=scientific['h'])
    (_,times,epochs,ne,freqs,ancient,ancient_haps,no_coals,current,logfreqs,log1freqs,
     derived_times,ancestral_times,h)=upstream_io.load_data(namespace)
    ne*=.5  # Required upstream conversion to the internal diploid parameter.
    assert times.shape[2]==200 and not no_coals
    z_probs=np.linspace(0.,1.,2000);z_probs[0]=1e-10;z_probs[-1]=1-1e-10
    z_bins=norm.ppf(z_probs);z_cdf=norm.cdf(z_bins)
    pool=ThreadPoolExecutor(max_workers=args.workers)

    def scores(s,module=fixed,indices=None,serial=False):
        selection=np.full(len(epochs),float(s))
        transition=module._nstep_log_trans_prob(ne[0],float(s),freqs,z_bins,z_cdf,z_cdf,h)
        def one(branch):
            matrix=module.backward_algorithm(selection,times[:,:,branch],derived_times,ancestral_times,
                      epochs,ne,h,freqs,logfreqs,log1freqs,z_bins,z_cdf,z_cdf,ancient,ancient_haps,
                      transition,noCoals=0,precomputematrixboolean=1,currFreq=current)
            return float(logsumexp(matrix[-2,:]))
        which=list(range(200)) if indices is None else list(indices)
        values=np.array(list(map(one,which)) if serial else list(pool.map(one,which)))
        assert np.isfinite(values).all()
        return values

    # Numerical reference gates precede the full estimator. The original three
    # problematic draws are tested against the prior full-range diagnostic.
    checks={}
    selected=[15,150,153]
    checks['parallel_serial_max_error']=float(np.max(np.abs(scores(0.,indices=selected)-scores(0.,indices=selected,serial=True))))
    assert checks['parallel_serial_max_error']==0
    if arm=='published':
        prior=old/'diagnostics/hmm_state_v01/branch_diagnostics.tsv'
        reference=list(csv.DictReader(prior.open(),delimiter='\t'))
        discrepancy=[]
        for s in (0.,-.1,-.03,.01):
            observed=scores((1.+s)-1.,indices=selected)
            expected=[float(next(r['log_score'] for r in reference if r['mode']=='full_range' and int(r['df'])==df
                                 and float(r['selection'])==s and int(r['branch_1based'])==b+1)) for b in selected]
            discrepancy.extend(np.abs(observed-np.array(expected)).tolist())
        checks['prior_regression_max_error']=max(discrepancy)
        assert checks['prior_regression_max_error']<1e-7

    neutral=scores(0.)
    cache={}
    ratios_by_s={}
    evaluation_rows=[]
    trace=(out/'branch_contributions.tsv').open('x')
    trace.write('evaluation\ts\tbranch_1based\tlog_ratio\n')
    def evaluate(s):
        key=round(float(s),14)
        if key not in cache:
            started=time.monotonic()
            ratios=np.zeros(200) if key==0 else scores(key)-neutral
            ll=float(logsumexp(ratios)-np.log(200))
            normalized=np.exp(ratios-logsumexp(ratios))
            record=dict(evaluation=len(cache),s=key,logLR=ll,ESS=float(1/np.square(normalized).sum()),
                        max_weight=float(normalized.max()),seconds=time.monotonic()-started)
            cache[key]=record;ratios_by_s[key]=ratios
            for i,v in enumerate(ratios):
                trace.write(f"{record['evaluation']}\t{key:.14g}\t{i+1}\t{v:.17g}\n")
            trace.flush();evaluation_rows.append(record)
            (out/'progress.json').write_text(json.dumps(evaluation_rows,indent=2)+'\n')
            print(json.dumps(record),flush=True)
        return cache[key]['logLR']
    try:
        profile=fit_profile(evaluate,config['search']['grid'],confidence=config['search']['confidence'],
                            xtol=config['search']['xtol'],max_evaluations=config['search']['max_evaluations'])
        best=round(profile['best_s'],14)
        # Both the numerator and neutral denominator are independently checked
        # on ALL 200 branch draws with full frequency updates.
        neutral_oracle=scores(0.,module=oracle)
        best_oracle=scores(best,module=oracle)
        oracle_ratios=best_oracle-neutral_oracle
        checks['oracle_neutral_max_error']=float(np.max(np.abs(neutral-neutral_oracle)))
        checks['oracle_best_ratio_max_error']=float(np.max(np.abs(ratios_by_s[best]-oracle_ratios)))
        checks['oracle_best_logLR']=float(logsumexp(oracle_ratios)-np.log(200))
        checks['neutral_logLR']=evaluate(0.)
        checks['best_matches_trace']=profile['logLR']==max(x['logLR'] for x in evaluation_rows)
        checks['all_200_draws_retained']=len(neutral)==200
        checks['repeat_best_max_error']=float(np.max(np.abs(scores(best)-neutral-ratios_by_s[best])))
        checks['PASS']=(checks['oracle_neutral_max_error']<1e-7 and checks['oracle_best_ratio_max_error']<1e-7
                        and checks['best_matches_trace'] and checks['repeat_best_max_error']<1e-7
                        and abs(checks['neutral_logLR'])<1e-10)
        (out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
        result=dict(status='COMPLETE',arm=arm,df=df,people=2048,AN=4096,original_AC=24,branch_samples=200,
                    profile=profile,at_best=cache[best],validation=checks,flips=identity['flips'],
                    original_derived=identity['original_derived'],converted_derived=identity['converted_derived'],
                    biological_acceptance=False,trajectory_generated=False,
                    limitation='Conditional on recoded branch, one topology and specified demographic model; cross-df/importance-sampling calibration pending',
                    end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print('SUMMARY',json.dumps({k:v for k,v in result.items() if k!='profile'}),flush=True)
        assert checks['PASS'],'Independent full-range validation failed'
        (out/'NUMERICAL.PASS').write_text('PASS\n')
    finally:
        trace.close();pool.shutdown()


if __name__=='__main__':
    main()
