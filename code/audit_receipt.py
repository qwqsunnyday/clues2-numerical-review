"""Independently verify returned aggregate results without invoking the HMM."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys


def logsumexp(values):
    largest=max(values)
    return largest+math.log(math.fsum(math.exp(x-largest) for x in values))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--baseline-grid',type=Path)
    args=p.parse_args()
    receipt=args.receipt.resolve()
    roots=sorted(receipt.glob('results_r02/*_df*'),
                 key=lambda path:(path.name.rsplit('_df',1)[0],int(path.name.rsplit('_df',1)[1])))
    assert roots, 'No result directories found'
    config=json.loads((receipt/'config.json').read_text())
    expected={(r['arm'],r['df']):r for r in config['runs']}
    found=set()
    jobs={}
    if not args.allow_partial:
        assert len(roots)==len(expected)==6
        with (receipt/'logs/accounting_final_v01.tsv').open() as stream:
            jobs={r['JobID']:r for r in csv.DictReader(stream,delimiter='|')}
        assert jobs, 'Final job accounting is empty'
    summaries=[]
    fixed_points={}
    shared_sources=None
    for root in roots:
        assert (root/'NUMERICAL.PASS').read_text().strip()=='PASS'
        result=json.loads((root/'summary.json').read_text())
        identity=json.loads((root/'identity.json').read_text())
        if shared_sources is None:
            shared_sources=identity['source_sha256']
        assert identity['source_sha256']==shared_sources, 'Source changed between configurations'
        key=(result['arm'],result['df'])
        assert key in expected and key not in found
        found.add(key)
        assert (identity['arm'],identity['df'])==key
        assert identity['input_sha256']==expected[key]['input_sha256']
        assert identity['parameters']==config['parameters']
        if jobs:
            job=jobs[f"{identity['array_job']}_{identity['array_task']}"]
            assert job['State']=='COMPLETED' and job['ExitCode']=='0:0'
            assert int(job['AllocCPUS'])==identity['workers']==4
        assert result['people']==2048 and result['AN']==4096 and result['original_AC']==24
        assert result['original_derived']==identity['original_derived']==24
        assert result['converted_derived']==identity['converted_derived']=={'published':22,'shapeit419':23}[key[0]]
        assert result['flips']==identity['flips']==24-result['converted_derived']
        for relative,digest in identity['source_sha256'].items():
            assert hashlib.sha256((root/'code_snapshot'/relative).read_bytes()).hexdigest()==digest
        assert result['branch_samples']==200 and result['validation']['PASS']
        progress=json.loads((root/'progress.json').read_text())
        grouped={}
        with (root/'branch_contributions.tsv').open() as stream:
            for row in csv.DictReader(stream,delimiter='\t'):
                i=int(row['evaluation']);b=int(row['branch_1based'])
                assert b not in grouped.get(i,{})
                assert float(row['s'])==progress[i]['s']
                grouped.setdefault(i,{})[b]=float(row['log_ratio'])
        assert len(grouped)==len(progress)==result['profile']['evaluations']
        max_error=0.
        for r in progress:
            contributions=grouped[r['evaluation']]
            assert set(contributions)==set(range(1,201))
            values=list(contributions.values());total=logsumexp(values)
            weights=[math.exp(x-total) for x in values]
            ll=total-math.log(200)
            ess=1/math.fsum(w*w for w in weights)
            assert abs(ll-r['logLR'])<1e-8
            assert abs(ess-r['ESS'])<1e-8
            assert abs(max(weights)-r['max_weight'])<1e-10
            max_error=max(max_error,abs(ll-r['logLR']))
            fixed_points[(result['arm'],result['df'],round(r['s'],14))]=r
        profile=result['profile'];best=max(progress,key=lambda r:(r['logLR'],-abs(r['s'])))
        assert abs(profile['logLR']-best['logLR'])<1e-12
        assert abs(profile['best_s']-best['s'])<1e-10
        assert result['at_best']==best
        points={r['s']:r['logLR'] for r in progress}
        assert points=={r['s']:r['logLR'] for r in profile['likelihood_points']}
        assert abs(points[0.])<1e-10
        validation=result['validation']
        assert validation==json.loads((root/'validation.json').read_text())
        assert validation['parallel_serial_max_error']==0
        for field in ('oracle_neutral_max_error','oracle_best_ratio_max_error','repeat_best_max_error'):
            assert validation[field]<1e-7
        assert abs(validation['oracle_best_logLR']-profile['logLR'])<1e-7
        assert validation['all_200_draws_retained'] and validation['best_matches_trace']
        assert not result['biological_acceptance'] and not result['trajectory_generated']
        assert any(c['lower']-1e-5<=profile['best_s']<=c['upper']+1e-5 for c in profile['confidence_components'])
        assert abs(profile['profile_drop']-NormalDist().inv_cdf(.975)**2/2)<1e-12
        assert all(abs(r['residual'])<=.002 for r in profile['confidence_roots'])
        for r in profile['confidence_roots']:
            evaluated=min(progress,key=lambda x:abs(x['s']-r['s']))
            assert abs(evaluated['s']-r['s'])<1e-10
            assert abs(evaluated['logLR']-(profile['logLR']-profile['profile_drop']))<.002
        summaries.append(dict(arm=result['arm'],df=result['df'],s=profile['best_s'],logLR=profile['logLR'],
                              confidence_components=json.dumps(profile['confidence_components'],separators=(',',':')),
                              contains_zero=any(c['lower']<=0<=c['upper'] for c in profile['confidence_components']),
                              ESS_at_best=result['at_best']['ESS'],max_weight_at_best=result['at_best']['max_weight'],
                              evaluations=profile['evaluations'],boundary=profile['optimum_at_boundary'],
                              max_logLR_recalc_error=max_error,numerical_PASS=True,biological_acceptance=False))
    if not args.allow_partial:
        assert found==set(expected)
    baseline_identity=None
    comparisons=[]
    if args.baseline_grid:
        baseline_identity=dict(path=str(args.baseline_grid.resolve()),
                               sha256=hashlib.sha256(args.baseline_grid.read_bytes()).hexdigest())
        with args.baseline_grid.open() as stream:
            for r in csv.DictReader(stream,delimiter='\t'):
                key=(r['arm'],int(r['df']),round(float(r['selection']),14))
                if key[:2] not in found:
                    assert args.allow_partial
                    continue
                fixed=fixed_points[key]
                comparisons.append(dict(arm=key[0],df=key[1],s=key[2],baseline_index=int(r['index']),
                                         old_logLR=float(r['logLR']),fixed_logLR=fixed['logLR'],
                                         old_ESS=float(r['ESS']),fixed_ESS=fixed['ESS'],
                                         old_max_weight=float(r['max_weight_fraction']),fixed_max_weight=fixed['max_weight']))
        assert comparisons
    dest=receipt/'independent_audit.json'
    with dest.open('x') as stream:
        json.dump(dict(status='PASS',configurations=len(summaries),results=summaries,
                       baseline_grid=baseline_identity,
                       audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       audit_python=sys.version),stream,indent=2)
        stream.write('\n')
    with (receipt/'auditor_snapshot.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    with (receipt/'comparison.tsv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]),delimiter='\t')
        writer.writeheader();writer.writerows(summaries)
    if comparisons:
        with (receipt/'before_after_grid.tsv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]),delimiter='\t')
            writer.writeheader();writer.writerows(comparisons)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
