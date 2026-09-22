"""Independent read-only-array audit; freeze a new audit receipt once."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
fix=root.parent
receipt=root/'receipt_ablation_v01'
accounting=list(csv.DictReader((receipt/'logs/accounting_ablation.tsv').open(),delimiter='|'))
names=['old_hmm_old_aggregation','old_hmm_normalized_aggregation','reset_hmm_old_aggregation','reset_hmm_normalized_aggregation']
checks=[]
for arm in ('published','shapeit419'):
    out=receipt/'results_ablation_v01'/arm
    identity=json.loads((out/'identity.json').read_text())
    validation=json.loads((out/'validation.json').read_text())
    assert (out/'NUMERICAL.PASS').read_text().strip()=='PASS' and validation['PASS']
    key=f"{identity['array_job']}_{identity['array_task']}"
    state=next(x for x in accounting if x['JobID']==key)
    assert state['State']=='COMPLETED' and state['ExitCode']=='0:0' and state['AllocCPUS']=='4'
    for name,digest in identity['source_sha256'].items():
        p=out/'code_snapshot'/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
        local=fix/name
        if name.startswith('results_r02/'):
            local=fix/'receipt_complete_v01'/name
        assert hashlib.sha256(local.read_bytes()).hexdigest()==digest,name
    spec=next(x for x in json.loads((fix/'config.json').read_text())['runs'] if x['arm']==arm and x['df']==1800)
    assert identity['input_sha256']==spec['input_sha256']
    records=list(csv.DictReader((out/'trajectory.tsv').open(),delimiter='\t'))
    assert len(records)==3996
    means={};posts={}
    for name in names:
        a=np.load(out/(name+'.npz'));freq=a['frequency'];post=a['posterior']
        assert post.shape==(999,1800) and np.isfinite(post).all() and np.min(post)>=0
        assert np.max(np.abs(post.sum(axis=1)-1))<1e-8
        assert freq[0]==0 and freq[-1]==1 and np.all(np.diff(freq)>0)
        mean=np.sum(post*freq,axis=1)
        supplied=np.array([float(r['mean']) for r in records if r['variant']==name])
        assert np.max(np.abs(mean-supplied))<1e-12
        posts[name]=post;means[name]=mean
    previous=np.load(root/'receipt_v01/results'/f'{arm}_mle/posterior.npz')['posterior']
    reproduction=float(np.max(np.abs(previous-posts[names[-1]])))
    assert reproduction<1e-8
    comparisons=[]
    for a,b in [(names[0],names[2]),(names[1],names[3]),(names[2],names[3]),(names[0],names[3])]:
        delta=np.abs(means[a]-means[b])*100
        comparisons.append(dict(before=a,after=b,max_mean_difference_pp=float(delta.max()),
            generation_at_max=int(delta.argmax()+1),max_posterior_total_variation=float((np.abs(posts[a]-posts[b]).sum(axis=1)*.5).max())))
    branch=json.loads((out/'branch_checks.json').read_text())
    assert len(branch)==200 and [x['branch'] for x in branch]==list(range(1,201))
    summary=dict(arm=arm,s=identity['s'],df=1800,PASS=True,job=key,
        posterior_reproduction_error=reproduction,comparisons=comparisons,
        max_old_evidence_spread=max(x['old_evidence_spread'] for x in branch),
        max_fixed_evidence_error=max(x['fixed_evidence_error'] for x in branch))
    checks.append(summary)
with (receipt/'independent_audit.json').open('x') as f:json.dump(checks,f,indent=2)
print(json.dumps(checks,indent=2))
