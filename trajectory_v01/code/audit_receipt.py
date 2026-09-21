"""Verify all four completed trajectory identities against frozen local sources."""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
fix=root.parent
receipt=root/'receipt_v01'
accounting=list(csv.DictReader((receipt/'logs/accounting.tsv').open(),delimiter='|'))
audit=[]
for arm in ('published','shapeit419'):
    for mode in ('mle','neutral'):
        out=receipt/'results'/f'{arm}_{mode}'
        identity=json.loads((out/'identity.json').read_text())
        val=json.loads((out/'validation.json').read_text())
        assert (out/'NUMERICAL.PASS').read_text().strip()=='PASS'
        key=f"{identity['array_job']}_{identity['array_task']}"
        state=next(x for x in accounting if x['JobID']==key)
        assert state['State']=='COMPLETED' and state['ExitCode']=='0:0' and state['AllocCPUS']=='4'
        assert val['PASS'] and val['branches']==200 and val['shape']==[999,1800]
        assert val['normalization_error']<1e-8 and val['forward_backward_evidence_error']<1e-7
        assert val['full_range_posterior_error']<1e-8 and not val['selection_uncertainty_included']
        for name,digest in identity['source_sha256'].items():
            snap=out/'code_snapshot'/name
            assert hashlib.sha256(snap.read_bytes()).hexdigest()==digest
            local=fix/name
            if name.startswith('results_r02/'):
                local=fix/'receipt_complete_v01'/name
            assert hashlib.sha256(local.read_bytes()).hexdigest()==digest,(name,'local differs')
        config=json.loads((fix/'config.json').read_text())
        spec=next(x for x in config['runs'] if x['arm']==arm and x['df']==1800)
        assert identity['input_sha256']==spec['input_sha256'] and identity['parameters']==config['parameters']
        profile=json.loads((fix/'receipt_complete_v01/results_r02'/f'{arm}_df1800/summary.json').read_text())
        expected=profile['at_best']['s'] if mode=='mle' else 0
        assert identity['s']==expected
        expected_ess=profile['at_best']['ESS'] if mode=='mle' else 200
        assert abs(val['ESS']-expected_ess)<1e-7
        audit.append(dict(arm=arm,mode=mode,job=key,s=expected,ESS=val['ESS'],PASS=True))
with (receipt/'independent_audit.json').open('x') as f:
    json.dump(dict(PASS=True,configurations=4,results=audit),f,indent=2)
print(json.dumps(audit))
