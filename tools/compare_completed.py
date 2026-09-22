"""Same-input v01/v02 likelihood comparison using returned full receipts."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
old = root.parent/'receipt_complete_v01/results_r02'
new = root/'receipt_v02/results_r03'
reports=[]
for path in sorted(new.glob('*_df*')):
    before=json.loads((old/path.name/'summary.json').read_text())
    after=json.loads((path/'summary.json').read_text())
    before_id=json.loads((old/path.name/'identity.json').read_text())
    after_id=json.loads((path/'identity.json').read_text())
    assert before_id['input_sha256']==after_id['input_sha256']
    assert before_id['parameters']==after_id['parameters']
    assert after['validation']['oracle_confidence_endpoint_max_error']<1e-7
    previous={float(r['s']):float(r['logLR']) for r in before['profile']['likelihood_points']}
    current={float(r['s']):float(r['logLR']) for r in after['profile']['likelihood_points']}
    common=previous.keys() & current.keys()
    def contributions(directory):
        return {(float(r['s']),int(r['branch_1based'])):float(r['log_ratio'])
                for r in csv.DictReader((directory/'branch_contributions.tsv').open(),delimiter='\t')}
    old_ratios=contributions(old/path.name)
    new_ratios=contributions(path)
    difference=max(abs(new_ratios[s,b]-old_ratios[s,b]) for s in common for b in range(1,201))
    reports.append(dict(run=path.name,matching_points=len(common),
        old_evaluations=len(previous),new_evaluations=len(current),
        max_logLR_change=max(abs(previous[s]-current[s]) for s in common),
        max_branch_log_ratio_change=difference,
        old_s=before['at_best']['s'],new_s=after['at_best']['s'],
        old_logLR=before['at_best']['logLR'],new_logLR=after['at_best']['logLR'],
        confidence_components_identical=before['profile']['confidence_components']==after['profile']['confidence_components'],
        confidence_reference_error=after['validation']['oracle_confidence_endpoint_max_error']))
assert len(reports)==6
with (root/'receipt_v02/v01_v02_comparison.json').open('x') as stream:
    json.dump(dict(configurations=6,results=reports),stream,indent=2)
print(json.dumps(reports,indent=2))
