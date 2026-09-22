"""Independent returned trajectory, identity, summary and path sanity checks."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
fix = root.parent
receipt = root/'receipt_v02'
accounting = {x['JobID']: x for x in csv.DictReader((receipt/'logs/accounting.tsv').open(),delimiter='|')}
config = json.loads((fix/'config.json').read_text())
reports = []
for arm in ('published','shapeit419'):
    for mode in ('mle','neutral'):
        directory = receipt/'results'/f'{arm}_{mode}'
        identity = json.loads((directory/'identity.json').read_text())
        validation = json.loads((directory/'validation.json').read_text())
        job = accounting[f"{identity['array_job']}_{identity['array_task']}"]
        assert job['State']=='COMPLETED' and job['ExitCode']=='0:0' and int(job['AllocCPUS'])==4
        assert (directory/'NUMERICAL.PASS').read_text().strip()=='PASS' and validation['PASS']
        for name,digest in identity['source_sha256'].items():
            assert hashlib.sha256((directory/'code_snapshot'/name).read_bytes()).hexdigest()==digest
            local = fix/name
            if name.startswith('results_r03/'):
                local = fix/'receipt_v02'/name
            assert hashlib.sha256(local.read_bytes()).hexdigest()==digest, name
        spec = next(x for x in config['runs'] if x['arm']==arm and x['df']==1800)
        assert identity['input_sha256']==spec['input_sha256'] and identity['parameters']==config['parameters']
        profile = json.loads((fix/'receipt_v02/results_r03'/f'{arm}_df1800/summary.json').read_text())
        assert identity['s']==(profile['at_best']['s'] if mode=='mle' else 0.)
        assert abs(validation['ESS']-(profile['at_best']['ESS'] if mode=='mle' else 200))<1e-7
        assert validation['branches']==200 and not validation['selection_uncertainty_included']
        assert validation['forward_backward_evidence_error']<1e-7
        assert validation['full_range_posterior_error']<1e-8
        data = np.load(directory/'posterior.npz')
        posterior, frequency, generation = data['posterior'],data['frequency'],data['generation']
        assert posterior.shape==(999,1800) and np.array_equal(generation,np.arange(1,1000))
        assert np.isfinite(posterior).all() and np.min(posterior)>=0
        assert np.max(np.abs(posterior.sum(axis=1)-1))<1e-8
        assert frequency[0]==0 and frequency[-1]==1 and np.all(np.diff(frequency)>0)
        mean = np.sum(posterior*frequency,axis=1)
        cumulative = posterior.cumsum(axis=1)
        quantiles = np.array([[frequency[np.searchsorted(row,p)] for row in cumulative] for p in (.025,.5,.975)])
        table = list(csv.DictReader((directory/'trajectory.tsv').open(),delimiter='\t'))
        expected = np.array([[float(r[k]) for k in ('mean','q025','median','q975')] for r in table])
        error = float(np.max(np.abs(expected-np.column_stack((mean,quantiles.T)))))
        assert error<1e-12
        paths = np.load(directory/'paths.npz')
        values, states = paths['frequency'], paths['state']
        assert np.min(states)>=0 and np.max(states)<len(frequency)
        assert states.shape==(512,999) and np.array_equal(values,frequency[states])
        assert np.array_equal(paths['generation'],generation)
        assert paths['branch_1based'].shape==(512,) and np.min(paths['branch_1based'])>=1 and np.max(paths['branch_1based'])<=200
        assert int(paths['seed'])==validation['path_seed']==20260921+int(identity['array_task'])
        # Analytic-mixture moments provide a distinct Monte Carlo sanity check.
        # Temporal draws are correlated; these are pointwise checks, not 999 trials.
        moments=[]
        for g in (25,50,100,200,500):
            i=g-1
            variance=float(np.sum(posterior[i]*(frequency-mean[i])**2))
            se=np.sqrt(variance/512)
            delta=abs(float(values[:,i].mean())-mean[i])
            pzero=float(posterior[i,0])
            # A normalized floating-point mixture can exceed one by roundoff.
            zero_se=np.sqrt(max(0.,pzero*(1-pzero))/512)
            zero_delta=abs(float(np.mean(values[:,i]==0))-pzero)
            assert delta<=6*se+1e-12 and zero_delta<=6*zero_se+2/512,(arm,mode,g)
            moments.append(dict(generation=g,mean_error=float(delta),analytic_se=float(se),pzero_error=zero_delta))
        previous=root.parents[1]/'trajectory_v01/receipt_v01/results'/f'{arm}_{mode}'
        old_identity=json.loads((previous/'identity.json').read_text())
        old_posterior=np.load(previous/'posterior.npz')
        np.testing.assert_array_equal(frequency,old_posterior['frequency'])
        np.testing.assert_array_equal(generation,old_posterior['generation'])
        reports.append(dict(arm=arm,mode=mode,s=identity['s'],summary_error=error,
                            same_s_as_v01=identity['s']==old_identity['s'],
                            v01_posterior_max_change=float(np.max(np.abs(posterior-old_posterior['posterior']))),
                            moments=moments,PASS=True))
with (receipt/'independent_audit.json').open('x') as stream:
    json.dump(dict(PASS=True,results=reports,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),stream,indent=2)
print(json.dumps(reports))
