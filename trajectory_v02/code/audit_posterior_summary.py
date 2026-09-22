"""Independent scalar checks of the saved decomposition and edge-case semantics."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from summarize_posterior import ROOT, ARMS, MODES, decompose, first_zero_quantiles


def main():
    # This exactly soluble mixture has a positive mean outside its [0,0] interval.
    frequency = np.array([0, .25, 1])
    posterior = np.array([[.98, .01, .01], [1, 0, 0], [.5, 0, .5]])
    summary, _ = decompose(posterior, frequency)
    assert summary['q975'][0] == 0 and summary['mean'][0] > 0
    np.testing.assert_allclose(summary['positive_mean'][0], .625)
    assert np.isnan(summary['positive_mean'][1]) and np.isnan(summary['positive_median'][1])
    times = first_zero_quantiles(np.array([1, 2, 3]), np.array([.1, .2, .6]))
    assert times['median'] == 3 and times['q975'] is None and times['q975_censored']
    errors, count = [], 0
    for arm in ARMS:
        path = ROOT/f'results/source_data/{arm}_posterior_decomposition_v03.tsv'
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        for mode in MODES:
            with np.load(ROOT/f'receipt_v02/results/{arm}_{mode}/posterior.npz') as data:
                frequency, posterior = data['frequency'], data['posterior']
            for generation in (1, 50, 100, 200, 500, 999):
                row = next(r for r in rows if r['mode'] == mode and int(r['generation']) == generation)
                weights = posterior[generation-1]
                weights = weights / weights.sum()
                mean = sum(float(w) * float(f) for w, f in zip(weights, frequency))
                errors.append(abs(mean - float(row['mean'])))
                positive = sum(float(v) for v in weights[1:])
                np.testing.assert_allclose(positive, float(row['p_positive']), rtol=1e-12)
                if positive > 0:
                    for q, key in ((.025, 'positive_q025'), (.5, 'positive_median'), (.975, 'positive_q975')):
                        cumulative = 0
                        for w, f in zip(weights[1:], frequency[1:]):
                            cumulative += float(w) / positive
                            if cumulative >= q:
                                break
                        assert f == float(row[key])
                else:
                    assert np.isnan(float(row['positive_mean']))
                count += 1
    assert max(errors) < 1e-14
    result = dict(exact_zero_mixture_PASS=True, undefined_positive_PASS=True, time_censoring_PASS=True,
                  independent_scalar_milestone_rows=count, maximum_mean_abs_error=max(errors),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), PASS=True)
    (ROOT/'results/posterior_decomposition_independent_check_v03.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
