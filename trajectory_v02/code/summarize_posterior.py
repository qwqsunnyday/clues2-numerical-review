"""Decompose existing posterior masses; no HMM rerun or path-tail estimator.

F=0 is absorbing backwards in this run. Therefore p_zero(t) is the CDF
of A = first recorded generation at zero, on the discrete 1..999 grid.
This model-conditioned hitting time is not a calibrated mutation age.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'v03'
ARMS = ('published', 'shapeit419')
MODES = ('mle', 'neutral')
QUANTILES = (.025, .5, .975)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discrete_quantiles(mass, frequency):
    cdf = np.cumsum(mass, axis=1)
    cdf[:, -1] = 1.0  # Prevent floating-point roundoff at the final bin.
    return np.array([frequency[(cdf >= q).argmax(axis=1)] for q in QUANTILES])


def decompose(raw, frequency):
    assert raw.ndim == 2 and raw.shape[1] == len(frequency)
    assert np.isfinite(raw).all() and (raw >= 0).all()
    assert frequency[0] == 0 and frequency[-1] == 1 and (np.diff(frequency) > 0).all()
    error = float(np.max(np.abs(raw.sum(axis=1) - 1)))
    assert error < 1e-8
    posterior = raw / raw.sum(axis=1, keepdims=True)
    # Sum positive bins directly: 1-p_zero loses precision in very small tails.
    positive = posterior[:, 1:].sum(axis=1)
    defined = positive > 0
    conditional = posterior[defined, 1:] / positive[defined, None]
    mean = posterior @ frequency
    q025, median, q975 = discrete_quantiles(posterior, frequency)
    positive_mean = np.full(len(mean), np.nan)
    positive_mean[defined] = conditional @ frequency[1:]
    positive_quantiles = np.full((3, len(mean)), np.nan)
    positive_quantiles[:, defined] = discrete_quantiles(conditional, frequency[1:])
    np.testing.assert_allclose(mean[defined], positive[defined] * positive_mean[defined], rtol=1e-12, atol=1e-15)
    assert (mean[~defined] == 0).all()
    result = dict(p_zero=posterior[:, 0], p_positive=positive, mean=mean,
                  q025=q025, median=median, q975=q975,
                  positive_mean=positive_mean, positive_q025=positive_quantiles[0],
                  positive_median=positive_quantiles[1], positive_q975=positive_quantiles[2],
                  positive_defined=defined, low_positive_mass=positive < .05)
    return result, error


def first_zero_quantiles(generation, p_zero):
    assert np.min(np.diff(p_zero)) >= -1e-12
    out = {}
    for name, q in zip(('q025', 'median', 'q975'), QUANTILES):
        indices = np.flatnonzero(p_zero >= q)
        out[name] = int(generation[indices[0]]) if len(indices) else None
        out[name + '_censored'] = not bool(len(indices))
    return out


def write_tsv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert (ROOT/'receipt_v02/independent_audit.json').is_file()
    output = ROOT/'results/source_data'
    output.mkdir(parents=True, exist_ok=True)
    checks_path = ROOT/f'results/posterior_decomposition_checks_{VERSION}.json'
    targets = [output/f'{arm}_posterior_decomposition_{VERSION}.tsv' for arm in ARMS]
    targets += [output/f'first_zero_times_{VERSION}.tsv', output/f'posterior_milestones_{VERSION}.tsv', checks_path]
    assert not any(p.exists() for p in targets), 'Preserve frozen outputs; choose a new version.'
    checks, time_rows, milestone_rows = {}, [], []
    kernel = ROOT.parent/'src/hmm_utils.py'
    for arm in ARMS:
        rows = []
        for mode in MODES:
            directory = ROOT/'receipt_v02/results'/f'{arm}_{mode}'
            identity = json.loads((directory/'identity.json').read_text())
            assert identity['source_sha256']['src/hmm_utils.py'] == sha(kernel)
            with np.load(directory/'posterior.npz') as data:
                generation, frequency, raw = data['generation'], data['frequency'], data['posterior']
                summary, row_error = decompose(raw, frequency)
            np.testing.assert_array_equal(generation, np.arange(1, 1000))
            time_summary = first_zero_quantiles(generation, summary['p_zero'])
            with np.load(directory/'paths.npz') as data:
                paths = data['frequency']
                np.testing.assert_array_equal(data['generation'], generation)
            # Every complete path must remain at zero after first entering it.
            zero = paths == 0
            assert (np.diff(zero.astype(int), axis=1) >= 0).all()
            empirical_cdf = zero.mean(axis=0)
            cdf_error = float(np.max(np.abs(empirical_cdf - summary['p_zero'])))
            # DKW bound, union across the four independently checked CDFs, alpha=.01.
            bound = float(np.sqrt(np.log(2 * len(ARMS) * len(MODES) / .01) / (2 * len(paths))))
            assert cdf_error <= bound
            for i, g in enumerate(generation):
                row = dict(arm=arm, mode=mode, s=identity['s'], generation=int(g))
                row.update({name: value[i].item() for name, value in summary.items()})
                rows.append(row)
                if g in (25, 50, 100, 200, 500, 999):
                    milestone_rows.append(row)
            time_rows.append(dict(arm=arm, mode=mode, s=identity['s'], **time_summary,
                                  last_generation=int(generation[-1]),
                                  p_first_zero_after_window=float(summary['p_positive'][-1])))
            checks[f'{arm}_{mode}'] = dict(
                posterior_sha256=sha(directory/'posterior.npz'), paths_sha256=sha(directory/'paths.npz'),
                identity_sha256=sha(directory/'identity.json'), kernel_sha256=sha(kernel),
                maximum_input_row_sum_error=row_error,
                decomposition_max_abs_error=float(np.max(np.abs(summary['mean'] - summary['p_positive'] * np.nan_to_num(summary['positive_mean'])))),
                p_zero_min_increment=float(np.diff(summary['p_zero']).min()),
                all_paths_zero_absorbing=True, path_count=len(paths),
                empirical_first_zero_cdf_max_error=cdf_error, simultaneous_dkw_bound=bound,
                undefined_positive_rows=int((~summary['positive_defined']).sum()),
                mean_outside_central95_rows=int(((summary['mean'] < summary['q025']) | (summary['mean'] > summary['q975'])).sum()),
                first_zero=time_rows[-1], PASS=True)
        write_tsv(output/f'{arm}_posterior_decomposition_{VERSION}.tsv', rows)
    write_tsv(output/f'first_zero_times_{VERSION}.tsv', time_rows)
    write_tsv(output/f'posterior_milestones_{VERSION}.tsv', milestone_rows)
    report = dict(summary_version=VERSION, input_analysis_run='conditional_trajectory_v02',
                  HMM_rerun=False, extra_smoothing=False, conditional_low_mass_display_threshold=.05,
                  undefined_conditional_value='nan; no positive posterior mass',
                  first_zero_definition='first zero at recorded generation, backwards in time; discrete CDF p_zero',
                  time_interpretation='conditional on fixed s, topology, recoding, demography and terminal bound; not selection onset or calibrated mutation age',
                  inputs=checks, script_sha256=sha(Path(__file__)),
                  sources={p.name:sha(p) for p in targets[:-1]}, PASS=True)
    checks_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(PASS=True, first_zero=time_rows, cdf_errors={k:v['empirical_first_zero_cdf_max_error'] for k,v in checks.items()})))


if __name__ == '__main__':
    main()
