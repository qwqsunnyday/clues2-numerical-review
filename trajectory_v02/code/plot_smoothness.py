"""One standalone phase plot: joint paths versus unsmoothed posterior summaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, '/REVIEW_ENV/figure_helpers')
from figure_style import new_figure, save_figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['published', 'shapeit419'], required=True)
    args = parser.parse_args()
    current = ROOT/'receipt_v02/results'/f'{args.arm}_mle'
    previous = ROOT.parents[1]/'trajectory_v01/receipt_v01/results'/f'{args.arm}_mle'
    assert (ROOT/'receipt_v02/independent_audit.json').is_file()
    now = np.load(current/'posterior.npz')
    old = np.load(previous/'posterior.npz')
    paths = np.load(current/'paths.npz')['frequency']
    generation, frequency, posterior = now['generation'], now['frequency'], now['posterior']
    np.testing.assert_array_equal(frequency, old['frequency'])
    np.testing.assert_array_equal(generation, old['generation'])
    mean = posterior @ frequency
    old_mean = old['posterior'] @ frequency
    cumulative = posterior.cumsum(axis=1)
    quantiles = np.array([frequency[(cumulative >= p).argmax(axis=1)] for p in (.025,.5,.975)])
    identity = json.loads((current/'identity.json').read_text())
    old_identity = json.loads((previous/'identity.json').read_text())
    # Keep all plots on the raw generation grid. No interpolation or smoothing.
    fig, ax = new_figure(183, 105)
    ax.fill_between(generation, quantiles[0]*100, quantiles[2]*100,
                    color='#3B6FB6', alpha=.12, lw=0, label='Pointwise 95% interval')
    for i in range(12):
        ax.plot(generation, paths[i]*100, color='#999999', alpha=.55, lw=.55,
                label='12 joint posterior paths' if i == 0 else None)
    overlap = np.max(np.abs(mean-old_mean)) < 1e-12
    if not overlap:
        ax.plot(generation, old_mean*100, color='#6E8E6B', ls='--', lw=1.3, label='v01 mean')
    ax.plot(generation, mean*100, color='#3B6FB6', lw=1.4,
            label='v01 / v02 mean (overlap)' if overlap else 'v02 mean')
    ax.plot(generation, quantiles[1]*100, color='#C17A3F', lw=1.15, label='v02 median')
    ax.scatter([0], [100*24/4096], s=15, marker='D', color='black', zorder=5,
               label='Present input: 24/4,096')
    ax.set_xlim(1000, -20)
    ax.set_ylim(0, max(np.max(paths[:12]), np.max(quantiles[2]), 24/4096)*107)
    ax.set_xlabel('Generations before present')
    ax.set_ylabel('Conditional allele frequency (%)')
    ax.legend(loc='upper left', fontsize=6.5)
    source = ROOT/'results/source_data'/f'{args.arm}_smoothness_v02.tsv'
    with source.open('x', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['generation','v01_mean','v02_mean','q025','median','q975','p_zero']+
                        [f'path_{i+1}' for i in range(12)])
        writer.writerows(zip(generation,old_mean,mean,*quantiles,posterior[:,0],*paths[:12]))
    base = ROOT/'figures/draft'/f'20260921_clues_{args.arm}_smoothness_v02'
    base.parent.mkdir(exist_ok=True)
    for extension in ('png','pdf','svg'):
        destination = base.with_suffix('.'+extension)
        assert not destination.exists()
        save_figure(fig,destination)
    metrics = dict(arm=args.arm,selection_v01=old_identity['s'],selection_v02=identity['s'],
        same_selection=identity['s']==old_identity['s'],means_overlap_at_1e12_tolerance=bool(overlap),
        maximum_mean_change_percentage_points=float(np.max(np.abs(mean-old_mean))*100),
        maximum_posterior_cell_change=float(np.max(np.abs(posterior-old['posterior']))),
        mean_total_variation=float(np.abs(np.diff(mean)).sum()),
        median_path_total_variation=float(np.median(np.abs(np.diff(paths,axis=1)).sum(axis=1))),
        path_slots_shown=list(range(1,13)),path_count=len(paths),extra_smoothing=False,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        posterior_sha256=hashlib.sha256((current/'posterior.npz').read_bytes()).hexdigest(),
        paths_sha256=hashlib.sha256((current/'paths.npz').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        milestones=[dict(generation=int(g),mean=float(mean[g-1]),median=float(quantiles[1,g-1]),
                         p_zero=float(posterior[g-1,0])) for g in (25,50,100,200,500)])
    (ROOT/'results'/f'{args.arm}_smoothness_checks.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
