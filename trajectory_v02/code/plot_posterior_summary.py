"""One standalone figure per invocation; same phase, MLE versus neutral.

The positive-only distribution uses an explicitly logarithmic y axis, with
faded tails where P(F>0)<5%. This display threshold never filters source rows.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get('BIOINFO_FIGURE_HELPERS', '/REVIEW_ENV/figure_helpers'))
from figure_style import new_figure, save_figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['published', 'shapeit419'], required=True)
    parser.add_argument('--view', choices=['frequency', 'positive_probability', 'positive_frequency'], required=True)
    args = parser.parse_args()
    source = ROOT/'results/source_data'/f'{args.arm}_posterior_decomposition_v03.tsv'
    checks = json.loads((ROOT/'results/posterior_decomposition_checks_v03.json').read_text(encoding='utf-8'))
    assert checks['PASS'] and checks['sources'][source.name] == hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    figure_id = f'clues_{args.arm}_{args.view}_v03'
    base = ROOT/'figures/draft'/f'20260921_{figure_id}'
    destinations = [base.with_suffix('.'+ext) for ext in ('png', 'pdf', 'svg')]
    assert not any(p.exists() for p in destinations)
    fig, ax = new_figure(183, 105)
    handles = []
    low_mass = []
    for mode, color, linestyle in [('mle', '#3B6FB6', '-'), ('neutral', '#666666', '--')]:
        subset = [r for r in rows if r['mode'] == mode]
        values = lambda key: np.array([float(r[key]) for r in subset])
        generation = values('generation')
        s = values('s')[0]
        label = f'MLE (s = {s:.4f})' if mode == 'mle' else 'Neutral (s = 0)'
        handles.append(Line2D([], [], color=color, ls=linestyle, lw=1.5, label=label))
        if args.view == 'positive_probability':
            ax.plot(generation, 100*values('p_positive'), color=color, ls=linestyle, lw=1.5)
            continue
        prefix = 'positive_' if args.view == 'positive_frequency' else ''
        low, median, high, mean = (100*values(prefix+key) for key in ('q025', 'median', 'q975', 'mean'))
        bounds = (.00005, 15) if prefix else (0, 1.4)
        assert np.nanmin(low) >= bounds[0] and np.nanmax(high) <= bounds[1]
        rare = values('p_positive') < .05
        # Show all defined conditional rows, while visually distinguishing the rare tail.
        segments = [(np.ones(len(generation), dtype=bool), .30), (~rare, 1.0)] if prefix else [(np.ones(len(generation), dtype=bool), 1.0)]
        for keep, opacity in segments:
            y = lambda value: np.where(keep, value, np.nan)
            ax.fill_between(generation, y(low), y(high), color=color, alpha=.10*opacity, lw=0)
            ax.plot(generation, y(median), color=color, ls=linestyle, lw=1.4, alpha=opacity)
            ax.plot(generation, y(mean), color=color, ls=':', lw=.85, alpha=opacity)
        low_mass.append(dict(mode=mode, low_positive_mass_rows=int(rare.sum()), undefined_rows=int(np.isnan(mean).sum())))
    if args.view == 'positive_probability':
        ax.set_ylim(0, 103)
        ax.set_ylabel('Posterior probability of positive frequency (%)')
        ax.axhline(50, color='#BBBBBB', lw=.5, ls=':', zorder=0)
    else:
        handles.extend([Line2D([], [], color='black', lw=1.4, label='Median'),
                        Line2D([], [], color='black', ls=':', lw=.85, label='Mean'),
                        Patch(facecolor='#999999', alpha=.18, label='Pointwise 95% interval')])
        if args.view == 'frequency':
            ax.set_ylim(0, 1.4)
            ax.set_ylabel('Allele frequency (%)')
            ax.scatter([0], [100*24/4096], s=14, marker='D', color='black', zorder=5)
            handles.append(Line2D([], [], marker='D', ls='none', color='black', markersize=3, label='Present input: 24/4,096'))
        else:
            ax.set_yscale('log')
            ax.set_ylim(.00005, 15)
            ax.set_yticks([.0001, .001, .01, .1, 1, 10])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
            ax.set_ylabel('Frequency given F > 0 (%, log scale)')
            handles.append(Line2D([], [], color='#888888', alpha=.3, lw=1.5, label='Faded: P(F > 0) < 5%'))
    ax.set_xlim(1000, -20)
    ax.set_xlabel('Generations before present')
    # An external legend keeps every time point and conditional tail unobscured.
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.015), ncol=3, fontsize=6.5,
              borderaxespad=0, columnspacing=1.6, handlelength=2.8)
    for destination in destinations:
        save_figure(fig, destination)
    record = dict(figure_id=figure_id, arm=args.arm, view=args.view, status='draft',
                  dimensions_mm=[183,105], extra_smoothing=False, positive_tail_display=low_mass,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destinations})
    (ROOT/'results'/f'{figure_id}_render.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(figure_id=figure_id, exports=len(destinations))))


if __name__ == '__main__':
    main()
