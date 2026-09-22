# Posterior presentation revision v03

This revision changes summaries and presentation of the existing conditional_trajectory_v02 outputs. It does not change the HMM, rerun inference, change selection parameters or add noise to a mean curve. Study posterior matrices, source tables, images and time estimates are not included in this public code-review repository.

## Scientific changes

For each phase, three separate figures compare the existing MLE and neutral scenarios on matching axes: the full frequency distribution, P(F_t>0), and the frequency distribution conditional on F_t>0. The full distribution now emphasizes its median and pointwise central 95% interval; its mean remains as a thin dotted line. Means may lie outside central intervals in highly zero-inflated posteriors.

Positive mass is summed directly rather than obtained by subtracting p_zero from one. Positive-only quantiles use the full posterior matrix, not the small number of survivors among the 512 sampled paths. Undefined positive-only summaries remain NaN. Conditional frequency has an explicitly logarithmic percentage axis. Fading when P(F_t>0)<5% is a display convention only; all source rows remain intact. No smoothing or interpolation is applied to the recorded generation grid.

The code checks p_zero monotonicity and that all existing paths remain at zero after entering it. The exact kernel identity is compared with the recorded run. In this backwards absorbing-zero model, p_zero is the CDF of the first recorded zero time. Discrete time quantiles preserve beyond-window probability and explicitly flag quantiles not reached by the terminal generation. This is a model-conditioned hitting-time summary, not selection onset or independently calibrated mutation age. Constant s does not contain a selection-start parameter.

## Entries and validation

- `trajectory_v02/code/summarize_posterior.py`: decomposition, complete source tables, first-zero quantiles and identity/numerical checks; refuses overwriting frozen outputs.
- `trajectory_v02/code/audit_posterior_summary.py`: exact zero-mixture, undefined-condition and right-censoring checks; independent scalar sums and conditional quantile checks at 24 saved rows.
- `trajectory_v02/code/plot_posterior_summary.py --arm published --view frequency`: one standalone plot per invocation; arm also accepts shapeit419 and view also accepts positive_probability or positive_frequency. The existing bioinfo-figure style helper is external; set BIOINFO_FIGURE_HELPERS to its directory.

The local four-input checks, scalar checks and six rendered PNG inspections passed. The four path-based empirical first-zero CDFs were within the predeclared DKW bound at familywise alpha=0.01; this assesses sampler consistency, not model calibration. PNG/PDF/SVG were exported as draft, and the original data/posterior matrices and older figures were preserved. These full-data checks cannot be independently reproduced from this code-only repository without the original result package.

No grid, topology, recoding, demographic or selection-parameter integration sensitivity was run in this presentation revision. Neutral scenarios also rise toward the present; conditional reconstructions are not an additional independent selection test. See SOURCE_INVENTORY.json for original/export hashes and the existing validation notes for the earlier numerical repairs.
