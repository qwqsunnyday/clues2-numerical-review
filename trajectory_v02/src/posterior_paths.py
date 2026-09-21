"""Joint posterior paths for a fixed transition and genealogy.

filtered[t] contains log unnormalized mass after the emission at generation
t+1. The terminal factor is uniform. Conditional on the next state j, the
previous state i has mass filtered[t,i] * transition[i,j]; the next emission
is constant in i. This preserves temporal dependence. It does not sample s.
"""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _draw(logmass, uniform):
    maximum = np.max(logmass)
    if not np.isfinite(maximum):
        raise ValueError('No finite conditional path mass')
    mass = np.exp(logmass - maximum)
    target = uniform * np.sum(mass)
    cumulative = 0.0
    last = -1
    for i in range(len(mass)):
        if mass[i] > 0:
            last = i
            cumulative += mass[i]
            if cumulative > target:
                return i
    return last  # Only floating-point rounding can reach this fallback.


@njit(cache=True, nogil=True)
def sample_paths(filtered, log_transition, uniforms):
    count, steps = uniforms.shape
    if steps != filtered.shape[0]:
        raise ValueError('Path length does not match filtered messages')
    states = np.empty((count, steps), dtype=np.int64)
    for k in range(count):
        states[k, steps-1] = _draw(filtered[-1], uniforms[k, steps-1])
        for t in range(steps-2, -1, -1):
            logmass = filtered[t] + log_transition[:, states[k, t+1]]
            states[k, t] = _draw(logmass, uniforms[k, t])
    return states
