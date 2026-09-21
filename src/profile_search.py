"""One-parameter bounded likelihood search and likelihood-ratio confidence sets.

No Gaussian is fitted to an optimizer's incidental evaluation history. The point
estimate is always the best evaluated point. Components touching the search edge
are explicitly censored there; a finite grid is not a proof of global optimality.
"""
import math
from statistics import NormalDist
from scipy.optimize import brentq, minimize_scalar


def fit_profile(evaluate, grid, confidence=0.95, xtol=1e-5, max_evaluations=160):
    grid = sorted(set(round(float(x),14) for x in grid))
    lower, upper = grid[0], grid[-1]
    if not lower < 0 < upper or not 0 < confidence < 1:
        raise ValueError('Grid must include both signs and confidence must be in (0,1)')
    cache = {}
    searches = []

    def value(x):
        # Operate on s itself. Rounding is only a cache key, never x=s+1.
        x = float(x)
        eps = 8 * math.ulp(max(1., abs(lower), abs(upper)))
        if x < lower-eps or x > upper+eps:
            raise ValueError('Attempt outside the fixed search domain')
        x = round(min(upper, max(lower, x)), 14)
        if x not in cache:
            if len(cache) >= max_evaluations:
                raise RuntimeError('Evaluation budget reached; no estimate is released')
            y = float(evaluate(x))
            if not math.isfinite(y):
                raise ValueError('Nonfinite likelihood')
            cache[x] = y
        return cache[x]

    for x in sorted(set(grid+[0.])):
        value(x)
    initial = dict(cache)
    flat = max(initial.values())-min(initial.values()) < 1e-8
    if not flat:
        for left, middle, right in zip(grid[:-2], grid[1:-1], grid[2:]):
            if initial[middle] >= max(initial[left], initial[right]):
                result = minimize_scalar(lambda x: -value(x), bounds=(left,right),
                                         method='bounded', options={'xatol':xtol,'maxiter':80})
                searches.append(dict(left=left,right=right,success=bool(result.success),
                                     x=float(result.x),message=str(result.message)))
                if not result.success:
                    raise RuntimeError('A candidate-peak refinement failed')
    drop = NormalDist().inv_cdf((1+confidence)/2)**2 / 2
    components = []
    roots = []
    for cycle in range(3):
        best_s = max(cache, key=lambda x:(cache[x], -abs(x)))
        best = cache[best_s]
        cutoff = best-drop
        xs = sorted(cache)
        roots = []
        for x in xs[1:-1]:
            if abs(cache[x]-cutoff) < 1e-10:
                roots.append(dict(s=x,residual=cache[x]-cutoff))
        for left,right in zip(xs[:-1],xs[1:]):
            yl,yr = cache[left]-cutoff,cache[right]-cutoff
            if yl*yr < 0:
                root = float(brentq(lambda x:value(x)-cutoff,left,right,
                                   xtol=min(xtol,1e-6),maxiter=80))
                residual = value(root)-cutoff
                if abs(residual)>0.002:
                    raise RuntimeError('Confidence endpoint residual too large')
                roots.append(dict(s=root,residual=residual))
        # CI refinement is also checked for a newly discovered better point.
        if max(cache.values()) > best+1e-8:
            continue
        bounds = [lower]+sorted({r['s'] for r in roots})+[upper]
        components = []
        for lo,hi in zip(bounds[:-1],bounds[1:]):
            # Evaluate inside each component, not arbitrarily close to a rounded
            # root. Any better point is included in the next search/CI cycle.
            inside = value((lo+hi)/2) >= cutoff
            if inside:
                components.append(dict(lower=lo,upper=hi,
                                       lower_censored=(lo==lower and cache[lower]>=cutoff),
                                       upper_censored=(hi==upper and cache[upper]>=cutoff)))
        if max(cache.values()) > best+1e-8:
            continue
        break
    else:
        raise RuntimeError('A new peak was found during confidence-set construction')
    # Select again after all evaluations, even if the change is below tolerance.
    best_s = max(cache, key=lambda x:(cache[x], -abs(x)))
    best = cache[best_s]
    contains = any(c['lower']-xtol <= best_s <= c['upper']+xtol for c in components)
    if not contains:
        raise RuntimeError('Confidence set does not contain the best evaluated point')
    boundary = min(abs(best_s-lower),abs(best_s-upper)) <= xtol
    return dict(best_s=best_s,logLR=best,domain=[lower,upper],confidence=confidence,
                profile_drop=drop,confidence_components=components,confidence_roots=roots,
                confidence_contains_best=contains,best_not_below_evaluated=True,
                optimum_at_boundary=boundary,flat_over_initial_grid=flat,
                disconnected_confidence_set=len(components)>1,searches=searches,
                evaluations=len(cache),initial_grid=grid,
                likelihood_points=[dict(s=x,logLR=cache[x]) for x in sorted(cache)],
                search_scope='Fixed grid with bounded refinement of its candidate peaks',
                interval_method='Conditional likelihood-ratio set; asymptotic chi-square(1), not calibrated population coverage')
