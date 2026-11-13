from __future__ import annotations
import numpy as np
class CandidateNotQualified(RuntimeError):pass
def _finite(value, shape=None):
    result = np.asarray(value, np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f'Expected {shape}, got {result.shape}')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite controller input')
    return result

def allocate_shared_budget(values, eligible, rescued, caps, sample_keys, *, seed):
    """Fixed-block budget allocation and dependent rounding, before outcomes.

    Geometry marginal trial probabilities respect caps. Systematic rounding keeps
    their marginals while respecting the integer block budget in every draw.
    Rescues are charged first and never receive a second geometry trial.
    """
    values = _finite(values)
    if values.ndim != 1:
        raise ValueError('Scheduling values must be a vector')
    for flags in [eligible, rescued]:
        array = np.asarray(flags)
        if array.dtype.kind not in 'biu' or not np.isin(array, [0, 1]).all():
            raise ValueError('Scheduling flags must be finite Boolean indicators')
    eligible = np.asarray(eligible, bool)
    rescued = np.asarray(rescued, bool)
    caps = _finite(caps, values.shape)
    n = len(values)
    if (n < 1 or values.ndim != 1 or eligible.shape != (n,) or rescued.shape != (n,)
            or len(sample_keys) != n or not all(isinstance(k, str) for k in sample_keys)
            or len(set(sample_keys)) != n
            or np.any(caps < 0) or np.any(caps > 1)):
        raise ValueError('Invalid fixed-block scheduling inputs')
    budget = n // 4
    rescues = int(rescued.sum())
    if rescues > budget:
        raise CandidateNotQualified('Presence rescues alone exceed the shared.25 budget')
    probability = np.zeros(n)
    remaining = float(budget - rescues)
    order = sorted(range(n), key=lambda j: (-float(values[j]), str(sample_keys[j])))
    for i in order:
        if not eligible[i] or rescued[i] or values[i] <= 0:
            continue
        probability[i] = min(float(caps[i]), remaining)
        remaining -= probability[i]
        if remaining <= 0:
            break
    # Stable identity order makes this a frozen policy, rather than input-order tuning.
    identity_order = sorted(range(n), key=lambda j: str(sample_keys[j]))
    total = np.cumsum(probability[identity_order])
    previous = np.r_[0., total[:-1]]
    offset = np.random.default_rng(int(seed)).random()
    draw = (np.floor(total + offset) - np.floor(previous + offset)).astype(bool)
    trial = np.zeros(n, bool)
    trial[identity_order] = draw
    if int(trial.sum()) + rescues > budget or np.any(trial & rescued):
        raise RuntimeError('Shared-budget accounting failed')
    return dict(probabilities=probability, geometry_trials=trial,
                rescue_trials=rescued.copy(), total_budget=budget,
                realized_total_trials=int(trial.sum()) + rescues)
