"""Frozen final-controller components; qualification is mandatory for activation.

These functions consume observable inputs only. They do not load annotations,
train heads, run extra Qwen forwards, or change the original SFT by default.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import numpy as np


class CandidateNotQualified(RuntimeError):
    pass


def _finite(value, shape=None):
    result = np.asarray(value, np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f'Expected {shape}, got {result.shape}')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite controller input')
    return result


@dataclass(frozen=True)
class FrozenPresenceReadout:
    """A diagnostic readout is not, by itself, an admitted final model."""
    mean: np.ndarray
    scale: np.ndarray
    basis: np.ndarray
    weight: np.ndarray

    @classmethod
    def from_model(cls, model):
        transform = model['transform']
        mean = _finite(transform['mean'])
        scale = _finite(transform['scale'], mean.shape)
        basis = _finite(transform['basis'])
        weight = _finite(model['weight'])
        if (mean.ndim != 1 or weight.ndim != 1 or weight.size == 0
                or basis.shape != (mean.size + 1, weight.size)):
            raise ValueError('Frozen readout shape mismatch')
        if np.any(scale < 1e-4):
            raise ValueError('Invalid frozen training scale')
        if not np.allclose(basis.T @ basis, np.eye(weight.size), atol=1e-10, rtol=0):
            raise ValueError('Fit-span basis is not orthonormal')
        arrays = [np.array(a, copy=True) for a in (mean, scale, basis, weight)]
        for array in arrays:
            array.flags.writeable = False
        return cls(*arrays)

    def scores(self, features):
        X = _finite(features)
        if X.ndim != 2 or X.shape[1] != self.mean.size:
            raise ValueError('Wrong observable feature width')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                Z = (X - self.mean) / self.scale
                Z /= np.maximum(1., np.linalg.norm(Z, axis=1, keepdims=True) / 5.)
                scores = (np.c_[np.ones(len(Z)), Z] @ self.basis) @ self.weight
        except FloatingPointError as exc:
            raise ValueError('Nonfinite frozen-readout computation') from exc
        return _finite(scores)


def require_presence_qualification(directory):
    """Read a prerequisite receipt before allowing any final-model activation.

    This is a research admission check, not a population risk certificate.
    It deliberately refuses the completed, failed risk-direct pilot.
    """
    directory = Path(directory)
    result_path = directory / 'result.json'
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    complete = json.loads((directory / 'complete.json').read_text('utf8'))
    if hashlib.sha256(result_bytes).hexdigest() != complete['result_sha256']:
        raise ValueError('Changed result bytes')
    arm = result.get('primary_arm')
    primary = result.get('results', {}).get(arm, {})
    calibration = primary.get('calibration') or {}
    if not calibration.get('feasible') or calibration.get('full_feasible_pairs', 0) <= 0:
        raise CandidateNotQualified('Presence calibration has no fully feasible policy')
    if not result.get('larger_stage_prerequisite') or not primary.get('validation'):
        raise CandidateNotQualified('Presence validation prerequisite did not pass')
    if (arm != 'risk_full_raw'
            or result.get('status') != 'RISK_DIRECT_SMALL_PREREQUISITE_PASSED'
            or complete.get('status') != result['status']):
        raise CandidateNotQualified('Frozen primary/status does not match the final protocol')
    v = primary['validation']
    limits = dict(FPR=.05, FNR=.05, newFP=.005, newMiss=.005,
                  geometric_harm_rate=.02, extra_decode_upper_per_query=.25)
    if any(not math.isfinite(v[k]) or v[k] > limit + 1e-12 for k, limit in limits.items()):
        raise CandidateNotQualified('Presence validation violates an original risk')
    if (not all(math.isfinite(v[k]) for k in ['net_original_utility', 'net_hard_iou'])
            or v['net_original_utility'] <= 0 or v['net_hard_iou'] < 0):
        raise CandidateNotQualified('Presence validation has no positive delivery gain')
    # All-risk calibration alone does not establish the required stable joint gain.
    stability_path = directory / 'stability_qualification.json'
    if not stability_path.is_file():
        raise CandidateNotQualified('Paired stability qualification is missing')
    stability = json.loads(stability_path.read_text('utf8'))
    bounds = [stability.get('paired_net_utility_lower', float('nan')),
              stability.get('paired_net_iou_lower', float('nan'))]
    if (stability.get('status') != 'PASSED'
            or stability.get('result_sha256') != hashlib.sha256(result_bytes).hexdigest()
            or not all(math.isfinite(x) for x in bounds)
            or bounds[0] <= 0 or bounds[1] < 0):
        raise CandidateNotQualified('Paired stability qualification did not pass')
    if result.get('complete_aggregate_auxiliary_budget_audit') is not True:
        raise CandidateNotQualified('Complete auxiliary budget audit is missing')
    # A result flag alone cannot admit weights that differ from the evaluated ones.
    manifest_path = directory / 'admission_manifest.json'
    if not manifest_path.is_file():
        raise CandidateNotQualified('Source/weight admission manifest is missing')
    manifest = json.loads(manifest_path.read_text('utf8'))
    required = ['result.json', 'complete.json', 'models.json', 'protocol.json',
                'stability_qualification.json']
    if set(manifest.get('files', {})) != set(required):
        raise CandidateNotQualified('Incomplete source/weight admission binding')
    for name in required:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest['files'][name]:
            raise ValueError('Changed admitted artifact: ' + name)
    protocol = json.loads((directory / 'protocol.json').read_text('utf8'))
    if protocol.get('primary_arm') != arm:
        raise CandidateNotQualified('Primary arm differs from the frozen protocol')
    expected_risks = dict(FPR=.05, FNR=.05, newFP=.005, newMiss=.005,
                          harm=.02, mean_extra_decode=.25, max_trials=1, original_cost=.03)
    if protocol.get('risks') != expected_risks or protocol.get('frozen_before_pilot') is not True:
        raise CandidateNotQualified('Frozen protocol risks differ from the original limits')
    models = json.loads((directory / 'models.json').read_text('utf8'))
    if models.get(arm, {}).get('selection') != calibration:
        raise CandidateNotQualified('Admitted thresholds differ from calibration')
    readout = FrozenPresenceReadout.from_model(models[arm])
    if readout.mean.size != 29 + 2560:
        raise CandidateNotQualified('Final raw-Qwen observable layout differs')
    return dict(arm=arm, result_sha256=hashlib.sha256(result_bytes).hexdigest(),
                population_risk_certified=False)


def changed_context39(forward, reverse, distance, anchor, proposal):
    """Exactly the compacted frozen522 task feature plus its even context.

    The source pair uses the original1536 layout, which contains five256-channel
    blocks, ten scalar slots and zero padding. No realized outcome is an input.
    """
    f = _finite(forward, (1536,))
    r = _finite(reverse, (1536,))
    d = float(distance)
    if not math.isfinite(d) or not 0 <= d <= 1:
        raise ValueError('Invalid predicted-mask distance')
    if not (1 <= anchor <= 8 and 1 <= proposal <= 8):
        raise ValueError('A geometry pair requires two legal present actions')
    if not np.array_equal(f[:256], r[:256]):
        raise ValueError('Semantic context changed when reversing the pair')
    if not (np.array_equal(f[256:512], r[512:768])
            and np.array_equal(f[512:768], r[256:512])):
        raise ValueError('Added/removed region features do not reverse exactly')
    semantic = (f[:256] + r[:256]) / 2
    difference = f - r
    region = f[256:512] - f[512:768]
    common = (f[768:1024] + r[768:1024]) / 2
    area = difference[1281]
    cost = .03 * (int(proposal > 1) - int(anchor > 1))
    compact = np.r_[np.sum(semantic * region),
                    np.sum(semantic * common * area), difference[1280:1290], cost]
    if compact.shape != (13,):
        raise ValueError('Compact feature layout changed')
    return np.r_[compact, d * compact, d * d * compact]


def odd_head_score(model, context):
    x = _finite(context, (39,))
    scale = _finite(model['scale'], (39,))
    weight = _finite(model['theta'], (39,))
    if np.any(scale < 1e-4):
        raise ValueError('Invalid frozen odd-head scale')
    z = x / scale
    z /= max(1., np.linalg.norm(z) / 5.)
    return float(z @ weight)


def geometry_accept(net_utility_score, hard_iou_score):
    # The utility target already includes the original.03 cost exactly once.
    if not all(math.isfinite(float(v)) for v in (net_utility_score, hard_iou_score)):
        raise ValueError('Nonfinite posterior scores')
    return bool(net_utility_score > 0 and hard_iou_score >= 0)


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


def deliver_or_rollback(baseline, candidate, *, accept):
    """Keep the actual original output object on rollback, not a reconstruction."""
    return candidate if accept else baseline
