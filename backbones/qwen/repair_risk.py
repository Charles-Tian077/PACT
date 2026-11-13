"""Finite-family, image-cluster risk certification for selective repair.

Calibration labels never enter model features. The entire hard policy (gate,
candidate filtering and argmax) is replayed, then certified simultaneously over
a PREDECLARED finite family. Bounded KL-Chernoff intervals use independent image
clusters; queries within an image may be arbitrarily dependent. Claims concern
image-balanced query risks under the stated same-distribution assumptions.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
from collections import defaultdict


THRESHOLDS = {'gate_offset':'evidence_gate_offset', 'gain_margin':'evidence_gain_margin',
              'repairability_threshold':'evidence_repairability_threshold',
              'harm_probability_threshold':'evidence_harm_probability_threshold'}
BOUND_KEYS = ('fp','fn','new_fp','new_miss','harm','repair_rate','gain_positive','gain_negative',
              'total_gain_positive','total_gain_negative')


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def protocol_sha256():
    root = Path(__file__).resolve().parent
    files = ['repair_risk.py','eval_sesame_qwen3vl.py','dataloaders/data_utils.py',
             'model/sesame_qwen3vl.py','model/evidence_policy.py','model/hierarchical_policy.py',
             'model/selective_repair_policy.py','model/qwen3vl_config.py',
             'model/convex_repair_policy.py','policy_optimization.py',
             'tools/collect_repair_audit.py']
    encoded = json.dumps({p:file_sha256(root/p) for p in files},sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def processor_sha256(checkpoint_path):
    root=Path(checkpoint_path).resolve().parent/'processor'
    if not root.is_dir():raise ValueError('A fixed saved processor is required for certification')
    files={str(p.relative_to(root)):file_sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}
    if not files:raise ValueError('Saved processor is empty')
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()


def frozen_context(cfg):
    # All method/data-resolution settings except the four certified thresholds.
    return {k:v for k,v in vars(cfg).items() if k not in THRESHOLDS.values() and
            k not in ('model_name_or_path','vision_pretrained')}


def validate_thresholds(values):
    if set(values) != set(THRESHOLDS):
        raise ValueError('Supply exactly the four selective-repair thresholds')
    values = {k:float(v) for k,v in values.items()}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Thresholds must be finite')
    if values['gain_margin'] < 0 or not all(0 <= values[k] <= 1 for k in
            ('repairability_threshold','harm_probability_threshold')):
        raise ValueError('Use nonnegative gain margin and probabilities in [0,1]')
    return values


def _f32(value):
    return struct.unpack('f',struct.pack('f',float(value)))[0]


def select_action(row, thresholds):
    """Replay V4 float32 filtering, switch ties and rejection ties exactly."""
    t = {k:_f32(v) for k,v in validate_thresholds(thresholds).items()}
    anchor = int(row['anchor_action']) - 1
    length = len(row['gains'])
    if not 0 <= anchor < length or any(len(row[k]) != length for k in
            ('benefit_probability','harm_probability','available')):
        raise ValueError('Malformed action diagnostics')
    scores = []
    for j,(gain,benefit,harm,valid) in enumerate(zip(row['gains'],row['benefit_probability'],
                                                  row['harm_probability'],row['available'])):
        if not all(math.isfinite(float(v)) for v in (gain,benefit,harm)):
            raise ValueError('Nonfinite action diagnostics')
        eligible = (bool(valid) and j!=anchor and benefit>=t['repairability_threshold'] and
                    harm<=t['harm_probability_threshold'])
        scores.append(0.0 if j==anchor else _f32(gain-t['gain_margin']) if eligible else -1e4)
    candidate = max(range(length),key=lambda j:scores[j])
    selected = candidate if scores[candidate]>0 else anchor
    return selected+1 if _f32(row['gate_score']-t['gate_offset'])>0 else 0


def _binary_kl(p,q):
    if q<=0:return 0.0 if p==0 else math.inf
    if q>=1:return 0.0 if p==1 else math.inf
    return ((p*math.log(p/q) if p>0 else 0.0) +
            ((1-p)*math.log((1-p)/(1-q)) if p<1 else 0.0))


def bounded_interval(values, failure_probability):
    """One lower + one upper KL-Chernoff bound for independent [0,1] losses.

    For X in [0,1], E exp(tX) <= 1-E[X]+E[X]exp(t). Chernoff and inversion give
    n*kl(empirical_mean, population_mean) <= log(1/delta), one tail at a time.
    The caller budgets each tail across policies/metrics. This is not a
    distribution-shift guarantee or a guarantee for each individual query.
    """
    n = len(values)
    if not n:return dict(mean=None,lower=0.0,upper=1.0,images=0)
    if not 0<failure_probability<1 or any(not math.isfinite(x) or not 0<=x<=1 for x in values):
        raise ValueError('Invalid bounded loss or confidence level')
    mean = math.fsum(values)/n
    radius = math.log(1/failure_probability)/n
    lo,hi=0.0,mean
    for _ in range(64):
        mid=(lo+hi)/2
        if _binary_kl(mean,mid)>radius:lo=mid
        else:hi=mid
    lower=lo if mean>0 else 0.0
    lo,hi=mean,1.0
    for _ in range(64):
        mid=(lo+hi)/2
        if _binary_kl(mean,mid)>radius:hi=mid
        else:lo=mid
    upper=hi if mean<1 else 1.0
    return dict(mean=mean,lower=lower,upper=upper,images=n)


def _cluster_losses(records,thresholds,tolerance):
    clusters = defaultdict(lambda:defaultdict(list))
    for row in records:
        action=select_action(row,thresholds)
        image=clusters[row['image_key']]
        anchor=int(row['anchor_action'])
        anchor_accept=row['anchor_gate']>0
        image['repair_rate'].append(float(action>1))
        if not row['gt_exists']:
            image['fp'].append(float(action>0))
            image['new_fp'].append(float(not anchor_accept and action>0))
            continue
        ious=row['candidate_iou']
        if ious is None or len(ious)!=len(row['gains']) or row.get('empty_positive_mask'):
            raise ValueError('Every positive certification query needs valid candidate IoUs')
        if any(not math.isfinite(v) or not 0<=v<=1 for v in ious):
            raise ValueError('Invalid candidate IoU')
        anchor_iou=ious[anchor-1]
        changed=action>0 and action!=anchor
        delta=ious[action-1]-anchor_iou if changed else 0.0
        total_delta=(ious[action-1] if action else 0.0)-(anchor_iou if anchor_accept else 0.0)
        image['fn'].append(float(action==0))
        image['new_miss'].append(float(anchor_accept and action==0))
        image['harm'].append(float(changed and delta < -tolerance))
        image['gain_positive'].append(max(delta,0.0))
        image['gain_negative'].append(max(-delta,0.0))
        image['total_gain_positive'].append(max(total_delta,0.0))
        image['total_gain_negative'].append(max(-total_delta,0.0))
    return {key:[math.fsum(v[key])/len(v[key]) for v in clusters.values() if v[key]] for key in BOUND_KEYS}


def certify_family(audit,family):
    meta=audit['metadata'];records=audit['predictions']
    if meta.get('role')!='calibration' or not meta.get('excluded_selection_reports'):
        raise ValueError('Certification requires held-out calibration and model-selection exclusions')
    if not meta.get('partition_manifest_sha256') or not records:
        raise ValueError('A predeclared image partition and nonempty records are required')
    if family.get('format_version')!=1 or family.get('predeclared') is not True:
        raise ValueError('A fixed, predeclared finite family is required')
    policies=[validate_thresholds(p) for p in family['policies']]
    family_hash=hashlib.sha256(json.dumps(family,sort_keys=True).encode()).hexdigest()
    if meta.get('family_sha256')!=family_hash:
        raise ValueError('Policy family differs from the family declared before training')
    if not policies or len({json.dumps(p,sort_keys=True) for p in policies})!=len(policies):
        raise ValueError('Policy family must be nonempty and contain no duplicates')
    targets=family['targets'];alpha=float(family['failure_probability'])
    slots=family['checkpoint_slots']
    if not slots or len(set(slots))!=len(slots) or meta.get('checkpoint_slot') not in slots:
        raise ValueError('Checkpoint slot was not declared before calibration')
    if not 0<alpha<1 or any(not 0<=float(targets[k])<=1 for k in
            ('fpr','fnr','new_fp','new_miss','harm','repair_rate','minimum_geometry_gain')):
        raise ValueError('Invalid risk targets')
    if abs(float(family['iou_tolerance'])-meta['frozen_context']['evidence_repair_iou_tolerance'])>1e-12:
        raise ValueError('Calibration cannot change the trained definition of harmful repair')
    for key,field in [('fpr','evidence_fpr_target'),('fnr','evidence_fnr_target'),
                      ('harm','evidence_repair_harm_target')]:
        if abs(float(targets[key])-meta['frozen_context'][field])>1e-12:
            raise ValueError('Certification must retain the predeclared training risk targets')
    identities=[r['sample_key'] for r in records]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate audit query')
    # Two tails per metric, all policies simultaneously. No subsequent adaptive
    # threshold search is covered by this certificate.
    tail=alpha/(len(slots)*2*len(policies)*len(BOUND_KEYS))
    reports=[]
    for index,thresholds in enumerate(policies):
        losses=_cluster_losses(records,thresholds,float(family['iou_tolerance']))
        bounds={k:bounded_interval(losses[k],tail) for k in BOUND_KEYS}
        gain_lower=bounds['gain_positive']['lower']-bounds['gain_negative']['upper']
        admissible=all(bounds[k]['images']>0 and bounds[k]['upper']<=targets[t]
            for k,t in (('fp','fpr'),('fn','fnr'),('new_fp','new_fp'),('new_miss','new_miss'),
                        ('harm','harm'),('repair_rate','repair_rate')))
        admissible=admissible and gain_lower>targets['minimum_geometry_gain']
        empirical_gain=(bounds['gain_positive']['mean'] or 0)-(bounds['gain_negative']['mean'] or 0)
        score=2*empirical_gain-meta['frozen_context']['evidence_cost']*(bounds['repair_rate']['mean'] or 0)
        reports.append(dict(index=index,thresholds=thresholds,bounds=bounds,
                            geometry_gain_lower=gain_lower,admissible=admissible,selection_score=score))
    feasible=[r for r in reports if r['admissible']]
    selected=max(feasible,key=lambda r:(r['selection_score'],-r['index'])) if feasible else None
    return dict(format_version=1,status='CERTIFIED' if selected else 'INFEASIBLE',
        scope='Simultaneous bounds for image-balanced query risks; independent held-out image clusters, frozen predictor and same-distribution deployment assumed. Not per-query safety.',
        failure_probability=alpha/len(slots),family_failure_probability=alpha,
        checkpoint_slot=meta['checkpoint_slot'],tail_probability=tail,targets=targets,
        score_definition='2 * image-balanced incremental hard IoU - evidence_cost * extra decoder rate',
        metadata=meta,calibration_image_keys=sorted({r['image_key'] for r in records}),
        selected=selected,policies=reports,
        family_sha256=family_hash,
        infeasible_action='Report infeasibility and the unchanged baseline separately; do not relax targets or call a baseline fallback certified.')


def apply_certificate(model,checkpoint_path,certificate,inference_dtype,hardware):
    if certificate.get('status')!='CERTIFIED' or not certificate.get('selected'):
        raise ValueError('No admissible calibrated policy; targets must not be silently relaxed')
    if model.cfg.evidence_policy_version not in (4,5) or not bool(model.evidence.rl_active.item()):
        raise ValueError('Risk certificates apply only to their bound V4/V5 policy')
    meta=certificate['metadata']
    if meta['checkpoint_sha256']!=file_sha256(checkpoint_path):
        raise ValueError('Certificate checkpoint hash mismatch')
    if meta['protocol_sha256']!=protocol_sha256() or meta['frozen_context']!=frozen_context(model.cfg):
        raise ValueError('Certificate source/preprocessing/configuration mismatch')
    if meta['processor_sha256']!=processor_sha256(checkpoint_path):
        raise ValueError('Certificate tokenizer/processor files differ')
    import torch
    if meta['torch_version']!=torch.__version__:
        raise ValueError('Certificate PyTorch numerical environment differs')
    if meta['inference_dtype']!=inference_dtype or meta['hardware']!=hardware:
        raise ValueError('Certificate hardware/numerical execution differs; recalibration is required')
    for key,value in validate_thresholds(certificate['selected']['thresholds']).items():
        setattr(model.cfg,THRESHOLDS[key],value)
    model.evidence.calibration_info={k:certificate[k] for k in
        ('status','scope','failure_probability','family_failure_probability','targets','selected','family_sha256')}


def assert_disjoint_images(samples,certificate):
    blocked=set(certificate['calibration_image_keys'])
    seen={}
    for sample in samples:
        path=sample['image_path']
        if path not in seen:seen[path]=file_sha256(path)
        if seen[path] in blocked:
            raise ValueError('Evaluation overlaps a calibration image; use the independent assessment partition')
