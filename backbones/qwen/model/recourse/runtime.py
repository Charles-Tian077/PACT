"""Single-trial inference component with exact output fallback.

This component preserves baseline existence decisions. It isolates GEOMETRIC
repair and cannot claim that baseline absolute FPR/FNR meets deployment targets.
The full research policy needs its separately evaluated existence decision.
"""
from __future__ import annotations
import contextlib
from dataclasses import asdict
import torch
from torch.nn import functional as F
from .core import FEATURE_WIDTH,paired_features,choose_proposal,accept_trial

@torch.inference_mode()
def generate_with_recourse(model,proposal_head,pair_head,*,feature_kind='changed',
                           utility_margin=0.,iou_margin=0.,annotation_domain_observable=False,**kwargs):
    if kwargs.get('audit_candidates'):raise ValueError('Deployment component cannot enumerate audit candidates')
    if feature_kind not in ('old','whole','changed'):raise ValueError('Unknown representation')
    original_meta=model.evidence.decision_metadata
    original_proposals=model._proposals;original_decode=model._sam_decode
    captured={};natives=[]
    def meta(features,prior,available):
        if 'features' in captured or features.shape[0]!=1:raise ValueError('Exactly one query per runtime call')
        captured.update(features=features[0].detach(),prior=prior[0].detach(),available=available[0].detach())
        return original_meta(features,prior,available)
    def proposals(*args,**kw):
        result=original_proposals(*args,**kw);captured['boxes']=result[0].detach();return result
    def decode(embedding,semantic,boxes,input_size):
        if 'embedding' not in captured:
            captured.update(embedding=embedding.detach(),semantic=semantic.detach(),input_size=tuple(input_size))
        result=original_decode(embedding,semantic,boxes,input_size);natives.append(result.detach());return result
    model.evidence.decision_metadata=meta;model._proposals=proposals;model._sam_decode=decode
    try:
        cuda_input=any(torch.is_tensor(value) and value.is_cuda for value in kwargs.values())
        baseline_context=torch.autocast('cuda',dtype=torch.bfloat16) if cuda_input else contextlib.nullcontext()
        with baseline_context:baseline=model.generate(**dict(kwargs,audit_candidates=False))
    finally:
        model.evidence.decision_metadata=original_meta;model._proposals=original_proposals;model._sam_decode=original_decode
    if len(baseline['evidence_actions'])!=1:raise ValueError('Exactly one baseline output required')
    action=int(baseline['evidence_actions'][0]);out=dict(baseline)
    base_record=dict(component='geometry_only',existence_policy='preserve_actual_baseline',
      baseline_action=action,delivered_action=action,additional_sam_decodes=0,
      absolute_risk_certified=False,geometry_head_trained=bool(getattr(pair_head,'trained',False)))
    if action<=0:
        out['recourse']=dict(base_record,reason='baseline_existence_rejected',accepted=False);return out
    features=captured['features'].float()
    if features.shape[1]>FEATURE_WIDTH:raise ValueError('Proposal feature width mismatch')
    cheap=F.pad(features,(0,FEATURE_WIDTH-features.shape[1]))
    cheap=cheap/cheap.norm(dim=-1,keepdim=True).clamp_min(1.)
    values=proposal_head(cheap)
    if not bool(torch.isfinite(values).all()):raise ValueError('Nonfinite proposal values')
    proposal=choose_proposal(values,captured['available'],action)
    if proposal is None:
        out['recourse']=dict(base_record,reason='no_positive_proposal',accepted=False);return out
    expected_calls=1+int(action>1)
    if len(natives)!=expected_calls:raise ValueError('Baseline decoder accounting mismatch')
    anchor_native=natives[-1][0]
    autocast=(torch.autocast('cuda',dtype=torch.bfloat16) if captured['embedding'].is_cuda else contextlib.nullcontext())
    with autocast:
        if proposal==1:new_native=natives[0][0];extra=0
        else:
            decoded=model._sam_decode(captured['embedding'],captured['semantic'],
              captured['boxes'][proposal-1:proposal],captured['input_size'])
            if decoded.shape[0]!=1:raise ValueError('Trial must preserve one-prompt decoder path')
            new_native=decoded[0];extra=1
        original_size=tuple(kwargs['sam_mask_shape_list'][0][1])
        new_full=model.sam.postprocess_masks(new_native[None,None],
          input_size=captured['input_size'],original_size=original_size)[0,0]
    old_full=baseline['pred_masks'][0]
    if old_full.ndim==3 and old_full.shape[0]==1:old_full=old_full[0]
    pair=paired_features(captured['embedding'],captured['semantic'],old_full,new_full,
      captured['input_size'],model.sam.image_encoder.img_size,anchor_native,new_native,
      features[action],features[proposal])
    result=pair_head(pair['forward'][feature_kind][None],pair['reverse'][feature_kind][None],
      pair['full_image_jaccard_distance'][None],annotation_domain_observable)
    quality=float(result['quality_delta_before_action_cost'][0]);geometry=float(result['hard_iou_delta'][0])
    if not (torch.isfinite(torch.tensor(quality)) and torch.isfinite(torch.tensor(geometry))):
        raise ValueError('Nonfinite pair estimate')
    decision=accept_trial(action,proposal,quality,geometry,model.cfg.evidence_cost,
      utility_margin,iou_margin,additional_sam_decodes=extra)
    if decision.accepted:
        out['pred_masks']=[new_full[None]];out['evidence_actions']=[proposal]
        out['baseline_policy_diagnostics']=baseline['policy_diagnostics']
        out['policy_diagnostics']=[dict(baseline['policy_diagnostics'][0],chosen_action=proposal,
          decision_protocol='external_recourse_geometry_component')]
    # A rejected trial retains every original tensor/decision by object identity.
    out['decoder_calls']=[int(baseline['decoder_calls'][0])+extra]
    out['recourse']=dict(base_record,**{k:v for k,v in asdict(decision).items()
      if k not in ('additional_sam_decodes','delivered_action')},
      delivered_action=decision.delivered_action,additional_sam_decodes=extra,
      estimated_hard_iou_delta=geometry,estimated_quality_delta=quality,
      actual_trial_cost_already_incurred=bool(extra),
      geometry_bound_uses_observable_valid_domain=annotation_domain_observable)
    return out
