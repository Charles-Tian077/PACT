"""Deployment-visible paired features and geometry-constrained repair heads.

No ground-truth paths, labels, IoUs or rewards enter these functions. Scoring
labels belong in the collector, after frozen inference has returned.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_WIDTH=1536

def _pool(embedding,weights):
    return (embedding*weights).sum((-2,-1))/weights.sum().clamp_min(1e-6)

def _pack(blocks):
    # Fixed block normalization, no fitted whitening/PCA or outcome-based scaling.
    items=[]
    for block in blocks:
        v=block.detach().float().flatten()
        items.append(v/v.norm().clamp_min(1.))
    value=torch.cat(items)
    if value.numel()>FEATURE_WIDTH:raise ValueError('Feature width exceeds frozen interface')
    return F.pad(value,(0,FEATURE_WIDTH-value.numel()))

def _boundary(mask):
    value=mask.float()[None,None]
    outer=F.max_pool2d(value,3,stride=1,padding=1)
    inner=1-F.max_pool2d(1-value,3,stride=1,padding=1)
    return (outer-inner)[0,0].clamp(0,1)

def paired_features(embedding,semantic,first_mask,second_mask,input_size,canvas_size,
                    first_native,second_native,old_first,old_second):
    """Return comparable representations for A/B/C/D and both pair directions.

Masks are native model predictions, reconstructed at original image resolution.
No annotation-derived ignore domain is accepted. Shape/region pooling uses only
predicted masks and the SAM image resize/pad geometry.
"""
    if first_mask.ndim!=2 or first_mask.shape!=second_mask.shape:raise ValueError('Mask geometry mismatch')
    emb=embedding.detach().float()
    if emb.ndim==4:
        if emb.shape[0]!=1:raise ValueError('One image per diagnostic forward')
        emb=emb[0]
    if emb.ndim!=3 or semantic.numel()!=emb.shape[0]:raise ValueError('Unexpected SAM representation')
    old_first=old_first.detach().to(emb.device).float()
    old_second=old_second.detach().to(emb.device).float()
    first=first_mask.to(emb.device)>0;second=second_mask.to(emb.device)>0
    union=(first|second).sum();intersection=(first&second).sum()
    distance=(1-intersection.float()/union) if int(union) else emb.new_zeros(())
    h,w=map(int,input_size)
    if not (0<h<=canvas_size and 0<w<=canvas_size):raise ValueError('Invalid SAM resize geometry')
    def weights(binary):
        image=F.interpolate(binary.float()[None,None],size=(h,w),mode='area')
        padded=F.pad(image,(0,canvas_size-w,0,canvas_size-h))
        return F.interpolate(padded,size=emb.shape[-2:],mode='area')[0,0]
    valid=weights(torch.ones_like(first))
    sem=semantic.detach().float().flatten()
    context=_pool(emb,valid)
    def direction(a,b,na,nb,oa,ob):
        add=b&~a;remove=a&~b;common=a&b
        wa,wb=weights(a),weights(b)
        wadd,wremove,wcommon=weights(add),weights(remove),weights(common)
        native_a=na.detach().to(emb.device).float();native_b=nb.detach().to(emb.device).float()
        scalars=emb.new_tensor([a.float().mean(),b.float().mean(),add.float().mean(),
          remove.float().mean(),common.float().mean(),float(distance),
          float(native_a.sigmoid().mean()),float(native_b.sigmoid().mean()),
          float(native_a.sigmoid().var(unbiased=False)),float(native_b.sigmoid().var(unbiased=False))])
        channels=emb.shape[0]
        if oa.numel()!=channels*4+13 or oa.shape!=ob.shape:
            raise ValueError('Original action-feature layout differs from frozen kernel')
        # Keep ALL original common evidence plus every varying ROI/box/type
        # coordinate. Pairwise differencing alone would discard query context.
        old=_pack([oa,(ob-oa)[channels:channels*2],(ob-oa)[channels*4:]])
        whole=_pack([sem,context,_pool(emb,wa),_pool(emb,wb),scalars])
        changed=_pack([sem,_pool(emb,wadd),_pool(emb,wremove),_pool(emb,wcommon),
          _pool(emb,weights((_boundary(a)>0)|(_boundary(b)>0))),scalars])
        return dict(old=old,whole=whole,changed=changed)
    forward=direction(first,second,first_native,second_native,old_first,old_second)
    reverse=direction(second,first,second_native,first_native,old_second,old_first)
    return dict(forward=forward,reverse=reverse,full_image_jaccard_distance=distance.detach(),
      predicted_masks_identical=bool(torch.equal(first,second)))

class PairHead(nn.Module):
    """Same parameter count across representations and structural ablations."""
    def __init__(self,width=FEATURE_WIDTH,hidden=32,structured=True):
        super().__init__();self.structured=bool(structured)
        self.network=nn.Sequential(nn.Linear(width,hidden),nn.SiLU(),nn.Linear(hidden,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,forward,reverse,distance,annotation_domain_observable=True):
        with torch.autocast(device_type=forward.device.type,enabled=False):
            raw=self.network(forward.float())
            if self.structured:
                odd=raw-self.network(reverse.float())
                # Hidden annotation domains require the trivial valid bound.
                d=distance.float() if annotation_domain_observable else (distance>0).float()
                geometry=d*torch.tanh(odd[...,0])
                quality=2*torch.tanh(odd[...,1])
            else:
                geometry=torch.tanh(raw[...,0]);quality=2*torch.tanh(raw[...,1])
        return dict(hard_iou_delta=geometry,quality_delta_before_action_cost=quality)

class ProposalHead(nn.Module):
    def __init__(self,width=FEATURE_WIDTH):
        super().__init__();self.value=nn.Linear(width,1)
        nn.init.zeros_(self.value.weight);nn.init.zeros_(self.value.bias)
    def forward(self,features):
        with torch.autocast(device_type=features.device.type,enabled=False):
            return self.value(features.float()).squeeze(-1)

@dataclass(frozen=True)
class RepairDecision:
    proposal:int|None
    accepted:bool
    delivered_action:int
    additional_sam_decodes:int
    original_action_cost_difference:float
    proposed_action_cost_difference:float
    reason:str

def choose_proposal(values,available,anchor,minimum_value=0.):
    """This function may see pre-computation signals only."""
    if values.ndim!=1 or values.shape!=available.shape:raise ValueError('Invalid proposal inputs')
    legal=available.bool().clone();legal[0]=False;legal[int(anchor)]=False
    masked=values.masked_fill(~legal,float('-inf'))
    if not bool(legal.any()) or not bool(masked.max()>minimum_value):return None
    return int(masked.argmax())

def accept_trial(anchor,proposal,quality_delta,hard_iou_delta,action_cost=.03,
                 utility_margin=0.,iou_margin=0.,additional_sam_decodes=1):
    if additional_sam_decodes not in (0,1):raise ValueError('At most one added decoder call')
    cost=action_cost*(int(proposal>1)-int(anchor>1))
    accepted=bool(quality_delta-cost>utility_margin and hard_iou_delta>iou_margin)
    return RepairDecision(proposal=proposal,accepted=accepted,
      delivered_action=proposal if accepted else anchor,additional_sam_decodes=additional_sam_decodes,
      original_action_cost_difference=cost if accepted else 0.,
      proposed_action_cost_difference=cost,
      reason='accepted' if accepted else 'exact_anchor_fallback')

class AcceptedUpdateBudget:
    """All head/auxiliary updates share one explicitly counted budget."""
    def __init__(self,limit=1500,completed=0):
        if not (0<=completed<=limit):raise ValueError('Invalid accepted update ledger')
        self.limit=int(limit);self.completed=int(completed)
    def update(self,optimizer,loss):
        if self.completed>=self.limit:raise RuntimeError('Accepted update budget exhausted')
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss; no optimizer update')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for group in optimizer.param_groups for p in group['params']):
            optimizer.zero_grad(set_to_none=True);raise ValueError('Nonfinite gradient; no optimizer update')
        optimizer.step();self.completed+=1
