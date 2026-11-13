"""Observable native-mask features from the evaluated implementation."""
import torch
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
