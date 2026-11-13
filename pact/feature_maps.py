import math,numpy as np
def _finite(value, shape=None):
    result = np.asarray(value, np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f'Expected {shape}, got {result.shape}')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite controller input')
    return result

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

def whole_context39(forward,reverse,distance,anchor,proposal):
 # Exact legacy task_interaction whole13 map. Whole and changed layouts differ.
 f=np.asarray(forward,np.float64);r=np.asarray(reverse,np.float64);d=float(distance)
 assert f.shape==r.shape==(1536,) and np.isfinite(f).all() and np.isfinite(r).all() and 0<=d<=1
 assert np.array_equal(f[:256],r[:256]) and np.array_equal(f[512:768],r[768:1024]) and np.array_equal(f[768:1024],r[512:768])
 semantic=(f[:256]+r[:256])/2
 difference=f-r;context=(f[256:512]+r[256:512])/2
 compact=np.r_[np.sum(semantic*(f[768:1024]-f[512:768])),np.sum(semantic*context*difference[1025]),difference[1024:1034],.03*(int(proposal>1)-int(anchor>1))]
 assert compact.shape==(13,)
 return np.r_[compact,d*compact,d*d*compact]
