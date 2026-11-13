from functools import lru_cache
import numpy as np
@lru_cache(None)
def projection(width,out):return np.random.default_rng(20261029+width+out).normal(0,1/np.sqrt(width),(width,out))

def contrast(first,second):
 x=first.copy();y=second.copy()
 assert len(x)==len(y)==1037
 for j in range(4):
  s=slice(j*256,(j+1)*256);x[s]/=max(1.,np.linalg.norm(x[s]));y[s]/=max(1.,np.linalg.norm(y[s]))
 d=y-x;c=(x[:256]+y[:256])/2
 return np.r_[d[:1024]@projection(1024,24),(c*d[256:512])@projection(256,16),d[1024:]]

def pre_candidates(prediction):
 # This function deliberately accepts NO pair masks, outcomes or GT objects.
 b=int(prediction['action']);anchor=int(prediction['policy_diagnostics']['anchor_action'])
 if b==0:return []
 assert b==anchor
 raw=prediction['features'][0,b].numpy().astype(float)
 p=np.random.default_rng(42+len(raw)).normal(0,1/np.sqrt(len(raw)),(len(raw),24))
 cheap=np.r_[(raw/max(1.,np.linalg.norm(raw)))@p,prediction['prior'][0].numpy().astype(float),float(prediction['policy_diagnostics']['anchor_gate'])]
 candidates=[]
 for a in range(2,9):
  if a==b or not bool(prediction['available'][0,a]):continue
  dc=.03*(1-int(b>1));new=prediction['features'][0,a].numpy().astype(float)
  candidates.append(dict(action=a,dc=dc,pre=np.r_[cheap,contrast(raw,new),dc]))
 return candidates
