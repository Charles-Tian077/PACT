import numpy as np
def design(m,X):
 z=(np.asarray(X,float)-m['mean'])/m['scale'];z/=np.maximum(1.,np.linalg.norm(z,axis=1,keepdims=True)/5.)
 return z if m['odd'] else np.c_[np.ones(len(z)),z]
def predict(m,X):return design(m,X)@np.asarray(m['theta'])
def observed(data,experts,ii,target='delivery',shared=None):
 n=len(ii);out=dict(proposal=np.zeros((2,n),int),value=np.zeros((2,n)),u=np.zeros((2,n)),j=np.zeros((2,n)))
 for e,expert in enumerate(experts):
  if not expert['active']:continue
  for offset,i in enumerate(ii):
   rows=np.flatnonzero((data['pair_query']==i)&np.isin(data['pair_action'],expert['support']))
   if not len(rows):continue
   if shared is None:
    values=predict(expert[target],data['pair_pre'][rows]);k=int(np.argmax(values));row=rows[k];value=float(values[k])
   else:
    action=int(shared['proposal'][e,offset]);found=rows[data['pair_action'][rows]==action]
    if not len(found):continue
    row=int(found[0]);value=float(shared['value'][e,offset])
   out['proposal'][e,offset]=int(data['pair_action'][row]);out['value'][e,offset]=value
   out['u'][e,offset]=float(predict(expert['u'],data['pair_'+expert['representation']][row:row+1])[0])
   out['j'][e,offset]=float(predict(expert['j'],data['pair_'+expert['representation']][row:row+1])[0])
 return out
def expected(data,ii,obs,thresholds,enabled=True,force=False):
 n=len(ii);b=data['baseline'][ii];valid=data['valid'][ii];du=np.zeros(n);dj=np.zeros(n);ha=np.zeros(n);ho=np.zeros(n);tr=np.zeros(n)
 for e in range(2):
  a=obs['proposal'][e];eligible=enabled&(b>0)&(a>0)&(obs['value'][e]>0)
  accepted=eligible if force else eligible&(obs['u'][e]>thresholds[0])&(obs['j'][e]>=thresholds[1])
  u=data['utility'][ii,a]-data['utility'][ii,b];j=data['iou'][ii,a]-data['iou'][ii,b]
  du+=.125*accepted*u;dj+=.125*accepted*j;ha+=.125*(accepted&(j<0));ho+=.125*(accepted&(j<-.02));tr+=.125*eligible
 return dict(netU=float(du.mean()),netIoU=float(dj[valid].mean()),harm=float(ha[valid].mean()),harm002=float(ho[valid].mean()),extraDecode=float(tr.mean()))
def metrics(data,ii,action,trial):
 b=data['baseline'][ii];y=data['positive'][ii];valid=data['valid'][ii];idx=np.arange(len(ii));u=data['utility'][ii];j=data['iou'][ii]
 du=u[idx,action]-u[idx,b];dj=j[idx,action]-j[idx,b]
 assert np.array_equal(action>0,b>0),'Presence may never change'
 assert valid.any() and np.isfinite(u).all() and np.isfinite(j).all(),'Invalid or non-finite native scores'
 inter=data['intersections'][ii][idx,action];union=data['unions'][ii][idx,action]
 return dict(FP=int(((action>0)&~y).sum()),FN=int(((action==0)&y).sum()),negative=int((~y).sum()),positive=int(y.sum()),FPR=float(((action>0)&~y).sum()/max((~y).sum(),1)),FNR=float(((action==0)&y).sum()/max(y.sum(),1)),newFP=0,newMiss=0,netU=float(du.mean()),netIoU=float(dj[valid].mean()),gIoU=float(j[idx,action][valid].mean()),cIoU=float(inter[valid].sum()/max(union[valid].sum(),1)),harm=float(((action!=b)&(dj<0)&valid).sum()/max(valid.sum(),1)),harm002=float(((action!=b)&(dj<-.02)&valid).sum()/max(valid.sum(),1)),extraDecode=float(trial.mean()),trials=int(trial.sum()),delivered_changes=int((action!=b).sum()),repairs=int(((dj>0)&valid).sum()),harms=int(((dj<0)&valid).sum()))
