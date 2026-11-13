"""Native one-trial Qwen/SAM execution extracted from the measured worker.

Requires the matching frozen SFT checkpoint and its original configuration.
No ground-truth labels are accepted by these functions.
"""
import time
import numpy as np
import torch
from .pre_features import pre_candidates
from .feature_maps import changed_context39,whole_context39
from .recourse_features import paired_features
from .budget import allocate_shared_budget
from . import frozen_math as M
def baseline(model,processor,row,capture):
 from dataloaders.data_utils import Qwen3VLImageProcessor,configure_vision_budget,encode_prompt,SESAME_SYSTEM_PROMPT
 configure_vision_budget(processor,model.cfg.min_pixels,model.cfg.max_pixels)
 processed=Qwen3VLImageProcessor(processor,image_size=model.sam.image_encoder.img_size).process_image(row['image_path'],process_qwen=False)
 messages=[dict(role='system',content=SESAME_SYSTEM_PROMPT),dict(role='user',content=[dict(type='image',image=processed['image']),dict(type='text',text=row['question'])])]
 encoded=encode_prompt(processor,messages,processed['image'],model.cfg.max_length);data={k:v.to('cuda') for k,v in encoded.items() if torch.is_tensor(v)}
 cell=dict(row=row,natives=[],prefills=0,decodes=0,original_size=processed['original_size'])
 decode=model._sam_decode;proposals=model._proposals;meta=model.evidence.decision_metadata;forward=model._backbone_forward
 def d(embedding,semantic,boxes,size):
  value=decode(embedding,semantic,boxes,size);cell['decodes']+=1;cell['natives'].append(value.detach())
  if capture:cell.update(embedding=embedding.detach(),semantic=semantic.detach(),input_size=tuple(size))
  return value
 def p(*a,**k):
  value=proposals(*a,**k)
  if capture:cell['boxes']=value[0].detach()
  return value
 def m(features,prior,available):
  if capture:cell.update(features=features.detach().cpu(),prior=prior.detach().cpu(),available=available.detach().cpu())
  return meta(features,prior,available)
 def f(*a,**k):cell['prefills']+=1;return forward(*a,**k)
 model._sam_decode=d;model._proposals=p;model.evidence.decision_metadata=m;model._backbone_forward=f
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):result=model.generate(**data,sam_images=processed['sam_image'].to('cuda'),sam_mask_shape_list=[(processed['sam_input_size'],processed['original_size'])],explain=False,audit_candidates=False)
 finally:model._sam_decode=decode;model._proposals=proposals;model.evidence.decision_metadata=meta;model._backbone_forward=forward
 cell.update(result=result,baseline_action=int(result['evidence_actions'][0]));assert cell['prefills']==1 and result['response_tokens']==[0]
 if capture:cell['prediction']=dict(features=cell['features'],prior=cell['prior'],available=cell['available'],action=cell['baseline_action'],policy_diagnostics=result['policy_diagnostics'][0])
 return cell

def schedule(cells,experts,target):
 keys=np.asarray([c['row']['sample_key'] for c in cells]);n=len(keys);order=np.argsort(keys,kind='stable');mix=np.arange(n)%2;np.random.default_rng(42).shuffle(mix);assigned=np.empty(n,int);assigned[order]=mix
 proposals=np.zeros(n,int);values=np.zeros(n);base=np.asarray([c['baseline_action'] for c in cells])
 for i,c in enumerate(cells):
  e=experts[assigned[i]];candidates=[x for x in pre_candidates(c['prediction']) if x['action'] in e['support']]
  if candidates:
   v=M.predict(e[target],np.asarray([x['pre'] for x in candidates]));j=int(np.argmax(v));proposals[i]=candidates[j]['action'];values[i]=v[j]
 labels=[str(k)+'#'+str(i) for i,k in enumerate(keys)];budget=allocate_shared_budget(values,(base>0)&(proposals>0),np.zeros(n,bool),np.full(n,.25),labels,seed=42)
 return dict(proposal=proposals,expert=assigned,trials=budget['geometry_trials'],value=values)

@torch.inference_mode()
def deliver(model,cells,plan,experts,rep,max_trials=None):
 output=[];extra=0
 for i,c in enumerate(cells):
  b=c['baseline_action'];action=b;mask=c['result']['pred_masks'][0];native=c['natives'][-1][0] if b>0 else None;accepted=False
  if plan['trials'][i]:
   a=int(plan['proposal'][i]);assert a>1 and b>0
   with torch.autocast('cuda',dtype=torch.bfloat16):
    new=model._sam_decode(c['embedding'],c['semantic'],c['boxes'][a-1:a],c['input_size'])[0]
    full=model.sam.postprocess_masks(new[None,None],c['input_size'],c['original_size'])[0,0]
   extra+=1;oldfull=mask[0] if mask.ndim==3 else mask;features=c['features'][0].to('cuda')
   pair=paired_features(c['embedding'],c['semantic'],oldfull,full,c['input_size'],model.sam.image_encoder.img_size,native,new,features[b],features[a]);function=changed_context39 if rep=='changed' else whole_context39
   phi=function(pair['forward'][rep].cpu().numpy(),pair['reverse'][rep].cpu().numpy(),float(pair['full_image_jaccard_distance']),b,a);e=experts[plan['expert'][i]]
   accepted=float(M.predict(e['u'],phi[None])[0])>0 and float(M.predict(e['j'],phi[None])[0])>=0
   if accepted:action=a;native=new;mask=full[None]
   else:assert mask is c['result']['pred_masks'][0]
  output.append(dict(action=action,native=native,mask=mask,accepted=accepted,trial=bool(plan['trials'][i])))
 assert extra==int(plan['trials'].sum()) and extra<=(len(cells)//4 if max_trials is None else max_trials)
 return output,extra

def streamed_block(model,processor,rows,experts,target,rep,arm):
 # For N divisible by4, no rescues and p_i<=.25, the shared cap never binds.
 # Thus every eligible query has p_i=.25; identity-order systematic rounding
 # can be applied causally and is exactly the existing block policy.
 n=len(rows);assert n%4==0;torch.cuda.synchronize();start=time.perf_counter()
 keys=np.asarray([row['sample_key'] for row in rows]);order=np.argsort(keys,kind='stable');mix=np.arange(n)%2;np.random.default_rng(42).shuffle(mix);expert_index=np.empty(n,int);expert_index[order]=mix
 offset=np.random.default_rng(42).random();mass=0.;cells=[None]*n;outputs=[None]*n;service=[];arrival=[];extra_total=0
 record=dict(proposal=np.zeros(n,int),value=np.zeros(n),expert=expert_index.copy(),trials=np.zeros(n,bool))
 for i in order:
  torch.cuda.synchronize();query_start=time.perf_counter();c=baseline(model,processor,rows[i],arm!='SFT');torch.cuda.synchronize();service.append(time.perf_counter()-query_start);cells[i]=c
  if arm=='SFT':
   outputs[i]=dict(action=c['baseline_action'],native=c['natives'][-1][0] if c['baseline_action']>0 else None,mask=c['result']['pred_masks'][0],trial=False)
  else:
   e=experts[expert_index[i]];cs=[x for x in pre_candidates(c['prediction']) if x['action'] in e['support']];a=0;v=0.
   if cs:
    values=M.predict(e[target],np.asarray([x['pre'] for x in cs]));j=int(np.argmax(values));a=int(cs[j]['action']);v=float(values[j])
   probability=.25 if c['baseline_action']>0 and a>0 and v>0 else 0.
   trial=bool(np.floor(mass+probability+offset)-np.floor(mass+offset));mass+=probability
   plan=dict(proposal=np.asarray([a]),expert=np.asarray([0]),trials=np.asarray([trial],bool))
   delivered,extra=deliver(model,[c],plan,[e,e],rep,max_trials=1);outputs[i]=delivered[0];extra_total+=extra
   record['proposal'][i]=a;record['value'][i]=v;record['trials'][i]=trial
  torch.cuda.synchronize();arrival.append(time.perf_counter()-start)
 torch.cuda.synchronize();duration=time.perf_counter()-start;assert extra_total<=n//4
 return dict(cells=cells,outputs=outputs,record=record,service=service,arrival=arrival,extra=extra_total,duration=duration)
