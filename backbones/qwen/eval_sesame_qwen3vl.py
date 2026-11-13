"""Strict checkpoint evaluation: image-grounded presence, masks, and actual cost."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from dataloaders.data_utils import Qwen3VLImageProcessor,configure_vision_budget,encode_prompt,SESAME_SYSTEM_PROMPT
from runtime_utils import load_bundle
from model.evidence_policy import mask_quality


def _as_bool(value,default=True):
    if value is None:
        return default
    return value.strip().lower() in {'true','1','yes','present','exists'} if isinstance(value,str) else bool(value)


def load_eval_samples(dataset_dir,split,dataset_names,num_samples=-1):
    """The optional limit is stratified within each dataset, independent of order."""
    root=Path(dataset_dir);samples=[]
    for name in dataset_names.split('||'):
        file=root/name/(split+'.json')
        if not file.exists():
            continue
        data=json.loads(file.read_text(encoding='utf-8'))
        data=data if isinstance(data,list) else data['annotations']
        group=[]
        for ann in data:
            expression=ann.get('expression',ann.get('sent',ann.get('question','')))
            image=ann.get('image_path',ann.get('image',''))
            if not expression or not image:
                raise ValueError(f'Invalid annotation in {file}')
            exists=_as_bool(ann.get('exists'),not name.lower().startswith(('fp','r-ref','negative','counterfactual')))
            def asset(key):
                path=ann.get(key)
                return str((root/path).resolve()) if path else None
            group.append({'id':ann.get('id'),'dataset':name,'image_path':str((root/image).resolve()),
                'expression':expression,'question':ann.get('question') or f'Locate the requested target if it is visible: {expression}.',
                'exists':exists,'mask_path':asset('mask_path'),'ignore_mask_path':asset('ignore_mask_path')})
        if 0<num_samples<len(group):
            present=[s for s in group if s['exists']];absent=[s for s in group if not s['exists']]
            if present and absent:
                np_=min(len(present),(num_samples+1)//2);nn_=min(len(absent),num_samples-np_)
                np_=min(len(present),num_samples-nn_)
            else:
                np_,nn_=min(num_samples,len(present)),min(num_samples,len(absent))
            group=[present[i] for i in np.linspace(0,len(present)-1,np_,dtype=int)] if np_ else []
            if nn_:
                group += [absent[i] for i in np.linspace(0,len(absent)-1,nn_,dtype=int)]
        samples.extend(group)
    return samples


def evaluate_see(predictions,gt_exists):
    tp=sum(bool(p) and bool(g) for p,g in zip(predictions,gt_exists))
    tn=sum(not bool(p) and not bool(g) for p,g in zip(predictions,gt_exists))
    fp=sum(bool(p) and not bool(g) for p,g in zip(predictions,gt_exists))
    fn=sum(not bool(p) and bool(g) for p,g in zip(predictions,gt_exists))
    return {'accuracy':(tp+tn)/max(tp+tn+fp+fn,1),'positive_recall':tp/(tp+fn) if tp+fn else None,
            'absence_fpr':fp/(fp+tn) if fp+tn else None,'tp':tp,'tn':tn,'fp':fp,'fn':fn}


def evaluate_segmentation(pred_masks,gt_masks):
    intersections=[];unions=[]
    for p,g in zip(pred_masks,gt_masks):
        p,g=p>0,g>0
        intersections.append(float((p&g).sum()));unions.append(float((p|g).sum()))
    return {'giou':float(np.mean([a/max(b,1) for a,b in zip(intersections,unions)])),
            'ciou':sum(intersections)/max(sum(unions),1),'num_samples':len(unions)}


@torch.inference_mode()
def evaluate_model(model,processor,samples,device,dtype=torch.bfloat16,max_new_tokens=64,visual_control='original',
                   candidate_audit=False):
    if not samples:
        raise ValueError('Empty evaluation set')
    if candidate_audit and (model.cfg.evidence_policy_version not in (4,5) or
            not model.evidence.rl_active.item() or visual_control!='original'):
        raise ValueError('Candidate auditing requires a trained/initialized V4/V5 with original images')
    was_training=model.training;model.eval()
    configure_vision_budget(processor,model.cfg.min_pixels,model.cfg.max_pixels)
    image_processor=Qwen3VLImageProcessor(processor,image_size=model.sam.image_encoder.img_size)
    records=[];actions=Counter();decoder_calls=0;tokens_total=0;visual_tokens=0
    unique_images=list(dict.fromkeys(s['image_path'] for s in samples))
    if visual_control=='shuffle' and len(unique_images)<2:
        raise ValueError('Image-shuffle control requires at least two distinct images')
    swaps={image:unique_images[(i+1)%len(unique_images)] for i,image in enumerate(unique_images)}
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for index,sample in enumerate(samples):
            start=time.perf_counter()
            # Controls alter both visual inputs; the query/label stays unchanged.
            image_path=swaps[sample['image_path']] if visual_control=='shuffle' else sample['image_path']
            processed=image_processor.process_image(image_path,process_qwen=False)
            if visual_control=='blank':
                processed['image']=Image.new('RGB',processed['image'].size)
                processed['sam_image'].zero_()
            messages=[{'role':'system','content':SESAME_SYSTEM_PROMPT},{'role':'user','content':[
                {'type':'image','image':processed['image']},{'type':'text','text':sample['question']}]}]
            encoded=encode_prompt(processor,messages,processed['image'],model.cfg.max_length)
            inputs={k:v.to(device) for k,v in encoded.items() if torch.is_tensor(v)}
            if device.type=='cuda':
                torch.cuda.synchronize(device)
            with torch.autocast(device.type,dtype=dtype,enabled=device.type=='cuda' and dtype!=torch.float32):
                result=model.generate(**inputs,sam_images=processed['sam_image'].to(device),
                    sam_mask_shape_list=[(processed['sam_input_size'],processed['original_size'])],max_new_tokens=max_new_tokens,
                    audit_candidates=candidate_audit and sample['exists'])
            if device.type=='cuda':
                torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-start
            present=bool(result['object_presence'][0])
            mask=result['pred_masks'][0]
            mask=mask[0].float().cpu() if mask is not None else None
            mask_area=float((mask>0).float().mean()) if mask is not None else 0.0
            intersection=union=iou=None
            empty_positive=False
            if sample['exists']:
                if not sample['mask_path']:
                    raise ValueError('Positive evaluation item is missing a mask: '+str(sample['id']))
                target=torch.from_numpy(np.array(Image.open(sample['mask_path']).convert('L'))>128)
                valid=torch.ones_like(target)
                if sample.get('ignore_mask_path'):
                    valid &= torch.from_numpy(np.array(Image.open(sample['ignore_mask_path']).convert('L')))<=128
                if mask is not None and mask.shape!=target.shape:
                    if visual_control=='original':
                        raise ValueError('Image/mask dimensions differ; do not silently resize evaluation masks')
                    mask=torch.nn.functional.interpolate(mask[None,None],size=target.shape,mode='bilinear',align_corners=False)[0,0]
                pred=torch.zeros_like(target) if mask is None else mask>0
                intersection=float((pred&target&valid).sum());union=float(((pred|target)&valid).sum())
                empty_positive=not bool((target&valid).any())
                if not empty_positive:
                    iou=intersection/max(union,1)
            response=(result.get('responses') or [processor.tokenizer.decode(result['output_ids'][0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)])[0]
            text_tokens=(result.get('response_tokens') or [result['output_ids'].shape[1]-inputs['input_ids'].shape[1]])[0]
            grid=inputs.get('image_grid_thw')
            nvisual=int(grid.prod(-1).sum())//(processor.image_processor.merge_size**2) if grid is not None else 0
            action=result.get('evidence_actions',[-1])[0]
            calls=result.get('decoder_calls',[int(mask is not None)])[0]
            decoder_calls+=calls;tokens_total+=text_tokens;visual_tokens+=nvisual;actions[str(action)]+=1
            record={'id':sample['id'],'dataset':sample['dataset'],'gt_exists':sample['exists'],
                'image_path':sample['image_path'],
                'pred_exists':present,'iou':iou,'intersection':intersection,'union':union,
                'empty_positive_mask':empty_positive,'response':response,'autoregressive_text_tokens':text_tokens,
                'mask_area_fraction':mask_area,
                'visual_tokens':nvisual,'sam_decoder_calls':calls,'action':action,
                'latency_seconds':None if candidate_audit else elapsed,
                'diagnostic_latency_seconds':elapsed if candidate_audit else None}
            diagnostic=(result.get('policy_diagnostics') or [None])[0]
            if diagnostic is not None:
                record['policy_diagnostics']=diagnostic
            if candidate_audit:
                record['candidate_iou']=None
                record['candidate_utility']=None
                if sample['exists']:
                    candidates=result['evidence_candidate_masks'][0]
                    record['candidate_iou']=[float(((m>0)&target&valid).sum())/
                        max(float((((m>0)|target)&valid).sum()),1) for m in candidates]
                    low=result['evidence_candidate_logits'][0]
                    size=processed['sam_input_size'];canvas=model.sam.image_encoder.img_size
                    grid_target=model._mask_on_sam_grid(target,size,canvas,low.shape[-2:],low.device,torch.float32)
                    grid_valid=model._mask_on_sam_grid(torch.ones_like(target),size,canvas,low.shape[-2:],low.device,torch.float32)
                    if sample.get('ignore_mask_path'):
                        grid_valid*=1-model._mask_on_sam_grid(~valid,size,canvas,low.shape[-2:],low.device,torch.float32)
                    quality=mask_quality(low,grid_target,grid_valid)
                    utilities=2*quality-1
                    utilities[1:]-=model.cfg.evidence_cost
                    record['candidate_utility']=utilities.tolist()
            records.append(record)
    finally:
        model.train(was_training)
    see=evaluate_see([r['pred_exists'] for r in records],[r['gt_exists'] for r in records])
    positives=[r for r in records if r['iou'] is not None]
    giou=float(np.mean([r['iou'] for r in positives])) if positives else None
    ciou=sum(r['intersection'] for r in positives)/max(sum(r['union'] for r in positives),1) if positives else None
    negative_recall=1-see['absence_fpr'] if see['absence_fpr'] is not None else None
    negatives=[r for r in records if not r['gt_exists']]
    if giou is not None and negative_recall is not None:
        joint=2*giou*negative_recall/max(giou+negative_recall,1e-9)
    else:
        joint=giou if giou is not None else negative_recall
    audited=[r for r in positives if r.get('candidate_iou') is not None]
    changed=[];opportunities=0;opportunities_taken=0
    for r in audited:
        d=r['policy_diagnostics'];anchor=d['anchor_action']-1;action=r['action']
        ious=r['candidate_iou'];utility=r['candidate_utility']
        safe_gains=[utility[j]-utility[anchor] for j,ok in enumerate(d['available'])
                    if ok and j!=anchor and ious[j]>=ious[anchor]-model.cfg.evidence_repair_iou_tolerance]
        opportunity=bool(safe_gains) and max(safe_gains)>model.cfg.evidence_gain_margin
        opportunities+=int(opportunity)
        if action>0 and action!=anchor+1:
            delta=ious[action-1]-ious[anchor]
            gain=utility[action-1]-utility[anchor]
            harmful=delta < -model.cfg.evidence_repair_iou_tolerance
            changed.append((delta,gain,harmful))
            opportunities_taken+=int(opportunity and not harmful and gain>model.cfg.evidence_gain_margin)
    harms=sum(x[2] for x in changed)
    repair_metrics={'scope':'Enumerated full-resolution IoU against frozen conditional SFT action; diagnostic cost excluded from deployment timing',
        'valid_positive_queries':len(audited),'changed_queries':len(changed),
        'additional_harm_rate':harms/len(audited) if audited else None,
        'harm_given_change':harms/len(changed) if changed else None,
        'mean_changed_iou_gain':float(np.mean([x[0] for x in changed])) if changed else None,
        'mean_changed_net_gain':float(np.mean([x[1] for x in changed])) if changed else None,
        'oracle_opportunities':opportunities,'opportunities_taken':opportunities_taken,
        'opportunity_recall':opportunities_taken/opportunities if opportunities else None,
        'iou_tolerance':model.cfg.evidence_repair_iou_tolerance if candidate_audit else None}
    return {'num_evaluated':len(records),'num_failures':0,'visual_control':visual_control,'see_metrics':see,
        'candidate_audit':candidate_audit,'repair_metrics':repair_metrics,
        'segmentation_metrics':{'giou':giou,'ciou':ciou,'num_positive_masks':len(positives),
            'empty_positive_masks_excluded':sum(r['empty_positive_mask'] for r in records)},
        'joint_score':joint,'joint_score_definition':'harmonic mean of positive gIoU and absence specificity; checkpoint-selection metric, not the official SESAME score',
        'absence_mask_metrics':{'nonempty_mask_rate':float(np.mean([r['mask_area_fraction']>0 for r in negatives])) if negatives else None,
            'mean_hallucinated_area_fraction':float(np.mean([r['mask_area_fraction'] for r in negatives])) if negatives else None},
        'cost':{'autoregressive_text_tokens_mean':tokens_total/len(records),'visual_tokens_mean':visual_tokens/len(records),
            'sam_decoder_calls_mean':None if candidate_audit else decoder_calls/len(records),'actions':dict(actions),
            'latency_median_seconds':None if candidate_audit else float(np.median([r['latency_seconds'] for r in records])),
            'latency_mean_seconds':None if candidate_audit else float(np.mean([r['latency_seconds'] for r in records])),
            'latency_p95_seconds':None if candidate_audit else float(np.percentile([r['latency_seconds'] for r in records],95)),
            'diagnostic_latency_mean_seconds':float(np.mean([r['diagnostic_latency_seconds'] for r in records])) if candidate_audit else None,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None},
        'predictions':records}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--model_path',default=None)
    p.add_argument('--vision_pretrained',default=None)
    p.add_argument('--dataset_dir',required=True)
    p.add_argument('--eval_datasets',default='refcoco||refcoco+||refcocog||fprefcoco||fprefcoco+||fprefcocog')
    p.add_argument('--eval_split',choices=['val','test'],default='val')
    p.add_argument('--num_samples',type=int,default=-1,help='Stratified limit per dataset; -1 evaluates all')
    p.add_argument('--output_dir',default='./eval_results')
    p.add_argument('--device',default='cuda')
    p.add_argument('--max_new_tokens',type=int,default=64)
    p.add_argument('--attn_implementation',default='sdpa')
    p.add_argument('--visual_control',choices=['original','blank','shuffle'],default='original')
    p.add_argument('--evidence_mode',choices=['learned','base'],default='learned',help='base bypasses the evidence policy, preserving the same trained backbone/heads')
    p.add_argument('--null_bias',type=float,default=None,help='Inference rejection calibration; choose on validation only')
    p.add_argument('--candidate_audit',action='store_true',help='Enumerate repair outcomes; explicitly excludes deployment timing')
    p.add_argument('--risk_certificate',default=None,help='Independent policy certificate bound to this exact checkpoint/protocol')
    a=p.parse_args();device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model,_,processor,_=load_bundle(a.checkpoint,a.model_path,a.vision_pretrained,a.attn_implementation)
    if device.type=='cpu':
        model=model.float()
    model.to(device)
    if a.evidence_mode=='base':model.cfg.evidence_enabled=False
    if a.null_bias is not None:model.cfg.evidence_null_bias=a.null_bias
    samples=load_eval_samples(a.dataset_dir,a.eval_split,a.eval_datasets,a.num_samples)
    if a.risk_certificate:
        if a.null_bias is not None or a.evidence_mode!='learned' or a.visual_control!='original':
            raise ValueError('Certified policies cannot change rejection bias, visual inputs or evidence mode')
        from repair_risk import apply_certificate,assert_disjoint_images
        certificate=json.loads(Path(a.risk_certificate).read_text(encoding='utf-8'))
        hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'
        apply_certificate(model,a.checkpoint,certificate,'bfloat16' if device.type=='cuda' else 'float32',hardware)
        assert_disjoint_images(samples,certificate)
    result=evaluate_model(model,processor,samples,device,dtype=torch.bfloat16 if device.type=='cuda' else torch.float32,
                          max_new_tokens=a.max_new_tokens,visual_control=a.visual_control,candidate_audit=a.candidate_audit)
    result['checkpoint']=str(Path(a.checkpoint).resolve());result['evidence_mode']=a.evidence_mode
    result['null_bias']=model.cfg.evidence_null_bias
    result['risk_certificate']=model.evidence.calibration_info if hasattr(model.evidence,'calibration_info') else None
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    (root/'eval_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2))


if __name__=='__main__':
    main()
