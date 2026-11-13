"""Two explicit stages: segmentation SFT, then a stationary visual bandit."""
import argparse
import contextlib
import json
import math
import os
from functools import partial
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from model.qwen3vl_config import get_default_config
from model.sesame_qwen3vl import build_sesame_qwen3vl,load_tokenizer_and_processor,add_seg_token,get_chat_eos_token_id
from dataloaders.data_utils import configure_vision_budget,collate_fn_train
from dataloaders.sesame_datasets import HybridDataset
from runtime_utils import move_to_device,seed_everything,seed_worker,unwrap,load_bundle,save_checkpoint,restore_rng,gather_rng_states
from training_normalization import policy_window_normalizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage',choices=['sft','rl','policy_sft'],default='sft')
    p.add_argument('--evidence_policy_version',type=int,choices=[2,3,4,5],default=None,
        help='Legacy default remains v3; explicitly select v4/v5; resume retains its version')
    p.add_argument('--evidence_solver_revision',type=int,choices=[1,2,3],default=None)
    for name in ('evidence_outcome_weight','evidence_repairability_threshold',
                 'evidence_harm_probability_threshold','evidence_repair_iou_tolerance',
                 'evidence_repair_harm_target','evidence_switch_temperature',
                 'evidence_primal_radius','evidence_dual_cap','evidence_ridge'):
        p.add_argument('--'+name,type=float,default=None)
    p.add_argument('--variant',choices=['sesame','prompt','evidence'],default='evidence')
    p.add_argument('--model_size',default='4B',choices=['2B','4B','8B','32B'])
    p.add_argument('--model_path',default=None)
    p.add_argument('--vision_pretrained',default=None)
    p.add_argument('--checkpoint',default=None,help='Initialize a new stage from a v2 checkpoint')
    p.add_argument('--resume',default=None,help='Resume optimizer/scheduler/scaler and random states')
    p.add_argument('--dataset_dir',required=True)
    p.add_argument('--dataset',default='refer_seg||correct_refer_seg||vqa||neg_refer_seg')
    p.add_argument('--sample_rates',default='9,3,3,3')
    p.add_argument('--dataset_specs',default=None,help='JSON mapping task aliases to root/names/paired')
    p.add_argument('--repair_partition',default=None,help='Immutable image reservation required for new V4/V5 policy stages')
    p.add_argument('--output_dir',default='./runs/evidence')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--stop_after_epoch',type=int,default=None,
        help='Stop at this completed epoch while preserving the full --epochs scheduler horizon')
    p.add_argument('--steps_per_epoch',type=int,default=500)
    p.add_argument('--batch_size',type=int,default=1)
    p.add_argument('--grad_accumulation_steps',type=int,default=8)
    p.add_argument('--lr',type=float,default=None)
    p.add_argument('--weight_decay',type=float,default=.01)
    p.add_argument('--warmup_ratio',type=float,default=.03)
    p.add_argument('--precision',choices=['fp32','fp16','bf16'],default='bf16')
    p.add_argument('--attn_implementation',default='sdpa',choices=['sdpa','eager','flash_attention_2'])
    p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--data_seed',type=int,default=None,
        help='Independent worker seed; v3/v4 default to --seed and re-seed per epoch')
    p.add_argument('--max_length',type=int,default=2048)
    p.add_argument('--max_pixels',type=int,default=1048576)
    p.add_argument('--lora_r',type=int,default=8)
    p.add_argument('--gradient_checkpointing',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--use_wandb',action='store_true')
    p.add_argument('--eval_every',type=int,default=1)
    p.add_argument('--val_samples_per_dataset',type=int,default=100)
    p.add_argument('--save_every',type=int,default=1)
    p.add_argument('--allow_data_overlap',action='store_true',help='Explicitly permit a non-SESAME image-overlap protocol')
    a = p.parse_args()
    if min(a.epochs,a.steps_per_epoch,a.batch_size,a.grad_accumulation_steps)<1:
        p.error('Training sizes must be positive')
    if a.stop_after_epoch is not None and not 1<=a.stop_after_epoch<=a.epochs:
        p.error('stop_after_epoch must be between 1 and epochs')
    if a.resume and a.checkpoint:
        p.error('Use either resume or checkpoint')
    if a.stage!='sft' and not (a.resume or a.checkpoint):
        p.error('RL requires an SFT checkpoint; random-policy RL is not a supported recipe')
    return a


def build_dataset(args,processor,world_size=1):
    names = args.dataset.split('||'); rates = [float(x) for x in args.sample_rates.split(',')]
    if len(names)!=len(rates):
        raise ValueError('Dataset/rate lengths differ')
    if args.stage!='sft':
        choices = [(n,r) for n,r in zip(names,rates) if n!='vqa']
        names,rates = map(list,zip(*choices))
    specs = json.loads(Path(args.dataset_specs).read_text(encoding='utf-8')) if args.dataset_specs else None
    if specs:
        for spec in specs.values():
            for key in ('root','eval_root'):
                if key in spec and not Path(spec[key]).is_absolute():
                    spec[key]=str((Path(args.dataset_dir)/spec[key]).resolve())
    return HybridDataset(args.dataset_dir,processor,
        samples_per_epoch=args.steps_per_epoch*args.batch_size*args.grad_accumulation_steps*world_size,
        dataset='||'.join(names),sample_rate=rates,dataset_specs=specs)


def split_preflight(dataset,allow_overlap=False,return_paths=False):
    """Check public split metadata before allocating GPU training memory."""
    train_assets,heldout_assets = set(),set()
    for ds in dataset.datasets:
        root = Path(ds.base_image_dir)
        for ann in getattr(ds,'annotations',getattr(ds,'vqa_data',[])):
            path = ann.get('image_path',ann.get('image',''))
            if path:
                train_assets.add((str(root),path))
        for name in getattr(ds,'dataset_names',[]):
            for split in ('val','test','testA','testB'):
                path = Path(getattr(ds,'eval_root',root))/name/(split+'.json')
                if path.exists():
                    annotations = json.loads(path.read_text(encoding='utf-8'))
                    annotations = annotations if isinstance(annotations,list) else annotations['annotations']
                    heldout_assets.update((str(path.parent.parent),a.get('image_path',a.get('image',''))) for a in annotations)
    # Resolve unique assets once; a COCO image can appear in hundreds of queries.
    canonical={(root,path):str((Path(root)/path).resolve()) for root,path in train_assets|heldout_assets}
    train_images={canonical[key] for key in train_assets}
    heldout={canonical[key] for key in heldout_assets}
    overlap = train_images&heldout
    if overlap and not allow_overlap:
        raise ValueError(f'{len(overlap)} train/heldout image overlaps. Use the prepared SESAME protocol or explicitly choose another protocol.')
    report={'train_images':len(train_images),'heldout_images':len(heldout),'overlap_images':len(overlap)}
    if return_paths:report.update(_train_paths=train_images,_heldout_paths=heldout)
    return report


def repair_partition_preflight(partition,train_paths,heldout_paths):
    """Validate the frozen evaluation groups independently of training task names.

    A reserved val dataset may not be sampled for training (e.g. ReasonSeg).
    Keep the original path split check, and additionally reject content aliases
    of reserved images anywhere in the training metadata.
    """
    from eval_sesame_qwen3vl import load_eval_samples
    from repair_risk import file_sha256
    groups=partition.get('dataset_groups')
    if not groups or any(g.get('split','val')!='val' for g in groups):
        raise ValueError('Reserved partitions require declared validation-only dataset groups')
    declared=set()
    for group in groups:
        declared.update(str(Path(s['image_path']).resolve()) for s in
            load_eval_samples(group['dataset_dir'],'val',group['names'],-1))
    expected={}
    for sample in partition['samples']:
        path=str(Path(sample['image_path']).resolve())
        if path in expected and expected[path]!=sample['image_key']:
            raise ValueError('Inconsistent reserved image identity')
        expected[path]=sample['image_key']
    reserved=set(expected)
    if not reserved or reserved & train_paths or not reserved<=declared:
        raise ValueError('Reserved images must belong exclusively to their frozen validation groups')
    for path,digest in expected.items():
        if file_sha256(path)!=digest:
            raise ValueError('Reserved image content changed: '+path)
    keys=set(expected.values())
    for path in sorted(train_paths):
        if file_sha256(path) in keys:
            raise ValueError('Training image content overlaps reserved calibration/assessment: '+path)
    return dict(passed=True,declared_validation_paths=len(declared),
        reserved_paths=len(reserved),reserved_image_identities=len(keys),
        extra_declared_validation_paths=len(reserved-heldout_paths),
        training_paths_content_checked=len(train_paths),training_content_overlaps=0)


def validation_samples(dataset,limit):
    from eval_sesame_qwen3vl import load_eval_samples
    samples = []
    visited = set()
    for ds in dataset.datasets:
        for name in getattr(ds,'dataset_names',[]):
            eval_root=getattr(ds,'eval_root',ds.base_image_dir)
            identity = (eval_root,name)
            if identity not in visited:
                visited.add(identity)
                samples.extend(load_eval_samples(eval_root,'val',name,limit))
    return samples


def checkpoint_admissible(result,anchor):
    """Validation guard; it does not guarantee performance on unseen data."""
    conditions=[(result['segmentation_metrics']['giou'],anchor['segmentation_metrics']['giou'],.002,True),
                (result['see_metrics']['positive_recall'],anchor['see_metrics']['positive_recall'],.005,True),
                (result['see_metrics']['absence_fpr'],anchor['see_metrics']['absence_fpr'],.005,False)]
    for current,baseline,tolerance,larger_is_better in conditions:
        if baseline is None:continue
        if current is None:return False
        if larger_is_better and current<baseline-tolerance:return False
        if not larger_is_better and current>baseline+tolerance:return False
    return True


def main():
    a = parse_args()
    world,rank,local = int(os.environ.get('WORLD_SIZE',1)),int(os.environ.get('RANK',0)),int(os.environ.get('LOCAL_RANK',0))
    if world>1:
        dist.init_process_group('nccl');torch.cuda.set_device(local)
    device = torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    seed_everything(a.seed+rank)
    dtype = {'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[a.precision]
    if device.type=='cpu':
        dtype = torch.float32
    checkpoint = None
    if a.resume or a.checkpoint:
        model,tokenizer,processor,checkpoint = load_bundle(a.resume or a.checkpoint,a.model_path,a.vision_pretrained,a.attn_implementation)
        if a.stage=='sft' and model.evidence.rl_active.item():
            raise ValueError('An RL checkpoint cannot resume SFT without an explicit stage conversion')
    else:
        if not a.vision_pretrained or not Path(a.vision_pretrained).is_file():
            raise FileNotFoundError('Supply --vision_pretrained with the official SAM checkpoint before loading the backbone.')
        cfg = get_default_config(a.model_size)
        cfg.model_name_or_path = a.model_path or cfg.model_name_or_path
        cfg.vision_pretrained = a.vision_pretrained or ''
        cfg.torch_dtype = {'bf16':'bfloat16','fp16':'float16','fp32':'float32'}[a.precision] if device.type=='cuda' else 'float32'
        cfg.attn_implementation,cfg.lora_r = a.attn_implementation,a.lora_r
        cfg.evidence_enabled = a.variant=='evidence'
        cfg.query_source = 'seg' if a.variant=='sesame' else 'prompt'
        if a.variant=='sesame':
            cfg.box_prompt_enabled=False;cfg.presence_gate=False
            cfg.box_loss_weight=0;cfg.presence_loss_weight=0
        tokenizer,processor = load_tokenizer_and_processor(cfg.model_name_or_path)
        marker = add_seg_token(tokenizer)
        cfg.eos_token_id = get_chat_eos_token_id(tokenizer)
        model = None
    cfg = model.cfg if model is not None else cfg
    if a.resume:
        if a.evidence_policy_version is not None and a.evidence_policy_version != cfg.evidence_policy_version:
            raise ValueError('Resume must retain the checkpoint policy version; convert only a new stage')
    else:
        requested_version = a.evidence_policy_version or 3
        if model is None:
            cfg.evidence_policy_version = requested_version
        elif a.stage != 'sft':
            model.upgrade_evidence_policy(requested_version)
        elif requested_version != cfg.evidence_policy_version:
            raise ValueError('Do not convert SFT weights implicitly; select their saved policy version')
    a.evidence_policy_version = cfg.evidence_policy_version
    requested_solver=a.evidence_solver_revision
    if requested_solver is not None and cfg.evidence_policy_version!=5:
        raise ValueError('Solver revisions apply only to V5')
    if a.resume:
        if requested_solver is not None and requested_solver!=cfg.evidence_solver_revision:
            raise ValueError('Resume must retain the solver revision')
    elif cfg.evidence_policy_version==5:
        cfg.evidence_solver_revision=3 if requested_solver is None else requested_solver
    a.evidence_solver_revision=cfg.evidence_solver_revision
    for name in ('evidence_outcome_weight','evidence_repairability_threshold',
                 'evidence_harm_probability_threshold','evidence_repair_iou_tolerance',
                 'evidence_repair_harm_target','evidence_switch_temperature',
                 'evidence_primal_radius','evidence_dual_cap','evidence_ridge'):
        value = getattr(a,name)
        if value is not None:
            if cfg.evidence_policy_version not in (4,5) or (a.resume and value != getattr(cfg,name)):
                raise ValueError('New outcome/risk settings require a new v4/v5 stage; resume keeps saved values')
            setattr(cfg,name,value)
        setattr(a,name,getattr(cfg,name))
    cfg.__post_init__()
    convex_policy = cfg.evidence_policy_version == 5
    if convex_policy and (a.stage=='sft' or world!=1 or a.precision=='fp16' or a.weight_decay!=0):
        raise ValueError('V5 requires a new/resumed policy stage, one GPU, bf16/fp32 and --weight_decay 0; ridge is explicit')
    if a.lr is None:
        a.lr = (checkpoint['args'].get('lr') if a.resume else
                .01 if convex_policy else 1e-4 if a.stage!='sft' else 2e-4)
    if a.resume and convex_policy:
        for name in ('lr','epochs','steps_per_epoch','grad_accumulation_steps','batch_size',
                     'warmup_ratio','weight_decay','seed','precision','dataset','sample_rates',
                     'dataset_specs','max_length','max_pixels','attn_implementation',
                     'val_samples_per_dataset','allow_data_overlap'):
            if getattr(a,name)!=checkpoint['args'].get(name):
                raise ValueError('V5 resume must preserve solver/data budget: '+name)
    if a.resume:
        saved_data_seed = checkpoint['args'].get('data_seed')
        if a.data_seed is not None and a.data_seed != saved_data_seed:
            raise ValueError('Resume must retain the saved data seed')
        a.data_seed = saved_data_seed
    elif a.data_seed is None and cfg.evidence_policy_version >= 3:
        a.data_seed = a.seed
    cfg.max_length,cfg.max_pixels = a.max_length,a.max_pixels
    configure_vision_budget(processor,cfg.min_pixels,cfg.max_pixels)
    data = build_dataset(a,processor,world)
    reserved_v4=cfg.evidence_policy_version>=4 and a.stage!='sft'
    protocol = split_preflight(data,a.allow_data_overlap,return_paths=reserved_v4)
    val = validation_samples(data,a.val_samples_per_dataset) if a.eval_every>0 else []
    if a.eval_every>0 and not val:
        raise ValueError('No validation samples found. Supply validation splits or explicitly set --eval_every 0 for a diagnostic run.')
    if reserved_v4:
        from repair_risk import file_sha256
        if not a.repair_partition or a.allow_data_overlap or not val:
            raise ValueError('Selective-repair comparisons require a reserved partition, disjoint training and validation')
        partition=json.loads(Path(a.repair_partition).read_text(encoding='utf-8'))
        train_paths,heldout_paths=protocol.pop('_train_paths'),protocol.pop('_heldout_paths')
        protocol['reservation_preflight']=repair_partition_preflight(partition,train_paths,heldout_paths)
        reserved_keys={s['image_key'] for s in partition['samples']}
        selection_keys={file_sha256(path) for path in {s['image_path'] for s in val}}
        if reserved_keys & selection_keys:
            raise ValueError('Checkpoint-selection images overlap reserved calibration/assessment')
        partition_hash=file_sha256(a.repair_partition)
        if a.resume and partition_hash!=checkpoint['args'].get('repair_partition_sha256'):
            raise ValueError('Resume must preserve the original reserved partition')
        a.repair_partition_sha256=partition_hash
        a.selection_image_keys=sorted(selection_keys)
        protocol['reserved_image_counts']=partition['counts']
    if model is None:
        model=build_sesame_qwen3vl(cfg,marker,lora=True,tokenizer=tokenizer)
    if a.stage!='sft':
        if not cfg.evidence_enabled:
            raise ValueError('RL needs a checkpoint with an evidence-policy SFT warm start')
        cfg.evidence_objective='expectation' if a.stage=='rl' else 'oracle_ce'
        if a.checkpoint and model.evidence.rl_active.item():
            raise ValueError('New RL/control stages start from SFT; trained policy checkpoints use --resume')
        if a.checkpoint:model.evidence.rl_active.fill_(False)
        model.start_rl_stage()
    elif a.gradient_checkpointing:
        model.qwen3vl.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if device.type=='cpu':
        model.float();cfg.torch_dtype='float32'
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if convex_policy:
        from policy_optimization import ProjectedMirrorProx
        optimizer = ProjectedMirrorProx(params,model.evidence,lr=a.lr,dual_lr=cfg.evidence_dual_lr,
            radius=cfg.evidence_primal_radius,ridge=cfg.evidence_ridge,dual_cap=cfg.evidence_dual_cap)
    else:
        optimizer = torch.optim.AdamW(params,lr=a.lr,betas=(.9,.95),weight_decay=a.weight_decay)
    total = a.epochs*a.steps_per_epoch;warm = max(1,round(total*a.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/warm, .5*(1+math.cos(math.pi*max(0,step-warm)/max(1,total-warm)))))
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and dtype==torch.float16)
    start = 0
    if a.resume:
        if checkpoint['args'].get('stage')!=a.stage:
            raise ValueError('Cross-stage initialization uses --checkpoint, not --resume')
        for key,obj in [('optimizer',optimizer),('scheduler',scheduler),('scaler',scaler)]:
            obj.load_state_dict(checkpoint[key+'_state_dict'])
        restore_rng(checkpoint);start=checkpoint['epoch']
    sampler = DistributedSampler(data,num_replicas=world,rank=rank,seed=a.seed)
    loader_generator = torch.Generator() if a.data_seed is not None else None
    loader = DataLoader(data,batch_size=a.batch_size,sampler=sampler,num_workers=a.num_workers,
        collate_fn=partial(collate_fn_train,processor=processor,max_length=cfg.max_length),
        worker_init_fn=seed_worker,pin_memory=device.type=='cuda',drop_last=True,generator=loader_generator)
    if world>1:
        model = DDP(model,device_ids=[local],find_unused_parameters=True)
    core = unwrap(model)
    output = Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        (output/'run_config.json').write_text(json.dumps({'args':vars(a),'protocol':protocol,
            'model_config':vars(cfg),'trainable_parameters':sum(p.numel() for p in params),
            'policy_recipe':{2:'legacy flat bandit',3:'window-balanced hierarchical gain bandit',
                4:'selective repairability and incremental IoU-harm bandit',
                5:'fixed-feature convex reward-derived policy fitting with projected extragradient'}[cfg.evidence_policy_version]},indent=2),encoding='utf-8')
        print('Trainable parameters:',sum(p.numel() for p in params),protocol,flush=True)
    wandb = None
    if a.use_wandb and rank==0:
        import wandb
        wandb.init(project='sesame-evidence',config=vars(a))
    best = checkpoint.get('best_score') if a.resume else None
    best = -float('inf') if best is None else best
    anchor=checkpoint.get('args',{}).get('validation_anchor') if a.resume else None
    if a.stage!='sft' and val and not a.resume:
        initial_rng=gather_rng_states()
        if rank==0:
            from eval_sesame_qwen3vl import evaluate_model
            initial=evaluate_model(core,processor,val,device,dtype=dtype,
                candidate_audit=cfg.evidence_policy_version>=4)
            anchor={k:initial[k] for k in ('see_metrics','segmentation_metrics')}
            a.validation_anchor=anchor;best=initial['joint_score']
            if best is None:raise ValueError('RL initialization has no usable validation score')
            (output/'validation_initial.json').write_text(json.dumps(initial,indent=2),encoding='utf-8')
            save_checkpoint(output/'best.pt',core,tokenizer,processor,optimizer,scheduler,scaler,0,0,a,initial_rng,best)
        if world>1:dist.barrier()
    elif anchor is not None:
        a.validation_anchor=anchor
    stop_epoch=a.epochs if a.stop_after_epoch is None else a.stop_after_epoch
    for epoch in range(start,stop_epoch):
        if loader_generator is not None:
            # Stage-independent and architecture-independent sampling. Restarting
            # at an epoch boundary uses the same worker streams as a continuous run.
            loader_generator.manual_seed(a.data_seed + rank + 1000003 * epoch)
        sampler.set_epoch(epoch);model.train();iterator=iter(loader)
        for step in range(a.steps_per_epoch):
            optimizer.zero_grad(set_to_none=True);metrics={};loss_sum=0.0
            window_normalizer = None
            window_batches = None
            if a.stage != 'sft' and cfg.evidence_policy_version >= 3:
                # Only host batches are prefetched. Forward/backward still run
                # one microbatch at a time with the original memory budget.
                window_batches = [next(iterator) for _ in range(a.grad_accumulation_steps)]
                window_normalizer = policy_window_normalizer(window_batches, device,
                    include_quality=cfg.evidence_policy_version>=4)
            if convex_policy:
                core.evidence.begin_window()
            for micro in range(a.grad_accumulation_steps):
                batch=move_to_device(window_batches[micro] if window_batches is not None else next(iterator),device)
                if window_normalizer is not None:
                    batch['policy_normalizer'] = window_normalizer
                sync = model.no_sync() if world>1 and micro<a.grad_accumulation_steps-1 else contextlib.nullcontext()
                with sync:
                    with torch.autocast(device.type,dtype=dtype,enabled=device.type=='cuda' and dtype!=torch.float32):
                        result=model(**batch);loss=result['loss']/a.grad_accumulation_steps
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss at epoch={epoch}, step={step}')
                    scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
                for k,v in result['policy_metrics'].items():
                    metrics[k]=metrics.get(k,0)+v.detach()
            scaler.unscale_(optimizer)
            gate_grad_norm = value_grad_norm = None
            solver_info = None
            if convex_policy:
                def unmodified_norm(parameters):
                    gradients=[p.grad.detach().norm().square() for p in parameters if p.grad is not None]
                    return torch.stack(gradients).sum().sqrt() if gradients else torch.zeros((),device=device)
                gate_grad_norm = unmodified_norm(core.evidence.gate_adapter.parameters())
                value_grad_norm = unmodified_norm(core.evidence.geometry_parameters())
                grad_norm = (gate_grad_norm.square()+value_grad_norm.square()).sqrt()
            elif a.stage != 'sft' and cfg.evidence_policy_version >= 3:
                # A large geometry gradient must not shrink the independent
                # risk-gate update, or confound RL versus CE gate behavior.
                gate_grad_norm = torch.nn.utils.clip_grad_norm_(core.evidence.gate_adapter.parameters(),1.0,
                    error_if_nonfinite=not scaler.is_enabled())
                geometry_parameters = (core.evidence.geometry_parameters() if cfg.evidence_policy_version>=4
                                       else core.evidence.policy.parameters())
                value_grad_norm = torch.nn.utils.clip_grad_norm_(geometry_parameters,1.0,
                    error_if_nonfinite=not scaler.is_enabled())
                grad_norm = (gate_grad_norm.square() + value_grad_norm.square()).sqrt()
            else:
                grad_norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=not scaler.is_enabled())
            if convex_policy:
                contexts = core.evidence.end_window()
                solver_info = optimizer.step(contexts=contexts,metrics=metrics)
                del contexts
                if solver_info['accepted_steps']!=epoch*a.steps_per_epoch+step+1:
                    raise RuntimeError('The solver update count differs from the declared training budget')
                stepped = True
            else:
                before=scaler.get_scale();scaler.step(optimizer);scaler.update()
                stepped=scaler.get_scale()>=before
            if stepped:
                scheduler.step()
            if metrics:
                metrics=core.evidence.update_constraints(metrics,update=stepped and not convex_policy)
            loss_value=torch.tensor(loss_sum,device=device)
            if world>1:dist.all_reduce(loss_value);loss_value/=world
            if step%10==0 and rank==0:
                count=max(float(metrics.get('count',1)),1)
                log={'epoch':epoch,'step':step,'loss':float(loss_value),'lr':scheduler.get_last_lr()[0],
                     'grad_norm':float(grad_norm) if torch.isfinite(grad_norm) else None,'optimizer_step':stepped,'kl_beta':float(core.evidence.kl_beta),
                     'lambda_fp':float(core.evidence.lambda_fp),'lambda_fn':float(core.evidence.lambda_fn)}
                log.update({k:float(metrics.get(k,0)) for k in ('positive_count','negative_count')})
                risk_keys = {'soft_fp','soft_fn','hard_fp','hard_fn','risk_fp_upper','risk_fn_upper'}
                positive_keys = {k for k in metrics if k.startswith('positive_') and k!='positive_count'}
                log.update({k:float(v)/count for k,v in metrics.items() if k not in
                    {'count','negative_count','positive_count'} | risk_keys | positive_keys})
                for key in positive_keys:
                    log[key] = float(metrics[key])/max(float(metrics.get('positive_count',0)),1)
                for cls in ('fp','fn'):
                    denom=max(float(metrics.get('negative_count' if cls=='fp' else 'positive_count',0)),1)
                    for kind in ('soft','hard'):
                        log[kind+'_'+cls]=float(metrics.get(kind+'_'+cls,0))/denom
                    upper_key = 'risk_'+cls+'_upper'
                    if upper_key in metrics:
                        log[upper_key] = float(metrics[upper_key])/denom
                if window_normalizer is not None:
                    log['window_positive_count'] = window_normalizer['positive_count']
                    log['window_negative_count'] = window_normalizer['negative_count']
                    log['gate_grad_norm'] = float(gate_grad_norm) if torch.isfinite(gate_grad_norm) else None
                    log['value_grad_norm'] = float(value_grad_norm) if torch.isfinite(value_grad_norm) else None
                if solver_info is not None:
                    log.update({'solver_'+key:value for key,value in solver_info.items() if key!='probe_metrics'})
                    log['solver_head_gradient_evaluations'] = 1+solver_info['head_probe_evaluations']
                    log['solver_ridge'] = cfg.evidence_ridge
                    log['solver_certificate_scope'] = 'current training window only'
                if cfg.evidence_policy_version >= 4:
                    log['lambda_harm'] = float(core.evidence.lambda_harm)
                    def rate(numerator, denominator):
                        n = float(metrics.get(denominator,0))
                        return float(metrics.get(numerator,0))/n if n>0 else None
                    log.update(
                        quality_query_count=float(metrics.get('positive_iou_valid',0)),
                        changed_query_count=float(metrics.get('positive_changed',0)),
                        oracle_opportunity_count=float(metrics.get('positive_repair_opportunity',0)),
                        repair_harm_rate=rate('positive_changed_harm','positive_iou_valid'),
                        repair_harm_given_change=rate('positive_changed_harm','positive_changed'),
                        repair_harm_upper=rate('positive_harm_upper','positive_iou_valid'),
                        repair_harm_proxy_floor=rate('positive_harm_proxy_floor','positive_iou_valid'),
                        changed_mean_iou_gain=rate('positive_changed_iou_gain','positive_changed'),
                        changed_mean_net_gain=rate('positive_changed_net_gain','positive_changed'),
                        repair_opportunity_recall=rate('positive_opportunity_taken','positive_repair_opportunity'))
                print(json.dumps(log),flush=True)
                with (output/'train_metrics.jsonl').open('a',encoding='utf-8') as file:
                    file.write(json.dumps(log)+'\n')
                if wandb:
                    wandb.log(log)
        if world>1:
            dist.barrier()
        rng_by_rank=gather_rng_states()
        if rank==0:
            if val and (epoch+1)%a.eval_every==0:
                from eval_sesame_qwen3vl import evaluate_model
                result=evaluate_model(core,processor,val,device,dtype=dtype,
                    candidate_audit=cfg.evidence_policy_version>=4 and a.stage!='sft')
                # Joint quality includes false rejections; neither positives nor absences can dominate alone.
                score=result['joint_score']
                if score is None:
                    raise ValueError('Validation has no usable segmentation targets or absence samples')
                admissible=a.stage=='sft' or anchor is None or checkpoint_admissible(result,anchor)
                if cfg.evidence_policy_version>=4 and a.stage!='sft':
                    harm_rate=result['repair_metrics']['additional_harm_rate']
                    admissible = admissible and harm_rate is not None and harm_rate<=cfg.evidence_repair_harm_target
                    result['repair_guard_scope']='Empirical model-selection guard; not independent risk certification'
                result['checkpoint_admissible']=admissible
                (output/f'validation_epoch_{epoch+1}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                if score>best and admissible:
                    best=score
                    save_checkpoint(output/'best.pt',core,tokenizer,processor,optimizer,scheduler,scaler,epoch+1,(epoch+1)*a.steps_per_epoch,a,rng_by_rank,best)
            if (epoch+1)%a.save_every==0 or epoch+1==a.epochs:
                save_checkpoint(output/'last.pt',core,tokenizer,processor,optimizer,scheduler,scaler,epoch+1,(epoch+1)*a.steps_per_epoch,a,rng_by_rank,best)
        if world>1:
            dist.barrier()
    if wandb:
        wandb.finish()
    if world>1:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
