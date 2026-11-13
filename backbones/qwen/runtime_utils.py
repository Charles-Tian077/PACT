"""Shared device transfer, seeding and checkpoint contracts for every entry point."""
import dataclasses
import os
import random
import numpy as np
import torch


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def read_checkpoint(path):
    # v2 checkpoints contain tensor state and primitive metadata only.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 2:
        raise ValueError('Legacy checkpoint has no reliable architecture metadata. Convert explicitly; silent partial loading is disabled.')
    return checkpoint


def restore_state(model, checkpoint):
    state = checkpoint['model_state_dict']
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    # Every parameter, including frozen base/reference parameters, must match.
    return unwrap(model).load_state_dict(state, strict=True)


def save_checkpoint(path, model, tokenizer, processor, optimizer=None,
                    scheduler=None, scaler=None, epoch=0, step=0, args=None,rng_by_rank=None,best_score=None):
    core = unwrap(model)
    payload = {
        'format_version': 2, 'model_state_dict': core.state_dict(),
        'model_config': dataclasses.asdict(core.cfg), 'seg_token_idx': core.seg_token_idx,
        'vocab_size': len(tokenizer), 'epoch': epoch, 'step': step,
        'args': vars(args) if args is not None else {},
        'best_score': best_score,
        'rng': {'torch': torch.get_rng_state(), 'python': random.getstate(),
                'numpy': {'kind': np.random.get_state()[0], 'keys': np.random.get_state()[1].tolist(),
                          'position': np.random.get_state()[2], 'has_gauss': np.random.get_state()[3],
                          'cached': np.random.get_state()[4]},
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    if rng_by_rank is not None:
        payload['rng_by_rank']=rng_by_rank
    for key, obj in [('optimizer', optimizer), ('scheduler', scheduler), ('scaler', scaler)]:
        if obj is not None:
            payload[key + '_state_dict'] = obj.state_dict()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = str(path) + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)
    processor.save_pretrained(os.path.join(os.path.dirname(path), 'processor'))


def restore_rng(checkpoint):
    ranks=checkpoint.get('rng_by_rank')
    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if ranks and torch.distributed.is_initialized() and len(ranks)!=torch.distributed.get_world_size():
        raise ValueError('Exact resume requires the same distributed world size')
    rng = ranks[rank] if ranks else checkpoint.get('rng')
    if not rng:
        return
    torch.set_rng_state(rng['torch'])
    random.setstate(rng['python'])
    n = rng['numpy']
    np.random.set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached']))
    if torch.cuda.is_available() and rng['cuda']:
        torch.cuda.set_rng_state_all(rng['cuda'])


def gather_rng_states():
    state=np.random.get_state()
    local={'torch':torch.get_rng_state(),'python':random.getstate(),
           'numpy':{'kind':state[0],'keys':state[1].tolist(),'position':state[2],
                    'has_gauss':state[3],'cached':state[4]},
           'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if torch.distributed.is_initialized():
        ranks=[None]*torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ranks,local)
        return ranks
    return [local]


def load_bundle(checkpoint_path, model_path=None, sam_path=None, attn_implementation=None):
    from model.qwen3vl_config import Qwen3VLSesameConfig
    from model.sesame_qwen3vl import build_sesame_qwen3vl, load_tokenizer_and_processor, add_seg_token
    checkpoint = read_checkpoint(checkpoint_path)
    cfg = Qwen3VLSesameConfig(**checkpoint['model_config'])
    if model_path:
        cfg.model_name_or_path = model_path
    if sam_path:
        cfg.vision_pretrained = sam_path
    if attn_implementation:
        cfg.attn_implementation = attn_implementation
    processor_path = os.path.join(os.path.dirname(checkpoint_path), 'processor')
    tokenizer, processor = load_tokenizer_and_processor(processor_path if os.path.isdir(processor_path) else cfg.model_name_or_path)
    marker = add_seg_token(tokenizer)
    if marker != checkpoint['seg_token_idx'] or len(tokenizer) != checkpoint['vocab_size']:
        raise ValueError('Checkpoint tokenizer vocabulary/SEG id does not match.')
    model = build_sesame_qwen3vl(cfg, marker, lora=cfg.use_lora, tokenizer=tokenizer)
    restore_state(model, checkpoint)
    if model.evidence.rl_active.item():
        model.start_rl_stage()
    return model, tokenizer, processor, checkpoint
