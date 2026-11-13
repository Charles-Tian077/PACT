"""Loss denominators for one complete gradient-accumulation window."""
import torch
import torch.distributed as dist


def policy_window_normalizer(batches, device, include_quality=False):
    """Count actual expanded queries, not loader items or dataset task weights.

    No images are moved to the GPU here. All ranks enter one collective before
    their microbatch forwards; individual class losses need no further counts.
    """
    positive = negative = pairs = quality = 0
    for batch in batches:
        labels = batch['exists_list']
        positive += sum(value is not None and bool(value) for value in labels)
        negative += sum(value is not None and not bool(value) for value in labels)
        if include_quality:
            masks = batch['masks_list']
            ignores = batch.get('ignore_masks_list') or [None] * len(labels)
            for label, mask, ignore in zip(labels, masks, ignores):
                if label is not None and bool(label) and mask is not None:
                    target = mask > 0.5
                    if ignore is not None:
                        target = target & (ignore <= 0.5)
                    quality += int(bool(target.any()))
        ids = batch.get('pair_ids') or [None] * len(labels)
        kinds = batch.get('pair_kinds') or [None] * len(labels)
        for j in range(len(labels)):
            for k in range(j + 1, len(labels)):
                if labels[j] is None or labels[k] is None or ids[j] is None or ids[j] != ids[k]:
                    continue
                if labels[j] != labels[k] or kinds[j] == 'irrelevant_change':
                    pairs += 1
    counts = torch.tensor([positive, negative, pairs, quality], device=device, dtype=torch.float64)
    world = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts)
        world = dist.get_world_size()
    positive, negative, pairs, quality = counts.tolist()
    return {'positive_count': positive, 'negative_count': negative,
            'pair_count': pairs, 'quality_count': quality, 'gradient_scale': world * len(batches),
            'accumulation_steps': len(batches)}
