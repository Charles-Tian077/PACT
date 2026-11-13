"""Finite-action visual verification with exact, KL-regularized bandit updates.

Action 0 rejects the target; action 1 accepts the current mask; remaining actions
request one decoder refinement. Rewards are enumerated during training, never
available at inference. This is a contextual bandit, not language-token GRPO.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist


def class_balanced_mean(values, labels, normalizer=None):
    """Global class means with correct DDP gradient scaling and missing classes."""
    if normalizer is not None:
        # The caller still divides every microbatch loss by accumulation_steps.
        # DDP subsequently averages ranks. Undo both averages here so the summed
        # gradients are the class means of the entire optimizer-step window.
        scale = normalizer['gradient_scale']
        counts = [normalizer['negative_count'], normalizer['positive_count']]
        classes = sum(count > 0 for count in counts)
        total = values.sum() * 0.0
        for cls, count in zip((False, True), counts):
            if count > 0:
                total = total + values[labels == cls].sum() * scale / count
        return total / max(classes, 1)
    total = values.sum() * 0.0
    classes = 0
    counts=torch.stack([(~labels).sum(),labels.sum()]).to(values.dtype).detach()
    scale=1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts);scale=dist.get_world_size()
    for index,cls in enumerate((False, True)):
        chosen = labels == cls
        count=counts[index]
        if count.item() > 0:
            total = total + values[chosen].sum() * scale / count
            classes += 1
    return total / max(classes, 1)


def global_mean(values, zero, normalizer=None, count_key='pair_count'):
    """All ranks participate, including ranks with no masks in a VQA batch."""
    if normalizer is not None:
        numerator = torch.stack(values).sum() if values else zero
        return numerator * normalizer['gradient_scale'] / max(normalizer[count_key], 1)
    count=zero.detach().new_tensor(len(values))
    world=1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count);world=dist.get_world_size()
    numerator=torch.stack(values).sum() if values else zero
    return numerator*world/count.clamp_min(1)


class EvidencePolicy(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feature_dim),
                                 nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, prior, available):
        with torch.autocast(device_type=features.device.type, enabled=False):
            scores = self.net(features.float()).squeeze(-1) + prior.float()
            return scores.masked_fill(~available, -1e4)


class EvidenceBandit(nn.Module):
    def __init__(self, feature_dim, cfg):
        super().__init__()
        self.policy = EvidencePolicy(feature_dim, cfg.evidence_hidden_dim)
        self.reference = copy.deepcopy(self.policy).requires_grad_(False)
        self.register_buffer('rl_active', torch.tensor(False))
        self.register_buffer('kl_beta', torch.tensor(float(cfg.evidence_kl_beta)))
        self.register_buffer('lambda_fp', torch.tensor(0.0))
        self.register_buffer('lambda_fn', torch.tensor(0.0))
        self.cfg = cfg

    def start_rl(self):
        if not bool(self.rl_active.item()):
            self.reference.load_state_dict(self.policy.state_dict())
            self.rl_active.fill_(True)

    def enable_policy_gradients(self):
        self.policy.requires_grad_(True)

    def select_actions(self, features, prior, available):
        scores = self.logits(features, prior, available)
        scores = scores.clone()
        scores[:, 0] += self.cfg.evidence_null_bias
        return scores.argmax(-1)

    def logits(self, features, prior, available):
        return self.policy(features.detach(), prior.detach(), available)

    def objective(self, features, prior, available, rewards, labels, normalizer=None):
        logits = self.logits(features, prior, available)
        log_probs = logits.log_softmax(-1)
        probs = log_probs.exp()
        rewards = rewards.detach().float()
        labels = labels.bool()
        reject = probs[:, 0]
        penalties = torch.zeros_like(rewards)
        penalties[:, 0] = self.lambda_fn.detach() * labels.float()
        penalties[:, 1:] = self.lambda_fp.detach() * (~labels).float().unsqueeze(1)
        utilities = rewards - penalties
        # An action-independent baseline preserves the exact expected gradient.
        advantages = utilities - utilities[:, 1:2]
        with torch.no_grad():
            reference_log_probs = self.reference(features.detach(), prior.detach(), available).log_softmax(-1)
        kl = (probs * (log_probs - reference_log_probs)).sum(-1).clamp_min(0)
        if bool(self.rl_active.item()) and self.cfg.evidence_objective=='expectation':
            per_example = -(probs * advantages).sum(-1) + self.kl_beta.detach() * kl
        elif bool(self.rl_active.item()):
            oracle=utilities.masked_fill(~available,-1e4).argmax(-1)
            per_example=F.cross_entropy(logits,oracle,reduction='none') + self.kl_beta.detach()*kl
        else:
            # Soft full-information imitation is the policy's cold start.
            target = (utilities / 0.2).masked_fill(~available, -1e4).softmax(-1)
            per_example = -(target * log_probs).sum(-1)
        loss = class_balanced_mean(per_example, labels, normalizer)
        expected = (probs * rewards).sum(-1)
        chosen = probs.argmax(-1)
        oracle = rewards.masked_fill(~available, -1e4).max(-1).values
        presence_baseline=torch.where(1-prior[:,0].exp()>=self.cfg.presence_threshold,1,0)
        metrics = {
            'reward': expected.detach().sum(),
            'base_reward': rewards[:, 1].sum(),
            'presence_baseline_reward':rewards.gather(1,presence_baseline[:,None]).sum(),
            'oracle_reward': oracle.sum(),
            'chosen_reward': rewards.gather(1, chosen[:, None]).sum(),
            'regret': (oracle - rewards.gather(1, chosen[:, None]).squeeze(1)).sum(),
            'kl': kl.detach().sum(),
            'entropy': -(probs * log_probs).sum(-1).detach().sum(),
            'refine_probability': probs[:, 2:].sum().detach(),
            'available_actions':available.float().sum(),
            'soft_fp': (1 - reject[~labels]).detach().sum(),
            'soft_fn': reject[labels].detach().sum(),
            'hard_fp': (chosen[~labels] != 0).float().sum(),
            'hard_fn': (chosen[labels] == 0).float().sum(),
            'negative_count': (~labels).sum().float(),
            'positive_count': labels.sum().float(),
            'count': labels.numel() * probs.new_ones(()),
        }
        return loss, metrics, probs

    @torch.no_grad()
    def update_constraints(self, metrics, update=True):
        """Call once per optimizer step, after aggregating all microbatches/ranks."""
        m = {key: value.detach().clone() for key, value in metrics.items()}
        if dist.is_available() and dist.is_initialized():
            keys=list(m);packed=torch.stack([m[key] for key in keys])
            dist.all_reduce(packed)
            m={key:packed[i] for i,key in enumerate(keys)}
        if bool(self.rl_active.item()) and update:
            lr = self.cfg.evidence_dual_lr
            if m['negative_count'] > 0:
                error = m['soft_fp'] / m['negative_count'] - self.cfg.evidence_fpr_target
                self.lambda_fp.add_(lr * error).clamp_(0, 10)
            if m['positive_count'] > 0:
                error = m['soft_fn'] / m['positive_count'] - self.cfg.evidence_fnr_target
                self.lambda_fn.add_(lr * error).clamp_(0, 10)
            kl = m['kl'] / m['count'].clamp_min(1)
            target = self.cfg.evidence_kl_target
            if kl > 1.5 * target:
                self.kl_beta.mul_(1.2)
            elif kl < target / 1.5:
                self.kl_beta.div_(1.2)
            self.kl_beta.clamp_(1e-4, 10)
            # A permanently fixed weak warm-start reference can prevent useful
            # learning. A lagged EMA reference controls local policy drift.
            decay = self.cfg.evidence_reference_decay
            for reference, current in zip(self.reference.parameters(), self.policy.parameters()):
                reference.lerp_(current.detach(), 1-decay)
        return m


def mask_quality(logits, target, valid):
    """Continuous IoU plus boundary F-score; excludes padding and ignore pixels."""
    pred = logits.sigmoid() * valid
    target = target.float() * valid
    intersection = (pred * target).sum((-2, -1))
    union = (pred + target - pred * target).sum((-2, -1)).clamp_min(1e-6)
    iou = intersection / union
    binary = (pred >= 0.5).float() * valid
    def boundary(x):
        eroded = -F.max_pool2d(-x[:, None], 3, 1, 1)[:, 0]
        return (x - eroded).clamp(0, 1) * valid
    pb, tb = boundary(binary), boundary(target.expand_as(binary))
    # Do not reward boundary matches on or immediately next to ignored pixels.
    valid_inner = -F.max_pool2d(-valid[None, None], 3, 1, 1)[0, 0]
    pb, tb = pb * valid_inner, tb * valid_inner
    pb_near = F.max_pool2d(pb[:, None], 3, 1, 1)[:, 0]
    tb_near = F.max_pool2d(tb[:, None], 3, 1, 1)[:, 0]
    precision = (pb * tb_near).sum((-2, -1)) / pb.sum((-2, -1)).clamp_min(1)
    recall = (tb * pb_near).sum((-2, -1)) / tb.sum((-2, -1)).clamp_min(1)
    boundary_f = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    return (0.8 * iou + 0.2 * boundary_f).clamp(0, 1)
