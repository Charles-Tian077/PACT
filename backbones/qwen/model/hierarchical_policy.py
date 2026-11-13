"""V3: a fixed SFT gate anchor and conditional, cost-aware mask utility learning.

The gate has its own small residual calibrator. The existing action MLP learns
counterfactual utility differences. Labels/rewards enter losses only; inference
uses the frozen anchor, observable features and predicted gains.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from .evidence_policy import EvidenceBandit, class_balanced_mean, global_mean


class HierarchicalEvidenceBandit(EvidenceBandit):
    def __init__(self, feature_dim, cfg):
        super().__init__(feature_dim, cfg)
        self.gate_adapter = nn.Linear(2 * feature_dim, 1)
        nn.init.zeros_(self.gate_adapter.weight)
        nn.init.zeros_(self.gate_adapter.bias)
        # SFT retains its original policy warm-up. Enable this only in a new
        # RL/control stage, after copying that policy into the fixed reference.
        self.gate_adapter.requires_grad_(False)
        if cfg.evidence_action_temperature <= 0 or cfg.evidence_risk_temperature <= 0:
            raise ValueError('Policy and risk temperatures must be positive')
        if cfg.evidence_gain_margin < 0 or cfg.evidence_value_weight <= 0:
            raise ValueError('Use a nonnegative gain margin and positive value supervision')
        if cfg.evidence_kl_beta < 0 or not all(0 <= value <= 1 for value in
                (cfg.evidence_fpr_target, cfg.evidence_fnr_target)):
            raise ValueError('Use nonnegative anchor KL and risk targets in [0, 1]')

    def logits(self, features, prior, available):
        if bool(self.rl_active.item()):
            raise RuntimeError('V3 has a binary gate and conditional gains; use select_actions, not flat argmax')
        return super().logits(features, prior, available)

    def start_rl(self):
        if not bool(self.rl_active.item()):
            self.reference.load_state_dict(self.policy.state_dict())
            # The reference preserves all SFT decisions. The value residual
            # begins at zero, so an untrained gain head cannot trigger repairs.
            nn.init.zeros_(self.policy.net[-1].weight)
            nn.init.zeros_(self.policy.net[-1].bias)
            nn.init.zeros_(self.gate_adapter.weight)
            nn.init.zeros_(self.gate_adapter.bias)
            self.kl_beta.fill_(self.cfg.evidence_kl_beta)
            self.lambda_fp.zero_()
            self.lambda_fn.zero_()
            self.rl_active.fill_(True)

    def enable_policy_gradients(self):
        super().enable_policy_gradients()
        self.gate_adapter.requires_grad_(True)

    def _decision(self, features, prior, available):
        features = features.detach().float()
        with torch.autocast(device_type=features.device.type, enabled=False):
            with torch.no_grad():
                anchor_scores = self.reference(features, prior.detach(), available)
                anchor_score, anchor_index = anchor_scores[:, 1:].max(-1)
                anchor_gate = anchor_score - anchor_scores[:, 0] - self.cfg.evidence_null_bias
            gate_features = torch.cat([features[:, 0], features[:, 1]], -1)
            gate = anchor_gate + self.gate_adapter(gate_features).squeeze(-1)
            raw = self.policy.net(features[:, 1:]).squeeze(-1)
            gains = raw - raw.gather(1, anchor_index[:, None])
            valid = available[:, 1:]
            masked_gains = gains.masked_fill(~valid, -1e4)
            log_conditional = (masked_gains / self.cfg.evidence_action_temperature).log_softmax(-1)
            conditional = log_conditional.exp()
            predicted_gain, candidate = masked_gains.max(-1)
            selected = torch.where(predicted_gain > self.cfg.evidence_gain_margin,
                                   candidate, anchor_index)
            # This is exactly the rule used at inference, including gate ties.
            chosen = torch.where(gate > 0, selected + 1, torch.zeros_like(selected))
            accept = gate.sigmoid()
            joint = torch.cat([(1 - accept)[:, None], accept[:, None] * conditional], -1)
        return {'gate': gate, 'anchor_gate': anchor_gate, 'anchor_index': anchor_index,
                'gains': gains, 'valid': valid, 'conditional': conditional,
                'log_conditional': log_conditional, 'selected': selected,
                'chosen': chosen, 'joint': joint, 'accept': accept}

    def select_actions(self, features, prior, available):
        if not bool(self.rl_active.item()):
            return super().select_actions(features, prior, available)
        return self._decision(features, prior, available)['chosen']

    def objective(self, features, prior, available, rewards, labels, normalizer=None):
        if not bool(self.rl_active.item()):
            return super().objective(features, prior, available, rewards, labels, normalizer)
        if normalizer is None:
            raise ValueError('V3 RL/control requires optimizer-window class denominators')
        state = self._decision(features, prior, available)
        gate, anchor_gate = state['gate'], state['anchor_gate']
        labels = labels.bool()
        rewards = rewards.detach().float()
        accept = state['accept']

        # Both are smooth upper bounds on the deterministic gate error:
        # gate>0 is a false accept for a negative; gate<=0 a false reject for
        # a positive. At the boundary the corresponding bound is exactly one.
        # The SAME surrogate is used in the primal loss and dual updates.
        temperature = self.cfg.evidence_risk_temperature
        risk_fp = F.softplus(gate / temperature) / math.log(2)
        risk_fn = F.softplus(-gate / temperature) / math.log(2)
        kl = (accept * (F.logsigmoid(gate) - F.logsigmoid(anchor_gate)) +
              (1 - accept) * (F.logsigmoid(-gate) - F.logsigmoid(-anchor_gate))).clamp_min(0)
        gate_loss = F.binary_cross_entropy_with_logits(gate, labels.float(), reduction='none')
        gate_loss = gate_loss + torch.where(labels, self.lambda_fn.detach() * risk_fn,
                                           self.lambda_fp.detach() * risk_fp)
        gate_loss = gate_loss + self.kl_beta.detach() * kl
        balanced_gate_loss = class_balanced_mean(gate_loss, labels, normalizer)

        # All non-null returns already include the decoder cost. Dense gain
        # supervision has no pi(a) multiplier, so a rare profitable action can
        # learn even when its current action probability is very small.
        nonnull = rewards[:, 1:]
        anchor_reward = nonnull.gather(1, state['anchor_index'][:, None])
        target_gains = nonnull - anchor_reward
        error = F.smooth_l1_loss(state['gains'], target_gains, reduction='none')
        valid = state['valid'].float()
        value_loss = (error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        if self.cfg.evidence_objective == 'expectation':
            decision_loss = -(state['conditional'] * target_gains).sum(-1)
        else:
            oracle_index = nonnull.masked_fill(~state['valid'], -1e4).argmax(-1)
            decision_loss = F.nll_loss(state['log_conditional'], oracle_index, reduction='none')
        geometry = decision_loss + self.cfg.evidence_value_weight * value_loss
        # Absence is learned by the gate. Constant rewards on absent objects
        # must not train the geometry branch to suppress every refinement.
        positive_geometry = global_mean(list(geometry[labels].unbind()), geometry.sum() * 0,
                                        normalizer, 'positive_count')
        loss = balanced_gate_loss + positive_geometry

        chosen, joint = state['chosen'], state['joint']
        oracle = rewards.masked_fill(~available, -1e4).max(-1).values
        selected_gain = target_gains.gather(1, state['selected'][:, None]).squeeze(1)
        predicted_selected = state['gains'].gather(1, state['selected'][:, None]).squeeze(1)
        best_gain = target_gains.masked_fill(~state['valid'], -1e4).max(-1).values
        presence_baseline = torch.where(1 - prior[:, 0].exp() >= self.cfg.presence_threshold, 1, 0)
        metrics = {
            'reward': (joint.detach() * rewards).sum(),
            'base_reward': rewards[:, 1].sum(),
            'presence_baseline_reward': rewards.gather(1, presence_baseline[:, None]).sum(),
            'oracle_reward': oracle.sum(),
            'chosen_reward': rewards.gather(1, chosen[:, None]).sum(),
            'regret': (oracle - rewards.gather(1, chosen[:, None]).squeeze(1)).sum(),
            'kl': kl.detach().sum(),
            'entropy': -(joint * joint.clamp_min(1e-12).log()).sum(-1).detach().sum(),
            'refine_probability': joint[:, 2:].detach().sum(),
            'hard_refine': (chosen > 1).float().sum(),
            'available_actions': available.float().sum(),
            'soft_fp': accept[~labels].detach().sum(),
            'soft_fn': (1 - accept[labels]).detach().sum(),
            'risk_fp_upper': risk_fp[~labels].detach().sum(),
            'risk_fn_upper': risk_fn[labels].detach().sum(),
            'hard_fp': (chosen[~labels] != 0).float().sum(),
            'hard_fn': (chosen[labels] == 0).float().sum(),
            'positive_oracle_gain': best_gain[labels].sum(),
            'positive_selected_gain': selected_gain[labels].sum(),
            'positive_predicted_gain': predicted_selected[labels].detach().sum(),
            'positive_gain_available': (best_gain[labels] > self.cfg.evidence_gain_margin).float().sum(),
            'positive_value_loss': value_loss[labels].detach().sum(),
            'negative_count': (~labels).sum().float(),
            'positive_count': labels.sum().float(),
            'count': labels.numel() * joint.new_ones(()),
        }
        return loss, metrics, joint

    @torch.no_grad()
    def update_constraints(self, metrics, update=True):
        if not bool(self.rl_active.item()):
            return super().update_constraints(metrics, update)
        result = {key: value.detach().clone() for key, value in metrics.items()}
        if dist.is_available() and dist.is_initialized():
            keys = list(result)
            packed = torch.stack([result[key] for key in keys])
            dist.all_reduce(packed)
            result = {key: packed[i] for i, key in enumerate(keys)}
        if update:
            for suffix, target in [('fp', self.cfg.evidence_fpr_target),
                                   ('fn', self.cfg.evidence_fnr_target)]:
                count = result['negative_count' if suffix == 'fp' else 'positive_count']
                if count > 0:
                    error = result['risk_' + suffix + '_upper'] / count - target
                    getattr(self, 'lambda_' + suffix).add_(self.cfg.evidence_dual_lr * error).clamp_(0, 10)
            # Keep both the SFT reference and beta fixed. An EMA reference can
            # track arbitrarily far from SFT while reporting a tiny local KL.
        return result
