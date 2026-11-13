"""V4: action repairability and bounded *additional* mask degradation.

One shared action MLP predicts net return and two outcomes. Hard selection and
the training harm bound use the same candidate and switch margin. The bound is
a training surrogate, not a finite-sample/generalization certificate. Independent
policy-level certification is implemented in repair_risk.py.
"""
import math
from itertools import chain
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from .hierarchical_policy import HierarchicalEvidenceBandit
from .evidence_policy import class_balanced_mean, global_mean


class SelectiveRepairBandit(HierarchicalEvidenceBandit):
    def __init__(self, feature_dim, cfg):
        super().__init__(feature_dim, cfg)
        # Reuse exactly the existing hidden representation: 258 extra parameters
        # at hidden_dim=128. No second encoder, critic, teacher or text rollout.
        self.outcome_head = nn.Linear(cfg.evidence_hidden_dim, 2)
        nn.init.zeros_(self.outcome_head.weight)
        nn.init.zeros_(self.outcome_head.bias)
        self.outcome_head.requires_grad_(False)
        self.register_buffer('lambda_harm', torch.tensor(0.0))
        self.calibration_info = None  # runtime certificate; never inferred from weights
        for name in ('evidence_repairability_threshold', 'evidence_harm_probability_threshold',
                     'evidence_repair_iou_tolerance', 'evidence_repair_harm_target'):
            if not 0 <= getattr(cfg, name) <= 1:
                raise ValueError(name + ' must be in [0, 1]')
        if cfg.evidence_switch_temperature <= 0 or cfg.evidence_outcome_weight <= 0:
            raise ValueError('Use positive switch temperature and outcome supervision')
        if not math.isfinite(cfg.evidence_gate_offset):
            raise ValueError('Gate offset must be finite')

    def start_rl(self):
        starting = not bool(self.rl_active.item())
        super().start_rl()
        if starting:
            nn.init.zeros_(self.outcome_head.weight)
            nn.init.zeros_(self.outcome_head.bias)
            self.lambda_harm.zero_()

    def enable_policy_gradients(self):
        super().enable_policy_gradients()
        self.outcome_head.requires_grad_(True)

    def geometry_parameters(self):
        return chain(self.policy.parameters(), self.outcome_head.parameters())

    def _action_hidden(self, features, anchor_index=None, valid=None):
        return self.policy.net[:-1](features[:, 1:])

    def _gate_features(self, features):
        return torch.cat([features[:, 0], features[:, 1]], -1)

    def _decision(self, features, prior, available):
        features = features.detach().float()
        with torch.autocast(device_type=features.device.type, enabled=False):
            with torch.no_grad():
                anchor_scores = self.reference(features, prior.detach(), available)
                anchor_score, anchor_index = anchor_scores[:, 1:].max(-1)
                anchor_gate = anchor_score - (anchor_scores[:, 0] + self.cfg.evidence_null_bias)
            gate_features = self._gate_features(features)
            gate_score = anchor_gate + self.gate_adapter(gate_features).squeeze(-1)
            gate = gate_score - self.cfg.evidence_gate_offset
            hidden = self._action_hidden(features, anchor_index, available[:,1:])
            raw = self.policy.net[-1](hidden).squeeze(-1)
            gains = raw - raw.gather(1, anchor_index[:, None])
            outcome_logits = self.outcome_head(hidden)
            benefit, harm = outcome_logits.sigmoid().unbind(-1)
            valid = available[:, 1:]
            is_anchor = torch.zeros_like(valid).scatter(1, anchor_index[:, None], True)
            # An anchor is a fallback, not a newly beneficial intervention.
            benefit = benefit.masked_fill(is_anchor, 0)
            harm = harm.masked_fill(is_anchor, 0)
            eligible = (valid & ~is_anchor &
                        (benefit >= self.cfg.evidence_repairability_threshold) &
                        (harm <= self.cfg.evidence_harm_probability_threshold))
            scores = (gains - self.cfg.evidence_gain_margin).masked_fill(~eligible, -1e4)
            # Include the frozen SFT conditional action with exactly zero score.
            scores = scores.masked_fill(is_anchor, 0)
            candidate_score, candidate = scores.max(-1)
            switch = candidate_score > 0
            selected = torch.where(switch, candidate, anchor_index)
            chosen = torch.where(gate > 0, selected + 1, torch.zeros_like(selected))
            log_conditional = (scores / self.cfg.evidence_action_temperature).log_softmax(-1)
            conditional = log_conditional.exp()
            accept = gate.sigmoid()
            joint = torch.cat([(1 - accept)[:, None], accept[:, None] * conditional], -1)
            # For the harm upper bound also consider a negative-margin alternate.
            alt_scores = (gains - self.cfg.evidence_gain_margin).masked_fill(~eligible, -1e4)
            best_alt_score, best_alt = alt_scores.max(-1)
        return dict(gate=gate, gate_score=gate_score, anchor_gate=anchor_gate, anchor_index=anchor_index,
                    gains=gains, valid=valid, eligible=eligible, is_anchor=is_anchor,
                    outcome_logits=outcome_logits, benefit=benefit, harm=harm,
                    conditional=conditional, log_conditional=log_conditional,
                    selected=selected, chosen=chosen, joint=joint, accept=accept,
                    best_alt=best_alt, best_alt_score=best_alt_score,
                    has_alternative=eligible.any(-1), hidden=hidden)

    @torch.no_grad()
    def decision_metadata(self, features, prior, available):
        if not bool(self.rl_active.item()):
            raise ValueError('Outcome diagnostics require an initialized V4 policy')
        state = self._decision(features, prior, available)
        rows = []
        for i in range(len(features)):
            rows.append(dict(anchor_action=int(state['anchor_index'][i]) + 1,
                             anchor_gate=float(state['anchor_gate'][i]),
                             gate_score=float(state['gate_score'][i]),
                             gate=float(state['gate'][i]),
                             gate_residual=float(state['gate'][i] - state['anchor_gate'][i]
                                                 + self.cfg.evidence_gate_offset),
                             gains=state['gains'][i].cpu().tolist(),
                             benefit_probability=state['benefit'][i].cpu().tolist(),
                             harm_probability=state['harm'][i].cpu().tolist(),
                             available=available[i, 1:].cpu().tolist(),
                             eligible=state['eligible'][i].cpu().tolist(),
                             chosen_action=int(state['chosen'][i]),
                             repairability_state=('absent' if state['gate'][i]<=0 else
                                 'repairable' if (state['eligible'][i] &
                                     (state['gains'][i]>self.cfg.evidence_gain_margin)).any() else 'retain'),
                             certificate_applied=self.calibration_info is not None))
        return rows

    def objective(self, features, prior, available, rewards, labels, normalizer=None, outcomes=None):
        if not bool(self.rl_active.item()):
            return super().objective(features, prior, available, rewards, labels, normalizer)
        if normalizer is None or outcomes is None:
            raise ValueError('V4 needs optimizer-window counts and measured candidate IoUs')
        if self.calibration_info is not None:
            raise ValueError('A certified inference policy cannot be fine-tuned without invalidating its certificate')
        state = self._decision(features, prior, available)
        labels = labels.bool()
        rewards = rewards.detach().float()
        ious = outcomes['hard_iou'].detach().float()
        quality_valid = outcomes['iou_valid'].bool() & labels
        if ious.shape != rewards.shape or outcomes['iou_valid'].shape != labels.shape:
            raise ValueError('Candidate outcome/label dimensions do not match')
        gate, anchor_gate, accept = state['gate'], state['anchor_gate'], state['accept']
        risk_fp = F.softplus(gate / self.cfg.evidence_risk_temperature) / math.log(2)
        risk_fn = F.softplus(-gate / self.cfg.evidence_risk_temperature) / math.log(2)
        kl = (accept * (F.logsigmoid(gate) - F.logsigmoid(anchor_gate)) +
              (1 - accept) * (F.logsigmoid(-gate) - F.logsigmoid(-anchor_gate))).clamp_min(0)
        gate_loss = F.binary_cross_entropy_with_logits(gate, labels.float(), reduction='none')
        gate_loss += torch.where(labels, self.lambda_fn.detach() * risk_fn,
                                 self.lambda_fp.detach() * risk_fp)
        gate_loss += self.kl_beta.detach() * kl

        nonnull = rewards[:, 1:]
        anchor_reward = nonnull.gather(1, state['anchor_index'][:, None])
        target_gains = nonnull - anchor_reward
        anchor_iou = ious[:, 1:].gather(1, state['anchor_index'][:, None])
        iou_gains = ious[:, 1:] - anchor_iou
        harmful = iou_gains < -self.cfg.evidence_repair_iou_tolerance
        profitable = (target_gains > self.cfg.evidence_gain_margin) & ~harmful
        valid_alt = state['valid'] & ~state['is_anchor']
        count_alt = valid_alt.float().sum(-1).clamp_min(1)
        outcome_target = torch.stack([profitable, harmful], -1).float()
        outcome_error = F.binary_cross_entropy_with_logits(state['outcome_logits'], outcome_target,
                                                          reduction='none').mean(-1)
        outcome_loss = (outcome_error * valid_alt).sum(-1) / count_alt
        valid = state['valid'].float()
        value_error = F.smooth_l1_loss(state['gains'], target_gains, reduction='none')
        value_loss = (value_error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        if self.cfg.evidence_objective == 'expectation':
            decision_loss = -(state['conditional'] * target_gains).sum(-1)
        else:
            admissible = state['eligible'] | state['is_anchor']
            oracle_index = target_gains.masked_fill(~admissible, -1e4).argmax(-1)
            decision_loss = F.nll_loss(state['log_conditional'], oracle_index, reduction='none')

        # For the SAME best eligible alternate as the hard decision:
        # I[switch & harmful] <= I[harmful(best_alt)] softplus(margin/T)/log(2).
        # Dropping the gate's acceptance event only makes this upper bound more
        # conservative. Control is per valid-positive query, not per repair.
        alt_harm = harmful.gather(1, state['best_alt'][:, None]).squeeze(1)
        risk_harm = (F.softplus(state['best_alt_score'] / self.cfg.evidence_switch_temperature)
                     / math.log(2) * alt_harm * state['has_alternative'] * quality_valid)
        geometry = decision_loss + self.cfg.evidence_value_weight * value_loss
        loss = class_balanced_mean(gate_loss, labels, normalizer)
        loss += global_mean(list(geometry[labels].unbind()), geometry.sum() * 0,
                            normalizer, 'positive_count')
        extra = self.cfg.evidence_outcome_weight * outcome_loss + self.lambda_harm.detach() * risk_harm
        loss += global_mean(list(extra[quality_valid].unbind()), extra.sum() * 0,
                            normalizer, 'quality_count')

        chosen = state['chosen']
        changed = (chosen > 0) & (state['selected'] != state['anchor_index'])
        selected_gain = target_gains.gather(1, state['selected'][:, None]).squeeze(1)
        selected_iou_gain = iou_gains.gather(1, state['selected'][:, None]).squeeze(1)
        selected_harm = harmful.gather(1, state['selected'][:, None]).squeeze(1)
        # Repairability is relative to the current finite action set and anchor.
        recoverable = (profitable & valid_alt).any(-1) & quality_valid
        predicted_recoverable = (state['eligible'] &
                                (state['gains']>self.cfg.evidence_gain_margin)).any(-1) & (gate > 0)
        executed_harm = changed & selected_harm & quality_valid
        oracle = rewards.masked_fill(~available, -1e4).max(-1).values
        presence_baseline = torch.where(1 - prior[:, 0].exp() >= self.cfg.presence_threshold, 1, 0)
        metrics = dict(
            reward=(state['joint'].detach() * rewards).sum(), base_reward=rewards[:, 1].sum(),
            presence_baseline_reward=rewards.gather(1, presence_baseline[:, None]).sum(),
            oracle_reward=oracle.sum(), chosen_reward=rewards.gather(1, chosen[:, None]).sum(),
            regret=(oracle - rewards.gather(1, chosen[:, None]).squeeze(1)).sum(),
            kl=kl.detach().sum(),
            entropy=-(state['joint'] * state['joint'].clamp_min(1e-12).log()).sum(-1).detach().sum(),
            refine_probability=state['joint'][:, 2:].detach().sum(),
            hard_refine=(chosen > 1).float().sum(), available_actions=available.float().sum(),
            soft_fp=accept[~labels].detach().sum(), soft_fn=(1 - accept[labels]).detach().sum(),
            risk_fp_upper=risk_fp[~labels].detach().sum(), risk_fn_upper=risk_fn[labels].detach().sum(),
            hard_fp=(chosen[~labels] != 0).float().sum(), hard_fn=(chosen[labels] == 0).float().sum(),
            positive_oracle_gain=target_gains.masked_fill(~state['valid'], -1e4).max(-1).values[labels].sum(),
            positive_selected_gain=selected_gain[labels].sum(),
            positive_predicted_gain=state['gains'].gather(1, state['selected'][:, None])[labels].detach().sum(),
            positive_value_loss=value_loss[labels].detach().sum(),
            positive_outcome_loss=outcome_loss[quality_valid].detach().sum(),
            positive_gain_available=recoverable.float().sum(),
            positive_repair_opportunity=recoverable.float().sum(),
            positive_opportunity_taken=(recoverable & changed & (selected_gain > 0) & ~selected_harm).float().sum(),
            positive_changed=(changed & quality_valid).float().sum(),
            positive_changed_harm=executed_harm.float().sum(),
            positive_changed_iou_gain=(selected_iou_gain * changed * quality_valid).sum(),
            positive_changed_net_gain=(selected_gain * changed * quality_valid).sum(),
            positive_harm_magnitude=((-selected_iou_gain).clamp_min(0) * executed_harm).sum(),
            positive_harm_upper=risk_harm.detach().sum(),
            positive_iou_valid=quality_valid.float().sum(),
            positive_repairability_tp=(predicted_recoverable & recoverable).float().sum(),
            positive_repairability_fp=(predicted_recoverable & ~recoverable & quality_valid).float().sum(),
            positive_repairability_fn=(~predicted_recoverable & recoverable).float().sum(),
            negative_count=(~labels).sum().float(), positive_count=labels.sum().float(),
            count=labels.numel() * state['joint'].new_ones(()))
        return loss, metrics, state['joint']

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
            for suffix, target, numerator, denominator in (
                ('fp', self.cfg.evidence_fpr_target, 'risk_fp_upper', 'negative_count'),
                ('fn', self.cfg.evidence_fnr_target, 'risk_fn_upper', 'positive_count'),
                ('harm', self.cfg.evidence_repair_harm_target, 'positive_harm_upper', 'positive_iou_valid')):
                if result[denominator] > 0:
                    error = result[numerator] / result[denominator] - target
                    getattr(self, 'lambda_' + suffix).add_(self.cfg.evidence_dual_lr * error).clamp_(0, 10)
        return result
