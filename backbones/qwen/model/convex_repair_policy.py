"""V5: convex, full-information fitting of a selective-repair policy.

The SFT representation and reference are fixed. Only affine gate, value and
outcome heads move. Hard eligibility is used at deployment, NEVER to mask the
training target. The hard-policy harm majorizer sums all harmful alternatives.
The bounded-dual saddle problem is solved by projected extragradient, not Adam.
"""
import math
import torch
from torch.nn import functional as F
from .selective_repair_policy import SelectiveRepairBandit
from .evidence_policy import class_balanced_mean, global_mean


class ConvexRepairBandit(SelectiveRepairBandit):
    def __init__(self, feature_dim, cfg):
        super().__init__(feature_dim, cfg)
        self._window_cache = None

    def enable_policy_gradients(self):
        super().enable_policy_gradients()
        # The old SFT action representation is retained, rather than randomly
        # replacing it. Its fixed normalization bounds head feature norms.
        self.policy.net[:-1].requires_grad_(False)
        self.policy.net[-1].bias.requires_grad_(False)  # cancels in every gain

    def _action_hidden(self, features, anchor_index=None, valid=None):
        if anchor_index is None:
            raise ValueError('V5 action conditioning requires the frozen SFT anchor')
        original=F.normalize(super()._action_hidden(features), dim=-1, eps=1e-6)
        if self.cfg.evidence_solver_revision==1:
            return original
        anchor=original.gather(1,anchor_index[:,None,None].expand(-1,1,original.shape[-1]))
        contrast=original-anchor
        norms=contrast.norm(dim=-1)
        if valid is not None:
            norms=norms.masked_fill(~valid,0)
        scale=norms.max(-1,keepdim=True).values.unsqueeze(-1).clamp_min(1e-6)
        # Keep the anchor's context, and condition only relative action effects.
        # Each available contrast has norm <=1, so hidden norm <=2. Exact
        # duplicate actions remain duplicates; no signal is invented for them.
        return anchor+contrast/scale

    def outcome_smoothness_upper_bound(self):
        # Mean BCE over two output events: Hessian <= (||h||^2+1)/8.
        # Available conditioned features satisfy ||h||<=2. Include the ridge.
        norm_bound=2. if self.cfg.evidence_solver_revision>=2 else 1.
        return self.cfg.evidence_outcome_weight*(norm_bound**2+1)/8+self.cfg.evidence_ridge

    @staticmethod
    def outcome_candidate_weights(target, alternate):
        """Per query/event, each observed label class gets equal total mass.

        Targets are detached measured outcomes. Weights are fixed in theta,
        so convexity and the same two-event BCE curvature bound are retained.
        Empty classes do not dilute the only observed class; an anchor never
        receives outcome supervision. Positive queries still use the original
        full-window quality denominator outside this per-query candidate loss.
        """
        target=target.detach()
        mask=alternate.unsqueeze(-1)
        positive=(target*mask).sum(1,keepdim=True)
        negative=((1-target)*mask).sum(1,keepdim=True)
        classes=((positive>0).float()+(negative>0).float()).clamp_min(1)
        return mask*torch.where(target>0,positive.clamp_min(1).reciprocal(),
            negative.clamp_min(1).reciprocal())/classes

    def fitting_block_smoothness_bounds(self):
        """Fixed fitting-loss bounds; risk curvature is checked by line search.

        Gate augmented feature norm^2 <=2. Pair difference norm^2 <=4,
        hence invariant-square curvature <=8 (ordered softplus <=1).
        Relative action features have norm <=1: softmax covariance <=1 and
        SmoothL1 curvature <=1. These are metric choices, not new loss weights.
        """
        cfg=self.cfg
        return dict(gate=(1+cfg.evidence_kl_beta)/2+8*cfg.evidence_pair_weight+cfg.evidence_ridge,
            value=1/cfg.evidence_action_temperature+cfg.evidence_value_weight+cfg.evidence_ridge,
            outcome=self.outcome_smoothness_upper_bound())

    def _gate_features(self, features):
        return F.normalize(super()._gate_features(features), dim=-1, eps=1e-6)

    def begin_window(self):
        if self._window_cache is not None:
            raise RuntimeError('A policy optimization window is already active')
        self._window_cache = []

    def end_window(self):
        if self._window_cache is None:
            raise RuntimeError('No policy optimization window is active')
        cached, self._window_cache = self._window_cache, None
        return cached

    def objective(self, features, prior, available, rewards, labels,
                  normalizer=None, outcomes=None, pair_relations=()):
        if not bool(self.rl_active.item()):
            raise ValueError('V5 is an explicit new policy stage from a saved SFT')
        if normalizer is None or outcomes is None:
            raise ValueError('V5 requires complete optimizer-window counts and candidate IoUs')
        if self.calibration_info is not None:
            raise ValueError('A certified policy is inference-only')
        if self._window_cache is not None:
            # No image features, masks, labels or outcomes are recomputed during
            # predictor/corrector trials. GT is a training target only.
            self._window_cache.append(dict(
                features=features.detach(), prior=prior.detach(),
                available=available.detach(), rewards=rewards.detach(),
                labels=labels.detach(), normalizer=dict(normalizer),
                outcomes={k:v.detach() for k,v in outcomes.items()},
                pair_relations=tuple(pair_relations)))
        state = self._decision(features, prior, available)
        labels = labels.bool()
        quality = outcomes['iou_valid'].bool() & labels
        rewards = rewards.detach().float()
        ious = outcomes['hard_iou'].detach().float()
        if ious.shape != rewards.shape or quality.shape != labels.shape:
            raise ValueError('Candidate outcome dimensions do not match')
        gate, gains = state['gate'], state['gains']
        valid = state['valid']
        if not bool(valid.any(-1).all()):
            raise ValueError('Each query must retain an available SFT conditional anchor')
        if not bool(valid.gather(1, state['anchor_index'][:,None]).all()):
            raise ValueError('The SFT conditional anchor must remain available')
        anchor_reward = rewards[:,1:].gather(1, state['anchor_index'][:,None])
        delta = rewards[:,1:] - anchor_reward
        anchor_iou = ious[:,1:].gather(1, state['anchor_index'][:,None])
        delta_iou = ious[:,1:] - anchor_iou
        alternate = valid & ~state['is_anchor']
        harmful = delta_iou < -self.cfg.evidence_repair_iou_tolerance
        profitable = (delta > self.cfg.evidence_gain_margin) & ~harmful
        differences=state['hidden']-state['hidden'].gather(1,
            state['anchor_index'][:,None,None].expand(-1,1,state['hidden'].shape[-1]))
        contrast_norm=differences.norm(dim=-1)
        # A fixed, parameter-independent capacity screen: such an action can
        # NEVER exceed the strict switching margin anywhere in the primal ball.
        # Omitting it from the upper bound is valid and avoids penalizing exact
        # duplicates which always tie the retained anchor and cannot execute.
        switchable=alternate & (self.cfg.evidence_primal_radius*contrast_norm >
                               self.cfg.evidence_gain_margin)
        if self.cfg.evidence_solver_revision==1:
            switchable=alternate
        tau = self.cfg.evidence_action_temperature
        # Uniform full-support reference on AVAILABLE actions. The exact
        # entropy-regularized reward optimizer is softmax(delta / tau).
        dense_logp = (gains / tau).masked_fill(~valid, -1e4).log_softmax(-1)
        target_logp = (delta / tau).masked_fill(~valid, -1e4).log_softmax(-1)
        target = target_logp.exp().detach()
        if self.cfg.evidence_objective == 'expectation':
            # tau KL(target || fitted policy): a convex fitting loss in affine
            # logits. This is reward-derived soft policy iteration, not PPO.
            decision_loss = tau * (target * (target_logp - dense_logp)).sum(-1)
        else:
            oracle = delta.masked_fill(~valid, -1e4).argmax(-1)
            decision_loss = tau * F.nll_loss(dense_logp, oracle, reduction='none')
        value_loss = (F.smooth_l1_loss(gains, delta, reduction='none') * valid).sum(-1)
        value_loss = value_loss / valid.sum(-1).clamp_min(1)
        target_outcome = torch.stack([profitable, harmful], -1).float()
        outcome_error = F.binary_cross_entropy_with_logits(
            state['outcome_logits'], target_outcome, reduction='none')
        if self.cfg.evidence_solver_revision>=3:
            outcome_loss=(outcome_error*self.outcome_candidate_weights(target_outcome,alternate)).sum(1).mean(-1)
        else:
            outcome_loss=outcome_error.mean(-1)
            outcome_loss=(outcome_loss*alternate).sum(-1)/alternate.sum(-1).clamp_min(1)

        risk_fp = F.softplus(gate / self.cfg.evidence_risk_temperature) / math.log(2)
        risk_fn = F.softplus(-gate / self.cfg.evidence_risk_temperature) / math.log(2)
        # Any executed harmful switch has at least ONE harmful alternate with
        # gain-margin>0. Its term exceeds one; summing is a convex upper bound
        # regardless of the learned eligibility screen or argmax identity.
        risk_harm = (F.softplus((gains-self.cfg.evidence_gain_margin) /
                     self.cfg.evidence_switch_temperature) / math.log(2)
                     * switchable * harmful).sum(-1)
        # Cauchy-Schwarz: every feasible value head has
        # gain_a >= -B*||h_a-h_anchor||. This is a lower bound on how small
        # the convex proxy can be, not on real deployment harm.
        minimum_gain=-self.cfg.evidence_primal_radius*contrast_norm
        harm_proxy_floor=(F.softplus((minimum_gain-self.cfg.evidence_gain_margin)/
            self.cfg.evidence_switch_temperature)/math.log(2)*switchable*harmful).sum(-1)
        anchor_accept = state['anchor_gate'].sigmoid().detach()
        # Direction matters: KL(fixed Bernoulli || current Bernoulli) is convex
        # in the current affine logit; the reversed KL need not be convex.
        anchor_kl = (F.binary_cross_entropy_with_logits(gate, anchor_accept, reduction='none')
            - F.binary_cross_entropy_with_logits(state['anchor_gate'], anchor_accept,
                                                 reduction='none').detach())
        gate_fit = F.binary_cross_entropy_with_logits(gate, labels.float(), reduction='none')
        gate_fit = gate_fit + self.kl_beta.detach() * anchor_kl
        zero = gate.sum()*0 + gains.sum()*0 + state['outcome_logits'].sum()*0
        loss = class_balanced_mean(gate_fit, labels, normalizer)

        def mean(values, mask, count_key):
            return global_mean(list(values[mask].unbind()), zero, normalizer, count_key)

        geometry = decision_loss + self.cfg.evidence_value_weight*value_loss
        geometry += self.cfg.evidence_outcome_weight*outcome_loss
        loss += mean(geometry, quality, 'quality_count')
        # Constraint coefficients are not divided by the number of classes.
        # They are the SAME class-conditional means used by the dual operator.
        loss += self.lambda_fp.detach()*mean(risk_fp, ~labels, 'negative_count')
        loss += self.lambda_fn.detach()*mean(risk_fn, labels, 'positive_count')
        loss += self.lambda_harm.detach()*mean(risk_harm, quality, 'quality_count')
        pair_losses = []
        for j,k,kind in pair_relations:
            if kind == 'invariant':
                pair_losses.append((gate[j]-gate[k]).square())
            elif kind == 'ordered':  # j=negative, k=positive
                pair_losses.append(F.softplus(self.cfg.evidence_pair_margin+gate[j]-gate[k]))
            else:
                raise ValueError('Unknown convex pair relation')
        loss += self.cfg.evidence_pair_weight*global_mean(pair_losses, zero, normalizer)

        # Reuse V4's HARD deployment diagnostics, without differentiating its
        # screened soft policy. Replace the changed mathematical quantities.
        with torch.no_grad():
            _, metrics, _ = super().objective(features, prior, available, rewards,
                                              labels, normalizer, outcomes)
            metrics['kl'] = anchor_kl.detach().sum()
            metrics['positive_harm_upper'] = risk_harm[quality].detach().sum()
            metrics['positive_harm_proxy_floor'] = harm_proxy_floor[quality].detach().sum()
            metrics['positive_unselectable_gain_opportunity'] = (
                (profitable & alternate & ~switchable).any(-1) & quality).float().sum()
            metrics['positive_value_loss'] = value_loss[quality].detach().sum()
            metrics['positive_outcome_loss'] = outcome_loss[quality].detach().sum()
            metrics['positive_dense_target_kl'] = (
                target*(target_logp-dense_logp)).sum(-1)[quality].detach().sum()
            # Exact gap identity is for the dense stochastic entropy subproblem,
            # NOT the screened hard deployment policy.
            dense = dense_logp.exp()
            metrics['positive_dense_entropy_gap'] = (
                tau*(dense*(dense_logp-target_logp)).sum(-1))[quality].detach().sum()
        return loss, metrics, state['joint']

    @staticmethod
    def dual_residuals(metrics, cfg, template):
        residuals = []
        for numerator,denominator,target in (
            ('risk_fp_upper','negative_count',cfg.evidence_fpr_target),
            ('risk_fn_upper','positive_count',cfg.evidence_fnr_target),
            ('positive_harm_upper','positive_iou_valid',cfg.evidence_repair_harm_target)):
            count = float(metrics.get(denominator, 0))
            residuals.append(template.new_tensor(0.) if count == 0 else
                             metrics[numerator].detach()/count-target)
        return torch.stack(residuals)

    def update_constraints(self, metrics, update=True):
        if update:
            raise RuntimeError('V5 duals must be updated atomically by ProjectedMirrorProx')
        return super().update_constraints(metrics, update=False)
