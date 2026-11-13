"""Projected extragradient on a compact convex policy-head saddle problem.

Every trial uses the SAME detached optimizer-window data. No new SAM calls or
neural-network rollouts occur inside this solver. Local line-search diagnostics
are NOT a global saddle gap or a population-risk certificate.
"""
import math
import torch


class ProjectedMirrorProx(torch.optim.Optimizer):
    def __init__(self, params, policy, lr, dual_lr, radius, ridge,
                 dual_cap=10., contraction=.5, acceptance=.9, max_backtracks=24):
        params = list(params)
        expected={id(policy.policy.net[-1].weight)} | {id(p) for p in policy.gate_adapter.parameters()}
        expected |= {id(p) for p in policy.outcome_head.parameters()}
        if {id(p) for p in params}!=expected or len(params)!=len(expected):
            raise ValueError('Convex solver requires exactly the five affine-head tensors')
        if any(p.requires_grad for p in policy.policy.net[:-1].parameters()) or any(
                p.requires_grad for p in policy.reference.parameters()):
            raise ValueError('Convex solver requires frozen SFT hidden/reference features')
        if not params or min(lr, dual_lr, radius, ridge, dual_cap) <= 0:
            raise ValueError('Use nonempty heads and positive solver parameters')
        if not 0 < contraction < 1 or not 0 < acceptance < 1 or max_backtracks < 1:
            raise ValueError('Invalid extragradient backtracking settings')
        super().__init__(params, dict(lr=lr, dual_ratio=dual_lr/lr, radius=radius,
            ridge=ridge, dual_cap=dual_cap, contraction=contraction,
            acceptance=acceptance, max_backtracks=max_backtracks, scale=1.,
            accepted_steps=0, total_probe_evaluations=0))
        self.policy = policy
        outcome_ids={id(p) for p in policy.outcome_head.parameters()}
        # Risk/value curvature can require a small step without requiring the
        # independent logistic outcome block to take that same small step.
        # The fixed metric comes from a proved BCE Hessian upper bound; it does
        # not change loss weights, labels, thresholds, domain or reward.
        outcome_factor=(1./(policy.outcome_smoothness_upper_bound()*lr)
                        if policy.cfg.evidence_solver_revision>=2 else 1.)
        self.param_groups[0]['primal_block_factors']=[outcome_factor if id(p) in outcome_ids else 1.
                                                    for p in params]
        self.param_groups[0]['outcome_block_factor']=outcome_factor
        if policy.cfg.evidence_solver_revision>=3:
            bounds=policy.fitting_block_smoothness_bounds()
            gate_ids={id(p) for p in policy.gate_adapter.parameters()}
            self.param_groups[0]['primal_block_factors']=[
                1/(lr*bounds['outcome' if id(p) in outcome_ids else 'gate' if id(p) in gate_ids else 'value'])
                for p in params]
            # Coordinate scaling r_j / b_j, mu_j=b_j*lambda_j with a
            # metric d_mu=dual_lr*b_j gives d_lambda=dual_lr/b_j.
            # Lagrangian lambda*r and the ORIGINAL lambda box are unchanged.
            budgets=[policy.cfg.evidence_fpr_target,policy.cfg.evidence_fnr_target,
                     policy.cfg.evidence_repair_harm_target]
            self.param_groups[0]['dual_block_factors']=[1/b if b>0 else 1. for b in budgets]
            self.param_groups[0]['fitting_block_bounds']=bounds
        if any(p.dtype != torch.float32 for p in params):
            raise ValueError('The small policy solver requires float32 parameters')
        if not all(math.isfinite(float(x)) for x in (lr,dual_lr,radius,ridge,dual_cap)):
            raise ValueError('Solver parameters must be finite')

    def _flat(self):
        return torch.cat([p.detach().reshape(-1) for p in self.param_groups[0]['params']])

    @torch.no_grad()
    def _put(self, flat):
        offset = 0
        for p in self.param_groups[0]['params']:
            p.copy_(flat[offset:offset+p.numel()].view_as(p)); offset += p.numel()
        if offset != flat.numel():
            raise ValueError('Wrong primal vector size')

    def _dual(self):
        return torch.stack([self.policy.lambda_fp, self.policy.lambda_fn, self.policy.lambda_harm]).clone()

    @torch.no_grad()
    def _put_dual(self, values):
        for name,value in zip(('lambda_fp','lambda_fn','lambda_harm'), values):
            getattr(self.policy,name).copy_(value)

    def _project(self, vector, diagonal):
        radius = self.param_groups[0]['radius']
        factors=self.param_groups[0].get('primal_block_factors',[1.])
        if all(value==factors[0] for value in factors):
            return vector*(radius/vector.norm().clamp_min(radius))
        if vector.norm()<=radius:
            return vector
        # Metric projection onto the ORIGINAL Euclidean ball. Ordinary radial
        # scaling would be incorrect for unequal block steps.
        lo=vector.new_tensor(0.,dtype=torch.float64)
        hi=(vector.norm().double()/radius)/diagonal.min().double()
        for _ in range(60):
            mid=(lo+hi)/2
            candidate=vector/(1+mid*diagonal)
            if candidate.norm()>radius:lo=mid
            else:hi=mid
        return vector/(1+hi*diagonal)

    def _probe(self, contexts):
        self.zero_grad(set_to_none=True)
        loss = self._flat().sum()*0
        metrics = {}
        for context in contexts:
            item, current, _ = self.policy.objective(**context)
            loss = loss + item/context['normalizer']['accumulation_steps']
            for key,value in current.items():
                metrics[key] = metrics.get(key, 0) + value.detach()
        parameters = self.param_groups[0]['params']
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        flat = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for p,g in zip(parameters,gradients)]).detach()
        flat += self.param_groups[0]['ridge']*self._flat()
        residual = self.policy.dual_residuals(metrics, self.policy.cfg, flat)
        return flat, residual, float(loss.detach()), metrics

    @torch.no_grad()
    def step(self, contexts=None, metrics=None):
        if contexts is None or metrics is None or not contexts:
            raise ValueError('Extragradient needs a detached full-window cache and first-pass metrics')
        group = self.param_groups[0]
        original, dual = self._flat(), self._dual()
        if not torch.isfinite(original).all() or not torch.isfinite(dual).all():
            raise FloatingPointError('Nonfinite primal/dual state')
        if bool(((dual < 0) | (dual > group['dual_cap'])).any()):
            raise ValueError('Resume state is outside the declared dual box')
        if original.norm() > group['radius']*(1+1e-5):
            raise ValueError('Resume state is outside the declared primal ball')
        parameters = group['params']
        first_gradient = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad)
                                    .detach().reshape(-1) for p in parameters])
        first_gradient += group['ridge']*original
        first_residual = self.policy.dual_residuals(metrics, self.policy.cfg, original)
        if not torch.isfinite(first_gradient).all() or not torch.isfinite(first_residual).all():
            raise FloatingPointError('Nonfinite first-pass saddle operator')
        # Linearized VI gap on B_2(radius) x [0,dual_cap]^3. By convexity /
        # concavity it upper-bounds THIS WINDOW's bounded saddle gap. It does
        # not certify the full training distribution or a hard deployment mask.
        window_gap = torch.dot(first_gradient, original) + group['radius']*first_gradient.norm()
        window_gap += (group['dual_cap']*first_residual.clamp_min(0)-dual*first_residual).sum()
        scale = min(1., group['scale']/group['contraction'])
        probe_evaluations=0
        try:
            for attempt in range(group['max_backtracks']):
                primal_step = group['lr']*scale
                dual_step = primal_step*group['dual_ratio']
                if 'dual_block_factors' in group:
                    dual_step = dual.new_tensor(group['dual_block_factors'])*dual_step
                if (not bool(torch.isfinite(dual_step).all() & (dual_step>0).all()) if torch.is_tensor(dual_step)
                    else dual_step<=0 or not math.isfinite(dual_step)) or primal_step<=0 or not math.isfinite(primal_step):
                    raise FloatingPointError('Invalid extragradient step size')
                block_factors=group.get('primal_block_factors',[1.]*len(parameters))
                diagonal=torch.cat([original.new_full((p.numel(),),primal_step*factor)
                    for p,factor in zip(parameters,block_factors)])
                predictor = self._project(original-diagonal*first_gradient,diagonal)
                predictor_dual = (dual+dual_step*first_residual).clamp(0,group['dual_cap'])
                self._put(predictor); self._put_dual(predictor_dual)
                # Only the affine heads are re-evaluated. Backbone, reference
                # features and all candidate labels remain frozen in contexts.
                same_point=torch.equal(predictor,original) and torch.equal(predictor_dual,dual)
                if same_point and self.policy.cfg.evidence_solver_revision>=2:
                    # An identical floating-point point has the same operator.
                    # Reuse its cached evaluation: recomputing through a different
                    # accumulation graph can introduce roundoff with zero RHS
                    # and cause endless false line-search failures at tiny LRs.
                    second_gradient,second_residual=first_gradient,first_residual
                    probe_loss,probe_metrics=None,metrics
                else:
                    with torch.enable_grad():
                        second_gradient,second_residual,probe_loss,probe_metrics = self._probe(contexts)
                    group['total_probe_evaluations'] += 1
                    probe_evaluations += 1
                lhs = (diagonal*(second_gradient-first_gradient).square()).sum()
                lhs += ((dual_step*(second_residual-first_residual).square()).sum() if torch.is_tensor(dual_step)
                        else dual_step*(second_residual-first_residual).square().sum())
                rhs = ((predictor-original).square()/diagonal).sum()
                rhs += (((predictor_dual-dual).square()/dual_step).sum() if torch.is_tensor(dual_step)
                        else (predictor_dual-dual).square().sum()/dual_step)
                finite = bool(torch.isfinite(lhs) & torch.isfinite(rhs))
                if finite and lhs <= group['acceptance']**2*rhs:
                    corrected = self._project(original-diagonal*second_gradient,diagonal)
                    corrected_dual = (dual+dual_step*second_residual).clamp(0,group['dual_cap'])
                    self._put(corrected); self._put_dual(corrected_dual)
                    group['scale'] = scale; group['accepted_steps'] += 1
                    group['numerically_stagnant_steps']=group.get('numerically_stagnant_steps',0)+int(same_point)
                    # These are WINDOW diagnostics, measured at the original
                    # or predictor point; never called global optimality gaps.
                    mapping = (((original-predictor)/diagonal).square().sum() +
                        (((dual-predictor_dual).square()/dual_step**2).sum() if torch.is_tensor(dual_step)
                         else (dual-predictor_dual).square().sum()/dual_step**2)).sqrt()
                    return dict(backtracks=attempt, head_probe_evaluations=probe_evaluations,
                        accepted_steps=group['accepted_steps'], primal_step=primal_step,
                        dual_step=dual_step.tolist() if torch.is_tensor(dual_step) else dual_step, window_projected_mapping=float(mapping),
                        outcome_step=primal_step*group.get('outcome_block_factor',1.),
                        window_saddle_gap_upper=float(window_gap.clamp_min(0)),
                        window_max_proxy_violation=float(first_residual.clamp_min(0).max()),
                        window_proxy_residuals=first_residual.tolist(),
                        window_complementarity=float((dual*first_residual).abs().max()),
                        dual_at_cap=bool((corrected_dual>=group['dual_cap']-1e-6).any()),
                        numerically_stagnant=same_point,
                        numerically_stagnant_steps=group['numerically_stagnant_steps'],
                        predictor_loss=probe_loss, line_search_lhs=float(lhs),
                        line_search_rhs=float(rhs), probe_metrics=probe_metrics,
                        primal_radius=float(corrected.norm()))
                scale *= group['contraction']
            raise RuntimeError('Extragradient line search exhausted; no update was committed')
        except Exception:
            self._put(original); self._put_dual(dual)
            raise
