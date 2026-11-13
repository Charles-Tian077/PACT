"""Qwen3-VL/SAM segmentation with a finite-action visual evidence policy."""
import os
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from .evidence_policy import EvidenceBandit, mask_quality, class_balanced_mean, global_mean
from .hierarchical_policy import HierarchicalEvidenceBandit
from .selective_repair_policy import SelectiveRepairBandit
from .convex_repair_policy import ConvexRepairBandit
from .token_adapters import TokenRowEmbedding, TokenRowLMHead


def dice_loss(inputs, targets, num_masks=1.0, scale=1000.0, eps=1e-6):
    prob = inputs.float().sigmoid().flatten(1)
    target = targets.float().flatten(1)
    return (1 - (2 * (prob * target).sum(-1) + eps) /
            (prob.sum(-1) + target.sum(-1) + eps)).sum() / max(float(num_masks), 1)


def sigmoid_ce_loss(inputs, targets, num_masks=1.0):
    return F.binary_cross_entropy_with_logits(inputs.float(), targets.float(), reduction='none').flatten(1).mean(1).sum() / max(float(num_masks), 1)


class SESAMEQwen3VL(nn.Module):
    def __init__(self, config, seg_token_idx, **kwargs):
        super().__init__()
        self.cfg, self.seg_token_idx = config, int(seg_token_idx)
        if config.query_source not in ('prompt', 'seg'):
            raise ValueError('query_source must be prompt or seg')
        if config.evidence_objective not in ('expectation','oracle_ce'):
            raise ValueError('Unsupported evidence objective')
        if config.evidence_policy_version not in (2, 3, 4, 5):
            raise ValueError('Unsupported evidence policy version')
        if config.evidence_enabled and config.query_source != 'prompt':
            raise ValueError('Evidence policy requires pre-answer prompt queries to prevent answer leakage.')
        if any(getattr(config, k, 0) for k in ('spatial_rl_weight', 'token_cost_loss_weight', 'mask_code_loss_weight', 'anchor_loss_weight')):
            raise ValueError('Legacy auxiliary losses are disabled in v2; use the evidence policy or the clean baseline.')
        self._init_qwen3vl()
        self._init_sam()
        self._init_projection()
        policy_class = {2: EvidenceBandit, 3: HierarchicalEvidenceBandit,
                        4: SelectiveRepairBandit, 5: ConvexRepairBandit}[config.evidence_policy_version]
        self.evidence = policy_class(self.sam.prompt_encoder.embed_dim * 4 + 13, config)
        if not config.evidence_enabled:
            self.evidence.requires_grad_(False)
        if not config.box_prompt_enabled:
            self.geometry_head.requires_grad_(False)
        if not config.presence_gate and not config.presence_loss_weight:
            self.existence_head.requires_grad_(False)

    def _init_qwen3vl(self):
        self.dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[self.cfg.torch_dtype]
        self.qwen3vl = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.model_name_or_path, torch_dtype=self.dtype,
            attn_implementation=self.cfg.attn_implementation)
        self.hidden_size = self.qwen3vl.config.text_config.hidden_size
        # LoRA and explicit token rows are enabled only after vocabulary resizing.
        self.qwen3vl.requires_grad_(False)

    def _init_sam(self):
        from segment_anything import sam_model_registry
        path = self.cfg.vision_pretrained
        if not os.path.isfile(path):
            if not self.cfg.allow_random_sam:
                raise FileNotFoundError('A pretrained SAM checkpoint is required: ' + str(path))
            path = None
        self.sam = sam_model_registry[self.cfg.sam_model_type](checkpoint=path)
        self.sam.requires_grad_(False)
        self.sam.mask_decoder.requires_grad_(self.cfg.train_mask_decoder)
        self.sam.to(dtype=self.dtype)

    def _init_projection(self):
        d = self.sam.prompt_encoder.embed_dim
        self.semantic_prompt_fc = nn.Sequential(nn.Linear(self.hidden_size, self.cfg.semantic_hidden_dim),
                                                nn.GELU(), nn.Linear(self.cfg.semantic_hidden_dim, d))
        self.geometry_head = nn.Sequential(nn.Linear(self.hidden_size, self.cfg.geometry_hidden_dim),
                                          nn.GELU(), nn.Linear(self.cfg.geometry_hidden_dim, 4), nn.Sigmoid())
        self.existence_head = nn.Sequential(nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, 1))

    def setup_lora(self, lora_config=None):
        targets = self.cfg.lora_target_modules
        # Restrict adapters to the text decoder; visual modules are never matched.
        pattern = r'.*language_model\.layers\.\d+\.self_attn\.(' + '|'.join(targets) + ')'
        options = dict(r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha,
                       lora_dropout=self.cfg.lora_dropout, target_modules=pattern,
                       bias='none', task_type='CAUSAL_LM')
        if lora_config:
            options.update(lora_config)
        self.qwen3vl = get_peft_model(self.qwen3vl, LoraConfig(**options))
        return self

    def enable_seg_token_rows(self):
        base = self.qwen3vl.get_base_model() if hasattr(self.qwen3vl, 'get_base_model') else self.qwen3vl
        if not isinstance(base.get_input_embeddings(), TokenRowEmbedding):
            base.set_input_embeddings(TokenRowEmbedding(base.get_input_embeddings(), self.seg_token_idx))
            base.set_output_embeddings(TokenRowLMHead(base.get_output_embeddings(), self.seg_token_idx))

    def start_rl_stage(self):
        self.evidence.start_rl()
        # The decoder and proposal generator define a stationary reward environment.
        self.requires_grad_(False)
        self.evidence.enable_policy_gradients()
        self.qwen3vl.eval()
        self.sam.eval()

    def upgrade_evidence_policy(self, version):
        """Explicit, in-memory SFT conversion; never rewrite a source checkpoint."""
        if version == self.cfg.evidence_policy_version:
            return
        if version not in (3, 4, 5) or self.cfg.evidence_policy_version not in (2, 3, 4):
            raise ValueError('Only an explicit SFT -> newer policy new-stage conversion is supported')
        if version <= self.cfg.evidence_policy_version:
            raise ValueError('Policy conversion must move to a newer version')
        if bool(self.evidence.rl_active.item()):
            raise ValueError('Cannot convert an active RL checkpoint; resume its saved policy version')
        previous = self.evidence
        self.cfg.evidence_policy_version = version
        policy_class = {3: HierarchicalEvidenceBandit, 4: SelectiveRepairBandit,
                        5: ConvexRepairBandit}[version]
        replacement = policy_class(self.sam.prompt_encoder.embed_dim * 4 + 13, self.cfg)
        replacement.to(device=next(previous.parameters()).device)
        replacement.policy.load_state_dict(previous.policy.state_dict(), strict=True)
        self.evidence = replacement

    def train(self, mode=True):
        super().train(mode)
        base=self.qwen3vl.get_base_model() if hasattr(self.qwen3vl,'get_base_model') else self.qwen3vl
        base.visual.eval()
        self.sam.image_encoder.eval()
        self.sam.prompt_encoder.eval()
        self.sam.mask_decoder.train(mode and self.cfg.train_mask_decoder)
        self.evidence.reference.eval()
        if bool(self.evidence.rl_active.item()):
            self.qwen3vl.eval()
            self.sam.eval()
        return self

    def _backbone_forward(self, **arguments):
        """Keep the final normalized states, without retaining every LLM layer."""
        base=self.qwen3vl.get_base_model() if hasattr(self.qwen3vl,'get_base_model') else self.qwen3vl
        states=[]
        hook=base.get_decoder().norm.register_forward_hook(lambda module,args,out:states.append(out))
        arguments['output_hidden_states']=False
        try:
            outputs=self.qwen3vl(**arguments)
        finally:
            hook.remove()
        if len(states)!=1:
            raise RuntimeError('Expected exactly one final text-normalization output')
        return outputs,states[0]

    @staticmethod
    def _cxcywh_to_xyxy(boxes):
        half = boxes[..., 2:].clamp_min(1e-3) / 2
        return torch.cat([(boxes[..., :2] - half).clamp(0, 1),
                          (boxes[..., :2] + half).clamp(0, 1)], -1)

    @staticmethod
    def _mask_to_box(mask):
        points = torch.nonzero(mask > 0.5, as_tuple=False)
        if not points.numel():
            return None
        low, high = points.min(0).values, points.max(0).values
        h, w = mask.shape[-2:]
        return mask.new_tensor([low[-1] / w, low[-2] / h, (high[-1] + 1) / w, (high[-2] + 1) / h])

    @staticmethod
    def _mask_on_sam_grid(mask, input_size, canvas_size, output_size, device, dtype):
        mask = torch.as_tensor(mask, device=device, dtype=dtype)
        while mask.ndim > 2:
            mask = mask[0]
        h, w = map(int, input_size)
        mask = F.interpolate(mask[None, None], size=(h, w), mode='nearest')
        mask = F.pad(mask, (0, canvas_size - w, 0, canvas_size - h))
        return F.interpolate(mask, size=output_size, mode='area')[0, 0]

    def _encode_sam(self, image):
        if image.ndim == 3:
            image = image[None]
        parameter = next(self.sam.parameters())
        image = image.to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            return self.sam.image_encoder(self.sam.preprocess(image))

    def _sam_decode(self, embedding, semantic, boxes, input_size):
        """SAM v1 repeats one image internally for a batch of K prompts."""
        parameter = next(self.sam.parameters())
        h, w = map(int, input_size)
        boxes = (boxes * boxes.new_tensor([w, h, w, h])).to(parameter.dtype)
        sparse, dense = self.sam.prompt_encoder(points=None, boxes=boxes if self.cfg.box_prompt_enabled else None, masks=None)
        token = semantic.to(parameter.dtype).reshape(1, 1, -1).expand(boxes.shape[0], -1, -1)
        sparse = torch.cat([sparse, token], 1)
        logits, _ = self.sam.mask_decoder(image_embeddings=embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense, multimask_output=False)
        return logits[:, 0]

    def _proposals(self, box, semantic, embedding, input_size):
        """Eight boxes: base, four shifts, expansion, two separated visual peaks."""
        with torch.no_grad():
            box = box.detach().float()
            center, size = (box[:2] + box[2:]) / 2, (box[2:] - box[:2]).clamp_min(1e-3)
            offsets = box.new_tensor([[0,0],[-.25,0],[.25,0],[0,-.25],[0,.25],[0,0]])
            centers = center[None] + offsets * size
            sizes = size[None].expand(6, -1).clone()
            sizes[5] *= 1.5
            f = embedding[0].float()
            score = (F.normalize(f, dim=0) * F.normalize(semantic.detach().float(), dim=0)[:, None, None]).sum(0)
            canvas = self.sam.image_encoder.img_size
            fh, fw = score.shape
            vh = max(1, min(fh, round(input_size[0] / canvas * fh)))
            vw = max(1, min(fw, round(input_size[1] / canvas * fw)))
            score[vh:] = -1e4
            score[:, vw:] = -1e4
            peaks = []
            for _ in range(2):
                index = int(score.argmax())
                y, x = index // fw, index % fw
                peaks.append(box.new_tensor([(x+.5)/vw, (y+.5)/vh]))
                radius = max(1, int(min(vh, vw) * .15))
                score[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1] = -1e4
            centers = torch.cat([centers, torch.stack(peaks)])
            sizes = torch.cat([sizes, size[None].expand(2,-1)])
            proposals = torch.cat([centers - sizes/2, centers + sizes/2], -1).clamp(0,1)
            proposals[0] = box
            available = ((proposals[:,2:] - proposals[:,:2]) > 1e-3).all(-1)
            for j in range(1, len(proposals)):
                if ((proposals[:j] - proposals[j]).abs().amax(-1) < 1e-3).any():
                    available[j] = False
            available[0] = True
            return proposals, torch.cat([available.new_ones(1), available])

    def _evidence_features(self, semantic, presence, embedding, base_mask, boxes, input_size):
        """Features use predicted evidence only; never a ground-truth mask/label."""
        with torch.no_grad():
            f = embedding[0].float()
            d, h, w = f.shape
            canvas = self.sam.image_encoder.img_size
            valid = torch.zeros(h,w,device=f.device)
            vh = max(1, min(h, round(input_size[0]/canvas*h)))
            vw = max(1, min(w, round(input_size[1]/canvas*w)))
            valid[:vh,:vw] = 1
            prob = F.interpolate(base_mask[None].float(), size=(h,w), mode='bilinear', align_corners=False)[0,0].sigmoid() * valid
            def pool(weights):
                return (f * weights).sum((-2,-1)) / weights.sum().clamp_min(1e-6)
            mask_feature, context = pool(prob), pool((valid-prob).clamp_min(0))
            yy, xx = torch.meshgrid((torch.arange(h,device=f.device)+.5)/vh,
                                    (torch.arange(w,device=f.device)+.5)/vw,indexing='ij')
            entropy = -(prob.clamp(1e-6,1-1e-6)*prob.clamp_min(1e-6).log() +
                        (1-prob).clamp_min(1e-6)*(1-prob).clamp_min(1e-6).log())
            stats = torch.stack([prob.sum()/valid.sum(), (entropy*valid).sum()/valid.sum(),
                                 presence.detach().float().sigmoid(), prob.max(),
                                 (prob.square()*valid).sum()/valid.sum(), valid.mean()])
            features = []
            all_boxes = torch.cat([boxes.new_zeros(1,4),boxes])
            for a, b in enumerate(all_boxes):
                roi = valid if a == 0 else ((xx>=b[0]) & (xx<=b[2]) & (yy>=b[1]) & (yy<=b[3])).float()*valid
                kind = f.new_tensor([a==0,a==1,a>1])
                features.append(torch.cat([semantic.detach().float(), pool(roi), mask_feature,
                                           context,b,stats,kind]))
            prior = F.logsigmoid(presence.detach().float()).expand(len(all_boxes)).clone()
            prior[0] = F.logsigmoid(-presence.detach().float())
            return torch.stack(features), prior

    def _query_states(self, hidden, input_ids, attention_mask, labels=None, prompt_positions=None):
        states, presence, markers = [], [], []
        for i in range(input_ids.shape[0]):
            valid = torch.where(attention_mask[i].bool())[0]
            if not valid.numel():
                raise ValueError('Empty attention mask')
            if prompt_positions is not None:
                p = int(prompt_positions[i])
            elif labels is not None:
                supervised = torch.where(labels[i] != -100)[0]
                if not supervised.numel():
                    raise ValueError('No supervised assistant tokens')
                p = int(supervised[0]) - 1
            else:
                p = int(valid[-1])
            if p < 0 or not attention_mask[i,p]:
                raise ValueError('Invalid prompt query position')
            presence.append(hidden[i,p])
            # A literal marker in the user prompt never triggers segmentation.
            positions = torch.where((input_ids[i] == self.seg_token_idx) & attention_mask[i].bool() &
                                    (torch.arange(input_ids.shape[1],device=input_ids.device)>p))[0]
            markers.append(bool(positions.numel()))
            q = p if self.cfg.query_source == 'prompt' or not positions.numel() else int(positions[-1])
            states.append(hidden[i,q])
        return torch.stack(states), torch.stack(presence), markers

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None,
                labels=None, masks_list=None, ignore_masks_list=None, sam_images=None,
                sam_mask_shape_list=None, exists_list=None, pair_ids=None, pair_kinds=None,
                presence_query_positions=None, spatial_rl_samples=0, inference=False, **kwargs):
        if spatial_rl_samples:
            raise ValueError('Gaussian-box GRPO was replaced by the finite-action evidence bandit.')
        attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        cache_requested=inference and kwargs.get('return_language_cache',False)
        arguments = dict(input_ids=input_ids,attention_mask=attention_mask,pixel_values=pixel_values,
                         image_grid_thw=image_grid_thw,use_cache=cache_requested)
        if labels is not None and not inference and not self.evidence.rl_active.item():
            arguments['labels'] = labels
        else:
            arguments['logits_to_keep'] = 1
        with torch.set_grad_enabled(self.training and not inference and not bool(self.evidence.rl_active.item())):
            outputs,hidden = self._backbone_forward(**arguments)
        q, p, markers = self._query_states(hidden,input_ids,attention_mask,labels,presence_query_positions)
        with torch.set_grad_enabled(self.training and not inference and not bool(self.evidence.rl_active.item())):
            semantic = self.semantic_prompt_fc(q.to(self.semantic_prompt_fc[0].weight.dtype))
            boxes = self._cxcywh_to_xyxy(self.geometry_head(q.to(self.geometry_head[0].weight.dtype)))
            presence = self.existence_head(p.to(self.existence_head[0].weight.dtype)).squeeze(-1)
        result = self._segment(semantic,boxes,presence,markers,sam_images,sam_mask_shape_list,
            exists_list,masks_list,ignore_masks_list,outputs.loss if hasattr(outputs,'loss') else None,
            pair_ids,pair_kinds,inference,image_keys=kwargs.get('image_keys'),
            policy_normalizer=kwargs.get('policy_normalizer'),
            audit_candidates=kwargs.get('audit_candidates', False))
        if cache_requested:
            result['_language_cache']=outputs.past_key_values
        return result

    def _segment(self, semantic, boxes, presence, markers, images, shapes, exists,
                  masks, ignores, text_loss=None, pair_ids=None, pair_kinds=None, inference=False,image_keys=None,
                  policy_normalizer=None, audit_candidates=False):
        batch = len(semantic)
        if images is None or shapes is None:
            raise ValueError('SAM images and shape metadata are required')
        images = list(images.unbind(0)) if torch.is_tensor(images) else list(images)
        if len(images) != batch or len(shapes) != batch:
            raise ValueError('Mismatched SAM image/query batch')
        exists = [None]*batch if exists is None else exists
        masks = [None]*batch if masks is None else masks
        ignores = [None]*batch if ignores is None else ignores
        zero = semantic.sum()*0
        losses = {k:zero for k in ('mask_loss','box_loss','presence_loss','evidence_loss','pair_loss')}
        predicted, actions, decisions = [None]*batch, [-1]*batch, [False]*batch
        policy_items, reward_items, supervised_presence = [], [], []
        outcome_items, outcome_valid = [], []
        diagnostics, audit_masks, audit_logits = [None]*batch, [None]*batch, [None]*batch
        v4_active = self.cfg.evidence_policy_version >= 4 and bool(self.evidence.rl_active.item())
        if audit_candidates and (not inference or not v4_active or not self.cfg.evidence_enabled):
            raise ValueError('Candidate audit is an explicit V4 inference diagnostic only')
        per_mask, per_box = [], []
        image_cache = {}
        for i in range(batch):
            if not inference and exists[i] is None:  # VQA contributes language supervision only.
                continue
            if not self.cfg.evidence_enabled:
                decode = (not self.cfg.presence_gate or bool(presence[i].sigmoid()>=self.cfg.presence_threshold)) if inference else bool(exists[i])
                decode = decode and (self.cfg.query_source=='prompt' or markers[i])
                if not decode:
                    continue
            input_size, original_size = shapes[i]
            cache_key=(image_keys[i],tuple(input_size)) if image_keys and image_keys[i] else None
            if cache_key is not None and cache_key in image_cache:
                embedding=image_cache[cache_key]
            else:
                embedding=self._encode_sam(images[i])
                if cache_key is not None:
                    image_cache[cache_key]=embedding
            frozen = inference or bool(self.evidence.rl_active.item())
            with torch.set_grad_enabled(self.training and not frozen):
                base_mask = self._sam_decode(embedding,semantic[i],boxes[i:i+1],input_size)
            selected = base_mask
            if self.cfg.evidence_enabled:
                candidates, available = self._proposals(boxes[i],semantic[i],embedding,input_size)
                features, prior = self._evidence_features(semantic[i],presence[i],embedding,base_mask,candidates,input_size)
                prior[1:] -= available[1:].sum().clamp_min(1).float().log()
                policy_items.append((i,features,prior,available))
                if inference:
                    if v4_active:
                        diagnostics[i] = self.evidence.decision_metadata(features[None],prior[None],available[None])[0]
                        action = diagnostics[i]['chosen_action']
                    else:
                        action = int(self.evidence.select_actions(features[None],prior[None],available[None]))
                    actions[i] = action
                    if audit_candidates:
                        # Match normal deployment's one-prompt decoder call.
                        # Batched prompts can change bf16 numerical outcomes.
                        other = torch.cat([self._sam_decode(embedding,semantic[i],candidates[j:j+1],input_size)
                                           for j in range(1,len(candidates))],0)
                        all_masks = torch.cat([base_mask,other],0)
                        audit_logits[i] = all_masks.detach().float().cpu()
                        # Diagnostics enumerate candidates. Never include this
                        # execution in normal-policy latency/cost comparisons.
                        audit_masks[i] = [self.sam.postprocess_masks(m[None,None],
                            input_size=tuple(input_size),original_size=tuple(original_size))[0,0].float().cpu()
                            for m in all_masks]
                    if action == 0:
                        continue
                    if action > 1:
                        selected = (all_masks[action-1:action] if audit_candidates else
                                    self._sam_decode(embedding,semantic[i],candidates[action-1:action],input_size))
                else:
                    # No language rollout; at most one batched refinement call per image.
                    with torch.no_grad():
                        if exists[i]:
                            if masks[i] is None:
                                raise ValueError('Positive training item has no mask')
                            other = (torch.cat([self._sam_decode(embedding,semantic[i].detach(),candidates[j:j+1],input_size)
                                                for j in range(1,len(candidates))],0) if v4_active else
                                     self._sam_decode(embedding,semantic[i].detach(),candidates[1:],input_size))
                            all_masks = torch.cat([base_mask.detach(),other],0)
                            canvas = self.sam.image_encoder.img_size
                            target = self._mask_on_sam_grid(masks[i],input_size,canvas,all_masks.shape[-2:],all_masks.device,torch.float32)
                            valid = self._mask_on_sam_grid(torch.ones_like(masks[i]),input_size,canvas,all_masks.shape[-2:],all_masks.device,torch.float32)
                            if ignores[i] is not None:
                                valid *= 1-self._mask_on_sam_grid(ignores[i],input_size,canvas,all_masks.shape[-2:],all_masks.device,torch.float32)
                            quality = mask_quality(all_masks,target,valid)
                            rewards = torch.cat([quality.new_tensor([-1.0]),2*quality-1])
                            if v4_active:
                                # Measure actual full-resolution binary IoU;
                                # the original soft-IoU/boundary reward is unchanged.
                                ious, usable = self._candidate_ious(all_masks,masks[i],ignores[i],input_size,original_size)
                                outcome_items.append(torch.cat([ious.new_zeros(1),ious]))
                                outcome_valid.append(usable)
                        else:
                            rewards = features.new_full((len(available),),-1.0)
                            rewards[0] = 1.0
                            if v4_active:
                                outcome_items.append(torch.zeros_like(rewards))
                                outcome_valid.append(False)
                        rewards[2:] -= self.cfg.evidence_cost
                        reward_items.append(rewards)
            full = self.sam.postprocess_masks(selected[:,None],input_size=tuple(input_size),original_size=tuple(original_size))[:,0]
            predicted[i] = full
            decisions[i] = True
            if not inference and exists[i] and not frozen:
                target = masks[i].to(device=full.device,dtype=torch.float32)
                if target.ndim==2:
                    target = target[None]
                if target.shape != full.shape:
                    raise ValueError('Target mask and original image dimensions differ')
                valid = torch.ones_like(target) if ignores[i] is None else (1-ignores[i].to(full.device).float()).reshape_as(target)
                prob = full.float().sigmoid()*valid
                bce = (F.binary_cross_entropy_with_logits(full.float(),target,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
                dice = 1-(2*(prob*target).sum()+1e-6)/(prob.sum()+(target*valid).sum()+1e-6)
                per_mask.append(self.cfg.bce_loss_weight*bce+self.cfg.dice_loss_weight*dice)
                gt_box = self._mask_to_box((target*valid)[0])
                if gt_box is not None:
                    per_box.append(F.smooth_l1_loss(boxes[i].float(),gt_box.float()))
        if inference:
            return {'pred_masks':predicted,'object_presence':torch.tensor(decisions,device=semantic.device),
                    'pred_boxes':boxes,'existence_logits':presence,'evidence_actions':actions,
                    'policy_diagnostics':diagnostics,
                    'evidence_candidate_masks':audit_masks if audit_candidates else None,
                    'evidence_candidate_logits':audit_logits if audit_candidates else None,
                    'candidate_audit':bool(audit_candidates),
                    'decoder_calls':[0 if a==-1 and not decisions[i] else 1+int(a>1) for i,a in enumerate(actions)]}
        if not self.evidence.rl_active.item():
            losses['mask_loss'] = global_mean(per_mask,zero)
            losses['box_loss'] = global_mean(per_box,zero)*self.cfg.box_loss_weight
        indices = [i for i,e in enumerate(exists) if e is not None]
        if not self.evidence.rl_active.item():
            labels = presence.new_tensor([exists[i] for i in indices])
            bce = F.binary_cross_entropy_with_logits(presence[indices].float(),labels.float(),reduction='none')
            losses['presence_loss'] = class_balanced_mean(bce,labels.bool())*self.cfg.presence_loss_weight
        metrics = {}
        if self.cfg.evidence_enabled:
            if policy_items:
                ids,features,priors,available = zip(*policy_items)
                feature_batch,prior_batch,available_batch,reward_batch = torch.stack(features),torch.stack(priors),torch.stack(available),torch.stack(reward_items)
            else:
                ids=[]
                feature_batch=semantic.new_empty((0,9,self.sam.prompt_encoder.embed_dim*4+13))
                prior_batch=semantic.new_empty((0,9))
                available_batch=torch.empty((0,9),device=semantic.device,dtype=torch.bool)
                reward_batch=semantic.new_empty((0,9))
            labels = torch.tensor([exists[i] for i in ids],device=semantic.device,dtype=torch.bool)
            extra = {}
            if v4_active:
                extra['outcomes'] = {'hard_iou':torch.stack(outcome_items) if outcome_items else reward_batch.clone(),
                    'iou_valid':torch.tensor(outcome_valid,device=semantic.device,dtype=torch.bool)}
            if self.cfg.evidence_policy_version == 5:
                relations = []
                if pair_ids:
                    for j in range(len(ids)):
                        for k in range(j+1,len(ids)):
                            a,b = ids[j],ids[k]
                            if pair_ids[a] is None or pair_ids[a]!=pair_ids[b]:
                                continue
                            if pair_kinds and pair_kinds[a]=='irrelevant_change' and exists[a]==exists[b]:
                                relations.append((j,k,'invariant'))
                            elif exists[a]!=exists[b]:
                                neg,pos = (j,k) if not exists[a] else (k,j)
                                relations.append((neg,pos,'ordered'))
                extra['pair_relations'] = relations
            policy_loss,metrics,probs = self.evidence.objective(feature_batch,prior_batch,available_batch,reward_batch,labels,
                                                             normalizer=policy_normalizer,**extra)
            losses['evidence_loss'] = policy_loss
            pairs = []
            if pair_ids and self.cfg.evidence_policy_version != 5:
                for j in range(len(ids)):
                    for k in range(j+1,len(ids)):
                        a,b = ids[j],ids[k]
                        if pair_ids[a] is None or pair_ids[a]!=pair_ids[b]:
                            continue
                        if pair_kinds and pair_kinds[a]=='irrelevant_change' and exists[a]==exists[b]:
                            pairs.append((probs[j,0]-probs[k,0]).square())
                        elif exists[a]!=exists[b]:
                            neg,pos = (j,k) if not exists[a] else (k,j)
                            pairs.append(F.relu(self.cfg.evidence_pair_margin-probs[neg,0]+probs[pos,0]))
            losses['pair_loss'] = global_mean(pairs,zero,policy_normalizer)*self.cfg.evidence_pair_weight
        ce = text_loss*self.cfg.ce_loss_weight if text_loss is not None else zero
        if self.evidence.rl_active.item():
            total = losses['evidence_loss']+losses['pair_loss']
        else:
            total = ce+losses['mask_loss']+losses['box_loss']+losses['presence_loss']+self.cfg.evidence_warmup_weight*losses['evidence_loss']+losses['pair_loss']
        return {'loss':total,'ce_loss':ce,**losses,'policy_metrics':metrics}

    @torch.no_grad()
    def _candidate_ious(self, masks, target, ignore, input_size, original_size):
        target = target.to(device=masks.device).reshape(tuple(original_size)) > 0.5
        valid = torch.ones_like(target) if ignore is None else ignore.to(masks.device).reshape_as(target) <= 0.5
        usable = bool((target & valid).any())
        values = []
        # One original-resolution mask at a time bounds memory regardless of
        # the number of actions; no additional SAM or LLM forward is required.
        for mask in masks:
            full = self.sam.postprocess_masks(mask[None,None],input_size=tuple(input_size),
                                             original_size=tuple(original_size))[0,0] > 0
            intersection = (full & target & valid).sum().float()
            union = ((full | target) & valid).sum().float()
            values.append(intersection / union.clamp_min(1))
        return torch.stack(values), usable

    @torch.inference_mode()
    def generate(self,input_ids,attention_mask=None,pixel_values=None,image_grid_thw=None,
                 sam_images=None,sam_mask_shape_list=None,max_new_tokens=64,**kwargs):
        if self.cfg.query_source=='prompt':
            explain=kwargs.get('explain',False)
            result = self.forward(input_ids=input_ids,attention_mask=attention_mask,pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,sam_images=sam_images,sam_mask_shape_list=sam_mask_shape_list,inference=True,
                return_language_cache=explain,audit_candidates=kwargs.get('audit_candidates',False))
            result['responses'] = ['Present. [SEG]' if p else 'Absent.' for p in result['object_presence'].tolist()]
            # This direct decision route deliberately has no autoregressive text tokens.
            result.update(output_ids=input_ids,response_tokens=[0]*len(input_ids),backbone_prefills=1)
            if explain:
                tokenizer=kwargs.get('tokenizer')
                if len(input_ids)!=1 or tokenizer is None:
                    raise ValueError('Optional free-form explanation currently requires batch_size=1 and tokenizer=')
                self._explain_cached(result,input_ids,attention_mask,tokenizer,max_new_tokens)
            return result
        return self._generate_cached(input_ids,attention_mask,pixel_values,image_grid_thw,
                                     sam_images,sam_mask_shape_list,max_new_tokens)

    def _generate_cached(self,ids,attention,pixels,grid,images,shapes,max_new_tokens):
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        attention = torch.ones_like(ids) if attention is None else attention
        if not attention[:,-1].all():
            raise ValueError('Generation requires left padding; training can use right padding.')
        out,hidden = self._backbone_forward(input_ids=ids,attention_mask=attention,pixel_values=pixels,image_grid_thw=grid,
                          use_cache=True,logits_to_keep=1)
        prompt = hidden[:,-1]
        query = prompt.clone()
        markers = torch.zeros(len(ids),device=ids.device,dtype=torch.bool)
        finished = torch.zeros_like(markers)
        eos = self.cfg.eos_token_id
        eos = [eos] if isinstance(eos,int) else (eos or [])
        generated = ids
        for step in range(max_new_tokens):
            token = out.logits[:,-1].argmax(-1)
            token = torch.where(finished,token.new_full(token.shape,eos[0] if eos else 0),token)
            generated = torch.cat([generated,token[:,None]],1)
            is_seg = (token==self.seg_token_idx)&~finished
            just_finished = torch.zeros_like(finished)
            for end in eos:
                just_finished |= token==end
            attention = torch.cat([attention,(~finished).long()[:,None]],1)
            finished |= just_finished
            if finished.all() and not is_seg.any():
                break
            if step==max_new_tokens-1 and not is_seg.any():
                break
            out,hidden = self._backbone_forward(input_ids=token[:,None],attention_mask=attention,
                past_key_values=out.past_key_values,use_cache=True,logits_to_keep=1)
            query[is_seg] = hidden[is_seg,-1]
            markers |= is_seg
        semantic = self.semantic_prompt_fc(query.to(self.semantic_prompt_fc[0].weight.dtype))
        boxes = self._cxcywh_to_xyxy(self.geometry_head(query.to(self.geometry_head[0].weight.dtype)))
        presence = self.existence_head(prompt.to(self.existence_head[0].weight.dtype)).squeeze(-1)
        result = self._segment(semantic,boxes,presence,markers.tolist(),images,shapes,None,None,None,inference=True)
        return {'output_ids':generated,'backbone_prefills':1,**result}

    def _explain_cached(self,result,input_ids,attention,tokenizer,max_new_tokens):
        """Optional language readout after the policy decision; no visual re-encoding."""
        prefix=tokenizer.encode(result['responses'][0]+' ',add_special_tokens=False)
        forced=input_ids.new_tensor([prefix])
        attention=torch.ones_like(input_ids) if attention is None else attention
        attention=torch.cat([attention,attention.new_ones((1,len(prefix)))],1)
        out,_=self._backbone_forward(input_ids=forced,attention_mask=attention,
            past_key_values=result.pop('_language_cache'),use_cache=True,logits_to_keep=1)
        eos=self.cfg.eos_token_id
        eos=[eos] if isinstance(eos,int) else (eos or [])
        generated=[]
        for step in range(max_new_tokens):
            token=out.logits[:,-1].argmax(-1)
            generated.append(int(token[0]))
            if int(token[0]) in eos or step==max_new_tokens-1:
                break
            attention=torch.cat([attention,attention.new_ones((1,1))],1)
            out,_=self._backbone_forward(input_ids=token[:,None],attention_mask=attention,
                past_key_values=out.past_key_values,use_cache=True,logits_to_keep=1)
        result['output_ids']=torch.cat([input_ids,forced,input_ids.new_tensor([generated])],1)
        result['responses']=[tokenizer.decode(prefix+generated,skip_special_tokens=True).strip()]
        result['response_tokens']=[len(generated)]
        result['forced_response_tokens']=[len(prefix)]


def load_tokenizer_and_processor(model_path):
    processor = AutoProcessor.from_pretrained(model_path)
    return processor.tokenizer,processor


def add_seg_token(tokenizer,seg_token='[SEG]'):
    tokenizer.add_tokens([seg_token],special_tokens=True)
    return tokenizer.convert_tokens_to_ids(seg_token)


def get_chat_eos_token_id(tokenizer):
    value = tokenizer.convert_tokens_to_ids('<|im_end|>')
    return value if value is not None and value!=tokenizer.unk_token_id else tokenizer.eos_token_id


def initialize_seg_token_embedding(model,tokenizer,seg_token_idx):
    embedding = model.qwen3vl.get_input_embeddings().weight
    sources = sorted({i for word in ('segment','mask','object') for i in tokenizer.encode(word,add_special_tokens=False) if i!=seg_token_idx})
    if sources:
        with torch.no_grad():
            embedding[seg_token_idx].copy_(embedding[sources].float().mean(0).to(embedding.dtype))
            output = model.qwen3vl.get_output_embeddings().weight
            output[seg_token_idx].copy_(output[sources].float().mean(0).to(output.dtype))


def build_sesame_qwen3vl(config,seg_token_idx,lora=True,tokenizer=None):
    model = SESAMEQwen3VL(config,seg_token_idx)
    if tokenizer is not None:
        model.qwen3vl.resize_token_embeddings(len(tokenizer))
        initialize_seg_token_embedding(model,tokenizer,seg_token_idx)
    config.use_lora = bool(lora)
    if lora:
        model.setup_lora()
    model.enable_seg_token_rows()
    return model
