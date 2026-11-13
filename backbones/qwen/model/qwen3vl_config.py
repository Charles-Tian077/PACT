"""
Qwen3-VL Configuration for SESAME Integration.

This module defines the configuration classes needed to integrate
Qwen3-VL as the backbone for SESAME's See-Say-Segment framework.
"""

import os
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union


@dataclass
class Qwen3VLSesameConfig:
    """
    Configuration for Qwen3-VL based SESAME model.
    
    This config bridges Qwen3-VL's native multimodal architecture with
    SESAME's segmentation requirements.
    """
    # === Qwen3-VL Base Model ===
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    architecture_version: int = 2
    query_source: str = "prompt"
    allow_random_sam: bool = False
    sam_model_type: str = "vit_h"
    use_lora: bool = True
    box_prompt_enabled: bool = True
    presence_gate: bool = True
    
    # === Vision tower: Qwen3-VL's native vision transformer ===
    # DeepStack configuration is read from the pretrained backbone itself.
    freeze_vision_tower: bool = True
    
    # === LoRA Configuration ===
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])
    
    # === Segmentation Head Configuration ===
    seg_token: str = "[SEG]"
    hidden_size: int = 4096  # Qwen3-VL-8B hidden size
    out_dim: int = 256  # Kept for loading legacy configs/checkpoints
    semantic_hidden_dim: int = 256
    geometry_hidden_dim: int = 128
    train_mask_decoder: bool = True
    vision_pretrained: str = ""  # Path to SAM checkpoint
    presence_threshold: float = 0.5
    
    # === Training Configuration ===
    ce_loss_weight: float = 1.0
    dice_loss_weight: float = 0.5
    bce_loss_weight: float = 2.0
    presence_loss_weight: float = 1.0
    box_loss_weight: float = 1.0
    mask_code_loss_weight: float = 0.0
    anchor_loss_weight: float = 0.0
    spatial_rl_weight: float = 0.0  # legacy Gaussian-box policy; disabled
    token_cost_loss_weight: float = 0.0
    eos_token_id: Optional[int] = None
    learnable_loss_scales: bool = False

    # A finite-action contextual bandit. No language rollouts or learned critic.
    evidence_enabled: bool = True
    evidence_hidden_dim: int = 128
    evidence_warmup_weight: float = 0.2
    evidence_kl_beta: float = 0.05
    evidence_kl_target: float = 0.05
    evidence_reference_decay: float = 0.95
    evidence_objective: str = 'expectation'
    evidence_null_bias: float = 0.0
    evidence_cost: float = 0.03
    evidence_dual_lr: float = 0.02
    evidence_fpr_target: float = 0.05
    evidence_fnr_target: float = 0.05
    evidence_pair_weight: float = 0.1
    evidence_pair_margin: float = 0.2
    # Missing fields in an existing checkpoint mean the original v2 algorithm.
    # V4 is explicitly selected for new stages; loading never silently converts.
    evidence_policy_version: int = 2
    evidence_value_weight: float = 1.0
    evidence_action_temperature: float = 0.2
    evidence_gain_margin: float = 0.0
    evidence_risk_temperature: float = 1.0
    # V4 reuses the action MLP; two scalar outputs predict safe benefit/harm.
    evidence_outcome_weight: float = 1.0
    evidence_repairability_threshold: float = 0.5
    evidence_harm_probability_threshold: float = 0.2
    evidence_repair_iou_tolerance: float = 0.02
    evidence_repair_harm_target: float = 0.02
    evidence_switch_temperature: float = 0.2
    evidence_gate_offset: float = 0.0
    # V5 freezes SFT features and optimizes affine heads on a compact domain.
    evidence_primal_radius: float = 10.0
    evidence_dual_cap: float = 10.0
    evidence_ridge: float = 1e-4
    # Old V5 bundles lacked a solver revision. Preserve their original recipe.
    evidence_solver_revision: int = 1
    
    # === Image Processing ===
    image_size: int = 1024
    max_pixels: int = 1048576  # at most 1024 Qwen3-VL image tokens
    min_pixels: int = 4096
    max_length: int = 2048
    
    # === Conversation ===
    system_prompt: str = (
        "You are a helpful assistant capable of seeing images, "
        "answering questions about them, and segmenting objects. "
        "If an object is not present in the image, politely say so "
        "and suggest alternatives."
    )
    
    # === Special Tokens ===
    # Qwen3-VL uses these natively:
    # vision_start_token_id: 151652
    # vision_end_token_id: 151653
    # image_token_id: 151655
    # video_token_id: 151656
    # We add [SEG] token for segmentation
    
    def __post_init__(self):
        if self.evidence_solver_revision not in (1,2,3):
            raise ValueError('Unsupported evidence solver revision')
        for name in ('evidence_repairability_threshold','evidence_harm_probability_threshold',
                     'evidence_repair_iou_tolerance','evidence_repair_harm_target'):
            if not 0<=getattr(self,name)<=1:
                raise ValueError(name+' must be in [0,1]')
        for name in ('evidence_outcome_weight','evidence_switch_temperature',
                     'evidence_primal_radius','evidence_dual_cap','evidence_ridge'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(name+' must be finite and positive')
        if not math.isfinite(self.evidence_gate_offset):
            raise ValueError('evidence_gate_offset must be finite')
        if isinstance(self.lora_target_modules, str):
            self.lora_target_modules = [
                x.strip() for x in self.lora_target_modules.split(",")
            ]


def get_default_config(model_size: str = "8B") -> Qwen3VLSesameConfig:
    """
    Get default configuration for different Qwen3-VL model sizes.
    
    Args:
        model_size: One of "2B", "4B", "8B", "32B"
    
    Returns:
        Qwen3VLSesameConfig with appropriate settings
    """
    configs = {
        "2B": Qwen3VLSesameConfig(
            model_name_or_path="Qwen/Qwen3-VL-2B-Instruct",
            hidden_size=2048,
            lora_r=4,
        ),
        "4B": Qwen3VLSesameConfig(
            model_name_or_path="Qwen/Qwen3-VL-4B-Instruct",
            hidden_size=2560,
            lora_r=4,
        ),
        "8B": Qwen3VLSesameConfig(
            model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
            hidden_size=4096,
            lora_r=8,
        ),
        "32B": Qwen3VLSesameConfig(
            model_name_or_path="Qwen/Qwen3-VL-32B-Instruct",
            hidden_size=5120,
            lora_r=8,
        ),
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    return configs[model_size]
