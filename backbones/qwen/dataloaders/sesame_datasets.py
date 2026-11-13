"""
Dataset classes for SESAME-Qwen3-VL.

Integrates RefCOCO, FP-RefCOCO, VQA, and other datasets
adapted for Qwen3-VL's processor and tokenizer.

NOTE: These classes follow the structure of the original SESAME
datasets but use Qwen3-VL's native image processing.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Dict, Any

from dataloaders.data_utils import Qwen3VLImageProcessor


class BaseSEGAMEDataset(Dataset):
    """Base class for SESAME datasets with Qwen3-VL processing."""
    
    def __init__(
        self,
        base_image_dir: str,
        processor,
        samples_per_epoch: int = 4000,
        image_size: int = 1024,
        num_classes_per_sample: int = 1,
    ):
        self.base_image_dir = base_image_dir
        self.processor = processor
        self.samples_per_epoch = samples_per_epoch
        self.image_size = image_size
        self.num_classes_per_sample = num_classes_per_sample
        
        self.image_processor = Qwen3VLImageProcessor(
            processor=processor,
            image_size=image_size,
        )
        
        # System prompt for SESAME
        self.system_prompt = (
            "Check whether the requested target is visible. Reply briefly. "
            "Say present or absent, name an alternative if known, and emit [SEG] "
            "only for a present target that should be segmented."
        )
    
    def __len__(self):
        return self.samples_per_epoch
    
    def _load_image(self, image_path: str, prepare_sam=True) -> Tuple[torch.Tensor, torch.Tensor, tuple, tuple]:
        """Load and process image for both Qwen3-VL and SAM."""
        if not os.path.exists(image_path):
            # Try relative to base dir
            alt_path = os.path.join(self.base_image_dir, image_path)
            if os.path.exists(alt_path):
                image_path = alt_path
        
        result = self.image_processor.process_image(image_path, process_qwen=False,prepare_sam=prepare_sam)
        
        return (
            result["image"],
            result["sam_image"],
            result["original_size"],
            result["sam_input_size"],
        )
    
    def _build_conversation(
        self,
        question: str,
        answer: str = "",
        need_seg: bool = True,
    ) -> List[Dict]:
        """Build a conversation in Qwen3-VL chat format."""
        content = []
        
        # Add image placeholder
        content.append({"type": "image", "image": "placeholder"})
        
        # [SEG] belongs to the assistant target. Putting it in the user prompt
        # leaks the action label and makes inference depend on a manual marker.
        content.append({"type": "text", "text": question})
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        
        if answer:
            messages.append({"role": "assistant", "content": answer})
        
        return messages


class ReferSegDataset(BaseSEGAMEDataset):
    """
    Referring Expression Segmentation Dataset.
    
    Supports RefCOCO, RefCOCO+, RefCOCOg, and their
    false-premise variants (FP-RefCOCO*).
    """
    
    def __init__(
        self,
        base_image_dir: str,
        processor,
        samples_per_epoch: int = 4000,
        image_size: int = 1024,
        num_classes_per_sample: int = 1,
        dataset_names: str = "refcoco||refcoco+||refcocog",
        split: str = "train",
        default_exists: bool = True,
    ):
        super().__init__(base_image_dir, processor, samples_per_epoch, image_size, num_classes_per_sample)
        
        self.split = split
        self.default_exists = default_exists
        self.dataset_names = dataset_names.split("||")
        
        # Load annotations
        self.annotations = []
        self._load_all_annotations()
        
        print(f"[ReferSegDataset] Loaded {len(self.annotations)} annotations "
              f"from {dataset_names} ({split})")
    
    def _load_all_annotations(self):
        """Load annotations from all specified datasets."""
        for ds_name in self.dataset_names:
            ds_path = os.path.join(self.base_image_dir, ds_name)
            if not os.path.exists(ds_path):
                print(f"  [WARN] Dataset path not found: {ds_path}")
                continue
            
            # Try to load annotations
            ann_file = os.path.join(ds_path, f"{self.split}.json")
            if os.path.exists(ann_file):
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different annotation formats
                if isinstance(data, list):
                    self.annotations.extend({**ann, '_asset_root': self.base_image_dir} for ann in data)
                elif isinstance(data, dict) and "annotations" in data:
                    self.annotations.extend({**ann, '_asset_root': self.base_image_dir} for ann in data['annotations'])
                
                print(f"  Loaded {ds_name}/{self.split}.json: "
                      f"{len(data) if isinstance(data, list) else len(data.get('annotations', []))} annotations")
    
    def __getitem__(self, idx):
        """Get a single training sample."""
        if not self.annotations:
            raise RuntimeError(
                f"No annotations loaded for {self.dataset_names} ({self.split}) under {self.base_image_dir}"
            )
        # Random selection
        return self._get_annotation(self.annotations[np.random.randint(0, len(self.annotations))])

    def _get_annotation(self, ann):
        
        # Extract fields (adapt to your annotation format)
        image_path = ann.get("image_path", ann.get("file_name", ""))
        expression = ann.get("expression", ann.get("sent", ""))
        exists_value = ann.get("exists", self.default_exists)
        if isinstance(exists_value, str):
            exists = exists_value.strip().lower() in {"1", "true", "yes", "present", "exists"}
        else:
            exists = bool(exists_value)
        
        asset_root = ann.get('_asset_root', self.base_image_dir)
        if not os.path.isabs(image_path):
            image_path = os.path.join(asset_root, image_path)
        # Load image
        image, sam_image, original_size, sam_input_size = self._load_image(image_path)
        
        # Build question
        # The input distribution must not depend on the ground-truth answer.
        question = ann.get('question') or f"Locate the requested target if it is visible: {expression}."
        if exists:
            answer = "Present. [SEG]"
        else:
            answer = "Absent."
            correction = ann.get("correction", ann.get("suggested_expression", ""))
            if correction:
                answer += f" Visible alternative: {correction}."
        
        # Keep missing masks as None. False-premise examples supervise existence only.
        masks = None
        if exists and ann.get("mask_path"):
            mask_path = ann["mask_path"]
            if not os.path.isabs(mask_path):
                mask_path = os.path.join(asset_root, mask_path)
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
                mask = (mask > 128).astype(np.float32)
                masks = torch.from_numpy(mask).unsqueeze(0)
        if exists and masks is None:
            raise FileNotFoundError(f"Positive annotation {ann.get('id')} has no readable mask")
        ignore_mask = None
        if ann.get('ignore_mask_path'):
            path = ann['ignore_mask_path']
            path = path if os.path.isabs(path) else os.path.join(asset_root,path)
            ignore_mask = torch.from_numpy((np.asarray(Image.open(path).convert('L'))>128).astype(np.float32)).unsqueeze(0)
        if masks is not None and tuple(masks.shape[-2:]) != tuple(original_size):
            raise ValueError(f"Image/mask geometry mismatch: {image_path}")
        sam_mask_shape = [sam_input_size, original_size]
        conversations = self._build_conversation(question, answer, need_seg=False)
        
        return {
            "image_path": image_path,
            "image": image,
            "sam_image": sam_image,
            "conversations": conversations,
            "masks": masks,
            "sam_mask_shape": sam_mask_shape,
            "exists": exists,
            "ignore_mask": ignore_mask,
            "pair_id": ann.get('pair_id'),
            "pair_kind": ann.get('pair_kind'),
        }


class PairedReferSegDataset(ReferSegDataset):
    """Explicit paired manifests; the collator keeps both members in one batch."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        groups = {}
        for ann in self.annotations:
            if ann.get('pair_id') is None:
                raise ValueError('Paired datasets require pair_id on every annotation')
            groups.setdefault(str(ann['pair_id']), []).append(ann)
        self.pairs = []
        for pair_id, members in groups.items():
            if len(members) != 2:
                raise ValueError(f'Pair {pair_id} must have exactly two members')
            if members[0].get('pair_kind') in ('target_removed', 'irrelevant_change'):
                if members[0].get('expression') != members[1].get('expression'):
                    raise ValueError('Visual counterfactual pairs require an identical query')
            if members[0].get('pair_kind') == 'irrelevant_change' and members[0]['exists'] != members[1]['exists']:
                raise ValueError('Irrelevant changes must preserve target existence')
            self.pairs.append(members)

    def __getitem__(self, idx):
        pair = self.pairs[np.random.randint(len(self.pairs))]
        return [self._get_annotation(ann) for ann in pair]


class HybridDataset(torch.utils.data.Dataset):
    """
    Hybrid dataset that samples from multiple dataset types.
    
    Similar to SESAME's HybridDataset but adapted for Qwen3-VL.
    """
    
    def __init__(
        self,
        base_image_dir: str,
        processor,
        samples_per_epoch: int = 4000,
        image_size: int = 1024,
        num_classes_per_sample: int = 1,
        dataset: str = "refer_seg||correct_refer_seg||vqa||neg_refer_seg",
        sample_rate: list = [9, 3, 3, 3],
        dataset_specs: Optional[dict] = None,
    ):
        self.samples_per_epoch = samples_per_epoch
        if not sample_rate or sum(sample_rate) <= 0:
            raise ValueError("sample_rate must contain positive weights")
        self.sample_rate = np.array(sample_rate, dtype=np.float64) / sum(sample_rate)
        
        self.datasets = []
        dataset_types = dataset.split("||")
        if len(sample_rate) != len(dataset_types):
            raise ValueError(f"sample_rate has {len(sample_rate)} values for {len(dataset_types)} datasets")
        for i, ds_type in enumerate(dataset_types):
            if dataset_specs and ds_type in dataset_specs:
                spec = dataset_specs[ds_type]
                cls = PairedReferSegDataset if spec.get('paired') else ReferSegDataset
                self.datasets.append(cls(spec['root'],processor,samples_per_epoch,
                    image_size,num_classes_per_sample,dataset_names=spec['names'],split='train'))
                self.datasets[-1].eval_root = spec.get('eval_root',spec['root'])
            elif ds_type == "refer_seg":
                self.datasets.append(
                    ReferSegDataset(
                        base_image_dir, processor, samples_per_epoch,
                        image_size, num_classes_per_sample,
                        dataset_names="refcoco||refcoco+||refcocog",
                        split="train",
                    )
                )
            elif ds_type == "neg_refer_seg":
                self.datasets.append(
                    ReferSegDataset(
                        base_image_dir, processor, samples_per_epoch,
                        image_size, num_classes_per_sample,
                        dataset_names="R-refcoco||R-refcoco+||R-refcocog",
                        split="train", default_exists=False,
                    )
                )
            elif ds_type == "counterfactual_refer_seg":
                # Expects reviewed factual/counterfactual pairs under
                # dataset/counterfactual/{train,val}.json. False examples must
                # point to the edited image and set exists=false.
                self.datasets.append(
                    ReferSegDataset(
                        base_image_dir, processor, samples_per_epoch,
                        image_size, num_classes_per_sample,
                        dataset_names="counterfactual",
                        split="train", default_exists=False,
                    )
                )
            elif ds_type == "correct_refer_seg":
                self.datasets.append(
                    ReferSegDataset(
                        base_image_dir, processor, samples_per_epoch,
                        image_size, num_classes_per_sample,
                        dataset_names="fprefcoco||fprefcoco+||fprefcocog",
                        split="train",
                    )
                )
            elif ds_type == "vqa":
                # VQA dataset - no segmentation needed
                self.datasets.append(
                    VQADataset(
                        base_image_dir, processor, samples_per_epoch,
                        image_size, split="train",
                    )
                )
            else:
                raise ValueError(f"Unknown dataset type: {ds_type}")

        empty = [kind for kind, ds in zip(dataset_types, self.datasets)
                 if (hasattr(ds, "annotations") and not ds.annotations)
                 or (hasattr(ds, "vqa_data") and not ds.vqa_data)]
        if empty:
            raise ValueError(f"No annotations found for requested datasets: {empty}; check --dataset_dir")
        if not self.datasets:
            raise ValueError("No valid datasets were configured")
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # Sample a task, then let that dataset choose a valid annotation.
        ds_idx = np.random.choice(len(self.datasets), p=self.sample_rate)
        return self.datasets[ds_idx][idx % len(self.datasets[ds_idx])]


class VQADataset(BaseSEGAMEDataset):
    """Simple VQA dataset for training general VL capabilities."""
    
    def __init__(
        self,
        base_image_dir: str,
        processor,
        samples_per_epoch: int = 4000,
        image_size: int = 1024,
        split: str = "train",
    ):
        super().__init__(base_image_dir, processor, samples_per_epoch, image_size)
        self.split = split
        self.vqa_data = []
        self._load_vqa_data()
    
    def _load_vqa_data(self):
        """Load VQA annotations."""
        vqa_path = os.path.join(self.base_image_dir, "llava_instruct_150k")
        if os.path.exists(vqa_path):
            for fname in ["llava_instruct_150k.json"]:
                fpath = os.path.join(vqa_path, fname)
                if os.path.exists(fpath):
                    with open(fpath, 'r', encoding='utf-8') as f:
                        self.vqa_data = json.load(f)
                    print(f"  Loaded VQA data: {len(self.vqa_data)} samples")
                    break
    
    def __getitem__(self, idx):
        if not self.vqa_data:
            raise RuntimeError(
                f"No VQA annotations found under {self.base_image_dir}/llava_instruct_150k"
            )
        
        sample = self.vqa_data[np.random.randint(0, len(self.vqa_data))]
        
        image_path = sample.get("image", "")
        conversations = sample.get("conversations", [])
        
        # Process conversations for Qwen3-VL format
        qwen_convs = []
        qwen_convs.append({"role": "system", "content": self.system_prompt})
        
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            content = turn["value"]
            # Replace image tokens
            content = content.replace("<image>", "")
            qwen_convs.append({"role": role, "content": content})
        
        # Add image to first user message
        for msg in qwen_convs:
            if msg["role"] == "user":
                msg["content"] = [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": msg["content"]},
                ]
                break
        
        # Load image
        image, sam_image, original_size, sam_input_size = self._load_image(image_path,prepare_sam=False)
        
        return {
            "image_path": image_path,
            "image": image,
            "sam_image": sam_image,
            "conversations": qwen_convs,
            "masks": None,
            "sam_mask_shape": [sam_input_size, original_size],
            "exists": None,
        }
