"""
ViT-B/16 loading and LoRA adapter attachment for TreeLoRA vision experiments.

The HuggingFace ViT stores Q, K, V as separate nn.Linear modules named
'query', 'key', 'value' — the same convention as BERT — so the single entry
    'vit': ['query', 'value']
added to TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING in
utils/my_peft/utils/other.py is sufficient to wire ViT into get_peft_model
with no other changes to the PEFT infrastructure.

The resulting LoRA structure per adapted layer is identical to the LLM case:
    frozen   lora_A / lora_B      [r_sum × d]  historical tasks
    trainable  loranew_A / loranew_B  [r × d]  current task

    y = W x  +  B_hist A_hist x  +  B_new A_new x    (scaled by alpha / r)

The KD_LoRA_Tree collects all parameters whose name contains 'loranew_A',
which is satisfied by both query and value adapters in every block, so the
tree machinery requires no modification.

lora_depth semantics for ViT:
    Each transformer block contributes 2 replacements (query + value).
    lora_depth=-1   all 12 blocks  (24 replacements)
    lora_depth=16   8 blocks
    lora_depth=4    2 blocks
"""

import os
import sys

from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.my_peft import get_peft_model, LoraConfig


def load_vit_b16(checkpoint_path):
    """
    Load the Google ViT-B/16 pretrained on ImageNet-21k from a local directory.

    Args:
        checkpoint_path: path to the downloaded model directory,
                         e.g. 'PTM/vit-base-patch16-224-in21k'

    Returns:
        model: HuggingFace ViTModel with all parameters frozen
    """
    model = AutoModel.from_pretrained(checkpoint_path)
    for p in model.parameters():
        p.requires_grad = False
    return model


def apply_lora_to_vit(model, r=8, lora_alpha=32, lora_dropout=0.1,
                       r_sum=0, lora_depth=-1):
    """
    Apply dual-adapter LoRA to the query and value projections of the ViT.

    Uses get_peft_model with no task_type so the returned PeftModel delegates
    its forward() transparently to the underlying ViT (pixel_values, etc.),
    with no LLM-specific wrapping.

    Args:
        model:       ViTModel from load_vit_b16
        r:           rank of the trainable (current-task) adapter
        lora_alpha:  LoRA scaling factor; effective scale = alpha / r
        lora_dropout: dropout probability on the LoRA path
        r_sum:       rank of the frozen (historical) adapter; 0 on task 0
        lora_depth:  max number of Linear replacements; -1 = all

    Returns:
        peft_model: PeftModel wrapping the ViT with LoRA applied in-place
    """
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r_sum=r_sum,
    )
    return get_peft_model(model, peft_config, depth=lora_depth)


def set_lora_trainable(model):
    """
    Freeze all parameters except the current-task LoRA matrices (loranew_A/B).

    Call this after apply_lora_to_vit.  The frozen lora_A/B (historical tasks)
    and all backbone weights remain unchanged.
    """
    for name, param in model.named_parameters():
        param.requires_grad = 'loranew_' in name

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[set_lora_trainable] {n_train:,} / {n_total:,} parameters trainable "
          f"({100 * n_train / n_total:.3f}%)")
    return model


def build_treelora_vit(checkpoint_path, r=8, lora_alpha=32, lora_dropout=0.1,
                        r_sum=0, lora_depth=-1):
    """
    Load iBOT ViT-B/16, apply LoRA to Q and V, and set trainable params.

    Convenience wrapper around load_vit_b16 → apply_lora_to_vit →
    set_lora_trainable for use in training scripts.

    Example (task 0):
        model = build_treelora_vit('PTM/vit-base-patch16-224-in21k')

    Example (task t > 0, after loading the saved adapter from task t-1):
        model = build_treelora_vit('PTM/vit-base-patch16-224-in21k', r_sum=8)
    """
    model = load_vit_b16(checkpoint_path)
    model = apply_lora_to_vit(model, r, lora_alpha, lora_dropout, r_sum, lora_depth)
    model = set_lora_trainable(model)
    return model
