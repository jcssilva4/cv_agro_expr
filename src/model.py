"""
src/model.py
────────────
EfficientNet-B0 transfer learning model built with the `timm` library.

Supports two-phase training:
  Phase 1 — freeze backbone, train only the classifier head.
  Phase 2 — unfreeze all parameters for full fine-tuning.

Usage:
    from src.model import build_model, freeze_backbone, unfreeze_all
    model = build_model(num_classes=38)
    freeze_backbone(model)          # Phase 1
    ...
    unfreeze_all(model)             # Phase 2
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    num_classes: int,
    model_name: str = "efficientnet_b0",
    pretrained: bool = True,
    drop_rate: float = 0.3,
) -> nn.Module:
    """
    Create an EfficientNet-B0 (or any timm model) with a custom head.

    Args:
        num_classes: Number of output classes (38 for PlantVillage).
        model_name:  Any architecture available in the timm registry.
        pretrained:  Load ImageNet pre-trained weights.
        drop_rate:   Dropout probability on the classifier head.

    Returns:
        A ready-to-train nn.Module.
    """
    import timm  # lazy import

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Phase-control helpers
# ──────────────────────────────────────────────────────────────────────────────

def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all parameters except the final classifier head.
    Works for timm EfficientNet where the head is named 'classifier'.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Backbone frozen — trainable params: {trainable:,}")


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze every parameter for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] All layers unfrozen — trainable params: {trainable:,}")


def count_parameters(model: nn.Module):
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM target-layer resolver
# ──────────────────────────────────────────────────────────────────────────────

def get_gradcam_target_layer(model: nn.Module, model_name: str = "efficientnet_b0"):
    """
    Return the last convolutional block suited for Grad-CAM visualisation.
    Adjust for the architecture in use.
    """
    if "efficientnet" in model_name:
        # timm's EfficientNet: last block of the feature extractor
        return model.blocks[-1]
    if "resnet" in model_name:
        return model.layer4[-1]
    # Generic fallback: last Conv2d layer
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv
