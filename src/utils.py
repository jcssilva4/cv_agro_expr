"""
src/utils.py
────────────
Visualisation helpers and Grad-CAM implementation.

Functions:
  denormalize          — ImageNet tensor → uint8 numpy image
  show_sample_grid     — Grid of images per class
  plot_class_dist      — Horizontal bar chart of class frequencies
  plot_training_curves — Loss/accuracy over epochs
  overlay_mask         — Colour mask blended on top of a PIL image
  plot_segmentation    — 4-panel segmentation results figure
  compute_gradcam      — Raw Grad-CAM heatmap computation
  apply_gradcam        — Overlay heatmap on an image
  plot_gradcam_grid    — Multi-image Grad-CAM comparison
"""

from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Tensor helpers
# ──────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalised tensor [3,H,W] → uint8 numpy [H,W,3]."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * _STD + _MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset visualisation
# ──────────────────────────────────────────────────────────────────────────────

def show_sample_grid(
    dataset,
    class_names: List[str],
    n_per_class: int = 3,
    max_classes: int = 10,
    figsize: tuple = (20, 16),
) -> plt.Figure:
    """Display a grid of sample images (rows = classes, cols = samples)."""
    n_cls = min(max_classes, len(class_names))

    # Collect n_per_class indices per class
    idx_by_class: Dict[int, list] = {i: [] for i in range(n_cls)}
    for tensor, lbl in dataset:
        if lbl < n_cls and len(idx_by_class[lbl]) < n_per_class:
            idx_by_class[lbl].append(tensor)
        if all(len(v) == n_per_class for v in idx_by_class.values()):
            break

    fig, axes = plt.subplots(n_cls, n_per_class, figsize=figsize)
    for row in range(n_cls):
        for col, tensor in enumerate(idx_by_class.get(row, [])):
            ax = axes[row][col]
            ax.imshow(denormalize(tensor))
            ax.axis("off")
            if col == 0:
                short = class_names[row].replace("___", "\n")
                ax.set_ylabel(short, fontsize=6, rotation=0,
                              labelpad=80, va="center")

    fig.suptitle("PlantVillage — Sample Images per Class", fontsize=12)
    plt.tight_layout()
    return fig


def plot_class_dist(
    class_names: List[str],
    counts: List[int],
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Horizontal bar chart of per-class image counts."""
    fig, ax = plt.subplots(figsize=figsize)
    colours = plt.cm.tab20(np.linspace(0, 1, len(class_names)))  # type: ignore[attr-defined]
    bars = ax.barh(class_names, counts, color=colours)
    ax.set_xlabel("Number of images", fontsize=11)
    ax.set_title("PlantVillage — Class Distribution", fontsize=13)
    ax.bar_label(bars, padding=3, fontsize=7)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Training curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: Dict[str, List[float]]) -> plt.Figure:
    """Plot train/val loss and accuracy curves from a history dict."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train loss",  markersize=4)
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val loss",    markersize=4)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train acc", markersize=4)
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val acc",   markersize=4)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Training Curves — EfficientNet-B0 on PlantVillage", fontsize=12)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation visualisation
# ──────────────────────────────────────────────────────────────────────────────

def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple = (0, 200, 0),
    alpha: float = 0.40,
) -> np.ndarray:
    """Blend a binary mask over a PIL image. Returns uint8 RGB numpy array."""
    img_arr  = np.array(image.convert("RGB")).copy()
    coloured = img_arr.copy()
    coloured[mask > 0] = color
    return cv2.addWeighted(img_arr, 1 - alpha, coloured, alpha, 0)


def plot_segmentation(
    color_img: Image.Image,
    seg_img: Optional[Image.Image],
    leaf_mask: np.ndarray,
    disease_stats: Dict,
    title: str = "",
) -> plt.Figure:
    """
    4-panel figure:
      (1) Original colour image
      (2) Pre-segmented image (or blank if not provided)
      (3) Leaf mask overlaid on the colour image
      (4) Pie chart — healthy vs diseased tissue
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(color_img)
    axes[0].set_title("Original Colour")
    axes[0].axis("off")

    if seg_img is not None:
        axes[1].imshow(seg_img)
        axes[1].set_title("Pre-Segmented")
    else:
        axes[1].text(0.5, 0.5, "N/A", ha="center", va="center")
    axes[1].axis("off")

    overlay = overlay_mask(color_img, leaf_mask)
    total   = disease_stats.get("total_px", disease_stats.get("leaf_area_px", 0))
    axes[2].imshow(overlay)
    axes[2].set_title(f"Leaf Mask\nArea: {total:,} px")
    axes[2].axis("off")

    # Pie chart
    dpct = disease_stats.get("disease_pct", 0.0)
    hpct = 100 - dpct
    axes[3].pie(
        [hpct, dpct],
        labels=[f"Healthy\n{hpct:.1f}%", f"Affected\n{dpct:.1f}%"],
        colors=["#4CAF50", "#F44336"],
        startangle=90,
    )
    axes[3].set_title("Tissue Composition")

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────

def compute_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: nn.Module,
    class_idx: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap (float32, values in [0, 1]) for one image.

    Args:
        model:        Trained classification model (eval mode recommended).
        image_tensor: Normalised tensor [3, H, W] (no batch dimension).
        target_layer: The convolutional layer to hook (e.g. model.blocks[-1]).
        class_idx:    Target class index; if None, uses the predicted class.
        device:       'cuda' or 'cpu'.

    Returns:
        Heatmap as a float32 numpy array [h_feat, w_feat] in [0, 1].
    """
    model.eval()
    inp = image_tensor.unsqueeze(0).to(device)

    gradients:  List[torch.Tensor] = []
    activations: List[torch.Tensor] = []

    def _fwd_hook(_module, _inp, output):
        activations.append(output)
        output.register_hook(gradients.append)

    handle = target_layer.register_forward_hook(_fwd_hook)
    output = model(inp)

    if class_idx is None:
        class_idx = int(output.argmax(dim=1).item())

    model.zero_grad()
    output[0, class_idx].backward()
    handle.remove()

    grads    = gradients[0].cpu().numpy()[0]   # [C, h, w]
    acts     = activations[0].detach().cpu().numpy()[0]  # [C, h, w]
    weights  = grads.mean(axis=(1, 2))          # GAP over spatial dims

    heatmap  = np.sum(weights[:, None, None] * acts, axis=0)
    heatmap  = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)
    return heatmap.astype(np.float32)


def apply_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.50,
) -> np.ndarray:
    """Resize heatmap to image size and blend as a JET colour overlay."""
    h, w   = original_image.shape[:2]
    resized = cv2.resize(heatmap, (w, h))
    jet     = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_image, 1 - alpha, jet_rgb, alpha, 0)


def plot_gradcam_grid(
    tensors: List[torch.Tensor],
    heatmaps: List[np.ndarray],
    predictions: List[str],
    ground_truths: List[str],
    figsize: tuple = (20, 5),
) -> plt.Figure:
    """
    Side-by-side grid: original | Grad-CAM overlay for each sample.
    """
    n = len(tensors)
    fig, axes = plt.subplots(2, n, figsize=figsize)

    for i in range(n):
        orig = denormalize(tensors[i])

        axes[0][i].imshow(orig)
        axes[0][i].axis("off")
        axes[0][i].set_title(f"GT: {ground_truths[i]}", fontsize=7)

        overlay = apply_gradcam(orig, heatmaps[i])
        axes[1][i].imshow(overlay)
        axes[1][i].axis("off")
        axes[1][i].set_title(f"Pred: {predictions[i]}", fontsize=7)

    axes[0][0].set_ylabel("Original",   fontsize=9, rotation=90)
    axes[1][0].set_ylabel("Grad-CAM",   fontsize=9, rotation=90)
    plt.suptitle("Grad-CAM Visualisation", fontsize=12)
    plt.tight_layout()
    return fig
