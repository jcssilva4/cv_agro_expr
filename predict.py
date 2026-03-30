"""
predict.py
──────────
Inference script: classify a single plant leaf image and quantify its leaf
area / disease extent.

Usage:
    # Classification + segmentation (needs a trained model)
    python predict.py --image path/to/leaf.jpg

    # With the matching pre-segmented image for disease-area estimation
    python predict.py --image path/to/leaf_color.jpg \\
                      --segmented path/to/leaf_seg.jpg

    # Segmentation only (no trained model required)
    python predict.py --image path/to/leaf.jpg --no-classify

    # Point to a different run directory
    python predict.py --image path/to/leaf.jpg --model-dir runs/my_exp
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import build_model
from src.segmentation import (
    analyse_leaf,
    segment_leaf_from_color,
    compute_leaf_area,
    get_contour_stats,
)

# ──────────────────────────────────────────────────────────────────────────────
# Transforms (same as test split in dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_classifier(model_dir: str, device: torch.device):
    """Load a trained model + class names from a run directory."""
    names_path = os.path.join(model_dir, "class_names.json")
    ckpt_path  = os.path.join(model_dir, "best_model.pth")

    if not os.path.exists(names_path):
        raise FileNotFoundError(
            f"class_names.json not found in '{model_dir}'. "
            "Run train.py first to create a checkpoint."
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"best_model.pth not found in '{model_dir}'. "
            "Run train.py first to create a checkpoint."
        )

    with open(names_path) as f:
        class_names = json.load(f)

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model, class_names


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def classify_image(model, image: Image.Image, class_names, device, top_k: int = 5):
    """Return the top-k predicted classes with confidence scores."""
    tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    top_probs, top_idx = probs.topk(min(top_k, len(class_names)))
    return [
        {"rank": i + 1, "class": class_names[idx], "confidence": round(p.item(), 4)}
        for i, (idx, p) in enumerate(zip(top_idx.cpu(), top_probs.cpu()))
    ]


def _print_classification(predictions):
    print("\n╔══ Disease Classification ════════════════════════╗")
    for pred in predictions:
        bar = "█" * int(pred["confidence"] * 30)
        print(f"  {pred['rank']}. {pred['class']:<45} {pred['confidence']*100:5.1f}%  {bar}")
    print("╚═══════════════════════════════════════════════════╝")


def _print_leaf_analysis(result):
    print("\n╔══ Leaf Area Analysis ═════════════════════════════╗")
    print(f"  Leaf area (colour-based mask): {result['leaf_area_px']:>10,} px")

    cs = result.get("contour_stats", {})
    if cs:
        print(f"  Circularity:                   {cs.get('circularity', 'N/A'):>10}")
        print(f"  Aspect ratio (w/h):            {cs.get('aspect_ratio', 'N/A'):>10}")
        bb = cs.get("bounding_box")
        if bb:
            print(f"  Bounding box (x,y,w,h):        {bb}")

    da = result.get("disease_analysis", {})
    if da:
        print(f"\n  ── Disease Extent (pre-segmented mask) ──")
        print(f"  Total leaf area:  {da['total_px']:>10,} px")
        print(f"  Healthy tissue:   {da['healthy_px']:>10,} px  ({100 - da['disease_pct']:.1f}%)")
        print(f"  Affected area:    {da['diseased_px']:>10,} px  ({da['disease_pct']:.1f}%)")

    print("╚═══════════════════════════════════════════════════╝")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PlantVillage leaf analyser")
    p.add_argument("--image",        required=True,
                   help="Path to the colour leaf image")
    p.add_argument("--segmented",    default=None,
                   help="Path to the background-removed version (optional)")
    p.add_argument("--model-dir",    default="runs/exp1",
                   help="Directory containing best_model.pth + class_names.json")
    p.add_argument("--top-k",        type=int, default=5,
                   help="Number of top predictions to display")
    p.add_argument("--no-classify",  action="store_true",
                   help="Skip classification (run segmentation only)")
    return p.parse_args()


def main():
    args = parse_args()

    color_image = Image.open(args.image).convert("RGB")
    seg_image   = Image.open(args.segmented).convert("RGB") if args.segmented else None
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Classification ────────────────────────────────────────────────────────
    if not args.no_classify:
        model, class_names = load_classifier(args.model_dir, device)
        predictions = classify_image(model, color_image, class_names, device, args.top_k)
        _print_classification(predictions)

    # ── Leaf area / segmentation ──────────────────────────────────────────────
    result = analyse_leaf(color_image, segmented_image=seg_image)
    _print_leaf_analysis(result)


if __name__ == "__main__":
    main()
