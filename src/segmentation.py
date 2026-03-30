"""
src/segmentation.py
───────────────────
Classical computer-vision routines for:
  1. Leaf segmentation from colour images (HSV thresholding + morphology).
  2. Leaf mask extraction from pre-segmented PlantVillage images.
  3. Leaf area quantification (pixel count + contour shape statistics).
  4. Disease area estimation (healthy green tissue vs total leaf area).

All functions accept PIL Images and return numpy arrays or plain dicts.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image → BGR numpy array (OpenCV convention)."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    if num_labels <= 1:
        return binary_mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Leaf segmentation — Method A: from colour image (HSV thresholding)
# ──────────────────────────────────────────────────────────────────────────────

def segment_leaf_from_color(image: Image.Image) -> np.ndarray:
    """
    Segment the leaf region from a colour image using HSV thresholding.

    Covers the full leaf colour spectrum (healthy green → diseased
    yellow/brown), then applies morphological operations to fill holes
    and remove noise, keeping the largest connected component.

    Returns:
        Binary mask (uint8) — 255 for leaf pixels, 0 for background.
    """
    bgr = _pil_to_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Broad hue range: yellow-green through dark green,
    # with minimum saturation/value to exclude white/grey backgrounds.
    lower = np.array([10,  25,  25])
    upper = np.array([90, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

    return _largest_component(mask)


# ──────────────────────────────────────────────────────────────────────────────
# Leaf segmentation — Method B: from pre-segmented PlantVillage image
# ──────────────────────────────────────────────────────────────────────────────

def segment_leaf_from_segmented(image: Image.Image) -> np.ndarray:
    """
    Extract the leaf mask from a pre-segmented (background-removed) image.
    The PlantVillage 'segmented' config stores images where the background
    is pure black — any pixel with max-channel value > 20 is leaf tissue.

    Returns:
        Binary mask (uint8) — 255 for leaf pixels, 0 for background.
    """
    arr  = np.array(image.convert("RGB"))
    mask = (arr.max(axis=2) > 20).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return _largest_component(mask)


# ──────────────────────────────────────────────────────────────────────────────
# Leaf area measurement
# ──────────────────────────────────────────────────────────────────────────────

def compute_leaf_area(mask: np.ndarray) -> int:
    """Return the leaf area in pixels (non-zero pixel count)."""
    return int(np.count_nonzero(mask))


def get_contour_stats(mask: np.ndarray) -> Dict:
    """
    Compute shape statistics from the largest contour of a leaf mask.

    Returns dict with:
        contour_area, perimeter, bounding_box (x,y,w,h),
        circularity, aspect_ratio.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}

    c         = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)

    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    return {
        "contour_area": float(area),
        "perimeter":    float(perimeter),
        "bounding_box": (int(x), int(y), int(w), int(h)),
        "circularity":  round(float(circularity), 4),
        "aspect_ratio": round(float(w) / float(h), 4) if h > 0 else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Disease area estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_disease_area(
    color_image: Image.Image,
    segmented_image: Image.Image,
) -> Dict:
    """
    Estimate what percentage of the leaf area shows disease symptoms.

    Strategy:
      - Total leaf area  → segment_leaf_from_segmented (precise boundary).
      - Healthy (green) → narrow HSV green mask applied inside the leaf region.
      - Diseased area   → total − healthy (captures yellowing, brown spots, etc.).

    Args:
        color_image:     Original RGB leaf image.
        segmented_image: Corresponding background-removed image from the
                         PlantVillage 'segmented' config.

    Returns:
        Dict with keys: total_px, healthy_px, diseased_px, disease_pct.
    """
    total_mask = segment_leaf_from_segmented(segmented_image)
    total_px   = compute_leaf_area(total_mask)

    if total_px == 0:
        return {"total_px": 0, "healthy_px": 0, "diseased_px": 0, "disease_pct": 0.0}

    bgr = _pil_to_bgr(color_image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Narrow "healthy green" range — excludes yellowing and brown lesions
    lower_healthy = np.array([30, 50, 40])
    upper_healthy = np.array([85, 255, 255])
    healthy_mask  = cv2.inRange(hsv, lower_healthy, upper_healthy)

    # Restrict healthy mask to actual leaf pixels
    healthy_in_leaf = cv2.bitwise_and(healthy_mask, healthy_mask, mask=total_mask)
    healthy_px  = compute_leaf_area(healthy_in_leaf)
    diseased_px = max(0, total_px - healthy_px)
    disease_pct = round(100.0 * diseased_px / total_px, 2)

    return {
        "total_px":    total_px,
        "healthy_px":  healthy_px,
        "diseased_px": diseased_px,
        "disease_pct": disease_pct,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Complete analysis pipeline (single-call convenience)
# ──────────────────────────────────────────────────────────────────────────────

def analyse_leaf(
    color_image: Image.Image,
    segmented_image: Optional[Image.Image] = None,
) -> Dict:
    """
    Run the full segmentation + area analysis pipeline on one leaf image.

    Returns a unified result dict with all available measurements.
    """
    color_mask    = segment_leaf_from_color(color_image)
    color_area    = compute_leaf_area(color_mask)
    contour_stats = get_contour_stats(color_mask)

    result = {
        "color_mask":      color_mask,
        "leaf_area_px":    color_area,
        "contour_stats":   contour_stats,
    }

    if segmented_image is not None:
        seg_mask          = segment_leaf_from_segmented(segmented_image)
        disease_stats     = estimate_disease_area(color_image, segmented_image)
        result["seg_mask"]        = seg_mask
        result["disease_analysis"] = disease_stats

    return result
