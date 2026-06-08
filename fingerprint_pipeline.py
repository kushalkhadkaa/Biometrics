from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, restoration, util


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class FingerprintConfig:
    """Configuration parameters for the conservative forensic fingerprint pipeline.
    
    Provides control over thresholds and kernel sizes across normalization, 
    segmentation, artifact masking, inpainting, and Gabor filtering.
    """

    input_dirs: tuple[str, ...] = ("utils/image", "utils")
    output_dir: str = "outputs"
    prefer_dark_ridges: bool = True
    auto_invert: bool = True
    clip_percentiles: tuple[float, float] = (1.0, 99.0)
    illumination_sigma: float = 26.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    segmentation_block_size: int = 23
    segmentation_variance_weight: float = 0.72
    segmentation_gradient_weight: float = 0.28
    segmentation_min_area_ratio: float = 0.002
    segmentation_close_radius: int = 9
    segmentation_open_radius: int = 3
    denoise_median_size: int = 3
    denoise_bilateral_diameter: int = 5
    denoise_bilateral_sigma_color: float = 35.0
    denoise_bilateral_sigma_space: float = 5.0
    line_hough_threshold: int = 80
    line_min_length_ratio: float = 0.34
    line_max_gap: int = 12
    line_mask_width: int = 5
    color_artifact_saturation: int = 55
    color_artifact_value_delta: int = 30
    dark_component_min_area: int = 20
    dark_component_max_area_ratio: float = 0.018
    remove_ambiguous_dark_inside_roi: bool = False
    artifact_mask_dilate: int = 1
    max_inpaint_component_area: int = 1600
    max_inpaint_component_width: int = 48
    max_inpaint_component_height: int = 48
    min_inpaint_support_ratio: float = 0.55
    inpaint_radius: int = 3
    ridge_period_px: float = 9.0
    gabor_orientations: int = 12
    gabor_kernel_size: int = 21
    gabor_blend: float = 0.32
    suppress_background: bool = True
    include_review_artifacts_in_reconstruction: bool = False
    force_reconstruct_artifacts: bool = False
    post_denoise_strength: float = 0.0
    final_cleanup_dilate: int = 0
    final_cleanup_radius: int = 0
    manual_rectangles: list[tuple[int, int, int, int]] = field(default_factory=list)
    manual_lines: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    manual_polygons: list[list[tuple[int, int]]] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Structure housing outputs and intermediate stages of the preprocessing run.
    
    Attributes:
        image_path: Source file path.
        config: The configuration used.
        stages: Dictionary mapping stage names (e.g. '01_normalized') to image arrays.
        masks: Map of detected masks (e.g. 'roi', 'artifact_remove').
        metrics: Quality and cleanup metrics.
        warnings: Warnings generated during the process.
    """
    image_path: Path
    config: FingerprintConfig
    stages: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    metrics: dict[str, float | int | str]
    warnings: list[str]


def discover_images(input_dirs: Iterable[str | Path] = ("utils/image", "utils")) -> list[Path]:
    """Scans configured directories for files with supported image extensions.

    Args:
        input_dirs: Iterable of directory paths to scan.

    Returns:
        Sorted list of unique Path objects pointing to detected images.
    """
    paths: list[Path] = []
    for directory in input_dirs:
        base = Path(directory)
        if not base.exists():
            continue
        paths.extend(p for p in base.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return sorted(dict.fromkeys(paths))


def load_rgb(path: str | Path, apply_exif_orientation: bool = True) -> np.ndarray:
    """Loads an image file as a NumPy array in RGB format.

    Corrects orientation issues automatically based on EXIF tags if requested.

    Args:
        path: Path to the image file.
        apply_exif_orientation: If True, transposes the image based on EXIF tag metadata.

    Returns:
        NumPy uint8 array of shape (H, W, 3).
    """
    image = Image.open(path).convert("RGB")
    if apply_exif_orientation:
        image = ImageOps.exif_transpose(image)
    return np.asarray(image)


def rgb_to_gray_float(rgb: np.ndarray) -> np.ndarray:
    """Converts an RGB image array to a normalized single-channel floating point array.

    Args:
        rgb: NumPy uint8 array of shape (H, W, 3).

    Returns:
        Single-channel float32 array in range [0.0, 1.0].
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return util.img_as_float32(gray)


def as_uint8(image: np.ndarray) -> np.ndarray:
    """Clips and casts a floating-point image array to 8-bit unsigned integer.

    Args:
        image: Single-channel or multi-channel float32 array in range [0.0, 1.0].

    Returns:
        NumPy uint8 array in range [0, 255].
    """
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def robust_rescale(gray: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    """Rescales image intensity using robust percentile bounds to handle outliers.

    Args:
        gray: Single-channel float32 image array.
        percentiles: Tuple of low and high percentiles (e.g. (1.0, 99.0)) to bound the range.

    Returns:
        Intensity-rescaled float32 image array.
    """
    lo, hi = np.percentile(gray[np.isfinite(gray)], percentiles)
    if hi <= lo:
        return np.clip(gray, 0.0, 1.0)
    return exposure.rescale_intensity(gray, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)


def maybe_invert(gray: np.ndarray, prefer_dark_ridges: bool = True) -> np.ndarray:
    """Inverts image polarity automatically if background and center median checks suggest it.

    In forensic science, fingerprints are standardly represented as dark ridges 
    on a light background. This function ensures consistency.

    Args:
        gray: Single-channel float32 image array.
        prefer_dark_ridges: If True, forces dark ridges on light background representation.

    Returns:
        Polarity-corrected float32 image array.
    """
    if not prefer_dark_ridges:
        return 1.0 - gray

    edge = max(8, min(gray.shape) // 16)
    border = np.concatenate(
        [
            gray[:edge, :].ravel(),
            gray[-edge:, :].ravel(),
            gray[:, :edge].ravel(),
            gray[:, -edge:].ravel(),
        ]
    )
    border_median = float(np.median(border))
    center = gray[edge:-edge, edge:-edge] if gray.shape[0] > 2 * edge and gray.shape[1] > 2 * edge else gray
    center_median = float(np.median(center))

    # If the border/background is dark and the fingerprint is comparatively light, invert.
    if border_median < 0.42 and center_median > border_median + 0.08:
        return 1.0 - gray
    return gray


def correct_illumination(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Applies a high-pass Gaussian filter to correct uneven background illumination.

    Subtracts low-frequency illumination variations estimated via a large Gaussian kernel.

    Args:
        gray: Single-channel float32 image array.
        sigma: Standard deviation of the Gaussian filter kernel.

    Returns:
        Corrected and normalized float32 image array.
    """
    background = filters.gaussian(gray, sigma=sigma, preserve_range=True)
    corrected = gray - background + float(np.median(background))
    return robust_rescale(corrected, (1.0, 99.0))


def clahe(gray: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Enhances local contrast using Contrast Limited Adaptive Histogram Equalization.

    Args:
        gray: Single-channel float32 image.
        clip_limit: Threshold for contrast limiting.
        tile_grid: Grid size for histogram contextual regions (e.g. 8 for an 8x8 grid).

    Returns:
        Contrast-enhanced float32 image.
    """
    tile = max(2, int(tile_grid))
    clahe_op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile, tile))
    return clahe_op.apply(as_uint8(gray)).astype(np.float32) / 255.0


def local_std(gray: np.ndarray, block_size: int) -> np.ndarray:
    """Calculates local standard deviation to assess texture strength.

    Args:
        gray: Single-channel float32 image.
        block_size: Window size for local calculation.

    Returns:
        Standard deviation map (float32).
    """
    block = max(3, int(block_size) | 1)
    mean = cv2.blur(gray.astype(np.float32), (block, block))
    mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (block, block))
    variance = np.maximum(mean_sq - mean**2, 0)
    return np.sqrt(variance)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Computes the magnitude of the image gradient using Sobel operators.

    Args:
        gray: Single-channel float32 image.

    Returns:
        Rescaled gradient magnitude map (float32).
    """
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return robust_rescale(mag, (1.0, 99.5))


def otsu_mask(score: np.ndarray) -> np.ndarray:
    """Creates a binary mask using Otsu global automatic thresholding.

    Args:
        score: Score/metric array.

    Returns:
        Boolean mask array of the same shape as input.
    """
    finite = score[np.isfinite(score)]
    if finite.size == 0 or np.ptp(finite) < 1e-6:
        return np.zeros(score.shape, dtype=bool)
    threshold = filters.threshold_otsu(finite)
    return score > threshold


def keep_large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Removes small connected components (noise) below a minimum area threshold.

    Args:
        mask: Boolean mask array.
        min_area: Minimum pixel area required to preserve a component.

    Returns:
        Filtered boolean mask array.
    """
    labeled = measure.label(mask)
    cleaned = np.zeros(mask.shape, dtype=bool)
    for region in measure.regionprops(labeled):
        if region.area >= min_area:
            cleaned[labeled == region.label] = True
    return cleaned


def segment_fingerprint(gray: np.ndarray, cfg: FingerprintConfig) -> tuple[np.ndarray, np.ndarray]:
    """Segments the fingerprint Region of Interest (ROI) from background noise.

    Combines texture variance and Sobel gradient magnitude scores to construct
    a robust classification map, which is then refined via morphological operations
    and convex hull modeling.

    Args:
        gray: Preprocessed single-channel float32 image.
        cfg: Configuration parameters defining block sizes, weights, and opening radii.

    Returns:
        Tuple:
            - Boolean mask indicating the segmented fingerprint ROI.
            - Rescaled quality score map (float32).
    """

    texture = robust_rescale(local_std(gray, cfg.segmentation_block_size), (2.0, 99.0))
    grad = gradient_magnitude(gray)
    score = cfg.segmentation_variance_weight * texture + cfg.segmentation_gradient_weight * grad
    mask = otsu_mask(score)

    open_radius = max(0, int(cfg.segmentation_open_radius))
    close_radius = max(0, int(cfg.segmentation_close_radius))
    if open_radius:
        mask = morphology.opening(mask, morphology.disk(open_radius))
    if close_radius:
        mask = morphology.closing(mask, morphology.disk(close_radius))
    mask = ndi.binary_fill_holes(mask)

    min_area = max(24, int(mask.size * cfg.segmentation_min_area_ratio))
    mask = keep_large_components(mask, min_area)
    if mask.any():
        mask = morphology.convex_hull_image(mask) & ndi.binary_dilation(mask, iterations=2)
    return mask.astype(bool), robust_rescale(score, (1.0, 99.0))


def ridge_orientation(gray: np.ndarray, sigma: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Estimates local ridge orientation and coherence from the structure tensor.

    The structure tensor J = [ [gxx, gxy], [gxy, gyy] ] characterizes the dominant 
    gradient directions. The orientation is computed perpendicular to the gradient.
    Coherence measures the local anisotropy (strength of alignment).

    Args:
        gray: Preprocessed single-channel float32 image.
        sigma: Standard deviation for Gaussian smoothing of the structure tensor components.

    Returns:
        Tuple:
            - local ridge orientation angles (float32 array) in radians [0, pi).
            - local coherence map (float32 array) in range [0.0, 1.0].
    """
    smoothed = filters.gaussian(gray, sigma=1.0, preserve_range=True)
    gx = cv2.Sobel(smoothed.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    gxx = filters.gaussian(gx * gx, sigma=sigma, preserve_range=True)
    gyy = filters.gaussian(gy * gy, sigma=sigma, preserve_range=True)
    gxy = filters.gaussian(gx * gy, sigma=sigma, preserve_range=True)

    orientation = 0.5 * np.arctan2(2 * gxy, gxx - gyy) + np.pi / 2.0
    orientation = np.mod(orientation, np.pi)
    coherence = np.sqrt((gxx - gyy) ** 2 + 4 * gxy**2) / (gxx + gyy + 1e-6)
    return orientation.astype(np.float32), np.clip(coherence, 0.0, 1.0).astype(np.float32)


def ridge_preserving_denoise(gray: np.ndarray, cfg: FingerprintConfig) -> np.ndarray:
    """Applies median and bilateral filtering to suppress sensor and compression noise.

    Bilateral filtering preserves sharp edges (such as fingerprint ridge boundaries) 
    while smoothing homogeneous or low-contrast regions.

    Args:
        gray: Single-channel float32 image.
        cfg: Configuration parameters defining bilateral sigma and diameters.

    Returns:
        Denoised single-channel float32 image.
    """
    denoised = gray.copy()
    median_size = int(cfg.denoise_median_size)
    if median_size >= 3:
        denoised = cv2.medianBlur(as_uint8(denoised), median_size | 1).astype(np.float32) / 255.0

    diameter = int(cfg.denoise_bilateral_diameter)
    if diameter >= 3:
        denoised = cv2.bilateralFilter(
            as_uint8(denoised),
            diameter | 1,
            float(cfg.denoise_bilateral_sigma_color),
            float(cfg.denoise_bilateral_sigma_space),
        ).astype(np.float32) / 255.0
    return denoised


def color_annotation_mask(rgb: np.ndarray, cfg: FingerprintConfig) -> np.ndarray:
    """Detects colored pen markings, stamps, and annotations.

    Translates to HSV color space to evaluate saturation. Specifically avoids 
    interpreting neutral gray fingerprint ridges or paper textures as color artifacts.

    Args:
        rgb: Input RGB image (uint8).
        cfg: Configuration parameters specifying saturation thresholds and delta values.

    Returns:
        Boolean mask array highlighting detected color artifacts.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    gray_value = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    colorfulness = saturation > int(cfg.color_artifact_saturation)
    value_delta = np.abs(value.astype(np.int16) - gray_value.astype(np.int16)) > int(cfg.color_artifact_value_delta)
    mask = colorfulness & value_delta
    return keep_large_components(mask.astype(bool), 13)


def border_frame_mask(gray: np.ndarray) -> np.ndarray:
    """Identifies and masks scanner borders, crop frames, and black background edges.

    Uses morphological reconstruction starting from the outer margins of the image 
    to prevent intersecting or overlapping central fingerprint details.

    Args:
        gray: Normalized single-channel float32 image.

    Returns:
        Boolean mask array of border artifacts.
    """
    binary_dark = gray < np.percentile(gray, 2.5)
    h, w = gray.shape
    border = np.zeros_like(binary_dark)
    margin_y = max(3, h // 45)
    margin_x = max(3, w // 45)
    border[:margin_y, :] = True
    border[-margin_y:, :] = True
    border[:, :margin_x] = True
    border[:, -margin_x:] = True
    seed = binary_dark & border
    grown = morphology.reconstruction(seed.astype(np.uint8), binary_dark.astype(np.uint8), method="dilation")
    return grown.astype(bool)


def straight_line_mask(gray: np.ndarray, cfg: FingerprintConfig) -> np.ndarray:
    """Detects straight ruled notebook lines or scanner scanlines.

    Applies Canny edge detection followed by the Probabilistic Hough Line Transform 
    to locate linear structures and generates a dilated masking map.

    Args:
        gray: Normalized single-channel float32 image.
        cfg: Configuration parameters specifying Hough thresholds and gaps.

    Returns:
        Boolean mask array of linear artifacts.
    """
    u8 = as_uint8(robust_rescale(gray, (1.0, 99.0)))
    edges = cv2.Canny(u8, 60, 160)
    min_len = max(16, int(min(gray.shape) * cfg.line_min_length_ratio))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(cfg.line_hough_threshold),
        minLineLength=min_len,
        maxLineGap=int(cfg.line_max_gap),
    )
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if lines is None:
        return mask.astype(bool)
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length >= min_len:
            cv2.line(mask, (x1, y1), (x2, y2), 255, max(1, int(cfg.line_mask_width)))
    return mask.astype(bool)


def dark_component_artifact_mask(gray: np.ndarray, roi: np.ndarray, cfg: FingerprintConfig) -> tuple[np.ndarray, np.ndarray]:
    """Identifies dark text components, print letters, and high-contrast annotations.

    Applies adaptive thresholding to detect local dark regions. Connected components 
    are analyzed based on their area, aspect ratio, solidity, and overlap with the ROI 
    to classify them as either definitive artifacts or review candidates.

    Args:
        gray: Single-channel float32 image.
        roi: Segemented fingerprint foreground mask.
        cfg: Configuration parameters for component dimensions and threshold parameters.

    Returns:
        Tuple:
            - Boolean mask indicating components to automatically remove.
            - Boolean mask indicating components flagged for manual review.
    """
    blur = filters.gaussian(gray, sigma=0.8, preserve_range=True)
    adaptive = cv2.adaptiveThreshold(
        as_uint8(1.0 - blur),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -2,
    ).astype(bool)
    adaptive = morphology.opening(adaptive, morphology.disk(1))

    protected_roi = ndi.binary_dilation(roi, iterations=8)
    labeled = measure.label(adaptive)
    remove = np.zeros_like(adaptive, dtype=bool)
    review = np.zeros_like(adaptive, dtype=bool)
    max_area = int(gray.size * cfg.dark_component_max_area_ratio)

    for region in measure.regionprops(labeled):
        area = int(region.area)
        if area < cfg.dark_component_min_area or area > max_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        if height <= 1 or width <= 1:
            continue
        aspect = max(width / height, height / width)
        extent = region.extent
        solidity = region.solidity
        region_mask = labeled == region.label
        overlap = float(np.mean(protected_roi[region_mask])) if np.any(region_mask) else 0.0

        if cfg.remove_ambiguous_dark_inside_roi:
            outside_roi = overlap < 0.02
            loose_outside_text = aspect > 2.7 or extent > 0.38 or area > 220
            stroke_inside_text = (
                (aspect > 3.0 and area >= 45 and extent > 0.08)
                or (area <= 3500 and extent > 0.42 and solidity > 0.38)
                or (area <= 6500 and aspect > 2.0 and extent > 0.24 and solidity > 0.34)
            )
            text_like = loose_outside_text if outside_roi else stroke_inside_text
            if not text_like:
                continue
            remove[region_mask] = True
        else:
            if overlap < 0.02:
                text_like = aspect > 2.7 or extent > 0.38 or area > 220
                if not text_like:
                    continue
                remove[region_mask] = True
            else:
                compact_nonridge_like = 35 <= area <= 260 and 0.45 <= extent <= 0.9 and aspect <= 3.2
                if compact_nonridge_like:
                    review[region_mask] = True

    return remove, review


def dark_ink_stroke_mask(gray: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Detects thick handwriting ink strokes using a local median subtraction method.

    Compares pixels with their local neighborhood median. Strong dark deviations 
    are grouped into ink stroke masks if they meet size and elongation criteria.

    Args:
        gray: Single-channel float32 image.
        roi: Boolean foreground fingerprint ROI mask.

    Returns:
        Boolean mask array of handwriting strokes.
    """
    base = filters.gaussian(gray, sigma=1.0, preserve_range=True)
    local_median = ndi.median_filter(base, size=23)
    diff = np.clip(local_median - base, 0.0, 1.0)
    if not np.any(roi):
        return np.zeros_like(gray, dtype=bool)

    threshold = float(np.percentile(diff[roi], 92))
    candidate = roi & (diff > threshold)
    candidate = morphology.closing(candidate, morphology.disk(2))
    candidate = keep_large_components(candidate, 30)

    labeled = measure.label(candidate)
    stroke_mask = np.zeros_like(candidate, dtype=bool)
    for region in measure.regionprops(labeled):
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        if height <= 1 or width <= 1:
            continue
        area = int(region.area)
        aspect = max(width / height, height / width)
        if area >= 40 and (aspect > 2.0 or area > 180):
            stroke_mask[labeled == region.label] = True
    return stroke_mask


def manual_artifact_mask(shape: tuple[int, int], cfg: FingerprintConfig) -> np.ndarray:
    """Generates a binary mask from user-specified manual annotation coordinates.

    Args:
        shape: Tuple of height and width (H, W).
        cfg: Configuration parameter containing manual rectangles, lines, and polygons.

    Returns:
        Boolean mask array of manual exclusions.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y, w, h in cfg.manual_rectangles:
        cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
    for x1, y1, x2, y2, width in cfg.manual_lines:
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, max(1, int(width)))
    for polygon in cfg.manual_polygons:
        if len(polygon) >= 3:
            pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
    return mask.astype(bool)


def build_artifact_masks(rgb: np.ndarray, gray: np.ndarray, roi: np.ndarray, cfg: FingerprintConfig) -> dict[str, np.ndarray]:
    """Aggregates all automated and manual masking modules into a unified directory map.

    Args:
        rgb: Input RGB image array.
        gray: Denoised single-channel float32 image array.
        roi: Segmented fingerprint ROI mask.
        cfg: Configuration settings.

    Returns:
        Dictionary mapping mask keys to their respective boolean arrays.
    """
    color_mask = color_annotation_mask(rgb, cfg)
    line_mask = straight_line_mask(gray, cfg)
    protected_roi = ndi.binary_dilation(roi, iterations=8)
    line_auto = line_mask & ~protected_roi
    line_review = line_mask & protected_roi
    border_mask = border_frame_mask(gray)
    dark_remove, dark_review = dark_component_artifact_mask(gray, roi, cfg)
    dark_stroke_mask = dark_ink_stroke_mask(gray, roi) if cfg.remove_ambiguous_dark_inside_roi else np.zeros_like(gray, dtype=bool)
    manual_mask = manual_artifact_mask(gray.shape, cfg)

    remove = color_mask | line_auto | border_mask | dark_remove | dark_stroke_mask | manual_mask
    if cfg.artifact_mask_dilate > 0:
        remove = ndi.binary_dilation(remove, iterations=int(cfg.artifact_mask_dilate))

    return {
        "artifact_remove": remove.astype(bool),
        "artifact_review": (dark_review | line_review).astype(bool),
        "artifact_color": color_mask.astype(bool),
        "artifact_lines": line_mask.astype(bool),
        "artifact_lines_auto": line_auto.astype(bool),
        "artifact_lines_review": line_review.astype(bool),
        "artifact_border": border_mask.astype(bool),
        "artifact_dark_auto": dark_remove.astype(bool),
        "artifact_dark_stroke": dark_stroke_mask.astype(bool),
        "artifact_manual": manual_mask.astype(bool),
    }


def component_support_ratio(mask: np.ndarray, component: np.ndarray, roi: np.ndarray, radius: int = 6) -> float:
    """Calculates the ratio of clean (non-masked) fingerprint pixels in the local neighborhood.

    Ensures that inpainting is supported by surrounding valid ridge structures.

    Args:
        mask: Boolean mask indicating all removed artifact pixels.
        component: Boolean mask of the specific connected component being evaluated.
        roi: Fingerprint foreground ROI mask.
        radius: Dilation radius used to define the neighborhood boundary.

    Returns:
        Float ratio in range [0.0, 1.0] of clean valid foreground pixels in the neighborhood.
    """
    around = ndi.binary_dilation(component, iterations=radius) & ~component
    if around.sum() == 0:
        return 0.0
    return float(np.mean(roi[around] & ~mask[around]))


def guarded_inpaint(gray: np.ndarray, artifact_mask: np.ndarray, roi: np.ndarray, cfg: FingerprintConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies a guarded inpainting strategy to restore ridge lines.

    Restores regions only if their size is within limits and they have strong
    structural support from neighboring clean ridges. Large gaps or unsupported 
    areas remain blocked (un-inpainted) to maintain forensic integrity and avoid 
    generating false details.

    Args:
        gray: Single-channel float32 image.
        artifact_mask: Boolean mask indicating pixels slated for removal.
        roi: Boolean foreground ROI mask.
        cfg: Configuration parameters controlling limits and support thresholds.

    Returns:
        Tuple:
            - Reconstructed single-channel float32 image.
            - Boolean mask indicating successful reconstruction (inpainted) regions.
            - Boolean mask indicating blocked (unaltered) regions.
    """
    allowed = np.zeros_like(artifact_mask, dtype=bool)
    blocked = np.zeros_like(artifact_mask, dtype=bool)
    labeled = measure.label(artifact_mask)

    for region in measure.regionprops(labeled):
        component = labeled == region.label
        minr, minc, maxr, maxc = region.bbox
        area = int(region.area)
        width = maxc - minc
        height = maxr - minr
        overlap_roi = bool(np.any(roi[component]))
        if not overlap_roi:
            continue
        if cfg.force_reconstruct_artifacts:
            allowed[component] = True
            continue
        support = component_support_ratio(artifact_mask, component, roi)

        small = (
            area <= cfg.max_inpaint_component_area
            and width <= cfg.max_inpaint_component_width
            and height <= cfg.max_inpaint_component_height
        )
        if small and support >= cfg.min_inpaint_support_ratio:
            allowed[component] = True
        else:
            blocked[component] = True

    if allowed.any():
        inpainted = cv2.inpaint(as_uint8(gray), allowed.astype(np.uint8) * 255, int(cfg.inpaint_radius), cv2.INPAINT_TELEA)
        reconstructed = inpainted.astype(np.float32) / 255.0
    else:
        reconstructed = gray.copy()

    # Keep blocked evidence visible and flagged; do not invent structure there.
    reconstructed[blocked] = gray[blocked]
    return reconstructed, allowed.astype(bool), blocked.astype(bool)


def post_denoise_clean_view(image: np.ndarray, roi: np.ndarray, strength: float) -> np.ndarray:
    """Performs an optional final bilateral filter pass over the fingerprint.

    Smoothes fine artifacts and granular noise while keeping ridges sharp.

    Args:
        image: Single-channel float32 image.
        roi: Boolean fingerprint ROI mask.
        strength: Control factor (0.0 to 1.0) defining the blend of the filter output.

    Returns:
        Filtered float32 image array.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return image.astype(np.float32)
    smooth = cv2.bilateralFilter(
        as_uint8(image),
        d=5,
        sigmaColor=18 + 42 * strength,
        sigmaSpace=4 + 5 * strength,
    ).astype(np.float32) / 255.0
    blend = 0.18 + 0.35 * strength
    out = image.copy()
    out[roi] = (1.0 - blend) * image[roi] + blend * smooth[roi]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def finalize_clean_display(
    image: np.ndarray,
    artifact_mask: np.ndarray,
    roi: np.ndarray,
    cfg: FingerprintConfig,
) -> np.ndarray:
    """Applies local inpaint clean-up focused strictly within masked artifact regions.

    Ensures that visual details outside the artifact masks remain completely untouched.

    Args:
        image: Single-channel float32 image.
        artifact_mask: Boolean mask indicating regions containing removed artifacts.
        roi: Foreground fingerprint ROI mask.
        cfg: Configuration parameters specifying dilation and radius parameters.

    Returns:
        Final display float32 image.
    """
    cleanup_mask = artifact_mask.copy()
    if cfg.final_cleanup_dilate > 0:
        cleanup_mask = ndi.binary_dilation(cleanup_mask, iterations=int(cfg.final_cleanup_dilate))
    cleanup_mask &= roi
    if not cleanup_mask.any() or cfg.final_cleanup_radius <= 0:
        return image.astype(np.float32)

    cleaned = cv2.inpaint(
        as_uint8(image),
        cleanup_mask.astype(np.uint8) * 255,
        int(cfg.final_cleanup_radius),
        cv2.INPAINT_TELEA,
    ).astype(np.float32) / 255.0
    out = image.copy()
    out[cleanup_mask] = cleaned[cleanup_mask]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def gabor_enhance(gray: np.ndarray, orientation: np.ndarray, roi: np.ndarray, coherence: np.ndarray, cfg: FingerprintConfig) -> np.ndarray:
    """Enhances ridge flow clarity by applying directionally selective Gabor filters.

    Applies a bank of Gabor filters tuned to the target ridge period and orientation angles. 
    Blend coefficients restrict changes to regions with confident coherence scores.

    Args:
        gray: Single-channel float32 image.
        orientation: Local ridge orientation angle map.
        roi: Boolean foreground ROI mask.
        coherence: Local structural coherence map.
        cfg: Configuration specifying frequency, blending, and orientation counts.

    Returns:
        Gabor-enhanced float32 image.
    """
    n = max(4, int(cfg.gabor_orientations))
    ksize = int(cfg.gabor_kernel_size) | 1
    frequency = 1.0 / max(3.0, float(cfg.ridge_period_px))
    normalized = robust_rescale(gray, (1.0, 99.0))
    centered = normalized - 0.5

    responses = []
    angles = np.linspace(0.0, np.pi, n, endpoint=False)
    for angle in angles:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma=4.0, theta=float(angle), lambd=1.0 / frequency, gamma=0.55, psi=0)
        kernel -= kernel.mean()
        denom = np.sum(np.abs(kernel)) + 1e-6
        kernel = kernel / denom
        response = cv2.filter2D(centered.astype(np.float32), cv2.CV_32F, ksize=cv2.CV_32F, kernel=kernel)
        responses.append(response)
    stack = np.stack(responses, axis=0)
    idx = np.argmin(np.abs(np.angle(np.exp(1j * (orientation[None, ...] - angles[:, None, None])))), axis=0)
    selected = np.take_along_axis(stack, idx[None, ...], axis=0)[0]
    selected = robust_rescale(selected, (1.0, 99.0))

    blend = np.clip(float(cfg.gabor_blend), 0.0, 0.65)
    confident_roi = roi & (coherence > 0.18)
    enhanced = normalized.copy()
    enhanced[confident_roi] = (1.0 - blend) * normalized[confident_roi] + blend * selected[confident_roi]
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def binarize_for_preview(enhanced: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Produces a binary preview map of the fingerprint ridges.

    Uses Sauvola local adaptive thresholding. Flagged for review-only purposes.

    Args:
        enhanced: Enhanced single-channel float32 image.
        roi: Boolean foreground ROI mask.

    Returns:
        Binary ridge map (float32 where ridges are 0.0 and background is 1.0).
    """

    out = np.ones_like(enhanced, dtype=np.float32)
    if np.any(roi):
        values = enhanced[roi]
        threshold = filters.threshold_sauvola(enhanced, window_size=25, k=0.2)
        ridges = enhanced < threshold
        out[roi] = np.where(ridges[roi], 0.0, 1.0)
    return out


def quality_metrics(
    original: np.ndarray,
    processed: np.ndarray,
    roi: np.ndarray,
    coherence: np.ndarray,
    artifact_mask: np.ndarray,
    inpaint_allowed: np.ndarray,
    inpaint_blocked: np.ndarray,
) -> dict[str, float | int | str]:
    """Computes comprehensive quality, contrast, noise, and reconstruction metrics.

    Args:
        original: Original grayscale float32 image array.
        processed: Final preprocessed float32 image array.
        roi: Boolean foreground ROI mask.
        coherence: Local structural coherence map.
        artifact_mask: Boolean mask indicating all detected artifacts.
        inpaint_allowed: Boolean mask of reconstructed pixels.
        inpaint_blocked: Boolean mask of blocked artifact pixels.

    Returns:
        Dictionary mapping metric names to computed values.
    """
    roi_pixels = int(roi.sum())
    total_pixels = int(roi.size)
    roi_values = processed[roi] if roi_pixels else processed.ravel()
    lap = cv2.Laplacian(as_uint8(processed), cv2.CV_32F)
    residual = original - filters.gaussian(original, sigma=1.0, preserve_range=True)
    roi_artifact_denominator = max(1, roi_pixels)
    metrics: dict[str, float | int | str] = {
        "roi_pixels": roi_pixels,
        "roi_coverage_pct": round(100.0 * roi_pixels / max(1, total_pixels), 2),
        "contrast_std_roi": round(float(np.std(roi_values)), 4),
        "mean_ridge_coherence_roi": round(float(np.mean(coherence[roi])) if roi_pixels else float(np.mean(coherence)), 4),
        "blur_laplacian_var": round(float(np.var(lap[roi])) if roi_pixels else float(np.var(lap)), 2),
        "noise_residual_mad": round(float(np.median(np.abs(residual - np.median(residual)))), 4),
        "artifact_remove_pct_total": round(100.0 * float(artifact_mask.sum()) / max(1, total_pixels), 3),
        "artifact_remove_pct_roi": round(100.0 * float((artifact_mask & roi).sum()) / roi_artifact_denominator, 3),
        "reconstructed_pct_total": round(100.0 * float(inpaint_allowed.sum()) / max(1, total_pixels), 3),
        "reconstructed_pct_roi": round(100.0 * float((inpaint_allowed & roi).sum()) / roi_artifact_denominator, 3),
        "blocked_artifact_pct_total": round(100.0 * float(inpaint_blocked.sum()) / max(1, total_pixels), 3),
        "blocked_artifact_pct_roi": round(100.0 * float((inpaint_blocked & roi).sum()) / roi_artifact_denominator, 3),
    }
    return metrics


def collect_warnings(metrics: dict[str, float | int | str], review_mask: np.ndarray) -> list[str]:
    """Generates standard forensic warning flags based on statistical quality limits.

    Args:
        metrics: Quality metrics dictionary.
        review_mask: Boolean mask of review candidates.

    Returns:
        List of warning string descriptions.
    """
    warnings: list[str] = []
    if float(metrics["reconstructed_pct_roi"]) > 2.0:
        warnings.append("More than 2% of ROI pixels were reconstructed; inspect provenance before forensic use.")
    if float(metrics["blocked_artifact_pct_roi"]) > 0.0:
        warnings.append("Some artifact regions were too large or unsupported for reconstruction and were left unchanged.")
    if review_mask.any():
        warnings.append("Ambiguous dark text/mark candidates overlap the ROI; review mask before enabling removal.")
    if float(metrics["mean_ridge_coherence_roi"]) < 0.18:
        warnings.append("Low orientation coherence; enhancement and reconstruction confidence are limited.")
    return warnings


def run_pipeline(
    image_path: str | Path,
    cfg: FingerprintConfig | None = None,
    progress_callback: Any | None = None,
) -> PipelineResult:
    """Executes the complete forensic fingerprint processing and reconstruction pipeline.

    Processes a target image file step-by-step: loads, normalizes, segments,
    denoises, extracts ridge parameters, masks and inpaints artifacts, applies
    Gabor filter enhancement, and computes diagnostic quality stats.

    Args:
        image_path: Path to the target image file.
        cfg: Configuration parameters. If None, default FingerprintConfig is initialized.
        progress_callback: Optional callable for reporting execution progress.

    Returns:
        PipelineResult object containing processed images, masks, and quality metrics.
    """
    cfg = cfg or FingerprintConfig()
    path = Path(image_path)

    def report(progress: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(progress, message)

    report(5, "Loading image")
    rgb = load_rgb(path)
    original_gray = rgb_to_gray_float(rgb)
    report(12, "Normalizing contrast range")
    normalized = robust_rescale(original_gray, cfg.clip_percentiles)
    if cfg.auto_invert:
        normalized = maybe_invert(normalized, cfg.prefer_dark_ridges)
    report(20, "Correcting illumination")
    corrected = correct_illumination(normalized, cfg.illumination_sigma)
    report(28, "Enhancing local contrast")
    contrasted = clahe(corrected, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
    report(36, "Segmenting fingerprint region")
    roi, roi_score = segment_fingerprint(contrasted, cfg)
    report(44, "Reducing acquisition noise")
    denoised = ridge_preserving_denoise(contrasted, cfg)
    report(52, "Estimating ridge orientation")
    orientation, coherence = ridge_orientation(denoised)
    report(62, "Detecting text and artifacts")
    masks = build_artifact_masks(rgb, denoised, roi, cfg)
    if cfg.include_review_artifacts_in_reconstruction:
        masks["artifact_remove"] = (masks["artifact_remove"] | masks["artifact_review"]).astype(bool)
    report(74, "Reconstructing interrupted ridge flow")
    reconstructed, inpaint_allowed, inpaint_blocked = guarded_inpaint(denoised, masks["artifact_remove"], roi, cfg)
    report(84, "Enhancing ridge clarity")
    enhanced = gabor_enhance(reconstructed, orientation, roi, coherence, cfg)
    enhanced = post_denoise_clean_view(enhanced, roi, cfg.post_denoise_strength)
    report(92, "Applying final cleanup")
    enhanced = finalize_clean_display(enhanced, masks["artifact_remove"], roi, cfg)
    if cfg.suppress_background:
        enhanced = np.where(roi, enhanced, 1.0).astype(np.float32)
    ridge_preview = binarize_for_preview(enhanced, roi)

    masks["roi"] = roi
    masks["roi_score"] = roi_score
    masks["reconstructed_pixels"] = inpaint_allowed
    masks["blocked_artifact_pixels"] = inpaint_blocked
    masks["orientation_coherence"] = coherence
    masks["orientation"] = orientation

    stages = {
        "00_original_gray": original_gray,
        "01_normalized": normalized,
        "02_illumination_corrected": corrected,
        "03_local_contrast": contrasted,
        "04_roi_isolated": np.where(roi, contrasted, 1.0),
        "05_ridge_preserving_denoise": denoised,
        "06_artifact_suppressed_reconstruction": reconstructed,
        "07_ridge_enhanced_analysis_view": enhanced,
        "08_preview_binary_ridge_map": ridge_preview,
    }
    metrics = quality_metrics(
        original_gray,
        enhanced,
        roi,
        coherence,
        masks["artifact_remove"],
        inpaint_allowed,
        inpaint_blocked,
    )
    warnings = collect_warnings(metrics, masks["artifact_review"])
    report(100, "Processing complete")
    return PipelineResult(path, cfg, stages, masks, metrics, warnings)


def mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlays a binary mask onto a grayscale image using a specified color and alpha blend.

    Args:
        image: Grayscale float32 image array.
        mask: Boolean mask array of the same shape.
        color: RGB float tuple (each channel in range [0.0, 1.0]).
        alpha: Transparency factor in range [0.0, 1.0].

    Returns:
        RGB float32 image array showing the mask overlay.
    """
    base = np.dstack([image, image, image]) if image.ndim == 2 else image.astype(np.float32) / 255.0
    overlay = base.copy()
    for channel, value in enumerate(color):
        overlay[..., channel] = np.where(mask, (1 - alpha) * overlay[..., channel] + alpha * value, overlay[..., channel])
    return np.clip(overlay, 0.0, 1.0)


def provenance_overlay(result: PipelineResult) -> np.ndarray:
    """Generates the three-color diagnostic overlay showing reconstruction and block status.

    Color coding:
      - Blue: Reconstructed (inpainted) pixels.
      - Red: Blocked (unaltered) artifact pixels.
      - Yellow: Pixels flagged for manual review.

    Args:
        result: The PipelineResult containing processed stage images and masks.

    Returns:
        RGB float32 image array.
    """
    image = result.stages["07_ridge_enhanced_analysis_view"]
    overlay = np.dstack([image, image, image])
    reconstructed = result.masks["reconstructed_pixels"]
    blocked = result.masks["blocked_artifact_pixels"]
    review = result.masks["artifact_review"]
    overlay[reconstructed] = 0.55 * overlay[reconstructed] + 0.45 * np.array([0.1, 0.55, 1.0])
    overlay[blocked] = 0.55 * overlay[blocked] + 0.45 * np.array([1.0, 0.1, 0.1])
    overlay[review] = 0.55 * overlay[review] + 0.45 * np.array([1.0, 0.8, 0.05])
    return np.clip(overlay, 0.0, 1.0)


def display_stage_grid(result: PipelineResult, max_cols: int = 3, figsize: tuple[int, int] = (15, 11)) -> None:
    """Renders a grid plot displaying all intermediate pipeline stages using Matplotlib.

    Args:
        result: The PipelineResult object.
        max_cols: Maximum number of grid columns.
        figsize: Output figure dimensions.
    """
    items: list[tuple[str, np.ndarray, str]] = [
        ("Original", result.stages["00_original_gray"], "gray"),
        ("Normalized", result.stages["01_normalized"], "gray"),
        ("Illumination corrected", result.stages["02_illumination_corrected"], "gray"),
        ("Local contrast", result.stages["03_local_contrast"], "gray"),
        ("ROI isolated", result.stages["04_roi_isolated"], "gray"),
        ("Denoised", result.stages["05_ridge_preserving_denoise"], "gray"),
        ("Remove mask", mask_overlay(result.stages["05_ridge_preserving_denoise"], result.masks["artifact_remove"]), "rgb"),
        ("Review mask", mask_overlay(result.stages["05_ridge_preserving_denoise"], result.masks["artifact_review"], (1.0, 0.85, 0.05)), "rgb"),
        ("Reconstruction", result.stages["06_artifact_suppressed_reconstruction"], "gray"),
        ("Ridge enhanced view", result.stages["07_ridge_enhanced_analysis_view"], "gray"),
        ("Preview ridge map", result.stages["08_preview_binary_ridge_map"], "gray"),
        ("Provenance overlay", provenance_overlay(result), "rgb"),
    ]
    cols = max(1, int(max_cols))
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for ax, (title, image, mode) in zip(axes.ravel(), items):
        if mode == "rgb":
            ax.imshow(image)
        else:
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.ravel()[len(items) :]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def display_comparison(result: PipelineResult, figsize: tuple[int, int] = (14, 5)) -> None:
    """Renders a comparison plot showing the original, processed, and provenance overlay.

    Args:
        result: The PipelineResult object.
        figsize: Plot figure dimensions.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(result.stages["00_original_gray"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original evidence image")
    axes[1].imshow(result.stages["07_ridge_enhanced_analysis_view"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Processed analysis view")
    axes[2].imshow(provenance_overlay(result))
    axes[2].set_title("Provenance: blue=repaired, red=blocked, yellow=review")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def metrics_frame(result: PipelineResult) -> pd.DataFrame:
    """Converts the quality metrics dictionary to a formatted Pandas DataFrame.

    Args:
        result: The PipelineResult object.

    Returns:
        Pandas DataFrame containing metrics.
    """
    frame = pd.DataFrame([result.metrics]).T
    frame.columns = ["value"]
    return frame


def save_pipeline_outputs(result: PipelineResult, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Saves all processed image stages, masks, overlays, and CSV metrics to disk.

    Args:
        result: The pipeline execution output.
        output_dir: Parent directory. If None, defaults to config output_dir.

    Returns:
        Dictionary mapping output file names/keys to their saved file paths.
    """
    out = Path(output_dir or result.config.output_dir)
    case_dir = out / result.image_path.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    for name, image in result.stages.items():
        path = case_dir / f"{name}.png"
        Image.fromarray(as_uint8(image)).save(path)
        saved[name] = path

    for name in ["artifact_remove", "artifact_review", "roi", "reconstructed_pixels", "blocked_artifact_pixels"]:
        path = case_dir / f"mask_{name}.png"
        Image.fromarray((result.masks[name].astype(np.uint8) * 255)).save(path)
        saved[f"mask_{name}"] = path

    Image.fromarray((provenance_overlay(result) * 255).astype(np.uint8)).save(case_dir / "provenance_overlay.png")
    pd.DataFrame([result.metrics]).to_csv(case_dir / "quality_metrics.csv", index=False)
    return saved


def process_batch(
    image_paths: Iterable[str | Path],
    cfg: FingerprintConfig | None = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Executes the pipeline in batch mode across multiple image file paths.

    Args:
        image_paths: Iterable of file paths to process.
        cfg: Pipeline configuration. If None, defaults are loaded.
        save_outputs: If True, saves files for each image to disk.

    Returns:
        Pandas DataFrame combining quality metrics and warnings across all files.
    """
    cfg = cfg or FingerprintConfig()
    rows: list[dict[str, Any]] = []
    for path in image_paths:
        result = run_pipeline(path, cfg)
        if save_outputs:
            save_pipeline_outputs(result, cfg.output_dir)
        row: dict[str, Any] = {"image": str(path), "warnings": " | ".join(result.warnings)}
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def contact_sheet(image_paths: Iterable[str | Path], thumb_size: tuple[int, int] = (220, 180), cols: int = 4) -> np.ndarray:
    """Combines multiple fingerprint thumbnails into a single structured image (contact sheet).

    Args:
        image_paths: Iterable of image file paths to display.
        thumb_size: Dimensions (width, height) of each thumbnail slot.
        cols: Number of columns in the grid.

    Returns:
        RGB float32 image array of the contact sheet.
    """
    paths = list(image_paths)
    if not paths:
        return np.ones((120, 320, 3), dtype=np.float32)
    thumbs: list[np.ndarray] = []
    for path in paths:
        rgb = load_rgb(path)
        image = Image.fromarray(rgb)
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", thumb_size, "white")
        canvas.paste(image, ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2))
        thumbs.append(np.asarray(canvas).astype(np.float32) / 255.0)
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.ones((rows * thumb_size[1], cols * thumb_size[0], 3), dtype=np.float32)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        y = r * thumb_size[1]
        x = c * thumb_size[0]
        sheet[y : y + thumb_size[1], x : x + thumb_size[0]] = thumb
    return sheet
