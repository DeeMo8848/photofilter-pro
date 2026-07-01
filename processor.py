"""
PhotoFilter Pro — Image Processing Engine
Implements all color grading & adjustment operations using numpy/scipy.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from PIL import Image

# ═══════════════════════════════════════════
#  GPU ACCELERATION (CuPy — NVIDIA only)
# ═══════════════════════════════════════════
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as _gpu_gaussian_filter
    _gpu_device_count = cp.cuda.runtime.getDeviceCount()
    HAS_GPU = _gpu_device_count > 0
except Exception:
    cp = None
    _gpu_gaussian_filter = None
    HAS_GPU = False

_use_gpu = False


def set_use_gpu(enabled: bool):
    """Enable/disable GPU acceleration. Raises RuntimeError if GPU not available."""
    global _use_gpu
    if enabled and not HAS_GPU:
        raise RuntimeError("CuPy 不可用：未安装或未检测到 NVIDIA GPU")
    _use_gpu = enabled


def get_gpu_status() -> dict:
    """Return GPU availability and status."""
    return {
        "available": HAS_GPU,
        "enabled": _use_gpu,
        "device_count": _gpu_device_count if HAS_GPU else 0
    }


def _gaussian_filter(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian filter — auto-routes to GPU when enabled."""
    if sigma <= 0:
        return arr.copy()
    if _use_gpu and HAS_GPU:
        gpu_arr = cp.asarray(arr)
        result = _gpu_gaussian_filter(gpu_arr, sigma=sigma)
        return cp.asnumpy(result)
    return gaussian_filter(arr, sigma=sigma)


# ═══════════════════════════════════════════
#  UTILITY: Color Space Conversions
# ═══════════════════════════════════════════

def rgb_to_hsl(img: np.ndarray) -> np.ndarray:
    """
    RGB [0,1] → HSL (H: 0-360, S: 0-1, L: 0-1)
    img: (H, W, 3) float32
    """
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.max(img, axis=-1)
    mn = np.min(img, axis=-1)
    d = mx - mn
    L = (mx + mn) / 2.0

    S = np.zeros_like(L, dtype=np.float32)
    mask = d > 1e-6
    denom = 1 - np.abs(2 * L[mask] - 1)
    S[mask] = np.where(denom > 1e-6, d[mask] / denom, 0.0)

    H = np.zeros_like(L, dtype=np.float32)
    # R is max
    cond_r = (mx == r) & mask
    H[cond_r] = 60 * (((g[cond_r] - b[cond_r]) / d[cond_r]) % 6)
    # G is max
    cond_g = (mx == g) & mask & ~cond_r
    H[cond_g] = 60 * (((b[cond_g] - r[cond_g]) / d[cond_g]) + 2)
    # B is max
    cond_b = (mx == b) & mask & ~cond_r & ~cond_g
    H[cond_b] = 60 * (((r[cond_b] - g[cond_b]) / d[cond_b]) + 4)

    return np.stack([H, S, L], axis=-1)


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """
    HSL → RGB [0,1]
    hsl: (H, W, 3) float32 (H: 0-360, S: 0-1, L: 0-1)
    """
    H, S, L = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    H = H / 60.0
    C = (1 - np.abs(2 * L - 1)) * S
    X = C * (1 - np.abs(H % 2 - 1))
    m = L - C / 2.0

    H_idx = np.floor(H).astype(int) % 6

    R, G, B = np.zeros_like(L), np.zeros_like(L), np.zeros_like(L)

    # Case 0: R=C, G=X, B=0
    mask0 = H_idx == 0
    R[mask0], G[mask0], B[mask0] = C[mask0], X[mask0], 0

    # Case 1: R=X, G=C, B=0
    mask1 = H_idx == 1
    R[mask1], G[mask1], B[mask1] = X[mask1], C[mask1], 0

    # Case 2: R=0, G=C, B=X
    mask2 = H_idx == 2
    R[mask2], G[mask2], B[mask2] = 0, C[mask2], X[mask2]

    # Case 3: R=0, G=X, B=C
    mask3 = H_idx == 3
    R[mask3], G[mask3], B[mask3] = 0, X[mask3], C[mask3]

    # Case 4: R=X, G=0, B=C
    mask4 = H_idx == 4
    R[mask4], G[mask4], B[mask4] = X[mask4], 0, C[mask4]

    # Case 5: R=C, G=0, B=X
    mask5 = H_idx == 5
    R[mask5], G[mask5], B[mask5] = C[mask5], 0, X[mask5]

    rgb = np.stack([R + m, G + m, B + m], axis=-1)
    return rgb


# ═══════════════════════════════════════════
#  UTILITY: Smoothstep / Curves
# ═══════════════════════════════════════════

def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep interpolation."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0, 1)
    return t * t * (3 - 2 * t)


def luminance(img: np.ndarray) -> np.ndarray:
    """Perceptual luminance (Rec.709 weights)."""
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


# ═══════════════════════════════════════════
#  BASIC ADJUSTMENTS
# ═══════════════════════════════════════════

def adjust_temperature(img: np.ndarray, temp: float) -> np.ndarray:
    """
    White balance temperature.
    temp: -100 (cool/blue) to +100 (warm/yellow)
    """
    if temp == 0:
        return img
    factor = temp / 100.0
    r_gain = 1.0 + max(factor, 0) * 0.25 + min(factor, 0) * 0.15
    b_gain = 1.0 - max(factor, 0) * 0.25 - min(factor, 0) * 0.15
    result = img.copy()
    result[..., 0] *= r_gain
    result[..., 2] *= b_gain
    return np.clip(result, 0, 1)


def adjust_tint(img: np.ndarray, tint: float) -> np.ndarray:
    """
    Green/Magenta tint.
    tint: -100 (green) to +100 (magenta)
    """
    if tint == 0:
        return img
    factor = tint / 100.0
    g_gain = 1.0 - factor * 0.25
    result = img.copy()
    result[..., 1] *= g_gain
    result[..., 0] *= 1.0 + factor * 0.1
    result[..., 2] *= 1.0 + factor * 0.1
    return np.clip(result, 0, 1)


def adjust_exposure(img: np.ndarray, ev: float) -> np.ndarray:
    """
    Exposure in stops.
    ev: -5.0 to +5.0
    """
    if ev == 0:
        return img
    return np.clip(img * (2.0 ** ev), 0, 1)


def adjust_contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    """
    Contrast adjustment around mid-gray.
    contrast: -100 to +100
    """
    if contrast == 0:
        return img
    factor = (100.0 + contrast) / 100.0
    return np.clip((img - 0.5) * factor + 0.5, 0, 1)


def adjust_highlights(img: np.ndarray, amount: float) -> np.ndarray:
    """amount: -100 to +100 (negative = recover highlights)"""
    if amount == 0:
        return img
    lum = luminance(img)
    # Soft mask for highlights (upper 40%)
    mask = smoothstep(0.4, 0.85, lum)
    factor = amount / 100.0
    if factor >= 0:
        return np.clip(img + mask[..., None] * factor * 0.4 * (1 - img), 0, 1)
    else:
        return np.clip(img + mask[..., None] * factor * 0.4 * img, 0, 1)


def adjust_shadows(img: np.ndarray, amount: float) -> np.ndarray:
    """amount: -100 to +100 (positive = lift shadows)"""
    if amount == 0:
        return img
    lum = luminance(img)
    # Soft mask for shadows (lower 60%)
    mask = 1.0 - smoothstep(0.15, 0.6, lum)
    factor = amount / 100.0
    if factor >= 0:
        return np.clip(img + mask[..., None] * factor * 0.5 * (1 - img), 0, 1)
    else:
        return np.clip(img + mask[..., None] * factor * 0.5 * img, 0, 1)


def adjust_whites(img: np.ndarray, amount: float) -> np.ndarray:
    """amount: -100 to +100 (push/pull white point)"""
    if amount == 0:
        return img
    lum = luminance(img)
    mask = smoothstep(0.7, 0.98, lum)
    factor = amount / 100.0
    return np.clip(img + mask[..., None] * factor * 0.3, 0, 1)


def adjust_blacks(img: np.ndarray, amount: float) -> np.ndarray:
    """amount: -100 to +100 (negative = deeper blacks)"""
    if amount == 0:
        return img
    lum = luminance(img)
    mask = 1.0 - smoothstep(0.02, 0.3, lum)
    factor = amount / 100.0
    return np.clip(img + mask[..., None] * factor * 0.3, 0, 1)


# ═══════════════════════════════════════════
#  PRESENCE (Texture / Clarity / Dehaze)
# ═══════════════════════════════════════════

def adjust_texture(img: np.ndarray, amount: float) -> np.ndarray:
    """
    Enhance fine texture (small-radius high-pass).
    amount: -100 to +100
    """
    if amount == 0:
        return img
    lum = luminance(img)
    blurred = _gaussian_filter(lum, sigma=1.8)
    detail = lum - blurred
    factor = amount / 100.0
    result = img + detail[..., None] * factor * 0.6
    return np.clip(result, 0, 1)


def adjust_clarity(img: np.ndarray, amount: float) -> np.ndarray:
    """
    Mid-tone contrast boost (larger radius).
    amount: -100 to +100
    """
    if amount == 0:
        return img
    lum = luminance(img)
    blurred = _gaussian_filter(lum, sigma=12.0)
    detail = lum - blurred
    # Soft midtone mask (avoid amplifying highlights/shadows)
    midtone_mask = 1.0 - np.abs(lum - 0.5) * 2.0
    factor = amount / 100.0
    result = img + detail[..., None] * factor * 0.35 * midtone_mask[..., None]
    return np.clip(result, 0, 1)


def adjust_dehaze(img: np.ndarray, amount: float) -> np.ndarray:
    """
    Remove atmospheric haze.
    amount: 0 to 100
    """
    if amount <= 0:
        return img
    factor = amount / 100.0

    # Boost local contrast
    lum = luminance(img)
    local_mean = _gaussian_filter(lum, sigma=25.0)
    detail = lum - local_mean
    result = img + detail[..., None] * factor * 0.4

    # Darken midtones deepen
    result = result * (1.0 - factor * 0.2)

    # Saturation boost
    hsl = rgb_to_hsl(np.clip(result, 0, 1))
    hsl[..., 1] = np.clip(hsl[..., 1] * (1.0 + factor * 0.35), 0, 1)
    result = hsl_to_rgb(hsl)

    return np.clip(result, 0, 1)


# ═══════════════════════════════════════════
#  COLOR (Vibrance / Saturation)
# ═══════════════════════════════════════════

def adjust_vibrance(img: np.ndarray, amount: float) -> np.ndarray:
    """
    Smart saturation — boosts muted colors more than already-saturated ones.
    amount: -100 to +100
    """
    if amount == 0:
        return img
    factor = amount / 100.0
    hsl = rgb_to_hsl(img)
    S = hsl[..., 1]
    if factor >= 0:
        # Boost less-saturated more
        weight = 1.0 - S
        S_new = S + factor * S * weight
    else:
        # Desaturate already-saturated more
        weight = S
        S_new = S + factor * S * weight
    hsl[..., 1] = np.clip(S_new, 0, 1)
    return hsl_to_rgb(hsl)


def adjust_saturation(img: np.ndarray, amount: float) -> np.ndarray:
    """
    Uniform saturation adjustment.
    amount: -100 to +100
    """
    if amount == 0:
        return img
    factor = 1.0 + amount / 100.0
    hsl = rgb_to_hsl(img)
    hsl[..., 1] = np.clip(hsl[..., 1] * factor, 0, 1)
    return hsl_to_rgb(hsl)


# ═══════════════════════════════════════════
#  HSL / COLOR GRADING
# ═══════════════════════════════════════════

# Color range centers (hue degrees)
COLOR_CENTERS = {
    'red': 0, 'orange': 30, 'yellow': 60, 'green': 120,
    'aqua': 180, 'blue': 240, 'purple': 270, 'magenta': 300
}

def apply_hsl_adjustments(img: np.ndarray, hsl_params: dict) -> np.ndarray:
    """
    Per-color HSL adjustments with smooth blending.
    hsl_params: {color: {hue, saturation, luminance}} where hue/sat/lum ∈ [-100, 100]
    """
    hsl = rgb_to_hsl(img)
    H, S, L = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    hue_shift = np.zeros_like(H, dtype=np.float32)
    sat_mult = np.ones_like(S, dtype=np.float32)
    lum_shift = np.zeros_like(L, dtype=np.float32)

    for color_name, params in hsl_params.items():
        center = COLOR_CENTERS.get(color_name, 0)
        # Angular distance on color wheel
        diff = np.abs(H - center)
        diff = np.minimum(diff, 360.0 - diff)
        # Cosine falloff over ±45°
        mask = np.where(diff < 45, 0.5 * (1.0 + np.cos(np.pi * diff / 45.0)), 0.0)

        h = params.get('hue', 0)
        s = params.get('saturation', 0)
        l = params.get('luminance', 0)

        if h != 0:
            hue_shift += mask * (h / 100.0) * 35.0
        if s != 0:
            sat_mult = np.where(mask > 0, sat_mult * (1.0 + mask * s / 100.0), sat_mult)
        if l != 0:
            lum_shift += mask * (l / 100.0) * 0.35

    H_new = (H + hue_shift) % 360.0
    S_new = np.clip(S * sat_mult, 0, 1)
    L_new = np.clip(L + lum_shift, 0, 1)

    hsl_new = np.stack([H_new, S_new, L_new], axis=-1)
    return hsl_to_rgb(hsl_new)


# ═══════════════════════════════════════════
#  DETAIL: Sharpening
# ═══════════════════════════════════════════

def adjust_sharpening(img: np.ndarray, amount: float, radius: float,
                      detail: float, masking: float) -> np.ndarray:
    """
    Unsharp mask sharpening.
    amount:  0-150
    radius:  0.5-3.0 px
    detail:  0-100 (0=suppress fine detail, 100=enhance all)
    masking: 0-100 (0=sharpen everything, 100=sharp edges only)
    """
    if amount <= 0:
        return img

    lum = luminance(img)
    blurred = _gaussian_filter(lum, sigma=radius)
    detail_layer = lum - blurred

    # Detail threshold (suppress low-amplitude detail like noise)
    if detail < 100:
        threshold = (1.0 - detail / 100.0) * 0.05
        detail_layer = np.where(np.abs(detail_layer) > threshold, detail_layer, 0.0)

    # Edge mask for masking parameter
    if masking > 0:
        gy = np.diff(lum, axis=0, prepend=lum[:1, :])
        gx = np.diff(lum, axis=1, prepend=lum[:, :1])
        edge_map = np.sqrt(gy ** 2 + gx ** 2)
        edge_map = edge_map / (edge_map.max() + 1e-8)
        edge_mask = smoothstep(masking / 100.0 * 0.3, 1.0, edge_map)
        detail_layer *= edge_mask

    factor = amount / 100.0 * 0.8
    result = img + detail_layer[..., None] * factor
    return np.clip(result, 0, 1)


# ═══════════════════════════════════════════
#  DETAIL: Noise Reduction
# ═══════════════════════════════════════════

def apply_noise_reduction(img: np.ndarray, luminance_nr: float, detail_nr: float,
                           contrast_nr: float, color_nr: float,
                           color_detail: float, color_smoothness: float) -> np.ndarray:
    """
    Noise reduction in YCbCr-like space.
    luminance_nr:     0-100
    detail_nr:        0-100 (preserves fine detail)
    contrast_nr:      0-100
    color_nr:         0-100
    color_detail:     0-100
    color_smoothness: 0-100
    """
    # Convert to YCbCr (simple)
    Y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    Cb = (img[..., 2] - Y) * 0.564 + 0.5
    Cr = (img[..., 0] - Y) * 0.713 + 0.5

    # Luminance noise reduction
    if luminance_nr > 0:
        sigma_lum = luminance_nr / 100.0 * 8.0
        Y_blurred = _gaussian_filter(Y, sigma=sigma_lum)
        Y_detail = Y - Y_blurred
        # Preserve detail above threshold
        detail_threshold = (1.0 - detail_nr / 100.0) * 0.08
        detail_mask = np.abs(Y_detail) > detail_threshold
        # Blend based on contrast parameter
        blend = contrast_nr / 100.0 * 0.8
        Y = Y * (1 - blend) + (Y_blurred + Y_detail * detail_mask) * blend

    # Color noise reduction
    if color_nr > 0:
        sigma_color = color_nr / 100.0 * 12.0 * (1.0 - color_smoothness / 200.0 + 0.5)
        cb_detail_threshold = (1.0 - color_detail / 100.0) * 0.05

        Cb_blur = _gaussian_filter(Cb, sigma=sigma_color)
        Cr_blur = _gaussian_filter(Cr, sigma=sigma_color)

        if color_detail > 0:
            Cb_d = Cb - Cb_blur
            Cr_d = Cr - Cr_blur
            Cb = Cb_blur + np.where(np.abs(Cb_d) > cb_detail_threshold, Cb_d, 0.0)
            Cr = Cr_blur + np.where(np.abs(Cr_d) > cb_detail_threshold, Cr_d, 0.0)
        else:
            Cb = Cb_blur
            Cr = Cr_blur

    # Convert back to RGB
    Cb_m = Cb - 0.5
    Cr_m = Cr - 0.5
    R = Y + 1.403 * Cr_m
    G = Y - 0.344 * Cb_m - 0.714 * Cr_m
    B = Y + 1.773 * Cb_m

    return np.clip(np.stack([R, G, B], axis=-1), 0, 1)


# ═══════════════════════════════════════════
#  MAIN PROCESSING PIPELINE
# ═══════════════════════════════════════════

def process_image(img: np.ndarray, params: dict) -> np.ndarray:
    """
    Full processing pipeline. Order matches Lightroom's engine.
    
    img: uint8 (H,W,3) or float32 (H,W,3) [0,1]
    params: dict with all adjustment parameters
    
    Returns: uint8 (H,W,3)
    """
    # Normalize to [0, 1] float32
    if img.dtype == np.uint8:
        arr = img.astype(np.float32) / 255.0
    else:
        arr = img.astype(np.float32)

    # === 1. White Balance ===
    arr = adjust_temperature(arr, params.get('temperature', 0))
    arr = adjust_tint(arr, params.get('tint', 0))

    # === 2. Tone ===
    arr = adjust_exposure(arr, params.get('exposure', 0))
    arr = adjust_contrast(arr, params.get('contrast', 0))
    arr = adjust_highlights(arr, params.get('highlights', 0))
    arr = adjust_shadows(arr, params.get('shadows', 0))
    arr = adjust_whites(arr, params.get('whites', 0))
    arr = adjust_blacks(arr, params.get('blacks', 0))

    # === 3. Presence ===
    arr = adjust_texture(arr, params.get('texture', 0))
    arr = adjust_clarity(arr, params.get('clarity', 0))
    arr = adjust_dehaze(arr, params.get('dehaze', 0))

    # === 4. Color ===
    arr = adjust_vibrance(arr, params.get('vibrance', 0))
    arr = adjust_saturation(arr, params.get('saturation', 0))

    # === 5. HSL / Color Grading ===
    hsl_params = params.get('hsl', {})
    if hsl_params:
        arr = apply_hsl_adjustments(arr, hsl_params)

    # === 6. Detail ===
    sharp = params.get('sharpening', {})
    if sharp.get('amount', 0) > 0:
        arr = adjust_sharpening(
            arr,
            sharp.get('amount', 0),
            sharp.get('radius', 1.0),
            sharp.get('detail', 25),
            sharp.get('masking', 0)
        )

    nr = params.get('noise_reduction', {})
    if any(nr.get(k, 0) > 0 for k in ['luminance', 'color']):
        arr = apply_noise_reduction(
            arr,
            nr.get('luminance', 0),
            nr.get('detail', 50),
            nr.get('contrast', 0),
            nr.get('color', 0),
            nr.get('color_detail', 50),
            nr.get('color_smoothness', 50)
        )

    # Clamp and convert back to uint8
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


# ═══════════════════════════════════════════
#  CONVENIENCE: PIL Image ↔ numpy
# ═══════════════════════════════════════════

def pil_to_array(img: Image.Image) -> np.ndarray:
    """PIL Image → numpy uint8 RGB array."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def array_to_pil(arr: np.ndarray) -> Image.Image:
    """numpy uint8 RGB array → PIL Image."""
    return Image.fromarray(arr.astype(np.uint8), 'RGB')


def load_and_process(path: str, params: dict) -> Image.Image:
    """Load image from file, apply params, return PIL Image."""
    img = Image.open(path)
    arr = pil_to_array(img)
    result = process_image(arr, params)
    return array_to_pil(result)


# ═══════════════════════════════════════════
#  DEFAULT PARAMETERS
# ═══════════════════════════════════════════

DEFAULT_PARAMS = {
    "temperature": 0,
    "tint": 0,
    "exposure": 0,
    "contrast": 0,
    "highlights": 0,
    "shadows": 0,
    "whites": 0,
    "blacks": 0,
    "texture": 0,
    "clarity": 0,
    "dehaze": 0,
    "vibrance": 0,
    "saturation": 0,
    "hsl": {
        "red":     {"hue": 0, "saturation": 0, "luminance": 0},
        "orange":  {"hue": 0, "saturation": 0, "luminance": 0},
        "yellow":  {"hue": 0, "saturation": 0, "luminance": 0},
        "green":   {"hue": 0, "saturation": 0, "luminance": 0},
        "aqua":    {"hue": 0, "saturation": 0, "luminance": 0},
        "blue":    {"hue": 0, "saturation": 0, "luminance": 0},
        "purple":  {"hue": 0, "saturation": 0, "luminance": 0},
        "magenta": {"hue": 0, "saturation": 0, "luminance": 0}
    },
    "sharpening": {
        "amount": 0,
        "radius": 1.0,
        "detail": 25,
        "masking": 0
    },
    "noise_reduction": {
        "luminance": 0,
        "detail": 50,
        "contrast": 0,
        "color": 0,
        "color_detail": 50,
        "color_smoothness": 50
    }
}
