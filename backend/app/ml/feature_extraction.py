# backend/app/ml/feature_extraction.py
"""
Texture-based feature extraction for certificate forgery detection.
Extracts LBP (Local Binary Pattern), GLCM (Gray-Level Co-occurrence Matrix),
and Gabor filter features from certificate images.
"""
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import entropy


def load_and_preprocess(image_path: str, target_size: tuple = (256, 256)) -> np.ndarray:
    """Load an image, convert to grayscale, and resize."""
    img = Image.open(image_path).convert("L")
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img, dtype=np.float64)


# ─── LBP (Local Binary Pattern) ───────────────────────────────

def compute_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """Compute a basic LBP descriptor for the image."""
    rows, cols = image.shape
    lbp = np.zeros_like(image, dtype=np.uint8)

    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            center = image[i, j]
            binary_str = 0
            for p in range(n_points):
                angle = 2 * np.pi * p / n_points
                y = i + int(round(radius * np.sin(angle)))
                x = j + int(round(radius * np.cos(angle)))
                binary_str |= (1 << p) if image[y, x] >= center else 0
            lbp[i, j] = binary_str

    return lbp


def lbp_features(image: np.ndarray) -> dict:
    """Extract LBP-based features: histogram entropy, mean, variance."""
    lbp = compute_lbp(image)
    hist, _ = np.histogram(lbp, bins=256, range=(0, 256), density=True)
    hist = hist + 1e-10  # avoid log(0)

    return {
        "lbp_entropy": float(entropy(hist)),
        "lbp_mean": float(np.mean(lbp)),
        "lbp_variance": float(np.var(lbp)),
    }


# ─── GLCM (Gray-Level Co-occurrence Matrix) ───────────────────

def compute_glcm(image: np.ndarray, levels: int = 32, dx: int = 1, dy: int = 0) -> np.ndarray:
    """Compute a normalized GLCM for the image."""
    # Quantize to fewer gray levels
    quantized = (image / (256.0 / levels)).astype(np.int32)
    quantized = np.clip(quantized, 0, levels - 1)

    glcm = np.zeros((levels, levels), dtype=np.float64)
    rows, cols = quantized.shape

    for i in range(max(0, -dy), rows - max(0, dy)):
        for j in range(max(0, -dx), cols - max(0, dx)):
            glcm[quantized[i, j], quantized[i + dy, j + dx]] += 1

    # Normalize
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def glcm_features(image: np.ndarray) -> dict:
    """Extract GLCM properties: contrast, dissimilarity, homogeneity, energy, correlation."""
    glcm = compute_glcm(image)
    levels = glcm.shape[0]

    i_idx, j_idx = np.meshgrid(range(levels), range(levels), indexing="ij")
    i_idx = i_idx.astype(np.float64)
    j_idx = j_idx.astype(np.float64)

    contrast = float(np.sum(glcm * (i_idx - j_idx) ** 2))
    dissimilarity = float(np.sum(glcm * np.abs(i_idx - j_idx)))
    homogeneity = float(np.sum(glcm / (1.0 + (i_idx - j_idx) ** 2)))
    energy = float(np.sum(glcm ** 2))

    # Correlation
    mu_i = np.sum(i_idx * glcm)
    mu_j = np.sum(j_idx * glcm)
    sigma_i = np.sqrt(np.sum(glcm * (i_idx - mu_i) ** 2))
    sigma_j = np.sqrt(np.sum(glcm * (j_idx - mu_j) ** 2))
    if sigma_i > 0 and sigma_j > 0:
        correlation = float(np.sum(glcm * (i_idx - mu_i) * (j_idx - mu_j)) / (sigma_i * sigma_j))
    else:
        correlation = 0.0

    return {
        "glcm_contrast": contrast,
        "glcm_dissimilarity": dissimilarity,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
    }


# ─── Gabor Filters ────────────────────────────────────────────

def gabor_features(image: np.ndarray, frequencies: tuple = (0.1, 0.2, 0.3, 0.4)) -> dict:
    """Extract Gabor filter energy across multiple frequencies and orientations."""
    energies = []
    for freq in frequencies:
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            # Build Gabor kernel
            kernel_size = 15
            sigma = 3.0
            x = np.arange(-kernel_size // 2, kernel_size // 2 + 1)
            y = np.arange(-kernel_size // 2, kernel_size // 2 + 1)
            X, Y = np.meshgrid(x, y)
            X_theta = X * np.cos(theta) + Y * np.sin(theta)
            Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
            kernel = np.exp(-0.5 * (X_theta ** 2 + Y_theta ** 2) / sigma ** 2) * np.cos(
                2 * np.pi * freq * X_theta
            )
            filtered = ndimage.convolve(image, kernel, mode="reflect")
            energies.append(np.mean(filtered ** 2))

    return {
        "gabor_energy": float(np.mean(energies)),
        "gabor_max_energy": float(np.max(energies)),
        "gabor_std_energy": float(np.std(energies)),
    }


# ─── Combined Feature Vector ──────────────────────────────────

def extract_features(image_path: str) -> dict:
    """Extract all texture features from an image file and return as a dict."""
    image = load_and_preprocess(image_path)

    features = {}
    features.update(lbp_features(image))
    features.update(glcm_features(image))
    features.update(gabor_features(image))

    return features


def features_to_vector(features: dict) -> list:
    """Convert feature dict to a fixed-order list for model input."""
    keys = [
        "lbp_entropy", "lbp_mean", "lbp_variance",
        "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
        "glcm_energy", "glcm_correlation",
        "gabor_energy", "gabor_max_energy", "gabor_std_energy",
    ]
    return [features.get(k, 0.0) for k in keys]


FEATURE_NAMES = [
    "lbp_entropy", "lbp_mean", "lbp_variance",
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
    "glcm_energy", "glcm_correlation",
    "gabor_energy", "gabor_max_energy", "gabor_std_energy",
]
