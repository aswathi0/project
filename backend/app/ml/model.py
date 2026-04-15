# backend/app/ml/model.py
"""
Model loading and prediction helper for the forgery detection pipeline.
Uses the trained GAN (autoencoder) model for anomaly-based detection.
"""
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from .gan_model import Generator, compute_anomaly_score

# ── Model paths ────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "model")
GEN_PATH = os.path.join(MODEL_DIR, "gan_generator.pth")
THRESH_PATH = os.path.join(MODEL_DIR, "gan_threshold.pth")

IMAGE_SIZE = 128

_generator = None
_threshold = None
_thresh_data = None
_device = None


def load_model():
    """Load the trained GAN generator and threshold (cached after first load)."""
    global _generator, _threshold, _thresh_data, _device

    if _generator is None:
        if not os.path.exists(GEN_PATH):
            raise FileNotFoundError(
                f"GAN model not found at {GEN_PATH}. "
                "Run 'python -m app.ml.gan_train' from the backend directory first."
            )

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load threshold data
        _thresh_data = torch.load(THRESH_PATH, map_location=_device, weights_only=True)
        _threshold = _thresh_data["threshold"]
        latent_dim = _thresh_data.get("latent_dim", 128)

        # Load generator
        _generator = Generator(latent_dim=latent_dim).to(_device)
        _generator.load_state_dict(torch.load(GEN_PATH, map_location=_device, weights_only=True))
        _generator.eval()

    return _generator, _threshold


def predict(image_path: str) -> dict:
    """
    Run the GAN-based prediction pipeline on a certificate image.

    Returns a dict with:
      - is_fake (bool)
      - confidence (float, 0-1)
      - final_score (float)
      - texture_score (float)
      - gan_score (float)
      - texture_features (dict)
      - processing_time_ms (int, placeholder, set by caller)
    """
    generator, threshold = load_model()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(_device)

    # Compute anomaly score
    with torch.no_grad():
        reconstructed = generator(image_tensor)
        anomaly_score = compute_anomaly_score(image_tensor, reconstructed).item()

    # Determine prediction
    is_fake = anomaly_score > threshold
    # Confidence: how far from threshold (normalized)
    confidence = min(1.0, abs(anomaly_score - threshold) / (threshold + 1e-10))

    # Component scores for the frontend display
    gan_score = min(1.0, anomaly_score / (2 * threshold + 1e-10))
    texture_score = gan_score  # Same model, single score
    final_score = gan_score

    # Compute some basic texture stats for display
    img_array = np.array(image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))
    lbp_entropy = float(np.std(img_array) / 255.0)
    glcm_contrast = float(np.mean(np.abs(np.diff(img_array.astype(float), axis=0))) / 255.0)
    gabor_energy = float(anomaly_score)

    return {
        "is_fake": is_fake,
        "confidence": round(confidence, 4),
        "final_score": round(final_score, 4),
        "texture_score": round(texture_score, 4),
        "gan_score": round(gan_score, 4),
        "texture_features": {
            "lbp_entropy": round(lbp_entropy, 4),
            "glm_contrast": round(glcm_contrast, 4),
            "gabor_energy": round(gabor_energy, 4),
        },
    }
