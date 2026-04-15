# backend/app/ml/train.py
"""
Training script for the certificate forgery detection model.

Generates a synthetic dataset of texture features (simulating real vs forged
certificates), splits into train/test sets, trains a Random Forest classifier,
prints accuracy metrics, and saves the model to disk.

Usage:
    cd c:\project1\backend
    python -m app.ml.train
"""
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .feature_extraction import FEATURE_NAMES

# ── Paths ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "forgery_detector.pkl")


def generate_synthetic_dataset(n_samples: int = 2000, seed: int = 42) -> tuple:
    """
    Generate a synthetic dataset with realistic texture feature distributions.

    Real certificates tend to have:
      - Higher LBP entropy (more uniform texture)
      - Lower GLCM contrast (smoother gradients)
      - Moderate, consistent Gabor energy

    Forged/manipulated certificates tend to have:
      - Lower LBP entropy (patchy texture from editing)
      - Higher GLCM contrast (artifacts from splicing)
      - Irregular Gabor energy (inconsistent frequency response)
    """
    rng = np.random.RandomState(seed)
    n_each = n_samples // 2

    # ── Real certificates (label = 0) ──────────────────────────
    real = np.column_stack([
        rng.normal(5.5, 0.4, n_each),       # lbp_entropy
        rng.normal(120, 15, n_each),         # lbp_mean
        rng.normal(1800, 300, n_each),       # lbp_variance
        rng.normal(50, 15, n_each),          # glcm_contrast
        rng.normal(5, 1.5, n_each),          # glcm_dissimilarity
        rng.normal(0.35, 0.05, n_each),      # glcm_homogeneity
        rng.normal(0.015, 0.004, n_each),    # glcm_energy
        rng.normal(0.85, 0.05, n_each),      # glcm_correlation
        rng.normal(300, 60, n_each),         # gabor_energy
        rng.normal(800, 120, n_each),        # gabor_max_energy
        rng.normal(200, 50, n_each),         # gabor_std_energy
    ])

    # ── Forged certificates (label = 1) ────────────────────────
    fake = np.column_stack([
        rng.normal(4.2, 0.6, n_each),       # lbp_entropy — lower
        rng.normal(95, 20, n_each),          # lbp_mean
        rng.normal(2500, 500, n_each),       # lbp_variance — higher
        rng.normal(150, 40, n_each),         # glcm_contrast — higher
        rng.normal(12, 3, n_each),           # glcm_dissimilarity — higher
        rng.normal(0.22, 0.06, n_each),      # glcm_homogeneity — lower
        rng.normal(0.008, 0.003, n_each),    # glcm_energy — lower
        rng.normal(0.65, 0.1, n_each),       # glcm_correlation — lower
        rng.normal(500, 100, n_each),        # gabor_energy — higher
        rng.normal(1300, 200, n_each),       # gabor_max_energy — higher
        rng.normal(350, 80, n_each),         # gabor_std_energy — higher
    ])

    X = np.vstack([real, fake])
    y = np.array([0] * n_each + [1] * n_each)

    return X, y


def train_model():
    """Train the forgery detection model and save to disk."""
    print("=" * 60)
    print("  CertiVerify — Forgery Detection Model Training")
    print("=" * 60)

    # ── Generate dataset ───────────────────────────────────────
    print("\n[1/4] Generating synthetic dataset...")
    X, y = generate_synthetic_dataset(n_samples=2000)
    print(f"       Dataset size: {len(X)} samples ({(y == 0).sum()} real, {(y == 1).sum()} forged)")
    print(f"       Features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")

    # ── Train/test split ───────────────────────────────────────
    print("\n[2/4] Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"       Train: {len(X_train)} samples")
    print(f"       Test:  {len(X_test)} samples")

    # ── Train classifier ───────────────────────────────────────
    print("\n[3/4] Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────
    print("\n[4/4] Evaluating model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'─' * 60}")
    print(f"  ✅  TEST ACCURACY: {accuracy * 100:.2f}%")
    print(f"{'─' * 60}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Forged"]))

    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted Real  Predicted Forged")
    print(f"  Actual Real        {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Actual Forged      {cm[1][0]:>5}          {cm[1][1]:>5}")

    # ── Feature importances ────────────────────────────────────
    print(f"\n  Feature Importances:")
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"    {rank:>2}. {FEATURE_NAMES[idx]:<25} {importances[idx]:.4f}")

    # ── Save model ─────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n  Model saved to: {MODEL_PATH}")
    print("=" * 60)

    return clf, accuracy


if __name__ == "__main__":
    train_model()
