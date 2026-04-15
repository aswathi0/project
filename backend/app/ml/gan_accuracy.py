# backend/app/ml/gan_accuracy.py
"""
GAN Accuracy Evaluation Script for Certificate Forgery Detection.

Computes detailed accuracy metrics for the trained GAN model:
  - Overall accuracy
  - Per-class accuracy (genuine vs fake)
  - Precision, Recall, F1-Score
  - Confusion matrix
  - ROC-AUC score

Usage:
    cd c:\\project1\\backend
    ..\\venv\\Scripts\\python -m app.ml.gan_accuracy
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from app.ml.gan_model import Generator, compute_anomaly_score

# ============================================================
#  Configuration
# ============================================================
DATASET_PATH = r"C:\Users\swath\OneDrive\Documents\m2\dataset"
GENUINE_DIR = os.path.join(DATASET_PATH, "genuine")
FAKE_DIR = os.path.join(DATASET_PATH, "fake")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "model")

IMAGE_SIZE = 128
BATCH_SIZE = 8


# ============================================================
#  Dataset
# ============================================================
class EvalDataset(Dataset):
    """Loads images with labels for evaluation."""

    def __init__(self, image_dir, label, transform=None):
        self.transform = transform
        self.label = label  # 0 = genuine, 1 = fake
        self.image_files = []

        for f in os.listdir(image_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_files.append(os.path.join(image_dir, f))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label, os.path.basename(img_path)


# ============================================================
#  Evaluate
# ============================================================
def evaluate():
    print("=" * 60)
    print("  CertiVerify - GAN Accuracy Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # -- Load model --
    gen_path = os.path.join(MODEL_DIR, "gan_generator.pth")
    thresh_path = os.path.join(MODEL_DIR, "gan_threshold.pth")

    if not os.path.exists(gen_path):
        print(f"\n  ERROR: Model not found at {gen_path}")
        print("  Run 'python -m app.ml.gan_train' first.")
        sys.exit(1)

    thresh_data = torch.load(thresh_path, map_location=device, weights_only=True)
    latent_dim = thresh_data.get("latent_dim", 128)
    threshold = thresh_data["threshold"]

    generator = Generator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True))
    generator.eval()

    print(f"  Threshold: {threshold:.6f}")

    # -- Prepare data --
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    genuine_dataset = EvalDataset(GENUINE_DIR, label=0, transform=transform)
    fake_dataset = EvalDataset(FAKE_DIR, label=1, transform=transform)

    print(f"\n  Genuine images: {len(genuine_dataset)}")
    print(f"  Fake images:    {len(fake_dataset)}")
    print(f"  Total images:   {len(genuine_dataset) + len(fake_dataset)}")

    # -- Compute anomaly scores --
    print("\n  Computing anomaly scores...")

    all_true_labels = []
    all_pred_labels = []
    all_scores = []

    # Process genuine images
    for i in range(len(genuine_dataset)):
        img_tensor, label, filename = genuine_dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            recon = generator(img_tensor)
            score = compute_anomaly_score(img_tensor, recon).item()

        predicted = 1 if score > threshold else 0
        all_true_labels.append(label)
        all_pred_labels.append(predicted)
        all_scores.append(score)

    # Process fake images
    for i in range(len(fake_dataset)):
        img_tensor, label, filename = fake_dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            recon = generator(img_tensor)
            score = compute_anomaly_score(img_tensor, recon).item()

        predicted = 1 if score > threshold else 0
        all_true_labels.append(label)
        all_pred_labels.append(predicted)
        all_scores.append(score)

    # -- Compute metrics --
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels)
    scores = np.array(all_scores)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    # -- Print results --
    print(f"\n{'=' * 60}")
    print(f"  ACCURACY RESULTS")
    print(f"{'=' * 60}")

    print(f"\n  Overall Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision:         {precision * 100:.2f}%")
    print(f"  Recall:            {recall * 100:.2f}%")
    print(f"  F1-Score:          {f1 * 100:.2f}%")
    print(f"  ROC-AUC:           {auc:.4f}")

    print(f"\n  {'=' * 50}")
    print(f"  Confusion Matrix")
    print(f"  {'=' * 50}")
    print(f"                      Predicted Genuine  Predicted Fake")
    print(f"  Actual Genuine           {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Actual Fake              {cm[1][0]:>5}          {cm[1][1]:>5}")

    # Per-class accuracy
    genuine_correct = cm[0][0]
    genuine_total = cm[0][0] + cm[0][1]
    fake_correct = cm[1][1]
    fake_total = cm[1][0] + cm[1][1]

    print(f"\n  Per-Class Accuracy:")
    print(f"    Genuine: {genuine_correct}/{genuine_total} ({genuine_correct / genuine_total * 100:.1f}%)")
    print(f"    Fake:    {fake_correct}/{fake_total} ({fake_correct / fake_total * 100:.1f}%)")

    # Detailed classification report
    print(f"\n  {'=' * 50}")
    print(f"  Classification Report")
    print(f"  {'=' * 50}")
    report = classification_report(y_true, y_pred, target_names=["Genuine", "Fake"])
    for line in report.split("\n"):
        print(f"  {line}")

    # Score distribution
    genuine_scores = scores[y_true == 0]
    fake_scores = scores[y_true == 1]

    print(f"\n  {'=' * 50}")
    print(f"  Score Distribution")
    print(f"  {'=' * 50}")
    print(f"  Genuine - Mean: {np.mean(genuine_scores):.6f}  Std: {np.std(genuine_scores):.6f}  "
          f"Min: {np.min(genuine_scores):.6f}  Max: {np.max(genuine_scores):.6f}")
    print(f"  Fake    - Mean: {np.mean(fake_scores):.6f}  Std: {np.std(fake_scores):.6f}  "
          f"Min: {np.min(fake_scores):.6f}  Max: {np.max(fake_scores):.6f}")
    print(f"  Threshold:       {threshold:.6f}")

    print(f"\n{'=' * 60}")

    return acc, precision, recall, f1, auc


if __name__ == "__main__":
    evaluate()
