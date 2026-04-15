# backend/app/ml/gan_test.py
"""
GAN Testing Script for Certificate Forgery Detection.

Loads the trained GAN generator and tests it on individual images or
the full dataset, showing per-image reconstruction error and prediction.

Usage:
    cd c:\\project1\\backend

    # Test on entire dataset:
    ..\\venv\\Scripts\\python -m app.ml.gan_test

    # Test on a single image:
    ..\\venv\\Scripts\\python -m app.ml.gan_test --image "path/to/image.jpg"
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from app.ml.gan_model import Generator, compute_anomaly_score

# ============================================================
#  Configuration
# ============================================================
DATASET_PATH = r"C:\Users\swath\OneDrive\Documents\m2\dataset"
GENUINE_DIR = os.path.join(DATASET_PATH, "genuine")
FAKE_DIR = os.path.join(DATASET_PATH, "fake")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "model")

IMAGE_SIZE = 128


# ============================================================
#  Dataset
# ============================================================
class TestDataset(Dataset):
    """Loads images from a directory for testing."""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
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
        return image, img_path


# ============================================================
#  Load model and threshold
# ============================================================
def load_model(device):
    """Load the trained generator and detection threshold."""
    gen_path = os.path.join(MODEL_DIR, "gan_generator.pth")
    thresh_path = os.path.join(MODEL_DIR, "gan_threshold.pth")

    if not os.path.exists(gen_path):
        print(f"ERROR: Generator model not found at {gen_path}")
        print("Run 'python -m app.ml.gan_train' first to train the model.")
        sys.exit(1)

    # Load threshold info
    thresh_data = torch.load(thresh_path, map_location=device, weights_only=True)
    latent_dim = thresh_data.get("latent_dim", 128)
    threshold = thresh_data["threshold"]

    # Load generator
    generator = Generator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True))
    generator.eval()

    return generator, threshold, thresh_data


# ============================================================
#  Test single image
# ============================================================
def test_single_image(image_path, generator, threshold, device):
    """Test a single image and return the result."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed = generator(image_tensor)
        anomaly_score = compute_anomaly_score(image_tensor, reconstructed).item()

    is_fake = anomaly_score > threshold
    confidence = min(1.0, abs(anomaly_score - threshold) / threshold) if threshold > 0 else 0.5

    return {
        "image": os.path.basename(image_path),
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "is_fake": is_fake,
        "prediction": "FAKE" if is_fake else "GENUINE",
        "confidence": confidence,
    }


# ============================================================
#  Test full dataset
# ============================================================
def test_dataset(generator, threshold, device):
    """Test on the entire genuine + fake dataset."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    print("=" * 70)
    print("  CertiVerify - GAN Testing")
    print("=" * 70)
    print(f"\n  Detection threshold: {threshold:.6f}")

    # -- Test genuine images --
    print(f"\n{'='*70}")
    print("  GENUINE CERTIFICATES")
    print(f"{'='*70}")

    genuine_dataset = TestDataset(GENUINE_DIR, transform=transform)
    genuine_correct = 0

    for i in range(len(genuine_dataset)):
        img_tensor, img_path = genuine_dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            recon = generator(img_tensor)
            score = compute_anomaly_score(img_tensor, recon).item()

        is_fake = score > threshold
        status = "CORRECT" if not is_fake else "WRONG"
        if not is_fake:
            genuine_correct += 1

        filename = os.path.basename(img_path)
        print(f"  [{status:>7}] {filename[:45]:<45}  Score: {score:.6f}  -> {'FAKE' if is_fake else 'GENUINE'}")

    print(f"\n  Genuine accuracy: {genuine_correct}/{len(genuine_dataset)} "
          f"({genuine_correct / len(genuine_dataset) * 100:.1f}%)")

    # -- Test fake images --
    print(f"\n{'='*70}")
    print("  FAKE CERTIFICATES")
    print(f"{'='*70}")

    fake_dataset = TestDataset(FAKE_DIR, transform=transform)
    fake_correct = 0

    for i in range(len(fake_dataset)):
        img_tensor, img_path = fake_dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            recon = generator(img_tensor)
            score = compute_anomaly_score(img_tensor, recon).item()

        is_fake = score > threshold
        status = "CORRECT" if is_fake else "WRONG"
        if is_fake:
            fake_correct += 1

        filename = os.path.basename(img_path)
        print(f"  [{status:>7}] {filename[:45]:<45}  Score: {score:.6f}  -> {'FAKE' if is_fake else 'GENUINE'}")

    print(f"\n  Fake detection accuracy: {fake_correct}/{len(fake_dataset)} "
          f"({fake_correct / len(fake_dataset) * 100:.1f}%)")

    # -- Summary --
    total = len(genuine_dataset) + len(fake_dataset)
    total_correct = genuine_correct + fake_correct
    print(f"\n{'='*70}")
    print(f"  OVERALL: {total_correct}/{total} correct ({total_correct / total * 100:.1f}%)")
    print(f"{'='*70}")


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Test GAN forgery detector")
    parser.add_argument("--image", type=str, help="Path to a single image to test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    generator, threshold, thresh_data = load_model(device)

    if args.image:
        # Single image test
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            sys.exit(1)

        result = test_single_image(args.image, generator, threshold, device)

        print(f"\n{'='*50}")
        print(f"  Image:      {result['image']}")
        print(f"  Score:      {result['anomaly_score']:.6f}")
        print(f"  Threshold:  {result['threshold']:.6f}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence'] * 100:.1f}%")
        print(f"{'='*50}")
    else:
        # Full dataset test
        test_dataset(generator, threshold, device)


if __name__ == "__main__":
    main()
