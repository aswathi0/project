# backend/app/ml/gan_train.py
"""
GAN Training Script for Certificate Forgery Detection.

Trains an Autoencoder-GAN on genuine certificate images so the generator
learns to reconstruct authentic textures. Forged certificates will produce
higher reconstruction error, enabling anomaly-based detection.

Dataset: C:\\Users\\swath\\OneDrive\\Documents\\m2\\dataset
  - genuine/   (real certificates - used for GAN training)
  - fake/      (forged certificates - used only for validation threshold)

Usage:
    cd c:\\project1\\backend
    ..\\venv\\Scripts\\python -m app.ml.gan_train

Outputs:
    backend/model/gan_generator.pth   - Trained generator weights
    backend/model/gan_discriminator.pth - Trained discriminator weights
    backend/model/gan_threshold.pth   - Anomaly threshold for detection
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from app.ml.gan_model import Generator, Discriminator, compute_anomaly_score, count_parameters

# ============================================================
#  Configuration
# ============================================================
DATASET_PATH = r"C:\Users\swath\OneDrive\Documents\m2\dataset"
GENUINE_DIR = os.path.join(DATASET_PATH, "genuine")
FAKE_DIR = os.path.join(DATASET_PATH, "fake")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
IMAGE_SIZE = 128
BATCH_SIZE = 8
LATENT_DIM = 128
NUM_EPOCHS = 100
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_RECON = 10.0  # Weight for reconstruction loss


# ============================================================
#  Dataset
# ============================================================
class CertificateDataset(Dataset):
    """Loads certificate images from a directory."""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = []

        for f in os.listdir(image_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_files.append(os.path.join(image_dir, f))

        print(f"  Loaded {len(self.image_files)} images from {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path


# ============================================================
#  Training
# ============================================================
def train():
    print("=" * 60)
    print("  CertiVerify - GAN Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # -- Data transforms --
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    transform_no_aug = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # -- Load genuine dataset for training --
    print("\n[1/5] Loading dataset...")
    genuine_dataset = CertificateDataset(GENUINE_DIR, transform=transform)
    genuine_loader = DataLoader(genuine_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # -- Load fake dataset for threshold calibration only --
    fake_dataset = CertificateDataset(FAKE_DIR, transform=transform_no_aug)

    print(f"  Training images (genuine): {len(genuine_dataset)}")
    print(f"  Validation images (fake):  {len(fake_dataset)}")

    # -- Initialize models --
    print("\n[2/5] Initializing GAN models...")
    generator = Generator(latent_dim=LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)

    print(f"  Generator parameters:     {count_parameters(generator):,}")
    print(f"  Discriminator parameters: {count_parameters(discriminator):,}")

    # -- Optimizers --
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))

    # -- Loss functions --
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.MSELoss()

    # -- Training loop --
    print(f"\n[3/5] Training for {NUM_EPOCHS} epochs...")
    print("-" * 60)

    generator.train()
    discriminator.train()

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_recon_loss = 0.0
        num_batches = 0

        for real_images, _ in genuine_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1).to(device) * 0.9   # Label smoothing
            fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1

            # ---- Train Discriminator ----
            optimizer_d.zero_grad()

            # Real images
            real_output = discriminator(real_images)
            d_loss_real = adversarial_loss(real_output, real_labels)

            # Reconstructed (fake) images
            reconstructed = generator(real_images)
            fake_output = discriminator(reconstructed.detach())
            d_loss_fake = adversarial_loss(fake_output, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_d.step()

            # ---- Train Generator ----
            optimizer_g.zero_grad()

            reconstructed = generator(real_images)
            gen_output = discriminator(reconstructed)

            # Generator losses: adversarial + reconstruction
            g_loss_adv = adversarial_loss(gen_output, real_labels)
            g_loss_recon = reconstruction_loss(reconstructed, real_images)
            g_loss = g_loss_adv + LAMBDA_RECON * g_loss_recon

            g_loss.backward()
            optimizer_g.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss_adv.item()
            epoch_recon_loss += g_loss_recon.item()
            num_batches += 1

        # Print progress
        avg_d = epoch_d_loss / max(num_batches, 1)
        avg_g = epoch_g_loss / max(num_batches, 1)
        avg_r = epoch_recon_loss / max(num_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
                  f"D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}  "
                  f"Recon: {avg_r:.4f}  Time: {elapsed:.0f}s")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"  Training completed in {total_time:.1f}s")

    # -- Compute anomaly threshold --
    print(f"\n[4/5] Computing anomaly detection threshold...")
    generator.eval()

    # Get reconstruction errors for genuine images
    genuine_scores = []
    genuine_loader_eval = DataLoader(
        CertificateDataset(GENUINE_DIR, transform=transform_no_aug),
        batch_size=BATCH_SIZE, shuffle=False
    )
    with torch.no_grad():
        for images, _ in genuine_loader_eval:
            images = images.to(device)
            recon = generator(images)
            scores = compute_anomaly_score(images, recon)
            genuine_scores.extend(scores.cpu().numpy().tolist())

    # Get reconstruction errors for fake images
    fake_scores = []
    fake_loader = DataLoader(fake_dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for images, _ in fake_loader:
            images = images.to(device)
            recon = generator(images)
            scores = compute_anomaly_score(images, recon)
            fake_scores.extend(scores.cpu().numpy().tolist())

    avg_genuine = sum(genuine_scores) / len(genuine_scores) if genuine_scores else 0
    avg_fake = sum(fake_scores) / len(fake_scores) if fake_scores else 0
    threshold = (avg_genuine + avg_fake) / 2

    print(f"  Avg genuine reconstruction error: {avg_genuine:.6f}")
    print(f"  Avg fake reconstruction error:    {avg_fake:.6f}")
    print(f"  Detection threshold:              {threshold:.6f}")

    # -- Save models --
    print(f"\n[5/5] Saving models...")
    gen_path = os.path.join(MODEL_DIR, "gan_generator.pth")
    disc_path = os.path.join(MODEL_DIR, "gan_discriminator.pth")
    thresh_path = os.path.join(MODEL_DIR, "gan_threshold.pth")

    torch.save(generator.state_dict(), gen_path)
    torch.save(discriminator.state_dict(), disc_path)
    torch.save({
        "threshold": threshold,
        "avg_genuine_score": avg_genuine,
        "avg_fake_score": avg_fake,
        "image_size": IMAGE_SIZE,
        "latent_dim": LATENT_DIM,
    }, thresh_path)

    print(f"  Generator saved:     {gen_path}")
    print(f"  Discriminator saved: {disc_path}")
    print(f"  Threshold saved:     {thresh_path}")
    print("=" * 60)
    print("  Training complete! Run gan_test.py to test on images")
    print("  or gan_accuracy.py to evaluate overall accuracy.")
    print("=" * 60)


if __name__ == "__main__":
    train()
