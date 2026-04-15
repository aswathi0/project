# backend/app/ml/gan_model.py
"""
GAN (Generative Adversarial Network) architecture for certificate forgery detection.

Uses an Autoencoder-GAN approach:
  - Generator (Autoencoder): learns to reconstruct genuine certificate textures
  - Discriminator: distinguishes real textures from generator reconstructions
  - Detection: forged certificates produce higher reconstruction error

This file defines the model architectures only. See gan_train.py, gan_test.py,
and gan_accuracy.py for training, testing, and evaluation.
"""
import torch
import torch.nn as nn


# ============================================================
#  Generator (Encoder-Decoder / Autoencoder)
# ============================================================
class Generator(nn.Module):
    """
    Convolutional autoencoder that learns to reconstruct genuine certificate images.
    Input:  (batch, 3, 128, 128)
    Output: (batch, 3, 128, 128) - reconstructed image
    """

    def __init__(self, latent_dim=128):
        super(Generator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # (3, 128, 128) -> (64, 64, 64)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 64, 64) -> (128, 32, 32)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (128, 32, 32) -> (256, 16, 16)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # (256, 16, 16) -> (512, 8, 8)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # (512, 8, 8) -> (latent_dim, 4, 4)
            nn.Conv2d(512, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # (latent_dim, 4, 4) -> (512, 8, 8)
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # (512, 8, 8) -> (256, 16, 16)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (256, 16, 16) -> (128, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (128, 32, 32) -> (64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (64, 64, 64) -> (3, 128, 128)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)


# ============================================================
#  Discriminator
# ============================================================
class Discriminator(nn.Module):
    """
    Classifies whether an image is a real genuine certificate or a
    reconstruction from the generator.
    Input:  (batch, 3, 128, 128)
    Output: (batch, 1) - probability of being real
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # (3, 128, 128) -> (64, 64, 64)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # (64, 64, 64) -> (128, 32, 32)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # (128, 32, 32) -> (256, 16, 16)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # (256, 16, 16) -> (512, 8, 8)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten -> FC
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
#  Anomaly Score Computation
# ============================================================
def compute_anomaly_score(original, reconstructed):
    """
    Compute anomaly score as the mean squared reconstruction error.
    Higher score = more likely to be forged.
    """
    mse = torch.mean((original - reconstructed) ** 2, dim=[1, 2, 3])
    return mse


# ============================================================
#  Model summary utility
# ============================================================
def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test of the architectures
    print("GAN Model Architecture Test")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    gen = Generator(latent_dim=128).to(device)
    disc = Discriminator().to(device)

    print(f"\nGenerator parameters:     {count_parameters(gen):,}")
    print(f"Discriminator parameters: {count_parameters(disc):,}")

    # Test forward pass
    test_input = torch.randn(2, 3, 128, 128).to(device)
    recon = gen(test_input)
    disc_out = disc(test_input)

    print(f"\nInput shape:          {test_input.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Discriminator output: {disc_out.shape}")

    score = compute_anomaly_score(test_input, recon)
    print(f"Anomaly scores:       {score.detach().cpu().numpy()}")
    print("\nAll checks passed!")
