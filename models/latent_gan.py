"""
Latent GAN (l-GAN) implementation for RL-GAN-Net
Operates on Global Feature Vectors (GFV) in latent space
Uses WGAN-GP for stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


class LatentGenerator(nn.Module):
    """
    Generator for Latent GAN that produces clean GFVs from noise.
    Input: Noise vector z (B, z_dim)
    Output: Clean Global Feature Vector (B, latent_dim)
    """
    
    def __init__(self, z_dim: int = 1, latent_dim: int = 128, 
                 hidden_dims: List[int] = [256, 512, 512, 256, 128]):
        super(LatentGenerator, self).__init__()
        
        self.z_dim = z_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build generator layers
        layers = []
        in_dim = z_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        
        # Final layer to latent space
        layers.extend([
            nn.Linear(in_dim, hidden_dims[-1]),
            nn.Tanh()  # Normalize output to [-1, 1]
        ])
        
        self.generator = nn.Sequential(*layers)
        
        # Ensure output dimension matches latent_dim
        assert hidden_dims[-1] == latent_dim, \
            f"Last hidden dim should be {latent_dim}, got {hidden_dims[-1]}"
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Noise vector (B, z_dim)
            
        Returns:
            Generated Global Feature Vector (B, latent_dim)
        """
        return self.generator(z)


class LatentDiscriminator(nn.Module):
    """
    Discriminator for Latent GAN that distinguishes real vs fake GFVs.
    Input: Global Feature Vector (B, latent_dim)
    Output: Realness score (B, 1)
    """
    
    def __init__(self, latent_dim: int = 128, 
                 hidden_dims: List[int] = [128, 256, 512, 256, 1]):
        super(LatentDiscriminator, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build discriminator layers
        layers = []
        in_dim = latent_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Use LayerNorm instead of BatchNorm for stability
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)  # Add dropout for regularization
            ])
            in_dim = hidden_dim
        
        # Final layer (no activation for WGAN)
        layers.append(nn.Linear(in_dim, hidden_dims[-1]))
        
        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, gfv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gfv: Global Feature Vector (B, latent_dim)
            
        Returns:
            Realness score (B, 1)
        """
        return self.discriminator(gfv)


class SpectralNorm(nn.Module):
    """
    Spectral Normalization for improved training stability.
    """
    
    def __init__(self, module: nn.Module, power_iterations: int = 1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.power_iterations = power_iterations
        
        # Get weight matrix
        weight = getattr(module, 'weight')
        height = weight.data.shape[0]
        width = weight.data.view(height, -1).shape[1]
        
        # Initialize u and v vectors
        u = torch.randn(height).normal_(0, 1)
        v = torch.randn(width).normal_(0, 1)
        
        self.register_buffer('u', u)
        self.register_buffer('v', v)
    
    def forward(self, *args):
        weight = getattr(self.module, 'weight')
        height = weight.data.shape[0]
        weight_mat = weight.data.view(height, -1)
        
        # Power iteration
        for _ in range(self.power_iterations):
            v = F.normalize(torch.mv(weight_mat.t(), self.u), dim=0, eps=1e-12)
            u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=1e-12)
        
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight_normalized = weight / sigma
        
        # Replace weight temporarily
        setattr(self.module, 'weight', weight_normalized)
        result = self.module(*args)
        setattr(self.module, 'weight', weight)
        
        # Update u and v
        self.u = u
        self.v = v
        
        return result


class ImprovedLatentDiscriminator(nn.Module):
    """
    Improved discriminator with spectral normalization for better stability.
    """
    
    def __init__(self, latent_dim: int = 128, 
                 hidden_dims: List[int] = [128, 256, 512, 256, 1],
                 use_spectral_norm: bool = True):
        super(ImprovedLatentDiscriminator, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.use_spectral_norm = use_spectral_norm
        
        # Build discriminator layers
        layers = []
        in_dim = latent_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            linear = nn.Linear(in_dim, hidden_dim)
            
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            
            layers.extend([
                linear,
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        
        # Final layer
        final_linear = nn.Linear(in_dim, hidden_dims[-1])
        if use_spectral_norm:
            final_linear = nn.utils.spectral_norm(final_linear)
        
        layers.append(final_linear)
        
        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, gfv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gfv: Global Feature Vector (B, latent_dim)
            
        Returns:
            Realness score (B, 1)
        """
        return self.discriminator(gfv)


class LatentGAN(nn.Module):
    """
    Complete Latent GAN combining generator and discriminator.
    """
    
    def __init__(self, z_dim: int = 1, latent_dim: int = 128,
                 generator_dims: List[int] = [256, 512, 512, 256, 128],
                 discriminator_dims: List[int] = [128, 256, 512, 256, 1],
                 use_improved_discriminator: bool = True):
        super(LatentGAN, self).__init__()
        
        self.z_dim = z_dim
        self.latent_dim = latent_dim
        
        # Generator
        self.generator = LatentGenerator(z_dim, latent_dim, generator_dims)
        
        # Discriminator
        if use_improved_discriminator:
            self.discriminator = ImprovedLatentDiscriminator(latent_dim, discriminator_dims)
        else:
            self.discriminator = LatentDiscriminator(latent_dim, discriminator_dims)
    
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate clean GFV from noise."""
        return self.generator(z)
    
    def discriminate(self, gfv: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs fake GFV."""
        return self.discriminator(gfv)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both generator and discriminator.
        
        Args:
            z: Noise vector (B, z_dim)
            
        Returns:
            Tuple of (generated_gfv, discriminator_output)
        """
        generated_gfv = self.generate(z)
        discriminator_output = self.discriminate(generated_gfv)
        return generated_gfv, discriminator_output


class GradientPenalty:
    """
    Gradient penalty implementation for WGAN-GP.
    """
    
    def __init__(self, lambda_gp: float = 10.0):
        self.lambda_gp = lambda_gp
    
    def __call__(self, discriminator: nn.Module, real_samples: torch.Tensor, 
                 fake_samples: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Compute gradient penalty.
        
        Args:
            discriminator: Discriminator network
            real_samples: Real GFV samples
            fake_samples: Fake GFV samples
            device: Device to run computation on
            
        Returns:
            Gradient penalty loss
        """
        batch_size = real_samples.size(0)
        
        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, device=device)
        
        # Interpolated samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        d_interpolated = discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return self.lambda_gp * gradient_penalty


class LatentGANTrainer:
    """
    Trainer class for Latent GAN with WGAN-GP loss.
    """
    
    def __init__(self, model: LatentGAN, device: torch.device,
                 generator_lr: float = 1e-4, discriminator_lr: float = 1e-4,
                 beta1: float = 0.5, beta2: float = 0.9, lambda_gp: float = 10.0):
        self.model = model
        self.device = device
        self.lambda_gp = lambda_gp
        
        # Optimizers
        self.generator_optimizer = torch.optim.Adam(
            model.generator.parameters(), lr=generator_lr, betas=(beta1, beta2)
        )
        self.discriminator_optimizer = torch.optim.Adam(
            model.discriminator.parameters(), lr=discriminator_lr, betas=(beta1, beta2)
        )
        
        # Gradient penalty
        self.gradient_penalty = GradientPenalty(lambda_gp)
    
    def train_discriminator(self, real_gfv: torch.Tensor, z: torch.Tensor) -> float:
        """
        Train discriminator for one step.
        
        Args:
            real_gfv: Real Global Feature Vectors
            z: Noise vectors
            
        Returns:
            Discriminator loss
        """
        self.discriminator_optimizer.zero_grad()
        
        # Real samples
        real_output = self.model.discriminate(real_gfv)
        
        # Fake samples
        with torch.no_grad():
            fake_gfv = self.model.generate(z)
        fake_output = self.model.discriminate(fake_gfv)
        
        # WGAN loss
        d_loss = torch.mean(fake_output) - torch.mean(real_output)
        
        # Gradient penalty
        gp = self.gradient_penalty(self.model.discriminator, real_gfv, fake_gfv, self.device)
        
        # Total loss
        total_loss = d_loss + gp
        total_loss.backward()
        self.discriminator_optimizer.step()
        
        return total_loss.item()
    
    def train_generator(self, z: torch.Tensor) -> float:
        """
        Train generator for one step.
        
        Args:
            z: Noise vectors
            
        Returns:
            Generator loss
        """
        self.generator_optimizer.zero_grad()
        
        # Generate fake samples
        fake_gfv = self.model.generate(z)
        fake_output = self.model.discriminate(fake_gfv)
        
        # WGAN loss (maximize discriminator output for fake samples)
        g_loss = -torch.mean(fake_output)
        
        g_loss.backward()
        self.generator_optimizer.step()
        
        return g_loss.item()


def test_latent_gan():
    """Test the Latent GAN implementation."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Parameters
    batch_size = 8
    z_dim = 1
    latent_dim = 128
    
    # Create test data
    real_gfv = torch.randn(batch_size, latent_dim).to(device)
    z = torch.randn(batch_size, z_dim).to(device)
    
    # Create model
    model = LatentGAN(z_dim=z_dim, latent_dim=latent_dim).to(device)
    
    print("Testing LatentGAN...")
    print(f"Real GFV shape: {real_gfv.shape}")
    print(f"Noise shape: {z.shape}")
    
    # Test generator
    with torch.no_grad():
        fake_gfv = model.generate(z)
        print(f"Generated GFV shape: {fake_gfv.shape}")
    
    # Test discriminator
    with torch.no_grad():
        real_output = model.discriminate(real_gfv)
        fake_output = model.discriminate(fake_gfv)
        print(f"Real output shape: {real_output.shape}")
        print(f"Fake output shape: {fake_output.shape}")
    
    # Test trainer
    trainer = LatentGANTrainer(model, device)
    
    # Test training steps
    d_loss = trainer.train_discriminator(real_gfv, z)
    g_loss = trainer.train_generator(z)
    
    print(f"Discriminator loss: {d_loss:.4f}")
    print(f"Generator loss: {g_loss:.4f}")
    
    print("LatentGAN test passed!")


if __name__ == "__main__":
    test_latent_gan() 