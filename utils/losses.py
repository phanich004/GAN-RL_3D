"""
Loss functions for RL-GAN-Net
Implements Chamfer Distance, Earth Mover's Distance, and other losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


def chamfer_distance_l2(pc1: torch.Tensor, pc2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute bidirectional Chamfer Distance between two point clouds using L2 norm.
    
    Args:
        pc1: Point cloud 1 of shape (B, N, 3)
        pc2: Point cloud 2 of shape (B, M, 3)
        
    Returns:
        Tuple of (dist1, dist2) where:
        dist1: Average distance from pc1 to pc2
        dist2: Average distance from pc2 to pc1
    """
    # Compute pairwise distances
    # pc1: (B, N, 3), pc2: (B, M, 3)
    # distances: (B, N, M)
    distances = torch.cdist(pc1, pc2, p=2)
    
    # Find minimum distances
    dist1 = torch.min(distances, dim=2)[0]  # (B, N) - for each point in pc1, find closest in pc2
    dist2 = torch.min(distances, dim=1)[0]  # (B, M) - for each point in pc2, find closest in pc1
    
    # Average over points
    dist1 = torch.mean(dist1, dim=1)  # (B,)
    dist2 = torch.mean(dist2, dim=1)  # (B,)
    
    return dist1, dist2


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        pc1: Point cloud 1 of shape (B, N, 3)
        pc2: Point cloud 2 of shape (B, M, 3)
        bidirectional: If True, compute bidirectional distance
        
    Returns:
        Chamfer distance tensor of shape (B,)
    """
    dist1, dist2 = chamfer_distance_l2(pc1, pc2)
    
    if bidirectional:
        return (dist1 + dist2) / 2.0
    else:
        return dist1


class ChamferLoss(nn.Module):
    """Chamfer Distance Loss"""
    
    def __init__(self, bidirectional: bool = True):
        super(ChamferLoss, self).__init__()
        self.bidirectional = bidirectional
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
        """
        return torch.mean(chamfer_distance(pred, target, self.bidirectional))


class EarthMoverDistance(nn.Module):
    """
    Earth Mover's Distance (Wasserstein Distance) for point clouds.
    This is a simplified approximation using optimal transport.
    """
    
    def __init__(self):
        super(EarthMoverDistance, self).__init__()
    
    def forward(self, pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
        """
        Compute EMD between two point clouds.
        This is a simplified version - for exact EMD, external libraries like geomloss are needed.
        
        Args:
            pc1: Point cloud 1 (B, N, 3)
            pc2: Point cloud 2 (B, N, 3) - must have same number of points
        """
        B, N, _ = pc1.shape
        
        # Simple approximation: sort points and compute L2 distance
        # This is not the true EMD but serves as a reasonable approximation
        pc1_sorted = torch.sort(pc1.view(B, -1), dim=1)[0]
        pc2_sorted = torch.sort(pc2.view(B, -1), dim=1)[0]
        
        return torch.mean(torch.norm(pc1_sorted - pc2_sorted, dim=1))


class GFVLoss(nn.Module):
    """Global Feature Vector Loss (L2 distance in latent space)"""
    
    def __init__(self):
        super(GFVLoss, self).__init__()
    
    def forward(self, pred_gfv: torch.Tensor, target_gfv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_gfv: Predicted Global Feature Vector (B, latent_dim)
            target_gfv: Target Global Feature Vector (B, latent_dim)
        """
        return F.mse_loss(pred_gfv, target_gfv)


class DiscriminatorLoss(nn.Module):
    """Discriminator Loss for GAN training"""
    
    def __init__(self, loss_type: str = "wgan-gp"):
        super(DiscriminatorLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            real_logits: Discriminator output for real samples
            fake_logits: Discriminator output for fake samples
        """
        if self.loss_type == "wgan-gp":
            # Wasserstein loss with gradient penalty
            return torch.mean(fake_logits) - torch.mean(real_logits)
        elif self.loss_type == "lsgan":
            # Least squares loss
            real_loss = F.mse_loss(real_logits, torch.ones_like(real_logits))
            fake_loss = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
            return (real_loss + fake_loss) / 2
        else:
            # Standard binary cross entropy
            real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            return (real_loss + fake_loss) / 2


class GeneratorLoss(nn.Module):
    """Generator Loss for GAN training"""
    
    def __init__(self, loss_type: str = "wgan-gp"):
        super(GeneratorLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fake_logits: Discriminator output for generated samples
        """
        if self.loss_type == "wgan-gp":
            return -torch.mean(fake_logits)
        elif self.loss_type == "lsgan":
            return F.mse_loss(fake_logits, torch.ones_like(fake_logits))
        else:
            return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def gradient_penalty(discriminator: nn.Module, real_samples: torch.Tensor, 
                    fake_samples: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    Args:
        discriminator: Discriminator network
        real_samples: Real samples
        fake_samples: Fake samples
        device: Device to run computation on
    """
    batch_size = real_samples.size(0)
    
    # Random weight term for interpolation
    alpha = torch.rand(batch_size, 1, 1, device=device)
    
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
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


class RewardFunction:
    """
    Reward function for RL agent based on completion quality.
    Combines Chamfer loss, GFV loss, and discriminator output.
    """
    
    def __init__(self, w_chamfer: float = 100.0, w_gfv: float = 10.0, w_discriminator: float = 0.01):
        self.w_chamfer = w_chamfer
        self.w_gfv = w_gfv
        self.w_discriminator = w_discriminator
        
        self.chamfer_loss = ChamferLoss()
        self.gfv_loss = GFVLoss()
    
    def compute_reward(self, pred_pc: torch.Tensor, target_pc: torch.Tensor,
                      pred_gfv: torch.Tensor, target_gfv: torch.Tensor,
                      discriminator_output: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for RL agent.
        
        Args:
            pred_pc: Predicted point cloud
            target_pc: Target point cloud
            pred_gfv: Predicted Global Feature Vector
            target_gfv: Target Global Feature Vector
            discriminator_output: Discriminator output for predicted sample
        """
        # Compute individual losses
        chamfer_loss = self.chamfer_loss(pred_pc, target_pc)
        gfv_loss = self.gfv_loss(pred_gfv, target_gfv)
        discriminator_loss = -torch.mean(discriminator_output)  # Higher discriminator output = better
        
        # Combine losses with weights (negative because we want to maximize reward)
        reward = -(self.w_chamfer * chamfer_loss + 
                  self.w_gfv * gfv_loss + 
                  self.w_discriminator * discriminator_loss)
        
        return reward 