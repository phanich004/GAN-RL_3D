"""
Autoencoder implementation for RL-GAN-Net
Converts point clouds to Global Feature Vectors (GFV) and reconstructs them
Based on PointNet architecture with adaptations for the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class PointNetEncoder(nn.Module):
    """
    PointNet-based encoder that converts point clouds to Global Feature Vectors.
    Input: Point cloud (B, N, 3)
    Output: Global Feature Vector (B, latent_dim)
    """
    
    def __init__(self, input_dim: int = 3, latent_dim: int = 128, 
                 hidden_dims: List[int] = [64, 128, 128, 256, 128]):
        super(PointNetEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Point-wise MLPs (1D convolutions)
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        
        # Final layer to latent dimension
        layers.extend([
            nn.Conv1d(in_dim, hidden_dims[-1], 1),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(inplace=True)
        ])
        
        self.point_mlp = nn.Sequential(*layers)
        
        # Global feature extraction
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input point cloud (B, N, 3)
            
        Returns:
            Global Feature Vector (B, latent_dim)
        """
        # x: (B, N, 3) -> (B, 3, N) for conv1d
        x = x.transpose(2, 1)
        
        # Point-wise feature extraction
        x = self.point_mlp(x)  # (B, hidden_dims[-1], N)
        
        # Global max pooling to get permutation invariant features
        x = torch.max(x, dim=2)[0]  # (B, hidden_dims[-1])
        
        # Final global feature vector
        gfv = self.global_mlp(x)  # (B, latent_dim)
        
        return gfv


class PointNetDecoder(nn.Module):
    """
    Decoder that reconstructs point clouds from Global Feature Vectors.
    Input: Global Feature Vector (B, latent_dim)
    Output: Point cloud (B, N, 3)
    """
    
    def __init__(self, latent_dim: int = 128, num_points: int = 2048,
                 hidden_dims: List[int] = [256, 256, 6144]):
        super(PointNetDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.hidden_dims = hidden_dims
        
        # MLP layers
        layers = []
        in_dim = latent_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        
        # Final layer to point coordinates
        layers.append(nn.Linear(in_dim, hidden_dims[-1]))
        
        self.mlp = nn.Sequential(*layers)
        
        # Reshape layer (6144 = 2048 * 3)
        assert hidden_dims[-1] == num_points * 3, \
            f"Last hidden dim should be {num_points * 3}, got {hidden_dims[-1]}"
    
    def forward(self, gfv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gfv: Global Feature Vector (B, latent_dim)
            
        Returns:
            Reconstructed point cloud (B, N, 3)
        """
        # Generate point coordinates
        x = self.mlp(gfv)  # (B, num_points * 3)
        
        # Reshape to point cloud
        x = x.view(-1, self.num_points, 3)  # (B, N, 3)
        
        return x


class PointCloudAutoencoder(nn.Module):
    """
    Complete autoencoder for point cloud completion.
    Combines encoder and decoder.
    """
    
    def __init__(self, input_dim: int = 3, latent_dim: int = 128, num_points: int = 2048,
                 encoder_dims: List[int] = [64, 128, 128, 256, 128],
                 decoder_dims: List[int] = [256, 256, 6144]):
        super(PointCloudAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_points = num_points
        
        # Encoder and decoder
        self.encoder = PointNetEncoder(input_dim, latent_dim, encoder_dims)
        self.decoder = PointNetDecoder(latent_dim, num_points, decoder_dims)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode point cloud to Global Feature Vector."""
        return self.encoder(x)
    
    def decode(self, gfv: torch.Tensor) -> torch.Tensor:
        """Decode Global Feature Vector to point cloud."""
        return self.decoder(gfv)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input point cloud (B, N, 3)
            
        Returns:
            Tuple of (reconstructed_pc, gfv)
        """
        gfv = self.encode(x)
        reconstructed = self.decode(gfv)
        return reconstructed, gfv


class PointNetWithAttention(nn.Module):
    """
    Enhanced PointNet with attention mechanism for better feature extraction.
    """
    
    def __init__(self, input_dim: int = 3, latent_dim: int = 128,
                 hidden_dims: List[int] = [64, 128, 128, 256, 128]):
        super(PointNetWithAttention, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Point-wise MLPs
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        
        layers.extend([
            nn.Conv1d(in_dim, hidden_dims[-1], 1),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(inplace=True)
        ])
        
        self.point_mlp = nn.Sequential(*layers)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1] // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dims[-1] // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Global feature extraction
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input point cloud (B, N, 3)
            
        Returns:
            Global Feature Vector (B, latent_dim)
        """
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1)
        
        # Point-wise features
        features = self.point_mlp(x)  # (B, hidden_dim, N)
        
        # Attention weights
        attention_weights = self.attention(features)  # (B, 1, N)
        
        # Weighted global pooling
        weighted_features = features * attention_weights
        global_features = torch.sum(weighted_features, dim=2)  # (B, hidden_dim)
        
        # Normalize by sum of attention weights to avoid scale issues
        attention_sum = torch.sum(attention_weights, dim=2) + 1e-8  # (B, 1)
        global_features = global_features / attention_sum
        
        # Final global feature vector
        gfv = self.global_mlp(global_features)
        
        return gfv


class AdaptivePointDecoder(nn.Module):
    """
    Adaptive decoder that can handle variable number of output points.
    """
    
    def __init__(self, latent_dim: int = 128, max_points: int = 2048,
                 hidden_dims: List[int] = [256, 512, 1024]):
        super(AdaptivePointDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.max_points = max_points
        
        # Feature expansion layers
        layers = []
        in_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        
        self.feature_mlp = nn.Sequential(*layers)
        
        # Point generation layers
        self.point_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], max_points * 3),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # Point refinement
        self.refine_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1)
        )
    
    def forward(self, gfv: torch.Tensor, num_points: int = None) -> torch.Tensor:
        """
        Args:
            gfv: Global Feature Vector (B, latent_dim)
            num_points: Number of points to generate (default: max_points)
            
        Returns:
            Generated point cloud (B, num_points, 3)
        """
        if num_points is None:
            num_points = self.max_points
        
        batch_size = gfv.shape[0]
        
        # Expand features
        features = self.feature_mlp(gfv)  # (B, hidden_dim)
        
        # Generate initial points
        points = self.point_mlp(features)  # (B, max_points * 3)
        points = points.view(batch_size, self.max_points, 3)
        
        # Select required number of points
        if num_points < self.max_points:
            points = points[:, :num_points, :]
        
        # Refine points
        points_t = points.transpose(2, 1)  # (B, 3, N)
        refined = self.refine_mlp(points_t)  # (B, 3, N)
        points = points_t + refined  # Residual connection
        points = points.transpose(2, 1)  # (B, N, 3)
        
        return points


def test_autoencoder():
    """Test the autoencoder implementation."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create test data
    batch_size, num_points = 4, 2048
    test_pc = torch.randn(batch_size, num_points, 3).to(device)
    
    # Test autoencoder
    model = PointCloudAutoencoder().to(device)
    
    print("Testing PointCloudAutoencoder...")
    print(f"Input shape: {test_pc.shape}")
    
    with torch.no_grad():
        reconstructed, gfv = model(test_pc)
        print(f"GFV shape: {gfv.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")
    
    print("Autoencoder test passed!")
    
    # Test individual components
    encoder = PointNetEncoder().to(device)
    decoder = PointNetDecoder().to(device)
    
    with torch.no_grad():
        gfv = encoder(test_pc)
        reconstructed = decoder(gfv)
        print(f"Encoder output shape: {gfv.shape}")
        print(f"Decoder output shape: {reconstructed.shape}")
    
    print("Individual components test passed!")


if __name__ == "__main__":
    test_autoencoder() 