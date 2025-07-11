"""
Data utilities for RL-GAN-Net
Handles point cloud processing, normalization, augmentation, and dataset operations
"""

import torch
import numpy as np
import random
from typing import Tuple, Optional, List
import h5py
import os
from torch.utils.data import Dataset


def normalize_point_cloud(pc):
    """
    Normalize point cloud to unit sphere centered at origin.
    
    Args:
        pc: Point cloud (numpy array or torch tensor) of shape (N, 3) or (B, N, 3)
        
    Returns:
        Normalized point cloud (same type as input)
    """
    is_numpy = isinstance(pc, np.ndarray)
    
    # Convert to torch if numpy
    if is_numpy:
        pc_tensor = torch.from_numpy(pc).float()
    else:
        pc_tensor = pc
    
    if pc_tensor.dim() == 2:
        # Single point cloud (N, 3)
        centroid = torch.mean(pc_tensor, dim=0, keepdim=True)
        pc_centered = pc_tensor - centroid
        scale = torch.max(torch.norm(pc_centered, dim=1))
        if scale > 0:
            pc_normalized = pc_centered / scale
        else:
            pc_normalized = pc_centered
    else:
        # Batch of point clouds (B, N, 3)
        centroid = torch.mean(pc_tensor, dim=1, keepdim=True)  # (B, 1, 3)
        pc_centered = pc_tensor - centroid
        scale = torch.max(torch.norm(pc_centered, dim=2), dim=1, keepdim=True)[0].unsqueeze(-1)  # (B, 1, 1)
        pc_normalized = pc_centered / scale
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        return pc_normalized.numpy()
    else:
        return pc_normalized


def center_point_cloud(pc: torch.Tensor) -> torch.Tensor:
    """
    Center point cloud at origin.
    
    Args:
        pc: Point cloud tensor of shape (N, 3) or (B, N, 3)
        
    Returns:
        Centered point cloud
    """
    if pc.dim() == 2:
        centroid = torch.mean(pc, dim=0, keepdim=True)
    else:
        centroid = torch.mean(pc, dim=1, keepdim=True)
    
    return pc - centroid


def random_rotation_matrix() -> np.ndarray:
    """Generate random 3D rotation matrix."""
    theta = np.random.uniform(0, 2 * np.pi, 3)
    
    # Rotation around x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta[0]), -np.sin(theta[0])],
        [0, np.sin(theta[0]), np.cos(theta[0])]
    ])
    
    # Rotation around y-axis
    Ry = np.array([
        [np.cos(theta[1]), 0, np.sin(theta[1])],
        [0, 1, 0],
        [-np.sin(theta[1]), 0, np.cos(theta[1])]
    ])
    
    # Rotation around z-axis
    Rz = np.array([
        [np.cos(theta[2]), -np.sin(theta[2]), 0],
        [np.sin(theta[2]), np.cos(theta[2]), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx


def rotate_point_cloud(pc: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Rotate point cloud by random or given rotation matrix.
    
    Args:
        pc: Point cloud tensor of shape (N, 3) or (B, N, 3)
        rotation_matrix: Optional rotation matrix. If None, random rotation is applied.
        
    Returns:
        Rotated point cloud
    """
    if rotation_matrix is None:
        rotation_matrix = torch.tensor(random_rotation_matrix(), dtype=pc.dtype, device=pc.device)
    
    if pc.dim() == 2:
        # Single point cloud (N, 3)
        return pc @ rotation_matrix.T
    else:
        # Batch of point clouds (B, N, 3)
        batch_size = pc.shape[0]
        if rotation_matrix.dim() == 2:
            # Same rotation for all in batch
            rotation_matrix = rotation_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        return torch.bmm(pc, rotation_matrix.transpose(-2, -1))


def jitter_point_cloud(pc: torch.Tensor, sigma: float = 0.01, clip: float = 0.05) -> torch.Tensor:
    """
    Add random jitter to point cloud.
    
    Args:
        pc: Point cloud tensor of shape (N, 3) or (B, N, 3)
        sigma: Standard deviation of Gaussian noise
        clip: Clipping value for noise
        
    Returns:
        Jittered point cloud
    """
    noise = torch.normal(0, sigma, size=pc.shape, device=pc.device)
    noise = torch.clamp(noise, -clip, clip)
    return pc + noise


def random_scale_point_cloud(pc: torch.Tensor, scale_low: float = 0.8, scale_high: float = 1.2) -> torch.Tensor:
    """
    Randomly scale point cloud.
    
    Args:
        pc: Point cloud tensor of shape (N, 3) or (B, N, 3)
        scale_low: Minimum scale factor
        scale_high: Maximum scale factor
        
    Returns:
        Scaled point cloud
    """
    if pc.dim() == 2:
        scale = random.uniform(scale_low, scale_high)
    else:
        batch_size = pc.shape[0]
        scale = torch.FloatTensor(batch_size, 1, 1).uniform_(scale_low, scale_high).to(pc.device)
    
    return pc * scale


def create_incomplete_point_cloud(pc: torch.Tensor, missing_ratio: float = 0.5, 
                                 method: str = "random") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create incomplete point cloud by removing points.
    
    Args:
        pc: Complete point cloud tensor of shape (N, 3) or (B, N, 3)
        missing_ratio: Ratio of points to remove (0-1)
        method: Method for point removal ("random", "sphere", "plane")
        
    Returns:
        Tuple of (incomplete_pc, mask) where mask indicates kept points
    """
    if pc.dim() == 2:
        # Single point cloud
        N = pc.shape[0]
        n_keep = int(N * (1 - missing_ratio))
        
        if method == "random":
            indices = torch.randperm(N)[:n_keep]
        elif method == "sphere":
            # Remove points within a sphere
            center = torch.mean(pc, dim=0)
            distances = torch.norm(pc - center, dim=1)
            _, indices = torch.topk(distances, n_keep, largest=True)
        elif method == "plane":
            # Remove points on one side of a random plane
            normal = torch.randn(3)
            normal = normal / torch.norm(normal)
            plane_point = torch.mean(pc, dim=0)
            distances = torch.sum((pc - plane_point) * normal, dim=1)
            _, indices = torch.topk(distances, n_keep, largest=True)
        
        mask = torch.zeros(N, dtype=torch.bool)
        mask[indices] = True
        incomplete_pc = pc[mask]
        
    else:
        # Batch of point clouds
        B, N, _ = pc.shape
        n_keep = int(N * (1 - missing_ratio))
        
        incomplete_pc_list = []
        mask_list = []
        
        for i in range(B):
            single_pc = pc[i]
            single_incomplete, single_mask = create_incomplete_point_cloud(
                single_pc, missing_ratio, method
            )
            incomplete_pc_list.append(single_incomplete)
            mask_list.append(single_mask)
        
        # Pad to same size
        max_points = max(pc.shape[1] for pc in incomplete_pc_list)
        incomplete_pc = torch.zeros(B, max_points, 3, device=pc.device)
        mask = torch.zeros(B, N, dtype=torch.bool, device=pc.device)
        
        for i, (inc_pc, m) in enumerate(zip(incomplete_pc_list, mask_list)):
            incomplete_pc[i, :inc_pc.shape[0]] = inc_pc
            mask[i] = m
    
    return incomplete_pc, mask


class PointCloudDataset(Dataset):
    """
    Point cloud dataset for training RL-GAN-Net.
    """
    
    def __init__(self, data_path: str, split: str = "train", num_points: int = 2048,
                 augment: bool = True, missing_ratio: float = 0.5):
        """
        Args:
            data_path: Path to dataset
            split: Data split ("train", "val", "test")
            num_points: Number of points per cloud
            augment: Whether to apply augmentation
            missing_ratio: Ratio of points to remove for incomplete clouds
        """
        self.data_path = data_path
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.missing_ratio = missing_ratio
        
        # Load data
        self.data = self._load_data()
    
    def _load_data(self) -> List[torch.Tensor]:
        """Load point cloud data from files."""
        data_file = os.path.join(self.data_path, f"{self.split}.h5")
        
        if not os.path.exists(data_file):
            # Generate synthetic data for testing
            print(f"Data file {data_file} not found. Generating synthetic data...")
            return self._generate_synthetic_data()
        
        with h5py.File(data_file, 'r') as f:
            point_clouds = f['point_clouds'][:]
        
        return [torch.tensor(pc, dtype=torch.float32) for pc in point_clouds]
    
    def _generate_synthetic_data(self) -> List[torch.Tensor]:
        """Generate synthetic point cloud data for testing."""
        n_samples = 1000 if self.split == "train" else 200
        data = []
        
        for _ in range(n_samples):
            # Generate random point clouds (sphere, cube, etc.)
            shape_type = random.choice(["sphere", "cube", "cylinder"])
            
            if shape_type == "sphere":
                # Random points on sphere
                theta = torch.rand(self.num_points) * 2 * np.pi
                phi = torch.acos(1 - 2 * torch.rand(self.num_points))
                x = torch.sin(phi) * torch.cos(theta)
                y = torch.sin(phi) * torch.sin(theta)
                z = torch.cos(phi)
                pc = torch.stack([x, y, z], dim=1)
                
            elif shape_type == "cube":
                # Random points on cube surface
                pc = torch.rand(self.num_points, 3) * 2 - 1
                # Project to cube surface
                max_coord = torch.max(torch.abs(pc), dim=1, keepdim=True)[0]
                pc = pc / max_coord
                
            else:  # cylinder
                # Random points on cylinder
                theta = torch.rand(self.num_points) * 2 * np.pi
                height = torch.rand(self.num_points) * 2 - 1
                x = torch.cos(theta)
                y = torch.sin(theta)
                z = height
                pc = torch.stack([x, y, z], dim=1)
            
            # Add some noise
            pc += torch.randn_like(pc) * 0.02
            data.append(pc)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pc = self.data[idx].clone()
        
        # Ensure correct number of points
        if pc.shape[0] > self.num_points:
            # Random sampling
            indices = torch.randperm(pc.shape[0])[:self.num_points]
            pc = pc[indices]
        elif pc.shape[0] < self.num_points:
            # Pad with random points
            n_pad = self.num_points - pc.shape[0]
            pad_points = pc[torch.randint(0, pc.shape[0], (n_pad,))]
            pc = torch.cat([pc, pad_points], dim=0)
        
        # Data augmentation
        if self.augment and self.split == "train":
            pc = jitter_point_cloud(pc)
            pc = rotate_point_cloud(pc)
            pc = random_scale_point_cloud(pc)
        
        # Normalize
        pc = normalize_point_cloud(pc)
        
        # Create incomplete version
        incomplete_pc, mask = create_incomplete_point_cloud(pc, self.missing_ratio)
        
        return {
            "complete": pc,
            "incomplete": incomplete_pc,
            "mask": mask
        }


def collate_fn(batch):
    """Custom collate function for point cloud batches."""
    complete_pcs = torch.stack([item["complete"] for item in batch])
    
    # Handle variable-size incomplete point clouds
    max_incomplete_points = max(item["incomplete"].shape[0] for item in batch)
    batch_size = len(batch)
    
    incomplete_pcs = torch.zeros(batch_size, max_incomplete_points, 3)
    masks = torch.stack([item["mask"] for item in batch])
    
    for i, item in enumerate(batch):
        n_points = item["incomplete"].shape[0]
        incomplete_pcs[i, :n_points] = item["incomplete"]
    
    return {
        "complete": complete_pcs,
        "incomplete": incomplete_pcs,
        "mask": masks
    } 