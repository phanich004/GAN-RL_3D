"""
Dataset loading and processing for RL-GAN-Net
Based on latent_3d_points dataset (ShapeNet subset) from the original paper
"""

import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import json
import pickle
from pathlib import Path
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm

from .data_utils import *


class ShapeNetDataset(Dataset):
    """
    Dataset loader for ShapeNet point clouds from latent_3d_points.
    
    This dataset contains:
    - Complete point clouds (ground truth)
    - Incomplete point clouds (with missing regions)
    - Global Feature Vectors (GFVs) from pre-trained autoencoder
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 num_points: int = 2048,
                 categories: Optional[List[str]] = None,
                 load_gfv: bool = False,
                 augment: bool = True):
        """
        Args:
            data_dir: Path to the dataset directory
            split: 'train', 'test', or 'val'
            num_points: Number of points to sample from each point cloud
            categories: List of ShapeNet categories to load (None for all)
            load_gfv: Whether to load pre-computed Global Feature Vectors
            augment: Whether to apply data augmentation
        """
        super(ShapeNetDataset, self).__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.categories = categories
        self.load_gfv = load_gfv
        self.augment = augment
        
        # ShapeNet category mapping
        self.category_mapping = {
            'airplane': '02691156',
            'car': '02958343',
            'chair': '03001627',
            'lamp': '03636649',
            'sofa': '04256520',
            'table': '04379243',
            'watercraft': '04530566',
            'cabinet': '02933112',
        }
        
        # Load data files
        self.data_files = self._load_data_files()
        
        print(f"Loaded {len(self.data_files)} {split} samples")
    
    def _load_data_files(self) -> List[Dict]:
        """Load and filter data files based on split and categories."""
        data_files = []
        
        # Check if we have the processed dataset
        processed_file = self.data_dir / f"{self.split}_data.json"
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                data_files = json.load(f)
        else:
            # Scan directory structure
            data_files = self._scan_directory()
            
            # Save processed file list
            with open(processed_file, 'w') as f:
                json.dump(data_files, f, indent=2)
        
        # Filter by categories if specified
        if self.categories:
            category_ids = [self.category_mapping.get(cat, cat) for cat in self.categories]
            data_files = [f for f in data_files if any(cat_id in f['path'] for cat_id in category_ids)]
        
        return data_files
    
    def _scan_directory(self) -> List[Dict]:
        """Scan directory structure to find point cloud files."""
        data_files = []
        
        # Look for .ply, .pts, .txt, or .h5 files
        for ext in ['*.ply', '*.pts', '*.txt', '*.h5']:
            for file_path in self.data_dir.rglob(ext):
                if self.split in str(file_path).lower():
                    data_files.append({
                        'path': str(file_path),
                        'category': self._extract_category(file_path),
                        'model_id': self._extract_model_id(file_path)
                    })
        
        return data_files
    
    def _extract_category(self, file_path: Path) -> str:
        """Extract category from file path."""
        path_parts = file_path.parts
        for part in path_parts:
            if part in self.category_mapping.values():
                # Find category name from ID
                for name, cat_id in self.category_mapping.items():
                    if cat_id == part:
                        return name
            elif part in self.category_mapping.keys():
                return part
        return 'unknown'
    
    def _extract_model_id(self, file_path: Path) -> str:
        """Extract model ID from file path."""
        return file_path.stem
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
            - complete_pc: Complete point cloud (N, 3)
            - incomplete_pc: Incomplete point cloud (M, 3) where M < N
            - gfv: Global Feature Vector if load_gfv=True (128,)
            - category: Category label
            - model_id: Model identifier
        """
        data_file = self.data_files[idx]
        
        # Load complete point cloud
        complete_pc = self._load_point_cloud(data_file['path'])
        
        # Sample to fixed number of points
        if len(complete_pc) > self.num_points:
            indices = np.random.choice(len(complete_pc), self.num_points, replace=False)
            complete_pc = complete_pc[indices]
        elif len(complete_pc) < self.num_points:
            # Pad with random points if too few
            padding_needed = self.num_points - len(complete_pc)
            padding = complete_pc[np.random.choice(len(complete_pc), padding_needed)]
            complete_pc = np.concatenate([complete_pc, padding], axis=0)
        
        # Generate incomplete point cloud
        incomplete_pc = self._create_incomplete_pc(complete_pc)
        
        # Apply data augmentation
        if self.augment and self.split == 'train':
            complete_pc = self._augment_point_cloud(complete_pc)
            incomplete_pc = self._augment_point_cloud(incomplete_pc)
        
        # Normalize point clouds
        complete_pc = normalize_point_cloud(complete_pc)
        incomplete_pc = normalize_point_cloud(incomplete_pc)
        
        result = {
            'complete_pc': torch.FloatTensor(complete_pc),
            'incomplete_pc': torch.FloatTensor(incomplete_pc),
            'category': data_file['category'],
            'model_id': data_file['model_id']
        }
        
        # Load GFV if requested
        if self.load_gfv:
            gfv_path = self._get_gfv_path(data_file['path'])
            if os.path.exists(gfv_path):
                with open(gfv_path, 'rb') as f:
                    gfv = pickle.load(f)
                result['gfv'] = torch.FloatTensor(gfv)
        
        return result
    
    def _load_point_cloud(self, file_path: str) -> np.ndarray:
        """Load point cloud from various file formats."""
        file_path = Path(file_path)
        
        if file_path.suffix == '.ply':
            return self._load_ply(file_path)
        elif file_path.suffix == '.pts':
            return self._load_pts(file_path)
        elif file_path.suffix == '.txt':
            return self._load_txt(file_path)
        elif file_path.suffix == '.h5':
            return self._load_h5(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_ply(self, file_path: Path) -> np.ndarray:
        """Load PLY file."""
        # Simple PLY loader - for more complex PLY files, use Open3D
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find header end
        header_end = 0
        for i, line in enumerate(lines):
            if line.strip() == 'end_header':
                header_end = i + 1
                break
        
        # Parse vertex data
        vertices = []
        for line in lines[header_end:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    vertices.append([x, y, z])
                except ValueError:
                    continue
        
        return np.array(vertices)
    
    def _load_pts(self, file_path: Path) -> np.ndarray:
        """Load PTS file."""
        return np.loadtxt(file_path)[:, :3]  # Take first 3 columns (x, y, z)
    
    def _load_txt(self, file_path: Path) -> np.ndarray:
        """Load TXT file."""
        return np.loadtxt(file_path)[:, :3]
    
    def _load_h5(self, file_path: Path) -> np.ndarray:
        """Load H5 file."""
        with h5py.File(file_path, 'r') as f:
            if 'data' in f:
                points = f['data'][:]
            elif 'points' in f:
                points = f['points'][:]
            else:
                # Take first dataset
                key = list(f.keys())[0]
                points = f[key][:]
        
        return points.reshape(-1, 3)
    
    def _create_incomplete_pc(self, complete_pc: np.ndarray) -> np.ndarray:
        """Create incomplete point cloud by removing regions."""
        # Random removal strategy from the paper
        removal_ratio = np.random.uniform(0.2, 0.5)  # Remove 20-50% of points
        num_keep = int(len(complete_pc) * (1 - removal_ratio))
        
        # Method 1: Random removal
        if np.random.random() < 0.5:
            indices = np.random.choice(len(complete_pc), num_keep, replace=False)
            return complete_pc[indices]
        
        # Method 2: Spatial removal (remove a region)
        else:
            # Choose random center point
            center_idx = np.random.randint(len(complete_pc))
            center = complete_pc[center_idx]
            
            # Calculate distances
            distances = np.linalg.norm(complete_pc - center, axis=1)
            
            # Remove points within a sphere
            radius = np.percentile(distances, removal_ratio * 100)
            keep_mask = distances > radius
            
            return complete_pc[keep_mask]
    
    def _augment_point_cloud(self, pc: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Convert to torch tensor for augmentation functions
        pc_tensor = torch.FloatTensor(pc)
        
        # Random rotation
        if np.random.random() < 0.5:
            pc_tensor = rotate_point_cloud(pc_tensor)
        
        # Random jittering
        if np.random.random() < 0.5:
            pc_tensor = jitter_point_cloud(pc_tensor)
        
        # Random scaling
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.8, 1.2)
            pc_tensor = pc_tensor * scale
        
        # Convert back to numpy
        return pc_tensor.numpy()
    
    def _get_gfv_path(self, pc_path: str) -> str:
        """Get path to corresponding GFV file."""
        pc_path = Path(pc_path)
        gfv_dir = pc_path.parent / 'gfv'
        gfv_path = gfv_dir / f"{pc_path.stem}.pkl"
        return str(gfv_path)


class DatasetDownloader:
    """Download and setup the latent_3d_points dataset."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_shapenet_subset(self):
        """Download ShapeNet subset used in latent_3d_points."""
        print("Setting up ShapeNet dataset for RL-GAN-Net...")
        
        # Instructions for manual download (as the dataset requires registration)
        print("""
        To use the exact dataset from the RL-GAN-Net paper:
        
        1. Visit: https://github.com/optas/latent_3d_points
        2. Follow their instructions to download the ShapeNet subset
        3. Extract the data to: {self.data_dir}
        
        The dataset should contain:
        - Point cloud files (.ply format)
        - Train/test splits
        - Categories: airplane, car, chair, lamp, sofa, table, watercraft, cabinet
        
        Alternatively, you can use synthetic data for testing by running:
        python -c "from utils.dataset import DatasetDownloader; d = DatasetDownloader('{self.data_dir}'); d.create_synthetic_data()"
        """)
    
    def create_synthetic_data(self, num_samples_per_category: int = 100):
        """Create synthetic point cloud data for testing."""
        print(f"Creating synthetic dataset with {num_samples_per_category} samples per category...")
        
        categories = ['airplane', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft', 'cabinet']
        
        for split in ['train', 'test', 'val']:
            split_dir = self.data_dir / split
            split_dir.mkdir(exist_ok=True)
            
            for category in categories:
                cat_dir = split_dir / category
                cat_dir.mkdir(exist_ok=True)
                
                num_samples = num_samples_per_category if split == 'train' else num_samples_per_category // 4
                
                for i in tqdm(range(num_samples), desc=f"Creating {split}/{category}"):
                    # Generate synthetic point cloud
                    pc = self._generate_synthetic_shape(category)
                    
                    # Save as text file
                    file_path = cat_dir / f"{category}_{i:04d}.txt"
                    np.savetxt(file_path, pc)
        
        print("Synthetic dataset created successfully!")
    
    def _generate_synthetic_shape(self, category: str, num_points: int = 2048) -> np.ndarray:
        """Generate synthetic point cloud for a given category."""
        if category == 'airplane':
            # Airplane-like shape (elongated with wings)
            body = np.random.randn(num_points//2, 3) * [2, 0.3, 0.3]
            wings = np.random.randn(num_points//2, 3) * [0.5, 2, 0.1]
            wings[:, 0] += 0.5  # Offset wings
            pc = np.concatenate([body, wings])
            
        elif category == 'car':
            # Car-like shape (rectangular)
            pc = np.random.randn(num_points, 3) * [2, 1, 0.8]
            
        elif category == 'chair':
            # Chair-like shape (seat + backrest)
            seat = np.random.randn(num_points//2, 3) * [1, 1, 0.1]
            backrest = np.random.randn(num_points//2, 3) * [1, 0.1, 1]
            backrest[:, 1] += 0.5  # Offset backrest
            pc = np.concatenate([seat, backrest])
            
        else:
            # Default random shape
            pc = np.random.randn(num_points, 3)
        
        # Sample to exact number of points
        if len(pc) > num_points:
            indices = np.random.choice(len(pc), num_points, replace=False)
            pc = pc[indices]
        
        return pc


def shapenet_collate_fn(batch):
    """Custom collate function for ShapeNet data."""
    result = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            if key == 'incomplete_pc':
                # Handle variable-length incomplete point clouds
                max_points = max(item[key].shape[0] for item in batch)
                batch_size = len(batch)
                padded_tensors = []
                
                for item in batch:
                    pc = item[key]
                    if pc.shape[0] < max_points:
                        # Pad with zeros or repeat last points
                        padding_needed = max_points - pc.shape[0]
                        if pc.shape[0] > 0:
                            # Repeat random points to pad
                            pad_indices = torch.randint(0, pc.shape[0], (padding_needed,))
                            pad_points = pc[pad_indices]
                            pc_padded = torch.cat([pc, pad_points], dim=0)
                        else:
                            # If empty, create zero points
                            pc_padded = torch.zeros(max_points, 3)
                    else:
                        pc_padded = pc
                    padded_tensors.append(pc_padded)
                
                result[key] = torch.stack(padded_tensors)
            else:
                # Normal stacking for fixed-size tensors
                result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


def create_dataloader(data_dir: str,
                     split: str = 'train',
                     batch_size: int = 32,
                     num_workers: int = 4,
                     **dataset_kwargs) -> DataLoader:
    """Create DataLoader for RL-GAN-Net training."""
    
    dataset = ShapeNetDataset(data_dir, split=split, **dataset_kwargs)
    
    # Disable pin_memory for MPS to avoid warnings
    pin_memory = False  # MPS doesn't support pinned memory yet
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=shapenet_collate_fn,
        pin_memory=pin_memory
    )


def setup_dataset(data_dir: str, synthetic: bool = False):
    """Setup the dataset for RL-GAN-Net training."""
    downloader = DatasetDownloader(data_dir)
    
    if synthetic:
        downloader.create_synthetic_data()
    else:
        downloader.download_shapenet_subset()
    
    print(f"Dataset setup complete in: {data_dir}")


# Example usage and testing
if __name__ == "__main__":
    # Test dataset loading
    data_dir = "./data/shapenet"
    
    # Create synthetic data for testing
    setup_dataset(data_dir, synthetic=True)
    
    # Test dataloader
    train_loader = create_dataloader(data_dir, split='train', batch_size=4)
    
    # Load a batch
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("Complete PC shape:", batch['complete_pc'].shape)
    print("Incomplete PC shape:", batch['incomplete_pc'].shape)
    print("Categories:", batch['category']) 