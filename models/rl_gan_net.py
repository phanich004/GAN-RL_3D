"""
Complete RL-GAN-Net model implementation
Combines autoencoder, latent GAN, and RL agent for point cloud completion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import yaml
from pathlib import Path

try:
    from .autoencoder import PointCloudAutoencoder
    from .latent_gan import LatentGAN, LatentGANTrainer
    from .rl_agent import DDPGAgent
    from ..utils.losses import RewardFunction, ChamferLoss, GFVLoss
except ImportError:
    # For direct execution testing
    import sys
    import os
    # Add parent directory to path for utils import
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Add current directory to path for model imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models.autoencoder import PointCloudAutoencoder
    from models.latent_gan import LatentGAN, LatentGANTrainer
    from models.rl_agent import DDPGAgent
    from utils.losses import RewardFunction, ChamferLoss, GFVLoss


class RLGANNet(nn.Module):
    """
    Complete RL-GAN-Net model that combines all components.
    """
    
    def __init__(self, config: Dict):
        super(RLGANNet, self).__init__()
        
        self.config = config
        self.device = torch.device(config['training']['device'])
        
        # Initialize components
        self._init_autoencoder()
        self._init_latent_gan()
        self._init_rl_agent()
        self._init_reward_function()
        
        # Training state
        self.training_phase = "autoencoder"  # autoencoder -> lgan -> rl_agent -> joint
        
    def _init_autoencoder(self):
        """Initialize the autoencoder."""
        ae_config = self.config['model']['autoencoder']
        self.autoencoder = PointCloudAutoencoder(
            input_dim=ae_config['input_dim'],
            latent_dim=ae_config['latent_dim'],
            num_points=ae_config['num_points'],
            encoder_dims=ae_config['encoder_dims'],
            decoder_dims=ae_config['decoder_dims']
        )
    
    def _init_latent_gan(self):
        """Initialize the latent GAN."""
        lgan_config = self.config['model']['lgan']
        self.latent_gan = LatentGAN(
            z_dim=lgan_config['z_dim'],
            latent_dim=lgan_config['latent_dim'],
            generator_dims=lgan_config['generator_dims'],
            discriminator_dims=lgan_config['discriminator_dims']
        )
    
    def _init_rl_agent(self):
        """Initialize the RL agent."""
        rl_config = self.config['model']['rl_agent']
        self.rl_agent = DDPGAgent(
            state_dim=rl_config['state_dim'],
            action_dim=rl_config['action_dim'],
            actor_hidden_dims=rl_config['hidden_dims'],
            critic_hidden_dims=rl_config['hidden_dims'],
            actor_lr=rl_config['actor_lr'],
            critic_lr=rl_config['critic_lr'],
            gamma=rl_config['gamma'],
            tau=rl_config['tau'],
            buffer_size=rl_config['buffer_size'],
            batch_size=rl_config['batch_size'],
            device=self.device
        )
    
    def _init_reward_function(self):
        """Initialize the reward function."""
        loss_config = self.config['loss']
        self.reward_function = RewardFunction(
            w_chamfer=loss_config['w_chamfer'],
            w_gfv=loss_config['w_gfv'],
            w_discriminator=loss_config['w_discriminator']
        )
    
    def encode_point_cloud(self, pc: torch.Tensor) -> torch.Tensor:
        """Encode point cloud to Global Feature Vector."""
        return self.autoencoder.encode(pc)
    
    def decode_gfv(self, gfv: torch.Tensor) -> torch.Tensor:
        """Decode Global Feature Vector to point cloud."""
        return self.autoencoder.decode(gfv)
    
    def generate_clean_gfv(self, z: torch.Tensor) -> torch.Tensor:
        """Generate clean GFV using latent GAN."""
        return self.latent_gan.generate(z)
    
    def select_z_vector(self, noisy_gfv: torch.Tensor) -> torch.Tensor:
        """Select z-vector using RL agent."""
        # Convert to numpy for RL agent
        if noisy_gfv.dim() == 1:
            noisy_gfv = noisy_gfv.unsqueeze(0)
        
        batch_size = noisy_gfv.shape[0]
        z_vectors = []
        
        for i in range(batch_size):
            state = noisy_gfv[i].detach().cpu().numpy()
            z = self.rl_agent.select_action(state, add_noise=True)
            z_vectors.append(z)
        
        return torch.tensor(z_vectors, dtype=torch.float32, device=self.device)
    
    def forward(self, incomplete_pc: torch.Tensor, mode: str = "inference") -> Dict[str, torch.Tensor]:
        """
        Forward pass through RL-GAN-Net.
        
        Args:
            incomplete_pc: Incomplete point cloud (B, N, 3)
            mode: "inference" or "training"
            
        Returns:
            Dictionary containing all intermediate and final results
        """
        results = {}
        
        # Encode incomplete point cloud to noisy GFV
        noisy_gfv = self.encode_point_cloud(incomplete_pc)
        results['noisy_gfv'] = noisy_gfv
        
        if mode == "inference" or self.training_phase in ["rl_agent", "joint"]:
            # Use RL agent to select z-vector
            z_vector = self.select_z_vector(noisy_gfv)
            results['z_vector'] = z_vector
            
            # Generate clean GFV using latent GAN
            clean_gfv = self.generate_clean_gfv(z_vector)
            results['clean_gfv'] = clean_gfv
            
            # Decode to completed point cloud
            completed_pc = self.decode_gfv(clean_gfv)
            results['completed_pc'] = completed_pc
            
            # Also get autoencoder reconstruction for comparison
            ae_reconstruction = self.decode_gfv(noisy_gfv)
            results['ae_reconstruction'] = ae_reconstruction
            
            # Use discriminator to decide between RL-GAN and AE output (hybrid approach)
            if mode == "inference":
                rl_gan_score = self.latent_gan.discriminate(clean_gfv)
                ae_score = self.latent_gan.discriminate(noisy_gfv)
                
                # Choose better reconstruction based on discriminator scores
                better_mask = (rl_gan_score > ae_score).float().unsqueeze(-1).unsqueeze(-1)
                hybrid_output = better_mask * completed_pc + (1 - better_mask) * ae_reconstruction
                results['hybrid_output'] = hybrid_output
                results['rl_gan_score'] = rl_gan_score
                results['ae_score'] = ae_score
        
        elif self.training_phase == "autoencoder":
            # Just autoencoder reconstruction
            ae_reconstruction = self.decode_gfv(noisy_gfv)
            results['ae_reconstruction'] = ae_reconstruction
            
        elif self.training_phase == "lgan":
            # Random z-vector for GAN training
            batch_size = noisy_gfv.shape[0]
            z_vector = torch.randn(batch_size, self.config['model']['lgan']['z_dim'], device=self.device)
            results['z_vector'] = z_vector
            
            clean_gfv = self.generate_clean_gfv(z_vector)
            results['clean_gfv'] = clean_gfv
            
            completed_pc = self.decode_gfv(clean_gfv)
            results['completed_pc'] = completed_pc
        
        return results
    
    def compute_reward(self, pred_pc: torch.Tensor, target_pc: torch.Tensor,
                      pred_gfv: torch.Tensor, target_gfv: torch.Tensor) -> torch.Tensor:
        """Compute reward for RL training."""
        # Get discriminator output for the predicted GFV
        discriminator_output = self.latent_gan.discriminate(pred_gfv)
        
        # Compute reward
        reward = self.reward_function.compute_reward(
            pred_pc, target_pc, pred_gfv, target_gfv, discriminator_output
        )
        
        return reward
    
    def set_training_phase(self, phase: str):
        """Set the current training phase."""
        valid_phases = ["autoencoder", "lgan", "rl_agent", "joint"]
        if phase not in valid_phases:
            raise ValueError(f"Invalid phase {phase}. Must be one of {valid_phases}")
        
        self.training_phase = phase
        
        # Freeze/unfreeze networks based on phase
        if phase == "autoencoder":
            self._set_requires_grad(self.autoencoder, True)
            self._set_requires_grad(self.latent_gan, False)
            
        elif phase == "lgan":
            self._set_requires_grad(self.autoencoder, False)
            self._set_requires_grad(self.latent_gan, True)
            
        elif phase == "rl_agent":
            self._set_requires_grad(self.autoencoder, False)
            self._set_requires_grad(self.latent_gan, False)
            # RL agent parameters are handled separately
            
        elif phase == "joint":
            self._set_requires_grad(self.autoencoder, True)
            self._set_requires_grad(self.latent_gan, True)
    
    def _set_requires_grad(self, module: nn.Module, requires_grad: bool):
        """Set requires_grad for all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = requires_grad
    
    def save_checkpoint(self, filepath: str, epoch: int, phase: str):
        """Save complete model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'config': self.config,
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'latent_gan_state_dict': self.latent_gan.state_dict(),
        }
        
        torch.save(checkpoint, filepath)
        
        # Save RL agent separately (different format)
        rl_filepath = filepath.replace('.pth', '_rl_agent.pth')
        self.rl_agent.save(rl_filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load complete model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        self.latent_gan.load_state_dict(checkpoint['latent_gan_state_dict'])
        
        # Load RL agent separately
        rl_filepath = filepath.replace('.pth', '_rl_agent.pth')
        self.rl_agent.load(rl_filepath)
        
        return checkpoint['epoch'], checkpoint['phase']


class RLGANNetEnvironment:
    """
    Environment for training the RL agent.
    Simulates the point cloud completion task.
    """
    
    def __init__(self, model: RLGANNet, dataset):
        self.model = model
        self.dataset = dataset
        self.current_batch = None
        self.current_step = 0
    
    def reset(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Reset environment with new batch.
        
        Args:
            batch: Dictionary containing 'incomplete' and 'complete' point clouds
            
        Returns:
            Initial state (noisy GFV)
        """
        self.current_batch = batch
        self.current_step = 0
        
        # Encode incomplete point cloud to get initial state
        incomplete_pc = batch['incomplete'].to(self.model.device)
        noisy_gfv = self.model.encode_point_cloud(incomplete_pc)
        
        # Return first item in batch as state
        return noisy_gfv[0].detach().cpu().numpy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: z-vector selected by RL agent
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Convert action to tensor
        z_vector = torch.tensor(action, dtype=torch.float32, device=self.model.device).unsqueeze(0)
        
        # Generate clean GFV using latent GAN
        clean_gfv = self.model.generate_clean_gfv(z_vector)
        
        # Decode to get completed point cloud
        completed_pc = self.model.decode_gfv(clean_gfv)
        
        # Get target point cloud and GFV
        target_pc = self.current_batch['complete'][0:1].to(self.model.device)
        target_gfv = self.model.encode_point_cloud(target_pc)
        
        # Compute reward
        reward = self.model.compute_reward(completed_pc, target_pc, clean_gfv, target_gfv)
        reward_value = reward.item()
        
        # For simplicity, each episode is one step
        done = True
        next_state = clean_gfv[0].detach().cpu().numpy()
        
        info = {
            'completed_pc': completed_pc,
            'target_pc': target_pc,
            'clean_gfv': clean_gfv,
            'target_gfv': target_gfv
        }
        
        self.current_step += 1
        
        return next_state, reward_value, done, info


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_rl_gan_net(config_path: str) -> RLGANNet:
    """Create RL-GAN-Net model from configuration."""
    config = load_config(config_path)
    model = RLGANNet(config)
    return model


def test_rl_gan_net():
    """Test the complete RL-GAN-Net model."""
    # Create a simple config for testing
    config = {
        'model': {
            'autoencoder': {
                'input_dim': 3,
                'latent_dim': 128,
                'num_points': 2048,
                'encoder_dims': [64, 128, 128, 256, 128],
                'decoder_dims': [256, 256, 6144]
            },
            'lgan': {
                'z_dim': 1,
                'latent_dim': 128,
                'generator_dims': [256, 512, 512, 256, 128],
                'discriminator_dims': [128, 256, 512, 256, 1]
            },
            'rl_agent': {
                'state_dim': 128,
                'action_dim': 1,
                'hidden_dims': [400, 400, 300, 300],
                'actor_lr': 1e-4,
                'critic_lr': 1e-3,
                'gamma': 0.99,
                'tau': 0.005,
                'buffer_size': 10000,
                'batch_size': 32
            }
        },
        'training': {
            'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
        },
        'loss': {
            'w_chamfer': 100.0,
            'w_gfv': 10.0,
            'w_discriminator': 0.01
        }
    }
    
    # Create model
    model = RLGANNet(config)
    model.to(model.device)
    
    print("Testing RLGANNet...")
    print(f"Device: {model.device}")
    
    # Test data
    batch_size = 4
    num_points = 2048
    incomplete_pc = torch.randn(batch_size, num_points, 3).to(model.device)
    
    # Test forward pass in different phases
    model.set_training_phase("autoencoder")
    results_ae = model(incomplete_pc, mode="training")
    print(f"Autoencoder phase - noisy GFV shape: {results_ae['noisy_gfv'].shape}")
    print(f"Autoencoder phase - reconstruction shape: {results_ae['ae_reconstruction'].shape}")
    
    model.set_training_phase("lgan")
    results_lgan = model(incomplete_pc, mode="training")
    print(f"LGAN phase - clean GFV shape: {results_lgan['clean_gfv'].shape}")
    print(f"LGAN phase - completed PC shape: {results_lgan['completed_pc'].shape}")
    
    model.set_training_phase("rl_agent")
    results_rl = model(incomplete_pc, mode="training")
    print(f"RL phase - z vector shape: {results_rl['z_vector'].shape}")
    print(f"RL phase - completed PC shape: {results_rl['completed_pc'].shape}")
    
    # Test inference mode
    results_inf = model(incomplete_pc, mode="inference")
    print(f"Inference - hybrid output shape: {results_inf['hybrid_output'].shape}")
    
    print("RLGANNet test passed!")


if __name__ == "__main__":
    test_rl_gan_net() 