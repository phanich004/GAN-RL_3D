"""
Test script for the complete RL-GAN-Net model
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.rl_gan_net import RLGANNet


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
    print("\n1. Testing Autoencoder phase...")
    model.set_training_phase("autoencoder")
    results_ae = model(incomplete_pc, mode="training")
    print(f"   Noisy GFV shape: {results_ae['noisy_gfv'].shape}")
    print(f"   Reconstruction shape: {results_ae['ae_reconstruction'].shape}")
    
    print("\n2. Testing LGAN phase...")
    model.set_training_phase("lgan")
    results_lgan = model(incomplete_pc, mode="training")
    print(f"   Clean GFV shape: {results_lgan['clean_gfv'].shape}")
    print(f"   Completed PC shape: {results_lgan['completed_pc'].shape}")
    
    print("\n3. Testing RL Agent phase...")
    model.set_training_phase("rl_agent")
    results_rl = model(incomplete_pc, mode="training")
    print(f"   Z vector shape: {results_rl['z_vector'].shape}")
    print(f"   Completed PC shape: {results_rl['completed_pc'].shape}")
    
    print("\n4. Testing Inference mode...")
    results_inf = model(incomplete_pc, mode="inference")
    print(f"   Hybrid output shape: {results_inf['hybrid_output'].shape}")
    print(f"   RL-GAN score shape: {results_inf['rl_gan_score'].shape}")
    print(f"   AE score shape: {results_inf['ae_score'].shape}")
    
    print("\n✅ RLGANNet test passed!")
    
    # Test saving and loading
    print("\n5. Testing checkpoint save/load...")
    checkpoint_path = "test_checkpoint.pth"
    model.save_checkpoint(checkpoint_path, epoch=1, phase="test")
    
    # Create new model and load checkpoint
    model2 = RLGANNet(config)
    model2.to(model.device)
    epoch, phase = model2.load_checkpoint(checkpoint_path)
    print(f"   Loaded checkpoint: epoch={epoch}, phase={phase}")
    
    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    rl_checkpoint_path = checkpoint_path.replace('.pth', '_rl_agent.pth')
    if os.path.exists(rl_checkpoint_path):
        os.remove(rl_checkpoint_path)
    
    print("✅ Checkpoint save/load test passed!")


if __name__ == "__main__":
    test_rl_gan_net() 