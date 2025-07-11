"""
Comprehensive training script for RL-GAN-Net
Follows the methodology from the original CVPR 2019 paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, Tuple

# Import our modules
from models.rl_gan_net import RLGANNet, RLGANNetEnvironment
from models.autoencoder import PointCloudAutoencoder
from models.latent_gan import LatentGAN, LatentGANTrainer
from models.rl_agent import DDPGAgent
from utils.dataset import create_dataloader, setup_dataset
from utils.losses import ChamferLoss, GFVLoss, RewardFunction
# from utils.data_utils import save_point_cloud_visualization  # Optional visualization function


class RLGANNetTrainer:
    """
    Comprehensive trainer for RL-GAN-Net following the paper's methodology.
    
    Training Phases:
    1. Train Autoencoder
    2. Generate GFVs using trained autoencoder
    3. Train Latent GAN on GFVs
    4. Train RL Agent
    5. Joint fine-tuning (optional)
    """
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device(self.config['training']['device'])
        print(f"Using device: {self.device}")
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Setup data loaders
        self.setup_data()
        
        # Initialize model
        self.model = RLGANNet(self.config)
        self.model.to(self.device)
        
        # Setup loss functions
        self.setup_losses()
        
        # Training state
        self.current_phase = "autoencoder"
        self.epoch = 0
        
    def setup_directories(self):
        """Setup output directories."""
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config['training']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path(self.config['training']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup tensorboard logging."""
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
    def setup_data(self):
        """Setup data loaders."""
        data_dir = self.config['data']['data_dir']
        
        # Setup dataset if needed
        if not os.path.exists(data_dir):
            print(f"Dataset not found at {data_dir}. Setting up synthetic dataset...")
            setup_dataset(data_dir, synthetic=True)
        
        # Create data loaders
        self.train_loader = create_dataloader(
            data_dir,
            split='train',
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            num_points=self.config['model']['autoencoder']['num_points'],
            augment=True
        )
        
        self.val_loader = create_dataloader(
            data_dir,
            split='test',
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            num_points=self.config['model']['autoencoder']['num_points'],
            augment=False
        )
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def setup_losses(self):
        """Setup loss functions."""
        self.chamfer_loss = ChamferLoss()
        self.gfv_loss = GFVLoss()
        self.reward_function = RewardFunction(
            w_chamfer=self.config['loss']['w_chamfer'],
            w_gfv=self.config['loss']['w_gfv'],
            w_discriminator=self.config['loss']['w_discriminator']
        )
    
    def train_autoencoder(self, num_epochs: int = 100):
        """Phase 1: Train the autoencoder."""
        print("\n" + "="*50)
        print("PHASE 1: Training Autoencoder")
        print("="*50)
        
        self.model.set_training_phase("autoencoder")
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.model.autoencoder.parameters(),
            lr=self.config['training']['autoencoder']['lr'],
            weight_decay=self.config['training']['autoencoder']['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config['training']['autoencoder']['scheduler_step'],
            gamma=self.config['training']['autoencoder']['scheduler_gamma']
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_autoencoder_epoch(optimizer)
            
            # Validation
            val_loss = self.validate_autoencoder()
            
            # Learning rate scheduling
            scheduler.step()
            
            # Logging
            self.writer.add_scalar('AE/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('AE/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('AE/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"autoencoder_best.pth", epoch, "autoencoder")
            
            # Save periodic checkpoint
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f"autoencoder_epoch_{epoch+1}.pth", epoch, "autoencoder")
        
        print(f"Autoencoder training completed. Best validation loss: {best_val_loss:.6f}")
    
    def train_autoencoder_epoch(self, optimizer: optim.Optimizer) -> float:
        """Train autoencoder for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training AE")):
            incomplete_pc = batch['incomplete_pc'].to(self.device)
            complete_pc = batch['complete_pc'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            results = self.model(incomplete_pc, mode="training")
            reconstruction = results['ae_reconstruction']
            
            # Compute loss
            loss = self.chamfer_loss(reconstruction, complete_pc)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                self.writer.add_scalar('AE/Batch_Loss', loss.item(), 
                                     self.epoch * len(self.train_loader) + batch_idx)
        
        return total_loss / len(self.train_loader)
    
    def validate_autoencoder(self) -> float:
        """Validate autoencoder."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating AE"):
                incomplete_pc = batch['incomplete_pc'].to(self.device)
                complete_pc = batch['complete_pc'].to(self.device)
                
                results = self.model(incomplete_pc, mode="training")
                reconstruction = results['ae_reconstruction']
                
                loss = self.chamfer_loss(reconstruction, complete_pc)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def generate_gfvs(self):
        """Phase 2: Generate Global Feature Vectors using trained autoencoder."""
        print("\n" + "="*50)
        print("PHASE 2: Generating Global Feature Vectors")
        print("="*50)
        
        self.model.eval()
        
        # Create directory for GFVs
        gfv_dir = Path(self.config['data']['data_dir']) / 'gfv'
        gfv_dir.mkdir(exist_ok=True)
        
        gfvs_clean = []
        gfvs_noisy = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Generating GFVs")):
                incomplete_pc = batch['incomplete_pc'].to(self.device)
                complete_pc = batch['complete_pc'].to(self.device)
                
                # Get GFVs
                clean_gfv = self.model.encode_point_cloud(complete_pc)
                noisy_gfv = self.model.encode_point_cloud(incomplete_pc)
                
                gfvs_clean.append(clean_gfv.cpu())
                gfvs_noisy.append(noisy_gfv.cpu())
        
        # Save GFVs
        gfvs_clean = torch.cat(gfvs_clean, dim=0)
        gfvs_noisy = torch.cat(gfvs_noisy, dim=0)
        
        torch.save(gfvs_clean, gfv_dir / 'clean_gfvs_train.pt')
        torch.save(gfvs_noisy, gfv_dir / 'noisy_gfvs_train.pt')
        
        print(f"Generated {len(gfvs_clean)} GFV pairs")
        print(f"Clean GFV shape: {gfvs_clean.shape}")
        print(f"Noisy GFV shape: {gfvs_noisy.shape}")
    
    def train_latent_gan(self, num_epochs: int = 200):
        """Phase 3: Train the Latent GAN."""
        print("\n" + "="*50)
        print("PHASE 3: Training Latent GAN")
        print("="*50)
        
        self.model.set_training_phase("lgan")
        
        # Load GFVs
        gfv_dir = Path(self.config['data']['data_dir']) / 'gfv'
        clean_gfvs = torch.load(gfv_dir / 'clean_gfvs_train.pt')
        
        # Setup GAN trainer
        gan_trainer = LatentGANTrainer(
            self.model.latent_gan,
            self.device,
            generator_lr=self.config['training']['lgan']['generator_lr'],
            discriminator_lr=self.config['training']['lgan']['discriminator_lr']
        )
        
        batch_size = self.config['training']['batch_size']
        best_generator_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            
            # Shuffle GFVs
            indices = torch.randperm(len(clean_gfvs))
            clean_gfvs_shuffled = clean_gfvs[indices]
            
            for i in tqdm(range(0, len(clean_gfvs), batch_size), desc=f"GAN Epoch {epoch+1}"):
                batch_gfvs = clean_gfvs_shuffled[i:i+batch_size].to(self.device)
                
                # Random z vectors
                z_dim = self.config['model']['lgan']['z_dim']
                z = torch.randn(len(batch_gfvs), z_dim, device=self.device)
                
                # Train discriminator
                d_loss = gan_trainer.train_discriminator(batch_gfvs, z)
                epoch_d_loss += d_loss
                
                # Train generator
                g_loss = gan_trainer.train_generator(z)
                epoch_g_loss += g_loss
            
            avg_g_loss = epoch_g_loss / (len(clean_gfvs) // batch_size)
            avg_d_loss = epoch_d_loss / (len(clean_gfvs) // batch_size)
            
            # Logging
            self.writer.add_scalar('GAN/Generator_Loss', avg_g_loss, epoch)
            self.writer.add_scalar('GAN/Discriminator_Loss', avg_d_loss, epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs} - G Loss: {avg_g_loss:.6f}, D Loss: {avg_d_loss:.6f}")
            
            # Save best model
            if avg_g_loss < best_generator_loss:
                best_generator_loss = avg_g_loss
                self.save_checkpoint(f"lgan_best.pth", epoch, "lgan")
            
            # Save periodic checkpoint
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(f"lgan_epoch_{epoch+1}.pth", epoch, "lgan")
        
        print(f"Latent GAN training completed. Best generator loss: {best_generator_loss:.6f}")
    
    def train_rl_agent(self, num_episodes: int = 1000):
        """Phase 4: Train the RL Agent."""
        print("\n" + "="*50)
        print("PHASE 4: Training RL Agent")
        print("="*50)
        
        self.model.set_training_phase("rl_agent")
        
        # Create RL environment
        env = RLGANNetEnvironment(self.model, self.train_loader.dataset)
        
        # Training loop
        episode_rewards = []
        best_avg_reward = float('-inf')
        
        for episode in tqdm(range(num_episodes), desc="Training RL"):
            # Sample random batch for this episode
            batch_idx = np.random.randint(len(self.train_loader.dataset))
            sample = self.train_loader.dataset[batch_idx]
            
            batch = {
                'incomplete_pc': sample['incomplete_pc'].unsqueeze(0),
                'complete_pc': sample['complete_pc'].unsqueeze(0)
            }
            
            # Reset environment
            state = env.reset(batch)
            
            episode_reward = 0.0
            done = False
            step = 0
            max_steps = 10  # Limit steps per episode
            
            while not done and step < max_steps:
                # Select action
                action = self.model.rl_agent.select_action(state, add_noise=True)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                self.model.rl_agent.store_experience(state, action, reward, next_state, done)
                
                # Update agent
                if len(self.model.rl_agent.replay_buffer) > self.model.rl_agent.batch_size:
                    actor_loss, critic_loss = self.model.rl_agent.update()
                    
                    # Log losses
                    if step % 5 == 0:
                        self.writer.add_scalar('RL/Actor_Loss', actor_loss, 
                                             episode * max_steps + step)
                        self.writer.add_scalar('RL/Critic_Loss', critic_loss, 
                                             episode * max_steps + step)
                
                state = next_state
                episode_reward += reward
                step += 1
            
            episode_rewards.append(episode_reward)
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                self.writer.add_scalar('RL/Episode_Reward', episode_reward, episode)
                self.writer.add_scalar('RL/Average_Reward', avg_reward, episode)
                
                print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.6f}, Avg: {avg_reward:.6f}")
                
                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.save_checkpoint(f"rl_agent_best.pth", episode, "rl_agent")
            
            # Save periodic checkpoint
            if (episode + 1) % 200 == 0:
                self.save_checkpoint(f"rl_agent_episode_{episode+1}.pth", episode, "rl_agent")
        
        print(f"RL Agent training completed. Best average reward: {best_avg_reward:.6f}")
    
    def joint_training(self, num_epochs: int = 50):
        """Phase 5: Joint fine-tuning of all components."""
        print("\n" + "="*50)
        print("PHASE 5: Joint Fine-tuning")
        print("="*50)
        
        self.model.set_training_phase("joint")
        
        # Setup optimizers for all components
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['joint']['lr'],
            weight_decay=self.config['training']['joint']['weight_decay']
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_joint_epoch(optimizer)
            
            # Validation
            val_loss = self.validate_joint()
            
            # Logging
            self.writer.add_scalar('Joint/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Joint/Val_Loss', val_loss, epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"joint_best.pth", epoch, "joint")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"joint_epoch_{epoch+1}.pth", epoch, "joint")
        
        print(f"Joint training completed. Best validation loss: {best_val_loss:.6f}")
    
    def train_joint_epoch(self, optimizer: optim.Optimizer) -> float:
        """Train all components jointly for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Joint Training")):
            incomplete_pc = batch['incomplete_pc'].to(self.device)
            complete_pc = batch['complete_pc'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            results = self.model(incomplete_pc, mode="training")
            
            # Compute combined loss
            chamfer_loss = self.chamfer_loss(results['completed_pc'], complete_pc)
            
            if 'clean_gfv' in results and 'noisy_gfv' in results:
                gfv_loss = self.gfv_loss(results['clean_gfv'], results['noisy_gfv'])
                loss = chamfer_loss + 0.1 * gfv_loss
            else:
                loss = chamfer_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate_joint(self) -> float:
        """Validate joint model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating Joint"):
                incomplete_pc = batch['incomplete_pc'].to(self.device)
                complete_pc = batch['complete_pc'].to(self.device)
                
                results = self.model(incomplete_pc, mode="inference")
                
                if 'hybrid_output' in results:
                    output = results['hybrid_output']
                else:
                    output = results['completed_pc']
                
                loss = self.chamfer_loss(output, complete_pc)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, filename: str, epoch: int, phase: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        self.model.save_checkpoint(str(checkpoint_path), epoch, phase)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        if checkpoint_path.exists():
            epoch, phase = self.model.load_checkpoint(str(checkpoint_path))
            print(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch}, phase {phase})")
            return epoch, phase
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0, "autoencoder"
    
    def train_full_pipeline(self):
        """Train the complete RL-GAN-Net pipeline."""
        print("Starting RL-GAN-Net training pipeline...")
        print(f"Config: {self.config}")
        
        # Phase 1: Train Autoencoder
        self.train_autoencoder(self.config['training']['autoencoder']['epochs'])
        
        # Phase 2: Generate GFVs
        self.generate_gfvs()
        
        # Phase 3: Train Latent GAN
        self.train_latent_gan(self.config['training']['lgan']['epochs'])
        
        # Phase 4: Train RL Agent
        self.train_rl_agent(self.config['training']['rl_agent']['episodes'])
        
        # Phase 5: Joint Fine-tuning (optional)
        if self.config['training']['joint']['enabled']:
            self.joint_training(self.config['training']['joint']['epochs'])
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE!")
        print("="*50)
        print("All components have been trained successfully.")
        print(f"Checkpoints saved in: {self.checkpoint_dir}")
        print(f"Logs saved in: {self.log_dir}")
    
    def close(self):
        """Clean up resources."""
        self.writer.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL-GAN-Net')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--phase', type=str, choices=['autoencoder', 'lgan', 'rl_agent', 'joint', 'full'],
                       default='full', help='Training phase to run')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Initialize trainer
    trainer = RLGANNetTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    try:
        # Run training based on phase
        if args.phase == 'autoencoder':
            trainer.train_autoencoder(trainer.config['training']['autoencoder']['epochs'])
        elif args.phase == 'lgan':
            trainer.generate_gfvs()
            trainer.train_latent_gan(trainer.config['training']['lgan']['epochs'])
        elif args.phase == 'rl_agent':
            trainer.train_rl_agent(trainer.config['training']['rl_agent']['episodes'])
        elif args.phase == 'joint':
            trainer.joint_training(trainer.config['training']['joint']['epochs'])
        elif args.phase == 'full':
            trainer.train_full_pipeline()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    main() 