"""
RL Agent implementation for RL-GAN-Net using DDPG
Controls the latent GAN by selecting appropriate z-vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Tuple, List, Optional
import copy


class Actor(nn.Module):
    """
    Actor network that maps states (noisy GFV) to actions (z-vector).
    """
    
    def __init__(self, state_dim: int = 128, action_dim: int = 1, 
                 hidden_dims: List[int] = [400, 400, 300, 300],
                 action_bound: float = 1.0):
        super(Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        
        # Build actor network
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        
        # Final layer with tanh activation to bound actions
        layers.extend([
            nn.Linear(in_dim, action_dim),
            nn.Tanh()
        ])
        
        self.actor = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Input state (noisy GFV) of shape (B, state_dim)
            
        Returns:
            Action (z-vector) of shape (B, action_dim)
        """
        action = self.actor(state)
        return action * self.action_bound


class Critic(nn.Module):
    """
    Critic network that estimates Q-values for state-action pairs.
    """
    
    def __init__(self, state_dim: int = 128, action_dim: int = 1,
                 hidden_dims: List[int] = [400, 432, 300, 300]):
        super(Critic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State processing layers
        self.state_fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.state_bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        # Combined state-action processing
        # Note: hidden_dims[1] = 432 = 400 + 32 (from action processing)
        self.action_fc = nn.Linear(action_dim, 32)
        self.combined_fc1 = nn.Linear(hidden_dims[0] + 32, hidden_dims[2])
        self.combined_bn1 = nn.BatchNorm1d(hidden_dims[2])
        
        self.combined_fc2 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.combined_bn2 = nn.BatchNorm1d(hidden_dims[3])
        
        # Output layer
        self.output_fc = nn.Linear(hidden_dims[3], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Input state (noisy GFV) of shape (B, state_dim)
            action: Input action (z-vector) of shape (B, action_dim)
            
        Returns:
            Q-value of shape (B, 1)
        """
        # Process state
        s = F.relu(self.state_bn1(self.state_fc1(state)))
        
        # Process action
        a = F.relu(self.action_fc(action))
        
        # Combine state and action
        combined = torch.cat([s, a], dim=1)
        
        x = F.relu(self.combined_bn1(self.combined_fc1(combined)))
        x = F.relu(self.combined_bn2(self.combined_fc2(x)))
        
        q_value = self.output_fc(x)
        
        return q_value


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experience tuples.
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', 
                                   ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.FloatTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences]).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """
    Ornstein-Uhlenbeck noise for exploration in continuous action spaces.
    """
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.2, dt: float = 1e-2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        """Reset the noise to the mean."""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """Sample noise."""
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state = self.state + dx
        return self.state


class DDPGAgent:
    """
    DDPG Agent for controlling the latent GAN.
    """
    
    def __init__(self, state_dim: int = 128, action_dim: int = 1,
                 actor_hidden_dims: List[int] = [400, 400, 300, 300],
                 critic_hidden_dims: List[int] = [400, 432, 300, 300],
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005,
                 buffer_size: int = 100000, batch_size: int = 64,
                 exploration_noise: float = 0.1, policy_noise: float = 0.2,
                 noise_clip: float = 0.5, policy_delay: int = 2,
                 device: torch.device = torch.device("cpu")):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = device
        
        # Networks
        self.actor = Actor(state_dim, action_dim, actor_hidden_dims).to(device)
        self.critic1 = Critic(state_dim, action_dim, critic_hidden_dims).to(device)
        self.critic2 = Critic(state_dim, action_dim, critic_hidden_dims).to(device)
        
        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Noise
        self.noise = OUNoise(action_dim)
        
        # Training step counter
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: Current state (noisy GFV)
            add_noise: Whether to add exploration noise
            
        Returns:
            Selected action (z-vector)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Set to eval mode to handle BatchNorm with batch size 1
        self.actor.eval()
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # Set back to train mode
        self.actor.train()
        
        if add_noise:
            action += self.exploration_noise * self.noise.sample()
            action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Tuple[float, float]:
        """
        Update the agent's networks.
        
        Returns:
            Tuple of (critic_loss, actor_loss)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        self.training_step += 1
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critics
        critic_loss = self._update_critics(states, actions, rewards, next_states, dones)
        
        # Update actor (delayed)
        actor_loss = 0.0
        if self.training_step % self.policy_delay == 0:
            actor_loss = self._update_actor(states)
            self._update_targets()
        
        return critic_loss, actor_loss
    
    def _update_critics(self, states: torch.Tensor, actions: torch.Tensor,
                       rewards: torch.Tensor, next_states: torch.Tensor,
                       dones: torch.Tensor) -> float:
        """Update critic networks."""
        with torch.no_grad():
            # Target actions with noise
            next_actions = self.actor_target(next_states)
            noise = torch.normal(0, self.policy_noise, size=next_actions.shape).to(self.device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            
            # Target Q-values
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        return (critic1_loss + critic2_loss).item() / 2
    
    def _update_actor(self, states: torch.Tensor) -> float:
        """Update actor network."""
        # Actor loss
        actor_actions = self.actor(states)
        actor_loss = -self.critic1(states, actor_actions).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_targets(self):
        """Update target networks with soft update."""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        """Save agent's networks."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent's networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])


def test_ddpg_agent():
    """Test the DDPG agent implementation."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Parameters
    state_dim = 128
    action_dim = 1
    batch_size = 32
    
    # Create agent
    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, 
                     batch_size=batch_size, device=device)
    
    print("Testing DDPGAgent...")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Test action selection
    test_state = np.random.randn(state_dim)
    action = agent.select_action(test_state)
    print(f"Selected action shape: {action.shape}")
    print(f"Action value: {action}")
    
    # Test experience storage and update
    for i in range(100):
        state = np.random.randn(state_dim)
        action = agent.select_action(state, add_noise=False)  # Disable noise for testing
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False
        
        agent.store_experience(state, action, reward, next_state, done)
    
    print(f"Replay buffer size: {len(agent.replay_buffer)}")
    
    # Test network update
    critic_loss, actor_loss = agent.update()
    print(f"Critic loss: {critic_loss:.4f}")
    print(f"Actor loss: {actor_loss:.4f}")
    
    print("DDPGAgent test passed!")


if __name__ == "__main__":
    test_ddpg_agent() 