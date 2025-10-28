import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm
import json

from dqn_agent import DQNAgent
from environment import CarRacingWrapper

class DQNTrainer:
    """Training manager for DQN agent"""
    
    def __init__(self, config):
        self.config = config
        
        # Create environment
        self.env = CarRacingWrapper(
            skip_frames=config['skip_frames'],
            stack_frames=config['stack_frames'],
            reward_shaping=config['reward_shaping']
        )
        
        # Create agent
        self.agent = DQNAgent(
            state_shape=(config['stack_frames'], 84, 84),
            n_actions=self.env.action_size,
            lr=config['learning_rate'],
            gamma=config['gamma'],
            epsilon_start=config['epsilon_start'],
            epsilon_end=config['epsilon_end'],
            epsilon_decay=config['epsilon_decay'],
            target_update=config['target_update'],
            device=config['device']
        )
        
        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.best_reward = -float('inf')
        
        # Create directories
        self.run_name = f"dqn_carracing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = f"runs/{self.run_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/checkpoints", exist_ok=True)
        
        # Tensorboard logging
        self.writer = SummaryWriter(f"{self.save_dir}/logs")
        
        # Save config
        with open(f"{self.save_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def train(self, num_episodes):
        """Main training loop"""
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Device: {self.agent.device}")
        print(f"Run directory: {self.save_dir}")
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            episode_reward, episode_length, avg_loss = self._train_episode()
            
            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if avg_loss is not None:
                self.losses.append(avg_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('Reward/Episode', episode_reward, episode)
            self.writer.add_scalar('Length/Episode', episode_length, episode)
            self.writer.add_scalar('Epsilon', self.agent.epsilon, episode)
            if avg_loss is not None:
                self.writer.add_scalar('Loss/Episode', avg_loss, episode)
            
            # Save best model
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.agent.save(f"{self.save_dir}/checkpoints/best_model.pth")
            
            # Periodic saves and logging
            if (episode + 1) % 100 == 0:
                self._log_progress(episode + 1)
                self.agent.save(f"{self.save_dir}/checkpoints/episode_{episode + 1}.pth")
            
            # Early stopping check
            if self._check_convergence(episode):
                print(f"Training converged at episode {episode + 1}")
                break
        
        self._save_training_plots()
        self.writer.close()
        print(f"Training completed! Best reward: {self.best_reward:.2f}")
    
    def _train_episode(self):
        """Train for one episode"""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        losses = []
        
        while True:
            # Select action
            action = self.agent.get_action(state, training=True)
            
            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            
            # Store experience
            self.agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            loss = self.agent.train_step(batch_size=self.config['batch_size'])
            if loss is not None:
                losses.append(loss)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        avg_loss = np.mean(losses) if losses else None
        return episode_reward, episode_length, avg_loss
    
    def _log_progress(self, episode):
        """Log training progress"""
        recent_rewards = self.episode_rewards[-100:]
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(self.episode_lengths[-100:])
        
        print(f"\nEpisode {episode}")
        print(f"Average Reward (last 100): {avg_reward:.2f}")
        print(f"Average Length (last 100): {avg_length:.1f}")
        print(f"Best Reward: {self.best_reward:.2f}")
        print(f"Epsilon: {self.agent.epsilon:.4f}")
        
        if len(self.losses) > 0:
            print(f"Average Loss: {np.mean(self.losses[-100:]):.6f}")
    
    def _check_convergence(self, episode):
        """Check if training has converged"""
        if episode < 200:  # Need minimum episodes
            return False
        
        recent_rewards = self.episode_rewards[-100:]
        if len(recent_rewards) < 100:
            return False
        
        # Check if average reward is stable and high
        avg_reward = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)
        
        # Convergence criteria
        return avg_reward > 500 and reward_std < 50
    
    def _save_training_plots(self):
        """Save training progress plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Moving average of rewards
        if len(self.episode_rewards) > 100:
            moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title('Moving Average Reward (100 episodes)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
        
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        
        # Training loss
        if len(self.losses) > 0:
            axes[1, 1].plot(self.losses)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, num_episodes=10, render=False):
        """Evaluate trained agent"""
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.agent.get_action(state, training=False)
                state, reward, done, _ = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    self.env.render()
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        avg_length = np.mean(eval_lengths)
        
        print(f"\nEvaluation Results:")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Average Length: {avg_length:.1f}")
        
        return eval_rewards, eval_lengths

def main():
    """Main training function"""
    config = {
        # Environment settings
        'skip_frames': 4,
        'stack_frames': 4,
        'reward_shaping': True,
        
        # Agent settings
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'target_update': 1000,
        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Training settings
        'num_episodes': 2000,
    }
    
    # Create trainer
    trainer = DQNTrainer(config)
    
    # Train agent
    trainer.train(config['num_episodes'])
    
    # Evaluate final performance
    trainer.evaluate(num_episodes=10)
    
    # Close environment
    trainer.env.close()

if __name__ == "__main__":
    main()