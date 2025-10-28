import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import json
import os
from collections import defaultdict
import pandas as pd

from dqn_agent import DQNAgent
from environment import CarRacingWrapper

class TrainingAnalyzer:
    """Analyze DQN training performance and diagnose issues"""
    
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        
    def _load_config(self):
        """Load training configuration"""
        config_path = os.path.join(self.run_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_metrics(self):
        """Load training metrics from tensorboard logs"""
        # This would typically parse tensorboard logs
        # For now, return empty dict - implement based on your logging format
        return {}
    
    def analyze_training_stability(self, rewards, losses, window=100):
        """Analyze training stability and identify failure modes"""
        print("=== Training Stability Analysis ===")
        
        # Reward stability
        reward_variance = np.var(rewards)
        reward_trend = np.polyfit(range(len(rewards)), rewards, 1)[0]
        
        print(f"Reward Variance: {reward_variance:.2f}")
        print(f"Reward Trend (slope): {reward_trend:.4f}")
        
        # Moving statistics
        if len(rewards) > window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            moving_std = pd.Series(rewards).rolling(window).std().dropna()
            
            # Identify unstable periods
            high_variance_periods = np.where(moving_std > np.percentile(moving_std, 90))[0]
            
            print(f"High variance periods: {len(high_variance_periods)} out of {len(moving_std)}")
            
            if len(high_variance_periods) > 0:
                print("Potential instability detected in episodes:", high_variance_periods[:10])
        
        # Loss analysis
        if len(losses) > 0:
            loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]
            print(f"Loss Trend (slope): {loss_trend:.6f}")
            
            if loss_trend > 0:
                print("WARNING: Loss is increasing - potential training instability")
        
        return {
            'reward_variance': reward_variance,
            'reward_trend': reward_trend,
            'high_variance_periods': high_variance_periods if len(rewards) > window else []
        }
    
    def diagnose_failure_modes(self, rewards, episode_lengths):
        """Diagnose common DQN failure modes"""
        print("\n=== Failure Mode Diagnosis ===")
        
        failure_modes = []
        
        # 1. Catastrophic forgetting
        if len(rewards) > 500:
            early_performance = np.mean(rewards[200:300])
            late_performance = np.mean(rewards[-100:])
            
            if late_performance < early_performance * 0.7:
                failure_modes.append("Catastrophic Forgetting")
                print("⚠️  Catastrophic forgetting detected - performance degraded significantly")
        
        # 2. Premature convergence
        recent_rewards = rewards[-200:] if len(rewards) > 200 else rewards
        if len(set(np.round(recent_rewards, 1))) < 5:  # Very low diversity
            failure_modes.append("Premature Convergence")
            print("⚠️  Premature convergence - agent stuck in local optimum")
        
        # 3. Exploration issues
        avg_episode_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) > 100 else np.mean(episode_lengths)
        if avg_episode_length < 50:  # Very short episodes
            failure_modes.append("Poor Exploration")
            print("⚠️  Poor exploration - episodes too short, agent may be stuck")
        
        # 4. Reward scale issues
        reward_range = np.max(rewards) - np.min(rewards)
        if reward_range > 1000:
            failure_modes.append("Reward Scale Issues")
            print("⚠️  Large reward range - consider reward normalization")
        
        # 5. Training instability
        if len(rewards) > 100:
            recent_std = np.std(rewards[-100:])
            if recent_std > 200:
                failure_modes.append("High Variance")
                print("⚠️  High reward variance - training unstable")
        
        if not failure_modes:
            print("✅ No major failure modes detected")
        
        return failure_modes
    
    def network_ablation_analysis(self, model_path):
        """Analyze network components through ablation"""
        print("\n=== Network Ablation Analysis ===")
        
        if not os.path.exists(model_path):
            print("Model file not found for ablation analysis")
            return
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent = DQNAgent(device=device)
        agent.load(model_path)
        
        # Analyze network weights
        conv_weights = []
        fc_weights = []
        
        for name, param in agent.q_network.named_parameters():
            if 'conv' in name and 'weight' in name:
                conv_weights.append(param.data.cpu().numpy().flatten())
            elif 'fc' in name and 'weight' in name:
                fc_weights.append(param.data.cpu().numpy().flatten())
        
        # Weight statistics
        if conv_weights:
            conv_weights_flat = np.concatenate(conv_weights)
            print(f"Conv weights - Mean: {np.mean(conv_weights_flat):.4f}, Std: {np.std(conv_weights_flat):.4f}")
            print(f"Conv weights - Dead neurons: {np.sum(np.abs(conv_weights_flat) < 1e-6)}/{len(conv_weights_flat)}")
        
        if fc_weights:
            fc_weights_flat = np.concatenate(fc_weights)
            print(f"FC weights - Mean: {np.mean(fc_weights_flat):.4f}, Std: {np.std(fc_weights_flat):.4f}")
            print(f"FC weights - Dead neurons: {np.sum(np.abs(fc_weights_flat) < 1e-6)}/{len(fc_weights_flat)}")
    
    def generate_diagnostic_report(self, rewards, losses, episode_lengths, model_path=None):
        """Generate comprehensive diagnostic report"""
        print("=" * 60)
        print("DQN TRAINING DIAGNOSTIC REPORT")
        print("=" * 60)
        
        # Basic statistics
        print(f"Total Episodes: {len(rewards)}")
        print(f"Best Reward: {np.max(rewards):.2f}")
        print(f"Final Average Reward (last 100): {np.mean(rewards[-100:]):.2f}")
        print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
        
        # Stability analysis
        stability_results = self.analyze_training_stability(rewards, losses)
        
        # Failure mode diagnosis
        failure_modes = self.diagnose_failure_modes(rewards, episode_lengths)
        
        # Network analysis
        if model_path:
            self.network_ablation_analysis(model_path)
        
        # Recommendations
        self._generate_recommendations(stability_results, failure_modes, rewards)
        
        return {
            'stability': stability_results,
            'failure_modes': failure_modes,
            'recommendations': self._generate_recommendations(stability_results, failure_modes, rewards)
        }
    
    def _generate_recommendations(self, stability_results, failure_modes, rewards):
        """Generate training improvement recommendations"""
        print("\n=== Recommendations ===")
        
        recommendations = []
        
        # Based on failure modes
        if "Catastrophic Forgetting" in failure_modes:
            recommendations.append("Reduce learning rate or increase target network update frequency")
            print("💡 Reduce learning rate or increase target network update frequency")
        
        if "Premature Convergence" in failure_modes:
            recommendations.append("Increase exploration (higher epsilon decay) or add noise to actions")
            print("💡 Increase exploration (higher epsilon decay) or add noise to actions")
        
        if "Poor Exploration" in failure_modes:
            recommendations.append("Improve reward shaping or increase initial epsilon")
            print("💡 Improve reward shaping or increase initial epsilon")
        
        if "Reward Scale Issues" in failure_modes:
            recommendations.append("Normalize rewards or adjust reward shaping parameters")
            print("💡 Normalize rewards or adjust reward shaping parameters")
        
        if "High Variance" in failure_modes:
            recommendations.append("Increase batch size or reduce learning rate")
            print("💡 Increase batch size or reduce learning rate")
        
        # Based on performance
        final_performance = np.mean(rewards[-100:]) if len(rewards) > 100 else np.mean(rewards)
        
        if final_performance < 200:
            recommendations.append("Performance too low - check reward shaping and network architecture")
            print("💡 Performance too low - check reward shaping and network architecture")
        
        if len(recommendations) == 0:
            print("✅ Training appears stable - consider fine-tuning hyperparameters for better performance")
            recommendations.append("Training stable - consider hyperparameter fine-tuning")
        
        return recommendations
    
    def plot_detailed_analysis(self, rewards, losses, episode_lengths, save_path=None):
        """Create detailed analysis plots"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Reward progression
        axes[0, 0].plot(rewards, alpha=0.7)
        if len(rewards) > 100:
            moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
            axes[0, 0].plot(range(99, len(rewards)), moving_avg, 'r-', linewidth=2, label='Moving Average')
        axes[0, 0].set_title('Reward Progression')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        
        # Reward distribution
        axes[0, 1].hist(rewards, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Reward Distribution')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Frequency')
        
        # Episode lengths
        axes[1, 0].plot(episode_lengths, alpha=0.7)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        
        # Loss progression
        if len(losses) > 0:
            axes[1, 1].plot(losses, alpha=0.7)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
        
        # Reward vs Episode Length scatter
        if len(rewards) == len(episode_lengths):
            axes[2, 0].scatter(episode_lengths, rewards, alpha=0.5)
            axes[2, 0].set_title('Reward vs Episode Length')
            axes[2, 0].set_xlabel('Episode Length')
            axes[2, 0].set_ylabel('Reward')
        
        # Performance over time (binned)
        if len(rewards) > 200:
            bin_size = len(rewards) // 10
            binned_rewards = [np.mean(rewards[i:i+bin_size]) for i in range(0, len(rewards), bin_size)]
            axes[2, 1].plot(binned_rewards, 'o-')
            axes[2, 1].set_title('Performance Over Time (Binned)')
            axes[2, 1].set_xlabel('Training Phase')
            axes[2, 1].set_ylabel('Average Reward')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def analyze_run(run_dir):
    """Analyze a specific training run"""
    analyzer = TrainingAnalyzer(run_dir)
    
    # Load data (you'll need to implement data loading based on your format)
    # For now, using dummy data
    rewards = np.random.randn(1000).cumsum() + 100
    losses = np.exp(-np.arange(500) / 100) + np.random.randn(500) * 0.1
    episode_lengths = np.random.poisson(200, 1000)
    
    # Generate report
    report = analyzer.generate_diagnostic_report(
        rewards, losses, episode_lengths,
        model_path=os.path.join(run_dir, 'checkpoints', 'best_model.pth')
    )
    
    # Create plots
    analyzer.plot_detailed_analysis(rewards, losses, episode_lengths)
    
    return report

if __name__ == "__main__":
    # Example usage
    run_dir = "runs/dqn_carracing_20241027_120000"  # Replace with actual run directory
    if os.path.exists(run_dir):
        analyze_run(run_dir)
    else:
        print(f"Run directory {run_dir} not found")
        print("Available runs:")
        if os.path.exists("runs"):
            for run in os.listdir("runs"):
                print(f"  - {run}")