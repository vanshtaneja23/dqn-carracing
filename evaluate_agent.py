#!/usr/bin/env python3
"""
Evaluate the trained DQN agent
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from environment import CarRacingWrapper
import os

def load_and_evaluate(model_path, num_episodes=10):
    """Load trained model and evaluate performance"""
    
    # Create environment and agent
    env = CarRacingWrapper(skip_frames=4, stack_frames=4, reward_shaping=True)
    agent = DQNAgent(
        state_shape=(4, 84, 84),
        n_actions=env.action_size,
        device='cpu'
    )
    
    # Load trained model
    print(f"Loading model from: {model_path}")
    agent.load(model_path)
    
    # Evaluate
    print(f"Evaluating for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Use trained policy (no exploration)
            action = agent.get_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1:2d}: Reward = {episode_reward:7.2f}, Length = {episode_length:4d}")
    
    # Statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Best Reward:    {max(episode_rewards):.2f}")
    print(f"Worst Reward:   {min(episode_rewards):.2f}")
    print(f"Average Length: {avg_length:.1f} steps")
    
    # Performance assessment
    if avg_reward > 400:
        print("🎉 EXCELLENT: Agent shows strong performance!")
    elif avg_reward > 200:
        print("✅ GOOD: Agent learned reasonable behavior")
    elif avg_reward > 0:
        print("⚠️  FAIR: Agent shows some learning but needs improvement")
    else:
        print("❌ POOR: Agent needs more training or hyperparameter tuning")
    
    env.close()
    return episode_rewards, episode_lengths

def compare_models(run_dir):
    """Compare different model checkpoints"""
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    
    if not os.path.exists(checkpoints_dir):
        print(f"No checkpoints found in {run_dir}")
        return
    
    # Test key checkpoints
    test_models = ['episode_100.pth', 'episode_500.pth', 'episode_1000.pth', 'best_model.pth']
    
    results = {}
    
    for model_name in test_models:
        model_path = os.path.join(checkpoints_dir, model_name)
        if os.path.exists(model_path):
            print(f"\n{'='*60}")
            print(f"Testing {model_name}")
            print(f"{'='*60}")
            
            try:
                rewards, lengths = load_and_evaluate(model_path, num_episodes=5)
                results[model_name] = {
                    'avg_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'avg_length': np.mean(lengths)
                }
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
    
    # Summary comparison
    if results:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Avg Reward':<12} {'Std Reward':<12} {'Avg Length':<12}")
        print("-" * 60)
        
        for model, stats in results.items():
            print(f"{model:<20} {stats['avg_reward']:>8.1f}    {stats['std_reward']:>8.1f}    {stats['avg_length']:>8.1f}")
    
    return results

def main():
    # Find latest run
    runs_dir = 'runs'
    if not os.path.exists(runs_dir):
        print("No training runs found!")
        return
    
    runs = [d for d in os.listdir(runs_dir) if d.startswith('dqn_carracing')]
    if not runs:
        print("No DQN training runs found!")
        return
    
    latest_run = max(runs, key=lambda x: os.path.getctime(os.path.join(runs_dir, x)))
    run_dir = os.path.join(runs_dir, latest_run)
    
    print(f"Evaluating latest run: {latest_run}")
    
    # Test best model
    best_model_path = os.path.join(run_dir, 'checkpoints', 'best_model.pth')
    if os.path.exists(best_model_path):
        print("\n" + "="*60)
        print("TESTING BEST MODEL")
        print("="*60)
        load_and_evaluate(best_model_path, num_episodes=10)
    
    # Compare different checkpoints
    compare_models(run_dir)

if __name__ == "__main__":
    main()