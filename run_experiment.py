#!/usr/bin/env python3
"""
Experiment runner for DQN CarRacing with different configurations
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
import argparse

from training import DQNTrainer
from analysis import analyze_run

def run_experiment(config_name, config, num_runs=1):
    """Run experiment with given configuration"""
    print(f"Running experiment: {config_name}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    results = []
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        # Create trainer
        trainer = DQNTrainer(config)
        
        # Train agent
        trainer.train(config['num_episodes'])
        
        # Evaluate performance
        eval_rewards, eval_lengths = trainer.evaluate(num_episodes=10)
        
        # Store results
        result = {
            'run': run + 1,
            'config_name': config_name,
            'final_reward': np.mean(trainer.episode_rewards[-100:]),
            'best_reward': trainer.best_reward,
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'convergence_episode': len(trainer.episode_rewards),
            'run_dir': trainer.save_dir
        }
        results.append(result)
        
        # Close environment
        trainer.env.close()
        
        print(f"Run {run + 1} completed:")
        print(f"  Final Reward: {result['final_reward']:.2f}")
        print(f"  Best Reward: {result['best_reward']:.2f}")
        print(f"  Eval Reward: {result['eval_reward_mean']:.2f} ± {result['eval_reward_std']:.2f}")
    
    return results

def get_baseline_config():
    """Get baseline configuration"""
    return {
        'skip_frames': 4,
        'stack_frames': 4,
        'reward_shaping': True,
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'target_update': 1000,
        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_episodes': 1000,
    }

def get_experiment_configs():
    """Get different experimental configurations"""
    baseline = get_baseline_config()
    
    configs = {
        'baseline': baseline,
        
        'high_lr': {**baseline, 'learning_rate': 5e-4},
        'low_lr': {**baseline, 'learning_rate': 5e-5},
        
        'fast_exploration': {**baseline, 'epsilon_decay': 0.99},
        'slow_exploration': {**baseline, 'epsilon_decay': 0.999},
        
        'large_batch': {**baseline, 'batch_size': 64},
        'small_batch': {**baseline, 'batch_size': 16},
        
        'frequent_target_update': {**baseline, 'target_update': 500},
        'rare_target_update': {**baseline, 'target_update': 2000},
        
        'no_reward_shaping': {**baseline, 'reward_shaping': False},
        
        'optimized': {
            **baseline,
            'learning_rate': 2e-4,
            'epsilon_decay': 0.997,
            'target_update': 750,
            'batch_size': 48,
        }
    }
    
    return configs

def main():
    parser = argparse.ArgumentParser(description='Run DQN CarRacing experiments')
    parser.add_argument('--config', type=str, default='baseline', 
                       help='Configuration to run (baseline, high_lr, etc.)')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs per configuration')
    parser.add_argument('--all', action='store_true',
                       help='Run all configurations')
    parser.add_argument('--analyze', type=str, default=None,
                       help='Analyze specific run directory')
    
    args = parser.parse_args()
    
    if args.analyze:
        print(f"Analyzing run: {args.analyze}")
        analyze_run(args.analyze)
        return
    
    configs = get_experiment_configs()
    
    if args.all:
        # Run all configurations
        all_results = []
        for config_name in configs:
            results = run_experiment(config_name, configs[config_name], args.runs)
            all_results.extend(results)
        
        # Save combined results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'experiment_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nAll experiments completed. Results saved to {results_file}")
        
        # Print summary
        print("\n=== EXPERIMENT SUMMARY ===")
        for config_name in configs:
            config_results = [r for r in all_results if r['config_name'] == config_name]
            if config_results:
                avg_final = np.mean([r['final_reward'] for r in config_results])
                avg_best = np.mean([r['best_reward'] for r in config_results])
                print(f"{config_name:20s}: Final={avg_final:6.1f}, Best={avg_best:6.1f}")
    
    else:
        # Run specific configuration
        if args.config not in configs:
            print(f"Unknown configuration: {args.config}")
            print(f"Available configurations: {list(configs.keys())}")
            return
        
        results = run_experiment(args.config, configs[args.config], args.runs)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'{args.config}_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nExperiment completed. Results saved to {results_file}")

if __name__ == "__main__":
    main()