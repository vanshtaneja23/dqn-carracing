#!/usr/bin/env python3
"""
Fine-tune your DQN agent for F1 tracks
"""

from training import DQNTrainer
import os

def fine_tune_for_f1():
    """Fine-tune your existing agent for F1 performance"""
    
    # Configuration for F1 fine-tuning
    config = {
        'skip_frames': 4,
        'stack_frames': 4,
        'reward_shaping': True,
        'learning_rate': 1e-5,      # Lower learning rate
        'gamma': 0.99,
        'epsilon_start': 0.3,       # Some exploration for new tracks
        'epsilon_end': 0.01,
        'epsilon_decay': 0.99,
        'target_update': 500,       # More frequent updates
        'batch_size': 32,
        'device': 'cpu',
        'num_episodes': 500,        # Shorter fine-tuning
    }
    
    # Create trainer
    trainer = DQNTrainer(config)
    
    # Load your pre-trained weights
    pretrained_path = "runs/dqn_carracing_20251027_184045/checkpoints/best_model.pth"
    if os.path.exists(pretrained_path):
        print("🏎️ Loading your pre-trained CarRacing agent...")
        trainer.agent.load(pretrained_path)
        print("✅ Pre-trained weights loaded!")
    else:
        print("⚠️ Pre-trained model not found. Training from scratch.")
    
    # Fine-tune on F1 tracks
    print("🏁 Starting F1 fine-tuning...")
    trainer.train(config['num_episodes'])
    
    print("🎉 F1 fine-tuning completed!")
    print(f"F1-adapted model saved in: {trainer.save_dir}")

if __name__ == "__main__":
    fine_tune_for_f1()
