#!/usr/bin/env python3
"""
Test script to verify the DQN setup works correctly
"""

import torch
import numpy as np
import sys
import traceback

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✅ Tensorboard")
    except ImportError as e:
        print(f"❌ Tensorboard import failed: {e}")
        return False
    
    try:
        import gymnasium as gym
        print(f"✅ Gymnasium")
    except ImportError as e:
        print(f"❌ Gymnasium import failed: {e}")
        return False
    
    return True

def test_dqn_agent():
    """Test DQN agent creation"""
    print("\nTesting DQN Agent...")
    
    try:
        from dqn_agent import DQNAgent, DQNNetwork, ExperienceReplay
        
        # Test network creation
        network = DQNNetwork(input_channels=4, n_actions=5)
        print("✅ DQN Network created")
        
        # Test agent creation
        agent = DQNAgent(
            state_shape=(4, 84, 84),
            n_actions=5,
            device='cpu'  # Use CPU for testing
        )
        print("✅ DQN Agent created")
        
        # Test experience replay
        replay = ExperienceReplay(capacity=1000)
        print("✅ Experience Replay created")
        
        return True
        
    except Exception as e:
        print(f"❌ DQN Agent test failed: {e}")
        traceback.print_exc()
        return False

def test_environment():
    """Test environment wrapper"""
    print("\nTesting Environment...")
    
    try:
        from environment import CarRacingWrapper
        
        # Real CarRacing-v3 only: the wrapper raises if Box2D is missing instead of mocking
        env = CarRacingWrapper(skip_frames=2, stack_frames=2)
        assert not env.is_mock, "training must never run on the mock environment"
        print("✅ Environment created (CarRacing-v3)")
        
        # Test reset
        state, info = env.reset()
        print(f"✅ Environment reset - State shape: {state.shape}")
        
        # Test step
        action = 0  # No action
        next_state, reward, done, info = env.step(action)
        print(f"✅ Environment step - Reward: {reward:.2f}, Done: {done}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        traceback.print_exc()
        return False

def test_training_components():
    """Test training components"""
    print("\nTesting Training Components...")
    
    try:
        from training import DQNTrainer
        
        # Create minimal config
        config = {
            'skip_frames': 2,
            'stack_frames': 2,
            'reward_shaping': True,
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update': 100,
            'batch_size': 8,
            'device': 'cpu',
            'num_episodes': 2,
        }
        
        # Create trainer
        trainer = DQNTrainer(config)
        print("✅ Trainer created")
        
        # Test short training run
        print("Running 2 test episodes...")
        trainer.train(2)
        print("✅ Training test completed")
        
        trainer.env.close()
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        traceback.print_exc()
        return False

def test_analysis():
    """Test analysis components"""
    print("\nTesting Analysis Components...")
    
    try:
        from analysis import TrainingAnalyzer
        
        # Create dummy data
        rewards = np.random.randn(100).cumsum() + 100
        losses = np.exp(-np.arange(50) / 10) + np.random.randn(50) * 0.1
        episode_lengths = np.random.poisson(200, 100)
        
        # Create analyzer
        analyzer = TrainingAnalyzer("dummy_run")
        
        # Test analysis functions
        stability_results = analyzer.analyze_training_stability(rewards, losses)
        print("✅ Stability analysis")
        
        failure_modes = analyzer.diagnose_failure_modes(rewards, episode_lengths)
        print("✅ Failure mode diagnosis")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("DQN CARRACING SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("DQN Agent", test_dqn_agent),
        ("Environment", test_environment),
        ("Training", test_training_components),
        ("Analysis", test_analysis),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Your DQN setup is ready.")
        print("\nTo start training, run:")
        print("  python training.py")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())