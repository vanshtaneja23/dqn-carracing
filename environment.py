import gymnasium as gym
import numpy as np
import cv2
from collections import deque

class CarRacingWrapper:
    """Wrapper for CarRacing environment with preprocessing and reward shaping"""
    
    def __init__(self, skip_frames=4, stack_frames=4, reward_shaping=True,
                 early_termination=None, use_mock=False):
        if use_mock:
            # Unit tests only: random frames and random rewards, NOT a real environment.
            self.env = MockCarRacingEnv()
        else:
            try:
                self.env = gym.make('CarRacing-v3', render_mode='rgb_array')
            except Exception as e:
                # Fail loudly. A silent fallback to the mock once trained a whole run on noise.
                raise RuntimeError(
                    "Could not create CarRacing-v3. Install Box2D: pip install 'gymnasium[box2d]'"
                ) from e
        self.is_mock = use_mock
        # End training episodes early when the car is stuck off-track (on by default with shaping)
        self.early_termination = reward_shaping if early_termination is None else early_termination
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.reward_shaping = reward_shaping
        
        # Frame stacking
        self.frame_stack = deque(maxlen=stack_frames)
        
        # Action space mapping
        self.action_space = self._create_action_space()
        
        # Reward shaping parameters
        self.prev_reward = 0
        self.negative_reward_counter = 0
        self.grass_penalty = 0
        
    def _create_action_space(self):
        """Create discrete action space from continuous actions"""
        return [
            [0, 0, 0],      # No action
            [-1, 0, 0],     # Turn left
            [1, 0, 0],      # Turn right
            [0, 1, 0],      # Accelerate
            [0, 0, 0.8],    # Brake
        ]
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for neural network"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Crop relevant area (remove score area)
        cropped = gray[0:84, 6:90]
        
        # Resize to 84x84
        resized = cv2.resize(cropped, (84, 84))
        
        return resized.astype(np.float32) / 255.0
    
    def _shape_reward(self, reward, done, info):
        """Apply reward shaping to improve training"""
        if not self.reward_shaping:
            return reward
        
        shaped_reward = reward
        
        # Penalize going off track (negative rewards)
        if reward < 0:
            self.negative_reward_counter += 1
            if self.negative_reward_counter > 20:  # Consecutive negative rewards
                shaped_reward = reward * 2  # Increase penalty
        else:
            self.negative_reward_counter = 0
        
        # Reward for maintaining speed
        if reward > 0:
            shaped_reward = reward * 1.5  # Boost positive rewards
        
        # Early termination penalty for poor performance
        if done and reward < 500:  # Didn't complete much of the track
            shaped_reward -= 100
        
        return shaped_reward
    
    def reset(self):
        """Reset environment and return initial state"""
        obs, info = self.env.reset()
        
        # Reset counters
        self.negative_reward_counter = 0
        self.grass_penalty = 0
        
        # Preprocess and stack initial frame
        processed_frame = self._preprocess_frame(obs)
        
        # Fill frame stack with initial frame
        for _ in range(self.stack_frames):
            self.frame_stack.append(processed_frame)
        
        return np.array(self.frame_stack), info
    
    def step(self, action_idx):
        """Execute action and return next state"""
        action = np.array(self.action_space[action_idx], dtype=np.float32)
        
        total_reward = 0
        done = False
        info = {}
        
        # Frame skipping
        for _ in range(self.skip_frames):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if done:
                break
        
        # Preprocess frame
        processed_frame = self._preprocess_frame(obs)
        self.frame_stack.append(processed_frame)
        
        # Apply reward shaping
        shaped_reward = self._shape_reward(total_reward, done, info)

        # Stop wasting frames once the car has been off-track for a long stretch
        if self.early_termination and self.negative_reward_counter > 50:
            done = True

        info = dict(info)
        info['raw_reward'] = total_reward  # standard CarRacing reward, before shaping
        return np.array(self.frame_stack), shaped_reward, done, info
    
    def render(self):
        """Render environment"""
        return self.env.render()
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    @property
    def action_size(self):
        """Get number of discrete actions"""
        return len(self.action_space)

class MockCarRacingEnv:
    """Mock environment for testing when CarRacing is not available"""
    
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.step_count = 0
        
    def reset(self, **kwargs):
        self.step_count = 0
        obs = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        obs = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        reward = np.random.randn() * 10  # Random reward for testing
        done = self.step_count > 1000  # End after 1000 steps
        return obs, reward, done, False, {}
    
    def render(self):
        return np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    
    def close(self):
        pass