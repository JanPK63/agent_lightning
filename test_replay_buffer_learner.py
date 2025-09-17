#!/usr/bin/env python3
"""
Test real ReplayBuffer and Learner implementations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl_orch'))

from rl_orch.core.replay_buffer import ReplayBuffer, EpisodeBuffer, BufferConfig, BufferManager
from rl_orch.core.learner import LearnerManager, LearnerConfig, LearnerFactory
from rl_orch.core.env_manager import Trajectory, TrajectoryBatch
import numpy as np

def create_mock_trajectory(length=10):
    """Create mock trajectory for testing"""
    states = [np.random.rand(4) for _ in range(length + 1)]  # +1 for final state
    actions = [np.random.randint(0, 2) for _ in range(length)]
    rewards = [np.random.rand() for _ in range(length)]
    dones = [False] * (length - 1) + [True]
    infos = [{}] * length
    
    return Trajectory(states, actions, rewards, dones, infos)

def test_replay_buffer():
    """Test ReplayBuffer functionality"""
    print("🧪 Testing ReplayBuffer...")
    
    try:
        # Create buffer config
        config = BufferConfig(
            size=1000,
            batch_size=32,
            min_size=10,
            prioritized=False
        )
        
        # Create buffer
        buffer = ReplayBuffer(config)
        
        # Create mock trajectories
        trajectories = [create_mock_trajectory(20) for _ in range(5)]
        traj_batch = TrajectoryBatch(trajectories)
        
        # Add trajectories to buffer
        buffer.add(traj_batch)
        print(f"Buffer size after adding: {buffer.size()}")
        
        # Test sampling
        if buffer.is_ready():
            batch = buffer.sample(16)
            print(f"Sampled batch - States: {len(batch['states'])}, Actions: {len(batch['actions'])}")
            print(f"Sample rewards: {batch['rewards'][:5]}")
        
        # Test episode buffer
        episode_config = BufferConfig(size=100, batch_size=16)
        episode_buffer = EpisodeBuffer(episode_config)
        
        episode_buffer.add(traj_batch)
        episode_batch = episode_buffer.sample()
        print(f"Episode buffer - Episodes: {episode_batch['num_episodes']}, Total steps: {len(episode_batch['states'])}")
        
        print("✅ ReplayBuffer test completed!")
        return True
        
    except Exception as e:
        print(f"❌ ReplayBuffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learners():
    """Test different learner implementations"""
    print("🧠 Testing Learners...")
    
    try:
        # Test PPO learner
        ppo_config = LearnerConfig(
            algorithm="ppo",
            learning_rate=3e-4,
            gamma=0.99,
            clip_epsilon=0.2
        )
        
        ppo_learner = LearnerFactory.create_learner("ppo", ppo_config)
        
        # Create mock batch
        batch = {
            "states": np.random.rand(32, 4),
            "actions": np.random.randint(0, 2, 32),
            "rewards": np.random.rand(32),
            "dones": np.random.choice([True, False], 32)
        }
        
        # Test PPO update
        loss, stats = ppo_learner.update(batch)
        print(f"PPO - Loss: {loss:.4f}, Stats: {list(stats.keys())}")
        
        # Test DQN learner
        dqn_config = LearnerConfig(algorithm="dqn", learning_rate=1e-3)
        dqn_learner = LearnerFactory.create_learner("dqn", dqn_config)
        
        loss, stats = dqn_learner.update(batch)
        print(f"DQN - Loss: {loss:.4f}, Epsilon: {stats.get('epsilon', 'N/A')}")
        
        # Test SAC learner
        sac_config = LearnerConfig(algorithm="sac", learning_rate=3e-4)
        sac_learner = LearnerFactory.create_learner("sac", sac_config)
        
        loss, stats = sac_learner.update(batch)
        print(f"SAC - Loss: {loss:.4f}, Temperature: {stats.get('temperature', 'N/A')}")
        
        # Test learner manager
        manager = LearnerManager()
        
        ppo_learner2 = manager.create_learner("ppo_agent", "ppo", ppo_config)
        dqn_learner2 = manager.create_learner("dqn_agent", "dqn", dqn_config)
        
        all_stats = manager.get_all_stats()
        print(f"Manager stats: {list(all_stats.keys())}")
        
        print("✅ Learners test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Learners test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_buffer_manager():
    """Test BufferManager functionality"""
    print("📦 Testing BufferManager...")
    
    try:
        manager = BufferManager()
        
        # Create different buffer types
        replay_config = BufferConfig(size=500, batch_size=16, min_size=5)
        episode_config = BufferConfig(size=100, batch_size=8)
        
        replay_buffer = manager.create_buffer("replay", replay_config, "replay")
        episode_buffer = manager.create_buffer("episode", episode_config, "episode")
        
        # Test with mock data
        trajectories = [create_mock_trajectory(15) for _ in range(3)]
        traj_batch = TrajectoryBatch(trajectories)
        
        # Add to both buffers
        replay_buffer.add(traj_batch)
        episode_buffer.add(traj_batch)
        
        print(f"Replay buffer size: {replay_buffer.size()}")
        print(f"Episode buffer size: {episode_buffer.size()}")
        
        # Test retrieval
        retrieved_replay = manager.get_buffer("replay")
        retrieved_episode = manager.get_buffer("episode")
        
        print(f"Retrieved buffers: {retrieved_replay is not None}, {retrieved_episode is not None}")
        
        # Clear all
        manager.clear_all()
        print("Cleared all buffers")
        
        print("✅ BufferManager test completed!")
        return True
        
    except Exception as e:
        print(f"❌ BufferManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_replay_buffer()
    success2 = test_learners()
    success3 = test_buffer_manager()
    
    sys.exit(0 if (success1 and success2 and success3) else 1)