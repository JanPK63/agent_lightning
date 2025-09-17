#!/usr/bin/env python3
"""
Test comprehensive environment manager
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl_orch'))

from rl_orch.core.comprehensive_env_manager import (
    ComprehensiveEnvManager, EnvSpec, RolloutConfig
)

def test_comprehensive_env_manager():
    """Test comprehensive environment manager functionality"""
    print("🧪 Testing Comprehensive Environment Manager...")
    
    try:
        # Create manager
        manager = ComprehensiveEnvManager(num_workers=2)
        
        # Register environment
        env_spec = EnvSpec(
            id="CartPole-v1",
            num_envs=2,
            seed=42,
            max_episode_steps=200
        )
        manager.register_environment("cartpole", env_spec)
        
        # Create rollout configuration
        config = RolloutConfig(
            max_steps_per_episode=200,
            max_episodes=5,
            deterministic=False,
            timeout=10.0
        )
        
        # Test single-threaded rollout collection
        print("Testing single-threaded rollout collection...")
        batch_single = manager.collect_rollouts(
            env_name="cartpole",
            policy_name="random",
            config=config,
            num_episodes=3,
            parallel=False
        )
        
        print(f"Single-threaded: {len(batch_single.trajectories)} episodes, "
              f"{batch_single.total_steps()} total steps")
        
        # Test parallel rollout collection
        print("Testing parallel rollout collection...")
        batch_parallel = manager.collect_rollouts(
            env_name="cartpole",
            policy_name="random",
            config=config,
            num_episodes=4,
            parallel=True
        )
        
        print(f"Parallel: {len(batch_parallel.trajectories)} episodes, "
              f"{batch_parallel.total_steps()} total steps")
        
        # Test environment listing
        envs = manager.env_manager.list_environments()
        print(f"Registered environments: {envs}")
        
        # Test trajectory analysis
        if batch_single.trajectories:
            first_traj = batch_single.trajectories[0]
            print(f"First trajectory: {len(first_traj)} steps, "
                  f"reward: {sum(first_traj.rewards):.2f}")
        
        # Cleanup
        manager.shutdown()
        
        print("✅ Comprehensive environment manager test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive environment manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_env_manager_components():
    """Test individual components"""
    print("🔧 Testing individual EnvManager components...")
    
    try:
        from rl_orch.core.comprehensive_env_manager import EnvManager, RolloutWorker
        
        # Test EnvManager
        env_manager = EnvManager()
        
        env_spec = EnvSpec(id="CartPole-v1", num_envs=1, seed=123)
        env_manager.register_env("test_env", env_spec)
        
        pool = env_manager.create_env_pool("test_env")
        print(f"Created pool with {pool.num_envs} environments")
        
        # Test RolloutWorker
        worker = RolloutWorker(env_manager, worker_id=1)
        
        # Create random policy
        random_policy = env_manager.env_pools["test_env"].envs[0].action_space.sample
        
        class SimplePolicy:
            def __init__(self, action_space):
                self.action_space = action_space
            
            def get_action(self, obs, deterministic=False):
                return self.action_space.sample()
            
            def reset(self):
                pass
        
        policy = SimplePolicy(env_manager.env_pools["test_env"].envs[0].action_space)
        
        config = RolloutConfig(max_steps_per_episode=50, timeout=5.0)
        
        # Collect single episode
        trajectory = worker.collect_episode("test_env", policy, config)
        print(f"Single episode: {len(trajectory)} steps, reward: {sum(trajectory.rewards)}")
        
        # Collect multiple episodes
        batch = worker.collect_rollouts("test_env", policy, config, num_episodes=2)
        print(f"Multiple episodes: {len(batch.trajectories)} episodes")
        
        # Cleanup
        env_manager.close_all()
        
        print("✅ Individual components test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Individual components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_comprehensive_env_manager()
    success2 = test_env_manager_components()
    
    sys.exit(0 if (success1 and success2) else 1)