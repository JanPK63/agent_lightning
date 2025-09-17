#!/usr/bin/env python3
"""
Test script for Gymnasium integration in RL Orchestrator
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl_orch'))

from rl_orch.core.env_manager import EnvironmentManager, EnvironmentPool
from rl_orch.core.config_models import EnvConfig

def test_gymnasium_integration():
    """Test real Gymnasium environment integration"""
    print("🧪 Testing Gymnasium Integration...")
    
    # Create environment manager
    env_manager = EnvironmentManager()
    
    # Test environment configuration
    env_cfg = EnvConfig(
        id="CartPole-v1",
        num_envs=2,
        seed=42
    )
    
    try:
        # Create environment pool
        print(f"Creating environment pool: {env_cfg.id} with {env_cfg.num_envs} envs")
        env_pool = env_manager.make_envs(env_cfg)
        
        # Test environment reset
        print("Testing environment reset...")
        state, info = env_pool.reset(0)
        print(f"Initial state shape: {len(state) if hasattr(state, '__len__') else 'scalar'}")
        
        # Test environment step
        print("Testing environment step...")
        action = env_pool.envs[0].action_space.sample()
        next_state, reward, terminated, truncated, info = env_pool.step(0, action)
        print(f"Action: {action}, Reward: {reward}, Done: {terminated or truncated}")
        
        # Test rollout collection
        print("Testing rollout collection...")
        mock_policy = {"type": "random"}  # Simple mock policy
        traj_batch = env_manager.collect_rollouts(env_pool, mock_policy, steps_per_epoch=100)
        
        print(f"Collected {len(traj_batch.trajectories)} trajectories")
        print(f"Total steps: {traj_batch.total_steps()}")
        
        if traj_batch.trajectories:
            first_traj = traj_batch.trajectories[0]
            print(f"First trajectory length: {len(first_traj)}")
            print(f"First trajectory reward: {sum(first_traj.rewards)}")
        
        # Test conversion to dict format
        batch_dict = traj_batch.to_dict()
        print(f"Dict format - States: {len(batch_dict['states'])}, Actions: {len(batch_dict['actions'])}")
        
        # Cleanup
        env_manager.close_all()
        print("✅ Gymnasium integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Gymnasium integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gymnasium_integration()
    sys.exit(0 if success else 1)