#!/usr/bin/env python3
"""
Test RL Orchestrator with real ReplayBuffer and Learner components
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl_orch'))

from rl_orch.core.orchestrator import RLOrchestrator
from rl_orch.core.config_models import *

def test_orchestrator_with_real_components():
    """Test orchestrator with real ReplayBuffer and Learner"""
    print("🚀 Testing RL Orchestrator with Real Components...")
    
    # Test PPO (on-policy)
    ppo_config = ExperimentConfig(
        name="test_ppo_real_components",
        run_id="ppo_real_001",
        env=EnvConfig(
            id="CartPole-v1",
            num_envs=2,
            seed=42
        ),
        policy=PolicyConfig(
            algo="ppo",
            network={"type": "mlp", "hidden_sizes": [64, 64]},
            discrete=True
        ),
        train=TrainConfig(
            epochs=2,
            steps_per_epoch=100,
            lr=3e-4,
            gamma=0.99,
            batch_size=32,
            off_policy=False  # On-policy
        ),
        buffer=BufferConfig(
            size=1000,
            batch_size=32
        ),
        eval=EvalConfig(
            frequency=1,
            episodes=3
        )
    )
    
    try:
        orchestrator = RLOrchestrator()
        
        print("Testing PPO (on-policy) with real components...")
        run_id = orchestrator.run_experiment(ppo_config)
        
        state = orchestrator.get_experiment_status(run_id)
        print(f"PPO experiment status: {state.status}")
        print(f"PPO completed epochs: {state.epoch}")
        
        print("✅ PPO test completed!")
        
    except Exception as e:
        print(f"❌ PPO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test DQN (off-policy)
    dqn_config = ExperimentConfig(
        name="test_dqn_real_components",
        run_id="dqn_real_001",
        env=EnvConfig(
            id="CartPole-v1",
            num_envs=1,
            seed=123
        ),
        policy=PolicyConfig(
            algo="dqn",
            network={"type": "mlp", "hidden_sizes": [32, 32]},
            discrete=True
        ),
        train=TrainConfig(
            epochs=3,
            steps_per_epoch=50,
            lr=1e-3,
            gamma=0.99,
            batch_size=16,
            off_policy=True  # Off-policy
        ),
        buffer=BufferConfig(
            size=500,
            batch_size=16
        ),
        eval=EvalConfig(
            frequency=2,
            episodes=2
        )
    )
    
    try:
        print("Testing DQN (off-policy) with real components...")
        run_id = orchestrator.run_experiment(dqn_config)
        
        state = orchestrator.get_experiment_status(run_id)
        print(f"DQN experiment status: {state.status}")
        print(f"DQN completed epochs: {state.epoch}")
        
        print("✅ DQN test completed!")
        
    except Exception as e:
        print(f"❌ DQN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test SAC (off-policy)
    sac_config = ExperimentConfig(
        name="test_sac_real_components",
        run_id="sac_real_001",
        env=EnvConfig(
            id="CartPole-v1",
            num_envs=1,
            seed=456
        ),
        policy=PolicyConfig(
            algo="sac",
            network={"type": "mlp", "hidden_sizes": [32, 32]},
            discrete=True
        ),
        train=TrainConfig(
            epochs=2,
            steps_per_epoch=40,
            lr=3e-4,
            gamma=0.99,
            batch_size=16,
            off_policy=True
        ),
        buffer=BufferConfig(
            size=300,
            batch_size=16
        )
    )
    
    try:
        print("Testing SAC (off-policy) with real components...")
        run_id = orchestrator.run_experiment(sac_config)
        
        state = orchestrator.get_experiment_status(run_id)
        print(f"SAC experiment status: {state.status}")
        print(f"SAC completed epochs: {state.epoch}")
        
        print("✅ SAC test completed!")
        
        # Test experiment listing
        all_experiments = orchestrator.list_experiments()
        print(f"Total experiments run: {len(all_experiments)}")
        
        print("✅ All real components tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ SAC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_orchestrator_with_real_components()
    sys.exit(0 if success else 1)