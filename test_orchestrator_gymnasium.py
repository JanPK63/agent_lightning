#!/usr/bin/env python3
"""
Test RL Orchestrator with real Gymnasium environments
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl_orch'))

from rl_orch.core.orchestrator import RLOrchestrator
from rl_orch.core.config_models import *

def test_orchestrator_with_gymnasium():
    """Test orchestrator with real Gymnasium environments"""
    print("🚀 Testing RL Orchestrator with Gymnasium...")
    
    # Create experiment configuration
    config = ExperimentConfig(
        name="test_cartpole_gymnasium",
        run_id="test_run_001",
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
            epochs=3,
            steps_per_epoch=50,
            lr=0.001,
            gamma=0.99,
            batch_size=32,
            off_policy=False
        ),
        eval=EvalConfig(
            frequency=1,
            episodes=5
        ),
        gates=GatesConfig(
            min_reward=10.0
        ),
        ckpt=CheckpointConfig(
            frequency=2
        )
    )
    
    try:
        # Create orchestrator
        orchestrator = RLOrchestrator()
        
        # Run experiment
        print(f"Starting experiment: {config.name}")
        run_id = orchestrator.run_experiment(config)
        
        # Check results
        state = orchestrator.get_experiment_status(run_id)
        print(f"Experiment status: {state.status}")
        print(f"Completed epochs: {state.epoch}")
        print(f"Checkpoints: {len(state.checkpoints)}")
        
        print("✅ RL Orchestrator with Gymnasium test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RL Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_orchestrator_with_gymnasium()
    sys.exit(0 if success else 1)