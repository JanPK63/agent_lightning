#!/usr/bin/env python3
"""
Test distributed execution with Ray
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl_orch'))

from rl_orch.core.orchestrator import RLOrchestrator
from rl_orch.core.config_models import *

def test_distributed_execution():
    """Test orchestrator with distributed Ray execution"""
    print("🚀 Testing Distributed Execution with Ray...")
    
    # Create experiment configuration with multiple workers
    config = ExperimentConfig(
        name="test_distributed_cartpole",
        run_id="distributed_test_001",
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
            lr=0.001,
            gamma=0.99,
            batch_size=32,
            off_policy=False
        ),
        resources=ResourceConfig(
            num_workers=3,  # Use 3 workers for distributed execution
            gpu=False
        ),
        eval=EvalConfig(
            frequency=1,
            episodes=3
        ),
        ckpt=CheckpointConfig(
            frequency=1
        )
    )
    
    try:
        # Create orchestrator
        orchestrator = RLOrchestrator()
        
        # Run distributed experiment
        print(f"Starting distributed experiment: {config.name}")
        print(f"Using {config.resources.num_workers} Ray workers")
        
        run_id = orchestrator.run_experiment(config)
        
        # Check results
        state = orchestrator.get_experiment_status(run_id)
        print(f"Experiment status: {state.status}")
        print(f"Completed epochs: {state.epoch}")
        print(f"Checkpoints: {len(state.checkpoints)}")
        
        print("✅ Distributed execution test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Distributed execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_worker_fallback():
    """Test fallback to single worker execution"""
    print("🔄 Testing single worker fallback...")
    
    config = ExperimentConfig(
        name="test_single_worker",
        run_id="single_worker_test_001",
        env=EnvConfig(
            id="CartPole-v1",
            num_envs=1,
            seed=42
        ),
        policy=PolicyConfig(
            algo="ppo",
            network={"type": "mlp", "hidden_sizes": [32, 32]},
            discrete=True
        ),
        train=TrainConfig(
            epochs=1,
            steps_per_epoch=50,
            lr=0.001,
            gamma=0.99,
            batch_size=16,
            off_policy=False
        ),
        resources=ResourceConfig(
            num_workers=1,  # Single worker - should use simple scheduler
            gpu=False
        )
    )
    
    try:
        orchestrator = RLOrchestrator()
        run_id = orchestrator.run_experiment(config)
        
        state = orchestrator.get_experiment_status(run_id)
        print(f"Single worker experiment status: {state.status}")
        
        print("✅ Single worker fallback test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Single worker test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_distributed_execution()
    success2 = test_single_worker_fallback()
    
    sys.exit(0 if (success1 and success2) else 1)