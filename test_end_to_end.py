#!/usr/bin/env python3
"""
End-to-end test of RL Orchestrator with all real components
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl_orch'))

from rl_orch.core.orchestrator import RLOrchestrator
from rl_orch.core.config_models import *
# CLI test will be simplified since RLController class doesn't exist
import yaml

def test_end_to_end_ppo():
    """Complete end-to-end PPO experiment"""
    print("🚀 End-to-End PPO Test...")
    
    config = ExperimentConfig(
        name="e2e_ppo_cartpole",
        run_id="e2e_ppo_001",
        env=EnvConfig(
            id="CartPole-v1",
            num_envs=4,
            seed=42
        ),
        policy=PolicyConfig(
            algo="ppo",
            network={"type": "mlp", "hidden_sizes": [64, 64]},
            discrete=True
        ),
        train=TrainConfig(
            epochs=5,
            steps_per_epoch=200,
            lr=3e-4,
            gamma=0.99,
            batch_size=64,
            off_policy=False
        ),
        buffer=BufferConfig(
            size=2000,
            batch_size=64
        ),
        eval=EvalConfig(
            frequency=2,
            episodes=5
        ),
        gates=GatesConfig(
            min_reward=50.0
        ),
        resources=ResourceConfig(
            num_workers=2,
            gpu=False
        ),
        ckpt=CheckpointConfig(
            frequency=2
        )
    )
    
    orchestrator = RLOrchestrator()
    run_id = orchestrator.run_experiment(config)
    
    state = orchestrator.get_experiment_status(run_id)
    print(f"✅ PPO E2E: {state.status}, epochs: {state.epoch}, checkpoints: {len(state.checkpoints)}")
    return state.status == "completed"

def test_end_to_end_dqn_distributed():
    """End-to-end DQN with distributed execution"""
    print("🔥 End-to-End DQN Distributed Test...")
    
    config = ExperimentConfig(
        name="e2e_dqn_distributed",
        run_id="e2e_dqn_001",
        env=EnvConfig(
            id="CartPole-v1",
            num_envs=3,
            seed=123
        ),
        policy=PolicyConfig(
            algo="dqn",
            network={"type": "mlp", "hidden_sizes": [128, 64]},
            discrete=True
        ),
        train=TrainConfig(
            epochs=4,
            steps_per_epoch=150,
            lr=1e-3,
            gamma=0.95,
            batch_size=32,
            off_policy=True
        ),
        buffer=BufferConfig(
            size=1500,
            batch_size=32
        ),
        eval=EvalConfig(
            frequency=1,
            episodes=3
        ),
        resources=ResourceConfig(
            num_workers=3,  # Distributed
            gpu=False
        )
    )
    
    orchestrator = RLOrchestrator()
    run_id = orchestrator.run_experiment(config)
    
    state = orchestrator.get_experiment_status(run_id)
    print(f"✅ DQN Distributed E2E: {state.status}, epochs: {state.epoch}")
    return state.status == "completed"

def test_cli_integration():
    """Test CLI integration via subprocess"""
    print("⚡ Testing CLI Integration...")
    
    config_data = {
        "name": "cli_test_experiment",
        "env": {"id": "CartPole-v1", "num_envs": 1, "seed": 456},
        "policy": {"algo": "ppo", "network": {"type": "mlp", "hidden_sizes": [32, 32]}, "discrete": True},
        "train": {"epochs": 1, "steps_per_epoch": 50, "lr": 0.001, "gamma": 0.99, "batch_size": 16, "off_policy": False}
    }
    
    with open("test_config.yaml", "w") as f:
        yaml.dump(config_data, f)
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "rl_orch/cli/rlctl.py", "launch", "-f", "test_config.yaml"], 
                              capture_output=True, text=True, timeout=30)
        
        os.remove("test_config.yaml")
        success = result.returncode == 0 and "launched" in result.stdout.lower()
        print(f"✅ CLI Integration: {success}")
        return success
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")
        return False

def test_comprehensive_pipeline():
    """Test complete pipeline with all features"""
    print("🌟 Comprehensive Pipeline Test...")
    
    config = ExperimentConfig(
        name="comprehensive_pipeline",
        run_id="comprehensive_001",
        env=EnvConfig(
            id="CartPole-v1",
            num_envs=2,
            seed=789
        ),
        policy=PolicyConfig(
            algo="sac",
            network={"type": "mlp", "hidden_sizes": [64, 32]},
            discrete=True
        ),
        train=TrainConfig(
            epochs=3,
            steps_per_epoch=120,
            lr=3e-4,
            gamma=0.99,
            batch_size=24,
            off_policy=True
        ),
        buffer=BufferConfig(
            size=1000,
            batch_size=24
        ),
        eval=EvalConfig(
            frequency=1,
            episodes=4,
            render=False
        ),
        gates=GatesConfig(
            min_reward=30.0,
            max_episodes_without_improvement=10
        ),
        resources=ResourceConfig(
            num_workers=2,
            gpu=False
        ),
        ckpt=CheckpointConfig(
            frequency=1,
            keep_last=3
        )
    )
    
    orchestrator = RLOrchestrator()
    
    # Test experiment lifecycle
    run_id = orchestrator.run_experiment(config)
    state = orchestrator.get_experiment_status(run_id)
    
    # Verify all components worked
    components_working = {
        "experiment_completed": state.status == "completed",
        "epochs_run": state.epoch >= 0,
        "checkpoints_created": len(state.checkpoints) > 0,
        "config_preserved": state.config.name == config.name
    }
    
    print(f"✅ Comprehensive Pipeline: {all(components_working.values())}")
    print(f"   Components: {components_working}")
    
    return all(components_working.values())

def main():
    """Run all end-to-end tests"""
    print("🎯 Starting End-to-End RL Orchestrator Tests...")
    print("=" * 60)
    
    tests = [
        ("PPO End-to-End", test_end_to_end_ppo),
        ("DQN Distributed", test_end_to_end_dqn_distributed),
        ("CLI Integration", test_cli_integration),
        ("Comprehensive Pipeline", test_comprehensive_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            results[test_name] = test_func()
            status = "✅ PASS" if results[test_name] else "❌ FAIL"
            print(f"   Result: {status}")
        except Exception as e:
            print(f"   Result: ❌ ERROR - {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS:")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n🎯 Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 ALL END-TO-END TESTS PASSED!")
        print("🚀 RL Orchestrator is fully functional!")
    else:
        print("⚠️  Some tests failed - check logs above")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)