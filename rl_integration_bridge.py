#!/usr/bin/env python3
"""
Integration bridge between Agent Lightning and RL Orchestrator
Connects the main system with the new RL training capabilities
"""

import sys
import os
from pathlib import Path

# Add rl_orch to path
sys.path.append(str(Path(__file__).parent / 'rl_orch'))

from agentlightning.trainer import Trainer as AgentLightningTrainer
from agentlightning.litagent import LitAgent
from rl_orch.core.orchestrator import RLOrchestrator
from rl_orch.core.config_models import *

class IntegratedRLTrainer:
    """Integrated trainer combining Agent Lightning and RL Orchestrator"""
    
    def __init__(self):
        self.agent_trainer = None
        self.rl_orchestrator = RLOrchestrator()
    
    def train_with_rl(self, 
                     agent: LitAgent,
                     rl_config: ExperimentConfig,
                     backend: str = None):
        """Train agent using RL orchestrator"""
        
        print(f"🚀 Starting integrated RL training for {rl_config.name}")
        
        # Run RL experiment
        run_id = self.rl_orchestrator.run_experiment(rl_config)
        
        # Get results
        state = self.rl_orchestrator.get_experiment_status(run_id)
        
        print(f"✅ RL training completed: {state.status}")
        print(f"   Epochs: {state.epoch}")
        print(f"   Checkpoints: {len(state.checkpoints)}")
        
        return run_id, state
    
    def train_with_agent_lightning(self,
                                  agent: LitAgent,
                                  backend: str,
                                  n_workers: int = 1):
        """Train using original Agent Lightning trainer"""
        
        print(f"🔥 Starting Agent Lightning training with {n_workers} workers")
        
        self.agent_trainer = AgentLightningTrainer(n_workers=n_workers)
        self.agent_trainer.fit(agent, backend)
        
        print("✅ Agent Lightning training completed")
    
    def hybrid_training(self,
                       agent: LitAgent,
                       rl_config: ExperimentConfig,
                       backend: str,
                       use_rl_first: bool = True):
        """Hybrid training using both systems"""
        
        print("🌟 Starting hybrid training approach")
        
        if use_rl_first:
            # Phase 1: RL training
            print("Phase 1: RL Orchestrator training")
            rl_run_id, rl_state = self.train_with_rl(agent, rl_config, backend)
            
            # Phase 2: Agent Lightning fine-tuning
            print("Phase 2: Agent Lightning fine-tuning")
            self.train_with_agent_lightning(agent, backend)
        else:
            # Phase 1: Agent Lightning training
            print("Phase 1: Agent Lightning training")
            self.train_with_agent_lightning(agent, backend)
            
            # Phase 2: RL optimization
            print("Phase 2: RL Orchestrator optimization")
            rl_run_id, rl_state = self.train_with_rl(agent, rl_config, backend)
        
        print("✅ Hybrid training completed")
        return rl_run_id, rl_state

def create_rl_config_from_agent(agent: LitAgent, 
                               algorithm: str = "ppo",
                               epochs: int = 5) -> ExperimentConfig:
    """Create RL config from Agent Lightning agent"""
    
    return ExperimentConfig(
        name=f"agent_rl_{algorithm}",
        env=EnvConfig(
            id="CartPole-v1",  # Default env
            num_envs=2,
            seed=42
        ),
        policy=PolicyConfig(
            algo=algorithm,
            network={"type": "mlp", "hidden_sizes": [64, 64]},
            discrete=True
        ),
        train=TrainConfig(
            epochs=epochs,
            steps_per_epoch=200,
            lr=3e-4,
            gamma=0.99,
            batch_size=32,
            off_policy=(algorithm != "ppo")
        ),
        buffer=BufferConfig(
            size=1000,
            batch_size=32
        ),
        eval=EvalConfig(
            frequency=2,
            episodes=3
        )
    )

def test_integration():
    """Test the integration bridge"""
    print("🧪 Testing RL Integration Bridge...")
    
    # Mock agent (in real use, would be actual LitAgent)
    class MockAgent(LitAgent):
        def training_rollout(self, task):
            return {"result": "mock_result"}
    
    agent = MockAgent()
    
    # Test RL-only training
    trainer = IntegratedRLTrainer()
    rl_config = create_rl_config_from_agent(agent, "ppo", 2)
    
    run_id, state = trainer.train_with_rl(agent, rl_config)
    
    print(f"✅ Integration test completed: {state.status}")
    return state.status == "completed"

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)