"""
Lightning Server with Hierarchical RL Configuration
Enhanced server setup for multi-agent reinforcement learning
"""

from agentlightning import AgentLightningServer
import os
import json
from pathlib import Path

def setup_hierarchical_rl_server():
    """Configure the Lightning Server for hierarchical RL training"""
    
    # Create necessary directories
    Path("./checkpoints").mkdir(exist_ok=True)
    Path("./data").mkdir(exist_ok=True)
    
    # Server configuration for hierarchical RL (metadata only)
    server_config = {
        "model_path": "gpt-4o",  # Primary LLM model
        "rl_algorithm": "LightningRL",  # Hierarchical RL algorithm
        "dataset_path": "data/train.jsonl",  # Training dataset
        "checkpoint_dir": "./checkpoints",  # For model recovery
        "batch_size": 32,
        "learning_rate": 1e-5,
        "num_epochs": 10,
        
        # Hierarchical RL specific settings
        "hierarchy_levels": ["high", "low"],  # High-level planning, low-level execution
        "discount_factor": 0.99,
        "entropy_coefficient": 0.01,
        "value_coefficient": 0.5,
        "max_grad_norm": 0.5,
        
        # Memory and context settings
        "context_window": 4096,
        "memory_buffer_size": 10000,
        "replay_batch_size": 128,
        
        # Multi-agent settings
        "num_agents": 4,
        "agent_types": ["researcher", "writer", "reviewer", "optimizer"],
        "shared_reward": True,
        "communication_enabled": True
    }
    
    # Initialize server with correct parameters
    print("⚡ Initializing Lightning Server with Hierarchical RL...")
    server = AgentLightningServer(host="0.0.0.0", port=8010, task_timeout_seconds=300.0)
    
    # Add custom reward shaping for hierarchical tasks
    server.reward_shaping = {
        "high_level_bonus": 0.3,  # Bonus for completing high-level goals
        "low_level_penalty": -0.1,  # Penalty for inefficient low-level actions
        "cooperation_bonus": 0.2,  # Bonus for agent cooperation
        "efficiency_multiplier": 1.5  # Multiplier for efficient task completion
    }
    
    return server, server_config

async def start_server():
    """Start the Lightning Server"""
    server, config = setup_hierarchical_rl_server()
    
    print(f"""
    ✅ Lightning Server Configuration:
    - Model: {config['model_path']}
    - RL Algorithm: {config['rl_algorithm']}
    - Batch Size: {config['batch_size']}
    - Learning Rate: {config['learning_rate']}
    - Hierarchy Levels: {config['hierarchy_levels']}
    - Number of Agents: {config['num_agents']}
    - Context Window: {config['context_window']}
    """)
    
    # Save configuration for client access
    with open("server_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("🚀 Starting Lightning Server on http://0.0.0.0:8010")
    print("   Use Ctrl+C to stop the server")
    
    try:
        # Start server on 0.0.0.0:8010
        await server.run_forever()
    except KeyboardInterrupt:
        print("\n⚫ Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(start_server())