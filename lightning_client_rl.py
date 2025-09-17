"""
Lightning Client for Agent Execution and Trajectory Collection
Handles agent execution, collects data trajectories, and sends them to the server
"""

from agentlightning import LightningClient
import json
import asyncio
from pathlib import Path
import time
from typing import Dict, List, Any

class RLLightningClient:
    """Enhanced Lightning Client for RL trajectory collection"""
    
    def __init__(self, server_url="http://localhost:8000"):
        """Initialize the Lightning Client"""
        
        # Load server configuration if available
        self.config = self.load_server_config()
        
        # Configure client for agent execution
        self.client = LightningClient(
            server_url=server_url,
            agent_function=self.multi_agent_mdp_function,  # Will be defined later
            num_workers=4,  # For parallel execution
            timeout=30,  # Prevent hangs
            batch_collection=True,  # Collect trajectories in batches
            trajectory_buffer_size=1000  # Buffer size for trajectories
        )
        
        # Trajectory storage
        self.trajectories = []
        self.episode_count = 0
        self.total_rewards = []
        
        print(f"✅ Lightning Client initialized")
        print(f"   Server: {server_url}")
        print(f"   Workers: 4")
        print(f"   Trajectory Buffer: 1000")
    
    def load_server_config(self):
        """Load server configuration"""
        config_path = Path("server_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print("⚠️  Server config not found, using defaults")
            return {}
    
    def multi_agent_mdp_function(self, state: Dict) -> Dict:
        """
        Placeholder for MDP agent function
        This will be replaced with actual MDP implementation
        """
        # For now, return a simple trajectory
        action = {
            "type": "response",
            "content": "Placeholder action",
            "agent": "default"
        }
        
        reward = 0.5  # Placeholder reward
        
        trajectory = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": state,  # Simplified for now
            "done": False,
            "timestamp": time.time()
        }
        
        return trajectory
    
    async def collect_trajectories(self, num_episodes=100):
        """Collect trajectories from agent execution"""
        print(f"\n📊 Starting trajectory collection for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Create initial state for episode
            initial_state = {
                "episode": episode,
                "task": f"Task_{episode}",
                "context": {},
                "hierarchy_level": "high" if episode % 2 == 0 else "low"
            }
            
            # Run agent and collect trajectory
            trajectory = await self.run_episode_async(initial_state)
            self.trajectories.append(trajectory)
            
            # Calculate episode reward
            episode_reward = sum(t.get("reward", 0) for t in trajectory)
            self.total_rewards.append(episode_reward)
            
            # Send batch to server every 10 episodes
            if (episode + 1) % 10 == 0:
                await self.send_trajectories_to_server()
                print(f"   Episode {episode + 1}/{num_episodes} - Avg Reward: {sum(self.total_rewards[-10:])/10:.3f}")
        
        # Send remaining trajectories
        if self.trajectories:
            await self.send_trajectories_to_server()
        
        print(f"\n✅ Trajectory collection complete!")
        print(f"   Total Episodes: {num_episodes}")
        print(f"   Average Reward: {sum(self.total_rewards)/len(self.total_rewards):.3f}")
    
    async def run_episode_async(self, initial_state: Dict) -> List[Dict]:
        """Run a single episode and collect trajectory"""
        trajectory = []
        state = initial_state
        done = False
        max_steps = 50
        step = 0
        
        while not done and step < max_steps:
            # Get action from agent (placeholder for now)
            action_result = self.multi_agent_mdp_function(state)
            trajectory.append(action_result)
            
            # Update state (simplified)
            state = action_result.get("next_state", state)
            done = action_result.get("done", False)
            step += 1
            
            # Small delay to simulate processing
            await asyncio.sleep(0.01)
        
        return trajectory
    
    async def send_trajectories_to_server(self):
        """Send collected trajectories to the server"""
        if not self.trajectories:
            return
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                # Prepare data for server
                data = {
                    "trajectories": self.trajectories,
                    "episode_count": self.episode_count,
                    "timestamp": time.time()
                }
                
                # Send to server
                response = await client.post(
                    f"{self.client.server_url}/rollouts",
                    json=data,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    self.episode_count += len(self.trajectories)
                    self.trajectories = []  # Clear buffer
                    print(f"   ✓ Sent {len(self.trajectories)} trajectories to server")
                else:
                    print(f"   ⚠️ Server response: {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ Error sending trajectories: {e}")
    
    def run(self):
        """Main execution loop"""
        print("\n🚀 Starting Lightning Client...")
        
        try:
            # Run trajectory collection
            asyncio.run(self.collect_trajectories(num_episodes=100))
            
            # Run standard client collection
            print("\n📡 Running standard client data collection...")
            self.client.run()
            
        except KeyboardInterrupt:
            print("\n⚫ Client stopped by user")
        except Exception as e:
            print(f"❌ Client error: {e}")
            raise

def main():
    """Main entry point for the client"""
    print("""
    ⚡ Lightning Client for RL Training
    ===================================
    This client will:
    1. Connect to the Lightning Server
    2. Execute agents and collect trajectories
    3. Send data back to server for RL training
    
    Make sure the server is running on http://localhost:8000
    """)
    
    # Check if server is accessible
    import requests
    try:
        response = requests.get("http://localhost:8000", timeout=2)
        print("✅ Server is accessible")
    except:
        print("⚠️  Warning: Cannot reach server at http://localhost:8000")
        print("   Make sure to start the server first with: python lightning_server_rl.py")
        print("   Continue anyway? (y/n): ", end="")
        if input().lower() != 'y':
            return
    
    # Initialize and run client
    client = RLLightningClient()
    client.run()

if __name__ == "__main__":
    main()