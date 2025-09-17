"""
Real ReplayBuffer implementation for RL training
Supports both on-policy and off-policy algorithms
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import threading
import logging

from .env_manager import TrajectoryBatch, Trajectory

logger = logging.getLogger(__name__)

@dataclass
class BufferConfig:
    """Replay buffer configuration"""
    size: int = 100000
    batch_size: int = 32
    min_size: int = 1000
    prioritized: bool = False
    alpha: float = 0.6  # Prioritization exponent
    beta: float = 0.4   # Importance sampling exponent

class Experience:
    """Single experience tuple"""
    
    def __init__(self, state, action, reward, next_state, done, info=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info or {}
        self.priority = 1.0  # For prioritized replay

class ReplayBuffer:
    """Standard replay buffer for off-policy algorithms"""
    
    def __init__(self, config: BufferConfig):
        self.config = config
        self.buffer = deque(maxlen=config.size)
        self.priorities = deque(maxlen=config.size) if config.prioritized else None
        self._lock = threading.Lock()
        self._position = 0
    
    def add_experience(self, experience: Experience):
        """Add single experience to buffer"""
        with self._lock:
            self.buffer.append(experience)
            if self.priorities is not None:
                self.priorities.append(experience.priority)
    
    def add_trajectory(self, trajectory: Trajectory):
        """Add trajectory to buffer by converting to experiences"""
        for i in range(len(trajectory.states) - 1):
            experience = Experience(
                state=trajectory.states[i],
                action=trajectory.actions[i],
                reward=trajectory.rewards[i],
                next_state=trajectory.states[i + 1],
                done=trajectory.dones[i],
                info=trajectory.infos[i] if i < len(trajectory.infos) else {}
            )
            self.add_experience(experience)
    
    def add(self, traj_batch: TrajectoryBatch):
        """Add trajectory batch to buffer"""
        for trajectory in traj_batch.trajectories:
            self.add_trajectory(trajectory)
        
        logger.debug(f"Added {len(traj_batch.trajectories)} trajectories to buffer. "
                    f"Buffer size: {len(self.buffer)}")
    
    def sample(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Sample batch from buffer"""
        batch_size = batch_size or self.config.batch_size
        
        with self._lock:
            if len(self.buffer) < self.config.min_size:
                raise ValueError(f"Buffer has {len(self.buffer)} experiences, "
                               f"minimum required: {self.config.min_size}")
            
            if self.config.prioritized and self.priorities:
                indices = self._sample_prioritized(batch_size)
            else:
                indices = random.sample(range(len(self.buffer)), 
                                      min(batch_size, len(self.buffer)))
            
            experiences = [self.buffer[i] for i in indices]
        
        return self._experiences_to_batch(experiences, indices)
    
    def _sample_prioritized(self, batch_size: int) -> List[int]:
        """Sample indices using prioritized replay"""
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.config.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return indices.tolist()
    
    def _experiences_to_batch(self, experiences: List[Experience], indices: List[int]) -> Dict[str, Any]:
        """Convert experiences to batch format"""
        states = [exp.state for exp in experiences]
        actions = [exp.action for exp in experiences]
        rewards = [exp.reward for exp in experiences]
        next_states = [exp.next_state for exp in experiences]
        dones = [exp.done for exp in experiences]
        
        batch = {
            "states": np.array(states) if states and isinstance(states[0], (int, float, np.ndarray)) else states,
            "actions": np.array(actions) if actions and isinstance(actions[0], (int, float)) else actions,
            "rewards": np.array(rewards),
            "next_states": np.array(next_states) if next_states and isinstance(next_states[0], (int, float, np.ndarray)) else next_states,
            "dones": np.array(dones),
            "indices": indices
        }
        
        # Add importance sampling weights for prioritized replay
        if self.config.prioritized:
            weights = self._compute_importance_weights(indices)
            batch["weights"] = weights
        
        return batch
    
    def _compute_importance_weights(self, indices: List[int]) -> np.ndarray:
        """Compute importance sampling weights"""
        priorities = np.array([self.priorities[i] for i in indices])
        probabilities = priorities ** self.config.alpha
        probabilities /= sum(self.priorities) ** self.config.alpha
        
        weights = (len(self.buffer) * probabilities) ** (-self.config.beta)
        weights /= weights.max()  # Normalize
        
        return weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for prioritized replay"""
        if not self.config.prioritized:
            return
        
        with self._lock:
            for idx, priority in zip(indices, priorities):
                if 0 <= idx < len(self.priorities):
                    self.priorities[idx] = priority
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough samples for training"""
        return len(self.buffer) >= self.config.min_size
    
    def clear(self):
        """Clear buffer"""
        with self._lock:
            self.buffer.clear()
            if self.priorities:
                self.priorities.clear()

class EpisodeBuffer:
    """Buffer for on-policy algorithms that stores complete episodes"""
    
    def __init__(self, config: BufferConfig):
        self.config = config
        self.episodes: List[Trajectory] = []
        self._lock = threading.Lock()
    
    def add(self, traj_batch: TrajectoryBatch):
        """Add trajectory batch"""
        with self._lock:
            self.episodes.extend(traj_batch.trajectories)
            
            # Keep only recent episodes if buffer is full
            if len(self.episodes) > self.config.size:
                self.episodes = self.episodes[-self.config.size:]
        
        logger.debug(f"Added {len(traj_batch.trajectories)} episodes to buffer. "
                    f"Buffer size: {len(self.episodes)}")
    
    def sample(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Sample episodes and convert to batch format"""
        with self._lock:
            if not self.episodes:
                raise ValueError("Episode buffer is empty")
            
            # For on-policy, typically use all recent data
            sampled_episodes = self.episodes[-self.config.batch_size:] if batch_size is None else self.episodes
        
        return self._episodes_to_batch(sampled_episodes)
    
    def _episodes_to_batch(self, episodes: List[Trajectory]) -> Dict[str, Any]:
        """Convert episodes to batch format"""
        all_states, all_actions, all_rewards, all_dones = [], [], [], []
        episode_lengths = []
        
        for episode in episodes:
            all_states.extend(episode.states[:-1])  # Exclude final state
            all_actions.extend(episode.actions)
            all_rewards.extend(episode.rewards)
            all_dones.extend(episode.dones)
            episode_lengths.append(len(episode.actions))
        
        return {
            "states": np.array(all_states) if all_states and isinstance(all_states[0], (int, float, np.ndarray)) else all_states,
            "actions": np.array(all_actions) if all_actions and isinstance(all_actions[0], (int, float)) else all_actions,
            "rewards": np.array(all_rewards),
            "dones": np.array(all_dones),
            "episode_lengths": episode_lengths,
            "num_episodes": len(episodes)
        }
    
    def size(self) -> int:
        """Get number of episodes in buffer"""
        return len(self.episodes)
    
    def total_steps(self) -> int:
        """Get total number of steps across all episodes"""
        return sum(len(episode.actions) for episode in self.episodes)
    
    def is_ready(self) -> bool:
        """Check if buffer has episodes for training"""
        return len(self.episodes) > 0
    
    def clear(self):
        """Clear buffer"""
        with self._lock:
            self.episodes.clear()

class BufferManager:
    """Manager for different types of replay buffers"""
    
    def __init__(self):
        self.buffers: Dict[str, Any] = {}
    
    def create_buffer(self, name: str, config: BufferConfig, buffer_type: str = "replay") -> Any:
        """Create buffer of specified type"""
        if buffer_type == "replay":
            buffer = ReplayBuffer(config)
        elif buffer_type == "episode":
            buffer = EpisodeBuffer(config)
        else:
            raise ValueError(f"Unknown buffer type: {buffer_type}")
        
        self.buffers[name] = buffer
        logger.info(f"Created {buffer_type} buffer: {name}")
        return buffer
    
    def get_buffer(self, name: str) -> Optional[Any]:
        """Get buffer by name"""
        return self.buffers.get(name)
    
    def clear_all(self):
        """Clear all buffers"""
        for buffer in self.buffers.values():
            buffer.clear()
        logger.info("Cleared all buffers")