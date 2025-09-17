"""
Advanced Memory Management System for Agent Lightning
Implements episodic, semantic, and working memory with retrieval mechanisms
Based on cognitive architectures for enhanced agent learning
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
import pickle
import hashlib


@dataclass
class MemoryEntry:
    """Single memory entry with metadata"""
    content: Dict
    timestamp: float
    importance: float
    access_count: int = 0
    last_accessed: float = None
    embedding: np.ndarray = None
    memory_type: str = "episodic"  # episodic, semantic, or procedural
    decay_rate: float = 0.99
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data
    
    def update_access(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def get_recency_score(self, current_time: float) -> float:
        """Calculate recency score with exponential decay"""
        if self.last_accessed is None:
            time_diff = current_time - self.timestamp
        else:
            time_diff = current_time - self.last_accessed
        
        # Exponential decay over time (in hours)
        hours_passed = time_diff / 3600
        return self.decay_rate ** hours_passed
    
    def get_relevance_score(self) -> float:
        """Calculate overall relevance combining importance, recency, and access frequency"""
        current_time = time.time()
        recency = self.get_recency_score(current_time)
        frequency = min(1.0, self.access_count / 10)  # Normalize frequency
        
        # Weighted combination
        score = (0.4 * self.importance + 
                0.3 * recency + 
                0.3 * frequency)
        
        return score


class MemoryManager:
    """
    Advanced memory management system for agents
    Implements cognitive-inspired memory architecture
    """
    
    def __init__(self, 
                 max_episodic_size: int = 10000,
                 max_semantic_size: int = 5000,
                 max_working_size: int = 20,
                 embedding_dim: int = 768,
                 persistence_path: Optional[Path] = None):
        """
        Initialize memory manager
        
        Args:
            max_episodic_size: Maximum episodic memory entries
            max_semantic_size: Maximum semantic memory entries
            max_working_size: Maximum working memory entries
            embedding_dim: Dimension of memory embeddings
            persistence_path: Path for persistent memory storage
        """
        # Memory stores
        self.episodic_memory: List[MemoryEntry] = []
        self.semantic_memory: Dict[str, MemoryEntry] = {}
        self.working_memory: deque = deque(maxlen=max_working_size)
        self.procedural_memory: Dict[str, Dict] = {}  # Skills and procedures
        
        # Size limits
        self.max_episodic_size = max_episodic_size
        self.max_semantic_size = max_semantic_size
        self.max_working_size = max_working_size
        
        # Embedding configuration
        self.embedding_dim = embedding_dim
        self.embedding_model = None  # Will be initialized if needed
        
        # Memory indices for fast retrieval
        self.episodic_index = {}  # Hash -> memory entry
        self.semantic_index = {}  # Concept -> memory entries
        
        # Persistence
        self.persistence_path = Path(persistence_path) if persistence_path else None
        if self.persistence_path:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            self.load_persistent_memory()
        
        # Statistics
        self.stats = {
            "total_stored": 0,
            "total_retrieved": 0,
            "consolidations": 0,
            "forgotten": 0
        }
        
        print(f"🧠 Memory Manager initialized")
        print(f"   Episodic capacity: {max_episodic_size}")
        print(f"   Semantic capacity: {max_semantic_size}")
        print(f"   Working memory: {max_working_size}")
    
    def store_episodic(self, content: Dict, importance: float = 0.5) -> str:
        """
        Store an episodic memory (specific experience)
        
        Args:
            content: Memory content
            importance: Importance score (0-1)
            
        Returns:
            Memory ID
        """
        # Create memory entry
        memory = MemoryEntry(
            content=content,
            timestamp=time.time(),
            importance=importance,
            memory_type="episodic"
        )
        
        # Generate embedding if model available
        if self.embedding_model:
            memory.embedding = self._generate_embedding(content)
        
        # Add to episodic memory
        self.episodic_memory.append(memory)
        
        # Create index entry
        memory_id = self._generate_memory_id(content)
        self.episodic_index[memory_id] = memory
        
        # Add to working memory
        self.working_memory.append(memory)
        
        # Consolidate if needed
        if len(self.episodic_memory) > self.max_episodic_size:
            self._consolidate_episodic_memory()
        
        self.stats["total_stored"] += 1
        
        return memory_id
    
    def store_semantic(self, concept: str, content: Dict, importance: float = 0.7) -> str:
        """
        Store semantic memory (general knowledge)
        
        Args:
            concept: Concept or category
            content: Knowledge content
            importance: Importance score
            
        Returns:
            Memory ID
        """
        # Check if concept exists and should be merged
        if concept in self.semantic_memory:
            existing = self.semantic_memory[concept]
            # Merge with existing knowledge
            merged_content = self._merge_semantic_knowledge(
                existing.content, content
            )
            existing.content = merged_content
            existing.importance = max(existing.importance, importance)
            existing.update_access()
            memory_id = concept
        else:
            # Create new semantic memory
            memory = MemoryEntry(
                content=content,
                timestamp=time.time(),
                importance=importance,
                memory_type="semantic"
            )
            
            if self.embedding_model:
                memory.embedding = self._generate_embedding(content)
            
            self.semantic_memory[concept] = memory
            memory_id = concept
            
            # Index by concept
            if concept not in self.semantic_index:
                self.semantic_index[concept] = []
            self.semantic_index[concept].append(memory)
        
        # Consolidate if needed
        if len(self.semantic_memory) > self.max_semantic_size:
            self._consolidate_semantic_memory()
        
        self.stats["total_stored"] += 1
        
        return memory_id
    
    def store_procedural(self, skill_name: str, procedure: Dict) -> str:
        """
        Store procedural memory (how to do things)
        
        Args:
            skill_name: Name of the skill/procedure
            procedure: Procedure details (steps, conditions, etc.)
            
        Returns:
            Skill ID
        """
        self.procedural_memory[skill_name] = {
            "procedure": procedure,
            "timestamp": time.time(),
            "success_count": 0,
            "failure_count": 0,
            "last_used": None
        }
        
        return skill_name
    
    def retrieve_relevant(self, query: Dict, k: int = 5, 
                         memory_types: List[str] = None) -> List[MemoryEntry]:
        """
        Retrieve k most relevant memories for a query
        
        Args:
            query: Query content
            k: Number of memories to retrieve
            memory_types: Types of memory to search (episodic, semantic, procedural)
            
        Returns:
            List of relevant memories
        """
        if memory_types is None:
            memory_types = ["episodic", "semantic"]
        
        all_candidates = []
        
        # Search episodic memory
        if "episodic" in memory_types:
            episodic_candidates = self._search_episodic(query, k * 2)
            all_candidates.extend(episodic_candidates)
        
        # Search semantic memory  
        if "semantic" in memory_types:
            semantic_candidates = self._search_semantic(query, k * 2)
            all_candidates.extend(semantic_candidates)
        
        # Rank by relevance
        ranked = self._rank_memories(all_candidates, query)
        
        # Update access statistics
        for memory in ranked[:k]:
            memory.update_access()
        
        self.stats["total_retrieved"] += min(k, len(ranked))
        
        return ranked[:k]
    
    def retrieve_recent(self, k: int = 5) -> List[MemoryEntry]:
        """Retrieve k most recent memories from working memory"""
        recent = list(self.working_memory)[-k:]
        recent.reverse()  # Most recent first
        
        for memory in recent:
            memory.update_access()
        
        return recent
    
    def retrieve_procedural(self, skill_name: str) -> Optional[Dict]:
        """Retrieve a procedural memory (skill)"""
        if skill_name in self.procedural_memory:
            skill = self.procedural_memory[skill_name]
            skill["last_used"] = time.time()
            return skill["procedure"]
        return None
    
    def update_procedural_performance(self, skill_name: str, success: bool):
        """Update performance statistics for a procedural memory"""
        if skill_name in self.procedural_memory:
            if success:
                self.procedural_memory[skill_name]["success_count"] += 1
            else:
                self.procedural_memory[skill_name]["failure_count"] += 1
    
    def consolidate_to_semantic(self, threshold: int = 3):
        """
        Consolidate repeated episodic memories into semantic memory
        This mimics how humans form general knowledge from specific experiences
        """
        # Group similar episodic memories
        patterns = {}
        
        for memory in self.episodic_memory:
            # Simple pattern extraction (can be made more sophisticated)
            pattern_key = self._extract_pattern(memory.content)
            if pattern_key not in patterns:
                patterns[pattern_key] = []
            patterns[pattern_key].append(memory)
        
        # Consolidate patterns that appear frequently
        for pattern_key, memories in patterns.items():
            if len(memories) >= threshold:
                # Create semantic memory from pattern
                consolidated = self._consolidate_pattern(memories)
                self.store_semantic(
                    concept=pattern_key,
                    content=consolidated,
                    importance=0.8
                )
                
                # Optionally remove consolidated episodic memories
                for memory in memories:
                    memory.importance *= 0.5  # Reduce importance
        
        self.stats["consolidations"] += 1
    
    def _search_episodic(self, query: Dict, limit: int) -> List[MemoryEntry]:
        """Search episodic memory"""
        if self.embedding_model:
            # Use embedding similarity
            query_embedding = self._generate_embedding(query)
            similarities = []
            
            for memory in self.episodic_memory:
                if memory.embedding is not None:
                    sim = self._cosine_similarity(query_embedding, memory.embedding)
                    similarities.append((memory, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [m for m, _ in similarities[:limit]]
        else:
            # Fallback to recency and importance
            scored = [(m, m.get_relevance_score()) for m in self.episodic_memory]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [m for m, _ in scored[:limit]]
    
    def _search_semantic(self, query: Dict, limit: int) -> List[MemoryEntry]:
        """Search semantic memory"""
        # Extract concepts from query
        query_concepts = self._extract_concepts(query)
        
        candidates = []
        for concept in query_concepts:
            if concept in self.semantic_memory:
                candidates.append(self.semantic_memory[concept])
        
        # Add related concepts
        for concept, memory in self.semantic_memory.items():
            if any(qc in concept for qc in query_concepts):
                if memory not in candidates:
                    candidates.append(memory)
        
        # Rank by relevance
        scored = [(m, m.get_relevance_score()) for m in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [m for m, _ in scored[:limit]]
    
    def _rank_memories(self, memories: List[MemoryEntry], query: Dict) -> List[MemoryEntry]:
        """Rank memories by relevance to query"""
        scored = []
        
        for memory in memories:
            # Calculate comprehensive score
            relevance = memory.get_relevance_score()
            
            # Boost recent memories
            if memory in self.working_memory:
                relevance *= 1.2
            
            # Boost semantic memories for general queries
            if memory.memory_type == "semantic":
                relevance *= 1.1
            
            scored.append((memory, relevance))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [m for m, _ in scored]
    
    def _consolidate_episodic_memory(self):
        """Remove least important episodic memories when at capacity"""
        # Sort by relevance
        self.episodic_memory.sort(key=lambda m: m.get_relevance_score())
        
        # Remove least relevant memories
        to_remove = len(self.episodic_memory) - self.max_episodic_size
        if to_remove > 0:
            removed = self.episodic_memory[:to_remove]
            self.episodic_memory = self.episodic_memory[to_remove:]
            
            # Update indices
            for memory in removed:
                memory_id = self._generate_memory_id(memory.content)
                if memory_id in self.episodic_index:
                    del self.episodic_index[memory_id]
            
            self.stats["forgotten"] += to_remove
    
    def _consolidate_semantic_memory(self):
        """Consolidate semantic memory when at capacity"""
        # Sort by relevance
        items = list(self.semantic_memory.items())
        items.sort(key=lambda x: x[1].get_relevance_score())
        
        # Remove least relevant
        to_remove = len(items) - self.max_semantic_size
        if to_remove > 0:
            for concept, _ in items[:to_remove]:
                del self.semantic_memory[concept]
                if concept in self.semantic_index:
                    del self.semantic_index[concept]
            
            self.stats["forgotten"] += to_remove
    
    def _merge_semantic_knowledge(self, existing: Dict, new: Dict) -> Dict:
        """Merge new knowledge with existing semantic memory"""
        merged = existing.copy()
        
        for key, value in new.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Merge lists
                merged[key] = list(set(merged[key] + value))
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                # Recursively merge dicts
                merged[key] = self._merge_semantic_knowledge(merged[key], value)
            else:
                # Keep the newer value
                merged[key] = value
        
        return merged
    
    def _extract_pattern(self, content: Dict) -> str:
        """Extract pattern from content for consolidation"""
        # Simple pattern extraction - can be made more sophisticated
        pattern_keys = []
        
        if "task_type" in content:
            pattern_keys.append(content["task_type"])
        if "action_type" in content:
            pattern_keys.append(content["action_type"])
        if "category" in content:
            pattern_keys.append(content["category"])
        
        return "_".join(pattern_keys) if pattern_keys else "general"
    
    def _consolidate_pattern(self, memories: List[MemoryEntry]) -> Dict:
        """Consolidate multiple memories into a pattern"""
        consolidated = {
            "pattern_type": "consolidated",
            "instance_count": len(memories),
            "common_features": {},
            "variations": []
        }
        
        # Extract common features
        if memories:
            first_content = memories[0].content
            for key in first_content:
                values = [m.content.get(key) for m in memories if key in m.content]
                if len(values) == len(memories):
                    # Common to all
                    if all(v == values[0] for v in values):
                        consolidated["common_features"][key] = values[0]
                    else:
                        consolidated["variations"].append({
                            "key": key,
                            "values": list(set(str(v) for v in values))
                        })
        
        return consolidated
    
    def _extract_concepts(self, content: Dict) -> List[str]:
        """Extract concepts from content"""
        concepts = []
        
        # Extract from various fields
        for key in ["concepts", "topics", "categories", "tags", "keywords"]:
            if key in content:
                if isinstance(content[key], list):
                    concepts.extend(content[key])
                else:
                    concepts.append(str(content[key]))
        
        # Extract from text content
        if "text" in content or "content" in content:
            text = content.get("text", content.get("content", ""))
            # Simple keyword extraction (can use NLP methods)
            words = text.lower().split()
            # Add significant words as concepts
            concepts.extend([w for w in words if len(w) > 5][:5])
        
        return list(set(concepts))
    
    def _generate_embedding(self, content: Dict) -> np.ndarray:
        """Generate embedding for content (placeholder - use real model in production)"""
        # In production, use sentence-transformers or similar
        # For now, create random embedding
        text = json.dumps(content)
        # Use hash for consistent pseudo-random embedding
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        return np.random.randn(self.embedding_dim)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _generate_memory_id(self, content: Dict) -> str:
        """Generate unique ID for memory"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    def save_persistent_memory(self):
        """Save memory to disk"""
        if not self.persistence_path:
            return
        
        # Save episodic memory
        episodic_file = self.persistence_path / "episodic_memory.pkl"
        with open(episodic_file, 'wb') as f:
            pickle.dump(self.episodic_memory, f)
        
        # Save semantic memory
        semantic_file = self.persistence_path / "semantic_memory.pkl"
        with open(semantic_file, 'wb') as f:
            pickle.dump(self.semantic_memory, f)
        
        # Save procedural memory
        procedural_file = self.persistence_path / "procedural_memory.pkl"
        with open(procedural_file, 'wb') as f:
            pickle.dump(self.procedural_memory, f)
        
        print(f"💾 Memory saved to {self.persistence_path}")
    
    def load_persistent_memory(self):
        """Load memory from disk"""
        if not self.persistence_path:
            return
        
        # Load episodic memory
        episodic_file = self.persistence_path / "episodic_memory.pkl"
        if episodic_file.exists():
            with open(episodic_file, 'rb') as f:
                self.episodic_memory = pickle.load(f)
        
        # Load semantic memory
        semantic_file = self.persistence_path / "semantic_memory.pkl"
        if semantic_file.exists():
            with open(semantic_file, 'rb') as f:
                self.semantic_memory = pickle.load(f)
        
        # Load procedural memory
        procedural_file = self.persistence_path / "procedural_memory.pkl"
        if procedural_file.exists():
            with open(procedural_file, 'rb') as f:
                self.procedural_memory = pickle.load(f)
        
        print(f"📂 Memory loaded from {self.persistence_path}")
    
    def get_statistics(self) -> Dict:
        """Get memory system statistics"""
        stats = self.stats.copy()
        stats.update({
            "episodic_count": len(self.episodic_memory),
            "semantic_count": len(self.semantic_memory),
            "procedural_count": len(self.procedural_memory),
            "working_count": len(self.working_memory),
            "episodic_capacity_used": len(self.episodic_memory) / self.max_episodic_size,
            "semantic_capacity_used": len(self.semantic_memory) / self.max_semantic_size
        })
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Testing Advanced Memory Management System")
    print("=" * 60)
    
    # Create memory manager
    memory = MemoryManager(
        max_episodic_size=1000,
        max_semantic_size=500,
        max_working_size=10,
        persistence_path=Path("./memory_storage")
    )
    
    # Store episodic memories
    print("\n📝 Storing episodic memories...")
    for i in range(5):
        memory.store_episodic(
            content={
                "experience": f"Task completion {i}",
                "result": "success",
                "context": {"difficulty": "medium"},
                "task_type": "problem_solving"
            },
            importance=0.5 + i * 0.1
        )
    
    # Store semantic knowledge
    print("\n📚 Storing semantic knowledge...")
    memory.store_semantic(
        concept="reinforcement_learning",
        content={
            "definition": "Learning through interaction with environment",
            "key_concepts": ["agent", "environment", "reward", "policy"],
            "applications": ["robotics", "game_playing", "optimization"]
        },
        importance=0.9
    )
    
    # Store procedural knowledge
    print("\n🔧 Storing procedural knowledge...")
    memory.store_procedural(
        skill_name="solve_math_problem",
        procedure={
            "steps": ["understand", "plan", "execute", "verify"],
            "tools": ["calculator", "formulas"],
            "conditions": {"requires": "mathematical_notation"}
        }
    )
    
    # Retrieve relevant memories
    print("\n🔍 Retrieving relevant memories...")
    query = {"task": "solve a problem", "context": {"type": "problem_solving"}}
    relevant = memory.retrieve_relevant(query, k=3)
    
    for i, mem in enumerate(relevant):
        print(f"  {i+1}. Type: {mem.memory_type}, Importance: {mem.importance:.2f}")
        print(f"     Content: {str(mem.content)[:100]}...")
    
    # Consolidate to semantic memory
    print("\n🔄 Consolidating episodic to semantic...")
    memory.consolidate_to_semantic(threshold=2)
    
    # Get statistics
    print("\n📊 Memory Statistics:")
    stats = memory.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Save memory
    memory.save_persistent_memory()
    
    print("\n✅ Memory system test complete!")