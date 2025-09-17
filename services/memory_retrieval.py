#!/usr/bin/env python3
"""
Advanced Memory Retrieval Algorithms for Agent Learning
Implements contextual, associative, and reinforcement-based retrieval
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    """Memory retrieval strategies"""
    SIMILARITY = "similarity"           # Vector similarity search
    TEMPORAL = "temporal"               # Time-based retrieval
    ASSOCIATIVE = "associative"         # Graph-based associations
    REINFORCEMENT = "reinforcement"     # RL-based prioritization
    HIERARCHICAL = "hierarchical"       # Tree-based retrieval
    CONTEXTUAL = "contextual"           # Context-aware retrieval
    HYBRID = "hybrid"                   # Multi-strategy combination


@dataclass
class RetrievalContext:
    """Context for memory retrieval"""
    agent_id: str
    current_task: Optional[str] = None
    current_state: Optional[Dict[str, Any]] = None
    query_embedding: Optional[np.ndarray] = None
    time_window: Optional[timedelta] = None
    importance_threshold: float = 0.0
    strength_threshold: float = 0.0
    max_results: int = 10
    strategies: List[RetrievalStrategy] = None


class MemoryRetriever:
    """Advanced memory retrieval system"""
    
    def __init__(self, db_pool, cache=None):
        self.db_pool = db_pool
        self.cache = cache
        self.strategy_weights = {
            RetrievalStrategy.SIMILARITY: 0.3,
            RetrievalStrategy.TEMPORAL: 0.15,
            RetrievalStrategy.ASSOCIATIVE: 0.2,
            RetrievalStrategy.REINFORCEMENT: 0.15,
            RetrievalStrategy.HIERARCHICAL: 0.1,
            RetrievalStrategy.CONTEXTUAL: 0.1
        }
    
    async def retrieve(self, context: RetrievalContext) -> List[Dict[str, Any]]:
        """
        Main retrieval method using multiple strategies
        """
        if not context.strategies:
            context.strategies = [RetrievalStrategy.HYBRID]
        
        # Single strategy retrieval
        if len(context.strategies) == 1 and context.strategies[0] != RetrievalStrategy.HYBRID:
            return await self._retrieve_single_strategy(context, context.strategies[0])
        
        # Hybrid retrieval - combine multiple strategies
        all_memories = {}
        strategy_scores = {}
        
        for strategy in [s for s in RetrievalStrategy if s != RetrievalStrategy.HYBRID]:
            memories = await self._retrieve_single_strategy(context, strategy)
            for mem in memories:
                mem_id = mem['id']
                if mem_id not in all_memories:
                    all_memories[mem_id] = mem
                    strategy_scores[mem_id] = {}
                strategy_scores[mem_id][strategy] = mem.get('score', 1.0)
        
        # Combine scores using weighted average
        final_scores = {}
        for mem_id, scores in strategy_scores.items():
            final_score = 0
            total_weight = 0
            for strategy, score in scores.items():
                weight = self.strategy_weights.get(strategy, 0.1)
                final_score += score * weight
                total_weight += weight
            final_scores[mem_id] = final_score / total_weight if total_weight > 0 else 0
        
        # Sort by final score and return top results
        sorted_memories = sorted(
            all_memories.values(),
            key=lambda m: final_scores.get(m['id'], 0),
            reverse=True
        )
        
        return sorted_memories[:context.max_results]
    
    async def _retrieve_single_strategy(
        self, context: RetrievalContext, strategy: RetrievalStrategy
    ) -> List[Dict[str, Any]]:
        """Retrieve using a single strategy"""
        
        if strategy == RetrievalStrategy.SIMILARITY:
            return await self._similarity_retrieval(context)
        elif strategy == RetrievalStrategy.TEMPORAL:
            return await self._temporal_retrieval(context)
        elif strategy == RetrievalStrategy.ASSOCIATIVE:
            return await self._associative_retrieval(context)
        elif strategy == RetrievalStrategy.REINFORCEMENT:
            return await self._reinforcement_retrieval(context)
        elif strategy == RetrievalStrategy.HIERARCHICAL:
            return await self._hierarchical_retrieval(context)
        elif strategy == RetrievalStrategy.CONTEXTUAL:
            return await self._contextual_retrieval(context)
        else:
            return []
    
    async def _similarity_retrieval(self, context: RetrievalContext) -> List[Dict[str, Any]]:
        """Vector similarity-based retrieval"""
        if not context.query_embedding:
            return []
        
        query = """
            SELECT *, 
                   1 - (embedding <=> %s::vector) as similarity_score
            FROM agent_memories
            WHERE agent_id = %s
              AND is_active = TRUE
              AND embedding IS NOT NULL
              AND strength >= %s
            ORDER BY similarity_score DESC
            LIMIT %s
        """
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (
                    context.query_embedding.tolist(),
                    context.agent_id,
                    context.strength_threshold,
                    context.max_results
                ))
                results = cur.fetchall()
                
                # Add score to results
                for r in results:
                    r['score'] = r['similarity_score']
                
                return results
        finally:
            self.db_pool.putconn(conn)
    
    async def _temporal_retrieval(self, context: RetrievalContext) -> List[Dict[str, Any]]:
        """Time-based retrieval with recency bias"""
        query = """
            SELECT *,
                   EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - accessed_at)) as time_diff,
                   EXP(-EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - accessed_at)) / 86400) as recency_score
            FROM agent_memories
            WHERE agent_id = %s
              AND is_active = TRUE
              AND strength >= %s
        """
        
        params = [context.agent_id, context.strength_threshold]
        
        if context.time_window:
            query += " AND accessed_at >= %s"
            params.append(datetime.now() - context.time_window)
        
        query += " ORDER BY recency_score DESC LIMIT %s"
        params.append(context.max_results)
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                results = cur.fetchall()
                
                for r in results:
                    r['score'] = r['recency_score']
                
                return results
        finally:
            self.db_pool.putconn(conn)
    
    async def _associative_retrieval(self, context: RetrievalContext) -> List[Dict[str, Any]]:
        """Graph-based associative retrieval"""
        
        # First get recent memories as seeds
        seed_query = """
            SELECT id FROM agent_memories
            WHERE agent_id = %s
              AND is_active = TRUE
            ORDER BY accessed_at DESC
            LIMIT 5
        """
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(seed_query, (context.agent_id,))
                seed_ids = [r['id'] for r in cur.fetchall()]
                
                if not seed_ids:
                    return []
                
                # Get associated memories using graph traversal
                assoc_query = """
                    WITH RECURSIVE associated AS (
                        -- Start with seed memories
                        SELECT m.*, 0 as depth, 1.0 as assoc_score
                        FROM agent_memories m
                        WHERE m.id = ANY(%s)
                        
                        UNION ALL
                        
                        -- Traverse associations
                        SELECT m.*, a.depth + 1, a.assoc_score * ma.strength as assoc_score
                        FROM associated a
                        JOIN memory_associations ma ON 
                            (ma.memory_a_id = a.id OR ma.memory_b_id = a.id)
                        JOIN agent_memories m ON 
                            (m.id = CASE 
                                WHEN ma.memory_a_id = a.id THEN ma.memory_b_id 
                                ELSE ma.memory_a_id 
                            END)
                        WHERE a.depth < 2  -- Max traversal depth
                          AND m.is_active = TRUE
                          AND m.strength >= %s
                    )
                    SELECT DISTINCT ON (id) *
                    FROM associated
                    ORDER BY id, assoc_score DESC
                    LIMIT %s
                """
                
                cur.execute(assoc_query, (
                    seed_ids,
                    context.strength_threshold,
                    context.max_results
                ))
                
                results = cur.fetchall()
                for r in results:
                    r['score'] = r['assoc_score']
                
                return results
                
        finally:
            self.db_pool.putconn(conn)
    
    async def _reinforcement_retrieval(self, context: RetrievalContext) -> List[Dict[str, Any]]:
        """RL-based retrieval using reward signals"""
        
        # Get memories that led to high rewards in similar states
        query = """
            SELECT m.*,
                   AVG(e.reward) as avg_reward,
                   COUNT(e.id) as use_count,
                   AVG(e.reward) * LOG(COUNT(e.id) + 1) as rl_score
            FROM agent_memories m
            JOIN experience_replay_buffer e ON e.memory_id = m.id
            WHERE m.agent_id = %s
              AND m.is_active = TRUE
              AND m.strength >= %s
              AND e.reward > 0
            GROUP BY m.id
            ORDER BY rl_score DESC
            LIMIT %s
        """
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (
                    context.agent_id,
                    context.strength_threshold,
                    context.max_results
                ))
                
                results = cur.fetchall()
                for r in results:
                    r['score'] = r['rl_score']
                
                return results
                
        finally:
            self.db_pool.putconn(conn)
    
    async def _hierarchical_retrieval(self, context: RetrievalContext) -> List[Dict[str, Any]]:
        """Tree-based hierarchical retrieval"""
        
        query = """
            WITH RECURSIVE memory_tree AS (
                -- Get root memories (consolidated or high importance)
                SELECT *, 0 as level, 1.0 as hier_score
                FROM agent_memories
                WHERE agent_id = %s
                  AND is_active = TRUE
                  AND (is_consolidated = TRUE OR importance IN ('critical', 'high'))
                  AND parent_memory_id IS NULL
                
                UNION ALL
                
                -- Get child memories
                SELECT m.*, mt.level + 1, mt.hier_score * 0.8 as hier_score
                FROM agent_memories m
                JOIN memory_tree mt ON m.parent_memory_id = mt.id
                WHERE m.is_active = TRUE
                  AND m.strength >= %s
            )
            SELECT * FROM memory_tree
            ORDER BY hier_score DESC, level ASC
            LIMIT %s
        """
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (
                    context.agent_id,
                    context.strength_threshold,
                    context.max_results
                ))
                
                results = cur.fetchall()
                for r in results:
                    r['score'] = r['hier_score']
                
                return results
                
        finally:
            self.db_pool.putconn(conn)
    
    async def _contextual_retrieval(self, context: RetrievalContext) -> List[Dict[str, Any]]:
        """Context-aware retrieval based on current state"""
        
        if not context.current_state:
            return []
        
        # Extract context features
        context_json = json.dumps(context.current_state)
        
        query = """
            SELECT *,
                   (CASE 
                    WHEN context::text LIKE %s THEN 0.5
                    ELSE 0
                   END +
                   CASE
                    WHEN source = %s THEN 0.3
                    ELSE 0
                   END +
                   strength * 0.2) as context_score
            FROM agent_memories
            WHERE agent_id = %s
              AND is_active = TRUE
              AND strength >= %s
            ORDER BY context_score DESC
            LIMIT %s
        """
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Simple context matching - can be enhanced
                context_pattern = f"%{context.current_task or ''}%"
                
                cur.execute(query, (
                    context_pattern,
                    context.current_task,
                    context.agent_id,
                    context.strength_threshold,
                    context.max_results
                ))
                
                results = cur.fetchall()
                for r in results:
                    r['score'] = r['context_score']
                
                return results
                
        finally:
            self.db_pool.putconn(conn)
    
    async def adaptive_retrieval(
        self, 
        agent_id: str,
        task_type: str,
        performance_history: List[float]
    ) -> Dict[str, float]:
        """
        Adaptively adjust strategy weights based on task performance
        Uses multi-armed bandit approach
        """
        
        # Calculate performance trend
        if len(performance_history) < 2:
            return self.strategy_weights
        
        recent_perf = np.mean(performance_history[-5:])
        overall_perf = np.mean(performance_history)
        
        # Adjust weights based on performance
        if recent_perf > overall_perf * 1.1:  # Recent improvement
            # Increase weights of recently used strategies
            logger.info(f"Performance improving for {agent_id}, maintaining strategy weights")
        elif recent_perf < overall_perf * 0.9:  # Recent decline
            # Explore different strategies
            logger.info(f"Performance declining for {agent_id}, exploring new strategies")
            
            # Add exploration noise
            noise = np.random.dirichlet(np.ones(len(self.strategy_weights)) * 0.5)
            weights = list(self.strategy_weights.values())
            adjusted = 0.8 * np.array(weights) + 0.2 * noise
            
            # Update weights
            for i, strategy in enumerate(self.strategy_weights.keys()):
                self.strategy_weights[strategy] = float(adjusted[i])
        
        # Normalize weights
        total = sum(self.strategy_weights.values())
        for k in self.strategy_weights:
            self.strategy_weights[k] /= total
        
        return self.strategy_weights
    
    async def get_memory_clusters(
        self, 
        agent_id: str,
        n_clusters: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Group memories into semantic clusters for better organization
        """
        
        # Get all active memories with embeddings
        query = """
            SELECT id, embedding, content, memory_type
            FROM agent_memories
            WHERE agent_id = %s
              AND is_active = TRUE
              AND embedding IS NOT NULL
            LIMIT 1000
        """
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (agent_id,))
                memories = cur.fetchall()
                
                if len(memories) < n_clusters:
                    return [memories]
                
                # Extract embeddings
                embeddings = np.array([m['embedding'] for m in memories])
                
                # Simple k-means clustering
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings)
                
                # Group memories by cluster
                clustered = [[] for _ in range(n_clusters)]
                for i, mem in enumerate(memories):
                    clustered[clusters[i]].append(mem)
                
                return clustered
                
        except ImportError:
            logger.warning("scikit-learn not installed, returning unclustered memories")
            return [memories]
        finally:
            self.db_pool.putconn(conn)