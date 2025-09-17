#!/usr/bin/env python3
"""
Memory Consolidation System for Agent Learning
Implements memory consolidation, pruning, and optimization strategies
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import logging
import asyncio
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ConsolidationType(str, Enum):
    """Types of memory consolidation"""
    SLEEP_BASED = "sleep_based"         # Periodic consolidation (like sleep)
    THRESHOLD_BASED = "threshold_based"  # When memory count exceeds threshold
    SIMILARITY_BASED = "similarity_based" # Merge similar memories
    IMPORTANCE_BASED = "importance_based" # Consolidate important memories
    REHEARSAL_BASED = "rehearsal_based"  # Based on repeated access
    ABSTRACTION = "abstraction"          # Create abstract concepts


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation"""
    agent_id: str
    consolidation_type: ConsolidationType
    strength_threshold: float = 0.7
    similarity_threshold: float = 0.85
    max_memories: int = 10000
    min_age_hours: int = 24
    batch_size: int = 100
    enable_pruning: bool = True
    enable_abstraction: bool = True


class MemoryConsolidator:
    """Advanced memory consolidation system"""
    
    def __init__(self, db_pool, cache=None, retriever=None):
        self.db_pool = db_pool
        self.cache = cache
        self.retriever = retriever
        
        # Consolidation parameters
        self.consolidation_rates = {
            "critical": 0.95,    # Almost always consolidate
            "high": 0.8,         # Often consolidate
            "medium": 0.5,       # Sometimes consolidate
            "low": 0.2,          # Rarely consolidate
            "temporary": 0.05    # Almost never consolidate
        }
    
    async def consolidate(self, config: ConsolidationConfig) -> Dict[str, Any]:
        """
        Main consolidation method
        """
        start_time = datetime.now()
        
        logger.info(f"Starting {config.consolidation_type} consolidation for agent {config.agent_id}")
        
        # Select consolidation strategy
        if config.consolidation_type == ConsolidationType.SLEEP_BASED:
            result = await self._sleep_consolidation(config)
        elif config.consolidation_type == ConsolidationType.THRESHOLD_BASED:
            result = await self._threshold_consolidation(config)
        elif config.consolidation_type == ConsolidationType.SIMILARITY_BASED:
            result = await self._similarity_consolidation(config)
        elif config.consolidation_type == ConsolidationType.IMPORTANCE_BASED:
            result = await self._importance_consolidation(config)
        elif config.consolidation_type == ConsolidationType.REHEARSAL_BASED:
            result = await self._rehearsal_consolidation(config)
        elif config.consolidation_type == ConsolidationType.ABSTRACTION:
            result = await self._abstraction_consolidation(config)
        else:
            result = {"error": "Unknown consolidation type"}
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        result['duration_seconds'] = duration
        
        # Log consolidation
        await self._log_consolidation(config, result)
        
        logger.info(f"Consolidation complete: {result}")
        return result
    
    async def _sleep_consolidation(self, config: ConsolidationConfig) -> Dict[str, Any]:
        """
        Sleep-based consolidation - simulates biological memory consolidation during sleep
        """
        consolidated = 0
        pruned = 0
        abstracted = 0
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Get eligible memories (old enough, not yet consolidated)
                query = """
                    SELECT * FROM agent_memories
                    WHERE agent_id = %s
                      AND is_active = TRUE
                      AND is_consolidated = FALSE
                      AND created_at < %s
                      AND strength >= %s
                    ORDER BY strength DESC, access_count DESC
                    LIMIT %s
                """
                
                min_age = datetime.now() - timedelta(hours=config.min_age_hours)
                cur.execute(query, (
                    config.agent_id,
                    min_age,
                    config.strength_threshold,
                    config.batch_size
                ))
                
                memories = cur.fetchall()
                
                for memory in memories:
                    # Determine if memory should be consolidated
                    importance = memory['importance']
                    consolidation_prob = self.consolidation_rates.get(importance, 0.5)
                    
                    # Add randomness (stochastic consolidation)
                    if np.random.random() < consolidation_prob:
                        # Consolidate memory
                        update_query = """
                            UPDATE agent_memories
                            SET is_consolidated = TRUE,
                                consolidation_date = CURRENT_TIMESTAMP,
                                memory_type = 'consolidated',
                                strength = LEAST(1.0, strength + 0.2)
                            WHERE id = %s
                        """
                        cur.execute(update_query, (memory['id'],))
                        consolidated += 1
                        
                        # Clear cache if exists
                        if self.cache:
                            self.cache.delete(f"memory:{memory['id']}")
                
                # Prune weak temporary memories
                if config.enable_pruning:
                    prune_query = """
                        UPDATE agent_memories
                        SET is_active = FALSE
                        WHERE agent_id = %s
                          AND strength < 0.1
                          AND importance IN ('temporary', 'low')
                          AND is_consolidated = FALSE
                          AND created_at < %s
                    """
                    cur.execute(prune_query, (config.agent_id, min_age))
                    pruned = cur.rowcount
                
                conn.commit()
                
        finally:
            self.db_pool.putconn(conn)
        
        return {
            "consolidated": consolidated,
            "pruned": pruned,
            "abstracted": abstracted,
            "total_processed": len(memories)
        }
    
    async def _similarity_consolidation(self, config: ConsolidationConfig) -> Dict[str, Any]:
        """
        Merge similar memories to reduce redundancy
        """
        merged = 0
        pruned = 0
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Find similar memory pairs using vector similarity
                query = """
                    SELECT a.id as id_a, b.id as id_b,
                           a.content as content_a, b.content as content_b,
                           1 - (a.embedding <=> b.embedding) as similarity
                    FROM agent_memories a
                    JOIN agent_memories b ON a.id < b.id
                    WHERE a.agent_id = %s
                      AND b.agent_id = %s
                      AND a.is_active = TRUE
                      AND b.is_active = TRUE
                      AND a.embedding IS NOT NULL
                      AND b.embedding IS NOT NULL
                      AND 1 - (a.embedding <=> b.embedding) > %s
                    ORDER BY similarity DESC
                    LIMIT 50
                """
                
                cur.execute(query, (
                    config.agent_id,
                    config.agent_id,
                    config.similarity_threshold
                ))
                
                similar_pairs = cur.fetchall()
                
                for pair in similar_pairs:
                    # Merge memories - keep stronger one, combine content
                    merged_content = self._merge_memory_content(
                        pair['content_a'],
                        pair['content_b']
                    )
                    
                    # Update stronger memory with merged content
                    update_query = """
                        UPDATE agent_memories
                        SET content = %s,
                            strength = LEAST(1.0, strength + 0.1),
                            reinforcement_count = reinforcement_count + 1
                        WHERE id = %s
                    """
                    cur.execute(update_query, (json.dumps(merged_content), pair['id_a']))
                    
                    # Deactivate weaker memory
                    deactivate_query = """
                        UPDATE agent_memories
                        SET is_active = FALSE
                        WHERE id = %s
                    """
                    cur.execute(deactivate_query, (pair['id_b'],))
                    
                    # Create association between them
                    assoc_query = """
                        INSERT INTO memory_associations (memory_a_id, memory_b_id, association_type, strength)
                        VALUES (%s, %s, 'merged', %s)
                        ON CONFLICT (memory_a_id, memory_b_id) DO UPDATE
                        SET strength = EXCLUDED.strength
                    """
                    cur.execute(assoc_query, (pair['id_a'], pair['id_b'], pair['similarity']))
                    
                    merged += 1
                
                conn.commit()
                
        finally:
            self.db_pool.putconn(conn)
        
        return {
            "merged": merged,
            "pruned": merged,  # Merged memories are effectively pruned
            "similarity_threshold": config.similarity_threshold
        }
    
    async def _importance_consolidation(self, config: ConsolidationConfig) -> Dict[str, Any]:
        """
        Consolidate based on memory importance
        """
        consolidated = 0
        pruned = 0
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Consolidate critical and high importance memories
                consolidate_query = """
                    UPDATE agent_memories
                    SET is_consolidated = TRUE,
                        consolidation_date = CURRENT_TIMESTAMP,
                        memory_type = 'consolidated'
                    WHERE agent_id = %s
                      AND is_active = TRUE
                      AND is_consolidated = FALSE
                      AND importance IN ('critical', 'high')
                      AND strength >= %s
                """
                cur.execute(consolidate_query, (config.agent_id, config.strength_threshold))
                consolidated = cur.rowcount
                
                # Prune low importance weak memories
                if config.enable_pruning:
                    prune_query = """
                        UPDATE agent_memories
                        SET is_active = FALSE
                        WHERE agent_id = %s
                          AND importance IN ('temporary', 'low')
                          AND strength < 0.3
                          AND is_consolidated = FALSE
                    """
                    cur.execute(prune_query, (config.agent_id,))
                    pruned = cur.rowcount
                
                conn.commit()
                
        finally:
            self.db_pool.putconn(conn)
        
        return {
            "consolidated": consolidated,
            "pruned": pruned,
            "importance_based": True
        }
    
    async def _rehearsal_consolidation(self, config: ConsolidationConfig) -> Dict[str, Any]:
        """
        Consolidate frequently accessed memories (rehearsal strengthens memory)
        """
        consolidated = 0
        strengthened = 0
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Find frequently accessed memories
                query = """
                    SELECT id, access_count, strength
                    FROM agent_memories
                    WHERE agent_id = %s
                      AND is_active = TRUE
                      AND is_consolidated = FALSE
                      AND access_count > 5
                    ORDER BY access_count DESC
                    LIMIT %s
                """
                
                cur.execute(query, (config.agent_id, config.batch_size))
                frequent_memories = cur.fetchall()
                
                for memory in frequent_memories:
                    # Strengthen based on access frequency
                    strength_boost = min(0.05 * np.log(memory['access_count'] + 1), 0.3)
                    new_strength = min(1.0, memory['strength'] + strength_boost)
                    
                    # Consolidate if strong enough
                    if new_strength >= config.strength_threshold:
                        update_query = """
                            UPDATE agent_memories
                            SET strength = %s,
                                is_consolidated = TRUE,
                                consolidation_date = CURRENT_TIMESTAMP,
                                memory_type = 'consolidated'
                            WHERE id = %s
                        """
                        cur.execute(update_query, (new_strength, memory['id']))
                        consolidated += 1
                    else:
                        # Just strengthen without consolidating
                        update_query = """
                            UPDATE agent_memories
                            SET strength = %s
                            WHERE id = %s
                        """
                        cur.execute(update_query, (new_strength, memory['id']))
                        strengthened += 1
                
                conn.commit()
                
        finally:
            self.db_pool.putconn(conn)
        
        return {
            "consolidated": consolidated,
            "strengthened": strengthened,
            "rehearsal_based": True
        }
    
    async def _abstraction_consolidation(self, config: ConsolidationConfig) -> Dict[str, Any]:
        """
        Create abstract concepts from concrete memories
        """
        abstractions_created = 0
        
        if not config.enable_abstraction:
            return {"abstractions_created": 0}
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Get memories by category/type for abstraction
                query = """
                    SELECT memory_type, tags, COUNT(*) as count,
                           array_agg(id) as memory_ids,
                           array_agg(content) as contents
                    FROM agent_memories
                    WHERE agent_id = %s
                      AND is_active = TRUE
                      AND is_consolidated = FALSE
                    GROUP BY memory_type, tags
                    HAVING COUNT(*) > 3
                """
                
                cur.execute(query, (config.agent_id,))
                groups = cur.fetchall()
                
                for group in groups:
                    if group['count'] < 5:
                        continue
                    
                    # Create abstract memory from group
                    abstract_content = self._create_abstraction(group['contents'])
                    
                    # Generate embedding for abstract memory
                    abstract_embedding = self._generate_abstract_embedding(group['memory_ids'])
                    
                    # Insert abstract memory
                    insert_query = """
                        INSERT INTO agent_memories
                        (agent_id, memory_type, importance, content, embedding, 
                         tags, is_consolidated, strength)
                        VALUES (%s, 'consolidated', 'high', %s, %s, %s, TRUE, 0.9)
                        RETURNING id
                    """
                    
                    tags = group['tags'] or []
                    tags.append('abstraction')
                    
                    cur.execute(insert_query, (
                        config.agent_id,
                        json.dumps(abstract_content),
                        abstract_embedding,
                        tags
                    ))
                    
                    abstract_id = cur.fetchone()['id']
                    
                    # Link concrete memories to abstraction
                    for memory_id in group['memory_ids'][:10]:  # Limit associations
                        assoc_query = """
                            INSERT INTO memory_associations 
                            (memory_a_id, memory_b_id, association_type, strength)
                            VALUES (%s, %s, 'abstraction', 0.7)
                            ON CONFLICT DO NOTHING
                        """
                        cur.execute(assoc_query, (abstract_id, memory_id))
                    
                    abstractions_created += 1
                
                conn.commit()
                
        finally:
            self.db_pool.putconn(conn)
        
        return {
            "abstractions_created": abstractions_created,
            "abstraction_enabled": True
        }
    
    async def _threshold_consolidation(self, config: ConsolidationConfig) -> Dict[str, Any]:
        """
        Consolidate when memory count exceeds threshold
        """
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Check current memory count
                count_query = """
                    SELECT COUNT(*) as count
                    FROM agent_memories
                    WHERE agent_id = %s AND is_active = TRUE
                """
                cur.execute(count_query, (config.agent_id,))
                current_count = cur.fetchone()['count']
                
                if current_count <= config.max_memories:
                    return {
                        "consolidated": 0,
                        "pruned": 0,
                        "current_count": current_count,
                        "threshold": config.max_memories
                    }
                
                # Need to reduce memory count
                to_remove = current_count - int(config.max_memories * 0.8)  # Keep 80% of max
                
                # Consolidate strong memories first
                consolidate_query = """
                    UPDATE agent_memories
                    SET is_consolidated = TRUE,
                        consolidation_date = CURRENT_TIMESTAMP,
                        memory_type = 'consolidated'
                    WHERE id IN (
                        SELECT id FROM agent_memories
                        WHERE agent_id = %s
                          AND is_active = TRUE
                          AND is_consolidated = FALSE
                          AND strength >= %s
                        ORDER BY strength DESC
                        LIMIT %s
                    )
                """
                cur.execute(consolidate_query, (
                    config.agent_id,
                    config.strength_threshold,
                    min(to_remove // 2, config.batch_size)
                ))
                consolidated = cur.rowcount
                
                # Prune weak memories
                prune_query = """
                    UPDATE agent_memories
                    SET is_active = FALSE
                    WHERE id IN (
                        SELECT id FROM agent_memories
                        WHERE agent_id = %s
                          AND is_active = TRUE
                          AND is_consolidated = FALSE
                        ORDER BY strength ASC, accessed_at ASC
                        LIMIT %s
                    )
                """
                cur.execute(prune_query, (
                    config.agent_id,
                    to_remove - consolidated
                ))
                pruned = cur.rowcount
                
                conn.commit()
                
        finally:
            self.db_pool.putconn(conn)
        
        return {
            "consolidated": consolidated,
            "pruned": pruned,
            "threshold_exceeded": True,
            "original_count": current_count
        }
    
    def _merge_memory_content(self, content_a: Dict, content_b: Dict) -> Dict:
        """
        Merge two memory contents intelligently
        """
        merged = {}
        
        # Combine all keys
        all_keys = set(content_a.keys()) | set(content_b.keys())
        
        for key in all_keys:
            val_a = content_a.get(key)
            val_b = content_b.get(key)
            
            if val_a is None:
                merged[key] = val_b
            elif val_b is None:
                merged[key] = val_a
            elif isinstance(val_a, list) and isinstance(val_b, list):
                # Combine lists, remove duplicates
                merged[key] = list(set(val_a + val_b))
            elif isinstance(val_a, dict) and isinstance(val_b, dict):
                # Recursively merge dicts
                merged[key] = self._merge_memory_content(val_a, val_b)
            elif val_a == val_b:
                merged[key] = val_a
            else:
                # Keep both values
                merged[key] = {"value_a": val_a, "value_b": val_b}
        
        return merged
    
    def _create_abstraction(self, contents: List[Dict]) -> Dict:
        """
        Create abstract concept from concrete memories
        """
        abstraction = {
            "type": "abstraction",
            "source_count": len(contents),
            "created_at": datetime.now().isoformat(),
            "patterns": [],
            "common_elements": {},
            "summary": ""
        }
        
        # Find common patterns
        if contents:
            # Count frequency of keys and values
            key_counts = {}
            value_counts = {}
            
            for content in contents:
                if isinstance(content, dict):
                    for key, value in content.items():
                        key_counts[key] = key_counts.get(key, 0) + 1
                        if isinstance(value, str):
                            value_counts[value] = value_counts.get(value, 0) + 1
            
            # Common elements appear in >50% of memories
            threshold = len(contents) * 0.5
            abstraction["common_elements"] = {
                k: v for k, v in key_counts.items() if v > threshold
            }
            
            # Create summary
            abstraction["summary"] = f"Abstract concept from {len(contents)} memories"
        
        return abstraction
    
    def _generate_abstract_embedding(self, memory_ids: List[str]) -> List[float]:
        """
        Generate embedding for abstract memory by averaging concrete embeddings
        """
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Get embeddings of concrete memories
                query = """
                    SELECT embedding
                    FROM agent_memories
                    WHERE id = ANY(%s) AND embedding IS NOT NULL
                """
                cur.execute(query, (memory_ids,))
                embeddings = [r['embedding'] for r in cur.fetchall()]
                
                if embeddings:
                    # Average embeddings
                    avg_embedding = np.mean(embeddings, axis=0)
                    # Normalize
                    norm = np.linalg.norm(avg_embedding)
                    if norm > 0:
                        avg_embedding = avg_embedding / norm
                    return avg_embedding.tolist()
                else:
                    # Random embedding as fallback
                    return np.random.randn(1536).tolist()
                    
        finally:
            self.db_pool.putconn(conn)
    
    async def _log_consolidation(self, config: ConsolidationConfig, result: Dict[str, Any]):
        """
        Log consolidation event
        """
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                query = """
                    INSERT INTO memory_consolidations
                    (agent_id, consolidation_type, memories_processed, 
                     memories_consolidated, memories_pruned, completed_at, duration_ms, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cur.execute(query, (
                    config.agent_id,
                    config.consolidation_type.value,
                    result.get('total_processed', 0),
                    result.get('consolidated', 0) + result.get('abstractions_created', 0),
                    result.get('pruned', 0),
                    datetime.now(),
                    int(result.get('duration_seconds', 0) * 1000),
                    json.dumps(result)
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log consolidation: {e}")
        finally:
            self.db_pool.putconn(conn)
    
    async def schedule_periodic_consolidation(
        self, 
        agent_id: str,
        interval_hours: int = 24
    ):
        """
        Schedule periodic consolidation for an agent
        """
        while True:
            await asyncio.sleep(interval_hours * 3600)
            
            config = ConsolidationConfig(
                agent_id=agent_id,
                consolidation_type=ConsolidationType.SLEEP_BASED
            )
            
            try:
                result = await self.consolidate(config)
                logger.info(f"Periodic consolidation for {agent_id}: {result}")
            except Exception as e:
                logger.error(f"Periodic consolidation failed for {agent_id}: {e}")