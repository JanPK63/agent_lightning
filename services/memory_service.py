#!/usr/bin/env python3
"""
Memory Service - Enterprise-grade memory persistence for AI agents
Implements episodic, semantic, and procedural memory with vector similarity search
"""

import os
import sys
import json
import asyncio
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass, field, asdict
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiohttp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

# Import memory modules
from memory_retrieval import MemoryRetriever, RetrievalContext, RetrievalStrategy
from memory_consolidation import MemoryConsolidator, ConsolidationConfig, ConsolidationType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory"""
    EPISODIC = "episodic"        # Specific experiences/events
    SEMANTIC = "semantic"        # General knowledge/facts
    PROCEDURAL = "procedural"    # How to do things
    WORKING = "working"          # Short-term active memory
    CONSOLIDATED = "consolidated" # Long-term consolidated memories


class MemoryImportance(str, Enum):
    """Memory importance levels"""
    CRITICAL = "critical"      # Must never forget
    HIGH = "high"              # Very important
    MEDIUM = "medium"          # Standard importance
    LOW = "low"                # Can be forgotten if needed
    TEMPORARY = "temporary"    # Will be deleted after consolidation


class MemoryCreate(BaseModel):
    """Memory creation model"""
    agent_id: str
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: MemoryImportance = MemoryImportance.MEDIUM
    content: Dict[str, Any]
    embedding: Optional[List[float]] = None
    context: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    tags: Optional[List[str]] = []
    parent_memory_id: Optional[str] = None
    related_memories: Optional[List[str]] = []


class MemoryQuery(BaseModel):
    """Memory query model"""
    agent_id: str
    query_embedding: Optional[List[float]] = None
    memory_types: Optional[List[MemoryType]] = None
    tags: Optional[List[str]] = None
    min_importance: Optional[MemoryImportance] = None
    min_strength: Optional[float] = 0.0
    limit: int = 10
    include_inactive: bool = False


class ExperienceReplay(BaseModel):
    """Experience replay buffer entry"""
    agent_id: str
    memory_id: Optional[str] = None
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    done: bool = False
    priority: float = 1.0


class MemoryService:
    """Main Memory Service for agent learning persistence"""
    
    def __init__(self):
        self.app = FastAPI(title="Memory Service", version="1.0.0")
        
        # Initialize Data Access Layer for standard operations
        self.dal = DataAccessLayer("memory_service")
        
        # Direct PostgreSQL connection for memory-specific operations
        self.db_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", 5432),
            database=os.getenv("POSTGRES_DB", "agent_lightning"),
            user=os.getenv("POSTGRES_USER", "agent_user"),
            password=os.getenv("POSTGRES_PASSWORD", "agent_password")
        )
        
        # Cache for frequently accessed memories
        self.cache = get_cache()
        
        # Initialize memory retriever and consolidator
        self.retriever = MemoryRetriever(self.db_pool, self.cache)
        self.consolidator = MemoryConsolidator(self.db_pool, self.cache, self.retriever)
        
        # Service URLs
        self.embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8015")
        
        logger.info("✅ Memory Service initialized with pgvector support and advanced retrieval/consolidation")
        
        self._setup_middleware()
        self._setup_routes()
        # Background tasks will be started on app startup
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a database query and return results"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                if cur.description:  # Query returns results
                    result = cur.fetchall()
                    conn.commit()
                    return result
                conn.commit()
                return []
        finally:
            self.db_pool.putconn(conn)
        
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            health_status = self.dal.health_check()
            
            # Check if pgvector is available
            vector_check = self.execute_query(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            )
            
            return {
                "service": "memory_service",
                "status": "healthy",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "pgvector": bool(vector_check),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/memories")
        async def create_memory(memory: MemoryCreate):
            """Create a new memory for an agent"""
            try:
                # Generate embedding if not provided
                if not memory.embedding and memory.content:
                    memory.embedding = await self._generate_embedding(
                        json.dumps(memory.content)
                    )
                
                # Create memory in database
                memory_id = str(uuid.uuid4())
                
                query = """
                    INSERT INTO agent_memories 
                    (id, agent_id, memory_type, importance, content, embedding, 
                     context, source, tags, parent_memory_id, related_memories)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                """
                
                result = self.execute_query(
                    query,
                    (
                        memory_id,
                        memory.agent_id,
                        memory.memory_type.value,
                        memory.importance.value,
                        json.dumps(memory.content),
                        memory.embedding,
                        json.dumps(memory.context) if memory.context else None,
                        memory.source,
                        memory.tags,
                        memory.parent_memory_id,
                        memory.related_memories
                    )
                )
                
                if result:
                    # Cache the memory
                    self.cache.set(
                        f"memory:{memory_id}",
                        result[0],
                        ttl=3600
                    )
                    
                    logger.info(f"Created memory {memory_id} for agent {memory.agent_id}")
                    return {"memory_id": memory_id, "status": "created"}
                
            except Exception as e:
                logger.error(f"Failed to create memory: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/memories/{memory_id}")
        async def get_memory(memory_id: str):
            """Get a specific memory"""
            try:
                # Check cache first
                cached = self.cache.get(f"memory:{memory_id}")
                if cached:
                    return cached
                
                # Query database
                query = "SELECT * FROM agent_memories WHERE id = %s"
                result = self.execute_query(query, (memory_id,))
                
                if result:
                    # Update access count and timestamp
                    update_query = """
                        UPDATE agent_memories 
                        SET accessed_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """
                    self.execute_query(update_query, (memory_id,))
                    
                    return result[0]
                else:
                    raise HTTPException(status_code=404, detail="Memory not found")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get memory: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/memories/query")
        async def query_memories(query: MemoryQuery):
            """Query memories with vector similarity search"""
            try:
                conditions = ["agent_id = %s"]
                params = [query.agent_id]
                
                # Add filters
                if query.memory_types:
                    type_placeholders = ", ".join(["%s"] * len(query.memory_types))
                    conditions.append(f"memory_type IN ({type_placeholders})")
                    params.extend([t.value for t in query.memory_types])
                
                if query.tags:
                    conditions.append("tags && %s")  # Array overlap operator
                    params.append(query.tags)
                
                if query.min_importance:
                    importance_values = {
                        MemoryImportance.TEMPORARY: 0,
                        MemoryImportance.LOW: 1,
                        MemoryImportance.MEDIUM: 2,
                        MemoryImportance.HIGH: 3,
                        MemoryImportance.CRITICAL: 4
                    }
                    conditions.append(
                        f"importance::text IN {tuple(k.value for k, v in importance_values.items() if v >= importance_values[query.min_importance])}"
                    )
                
                if query.min_strength:
                    conditions.append("strength >= %s")
                    params.append(query.min_strength)
                
                if not query.include_inactive:
                    conditions.append("is_active = TRUE")
                
                # Build query
                if query.query_embedding:
                    # Vector similarity search
                    sql = f"""
                        SELECT *, 
                               1 - (embedding <=> %s::vector) as similarity
                        FROM agent_memories
                        WHERE {' AND '.join(conditions)}
                        ORDER BY similarity DESC
                        LIMIT %s
                    """
                    params = [query.query_embedding] + params + [query.limit]
                else:
                    # Regular query
                    sql = f"""
                        SELECT *
                        FROM agent_memories
                        WHERE {' AND '.join(conditions)}
                        ORDER BY strength DESC, created_at DESC
                        LIMIT %s
                    """
                    params.append(query.limit)
                
                results = self.execute_query(sql, params)
                
                return {
                    "memories": results,
                    "count": len(results)
                }
                
            except Exception as e:
                logger.error(f"Failed to query memories: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/memories/{agent_id}/consolidate")
        async def consolidate_memories(
            agent_id: str,
            background_tasks: BackgroundTasks,
            threshold: float = 0.7
        ):
            """Consolidate agent memories"""
            try:
                # Run consolidation in background
                background_tasks.add_task(
                    self._consolidate_agent_memories,
                    agent_id,
                    threshold
                )
                
                return {
                    "status": "consolidation_started",
                    "agent_id": agent_id,
                    "threshold": threshold
                }
                
            except Exception as e:
                logger.error(f"Failed to start consolidation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/experience-replay")
        async def add_experience(experience: ExperienceReplay):
            """Add experience to replay buffer"""
            try:
                exp_id = str(uuid.uuid4())
                
                query = """
                    INSERT INTO experience_replay_buffer
                    (id, agent_id, memory_id, state, action, reward, next_state, done, priority)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """
                
                result = self.execute_query(
                    query,
                    (
                        exp_id,
                        experience.agent_id,
                        experience.memory_id,
                        json.dumps(experience.state),
                        json.dumps(experience.action),
                        experience.reward,
                        json.dumps(experience.next_state),
                        experience.done,
                        experience.priority
                    )
                )
                
                if result:
                    logger.info(f"Added experience {exp_id} for agent {experience.agent_id}")
                    return {"experience_id": exp_id, "status": "added"}
                    
            except Exception as e:
                logger.error(f"Failed to add experience: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/experience-replay/{agent_id}/sample")
        async def sample_experiences(
            agent_id: str,
            batch_size: int = 32,
            prioritized: bool = True
        ):
            """Sample experiences for training"""
            try:
                if prioritized:
                    # Prioritized experience replay
                    query = """
                        SELECT * FROM experience_replay_buffer
                        WHERE agent_id = %s AND is_processed = FALSE
                        ORDER BY priority DESC, RANDOM()
                        LIMIT %s
                    """
                else:
                    # Random sampling
                    query = """
                        SELECT * FROM experience_replay_buffer
                        WHERE agent_id = %s AND is_processed = FALSE
                        ORDER BY RANDOM()
                        LIMIT %s
                    """
                
                experiences = self.execute_query(query, (agent_id, batch_size))
                
                # Mark as sampled
                if experiences:
                    exp_ids = [exp['id'] for exp in experiences]
                    update_query = """
                        UPDATE experience_replay_buffer
                        SET sampling_count = sampling_count + 1
                        WHERE id = ANY(%s)
                    """
                    self.execute_query(update_query, (exp_ids,))
                
                return {
                    "experiences": experiences,
                    "batch_size": len(experiences)
                }
                
            except Exception as e:
                logger.error(f"Failed to sample experiences: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/memories/{agent_id}/decay")
        async def decay_memories(agent_id: str):
            """Apply decay to agent memories"""
            try:
                # Call the decay function
                query = "SELECT decay_memory_strength(%s)"
                self.execute_query(query, (agent_id,))
                
                # Prune very weak memories
                prune_query = """
                    UPDATE agent_memories
                    SET is_active = FALSE
                    WHERE agent_id = %s 
                    AND strength < 0.05 
                    AND memory_type != 'consolidated'
                    AND importance IN ('temporary', 'low')
                """
                pruned = self.execute_query(prune_query, (agent_id,))
                
                return {
                    "status": "decay_applied",
                    "agent_id": agent_id,
                    "memories_pruned": len(pruned) if pruned else 0
                }
                
            except Exception as e:
                logger.error(f"Failed to decay memories: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using embedding service or fallback"""
        try:
            # Try to use embedding service
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.embedding_service_url}/embed",
                    json={"text": text}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["embedding"]
        except:
            pass
        
        # Fallback to random embedding for now
        # In production, use OpenAI, Anthropic, or local embedding model
        logger.warning("Using random embedding as fallback")
        return np.random.randn(1536).tolist()
    
    async def _consolidate_agent_memories(self, agent_id: str, threshold: float):
        """Background task to consolidate memories using advanced consolidator"""
        try:
            # Use the new consolidator with configurable strategy
            config = ConsolidationConfig(
                agent_id=agent_id,
                consolidation_type=ConsolidationType.SLEEP_BASED,
                strength_threshold=threshold,
                enable_pruning=True,
                enable_abstraction=True
            )
            
            result = await self.consolidator.consolidate(config)
            
            logger.info(
                f"Consolidation complete for {agent_id}: "
                f"{result.get('consolidated', 0)} consolidated, "
                f"{result.get('pruned', 0)} pruned, "
                f"{result.get('abstractions_created', 0)} abstractions created"
            )
                
        except Exception as e:
            logger.error(f"Consolidation failed for {agent_id}: {e}")
    
    async def _setup_background_tasks(self):
        """Setup periodic background tasks"""
        async def periodic_decay():
            """Periodically decay all memories"""
            while True:
                await asyncio.sleep(3600)  # Every hour
                try:
                    # Get all active agents
                    query = "SELECT DISTINCT agent_id FROM agent_memories WHERE is_active = TRUE"
                    agents = self.execute_query(query)
                    
                    for agent in agents:
                        await self.decay_memories(agent['agent_id'])
                        
                    logger.info(f"Applied decay to {len(agents)} agents")
                except Exception as e:
                    logger.error(f"Periodic decay failed: {e}")
        
        # Start background tasks
        asyncio.create_task(periodic_decay())
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Memory Service starting up...")
        
        # Verify pgvector is installed
        check = self.execute_query(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        if not check:
            logger.warning("pgvector extension not found - vector search disabled")
        else:
            logger.info("✅ pgvector extension found - vector search enabled")
        
        # Start background tasks
        await self._setup_background_tasks()
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Memory Service shutting down...")
        self.dal.cleanup()
        self.db_pool.closeall()


def main():
    """Main entry point"""
    import uvicorn
    
    service = MemoryService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("MEMORY_SERVICE_PORT", 8012))
    logger.info(f"Starting Memory Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()