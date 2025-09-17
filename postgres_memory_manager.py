"""
PostgreSQL Memory Manager for Agent Lightning
Implements persistent episodic, semantic, procedural, and working memory
"""

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, Json
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading
import logging
from dataclasses import dataclass, asdict
import uuid

@dataclass
class MemoryItem:
    id: str
    memory_type: str  # episodic, semantic, procedural, working
    agent_id: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    relevance_score: float
    tags: List[str]
    embedding_vector: Optional[List[float]] = None

class PostgreSQLMemoryManager:
    """
    Advanced PostgreSQL-based memory management system
    Implements all four memory types from LEARNING_CAPABILITIES.md
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 database: str = "agent_lightning_memory", 
                 user: str = None,  # Use current user for Homebrew PostgreSQL
                 password: str = None,  # No password needed for local connection
                 port: int = 5432,
                 min_connections: int = 5,
                 max_connections: int = 20):
        
        self.connection_pool = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Connection parameters for Homebrew PostgreSQL
        import os
        self.db_params = {
            'host': host,
            'database': database,
            'user': user or os.getenv('USER'),  # Use current system user
            'port': port
        }
        # Only add password if provided
        if password:
            self.db_params['password'] = password
        
        try:
            # Initialize connection pool and database
            self._initialize_database()
            self._create_connection_pool(min_connections, max_connections)
            self._create_tables()
            self.logger.info("PostgreSQL Memory Manager fully initialized")
        except Exception as e:
            self.logger.error(f"PostgreSQL initialization failed: {e}")
            raise ConnectionError(f"PostgreSQL connection failed: {e}")
        
        # Memory statistics cache
        self._stats_cache = {}
        self._cache_expiry = datetime.now()
    
    def _initialize_database(self):
        """Create database if it doesn't exist"""
        try:
            # Test connection to target database first
            test_conn = psycopg2.connect(**self.db_params)
            test_conn.close()
            self.logger.info(f"Connected to existing database: {self.db_params['database']}")
        except psycopg2.OperationalError:
            # Database doesn't exist, create it
            try:
                temp_params = self.db_params.copy()
                temp_params['database'] = 'postgres'
                
                conn = psycopg2.connect(**temp_params)
                conn.autocommit = True
                
                with conn.cursor() as cur:
                    cur.execute(f"CREATE DATABASE {self.db_params['database']}")
                    self.logger.info(f"Created database: {self.db_params['database']}")
                
                conn.close()
            except Exception as e:
                self.logger.error(f"Database creation failed: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_connection_pool(self, min_conn: int, max_conn: int):
        """Create threaded connection pool"""
        try:
            self.connection_pool = ThreadedConnectionPool(
                min_conn, max_conn, **self.db_params
            )
            self.logger.info(f"Connection pool created: {min_conn}-{max_conn} connections")
        except Exception as e:
            self.logger.error(f"Connection pool creation failed: {e}")
            raise
    
    def _get_connection(self):
        """Get connection from pool"""
        return self.connection_pool.getconn()
    
    def _put_connection(self, conn):
        """Return connection to pool"""
        self.connection_pool.putconn(conn)
    
    def _create_tables(self):
        """Create all memory tables with proper indexes and constraints"""
        conn = self._get_connection()
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                # Main memory table with JSONB for efficient querying
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_memory (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        memory_type VARCHAR(20) NOT NULL CHECK (memory_type IN ('episodic', 'semantic', 'procedural', 'working')),
                        agent_id VARCHAR(100) NOT NULL,
                        content JSONB NOT NULL,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        relevance_score FLOAT DEFAULT 0.0,
                        tags TEXT[] DEFAULT '{}',
                        embedding_vector FLOAT[] DEFAULT NULL,
                        expires_at TIMESTAMP WITH TIME ZONE DEFAULT NULL
                    )
                """)
                
                # Task history table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS task_history (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        agent_id VARCHAR(100) NOT NULL,
                        task_description TEXT NOT NULL,
                        task_type VARCHAR(50),
                        result JSONB,
                        status VARCHAR(20) DEFAULT 'completed',
                        execution_time FLOAT,
                        tokens_used INTEGER,
                        cost FLOAT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{}'
                    )
                """)
                
                # Agent learning sessions
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS learning_sessions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        agent_id VARCHAR(100) NOT NULL,
                        session_type VARCHAR(50) NOT NULL,
                        knowledge_items_processed INTEGER DEFAULT 0,
                        knowledge_integrated INTEGER DEFAULT 0,
                        performance_improvement FLOAT DEFAULT 0.0,
                        session_data JSONB DEFAULT '{}',
                        started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP WITH TIME ZONE,
                        status VARCHAR(20) DEFAULT 'active'
                    )
                """)
                
                # Cross-agent knowledge transfer
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_transfer (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        source_agent_id VARCHAR(100) NOT NULL,
                        target_agent_id VARCHAR(100) NOT NULL,
                        knowledge_type VARCHAR(50) NOT NULL,
                        transfer_data JSONB NOT NULL,
                        success_rate FLOAT DEFAULT 0.0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        applied_at TIMESTAMP WITH TIME ZONE
                    )
                """)
                
                # Performance metrics
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        agent_id VARCHAR(100) NOT NULL,
                        metric_type VARCHAR(50) NOT NULL,
                        metric_value FLOAT NOT NULL,
                        context JSONB DEFAULT '{}',
                        recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_memory_agent_type ON agent_memory(agent_id, memory_type)",
                    "CREATE INDEX IF NOT EXISTS idx_memory_created ON agent_memory(created_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_memory_accessed ON agent_memory(last_accessed DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_memory_relevance ON agent_memory(relevance_score DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_memory_tags ON agent_memory USING GIN(tags)",
                    "CREATE INDEX IF NOT EXISTS idx_memory_content ON agent_memory USING GIN(content)",
                    "CREATE INDEX IF NOT EXISTS idx_task_history_agent ON task_history(agent_id, created_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_learning_sessions_agent ON learning_sessions(agent_id, started_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_knowledge_transfer_agents ON knowledge_transfer(source_agent_id, target_agent_id)",
                    "CREATE INDEX IF NOT EXISTS idx_performance_metrics ON performance_metrics(agent_id, metric_type, recorded_at DESC)"
                ]
                
                for index_sql in indexes:
                    cur.execute(index_sql)
                
                # Enable vector extension if available (for embeddings)
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_embedding ON agent_memory USING ivfflat (embedding_vector vector_cosine_ops)")
                except:
                    pass  # Vector extension not available
                
                conn.commit()
                self.logger.info("Database tables and indexes created successfully")
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Table creation failed: {e}")
            raise
        finally:
            self._put_connection(conn)
    
    # Episodic Memory Methods
    def store_episodic_memory(self, agent_id: str, conversation_data: Dict[str, Any], 
                            context: Dict[str, Any] = None) -> str:
        """Store conversation/interaction episode"""
        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            memory_type="episodic",
            agent_id=agent_id,
            content={
                "conversation": conversation_data,
                "context": context or {},
                "episode_type": "conversation"
            },
            metadata={
                "participants": conversation_data.get("participants", []),
                "duration": conversation_data.get("duration"),
                "outcome": conversation_data.get("outcome")
            },
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            relevance_score=1.0,
            tags=conversation_data.get("tags", [])
        )
        
        return self._store_memory_item(memory_item)
    
    def get_episodic_memories(self, agent_id: str, limit: int = 50, 
                            time_range: Tuple[datetime, datetime] = None) -> List[MemoryItem]:
        """Retrieve episodic memories (conversation history)"""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT * FROM agent_memory 
                    WHERE agent_id = %s AND memory_type = 'episodic'
                """
                params = [agent_id]
                
                if time_range:
                    query += " AND created_at BETWEEN %s AND %s"
                    params.extend(time_range)
                
                query += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)
                
                cur.execute(query, params)
                return [self._row_to_memory_item(row) for row in cur.fetchall()]
                
        finally:
            self._put_connection(conn)
    
    # Semantic Memory Methods
    def store_semantic_knowledge(self, agent_id: str, knowledge_data: Dict[str, Any], 
                               category: str, source: str = "learned") -> str:
        """Store factual knowledge in semantic memory"""
        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            memory_type="semantic",
            agent_id=agent_id,
            content={
                "knowledge": knowledge_data,
                "category": category,
                "source": source,
                "confidence": knowledge_data.get("confidence", 0.8)
            },
            metadata={
                "category": category,
                "source": source,
                "verified": knowledge_data.get("verified", False),
                "references": knowledge_data.get("references", [])
            },
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            relevance_score=knowledge_data.get("importance", 0.5),
            tags=knowledge_data.get("tags", [category])
        )
        
        return self._store_memory_item(memory_item)
    
    def search_semantic_knowledge(self, agent_id: str, query: str, 
                                category: str = None, limit: int = 10) -> List[MemoryItem]:
        """Search semantic knowledge base"""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                search_query = """
                    SELECT *, ts_rank(to_tsvector('english', content::text), plainto_tsquery('english', %s)) as rank
                    FROM agent_memory 
                    WHERE agent_id = %s AND memory_type = 'semantic'
                    AND to_tsvector('english', content::text) @@ plainto_tsquery('english', %s)
                """
                params = [query, agent_id, query]
                
                if category:
                    search_query += " AND metadata->>'category' = %s"
                    params.append(category)
                
                search_query += " ORDER BY rank DESC, relevance_score DESC LIMIT %s"
                params.append(limit)
                
                cur.execute(search_query, params)
                results = [self._row_to_memory_item(row) for row in cur.fetchall()]
                
                # Update access counts
                for item in results:
                    self._update_access_count(item.id)
                
                return results
                
        finally:
            self._put_connection(conn)
    
    # Procedural Memory Methods
    def store_procedural_skill(self, agent_id: str, skill_data: Dict[str, Any], 
                             skill_type: str) -> str:
        """Store learned skills and patterns"""
        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            memory_type="procedural",
            agent_id=agent_id,
            content={
                "skill": skill_data,
                "skill_type": skill_type,
                "proficiency": skill_data.get("proficiency", 0.5),
                "usage_patterns": skill_data.get("usage_patterns", []),
                "success_rate": skill_data.get("success_rate", 0.0)
            },
            metadata={
                "skill_type": skill_type,
                "learned_from": skill_data.get("learned_from", "experience"),
                "complexity": skill_data.get("complexity", "medium"),
                "prerequisites": skill_data.get("prerequisites", [])
            },
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            relevance_score=skill_data.get("proficiency", 0.5),
            tags=skill_data.get("tags", [skill_type])
        )
        
        return self._store_memory_item(memory_item)
    
    def get_procedural_skills(self, agent_id: str, skill_type: str = None) -> List[MemoryItem]:
        """Get learned skills and patterns"""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT * FROM agent_memory 
                    WHERE agent_id = %s AND memory_type = 'procedural'
                """
                params = [agent_id]
                
                if skill_type:
                    query += " AND metadata->>'skill_type' = %s"
                    params.append(skill_type)
                
                query += " ORDER BY relevance_score DESC, access_count DESC"
                
                cur.execute(query, params)
                return [self._row_to_memory_item(row) for row in cur.fetchall()]
                
        finally:
            self._put_connection(conn)
    
    # Working Memory Methods
    def store_working_memory(self, agent_id: str, context_data: Dict[str, Any], 
                           ttl_minutes: int = 60) -> str:
        """Store active context (expires automatically)"""
        expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
        
        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            memory_type="working",
            agent_id=agent_id,
            content={
                "context": context_data,
                "active_tasks": context_data.get("active_tasks", []),
                "current_focus": context_data.get("current_focus"),
                "temporary_data": context_data.get("temporary_data", {})
            },
            metadata={
                "ttl_minutes": ttl_minutes,
                "expires_at": expires_at.isoformat(),
                "priority": context_data.get("priority", "normal")
            },
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            relevance_score=1.0,
            tags=context_data.get("tags", ["working"])
        )
        
        return self._store_memory_item(memory_item, expires_at=expires_at)
    
    def get_working_memory(self, agent_id: str) -> List[MemoryItem]:
        """Get active working memory (non-expired)"""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM agent_memory 
                    WHERE agent_id = %s AND memory_type = 'working'
                    AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    ORDER BY created_at DESC
                """, (agent_id,))
                
                return [self._row_to_memory_item(row) for row in cur.fetchall()]
                
        finally:
            self._put_connection(conn)
    
    def clear_working_memory(self, agent_id: str):
        """Clear all working memory for agent"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM agent_memory 
                    WHERE agent_id = %s AND memory_type = 'working'
                """, (agent_id,))
                conn.commit()
                
        finally:
            self._put_connection(conn)
    
    # Task History Methods
    def store_task_result(self, agent_id: str, task_description: str, result: Dict[str, Any],
                         execution_time: float = None, tokens_used: int = None, cost: float = None) -> str:
        """Store task execution result"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO task_history 
                    (agent_id, task_description, result, execution_time, tokens_used, cost, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    agent_id, task_description, Json(result), 
                    execution_time, tokens_used, cost, Json({})
                ))
                
                task_id = cur.fetchone()[0]
                conn.commit()
                
                # Also store as episodic memory
                self.store_episodic_memory(agent_id, {
                    "type": "task_execution",
                    "task": task_description,
                    "result": result,
                    "execution_time": execution_time
                })
                
                return str(task_id)
                
        finally:
            self._put_connection(conn)
    
    def get_task_history(self, agent_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get task execution history"""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if agent_id:
                    cur.execute("""
                        SELECT * FROM task_history 
                        WHERE agent_id = %s 
                        ORDER BY created_at DESC 
                        LIMIT %s
                    """, (agent_id, limit))
                else:
                    cur.execute("""
                        SELECT * FROM task_history 
                        ORDER BY created_at DESC 
                        LIMIT %s
                    """, (limit,))
                
                return [dict(row) for row in cur.fetchall()]
                
        finally:
            self._put_connection(conn)
    
    # Learning and Analytics Methods
    def record_learning_session(self, agent_id: str, session_type: str, 
                              knowledge_processed: int, knowledge_integrated: int,
                              performance_improvement: float = 0.0) -> str:
        """Record a learning session"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO learning_sessions 
                    (agent_id, session_type, knowledge_items_processed, 
                     knowledge_integrated, performance_improvement, completed_at, status)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, 'completed')
                    RETURNING id
                """, (agent_id, session_type, knowledge_processed, 
                      knowledge_integrated, performance_improvement))
                
                session_id = cur.fetchone()[0]
                conn.commit()
                return str(session_id)
                
        finally:
            self._put_connection(conn)
    
    def get_memory_statistics(self, agent_id: str = None) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        # Check cache first
        cache_key = f"stats_{agent_id or 'all'}"
        if (cache_key in self._stats_cache and 
            datetime.now() < self._cache_expiry):
            return self._stats_cache[cache_key]
        
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Memory counts by type
                if agent_id:
                    cur.execute("""
                        SELECT memory_type, COUNT(*) as count, 
                               AVG(relevance_score) as avg_relevance,
                               SUM(access_count) as total_accesses
                        FROM agent_memory 
                        WHERE agent_id = %s 
                        GROUP BY memory_type
                    """, (agent_id,))
                else:
                    cur.execute("""
                        SELECT memory_type, COUNT(*) as count,
                               AVG(relevance_score) as avg_relevance,
                               SUM(access_count) as total_accesses
                        FROM agent_memory 
                        GROUP BY memory_type
                    """)
                
                memory_stats = {row['memory_type']: dict(row) for row in cur.fetchall()}
                
                # Task statistics
                if agent_id:
                    cur.execute("""
                        SELECT COUNT(*) as total_tasks,
                               AVG(execution_time) as avg_execution_time,
                               SUM(tokens_used) as total_tokens,
                               SUM(cost) as total_cost
                        FROM task_history 
                        WHERE agent_id = %s
                    """, (agent_id,))
                else:
                    cur.execute("""
                        SELECT COUNT(*) as total_tasks,
                               AVG(execution_time) as avg_execution_time,
                               SUM(tokens_used) as total_tokens,
                               SUM(cost) as total_cost
                        FROM task_history
                    """)
                
                task_stats = dict(cur.fetchone())
                
                # Learning statistics
                if agent_id:
                    cur.execute("""
                        SELECT COUNT(*) as learning_sessions,
                               SUM(knowledge_items_processed) as total_knowledge_processed,
                               SUM(knowledge_integrated) as total_knowledge_integrated,
                               AVG(performance_improvement) as avg_improvement
                        FROM learning_sessions 
                        WHERE agent_id = %s
                    """, (agent_id,))
                else:
                    cur.execute("""
                        SELECT COUNT(*) as learning_sessions,
                               SUM(knowledge_items_processed) as total_knowledge_processed,
                               SUM(knowledge_integrated) as total_knowledge_integrated,
                               AVG(performance_improvement) as avg_improvement
                        FROM learning_sessions
                    """)
                
                learning_stats = dict(cur.fetchone())
                
                stats = {
                    "memory_by_type": memory_stats,
                    "task_statistics": task_stats,
                    "learning_statistics": learning_stats,
                    "generated_at": datetime.now().isoformat()
                }
                
                # Cache for 5 minutes
                self._stats_cache[cache_key] = stats
                self._cache_expiry = datetime.now() + timedelta(minutes=5)
                
                return stats
                
        finally:
            self._put_connection(conn)
    
    # Utility Methods
    def _store_memory_item(self, item: MemoryItem, expires_at: datetime = None) -> str:
        """Store a memory item in the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO agent_memory 
                    (id, memory_type, agent_id, content, metadata, created_at, 
                     last_accessed, access_count, relevance_score, tags, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    item.id, item.memory_type, item.agent_id,
                    Json(item.content), Json(item.metadata),
                    item.created_at, item.last_accessed,
                    item.access_count, item.relevance_score,
                    item.tags, expires_at
                ))
                conn.commit()
                return item.id
                
        finally:
            self._put_connection(conn)
    
    def _row_to_memory_item(self, row: Dict) -> MemoryItem:
        """Convert database row to MemoryItem"""
        return MemoryItem(
            id=str(row['id']),
            memory_type=row['memory_type'],
            agent_id=row['agent_id'],
            content=row['content'],
            metadata=row['metadata'],
            created_at=row['created_at'],
            last_accessed=row['last_accessed'],
            access_count=row['access_count'],
            relevance_score=row['relevance_score'],
            tags=row['tags'] or [],
            embedding_vector=row.get('embedding_vector')
        )
    
    def _update_access_count(self, memory_id: str):
        """Update access count and last accessed time"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE agent_memory 
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (memory_id,))
                conn.commit()
                
        finally:
            self._put_connection(conn)
    
    def cleanup_expired_memory(self):
        """Clean up expired working memory"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM agent_memory 
                    WHERE expires_at IS NOT NULL 
                    AND expires_at < CURRENT_TIMESTAMP
                """)
                deleted_count = cur.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} expired memory items")
                
        finally:
            self._put_connection(conn)
    
    def close(self):
        """Close connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("Connection pool closed")