"""
Enterprise Database Manager
Production-grade database operations with connection pooling and monitoring
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import asyncpg
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Enterprise database manager with connection pooling"""
    
    def __init__(self):
        self.pool = None
        self.connection_string = os.getenv(
            'DATABASE_URL', 
            'postgresql://agent_user:agent_pass@postgres:5432/agent_lightning'
        )
        
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
            await self._ensure_tables()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _ensure_tables(self):
        """Ensure required tables exist"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    status VARCHAR(50) NOT NULL,
                    created_by VARCHAR(255),
                    created_at TIMESTAMP DEFAULT NOW(),
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    progress FLOAT DEFAULT 0.0,
                    metadata JSONB DEFAULT '{}'
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_tasks (
                    id VARCHAR(255) PRIMARY KEY,
                    workflow_id VARCHAR(255) REFERENCES workflows(id),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    status VARCHAR(50) NOT NULL,
                    assigned_agent VARCHAR(255),
                    result TEXT,
                    error_message TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    retry_attempts INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}'
                )
            """)
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        async with self.pool.acquire() as conn:
            yield conn
    
    async def save_workflow(self, workflow_data: Dict[str, Any]):
        """Save workflow to database"""
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO workflows (id, name, description, status, created_by, created_at, started_at, completed_at, progress, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    progress = EXCLUDED.progress,
                    metadata = EXCLUDED.metadata
            """, 
                workflow_data['workflow_id'],
                workflow_data['name'],
                workflow_data['description'],
                workflow_data['status'],
                workflow_data['created_by'],
                workflow_data.get('created_at'),
                workflow_data.get('started_at'),
                workflow_data.get('completed_at'),
                workflow_data.get('progress', 0.0),
                json.dumps(workflow_data.get('metadata', {}))
            )
    
    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow by ID"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM workflows WHERE id = $1", workflow_id
            )
            return dict(row) if row else None
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()


# Global database manager instance
db_manager = DatabaseManager()