#!/usr/bin/env python3
"""
Idempotent State Management Service
Ensures operations can be safely retried without side effects
Tracks operation states and prevents duplicate executions
Critical for enterprise-grade reliability
"""

import os
import sys
import json
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass, asdict
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
import redis

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.cache import get_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OperationState(str, Enum):
    """Operation execution states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"


class OperationType(str, Enum):
    """Types of operations that need idempotency"""
    TASK_ASSIGNMENT = "task_assignment"
    AGENT_EXECUTION = "agent_execution"
    MODEL_TRAINING = "model_training"
    DATA_PROCESSING = "data_processing"
    DEPLOYMENT = "deployment"
    DATABASE_WRITE = "database_write"
    API_CALL = "api_call"
    FILE_OPERATION = "file_operation"


@dataclass
class OperationKey:
    """Unique key for an operation"""
    operation_type: OperationType
    entity_id: str  # Task ID, Agent ID, etc.
    params_hash: str  # Hash of operation parameters
    
    def to_string(self) -> str:
        """Convert to string key"""
        return f"{self.operation_type.value}:{self.entity_id}:{self.params_hash}"


class IdempotencyRequest(BaseModel):
    """Request to register an idempotent operation"""
    operation_type: OperationType
    entity_id: str = Field(description="ID of entity being operated on")
    operation_params: Dict[str, Any] = Field(description="Parameters of the operation")
    ttl_seconds: int = Field(default=3600, description="Time to live in seconds")
    retry_count: int = Field(default=0, description="Current retry attempt")


class IdempotencyCheck(BaseModel):
    """Check if operation can proceed"""
    operation_type: OperationType
    entity_id: str
    operation_params: Dict[str, Any]


class OperationResult(BaseModel):
    """Result of an operation"""
    operation_id: str
    state: OperationState
    can_proceed: bool
    existing_result: Optional[Dict[str, Any]] = None
    message: str
    retry_after: Optional[int] = None  # Seconds to wait before retry


class IdempotentStateManager:
    """Manages idempotent state for all operations"""
    
    def __init__(self):
        self.app = FastAPI(title="Idempotent State Manager", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("idempotent_state")
        self.cache = get_cache()
        
        # Database connection
        self.db_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agent_lightning"),
            "user": os.getenv("POSTGRES_USER", "agent_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "agent_password")
        }
        
        # Operation tracking
        self.operations = {}
        self.lock_timeout = 30  # seconds
        
        # Create database table
        self._create_tables()
        
        logger.info("✅ Idempotent State Manager initialized")
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _create_tables(self):
        """Create idempotency tracking tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Create idempotent_operations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS idempotent_operations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    operation_key VARCHAR(500) UNIQUE NOT NULL,
                    operation_type VARCHAR(50) NOT NULL,
                    entity_id VARCHAR(200) NOT NULL,
                    params_hash VARCHAR(64) NOT NULL,
                    state VARCHAR(20) NOT NULL,
                    result JSONB,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    locked_until TIMESTAMP,
                    locked_by VARCHAR(100)
                );
                
                CREATE INDEX IF NOT EXISTS idx_operation_key ON idempotent_operations(operation_key);
                CREATE INDEX IF NOT EXISTS idx_entity_id ON idempotent_operations(entity_id);
                CREATE INDEX IF NOT EXISTS idx_state ON idempotent_operations(state);
                CREATE INDEX IF NOT EXISTS idx_expires_at ON idempotent_operations(expires_at);
            """)
            
            # Create operation_history table for audit
            cur.execute("""
                CREATE TABLE IF NOT EXISTS operation_history (
                    id SERIAL PRIMARY KEY,
                    operation_id UUID REFERENCES idempotent_operations(id),
                    state_from VARCHAR(20),
                    state_to VARCHAR(20),
                    details JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_operation_history_id ON operation_history(operation_id);
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
    
    def _hash_params(self, params: Dict[str, Any]) -> str:
        """Create deterministic hash of parameters"""
        # Sort keys for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.sha256(sorted_params.encode()).hexdigest()
    
    def _generate_operation_key(self, operation_type: OperationType, 
                               entity_id: str, params: Dict[str, Any]) -> OperationKey:
        """Generate unique operation key"""
        params_hash = self._hash_params(params)
        return OperationKey(operation_type, entity_id, params_hash)
    
    async def check_operation(self, request: IdempotencyCheck) -> OperationResult:
        """Check if operation can proceed (idempotency check)"""
        try:
            # Generate operation key
            op_key = self._generate_operation_key(
                request.operation_type,
                request.entity_id,
                request.operation_params
            )
            key_str = op_key.to_string()
            
            # Check cache first
            cached = self.cache.get(f"idempotent:{key_str}")
            if cached:
                state = OperationState(cached.get("state"))
                
                if state == OperationState.COMPLETED:
                    # Operation already completed, return cached result
                    return OperationResult(
                        operation_id=cached.get("id"),
                        state=state,
                        can_proceed=False,
                        existing_result=cached.get("result"),
                        message="Operation already completed"
                    )
                elif state == OperationState.IN_PROGRESS:
                    # Operation in progress, check lock
                    locked_until = cached.get("locked_until")
                    if locked_until and datetime.fromisoformat(locked_until) > datetime.utcnow():
                        return OperationResult(
                            operation_id=cached.get("id"),
                            state=state,
                            can_proceed=False,
                            message="Operation in progress",
                            retry_after=30
                        )
            
            # Check database
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT id, state, result, locked_until, retry_count, expires_at
                FROM idempotent_operations
                WHERE operation_key = %s
            """, (key_str,))
            
            row = cur.fetchone()
            
            if row:
                state = OperationState(row["state"])
                
                # Check expiration
                if row["expires_at"] and row["expires_at"] < datetime.utcnow():
                    # Operation expired, can retry
                    cur.execute("""
                        UPDATE idempotent_operations
                        SET state = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (OperationState.EXPIRED.value, row["id"]))
                    conn.commit()
                    conn.close()
                    
                    return OperationResult(
                        operation_id=str(row["id"]),
                        state=OperationState.EXPIRED,
                        can_proceed=True,
                        message="Previous operation expired, can retry"
                    )
                
                if state == OperationState.COMPLETED:
                    conn.close()
                    return OperationResult(
                        operation_id=str(row["id"]),
                        state=state,
                        can_proceed=False,
                        existing_result=row["result"],
                        message="Operation already completed"
                    )
                
                elif state == OperationState.IN_PROGRESS:
                    # Check if lock expired
                    if row["locked_until"] and row["locked_until"] < datetime.utcnow():
                        # Lock expired, operation likely failed
                        cur.execute("""
                            UPDATE idempotent_operations
                            SET state = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (OperationState.FAILED.value, row["id"]))
                        conn.commit()
                        conn.close()
                        
                        return OperationResult(
                            operation_id=str(row["id"]),
                            state=OperationState.FAILED,
                            can_proceed=True,
                            message="Previous operation timed out, can retry"
                        )
                    else:
                        conn.close()
                        return OperationResult(
                            operation_id=str(row["id"]),
                            state=state,
                            can_proceed=False,
                            message="Operation in progress",
                            retry_after=30
                        )
                
                elif state == OperationState.FAILED:
                    # Check retry count
                    if row["retry_count"] < 3:
                        conn.close()
                        return OperationResult(
                            operation_id=str(row["id"]),
                            state=state,
                            can_proceed=True,
                            message=f"Previous operation failed, retry {row['retry_count'] + 1}/3"
                        )
                    else:
                        conn.close()
                        return OperationResult(
                            operation_id=str(row["id"]),
                            state=state,
                            can_proceed=False,
                            message="Maximum retries exceeded"
                        )
            
            conn.close()
            
            # No existing operation, can proceed
            return OperationResult(
                operation_id="",
                state=OperationState.PENDING,
                can_proceed=True,
                message="No existing operation, can proceed"
            )
            
        except Exception as e:
            logger.error(f"Failed to check operation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def register_operation(self, request: IdempotencyRequest) -> OperationResult:
        """Register a new idempotent operation"""
        try:
            # First check if operation exists
            check_result = await self.check_operation(IdempotencyCheck(
                operation_type=request.operation_type,
                entity_id=request.entity_id,
                operation_params=request.operation_params
            ))
            
            if not check_result.can_proceed:
                return check_result
            
            # Generate operation key
            op_key = self._generate_operation_key(
                request.operation_type,
                request.entity_id,
                request.operation_params
            )
            key_str = op_key.to_string()
            
            # Create new operation
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            operation_id = str(uuid.uuid4())
            expires_at = datetime.utcnow() + timedelta(seconds=request.ttl_seconds)
            locked_until = datetime.utcnow() + timedelta(seconds=self.lock_timeout)
            
            try:
                cur.execute("""
                    INSERT INTO idempotent_operations 
                    (id, operation_key, operation_type, entity_id, params_hash, 
                     state, retry_count, expires_at, locked_until)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (operation_key) DO UPDATE
                    SET state = %s, 
                        retry_count = idempotent_operations.retry_count + 1,
                        locked_until = %s,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id, state, retry_count
                """, (
                    operation_id, key_str, request.operation_type.value,
                    request.entity_id, op_key.params_hash,
                    OperationState.IN_PROGRESS.value, request.retry_count,
                    expires_at, locked_until,
                    OperationState.IN_PROGRESS.value, locked_until
                ))
                
                result = cur.fetchone()
                if result:
                    operation_id = str(result[0])
                
                # Log to history
                cur.execute("""
                    INSERT INTO operation_history (operation_id, state_from, state_to, details)
                    VALUES (%s, %s, %s, %s)
                """, (
                    operation_id, OperationState.PENDING.value,
                    OperationState.IN_PROGRESS.value,
                    json.dumps({"entity_id": request.entity_id})
                ))
                
                conn.commit()
                
            except psycopg2.IntegrityError:
                # Operation already exists, get current state
                conn.rollback()
                cur.execute("""
                    SELECT id, state, result FROM idempotent_operations
                    WHERE operation_key = %s
                """, (key_str,))
                row = cur.fetchone()
                conn.close()
                
                if row:
                    return OperationResult(
                        operation_id=str(row[0]),
                        state=OperationState(row[1]),
                        can_proceed=False,
                        existing_result=row[2],
                        message="Operation already registered"
                    )
            
            conn.close()
            
            # Cache the operation
            cache_data = {
                "id": operation_id,
                "state": OperationState.IN_PROGRESS.value,
                "locked_until": locked_until.isoformat()
            }
            self.cache.set(f"idempotent:{key_str}", cache_data, ttl=request.ttl_seconds)
            
            logger.info(f"Registered operation {operation_id} with key {key_str}")
            
            return OperationResult(
                operation_id=operation_id,
                state=OperationState.IN_PROGRESS,
                can_proceed=True,
                message="Operation registered and locked"
            )
            
        except Exception as e:
            logger.error(f"Failed to register operation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def complete_operation(self, operation_id: str, result: Dict[str, Any], success: bool = True):
        """Mark operation as completed"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get operation details
            cur.execute("""
                SELECT operation_key, state FROM idempotent_operations
                WHERE id = %s
            """, (operation_id,))
            
            row = cur.fetchone()
            if not row:
                conn.close()
                raise HTTPException(status_code=404, detail="Operation not found")
            
            new_state = OperationState.COMPLETED if success else OperationState.FAILED
            
            # Update operation
            cur.execute("""
                UPDATE idempotent_operations
                SET state = %s, result = %s, updated_at = CURRENT_TIMESTAMP,
                    locked_until = NULL
                WHERE id = %s
            """, (new_state.value, json.dumps(result), operation_id))
            
            # Log to history
            cur.execute("""
                INSERT INTO operation_history (operation_id, state_from, state_to, details)
                VALUES (%s, %s, %s, %s)
            """, (
                operation_id, row["state"], new_state.value,
                json.dumps({"success": success})
            ))
            
            conn.commit()
            conn.close()
            
            # Update cache
            cache_data = {
                "id": operation_id,
                "state": new_state.value,
                "result": result
            }
            self.cache.set(f"idempotent:{row['operation_key']}", cache_data, ttl=3600)
            
            logger.info(f"Completed operation {operation_id} with state {new_state.value}")
            
            return {"status": "completed", "operation_id": operation_id}
            
        except Exception as e:
            logger.error(f"Failed to complete operation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "idempotent_state_manager",
                "status": "healthy",
                "operations_tracked": len(self.operations),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/check")
        async def check_operation(request: IdempotencyCheck):
            """Check if operation can proceed"""
            return await self.check_operation(request)
        
        @self.app.post("/register")
        async def register_operation(request: IdempotencyRequest):
            """Register new idempotent operation"""
            return await self.register_operation(request)
        
        @self.app.post("/complete/{operation_id}")
        async def complete_operation(operation_id: str, result: dict):
            """Mark operation as completed"""
            return await self.complete_operation(
                operation_id,
                result.get("data", {}),
                result.get("success", True)
            )
        
        @self.app.get("/operation/{operation_id}")
        async def get_operation(operation_id: str):
            """Get operation details"""
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                cur.execute("""
                    SELECT * FROM idempotent_operations WHERE id = %s
                """, (operation_id,))
                
                row = cur.fetchone()
                conn.close()
                
                if not row:
                    raise HTTPException(status_code=404, detail="Operation not found")
                
                return row
                
            except Exception as e:
                logger.error(f"Failed to get operation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/operations/entity/{entity_id}")
        async def get_entity_operations(entity_id: str):
            """Get all operations for an entity"""
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                cur.execute("""
                    SELECT * FROM idempotent_operations 
                    WHERE entity_id = %s
                    ORDER BY created_at DESC
                    LIMIT 100
                """, (entity_id,))
                
                rows = cur.fetchall()
                conn.close()
                
                return {"entity_id": entity_id, "operations": rows}
                
            except Exception as e:
                logger.error(f"Failed to get entity operations: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/cleanup")
        async def cleanup_expired():
            """Clean up expired operations"""
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor()
                
                # Delete expired operations
                cur.execute("""
                    DELETE FROM idempotent_operations
                    WHERE expires_at < CURRENT_TIMESTAMP
                    AND state IN (%s, %s)
                """, (OperationState.EXPIRED.value, OperationState.COMPLETED.value))
                
                deleted = cur.rowcount
                conn.commit()
                conn.close()
                
                logger.info(f"Cleaned up {deleted} expired operations")
                
                return {"cleaned": deleted}
                
            except Exception as e:
                logger.error(f"Failed to cleanup: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Idempotent State Manager starting up...")
        
        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired operations"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor()
                
                # Clean up expired operations
                cur.execute("""
                    UPDATE idempotent_operations
                    SET state = %s
                    WHERE expires_at < CURRENT_TIMESTAMP
                    AND state NOT IN (%s, %s, %s)
                """, (
                    OperationState.EXPIRED.value,
                    OperationState.COMPLETED.value,
                    OperationState.FAILED.value,
                    OperationState.EXPIRED.value
                ))
                
                # Release stale locks
                cur.execute("""
                    UPDATE idempotent_operations
                    SET locked_until = NULL, state = %s
                    WHERE locked_until < CURRENT_TIMESTAMP
                    AND state = %s
                """, (OperationState.FAILED.value, OperationState.IN_PROGRESS.value))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Idempotent State Manager shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = IdempotentStateManager()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("IDEMPOTENT_PORT", 8026))
    logger.info(f"Starting Idempotent State Manager on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()