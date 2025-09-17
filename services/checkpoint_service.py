#!/usr/bin/env python3
"""
Checkpoint Service - Enterprise-grade model and state persistence
Handles checkpoint creation, storage, restoration, and versioning
"""

import os
import sys
import json
import asyncio
import uuid
import pickle
import hashlib
import gzip
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import aiofiles
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointType(str, Enum):
    """Types of checkpoints"""
    MODEL_STATE = "model_state"
    TRAINING_STATE = "training_state"
    AGENT_STATE = "agent_state"
    WORKFLOW_STATE = "workflow_state"
    FULL_SNAPSHOT = "full_snapshot"


class CheckpointStatus(str, Enum):
    """Checkpoint status"""
    CREATING = "creating"
    READY = "ready"
    RESTORING = "restoring"
    ARCHIVED = "archived"
    FAILED = "failed"
    DELETED = "deleted"


class CheckpointCreate(BaseModel):
    """Checkpoint creation request"""
    agent_id: str
    checkpoint_type: CheckpointType
    name: Optional[str] = None
    description: Optional[str] = None
    state_data: Optional[Dict[str, Any]] = None
    model_architecture: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    epoch: Optional[int] = None
    step: Optional[int] = None
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = []
    is_best_model: bool = False
    parent_checkpoint_id: Optional[str] = None


class CheckpointRestore(BaseModel):
    """Checkpoint restoration request"""
    checkpoint_id: str
    agent_id: str
    restore_memory: bool = True
    restore_model: bool = True
    restore_optimizer: bool = True


class CheckpointSchedule(BaseModel):
    """Checkpoint scheduling configuration"""
    agent_id: str
    schedule_type: str = "periodic"  # periodic, on_improvement, on_milestone
    interval_minutes: Optional[int] = 60
    improvement_threshold: Optional[float] = 0.01
    max_checkpoints: int = 10
    retention_days: int = 30
    keep_best_n: int = 3


@dataclass
class CheckpointMetadata:
    """Checkpoint metadata"""
    id: str
    agent_id: str
    checkpoint_type: CheckpointType
    created_at: datetime
    file_size_bytes: int
    checksum: str
    storage_path: str
    compression_type: str = "gzip"


class CheckpointService:
    """Main Checkpoint Service for model recovery"""
    
    def __init__(self):
        self.app = FastAPI(title="Checkpoint Service", version="1.0.0")
        
        # Initialize Data Access Layer
        self.dal = DataAccessLayer("checkpoint_service")
        
        # Direct PostgreSQL connection for checkpoint operations
        self.db_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", 5432),
            database=os.getenv("POSTGRES_DB", "agent_lightning"),
            user=os.getenv("POSTGRES_USER", "agent_user"),
            password=os.getenv("POSTGRES_PASSWORD", "agent_password")
        )
        
        # Cache for checkpoint metadata
        self.cache = get_cache()
        
        # Storage configuration
        self.storage_base = os.getenv("CHECKPOINT_STORAGE", "/tmp/checkpoints")
        Path(self.storage_base).mkdir(parents=True, exist_ok=True)
        
        # S3 configuration (optional)
        self.s3_bucket = os.getenv("CHECKPOINT_S3_BUCKET")
        self.s3_client = None
        if self.s3_bucket:
            try:
                import boto3
                self.s3_client = boto3.client('s3')
                logger.info(f"✅ S3 storage enabled: {self.s3_bucket}")
            except ImportError:
                logger.warning("boto3 not installed, S3 storage disabled")
        
        logger.info(f"✅ Checkpoint Service initialized with storage at {self.storage_base}")
        
        self._setup_middleware()
        self._setup_routes()
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a database query"""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                if cur.description:
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
            
            # Check storage availability
            storage_ok = os.path.exists(self.storage_base) and os.access(self.storage_base, os.W_OK)
            
            return {
                "service": "checkpoint_service",
                "status": "healthy",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "storage": storage_ok,
                "s3_enabled": self.s3_client is not None,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/checkpoints")
        async def create_checkpoint(
            checkpoint_data: CheckpointCreate,
            background_tasks: BackgroundTasks
        ):
            """Create a new checkpoint"""
            try:
                checkpoint_id = str(uuid.uuid4())
                
                # Create storage path
                storage_path = self._get_storage_path(
                    checkpoint_data.agent_id,
                    checkpoint_id
                )
                
                # Start checkpoint creation in background
                background_tasks.add_task(
                    self._create_checkpoint_async,
                    checkpoint_id,
                    checkpoint_data,
                    storage_path
                )
                
                # Create initial database entry
                query = """
                    INSERT INTO checkpoints 
                    (id, agent_id, checkpoint_type, checkpoint_status, storage_path,
                     name, description, tags, epoch, step, training_loss, 
                     validation_loss, metrics, model_architecture, hyperparameters,
                     is_best_model, parent_checkpoint_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                """
                
                result = self.execute_query(
                    query,
                    (
                        checkpoint_id,
                        checkpoint_data.agent_id,
                        checkpoint_data.checkpoint_type.value,
                        CheckpointStatus.CREATING.value,
                        storage_path,
                        checkpoint_data.name or f"Checkpoint {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        checkpoint_data.description,
                        checkpoint_data.tags,
                        checkpoint_data.epoch,
                        checkpoint_data.step,
                        checkpoint_data.training_loss,
                        checkpoint_data.validation_loss,
                        json.dumps(checkpoint_data.metrics) if checkpoint_data.metrics else None,
                        json.dumps(checkpoint_data.model_architecture) if checkpoint_data.model_architecture else None,
                        json.dumps(checkpoint_data.hyperparameters) if checkpoint_data.hyperparameters else None,
                        checkpoint_data.is_best_model,
                        checkpoint_data.parent_checkpoint_id
                    )
                )
                
                if result:
                    logger.info(f"Creating checkpoint {checkpoint_id} for agent {checkpoint_data.agent_id}")
                    return {
                        "checkpoint_id": checkpoint_id,
                        "status": "creating",
                        "storage_path": storage_path
                    }
                
            except Exception as e:
                logger.error(f"Failed to create checkpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/checkpoints")
        async def list_checkpoints(
            agent_id: Optional[str] = None,
            checkpoint_type: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 20
        ):
            """List checkpoints with optional filtering"""
            try:
                conditions = []
                params = []
                
                if agent_id:
                    conditions.append("agent_id = %s")
                    params.append(agent_id)
                
                if checkpoint_type:
                    conditions.append("checkpoint_type = %s")
                    params.append(checkpoint_type)
                
                if status:
                    conditions.append("checkpoint_status = %s")
                    params.append(status)
                
                where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                
                query = f"""
                    SELECT * FROM checkpoints
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s
                """
                params.append(limit)
                
                checkpoints = self.execute_query(query, tuple(params))
                
                return {
                    "checkpoints": checkpoints,
                    "count": len(checkpoints)
                }
                
            except Exception as e:
                logger.error(f"Failed to list checkpoints: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/checkpoints/{checkpoint_id}")
        async def get_checkpoint(checkpoint_id: str):
            """Get checkpoint details"""
            try:
                query = "SELECT * FROM checkpoints WHERE id = %s"
                result = self.execute_query(query, (checkpoint_id,))
                
                if result:
                    return result[0]
                else:
                    raise HTTPException(status_code=404, detail="Checkpoint not found")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get checkpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/checkpoints/restore")
        async def restore_checkpoint(
            restore_request: CheckpointRestore,
            background_tasks: BackgroundTasks
        ):
            """Restore from checkpoint"""
            try:
                # Verify checkpoint exists
                checkpoint = await get_checkpoint(restore_request.checkpoint_id)
                
                if checkpoint['checkpoint_status'] != 'ready':
                    raise HTTPException(
                        status_code=400,
                        detail=f"Checkpoint is not ready for restoration: {checkpoint['checkpoint_status']}"
                    )
                
                # Start restoration in background
                restoration_id = str(uuid.uuid4())
                
                background_tasks.add_task(
                    self._restore_checkpoint_async,
                    restoration_id,
                    restore_request,
                    checkpoint
                )
                
                # Log restoration attempt
                query = """
                    INSERT INTO checkpoint_restorations
                    (id, checkpoint_id, agent_id, success)
                    VALUES (%s, %s, %s, FALSE)
                    RETURNING id
                """
                
                self.execute_query(
                    query,
                    (restoration_id, restore_request.checkpoint_id, restore_request.agent_id)
                )
                
                return {
                    "restoration_id": restoration_id,
                    "checkpoint_id": restore_request.checkpoint_id,
                    "status": "restoring"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to restore checkpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/checkpoints/{checkpoint_id}")
        async def delete_checkpoint(checkpoint_id: str, soft_delete: bool = True):
            """Delete or archive checkpoint"""
            try:
                if soft_delete:
                    # Soft delete - mark as deleted
                    query = """
                        UPDATE checkpoints
                        SET checkpoint_status = 'deleted'
                        WHERE id = %s
                    """
                else:
                    # Hard delete - remove from database and storage
                    checkpoint = await get_checkpoint(checkpoint_id)
                    
                    # Delete file
                    if os.path.exists(checkpoint['storage_path']):
                        os.remove(checkpoint['storage_path'])
                    
                    # Delete from S3 if applicable
                    if self.s3_client and checkpoint['storage_path'].startswith('s3://'):
                        # Parse S3 path and delete
                        pass
                    
                    query = "DELETE FROM checkpoints WHERE id = %s"
                
                self.execute_query(query, (checkpoint_id,))
                
                return {
                    "checkpoint_id": checkpoint_id,
                    "deleted": True,
                    "soft_delete": soft_delete
                }
                
            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/checkpoints/{checkpoint_id}/mark-best")
        async def mark_best_checkpoint(checkpoint_id: str):
            """Mark checkpoint as best model"""
            try:
                # Get checkpoint to find agent_id
                checkpoint = await get_checkpoint(checkpoint_id)
                
                # Call stored procedure
                query = "SELECT mark_best_checkpoint(%s, %s)"
                self.execute_query(query, (checkpoint['agent_id'], checkpoint_id))
                
                return {
                    "checkpoint_id": checkpoint_id,
                    "is_best_model": True
                }
                
            except Exception as e:
                logger.error(f"Failed to mark best checkpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/checkpoint-schedules")
        async def create_schedule(schedule: CheckpointSchedule):
            """Create checkpoint schedule for agent"""
            try:
                schedule_id = str(uuid.uuid4())
                
                query = """
                    INSERT INTO checkpoint_schedules
                    (id, agent_id, schedule_type, interval_minutes, 
                     improvement_threshold, max_checkpoints, retention_days, keep_best_n)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                """
                
                result = self.execute_query(
                    query,
                    (
                        schedule_id,
                        schedule.agent_id,
                        schedule.schedule_type,
                        schedule.interval_minutes,
                        schedule.improvement_threshold,
                        schedule.max_checkpoints,
                        schedule.retention_days,
                        schedule.keep_best_n
                    )
                )
                
                if result:
                    logger.info(f"Created checkpoint schedule for agent {schedule.agent_id}")
                    return result[0]
                    
            except Exception as e:
                logger.error(f"Failed to create schedule: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/model-registry")
        async def list_models(
            agent_id: Optional[str] = None,
            is_deployed: Optional[bool] = None
        ):
            """List models in registry"""
            try:
                conditions = []
                params = []
                
                if agent_id:
                    conditions.append("agent_id = %s")
                    params.append(agent_id)
                
                if is_deployed is not None:
                    conditions.append("is_deployed = %s")
                    params.append(is_deployed)
                
                where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                
                query = f"""
                    SELECT * FROM model_registry
                    {where_clause}
                    ORDER BY created_at DESC
                """
                
                models = self.execute_query(query, tuple(params))
                
                return {
                    "models": models,
                    "count": len(models)
                }
                
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/checkpoints/{checkpoint_id}/download")
        async def download_checkpoint(checkpoint_id: str):
            """Download checkpoint file"""
            try:
                checkpoint = await get_checkpoint(checkpoint_id)
                storage_path = checkpoint['storage_path']
                
                if not os.path.exists(storage_path):
                    raise HTTPException(status_code=404, detail="Checkpoint file not found")
                
                return FileResponse(
                    storage_path,
                    media_type='application/octet-stream',
                    filename=f"checkpoint_{checkpoint_id}.pkl.gz"
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to download checkpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _get_storage_path(self, agent_id: str, checkpoint_id: str) -> str:
        """Generate storage path for checkpoint"""
        return os.path.join(
            self.storage_base,
            agent_id,
            f"checkpoint_{checkpoint_id}.pkl.gz"
        )
    
    async def _create_checkpoint_async(
        self, 
        checkpoint_id: str,
        checkpoint_data: CheckpointCreate,
        storage_path: str
    ):
        """Asynchronously create checkpoint"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            
            # Prepare checkpoint data
            checkpoint_content = {
                "checkpoint_id": checkpoint_id,
                "agent_id": checkpoint_data.agent_id,
                "checkpoint_type": checkpoint_data.checkpoint_type.value,
                "created_at": datetime.now().isoformat(),
                "state_data": checkpoint_data.state_data,
                "model_architecture": checkpoint_data.model_architecture,
                "hyperparameters": checkpoint_data.hyperparameters,
                "epoch": checkpoint_data.epoch,
                "step": checkpoint_data.step,
                "metrics": checkpoint_data.metrics
            }
            
            # Serialize and compress
            serialized = pickle.dumps(checkpoint_content)
            compressed = gzip.compress(serialized)
            
            # Calculate checksum
            checksum = hashlib.sha256(compressed).hexdigest()
            
            # Write to file
            async with aiofiles.open(storage_path, 'wb') as f:
                await f.write(compressed)
            
            file_size = len(compressed)
            
            # Upload to S3 if configured
            if self.s3_client and self.s3_bucket:
                s3_key = f"checkpoints/{checkpoint_data.agent_id}/{checkpoint_id}.pkl.gz"
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=compressed
                )
                storage_path = f"s3://{self.s3_bucket}/{s3_key}"
            
            # Update database
            query = """
                UPDATE checkpoints
                SET checkpoint_status = %s,
                    file_size_bytes = %s,
                    checksum = %s,
                    compression_type = %s
                WHERE id = %s
            """
            
            self.execute_query(
                query,
                (
                    CheckpointStatus.READY.value,
                    file_size,
                    checksum,
                    "gzip",
                    checkpoint_id
                )
            )
            
            # Cache metadata
            self.cache.set(
                f"checkpoint:{checkpoint_id}",
                {
                    "id": checkpoint_id,
                    "status": "ready",
                    "checksum": checksum,
                    "size": file_size
                },
                ttl=3600
            )
            
            logger.info(f"Checkpoint {checkpoint_id} created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint {checkpoint_id}: {e}")
            
            # Update status to failed
            query = """
                UPDATE checkpoints
                SET checkpoint_status = %s
                WHERE id = %s
            """
            self.execute_query(query, (CheckpointStatus.FAILED.value, checkpoint_id))
    
    async def _restore_checkpoint_async(
        self,
        restoration_id: str,
        restore_request: CheckpointRestore,
        checkpoint: Dict
    ):
        """Asynchronously restore checkpoint"""
        try:
            start_time = datetime.now()
            storage_path = checkpoint['storage_path']
            
            # Download from S3 if needed
            if storage_path.startswith('s3://'):
                # Parse S3 path and download
                local_path = f"/tmp/restore_{checkpoint['id']}.pkl.gz"
                # self.s3_client.download_file(...)
                storage_path = local_path
            
            # Read and decompress checkpoint
            async with aiofiles.open(storage_path, 'rb') as f:
                compressed = await f.read()
            
            # Verify checksum
            checksum = hashlib.sha256(compressed).hexdigest()
            if checksum != checkpoint['checksum']:
                raise ValueError("Checkpoint checksum mismatch")
            
            # Decompress and deserialize
            decompressed = gzip.decompress(compressed)
            checkpoint_content = pickle.loads(decompressed)
            
            # Restore to agent (this would integrate with your agent system)
            # For now, we'll just update the database
            
            restoration_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Update restoration record
            query = """
                UPDATE checkpoint_restorations
                SET success = TRUE,
                    restoration_time_ms = %s
                WHERE id = %s
            """
            self.execute_query(query, (restoration_time, restoration_id))
            
            # Update checkpoint restored count
            query = """
                UPDATE checkpoints
                SET restored_count = restored_count + 1,
                    restored_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """
            self.execute_query(query, (checkpoint['id'],))
            
            logger.info(f"Checkpoint {checkpoint['id']} restored successfully in {restoration_time}ms")
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            
            # Update restoration record
            query = """
                UPDATE checkpoint_restorations
                SET success = FALSE,
                    error_message = %s
                WHERE id = %s
            """
            self.execute_query(query, (str(e), restoration_id))
    
    async def cleanup_old_checkpoints(self):
        """Periodic cleanup of old checkpoints"""
        try:
            # Call cleanup function
            query = "SELECT cleanup_old_checkpoints()"
            self.execute_query(query)
            
            logger.info("Old checkpoints cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Checkpoint Service starting up...")
        
        # Schedule periodic cleanup
        asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Run cleanup every 24 hours"""
        while True:
            await asyncio.sleep(86400)  # 24 hours
            await self.cleanup_old_checkpoints()
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Checkpoint Service shutting down...")
        self.dal.cleanup()
        self.db_pool.closeall()


def main():
    """Main entry point"""
    import uvicorn
    
    service = CheckpointService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("CHECKPOINT_SERVICE_PORT", 8013))
    logger.info(f"Starting Checkpoint Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()