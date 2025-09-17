#!/usr/bin/env python3
"""
Task History Service
Comprehensive audit logging and tracking for all task operations
Provides full traceability and debugging capabilities
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.cache import get_cache
from shared.events import EventChannel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoryAction(str, Enum):
    """Types of actions that are logged"""
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRIED = "task_retried"
    TASK_CANCELLED = "task_cancelled"
    TASK_VALIDATED = "task_validated"
    AGENT_SELECTED = "agent_selected"
    AGENT_REJECTED = "agent_rejected"
    ORCHESTRATION = "orchestration"
    STATE_CHANGE = "state_change"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"


class HistoryEntry(BaseModel):
    """A single history entry"""
    task_id: str = Field(description="Task ID")
    agent_id: Optional[str] = Field(default=None, description="Agent ID involved")
    action: HistoryAction = Field(description="Action type")
    status: Optional[str] = Field(default=None, description="Task status after action")
    details: Dict[str, Any] = Field(default={}, description="Additional details")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata")
    user_id: Optional[str] = Field(default=None, description="User who triggered action")
    service_name: Optional[str] = Field(default=None, description="Service that logged action")


class TaskTimeline(BaseModel):
    """Complete timeline for a task"""
    task_id: str
    description: str
    created_at: str
    current_status: str
    assigned_agent: Optional[str]
    total_duration_seconds: Optional[float]
    events: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class AgentPerformance(BaseModel):
    """Agent performance metrics"""
    agent_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_duration_seconds: float
    success_rate: float
    recent_tasks: List[Dict[str, Any]]


class TaskHistoryService:
    """Service for comprehensive task history tracking"""
    
    def __init__(self):
        self.app = FastAPI(title="Task History Service", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("task_history")
        self.cache = get_cache()
        
        # Database connection
        self.db_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agent_lightning"),
            "user": os.getenv("POSTGRES_USER", "agent_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "agent_password")
        }
        
        # Ensure tables exist
        self._ensure_tables()
        
        logger.info("✅ Task History Service initialized")
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_listeners()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _ensure_tables(self):
        """Ensure task_history table exists with proper indexes"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'task_history'
                )
            """)
            
            if not cur.fetchone()[0]:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE task_history (
                        id SERIAL PRIMARY KEY,
                        task_id UUID NOT NULL,
                        agent_id VARCHAR(100),
                        action VARCHAR(50) NOT NULL,
                        status VARCHAR(20),
                        details JSONB,
                        metadata JSONB,
                        user_id VARCHAR(100),
                        service_name VARCHAR(100),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX idx_task_history_task_id ON task_history(task_id);
                    CREATE INDEX idx_task_history_agent_id ON task_history(agent_id);
                    CREATE INDEX idx_task_history_action ON task_history(action);
                    CREATE INDEX idx_task_history_timestamp ON task_history(timestamp);
                    CREATE INDEX idx_task_history_service ON task_history(service_name);
                """)
                
                logger.info("Created task_history table")
            
            # Ensure all columns exist (for existing tables)
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'task_history'
            """)
            existing_columns = [row[0] for row in cur.fetchall()]
            
            if 'metadata' not in existing_columns:
                cur.execute("ALTER TABLE task_history ADD COLUMN metadata JSONB")
            if 'user_id' not in existing_columns:
                cur.execute("ALTER TABLE task_history ADD COLUMN user_id VARCHAR(100)")
            if 'service_name' not in existing_columns:
                cur.execute("ALTER TABLE task_history ADD COLUMN service_name VARCHAR(100)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to ensure tables: {e}")
    
    async def log_history(self, entry: HistoryEntry) -> Dict[str, Any]:
        """Log a history entry"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO task_history 
                (task_id, agent_id, action, status, details, metadata, user_id, service_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, timestamp
            """, (
                entry.task_id, entry.agent_id, entry.action.value, entry.status,
                json.dumps(entry.details), json.dumps(entry.metadata) if entry.metadata else None,
                entry.user_id, entry.service_name
            ))
            
            result = cur.fetchone()
            history_id = result[0]
            timestamp = result[1]
            
            conn.commit()
            conn.close()
            
            # Emit event
            event_data = {
                "history_id": history_id,
                "task_id": entry.task_id,
                "action": entry.action.value,
                "timestamp": timestamp.isoformat()
            }
            self.dal.event_bus.emit(EventChannel.SYSTEM_EVENT, event_data)
            
            # Cache recent entries
            cache_key = f"task_history:{entry.task_id}:recent"
            cached = self.cache.get(cache_key) or []
            cached.append({
                "id": history_id,
                "action": entry.action.value,
                "timestamp": timestamp.isoformat(),
                "agent_id": entry.agent_id
            })
            # Keep only last 20 entries in cache
            if len(cached) > 20:
                cached = cached[-20:]
            self.cache.set(cache_key, cached, ttl=3600)
            
            logger.info(f"Logged history: task={entry.task_id}, action={entry.action.value}")
            
            return {
                "id": history_id,
                "timestamp": timestamp.isoformat(),
                "status": "logged"
            }
            
        except Exception as e:
            logger.error(f"Failed to log history: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_task_timeline(self, task_id: str) -> TaskTimeline:
        """Get complete timeline for a task"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get task details
            cur.execute("""
                SELECT description, status, agent_id, created_at, completed_at
                FROM tasks WHERE id = %s
            """, (task_id,))
            
            task = cur.fetchone()
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            # Get all history entries
            cur.execute("""
                SELECT * FROM task_history 
                WHERE task_id = %s 
                ORDER BY timestamp ASC
            """, (task_id,))
            
            events = cur.fetchall()
            
            # Calculate metrics
            created_at = task["created_at"]
            completed_at = task.get("completed_at")
            
            duration = None
            if completed_at:
                duration = (completed_at - created_at).total_seconds()
            
            # Count action types
            action_counts = {}
            for event in events:
                action = event["action"]
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Format events
            formatted_events = []
            for event in events:
                formatted_events.append({
                    "timestamp": event["timestamp"].isoformat(),
                    "action": event["action"],
                    "agent_id": event.get("agent_id"),
                    "status": event.get("status"),
                    "details": event.get("details", {}),
                    "service": event.get("service_name")
                })
            
            conn.close()
            
            return TaskTimeline(
                task_id=task_id,
                description=task["description"],
                created_at=created_at.isoformat(),
                current_status=task["status"],
                assigned_agent=task.get("agent_id"),
                total_duration_seconds=duration,
                events=formatted_events,
                metrics={
                    "total_events": len(events),
                    "action_counts": action_counts,
                    "retry_count": action_counts.get(HistoryAction.TASK_RETRIED.value, 0),
                    "error_count": action_counts.get(HistoryAction.ERROR_OCCURRED.value, 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get timeline: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_agent_performance(self, agent_id: str, 
                                   days: int = 7) -> AgentPerformance:
        """Get performance metrics for an agent"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            since = datetime.utcnow() - timedelta(days=days)
            
            # Get task counts
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT task_id) as total_tasks,
                    COUNT(DISTINCT CASE WHEN action = %s THEN task_id END) as completed,
                    COUNT(DISTINCT CASE WHEN action = %s THEN task_id END) as failed
                FROM task_history
                WHERE agent_id = %s AND timestamp >= %s
            """, (
                HistoryAction.TASK_COMPLETED.value,
                HistoryAction.TASK_FAILED.value,
                agent_id, since
            ))
            
            counts = cur.fetchone()
            
            # Get average duration for completed tasks
            cur.execute("""
                SELECT AVG(
                    EXTRACT(EPOCH FROM (completed_at - started_at))
                ) as avg_duration
                FROM tasks
                WHERE agent_id = %s 
                AND completed_at IS NOT NULL
                AND started_at IS NOT NULL
                AND created_at >= %s
            """, (agent_id, since))
            
            avg_duration = cur.fetchone()["avg_duration"] or 0
            
            # Get recent tasks
            cur.execute("""
                SELECT DISTINCT ON (t.id)
                    t.id, t.description, t.status, t.created_at, t.completed_at
                FROM tasks t
                JOIN task_history th ON t.id = th.task_id::uuid
                WHERE th.agent_id = %s
                ORDER BY t.id, t.created_at DESC
                LIMIT 10
            """, (agent_id,))
            
            recent_tasks = []
            for task in cur.fetchall():
                recent_tasks.append({
                    "task_id": str(task["id"]),
                    "description": task["description"],
                    "status": task["status"],
                    "created_at": task["created_at"].isoformat()
                })
            
            conn.close()
            
            total = counts["total_tasks"] or 1  # Avoid division by zero
            success_rate = (counts["completed"] or 0) / total
            
            return AgentPerformance(
                agent_id=agent_id,
                total_tasks=counts["total_tasks"] or 0,
                completed_tasks=counts["completed"] or 0,
                failed_tasks=counts["failed"] or 0,
                average_duration_seconds=avg_duration,
                success_rate=success_rate,
                recent_tasks=recent_tasks
            )
            
        except Exception as e:
            logger.error(f"Failed to get agent performance: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_event_listeners(self):
        """Setup event listeners for automatic logging"""
        
        def on_task_event(event):
            """Handle task events"""
            try:
                # Map event types to history actions
                action_map = {
                    "task_created": HistoryAction.TASK_CREATED,
                    "task_started": HistoryAction.TASK_STARTED,
                    "task_completed": HistoryAction.TASK_COMPLETED,
                    "task_failed": HistoryAction.TASK_FAILED
                }
                
                event_type = event.data.get("type")
                if event_type in action_map:
                    entry = HistoryEntry(
                        task_id=event.data.get("task_id"),
                        agent_id=event.data.get("agent_id"),
                        action=action_map[event_type],
                        status=event.data.get("status"),
                        details=event.data.get("details", {}),
                        service_name=event.data.get("service", "unknown")
                    )
                    
                    # Log asynchronously
                    asyncio.create_task(self.log_history(entry))
                    
            except Exception as e:
                logger.error(f"Failed to handle event: {e}")
        
        # Subscribe to task events
        self.dal.event_bus.on(EventChannel.TASK_CREATED, on_task_event)
        self.dal.event_bus.on(EventChannel.TASK_STARTED, on_task_event)
        self.dal.event_bus.on(EventChannel.TASK_COMPLETED, on_task_event)
        self.dal.event_bus.on(EventChannel.TASK_FAILED, on_task_event)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "task_history",
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/log")
        async def log_entry(entry: HistoryEntry):
            """Log a history entry"""
            return await self.log_history(entry)
        
        @self.app.get("/task/{task_id}/timeline")
        async def get_timeline(task_id: str):
            """Get complete timeline for a task"""
            return await self.get_task_timeline(task_id)
        
        @self.app.get("/task/{task_id}/history")
        async def get_task_history(
            task_id: str,
            limit: int = Query(100, description="Maximum entries to return")
        ):
            """Get history entries for a task"""
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                cur.execute("""
                    SELECT * FROM task_history
                    WHERE task_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (task_id, limit))
                
                entries = cur.fetchall()
                conn.close()
                
                # Format timestamps
                for entry in entries:
                    entry["timestamp"] = entry["timestamp"].isoformat()
                
                return {
                    "task_id": task_id,
                    "count": len(entries),
                    "entries": entries
                }
                
            except Exception as e:
                logger.error(f"Failed to get task history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agent/{agent_id}/performance")
        async def get_agent_perf(
            agent_id: str,
            days: int = Query(7, description="Days to look back")
        ):
            """Get agent performance metrics"""
            return await self.get_agent_performance(agent_id, days)
        
        @self.app.get("/agent/{agent_id}/history")
        async def get_agent_history(
            agent_id: str,
            limit: int = Query(100, description="Maximum entries")
        ):
            """Get history entries for an agent"""
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                cur.execute("""
                    SELECT * FROM task_history
                    WHERE agent_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (agent_id, limit))
                
                entries = cur.fetchall()
                conn.close()
                
                for entry in entries:
                    entry["timestamp"] = entry["timestamp"].isoformat()
                
                return {
                    "agent_id": agent_id,
                    "count": len(entries),
                    "entries": entries
                }
                
            except Exception as e:
                logger.error(f"Failed to get agent history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/recent")
        async def get_recent_history(
            limit: int = Query(50, description="Maximum entries"),
            action: Optional[str] = Query(None, description="Filter by action")
        ):
            """Get recent history entries"""
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                if action:
                    cur.execute("""
                        SELECT * FROM task_history
                        WHERE action = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (action, limit))
                else:
                    cur.execute("""
                        SELECT * FROM task_history
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (limit,))
                
                entries = cur.fetchall()
                conn.close()
                
                for entry in entries:
                    entry["timestamp"] = entry["timestamp"].isoformat()
                
                return {
                    "count": len(entries),
                    "entries": entries
                }
                
            except Exception as e:
                logger.error(f"Failed to get recent history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/summary")
        async def get_analytics_summary(
            days: int = Query(7, description="Days to analyze")
        ):
            """Get analytics summary"""
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                since = datetime.utcnow() - timedelta(days=days)
                
                # Get action counts
                cur.execute("""
                    SELECT action, COUNT(*) as count
                    FROM task_history
                    WHERE timestamp >= %s
                    GROUP BY action
                    ORDER BY count DESC
                """, (since,))
                
                action_counts = {row["action"]: row["count"] for row in cur.fetchall()}
                
                # Get top agents
                cur.execute("""
                    SELECT agent_id, COUNT(DISTINCT task_id) as task_count
                    FROM task_history
                    WHERE agent_id IS NOT NULL AND timestamp >= %s
                    GROUP BY agent_id
                    ORDER BY task_count DESC
                    LIMIT 10
                """, (since,))
                
                top_agents = cur.fetchall()
                
                # Get error rate
                cur.execute("""
                    SELECT 
                        COUNT(CASE WHEN action = %s THEN 1 END) as errors,
                        COUNT(*) as total
                    FROM task_history
                    WHERE timestamp >= %s
                """, (HistoryAction.ERROR_OCCURRED.value, since))
                
                error_stats = cur.fetchone()
                error_rate = (error_stats["errors"] / error_stats["total"]) if error_stats["total"] > 0 else 0
                
                conn.close()
                
                return {
                    "period_days": days,
                    "since": since.isoformat(),
                    "action_counts": action_counts,
                    "top_agents": top_agents,
                    "error_rate": error_rate,
                    "total_events": error_stats["total"]
                }
                
            except Exception as e:
                logger.error(f"Failed to get analytics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Task History Service starting up...")
        
        # Start periodic cleanup
        asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodically clean up old history entries"""
        while True:
            try:
                await asyncio.sleep(86400)  # Daily
                
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor()
                
                # Keep only last 90 days of history
                cutoff = datetime.utcnow() - timedelta(days=90)
                
                cur.execute("""
                    DELETE FROM task_history
                    WHERE timestamp < %s
                """, (cutoff,))
                
                deleted = cur.rowcount
                conn.commit()
                conn.close()
                
                logger.info(f"Cleaned up {deleted} old history entries")
                
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Task History Service shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = TaskHistoryService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("TASK_HISTORY_PORT", 8027))
    logger.info(f"Starting Task History Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()