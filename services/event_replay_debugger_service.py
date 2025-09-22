#!/usr/bin/env python3
"""
Event Replay Debugger Service for Agent Lightning
Provides REST API for debugging event replay functionality
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn
import logging

from ..shared.event_replay_debugger import event_replay_debugger
from ..shared.database import get_db_session
from ..shared.cache import get_cache
from ..shared.data_access import DataAccessLayer
from monitoring.metrics import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class StartReplayRequest(BaseModel):
    """Request to start event replay"""
    aggregate_id: str = Field(description="Aggregate ID to replay events for")
    from_version: int = Field(default=1, description="Starting version for replay")


class AddBreakpointRequest(BaseModel):
    """Request to add a breakpoint"""
    event_type: Optional[str] = Field(default=None, description="Event type to break on")
    condition: Optional[str] = Field(default=None, description="Condition expression to break on")


class AddWatchRequest(BaseModel):
    """Request to add a watch expression"""
    expression: str = Field(description="Watch expression to evaluate")


class DebugSessionResponse(BaseModel):
    """Response containing debug session information"""
    session_id: str
    state: str
    aggregate_id: Optional[str]
    from_version: int
    current_event: Optional[Dict[str, Any]]
    breakpoints: List[Dict[str, Any]]
    timeline_length: int
    watched_values: Dict[str, Any]
    last_error: Optional[Dict[str, Any]]


class EventReplayDebuggerService:
    """Service for event replay debugging with REST API"""

    def __init__(self):
        self.app = FastAPI(title="Event Replay Debugger Service", version="1.0.0")
        self.debugger = event_replay_debugger
        self.dal = DataAccessLayer("event_replay_debugger")
        self.cache = get_cache()
        self.metrics_collector = MetricsCollector()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Setup routes
        self._setup_routes()

        logger.info("✅ Event Replay Debugger Service initialized")

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                health_status = await self._get_health_status()
                status_code = 200 if health_status['database'] and health_status['cache'] else 503
                self.metrics_collector.increment_request("health", "GET", str(status_code))
                return {
                    "service": "event_replay_debugger",
                    "status": "healthy" if status_code == 200 else "degraded",
                    "timestamp": datetime.utcnow().isoformat(),
                    **health_status
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                self.metrics_collector.increment_error("health", type(e).__name__)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/sessions/start")
        async def start_replay_session(request: StartReplayRequest, background_tasks: BackgroundTasks):
            """Start a new event replay debugging session"""
            try:
                session_id = str(uuid.uuid4())

                # Start replay in background
                background_tasks.add_task(self._start_replay_async, session_id, request)

                return {
                    "session_id": session_id,
                    "status": "starting",
                    "message": "Event replay session started"
                }

            except Exception as e:
                logger.error(f"Failed to start replay session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/sessions/{session_id}/pause")
        async def pause_replay_session(session_id: str):
            """Pause event replay session"""
            try:
                await self.debugger.pause_replay()
                return {"status": "paused", "message": "Replay session paused"}
            except Exception as e:
                logger.error(f"Failed to pause replay session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/sessions/{session_id}/resume")
        async def resume_replay_session(session_id: str):
            """Resume event replay session"""
            try:
                await self.debugger.resume_replay()
                return {"status": "resumed", "message": "Replay session resumed"}
            except Exception as e:
                logger.error(f"Failed to resume replay session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/sessions/{session_id}/step")
        async def step_replay_session(session_id: str):
            """Step through event replay session"""
            try:
                await self.debugger.step_replay()
                return {"status": "stepped", "message": "Stepped to next event"}
            except Exception as e:
                logger.error(f"Failed to step replay session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/sessions/{session_id}/stop")
        async def stop_replay_session(session_id: str):
            """Stop event replay session"""
            try:
                await self.debugger.stop_replay()
                return {"status": "stopped", "message": "Replay session stopped"}
            except Exception as e:
                logger.error(f"Failed to stop replay session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/sessions/{session_id}/breakpoints")
        async def add_breakpoint(session_id: str, request: AddBreakpointRequest):
            """Add a breakpoint to the replay session"""
            try:
                breakpoint_id = self.debugger.add_breakpoint(
                    request.event_type,
                    request.condition
                )
                return {
                    "breakpoint_id": breakpoint_id,
                    "message": "Breakpoint added successfully"
                }
            except Exception as e:
                logger.error(f"Failed to add breakpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/sessions/{session_id}/breakpoints/{breakpoint_id}")
        async def remove_breakpoint(session_id: str, breakpoint_id: str):
            """Remove a breakpoint from the replay session"""
            try:
                success = self.debugger.remove_breakpoint(breakpoint_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Breakpoint not found")
                return {"message": "Breakpoint removed successfully"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to remove breakpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/sessions/{session_id}/watches")
        async def add_watch(session_id: str, request: AddWatchRequest):
            """Add a watch expression to the replay session"""
            try:
                self.debugger.add_watch(request.expression)
                return {"message": "Watch expression added successfully"}
            except Exception as e:
                logger.error(f"Failed to add watch: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/sessions/{session_id}/watches/{expression}")
        async def remove_watch(session_id: str, expression: str):
            """Remove a watch expression from the replay session"""
            try:
                self.debugger.remove_watch(expression)
                return {"message": "Watch expression removed successfully"}
            except Exception as e:
                logger.error(f"Failed to remove watch: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/sessions/{session_id}")
        async def get_debug_session(session_id: str):
            """Get debug session information"""
            try:
                debug_info = self.debugger.get_debug_info()
                return DebugSessionResponse(
                    session_id=session_id,
                    **debug_info
                )
            except Exception as e:
                logger.error(f"Failed to get debug session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/sessions/{session_id}/timeline")
        async def get_execution_timeline(session_id: str):
            """Get execution timeline for the debug session"""
            try:
                timeline = self.debugger.get_execution_timeline()
                return {
                    "timeline": [
                        {
                            "event_id": entry['event'].event_id,
                            "event_type": entry['event'].event_type,
                            "version": entry['event'].version,
                            "timestamp": entry['event'].timestamp.isoformat(),
                            "breakpoint_hit": entry['breakpoint_hit']
                        }
                        for entry in timeline
                    ]
                }
            except Exception as e:
                logger.error(f"Failed to get execution timeline: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/sessions/{session_id}/export")
        async def export_debug_session(session_id: str):
            """Export complete debug session data"""
            try:
                export_data = self.debugger.export_debug_session()
                return export_data
            except Exception as e:
                logger.error(f"Failed to export debug session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/sessions")
        async def list_debug_sessions():
            """List all active debug sessions"""
            try:
                sessions = []
                for session_id, session_data in self.active_sessions.items():
                    sessions.append({
                        "session_id": session_id,
                        "aggregate_id": session_data.get("aggregate_id"),
                        "state": session_data.get("state", "unknown"),
                        "started_at": session_data.get("started_at")
                    })
                return {"sessions": sessions}
            except Exception as e:
                logger.error(f"Failed to list debug sessions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _start_replay_async(self, session_id: str, request: StartReplayRequest):
        """Start replay asynchronously"""
        try:
            # Store session info
            self.active_sessions[session_id] = {
                "aggregate_id": request.aggregate_id,
                "state": "starting",
                "started_at": datetime.utcnow().isoformat()
            }

            # Start the replay
            await self.debugger.start_replay(request.aggregate_id, request.from_version)

            # Update session state
            self.active_sessions[session_id]["state"] = "running"

        except Exception as e:
            logger.error(f"Failed to start replay for session {session_id}: {e}")
            self.active_sessions[session_id]["state"] = "error"
            self.active_sessions[session_id]["error"] = str(e)

    async def _get_health_status(self) -> Dict[str, Any]:
        """Get health status of service dependencies"""
        try:
            # Check database
            db_session = get_db_session()
            db_healthy = db_session is not None
            if db_session:
                db_session.close()

            # Check cache
            cache_healthy = self.cache is not None

            return {
                "database": db_healthy,
                "cache": cache_healthy,
                "active_sessions": len(self.active_sessions)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "database": False,
                "cache": False,
                "active_sessions": len(self.active_sessions),
                "error": str(e)
            }


# Global service instance
event_replay_debugger_service = EventReplayDebuggerService()


def start_service(host: str = "0.0.0.0", port: int = 8012):
    """Start the Event Replay Debugger Service"""
    logger.info(f"Starting Event Replay Debugger Service on {host}:{port}")
    uvicorn.run(
        event_replay_debugger_service.app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    start_service()