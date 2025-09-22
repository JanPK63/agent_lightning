"""
Data models and configuration for Agent Lightning Monitoring Dashboard
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time"""
    timestamp: datetime
    agent_id: str
    metric_name: str
    value: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class DashboardConfig:
    """Configuration for monitoring dashboard"""
    refresh_interval: int = 1  # seconds
    max_data_points: int = 1000
    metrics_retention: int = 3600  # seconds
    alert_thresholds: Dict = field(default_factory=dict)
    dashboard_port: int = 8501
    websocket_port: int = 8765
    enhanced_api_url: str = "http://localhost:8002"  # Enhanced API endpoint


# Pydantic schemas for request/response validation
class TaskAssignmentRequest(BaseModel):
    """Validated request for task assignment"""
    task_description: str = Field(..., min_length=1, max_length=10000, description="Task description")
    agent_type: Optional[str] = Field(None, description="Specific agent type or 'auto'")
    priority: str = Field("normal", pattern="^(low|normal|high|urgent)$", description="Task priority")
    ai_model: str = Field("gpt-4o", description="AI model to use")
    reference_task_id: Optional[str] = Field(None, description="Reference to previous task")
    deployment_config: Optional[Dict[str, Any]] = Field(None, description="Deployment configuration")

    @field_validator('task_description')
    @classmethod
    def validate_task_description(cls, v):
        if not v.strip():
            raise ValueError('Task description cannot be empty')
        return v.strip()


class TaskAssignmentResponse(BaseModel):
    """Validated response from task assignment"""
    task_id: str = Field(..., description="Unique task identifier")
    agent_assigned: str = Field(..., description="Agent that was assigned")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Assignment confidence")
    status: str = Field(..., pattern="^(assigned|queued|failed)$", description="Task status")


class TaskFeedbackRequest(BaseModel):
    """Validated feedback request"""
    task_id: str = Field(..., description="Task identifier")
    agent_id: str = Field(..., description="Agent identifier")
    success: bool = Field(..., description="Whether task was successful")
    quality_score: int = Field(..., ge=0, le=10, description="Quality score 0-10")


class APIConnectionRequest(BaseModel):
    """Validated API connection request"""
    username: str = Field(..., description="API username")
    password: str = Field(..., description="API password")


class ChatRequest(BaseModel):
    """Validated chat request"""
    message: str = Field(..., min_length=1, max_length=2000, description="Chat message")
    agent_id: str = Field(..., description="Agent to chat with")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ChatResponse(BaseModel):
    """Validated chat response"""
    response: str = Field(..., description="Agent response")
    agent_id: str = Field(..., description="Responding agent")
    timestamp: str = Field(..., description="Response timestamp")