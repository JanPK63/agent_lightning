"""
Enterprise Data Models
Production-grade data models with validation and serialization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import json


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class Task:
    """Enterprise task model"""
    id: str
    workflow_id: str
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'workflow_id': self.workflow_id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'assigned_agent': self.assigned_agent,
            'result': self.result,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retry_attempts': self.retry_attempts,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            workflow_id=data['workflow_id'],
            name=data['name'],
            description=data['description'],
            status=TaskStatus(data.get('status', 'pending')),
            assigned_agent=data.get('assigned_agent'),
            result=data.get('result'),
            error_message=data.get('error_message'),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            retry_attempts=data.get('retry_attempts', 0),
            metadata=data.get('metadata', {})
        )


@dataclass
class Workflow:
    """Enterprise workflow model"""
    id: str
    name: str
    description: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_by: str = ""
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    tasks: List[Task] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
            'tasks': [task.to_dict() for task in self.tasks],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            status=WorkflowStatus(data.get('status', 'pending')),
            created_by=data.get('created_by', ''),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            progress=data.get('progress', 0.0),
            tasks=[Task.from_dict(task_data) for task_data in data.get('tasks', [])],
            metadata=data.get('metadata', {})
        )