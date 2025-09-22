class TaskAssignmentRequest(BaseModel):
    """Request for task assignment with capability validation"""
    task_id: str
    description: str
    priority: int = 5
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None
    force_execute: bool = False  # Allow overriding confidence checks