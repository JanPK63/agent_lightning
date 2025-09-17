# Solution Architecture: Workflow Engine Integration

**Document ID:** SA-004  
**Date:** 2025-09-06  
**Status:** For Review  
**Priority:** High  
**Author:** System Architect  
**Dependencies:** SA-001 (Database), SA-002 (Redis), SA-003 (Microservices)  

## Executive Summary

This solution architecture details the integration of the Workflow Engine service with the shared PostgreSQL database and Redis event bus. The Workflow Engine is critical for orchestrating multi-step processes, managing task execution across agents, and ensuring reliable workflow completion even in failure scenarios.

## Problem Statement

### Current Issues

- **In-Memory Workflow State:** Workflows are stored in memory, lost on service restart
- **No Persistence:** Running workflows cannot be resumed after crashes
- **Isolated Execution:** Workflows cannot coordinate across services
- **No Audit Trail:** No record of workflow execution history
- **Limited Scalability:** Cannot distribute workflow execution across instances

### Business Impact

- Workflow progress lost during service restarts
- Cannot handle long-running workflows reliably
- No visibility into workflow execution history
- Cannot scale workflow processing horizontally
- Poor recovery from failures

## Proposed Solution

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Workflow Engine Service                      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Workflow Orchestrator                   │   │
│  │                                                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │ Workflow │  │   Task   │  │ Execution│             │   │
│  │  │ Manager  │  │Scheduler │  │  Engine  │             │   │
│  │  └─────┬────┘  └─────┬────┘  └─────┬────┘             │   │
│  │        │             │              │                   │   │
│  └────────┼─────────────┼──────────────┼───────────────────┘   │
│           │             │              │                        │
│  ┌────────▼─────────────▼──────────────▼────────────────────┐   │
│  │              Data Access Layer (DAL)                     │   │
│  │                                                          │   │
│  │  - Workflow CRUD operations                              │   │
│  │  - Task assignment and tracking                          │   │
│  │  - State persistence                                     │   │
│  │  - Event publishing                                      │   │
│  └────────┬──────────────────────────┬──────────────────────┘   │
│           │                          │                          │
└───────────┼──────────────────────────┼──────────────────────────┘
            │                          │
     ┌──────▼──────┐            ┌──────▼──────┐
     │  PostgreSQL │            │    Redis    │
     │             │            │             │
     │ • Workflows │            │ • Events    │
     │ • Tasks     │            │ • Queue     │
     │ • History   │            │ • Locks     │
     └─────────────┘            └─────────────┘
```

### Integration Components

#### 1. Workflow Persistence

```python
# Workflow state in PostgreSQL
class WorkflowState:
    - workflow_id: UUID
    - name: str
    - status: enum
    - steps: JSONB
    - current_step: int
    - context: JSONB
    - created_by: str
    - assigned_to: str
    - started_at: timestamp
    - completed_at: timestamp
    - error_log: JSONB
```

#### 2. Task Management

```python
# Task assignment and tracking
class WorkflowTaskManager:
    def assign_task(self, workflow_id: str, agent_id: str, task_data: dict):
        """Assign task to agent with workflow context"""
        task = self.dal.create_task({
            'agent_id': agent_id,
            'workflow_id': workflow_id,
            'description': task_data['description'],
            'context': {
                'workflow_id': workflow_id,
                'step_number': task_data['step'],
                'retry_count': 0
            }
        })
        
        # Emit task assignment event
        self.event_bus.emit(EventChannel.TASK_ASSIGNED, {
            'task_id': task['id'],
            'agent_id': agent_id,
            'workflow_id': workflow_id
        })
        
        return task
```

#### 3. Event-Driven Orchestration

```python
# Event handlers for workflow coordination
class WorkflowEventHandlers:
    
    def on_task_completed(self, event):
        """Handle task completion"""
        task_id = event.data['task_id']
        workflow_id = event.data.get('workflow_id')
        
        if workflow_id:
            # Update workflow progress
            workflow = self.dal.get_workflow(workflow_id)
            current_step = workflow['context']['current_step']
            
            # Move to next step
            self.execute_next_step(workflow_id, current_step + 1)
    
    def on_task_failed(self, event):
        """Handle task failure with retry logic"""
        task_id = event.data['task_id']
        error = event.data['error']
        workflow_id = event.data.get('workflow_id')
        
        if workflow_id:
            # Implement retry logic
            retry_count = event.data.get('retry_count', 0)
            if retry_count < MAX_RETRIES:
                self.retry_task(task_id, workflow_id, retry_count + 1)
            else:
                self.fail_workflow(workflow_id, error)
```

#### 4. Execution Queue

```python
# Redis-based execution queue
class WorkflowQueue:
    def __init__(self):
        self.redis = get_cache()
        self.queue_key = "workflow:queue"
        self.processing_key = "workflow:processing"
    
    def enqueue(self, workflow_id: str, priority: int = 5):
        """Add workflow to execution queue"""
        score = time.time() - (priority * 1000)  # Higher priority = lower score
        self.redis.zadd(self.queue_key, {workflow_id: score})
    
    def dequeue(self) -> Optional[str]:
        """Get next workflow to execute"""
        # Atomic move from queue to processing
        workflow_id = self.redis.zpopmin(self.queue_key)
        if workflow_id:
            self.redis.hset(self.processing_key, workflow_id, time.time())
            return workflow_id
        return None
```

### Implementation Plan

#### Phase 1: Database Schema (Day 1)
1. Extend workflow table with execution fields
2. Add workflow_id foreign key to tasks table
3. Create workflow_history table for audit trail
4. Add indexes for performance

#### Phase 2: Core Integration (Day 2)
1. Implement DataAccessLayer usage
2. Replace in-memory storage with database
3. Add Redis queue for execution
4. Implement event handlers

#### Phase 3: Advanced Features (Day 3)
1. Implement retry logic with exponential backoff
2. Add workflow versioning
3. Create checkpoint/resume capability
4. Add parallel step execution

#### Phase 4: Testing & Migration (Day 4)
1. Test workflow persistence
2. Test failure recovery
3. Migrate existing workflows
4. Performance testing

### Key Features

#### 1. Workflow Persistence
- All workflow state stored in PostgreSQL
- Complete execution history maintained
- Audit trail for compliance

#### 2. Failure Recovery
```python
class WorkflowRecovery:
    async def recover_interrupted_workflows(self):
        """Recover workflows interrupted by service restart"""
        # Find all workflows that were running
        interrupted = self.dal.get_workflows_by_status('running')
        
        for workflow in interrupted:
            # Check last checkpoint
            last_checkpoint = workflow['context'].get('checkpoint')
            if last_checkpoint:
                # Resume from checkpoint
                await self.resume_from_checkpoint(workflow['id'], last_checkpoint)
            else:
                # Restart workflow
                await self.restart_workflow(workflow['id'])
```

#### 3. Distributed Execution
```python
class DistributedExecutor:
    def can_execute(self, workflow_id: str) -> bool:
        """Check if this instance can execute workflow"""
        # Use distributed lock
        lock_key = f"workflow:lock:{workflow_id}"
        return self.cache.acquire_lock(lock_key, timeout=300)
    
    async def execute_distributed(self, workflow_id: str):
        """Execute workflow with distributed locking"""
        if not self.can_execute(workflow_id):
            logger.info(f"Workflow {workflow_id} being executed by another instance")
            return
        
        try:
            await self.execute_workflow(workflow_id)
        finally:
            self.cache.release_lock(f"workflow:lock:{workflow_id}")
```

#### 4. Monitoring & Metrics
```python
class WorkflowMetrics:
    def record_execution_time(self, workflow_id: str, duration: float):
        self.dal.record_metric('workflow_execution_time', duration, {
            'workflow_id': workflow_id,
            'service': 'workflow_engine'
        })
    
    def record_step_completion(self, workflow_id: str, step: int, success: bool):
        self.dal.record_metric('workflow_step_completion', 1, {
            'workflow_id': workflow_id,
            'step': step,
            'success': success
        })
```

## Success Metrics

### Technical Metrics
- ✅ 100% workflow state persistence
- ✅ Zero data loss on service restart
- ✅ < 100ms workflow creation latency
- ✅ Successful recovery from failures
- ✅ Horizontal scaling capability

### Business Metrics
- ✅ Complete workflow execution history
- ✅ 99.9% workflow completion rate
- ✅ Reduced manual intervention by 80%
- ✅ Full audit trail for compliance
- ✅ Improved visibility into processes

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Database becomes bottleneck | Medium | High | Implement caching, optimize queries |
| Long-running workflows timeout | Medium | Medium | Implement checkpointing |
| Deadlocks in parallel execution | Low | High | Use proper locking strategies |
| Event loss during high load | Low | Medium | Implement event persistence |

## Testing Strategy

### Unit Tests
```python
def test_workflow_persistence():
    """Test workflow saved to database"""
    workflow = dal.create_workflow({...})
    assert workflow['id'] in database
    
def test_workflow_recovery():
    """Test workflow recovery after crash"""
    # Simulate crash
    # Verify workflow resumes correctly
```

### Integration Tests
- Test workflow execution across services
- Test event propagation
- Test failure scenarios
- Test concurrent execution

### Performance Tests
- Load test with 1000+ concurrent workflows
- Measure execution latency
- Test queue throughput
- Verify horizontal scaling

## Migration Strategy

1. Deploy new integrated service alongside existing
2. Route new workflows to integrated service
3. Migrate existing in-memory workflows
4. Monitor for issues
5. Decommission old service

## Rollback Plan

```bash
# 1. Stop integrated service
systemctl stop workflow-engine-integrated

# 2. Start original service
systemctl start workflow-engine

# 3. Restore from backup if needed
psql -U agent_user -d agent_lightning < workflow_backup.sql
```

## Approval

**Review Checklist:**
- [ ] Database schema supports all requirements
- [ ] Event handling is comprehensive
- [ ] Failure recovery is robust
- [ ] Performance requirements met
- [ ] Testing coverage adequate

**Sign-off Required From:**
- [ ] Technical Lead
- [ ] Workflow Team
- [ ] Database Team
- [ ] DevOps Team

---

**Next Steps After Approval:**
1. Implement database schema changes
2. Develop integrated service
3. Create comprehensive tests
4. Deploy to staging environment
5. Monitor and optimize

**Related Documents:**
- SA-001: Database Persistence Layer
- SA-002: Redis Cache & Event Bus
- SA-003: Microservices Integration
- Next: SA-005 Authentication & Authorization