# Solution Architecture: Microservices Integration

**Document ID:** SA-003  
**Date:** 2025-09-06  
**Status:** For Review  
**Priority:** Critical  
**Author:** System Architect  
**Dependencies:** SA-001 (Database), SA-002 (Redis Cache)  

## Executive Summary

This solution architecture addresses the critical issue of disconnected microservices operating with isolated data stores. It provides a comprehensive plan to integrate all services with the shared PostgreSQL database and Redis cache, ensuring data consistency, eliminating duplication, and enabling true microservices architecture with shared state management.

## Problem Statement

### Current Issues

- **Data Isolation:** Each service has its own in-memory storage
- **Inconsistent State:** Services show different data (1 agent vs 10 agents)
- **No Data Sharing:** Services cannot access each other's data
- **Memory Waste:** Duplicate data in multiple services
- **No Persistence:** Service restarts lose all data

### Business Impact

- System shows inconsistent information
- Cannot scale services independently
- High memory usage from duplication
- No audit trail or history
- Poor user experience with stale data

## Proposed Solution

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Microservices Layer                          │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Agent     │  │  Workflow   │  │ Integration │            │
│  │  Designer   │  │   Engine    │  │     Hub     │  ...       │
│  │   (8002)    │  │   (8003)    │  │   (8004)    │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │                │                │                     │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                          │
                    ┌─────▼─────┐
                    │   Shared  │
                    │   Layer   │
                    └─────┬─────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
   ┌────▼────┐                        ┌────▼────┐
   │  Redis  │                        │Postgres │
   │  Cache  │◄───────────────────────┤Database │
   │ & Events│                        │         │
   └─────────┘                        └─────────┘
   
   Cache Strategy:                    Data Storage:
   • Read-through                     • Agents
   • Write-through                    • Tasks
   • Event propagation                • Knowledge
   • Distributed locks                • Workflows
```

### Integration Strategy

#### Phase 1: Shared Data Access Layer

```python
# shared/data_access.py
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from shared.database import db_manager
from shared.cache import get_cache
from shared.events import EventBus, EventChannel
from shared.models import Agent, Task, Knowledge, Workflow
import logging

logger = logging.getLogger(__name__)

class DataAccessLayer:
    """Unified data access for all microservices"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.db = db_manager
        self.cache = get_cache()
        self.event_bus = EventBus(service_name)
        self.event_bus.start()
    
    # ============== Agent Operations ==============
    
    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get agent with caching"""
        # Try cache first
        cache_key = f"agent:{agent_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Load from database
        with self.db.get_db() as session:
            agent = session.query(Agent).filter(
                Agent.id == agent_id
            ).first()
            
            if agent:
                agent_dict = agent.to_dict()
                # Cache for 1 hour
                self.cache.set(cache_key, agent_dict, ttl=3600)
                return agent_dict
        
        return None
    
    def list_agents(self) -> List[Dict]:
        """List all agents with caching"""
        cache_key = "agents:all"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        with self.db.get_db() as session:
            agents = session.query(Agent).all()
            agent_list = [a.to_dict() for a in agents]
            
            # Cache for 5 minutes
            self.cache.set(cache_key, agent_list, ttl=300)
            return agent_list
    
    def create_agent(self, agent_data: Dict) -> Dict:
        """Create agent with cache invalidation"""
        with self.db.get_db() as session:
            agent = Agent(**agent_data)
            session.add(agent)
            session.commit()
            
            agent_dict = agent.to_dict()
            
            # Update cache
            self.cache.set(f"agent:{agent.id}", agent_dict, ttl=3600)
            self.cache.delete("agents:all")  # Invalidate list
            
            # Emit event
            self.event_bus.emit(
                EventChannel.AGENT_CREATED,
                {"agent_id": agent.id, "agent": agent_dict}
            )
            
            return agent_dict
    
    def update_agent(self, agent_id: str, updates: Dict) -> Optional[Dict]:
        """Update agent with cache synchronization"""
        with self.db.get_db() as session:
            agent = session.query(Agent).filter(
                Agent.id == agent_id
            ).first()
            
            if not agent:
                return None
            
            for key, value in updates.items():
                setattr(agent, key, value)
            
            session.commit()
            agent_dict = agent.to_dict()
            
            # Update cache
            self.cache.set(f"agent:{agent_id}", agent_dict, ttl=3600)
            self.cache.delete("agents:all")
            
            # Emit event
            self.event_bus.emit(
                EventChannel.AGENT_UPDATED,
                {"agent_id": agent_id, "updates": updates}
            )
            
            return agent_dict
    
    # ============== Task Operations ==============
    
    def create_task(self, task_data: Dict) -> Dict:
        """Create task with event emission"""
        with self.db.get_db() as session:
            task = Task(**task_data)
            session.add(task)
            session.commit()
            
            task_dict = task.to_dict()
            
            # Cache task
            self.cache.set(f"task:{task.id}", task_dict, ttl=900)
            
            # Emit event
            self.event_bus.emit(
                EventChannel.TASK_CREATED,
                {"task_id": str(task.id), "task": task_dict}
            )
            
            return task_dict
    
    def update_task_status(self, task_id: str, status: str, 
                          result: Optional[Dict] = None) -> Optional[Dict]:
        """Update task status with events"""
        with self.db.get_db() as session:
            task = session.query(Task).filter(
                Task.id == task_id
            ).first()
            
            if not task:
                return None
            
            task.status = status
            if result:
                task.result = result
            
            if status == "started":
                task.started_at = datetime.utcnow()
                event_channel = EventChannel.TASK_STARTED
            elif status == "completed":
                task.completed_at = datetime.utcnow()
                event_channel = EventChannel.TASK_COMPLETED
            elif status == "failed":
                task.completed_at = datetime.utcnow()
                event_channel = EventChannel.TASK_FAILED
            else:
                event_channel = EventChannel.TASK_PROGRESS
            
            session.commit()
            task_dict = task.to_dict()
            
            # Update cache
            self.cache.set(f"task:{task_id}", task_dict, ttl=900)
            
            # Emit event
            self.event_bus.emit(
                event_channel,
                {"task_id": str(task_id), "status": status, "result": result}
            )
            
            return task_dict
```

#### Phase 2: Service Migration Pattern

```python
# services/agent_designer_service_integrated.py
from shared.data_access import DataAccessLayer
from shared.cache_decorators import cached, cache_lock
from fastapi import FastAPI, HTTPException
import logging

logger = logging.getLogger(__name__)

class AgentDesignerService:
    """Agent Designer Service with shared data layer"""
    
    def __init__(self):
        self.dal = DataAccessLayer("agent_designer")
        self.app = FastAPI(title="Agent Designer Service")
        self.setup_routes()
        self.setup_event_handlers()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/agents")
        async def list_agents():
            """List all agents from shared database"""
            try:
                agents = self.dal.list_agents()
                return {"agents": agents, "count": len(agents)}
            except Exception as e:
                logger.error(f"Failed to list agents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agents/{agent_id}")
        async def get_agent(agent_id: str):
            """Get specific agent from shared database"""
            agent = self.dal.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            return agent
        
        @self.app.post("/agents")
        async def create_agent(agent_data: dict):
            """Create new agent in shared database"""
            try:
                # Use distributed lock to prevent race conditions
                with self.dal.cache.lock(f"create_agent:{agent_data['id']}"):
                    # Check if already exists
                    existing = self.dal.get_agent(agent_data['id'])
                    if existing:
                        raise HTTPException(status_code=409, detail="Agent already exists")
                    
                    agent = self.dal.create_agent(agent_data)
                    return agent
            except Exception as e:
                logger.error(f"Failed to create agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/agents/{agent_id}")
        async def update_agent(agent_id: str, updates: dict):
            """Update agent in shared database"""
            agent = self.dal.update_agent(agent_id, updates)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            return agent
    
    def setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""
        
        def on_task_completed(event):
            """Handle task completion events"""
            task_id = event.data.get('task_id')
            logger.info(f"Task {task_id} completed, updating agent metrics")
            # Update agent performance metrics
            # This would trigger cache invalidation
        
        def on_workflow_started(event):
            """Handle workflow events"""
            workflow_id = event.data.get('workflow_id')
            logger.info(f"Workflow {workflow_id} started")
            # Update agent availability
        
        # Register handlers
        self.dal.event_bus.on(EventChannel.TASK_COMPLETED, on_task_completed)
        self.dal.event_bus.on(EventChannel.WORKFLOW_STARTED, on_workflow_started)
```

### Migration Plan

#### Step 1: Create Shared Data Access Layer (Day 1)
1. Implement DataAccessLayer class
2. Add CRUD operations for all entities
3. Integrate caching strategies
4. Add event emission
5. Create unit tests

#### Step 2: Migrate Agent Designer Service (Day 2)
1. Replace in-memory storage with DAL
2. Update all endpoints
3. Add event handlers
4. Test with shared database
5. Verify cache operations

#### Step 3: Migrate Workflow Engine (Day 3)
1. Replace in-memory workflows
2. Use shared task management
3. Implement workflow persistence
4. Add workflow events
5. Test orchestration

#### Step 4: Migrate Integration Hub (Day 4)
1. Connect to shared database
2. Store integration configs
3. Cache API responses
4. Add webhook events
5. Test external integrations

#### Step 5: Migrate Remaining Services (Days 5-6)
1. AI Model Service
2. Auth Service
3. WebSocket Service
4. Transaction Coordinator
5. Service Discovery

### Data Consistency Strategy

#### 1. Transaction Management
```python
from contextlib import contextmanager

@contextmanager
def distributed_transaction():
    """Manage distributed transactions"""
    tx_id = str(uuid.uuid4())
    
    try:
        # Begin transaction
        logger.info(f"Starting distributed transaction {tx_id}")
        
        # Acquire distributed lock
        with cache.lock(f"tx:{tx_id}", timeout=30):
            yield tx_id
            
        # Commit successful
        logger.info(f"Transaction {tx_id} committed")
        
    except Exception as e:
        # Rollback on error
        logger.error(f"Transaction {tx_id} failed: {e}")
        # Emit rollback event
        event_bus.emit(EventChannel.SYSTEM_ALERT, {
            "type": "transaction_failed",
            "tx_id": tx_id,
            "error": str(e)
        })
        raise
```

#### 2. Event Sourcing
```python
class EventStore:
    """Store all state changes as events"""
    
    def append(self, aggregate_id: str, event: Dict):
        """Append event to store"""
        event_data = {
            "aggregate_id": aggregate_id,
            "event_type": event['type'],
            "event_data": event['data'],
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name
        }
        
        # Store in database
        with self.db.get_db() as session:
            event_record = EventLog(**event_data)
            session.add(event_record)
            session.commit()
        
        # Publish to event bus
        self.event_bus.emit(event['type'], event['data'])
    
    def replay(self, aggregate_id: str) -> List[Dict]:
        """Replay events for aggregate"""
        with self.db.get_db() as session:
            events = session.query(EventLog).filter(
                EventLog.aggregate_id == aggregate_id
            ).order_by(EventLog.timestamp).all()
            
            return [e.to_dict() for e in events]
```

## Implementation Guidelines

### 1. Service Template
Each service should follow this pattern:
- Use DataAccessLayer for all data operations
- Never store state in memory
- Emit events for all state changes
- Subscribe to relevant events
- Implement health checks
- Use distributed locking for critical sections

### 2. Testing Strategy
```python
# tests/test_service_integration.py
import pytest
from shared.data_access import DataAccessLayer

@pytest.fixture
def dal():
    """Provide data access layer for tests"""
    return DataAccessLayer("test_service")

def test_agent_persistence(dal):
    """Test agent persists across service restarts"""
    # Create agent
    agent = dal.create_agent({
        "id": "test_agent",
        "name": "Test Agent",
        "model": "test-model"
    })
    
    # Verify in database
    retrieved = dal.get_agent("test_agent")
    assert retrieved['id'] == "test_agent"
    
    # Verify in cache
    cached = dal.cache.get("agent:test_agent")
    assert cached is not None

def test_event_propagation(dal):
    """Test events propagate between services"""
    events_received = []
    
    def handler(event):
        events_received.append(event)
    
    # Subscribe to events
    dal.event_bus.on(EventChannel.TASK_CREATED, handler)
    
    # Create task
    task = dal.create_task({
        "agent_id": "test_agent",
        "description": "Test task"
    })
    
    # Verify event received
    time.sleep(1)
    assert len(events_received) == 1
```

## Rollback Plan

### Rollback Strategy
1. Keep old service code in separate files
2. Use feature flags for gradual rollout
3. Maintain backward compatibility
4. Test rollback procedure
5. Monitor for issues

### Rollback Procedure
```bash
# 1. Switch feature flag
export USE_SHARED_DATA=false

# 2. Restart services with old code
systemctl restart agent-lightning-*

# 3. Verify services running
curl http://localhost:8002/health

# 4. Check logs for errors
tail -f /var/log/agent-lightning/*.log
```

## Success Metrics

### Technical Metrics
- ✅ All services using shared database
- ✅ Zero in-memory state storage
- ✅ 100% data consistency
- ✅ < 50ms data access latency
- ✅ Event propagation < 10ms

### Business Metrics
- ✅ Consistent data across all services
- ✅ Services can scale independently
- ✅ 50% reduction in memory usage
- ✅ Full audit trail available
- ✅ Zero data loss on restart

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Data corruption during migration | Low | High | Comprehensive testing, backups |
| Performance degradation | Medium | Medium | Caching, query optimization |
| Service communication failure | Low | High | Event retry, dead letter queue |
| Database becomes bottleneck | Medium | High | Connection pooling, read replicas |
| Cache inconsistency | Low | Medium | TTL strategy, invalidation |

## Migration Timeline

### Week 1
- Day 1: Implement DataAccessLayer
- Day 2: Migrate Agent Designer Service
- Day 3: Migrate Workflow Engine
- Day 4: Migrate Integration Hub
- Day 5: Migrate AI Model & Auth Services

### Week 2
- Day 1: Migrate WebSocket Service
- Day 2: Integration testing
- Day 3: Performance testing
- Day 4: Documentation update
- Day 5: Production deployment

## Monitoring & Observability

### Key Metrics
```python
# Metrics to track
service_data_latency_seconds
database_connections_active
cache_hit_ratio
events_published_total
events_failed_total
transaction_duration_seconds
```

### Dashboards
- Service health status
- Data access latency
- Cache performance
- Event flow visualization
- Error rates

## Approval

**Review Checklist:**
- [ ] Migration plan covers all services
- [ ] Data consistency strategy is sound
- [ ] Performance requirements are met
- [ ] Rollback plan is comprehensive
- [ ] Testing strategy is adequate

**Sign-off Required From:**
- [ ] Technical Lead
- [ ] Service Owners
- [ ] Database Team
- [ ] DevOps Team

---

**Next Steps After Approval:**
1. Implement DataAccessLayer
2. Begin service migration
3. Set up monitoring
4. Execute testing plan

**Related Documents:**
- SA-001: Database Persistence Layer
- SA-002: Redis Cache & Event Bus
- SYSTEM_ARCHITECTURE_ANALYSIS.md
- Next: SA-004 Service Discovery