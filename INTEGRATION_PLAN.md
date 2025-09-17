# Agent Lightning Integration Plan
## Connecting Microservices to Existing Data Layer

**Date:** 2025-09-05  
**Version:** 1.0  
**Objective:** Properly integrate microservices with the main application's data layer

## Current State vs Target State

### Current State (BROKEN)
```
┌─────────────────────────┐     ┌─────────────────────────┐
│   Main Application      │     │     Microservices       │
│      (Port 8051)        │     │   (Ports 8002-8007)     │
├─────────────────────────┤     ├─────────────────────────┤
│ • 10 Agents            │     │ • 1 Test Agent          │
│ • Full Knowledge Base   │     │ • Empty Knowledge       │
│ • Task History         │ ❌   │ • Isolated Tasks        │
│ • User Sessions        │     │ • No Shared Sessions    │
│ • Real Metrics         │     │ • Separate Metrics      │
└─────────────────────────┘     └─────────────────────────┘
         ↓                               ↓
    In-Memory Dict                  In-Memory Dict
    (Data Lost on                  (Different Data!)
     Restart)
```

### Target State (FIXED)
```
┌─────────────────────────┐     ┌─────────────────────────┐
│   Main Application      │     │     Microservices       │
│      (Port 8051)        │     │   (Ports 8002-8007)     │
├─────────────────────────┤     ├─────────────────────────┤
│ • Agent UI Layer       │     │ • Agent Designer API    │
│ • Dashboard Views      │     │ • Workflow Engine API   │
│ • WebSocket Client     │ ✅   │ • Integration Hub API   │
│ • Monitoring UI        │     │ • AI Model Service API  │
│ • Visual Builder UI    │     │ • Auth Service API      │
└─────────────────────────┘     └─────────────────────────┘
         ↓                               ↓
         └───────────┬───────────────────┘
                     ↓
            ┌────────────────┐
            │  PostgreSQL    │
            │   Database     │
            ├────────────────┤
            │ • Agents Table │
            │ • Tasks Table  │
            │ • Knowledge TB │
            │ • Sessions TB  │
            │ • Metrics TB   │
            └────────────────┘
                     ↓
            ┌────────────────┐
            │  Redis Cache   │
            │  & Pub/Sub     │
            └────────────────┘
```

## Step-by-Step Integration Plan

### Step 1: Create Shared Data Models

**File:** `shared/models.py`
```python
from sqlalchemy import Column, String, Integer, JSON, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Agent(Base):
    __tablename__ = 'agents'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    model = Column(String, nullable=False)
    specialization = Column(String)
    status = Column(String, default='idle')
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(String, primary_key=True)
    agent_id = Column(String)
    description = Column(Text)
    status = Column(String, default='pending')
    priority = Column(String, default='normal')
    result = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

class Knowledge(Base):
    __tablename__ = 'knowledge'
    
    id = Column(String, primary_key=True)
    agent_id = Column(String)
    category = Column(String)
    content = Column(Text)
    source = Column(String)
    metadata = Column(JSON)
    usage_count = Column(Integer, default=0)
    relevance_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Workflow(Base):
    __tablename__ = 'workflows'
    
    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(Text)
    steps = Column(JSON)
    status = Column(String)
    created_by = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### Step 2: Create Database Connection Manager

**File:** `shared/database.py`
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://agent_user:agent_pass@localhost:5432/agent_lightning"
)

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)

SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

@contextmanager
def get_db():
    """Provide a transactional scope around operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def init_database():
    """Initialize database tables."""
    from shared.models import Base
    Base.metadata.create_all(bind=engine)
```

### Step 3: Create Shared Cache Manager

**File:** `shared/cache.py`
```python
import redis
import json
import pickle
from typing import Any, Optional
import os

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0,
            decode_responses=False
        )
        self.pubsub = self.redis_client.pubsub()
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cache value."""
        serialized = pickle.dumps(value)
        if ttl:
            self.redis_client.setex(key, ttl, serialized)
        else:
            self.redis_client.set(key, serialized)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache value."""
        value = self.redis_client.get(key)
        if value:
            return pickle.loads(value)
        return None
    
    def publish(self, channel: str, message: dict):
        """Publish message to channel."""
        self.redis_client.publish(channel, json.dumps(message))
    
    def subscribe(self, channels: list):
        """Subscribe to channels."""
        self.pubsub.subscribe(*channels)
        return self.pubsub

cache = CacheManager()
```

### Step 4: Update Main Application Data Access

**File:** `monitoring_dashboard_updated.py` (excerpt)
```python
from shared.database import get_db, init_database
from shared.models import Agent, Task, Knowledge, Workflow
from shared.cache import cache

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_database()
    await load_agents_from_db()

async def load_agents_from_db():
    """Load all agents from database."""
    with get_db() as db:
        db_agents = db.query(Agent).all()
        
        for db_agent in db_agents:
            # Recreate agent instances from DB
            agent_class = get_agent_class(db_agent.specialization)
            agent = agent_class(
                agent_id=db_agent.id,
                name=db_agent.name,
                model=db_agent.model
            )
            # Restore knowledge from DB
            knowledge_items = db.query(Knowledge).filter(
                Knowledge.agent_id == db_agent.id
            ).all()
            agent.knowledge_base.load_from_db(knowledge_items)
            
            app.state.agents[db_agent.id] = agent
            
            # Cache agent data
            cache.set(f"agent:{db_agent.id}", agent.to_dict(), ttl=3600)

@app.post("/agents/{agent_id}/tasks")
async def create_task(agent_id: str, task: TaskCreate):
    """Create task and persist to DB."""
    with get_db() as db:
        # Create in database
        db_task = Task(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            description=task.description,
            priority=task.priority
        )
        db.add(db_task)
        db.commit()
        
        # Publish event
        cache.publish("task_created", {
            "task_id": db_task.id,
            "agent_id": agent_id,
            "description": task.description
        })
        
        # Process task
        agent = app.state.agents.get(agent_id)
        if agent:
            result = await agent.process_task(task.description)
            
            # Update database
            db_task.status = "completed"
            db_task.result = result
            db_task.completed_at = datetime.utcnow()
            db.commit()
            
            return result
```

### Step 5: Update Microservices to Use Shared Data

**File:** `services/agent_designer_service_fixed.py` (excerpt)
```python
from shared.database import get_db
from shared.models import Agent, Knowledge
from shared.cache import cache

class AgentDesignerService:
    def __init__(self):
        # No more in-memory storage!
        # self.agents = {}  # REMOVE THIS
        self.init_event_listeners()
    
    def init_event_listeners(self):
        """Subscribe to relevant events."""
        pubsub = cache.subscribe(['agent_created', 'agent_updated'])
        
        # Start listener thread
        import threading
        thread = threading.Thread(target=self.listen_for_events, args=(pubsub,))
        thread.daemon = True
        thread.start()
    
    def listen_for_events(self, pubsub):
        """Listen for cache events."""
        for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                self.handle_event(message['channel'], data)
    
    async def create_agent(self, agent_data: dict) -> dict:
        """Create agent in shared database."""
        with get_db() as db:
            # Check if agent exists
            existing = db.query(Agent).filter(
                Agent.id == agent_data['id']
            ).first()
            
            if existing:
                raise ValueError(f"Agent {agent_data['id']} already exists")
            
            # Create in database
            db_agent = Agent(
                id=agent_data['id'],
                name=agent_data['name'],
                model=agent_data['model'],
                specialization=agent_data['specialization'],
                config=agent_data.get('config', {})
            )
            db.add(db_agent)
            db.commit()
            
            # Cache and publish
            cache.set(f"agent:{db_agent.id}", agent_data, ttl=3600)
            cache.publish("agent_created", agent_data)
            
            return agent_data
    
    async def get_agent(self, agent_id: str) -> dict:
        """Get agent from cache or database."""
        # Try cache first
        cached = cache.get(f"agent:{agent_id}")
        if cached:
            return cached
        
        # Fall back to database
        with get_db() as db:
            db_agent = db.query(Agent).filter(
                Agent.id == agent_id
            ).first()
            
            if not db_agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent_data = {
                "id": db_agent.id,
                "name": db_agent.name,
                "model": db_agent.model,
                "specialization": db_agent.specialization,
                "status": db_agent.status,
                "config": db_agent.config
            }
            
            # Update cache
            cache.set(f"agent:{agent_id}", agent_data, ttl=3600)
            
            return agent_data
    
    async def list_agents(self) -> list:
        """List all agents from database."""
        with get_db() as db:
            db_agents = db.query(Agent).all()
            
            agents = []
            for db_agent in db_agents:
                agent_data = {
                    "id": db_agent.id,
                    "name": db_agent.name,
                    "model": db_agent.model,
                    "specialization": db_agent.specialization,
                    "status": db_agent.status
                }
                agents.append(agent_data)
            
            return agents
```

### Step 6: Update API Gateway Configuration

**File:** `gateway_config_fixed.yaml`
```yaml
services:
  - name: agent_designer
    url: http://localhost:8002
    routes:
      - path: /agents
        methods: [GET, POST, PUT, DELETE]
    database: shared
    cache: shared
    
  - name: workflow_engine
    url: http://localhost:8003
    routes:
      - path: /workflows
        methods: [GET, POST, PUT, DELETE]
    database: shared
    cache: shared
    
  - name: main_dashboard
    url: http://localhost:8051
    routes:
      - path: /dashboard
        methods: [GET]
    database: shared
    cache: shared

database:
  type: postgresql
  host: localhost
  port: 5432
  name: agent_lightning
  user: agent_user
  pool_size: 20

cache:
  type: redis
  host: localhost
  port: 6379
  db: 0

events:
  channels:
    - agent_created
    - agent_updated
    - task_created
    - task_completed
    - workflow_started
    - workflow_completed
```

### Step 7: Migration Script

**File:** `scripts/migrate_to_shared_db.py`
```python
#!/usr/bin/env python3
"""
Migrate existing in-memory data to shared database.
"""

import json
import os
from pathlib import Path
from shared.database import get_db, init_database
from shared.models import Agent, Knowledge, Task, Workflow

def migrate_knowledge_base():
    """Migrate .agent-knowledge/ files to database."""
    knowledge_dir = Path(".agent-knowledge")
    
    with get_db() as db:
        for json_file in knowledge_dir.glob("*.json"):
            agent_id = json_file.stem
            
            with open(json_file) as f:
                knowledge_items = json.load(f)
            
            for item in knowledge_items:
                db_knowledge = Knowledge(
                    id=item['id'],
                    agent_id=agent_id,
                    category=item['category'],
                    content=item['content'],
                    source=item['source'],
                    metadata=item.get('metadata', {}),
                    usage_count=item.get('usage_count', 0),
                    relevance_score=item.get('relevance_score', 1.0)
                )
                
                # Check if exists
                existing = db.query(Knowledge).filter(
                    Knowledge.id == item['id']
                ).first()
                
                if not existing:
                    db.add(db_knowledge)
        
        db.commit()
        print(f"✅ Migrated knowledge base to database")

def create_default_agents():
    """Create the 10 default agents in database."""
    default_agents = [
        {"id": "researcher", "name": "Research Specialist", "model": "claude-3-haiku", "specialization": "researcher"},
        {"id": "developer", "name": "Code Developer", "model": "claude-3-sonnet", "specialization": "developer"},
        {"id": "reviewer", "name": "Code Reviewer", "model": "claude-3-haiku", "specialization": "reviewer"},
        {"id": "optimizer", "name": "Performance Optimizer", "model": "claude-3-haiku", "specialization": "optimizer"},
        {"id": "writer", "name": "Documentation Writer", "model": "claude-3-haiku", "specialization": "writer"},
        {"id": "security_expert", "name": "Security Expert", "model": "claude-3-opus", "specialization": "security_expert"},
        {"id": "data_scientist", "name": "Data Scientist", "model": "claude-3-sonnet", "specialization": "data_scientist"},
        {"id": "devops_engineer", "name": "DevOps Engineer", "model": "claude-3-haiku", "specialization": "devops_engineer"},
        {"id": "blockchain_developer", "name": "Blockchain Developer", "model": "claude-3-haiku", "specialization": "blockchain_developer"},
        {"id": "system_architect", "name": "System Architect", "model": "claude-3-opus", "specialization": "system_architect"}
    ]
    
    with get_db() as db:
        for agent_data in default_agents:
            existing = db.query(Agent).filter(
                Agent.id == agent_data['id']
            ).first()
            
            if not existing:
                db_agent = Agent(**agent_data)
                db.add(db_agent)
        
        db.commit()
        print(f"✅ Created {len(default_agents)} agents in database")

if __name__ == "__main__":
    print("🚀 Starting migration to shared database...")
    
    # Initialize database
    init_database()
    print("✅ Database initialized")
    
    # Create default agents
    create_default_agents()
    
    # Migrate knowledge base
    migrate_knowledge_base()
    
    print("✨ Migration complete!")
```

## Implementation Timeline

### Day 1: Database Setup
1. Install PostgreSQL
2. Create database schema
3. Run migration script
4. Test connections

### Day 2: Update Main Application
1. Update monitoring_dashboard.py
2. Update agent classes
3. Test with database

### Day 3: Update Microservices
1. Update all service files
2. Remove in-memory storage
3. Connect to shared DB

### Day 4: Add Redis Cache
1. Install Redis
2. Implement cache layer
3. Add pub/sub events

### Day 5: Testing & Validation
1. Integration testing
2. Performance testing
3. Data consistency checks

## Success Criteria

✅ All 10 agents visible in all services  
✅ Tasks shared across services  
✅ Knowledge base persisted  
✅ Data survives restarts  
✅ Real-time updates via Redis pub/sub  
✅ Proper authentication flow  
✅ No data isolation  

## Next Steps After Integration

1. **Containerization**: Docker images for all services
2. **Orchestration**: Kubernetes deployment
3. **Monitoring**: Prometheus + Grafana
4. **CI/CD**: GitHub Actions pipeline
5. **Security**: TLS, secrets management
6. **Scaling**: Horizontal pod autoscaling