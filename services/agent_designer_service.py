#!/usr/bin/env python3
"""
Agent Designer Microservice
Handles all agent design, creation, and management operations
"""

import os
import sys
import json
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
import logging

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
# Database imports - will use in-memory for now
# import redis
# import psycopg2
# from psycopg2.extras import RealDictCursor
# import asyncpg

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    DRAFT = "draft"
    TESTING = "testing"
    ACTIVE = "active"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class AgentType(str, Enum):
    """Types of agents"""
    CONVERSATIONAL = "conversational"
    TASK_EXECUTOR = "task_executor"
    DATA_PROCESSOR = "data_processor"
    INTEGRATION = "integration"
    MONITORING = "monitoring"
    CUSTOM = "custom"


# Pydantic Models
class AgentConfig(BaseModel):
    """Agent configuration model"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    agent_type: AgentType
    capabilities: List[str] = Field(default_factory=list)
    integrations: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    ai_model_config: Dict[str, Any] = Field(default_factory=dict)  # Renamed to avoid conflict
    workflow_config: Optional[Dict[str, Any]] = None
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    resource_limits: Dict[str, Any] = Field(default_factory=dict)


class Agent(BaseModel):
    """Agent model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str
    name: str
    description: Optional[str] = None
    agent_type: AgentType
    status: AgentStatus = AgentStatus.DRAFT
    configuration: AgentConfig
    version: int = 1
    created_by: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    deployed_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentTemplate(BaseModel):
    """Agent template model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: str
    description: str
    configuration: AgentConfig
    is_public: bool = False
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    usage_count: int = 0


class AgentWorkflow(BaseModel):
    """Agent workflow model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    workflow_data: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class CreateAgentRequest(BaseModel):
    """Request model for creating an agent"""
    name: str
    description: Optional[str] = None
    agent_type: AgentType
    configuration: Optional[AgentConfig] = None
    template_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class UpdateAgentRequest(BaseModel):
    """Request model for updating an agent"""
    name: Optional[str] = None
    description: Optional[str] = None
    configuration: Optional[AgentConfig] = None
    status: Optional[AgentStatus] = None
    tags: Optional[List[str]] = None


class DeployAgentRequest(BaseModel):
    """Request model for deploying an agent"""
    environment: str = "production"
    replicas: int = 1
    auto_scale: bool = False
    max_replicas: int = 5
    resource_overrides: Optional[Dict[str, Any]] = None


class AgentDesignerService:
    """Main Agent Designer Service class"""
    
    def __init__(self):
        self.app = FastAPI(title="Agent Designer Service", version="1.0.0")
        self.db_pool = None  # Will use in-memory for now
        self.redis_client = None  # Will use in-memory for now
        self.templates_cache: Dict[str, AgentTemplate] = {}
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_event_handlers(self):
        """Setup startup and shutdown event handlers"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize connections on startup"""
            await self._init_database()
            await self._init_redis()
            await self._load_templates()
            logger.info("Agent Designer Service started successfully")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup connections on shutdown"""
            if self.db_pool:
                await self.db_pool.close()
            if self.redis_client:
                await self.redis_client.close()
            logger.info("Agent Designer Service shut down")
    
    async def _init_database(self):
        """Initialize PostgreSQL connection pool"""
        try:
            # Use in-memory simulation if PostgreSQL not available
            db_url = os.getenv("DATABASE_URL", "postgresql://localhost/agent_designer")
            
            # For now, we'll simulate the database in memory
            # In production, uncomment the following:
            # self.db_pool = await asyncpg.create_pool(db_url, min_size=5, max_size=20)
            
            logger.info("Database connection initialized (simulated)")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}. Using in-memory storage.")
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost")
            # For now, we'll simulate Redis in memory
            # In production, uncomment the following:
            # self.redis_client = await aioredis.from_url(redis_url)
            
            logger.info("Redis connection initialized (simulated)")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
    
    async def _load_templates(self):
        """Load agent templates"""
        # Load default templates
        self.templates_cache = {
            "conversational-basic": AgentTemplate(
                id="template-1",
                name="Basic Conversational Agent",
                category="conversational",
                description="A simple conversational agent for customer support",
                configuration=AgentConfig(
                    name="Conversational Agent",
                    agent_type=AgentType.CONVERSATIONAL,
                    capabilities=["chat", "qa", "sentiment_analysis"],
                    ai_model_config={"model": "gpt-4", "temperature": 0.7}
                ),
                is_public=True,
                tags=["starter", "support", "chat"]
            ),
            "task-executor-basic": AgentTemplate(
                id="template-2",
                name="Task Executor Agent",
                category="automation",
                description="An agent for executing automated tasks",
                configuration=AgentConfig(
                    name="Task Executor",
                    agent_type=AgentType.TASK_EXECUTOR,
                    capabilities=["task_scheduling", "workflow_execution", "error_handling"],
                    ai_model_config={"model": "gpt-3.5-turbo", "temperature": 0.3}
                ),
                is_public=True,
                tags=["automation", "tasks", "workflow"]
            ),
            "data-processor-basic": AgentTemplate(
                id="template-3",
                name="Data Processing Agent",
                category="data",
                description="An agent for processing and analyzing data",
                configuration=AgentConfig(
                    name="Data Processor",
                    agent_type=AgentType.DATA_PROCESSOR,
                    capabilities=["data_extraction", "transformation", "analysis"],
                    ai_model_config={"model": "gpt-4", "temperature": 0.2}
                ),
                is_public=True,
                tags=["data", "analytics", "processing"]
            )
        }
        logger.info(f"Loaded {len(self.templates_cache)} agent templates")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            """Service health check"""
            return {
                "status": "healthy",
                "service": "agent_designer",
                "timestamp": datetime.now().isoformat()
            }
        
        # Agent CRUD operations
        @self.app.post("/api/v1/agents", response_model=Agent)
        async def create_agent(request: CreateAgentRequest, req: Request):
            """Create a new agent"""
            # Simulate user context (in production, extract from JWT)
            user_id = req.headers.get("X-User-ID", "default-user")
            org_id = req.headers.get("X-Organization-ID", "default-org")
            
            # Create configuration from template if provided
            if request.template_id:
                template = self.templates_cache.get(request.template_id)
                if not template:
                    raise HTTPException(status_code=404, detail="Template not found")
                configuration = template.configuration
            else:
                configuration = request.configuration or AgentConfig(
                    name=request.name,
                    agent_type=request.agent_type,
                    capabilities=[],
                    ai_model_config={}
                )
            
            # Create agent
            agent = Agent(
                organization_id=org_id,
                name=request.name,
                description=request.description,
                agent_type=request.agent_type,
                configuration=configuration,
                created_by=user_id,
                tags=request.tags
            )
            
            # Store agent (simulated - in production, save to database)
            await self._store_agent(agent)
            
            logger.info(f"Created agent: {agent.id}")
            return agent
        
        @self.app.get("/api/v1/agents", response_model=List[Agent])
        async def list_agents(req: Request, status: Optional[AgentStatus] = None):
            """List all agents for an organization"""
            org_id = req.headers.get("X-Organization-ID", "default-org")
            
            # Retrieve agents (simulated)
            agents = await self._get_agents(org_id, status)
            return agents
        
        @self.app.get("/api/v1/agents/{agent_id}", response_model=Agent)
        async def get_agent(agent_id: str):
            """Get a specific agent"""
            agent = await self._get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            return agent
        
        @self.app.put("/api/v1/agents/{agent_id}", response_model=Agent)
        async def update_agent(agent_id: str, request: UpdateAgentRequest):
            """Update an agent"""
            agent = await self._get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            # Update fields
            if request.name:
                agent.name = request.name
            if request.description is not None:
                agent.description = request.description
            if request.configuration:
                agent.configuration = request.configuration
            if request.status:
                agent.status = request.status
            if request.tags is not None:
                agent.tags = request.tags
            
            agent.updated_at = datetime.now()
            agent.version += 1
            
            # Store updated agent
            await self._store_agent(agent)
            
            logger.info(f"Updated agent: {agent_id}")
            return agent
        
        @self.app.delete("/api/v1/agents/{agent_id}")
        async def delete_agent(agent_id: str):
            """Delete an agent"""
            agent = await self._get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            # Delete agent (simulated)
            await self._delete_agent(agent_id)
            
            logger.info(f"Deleted agent: {agent_id}")
            return {"message": "Agent deleted successfully"}
        
        # Agent deployment
        @self.app.post("/api/v1/agents/{agent_id}/deploy")
        async def deploy_agent(agent_id: str, request: DeployAgentRequest):
            """Deploy an agent"""
            agent = await self._get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            if agent.status not in [AgentStatus.TESTING, AgentStatus.ACTIVE]:
                raise HTTPException(
                    status_code=400,
                    detail="Agent must be in testing or active status to deploy"
                )
            
            # Simulate deployment
            deployment_id = str(uuid.uuid4())
            agent.status = AgentStatus.DEPLOYED
            agent.deployed_at = datetime.now()
            agent.metadata["deployment_id"] = deployment_id
            agent.metadata["deployment_config"] = asdict(request)
            
            await self._store_agent(agent)
            
            logger.info(f"Deployed agent: {agent_id} with deployment ID: {deployment_id}")
            return {
                "deployment_id": deployment_id,
                "status": "deployed",
                "agent_id": agent_id,
                "environment": request.environment,
                "replicas": request.replicas
            }
        
        # Template operations
        @self.app.get("/api/v1/templates", response_model=List[AgentTemplate])
        async def list_templates(category: Optional[str] = None):
            """List available agent templates"""
            templates = list(self.templates_cache.values())
            
            if category:
                templates = [t for t in templates if t.category == category]
            
            return templates
        
        @self.app.get("/api/v1/templates/{template_id}", response_model=AgentTemplate)
        async def get_template(template_id: str):
            """Get a specific template"""
            template = self.templates_cache.get(template_id)
            if not template:
                raise HTTPException(status_code=404, detail="Template not found")
            return template
        
        # Workflow operations
        @self.app.post("/api/v1/agents/{agent_id}/workflow", response_model=AgentWorkflow)
        async def create_workflow(agent_id: str, workflow_data: Dict[str, Any]):
            """Create or update agent workflow"""
            agent = await self._get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            workflow = AgentWorkflow(
                agent_id=agent_id,
                workflow_data=workflow_data
            )
            
            # Store workflow (simulated)
            logger.info(f"Created workflow for agent: {agent_id}")
            return workflow
        
        @self.app.get("/api/v1/agents/{agent_id}/workflow", response_model=AgentWorkflow)
        async def get_workflow(agent_id: str):
            """Get agent workflow"""
            # Simulated retrieval
            workflow = AgentWorkflow(
                agent_id=agent_id,
                workflow_data={
                    "nodes": [],
                    "edges": [],
                    "metadata": {}
                }
            )
            return workflow
    
    # Storage methods (simulated for now)
    _agents_storage: Dict[str, Agent] = {}
    
    async def _store_agent(self, agent: Agent):
        """Store agent in database"""
        self._agents_storage[agent.id] = agent
        
        # Cache in Redis if available
        if self.redis_client:
            await self.redis_client.set(
                f"agent:{agent.id}",
                agent.json(),
                ex=3600
            )
    
    async def _get_agent(self, agent_id: str) -> Optional[Agent]:
        """Retrieve agent from database"""
        return self._agents_storage.get(agent_id)
    
    async def _get_agents(self, org_id: str, status: Optional[AgentStatus] = None) -> List[Agent]:
        """Retrieve agents for organization"""
        agents = [
            agent for agent in self._agents_storage.values()
            if agent.organization_id == org_id
        ]
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        return agents
    
    async def _delete_agent(self, agent_id: str):
        """Delete agent from database"""
        if agent_id in self._agents_storage:
            del self._agents_storage[agent_id]


def create_service():
    """Create and return the service instance"""
    return AgentDesignerService()


if __name__ == "__main__":
    import uvicorn
    
    print("Agent Designer Microservice")
    print("=" * 60)
    
    service = create_service()
    
    print("\n📦 Starting Agent Designer Service on port 8001")
    print("\nEndpoints:")
    print("  • GET  /health - Health check")
    print("  • POST /api/v1/agents - Create agent")
    print("  • GET  /api/v1/agents - List agents")
    print("  • GET  /api/v1/agents/{id} - Get agent")
    print("  • PUT  /api/v1/agents/{id} - Update agent")
    print("  • DEL  /api/v1/agents/{id} - Delete agent")
    print("  • POST /api/v1/agents/{id}/deploy - Deploy agent")
    print("  • GET  /api/v1/templates - List templates")
    
    uvicorn.run(service.app, host="0.0.0.0", port=8001, reload=False)