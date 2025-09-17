"""
Knowledge Manager Service - FastAPI Server
Provides REST API for knowledge management across all agents
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import asyncio
import sys
import os

# Import the KnowledgeManager class
sys.path.append('/app')
from knowledge_manager import KnowledgeManager

app = FastAPI(title="Knowledge Manager Service", version="1.0.0")
km = KnowledgeManager()

class KnowledgeRequest(BaseModel):
    agent_name: str
    category: str
    content: str
    source: str = "api"
    metadata: Optional[dict] = None

class SearchRequest(BaseModel):
    agent_name: str
    query: str
    category: Optional[str] = None
    limit: int = 10

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "knowledge-manager", "port": 8014}

@app.post("/knowledge")
async def add_knowledge(request: KnowledgeRequest):
    try:
        item = km.add_knowledge(
            agent_name=request.agent_name,
            category=request.category,
            content=request.content,
            source=request.source,
            metadata=request.metadata
        )
        return {"status": "success", "item_id": item.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_knowledge(request: SearchRequest):
    try:
        results = km.search_knowledge(
            agent_name=request.agent_name,
            query=request.query,
            category=request.category,
            limit=request.limit
        )
        return {
            "status": "success",
            "results": [{
                "id": item.id,
                "category": item.category,
                "content": item.content,
                "source": item.source,
                "usage_count": item.usage_count,
                "relevance_score": item.relevance_score
            } for item in results]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics/{agent_name}")
async def get_statistics(agent_name: str):
    try:
        stats = km.get_statistics(agent_name)
        return {"status": "success", "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_name}/knowledge")
async def get_agent_knowledge(agent_name: str):
    try:
        items = km.knowledge_bases.get(agent_name, [])
        return {
            "status": "success",
            "agent": agent_name,
            "total_items": len(items),
            "knowledge_items": [{
                "id": item.id,
                "category": item.category,
                "content": item.content[:200] + "..." if len(item.content) > 200 else item.content,
                "source": item.source,
                "usage_count": item.usage_count
            } for item in items]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List all agents with knowledge bases"""
    try:
        agents = list(km.knowledge_bases.keys())
        return {
            "status": "success",
            "agents": agents,
            "total_agents": len(agents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def initialize_demo_data():
    """Initialize with demo knowledge"""
    # Add knowledge for full-stack developer
    km.add_knowledge(
        "full_stack_developer",
        "best_practices",
        "Always use parameterized queries to prevent SQL injection attacks",
        "security_guide"
    )
    
    km.add_knowledge(
        "full_stack_developer",
        "code_examples",
        "React hook for API calls: const useApi = (url) => { const [data, setData] = useState(null); ... }",
        "react_patterns"
    )
    
    # Add knowledge for other agents
    km.add_knowledge(
        "devops_engineer",
        "best_practices",
        "Use infrastructure as code for reproducible deployments",
        "devops_guide"
    )
    
    km.add_knowledge(
        "security_expert",
        "best_practices",
        "Implement zero-trust security architecture",
        "security_framework"
    )
    
    print("✅ Knowledge Manager initialized with demo data")

if __name__ == "__main__":
    # Initialize demo data
    asyncio.run(initialize_demo_data())
    
    print("🚀 Starting Knowledge Manager Service on http://0.0.0.0:8014")
    uvicorn.run(app, host="0.0.0.0", port=8014)