"""
Enterprise Memory Manager Server
FastAPI server for distributed memory management
"""

import asyncio
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise Memory Manager", version="1.0.0")

class MemoryItem(BaseModel):
    id: str
    agent_id: str
    content: Dict[str, Any]
    memory_type: str  # episodic, semantic, procedural
    importance: float = 0.5
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class MemoryStore:
    """Enterprise memory storage"""
    
    def __init__(self):
        self.episodic_memory: Dict[str, List[MemoryItem]] = {}
        self.semantic_memory: Dict[str, List[MemoryItem]] = {}
        self.procedural_memory: Dict[str, List[MemoryItem]] = {}
        self.working_memory: Dict[str, List[MemoryItem]] = {}
        
    def store_memory(self, item: MemoryItem):
        """Store memory item by type"""
        agent_memories = getattr(self, f"{item.memory_type}_memory")
        if item.agent_id not in agent_memories:
            agent_memories[item.agent_id] = []
        agent_memories[item.agent_id].append(item)
        
        # Maintain capacity limits
        if len(agent_memories[item.agent_id]) > 1000:
            agent_memories[item.agent_id] = agent_memories[item.agent_id][-1000:]
    
    def retrieve_memories(self, agent_id: str, memory_type: str, limit: int = 10) -> List[MemoryItem]:
        """Retrieve memories by type and importance"""
        agent_memories = getattr(self, f"{memory_type}_memory")
        memories = agent_memories.get(agent_id, [])
        return sorted(memories, key=lambda x: x.importance, reverse=True)[:limit]
    
    def get_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for agent"""
        return {
            "episodic_count": len(self.episodic_memory.get(agent_id, [])),
            "semantic_count": len(self.semantic_memory.get(agent_id, [])),
            "procedural_count": len(self.procedural_memory.get(agent_id, [])),
            "working_count": len(self.working_memory.get(agent_id, [])),
            "total_memories": sum([
                len(self.episodic_memory.get(agent_id, [])),
                len(self.semantic_memory.get(agent_id, [])),
                len(self.procedural_memory.get(agent_id, [])),
                len(self.working_memory.get(agent_id, []))
            ])
        }

# Global memory store
memory_store = MemoryStore()

class StoreMemoryRequest(BaseModel):
    agent_id: str
    content: Dict[str, Any]
    memory_type: str
    importance: float = 0.5
    metadata: Dict[str, Any] = {}

class RetrieveMemoryRequest(BaseModel):
    agent_id: str
    memory_type: str
    limit: int = 10

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "memory-manager", "port": 8012}

@app.post("/memory/store")
async def store_memory(request: StoreMemoryRequest):
    """Store memory item"""
    try:
        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            agent_id=request.agent_id,
            content=request.content,
            memory_type=request.memory_type,
            importance=request.importance,
            timestamp=datetime.utcnow(),
            metadata=request.metadata
        )
        
        memory_store.store_memory(memory_item)
        
        return {
            "status": "success",
            "memory_id": memory_item.id,
            "stored_at": memory_item.timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/retrieve")
async def retrieve_memories(request: RetrieveMemoryRequest):
    """Retrieve memories by type"""
    try:
        memories = memory_store.retrieve_memories(
            request.agent_id,
            request.memory_type,
            request.limit
        )
        
        return {
            "status": "success",
            "agent_id": request.agent_id,
            "memory_type": request.memory_type,
            "count": len(memories),
            "memories": [
                {
                    "id": mem.id,
                    "content": mem.content,
                    "importance": mem.importance,
                    "timestamp": mem.timestamp,
                    "metadata": mem.metadata
                }
                for mem in memories
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/statistics/{agent_id}")
async def get_memory_statistics(agent_id: str):
    """Get memory statistics for agent"""
    try:
        stats = memory_store.get_statistics(agent_id)
        return {"status": "success", "agent_id": agent_id, "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/agents")
async def list_agents():
    """List all agents with memories"""
    try:
        all_agents = set()
        for memory_type in ['episodic', 'semantic', 'procedural', 'working']:
            agent_memories = getattr(memory_store, f"{memory_type}_memory")
            all_agents.update(agent_memories.keys())
        
        return {
            "status": "success",
            "agents": list(all_agents),
            "total_agents": len(all_agents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/{agent_id}")
async def clear_agent_memory(agent_id: str):
    """Clear all memories for an agent"""
    try:
        for memory_type in ['episodic', 'semantic', 'procedural', 'working']:
            agent_memories = getattr(memory_store, f"{memory_type}_memory")
            if agent_id in agent_memories:
                del agent_memories[agent_id]
        
        return {"status": "success", "message": f"Cleared all memories for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_system_metrics():
    """Get system-wide memory metrics"""
    try:
        total_memories = 0
        agent_count = 0
        memory_types = {}
        
        for memory_type in ['episodic', 'semantic', 'procedural', 'working']:
            agent_memories = getattr(memory_store, f"{memory_type}_memory")
            type_count = sum(len(memories) for memories in agent_memories.values())
            memory_types[memory_type] = type_count
            total_memories += type_count
            agent_count = max(agent_count, len(agent_memories))
        
        return {
            "status": "success",
            "total_memories": total_memories,
            "total_agents": agent_count,
            "memory_by_type": memory_types,
            "system_health": "healthy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🧠 Starting Enterprise Memory Manager on http://0.0.0.0:8012")
    uvicorn.run(app, host="0.0.0.0", port=8012)