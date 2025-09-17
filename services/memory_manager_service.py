#!/usr/bin/env python3
"""
Enterprise Memory Manager Service
Handles shared memory, persistence, and cross-agent memory coordination
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import asyncio
from datetime import datetime

app = FastAPI(title="Memory Manager Service", version="1.0.0")

class MemoryEntry(BaseModel):
    agent_id: str
    memory_type: str
    content: dict
    timestamp: str = None
    ttl: int = 3600  # Time to live in seconds

class MemoryQuery(BaseModel):
    agent_id: str = None
    memory_type: str = None
    limit: int = 100

# Enterprise memory storage
memory_store = {}
shared_memory = {}
agent_contexts = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "memory-manager"}

@app.post("/memory/store")
async def store_memory(entry: MemoryEntry):
    if not entry.timestamp:
        entry.timestamp = datetime.now().isoformat()
    
    memory_id = f"{entry.agent_id}_{entry.memory_type}_{len(memory_store)}"
    memory_store[memory_id] = entry.dict()
    
    # Update agent context
    if entry.agent_id not in agent_contexts:
        agent_contexts[entry.agent_id] = []
    agent_contexts[entry.agent_id].append(memory_id)
    
    return {"memory_id": memory_id, "status": "stored"}

@app.post("/memory/query")
async def query_memory(query: MemoryQuery):
    results = []
    
    for memory_id, memory in memory_store.items():
        if query.agent_id and memory["agent_id"] != query.agent_id:
            continue
        if query.memory_type and memory["memory_type"] != query.memory_type:
            continue
        
        results.append({"id": memory_id, **memory})
        
        if len(results) >= query.limit:
            break
    
    return {"results": results, "count": len(results)}

@app.post("/memory/shared/store")
async def store_shared_memory(key: str, value: dict):
    shared_memory[key] = {
        "value": value,
        "timestamp": datetime.now().isoformat(),
        "access_count": 0
    }
    return {"status": "stored", "key": key}

@app.get("/memory/shared/{key}")
async def get_shared_memory(key: str):
    if key not in shared_memory:
        raise HTTPException(status_code=404, detail="Shared memory not found")
    
    shared_memory[key]["access_count"] += 1
    return shared_memory[key]

@app.get("/memory/agent/{agent_id}/context")
async def get_agent_context(agent_id: str):
    if agent_id not in agent_contexts:
        return {"agent_id": agent_id, "memories": [], "count": 0}
    
    memories = []
    for memory_id in agent_contexts[agent_id]:
        if memory_id in memory_store:
            memories.append(memory_store[memory_id])
    
    return {"agent_id": agent_id, "memories": memories, "count": len(memories)}

@app.delete("/memory/cleanup")
async def cleanup_expired_memory():
    """Clean up expired memory entries"""
    current_time = datetime.now()
    expired_count = 0
    
    # This would implement TTL cleanup in production
    # For now, just return success
    
    return {"status": "completed", "expired_count": expired_count}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)