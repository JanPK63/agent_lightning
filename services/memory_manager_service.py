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

# Import Prometheus metrics
from monitoring.metrics import get_metrics_collector

# Initialize metrics collector
metrics_collector = get_metrics_collector("memory_manager")

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
    """Health check endpoint"""
    try:
        metrics_collector.increment_request("health", "GET", "200")
        return {"status": "healthy", "service": "memory-manager"}
    except Exception as e:
        metrics_collector.increment_error("health", type(e).__name__)
        raise

@app.post("/memory/store")
async def store_memory(entry: MemoryEntry):
    """Store memory entry"""
    try:
        if not entry.timestamp:
            entry.timestamp = datetime.now().isoformat()

        memory_id = f"{entry.agent_id}_{entry.memory_type}_{len(memory_store)}"
        memory_store[memory_id] = entry.dict()

        # Update agent context
        if entry.agent_id not in agent_contexts:
            agent_contexts[entry.agent_id] = []
        agent_contexts[entry.agent_id].append(memory_id)

        metrics_collector.increment_request("store_memory", "POST", "200")
        return {"memory_id": memory_id, "status": "stored"}
    except Exception as e:
        metrics_collector.increment_error("store_memory", type(e).__name__)
        raise

@app.post("/memory/query")
async def query_memory(query: MemoryQuery):
    """Query memory entries"""
    try:
        results = []

        for memory_id, memory in memory_store.items():
            if query.agent_id and memory["agent_id"] != query.agent_id:
                continue
            if query.memory_type and memory["memory_type"] != query.memory_type:
                continue

            results.append({"id": memory_id, **memory})

            if len(results) >= query.limit:
                break

        metrics_collector.increment_request("query_memory", "POST", "200")
        return {"results": results, "count": len(results)}
    except Exception as e:
        metrics_collector.increment_error("query_memory", type(e).__name__)
        raise

@app.post("/memory/shared/store")
async def store_shared_memory(key: str, value: dict):
    """Store shared memory"""
    try:
        shared_memory[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }
        metrics_collector.increment_request("store_shared_memory", "POST", "200")
        return {"status": "stored", "key": key}
    except Exception as e:
        metrics_collector.increment_error("store_shared_memory", type(e).__name__)
        raise

@app.get("/memory/shared/{key}")
async def get_shared_memory(key: str):
    """Get shared memory"""
    try:
        if key not in shared_memory:
            metrics_collector.increment_request("get_shared_memory", "GET", "404")
            raise HTTPException(status_code=404, detail="Shared memory not found")

        shared_memory[key]["access_count"] += 1
        metrics_collector.increment_request("get_shared_memory", "GET", "200")
        return shared_memory[key]
    except HTTPException:
        raise
    except Exception as e:
        metrics_collector.increment_error("get_shared_memory", type(e).__name__)
        raise

@app.get("/memory/agent/{agent_id}/context")
async def get_agent_context(agent_id: str):
    """Get agent context"""
    try:
        if agent_id not in agent_contexts:
            metrics_collector.increment_request("get_agent_context", "GET", "200")
            return {"agent_id": agent_id, "memories": [], "count": 0}

        memories = []
        for memory_id in agent_contexts[agent_id]:
            if memory_id in memory_store:
                memories.append(memory_store[memory_id])

        metrics_collector.increment_request("get_agent_context", "GET", "200")
        return {"agent_id": agent_id, "memories": memories, "count": len(memories)}
    except Exception as e:
        metrics_collector.increment_error("get_agent_context", type(e).__name__)
        raise

@app.delete("/memory/cleanup")
async def cleanup_expired_memory():
    """Clean up expired memory entries"""
    try:
        # This would implement TTL cleanup in production
        # For now, just return success

        metrics_collector.increment_request("cleanup_memory", "DELETE", "200")
        return {"status": "completed", "expired_count": 0}
    except Exception as e:
        metrics_collector.increment_error("cleanup_memory", type(e).__name__)
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)