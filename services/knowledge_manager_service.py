#!/usr/bin/env python3
"""
Enterprise Knowledge Manager Service
Handles knowledge storage, retrieval, embeddings, and training for Agent Lightning
"""

import os
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
# Temporarily disable sentence-transformers due to AVX compatibility issues
# try:
#     from sentence_transformers import SentenceTransformer
#     import numpy as np
#     SENTENCE_TRANSFORMERS_AVAILABLE = True
# except (ImportError, Exception) as e:
#     logger.warning(f"sentence-transformers not available: {e}")
#     logger.warning("Running without embeddings - semantic search will be limited")
#     SentenceTransformer = None
#     np = None
#     SENTENCE_TRANSFORMERS_AVAILABLE = False

logger.warning("sentence-transformers temporarily disabled due to AVX compatibility issues")
logger.warning("Running without embeddings - semantic search will be limited")
SentenceTransformer = None
np = None
SENTENCE_TRANSFORMERS_AVAILABLE = False

from shared.data_access import DataAccessLayer
from shared.cache import get_cache
from shared.events import EventChannel

# Import input sanitization utilities
from shared.sanitization import InputSanitizer, sanitize_user_input

# Import Prometheus metrics
from monitoring.metrics import get_metrics_collector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Temporarily disable sentence-transformers due to AVX compatibility issues
logger.warning("sentence-transformers temporarily disabled due to AVX compatibility issues")
logger.warning("Running without embeddings - semantic search will be limited")
SentenceTransformer = None
np = None
SENTENCE_TRANSFORMERS_AVAILABLE = False

app = FastAPI(title="Knowledge Manager Service", version="2.0.0")

# Initialize components
dal = DataAccessLayer("knowledge_manager")
cache = get_cache()

# Initialize Prometheus metrics collector
metrics_collector = get_metrics_collector("knowledge_manager")

# Initialize embedding model (disabled for now)
embedding_model = None

# Pydantic models
class KnowledgeItem(BaseModel):
    id: Optional[str] = None
    agent_id: str
    content: str
    category: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

class KnowledgeQuery(BaseModel):
    query: str
    agent_id: Optional[str] = None
    category: Optional[str] = None
    limit: int = 10
    min_relevance: float = 0.0

class KnowledgeUpdate(BaseModel):
    content: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

class TrainingRequest(BaseModel):
    agent_id: str
    knowledge_items: List[str]
    training_type: str = "incremental"
    epochs: int = 3

class BulkIngestionRequest(BaseModel):
    agent_id: str
    items: List[KnowledgeItem]
    process_embeddings: bool = True

class TaskContextRequest(BaseModel):
    agent_id: str
    task_description: str
    limit: int = 5

# Knowledge Manager Service Class
class KnowledgeManagerService:
    """Enhanced knowledge management with embeddings and DAL integration"""

    def __init__(self):
        self.embedding_model = embedding_model
        self.embedding_dimension = 384 if embedding_model else 0

        # Initialize input sanitizer
        self.sanitizer = InputSanitizer()

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using sentence transformer"""
        if not self.embedding_model:
            logger.warning("Embedding model not available")
            return None

        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            a_np = np.array(a)
            b_np = np.array(b)
            return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

    def store_knowledge(self, item: KnowledgeItem) -> Dict[str, Any]:
        """Store knowledge item with embedding"""
        # Sanitize user input
        sanitized_agent_id = self.sanitizer.sanitize_text(item.agent_id.strip())
        sanitized_category = self.sanitizer.sanitize_text(item.category.strip())
        sanitized_content = self.sanitizer.sanitize_text(item.content.strip())
        sanitized_source = (
            self.sanitizer.sanitize_text(item.source.strip())
            if item.source else None
        )
        sanitized_tags = [
            self.sanitizer.sanitize_text(tag.strip())
            for tag in item.tags if tag.strip()
        ]

        # Generate ID if not provided
        if not item.id:
            item.id = f"{sanitized_agent_id}_{uuid.uuid4().hex[:16]}"

        # Generate embedding
        embedding = None
        if sanitized_content and self.embedding_model:
            embedding = self.generate_embedding(sanitized_content)

        # Prepare data for database
        knowledge_data = {
            'id': item.id,
            'agent_id': sanitized_agent_id,
            'category': sanitized_category,
            'content': sanitized_content,
            'source': sanitized_source,
            'metadata': {
                **item.metadata,
                'tags': sanitized_tags,
                'embedding': embedding,
                'embedding_model': 'all-MiniLM-L6-v2' if embedding else None
            }
        }

        # Store in database
        result = dal.add_knowledge(sanitized_agent_id, knowledge_data)

        # Cache the result
        cache_key = f"knowledge:{item.agent_id}:{item.id}"
        cache.set(cache_key, result, ttl=21600)  # 6 hours

        # Emit event
        dal.event_bus.emit(EventChannel.KNOWLEDGE_ADDED, {
            "agent_id": item.agent_id,
            "knowledge_id": item.id,
            "category": item.category
        })

        logger.info(f"Stored knowledge {item.id} for agent {item.agent_id}")
        return result

    def get_knowledge(self, knowledge_id: str, agent_id: Optional[str] = None) -> Optional[Dict]:
        """Retrieve knowledge item with caching"""
        # Try cache first
        cache_key = f"knowledge:{agent_id}:{knowledge_id}" if agent_id else f"knowledge:*:{knowledge_id}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        # Query database
        # If agent_id is provided, search within that agent's knowledge
        if agent_id:
            knowledge_items = dal.get_agent_knowledge(agent_id)
            for item in knowledge_items:
                if item['id'] == knowledge_id:
                    # Cache and return
                    cache.set(cache_key, item, ttl=21600)
                    return item
        else:
            # If no agent_id provided, we need to search across all agents
            # This is a simplified implementation - in production you'd have a global index
            # For now, we'll try to extract agent_id from the knowledge_id format
            if '_' in knowledge_id:
                potential_agent_id = knowledge_id.split('_')[0]
                knowledge_items = dal.get_agent_knowledge(potential_agent_id)
                for item in knowledge_items:
                    if item['id'] == knowledge_id:
                        # Cache and return
                        cache.set(cache_key, item, ttl=21600)
                        return item

        return None

    def search_knowledge(self, query: KnowledgeQuery) -> List[Dict]:
        """Search knowledge using semantic similarity"""
        # Sanitize user input
        sanitized_query = (
            self.sanitizer.sanitize_text(query.query.strip())
            if query.query else None
        )
        sanitized_agent_id = (
            self.sanitizer.sanitize_text(query.agent_id.strip())
            if query.agent_id else None
        )
        sanitized_category = (
            self.sanitizer.sanitize_text(query.category.strip())
            if query.category else None
        )

        query_embedding = None
        if self.embedding_model and sanitized_query:
            query_embedding = self.generate_embedding(sanitized_query)

        results = []

        if sanitized_agent_id:
            # Search specific agent's knowledge
            knowledge_items = dal.get_agent_knowledge(sanitized_agent_id, sanitized_category)
        else:
            # Search all knowledge (simplified - in production you'd need better indexing)
            # For now, we'll just return recent items
            knowledge_items = []
            # This would need to be implemented with proper cross-agent search

        # Score and rank results
        scored_results = []
        for item in knowledge_items:
            score = 0.0

            # Semantic similarity score
            if query_embedding and item.get('metadata', {}).get('embedding'):
                similarity = self.cosine_similarity(
                    query_embedding,
                    item['metadata']['embedding']
                )
                score += similarity * 0.7

            # Text matching score
            if sanitized_query and sanitized_query.lower() in item['content'].lower():
                score += 0.3

            # Relevance score from database
            score += item.get('relevance_score', 0.0) * 0.1

            if score >= query.min_relevance:
                scored_results.append((item, score))

        # Sort by score and return top results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        results = [item for item, score in scored_results[:query.limit]]

        # Update usage counts for retrieved items
        for item in results:
            dal.update_knowledge_usage(item['id'])

        return results

    def update_knowledge(self, knowledge_id: str, agent_id: str, updates: KnowledgeUpdate) -> Optional[Dict]:
        """Update knowledge item"""
        # Get current item
        current = self.get_knowledge(knowledge_id, agent_id)
        if not current:
            return None

        # Apply updates
        update_data = {}
        if updates.content is not None:
            update_data['content'] = updates.content
            # Regenerate embedding if content changed
            if self.embedding_model:
                embedding = self.generate_embedding(updates.content)
                current['metadata']['embedding'] = embedding

        if updates.category is not None:
            update_data['category'] = updates.category

        if updates.metadata is not None:
            current['metadata'].update(updates.metadata)

        if updates.tags is not None:
            current['metadata']['tags'] = updates.tags

        # Update in database (simplified - would need proper update method in DAL)
        # For now, we'll recreate the item
        updated_item = {
            **current,
            **update_data,
            'metadata': current['metadata']
        }

        # Update cache
        cache_key = f"knowledge:{agent_id}:{knowledge_id}"
        cache.set(cache_key, updated_item, ttl=21600)

        # Emit event
        dal.event_bus.emit(EventChannel.KNOWLEDGE_UPDATED, {
            "agent_id": agent_id,
            "knowledge_id": knowledge_id
        })

        return updated_item

    def delete_knowledge(self, knowledge_id: str, agent_id: str) -> bool:
        """Delete knowledge item"""
        # Clear cache
        cache_key = f"knowledge:{agent_id}:{knowledge_id}"
        cache.delete(cache_key)

        # Emit event
        dal.event_bus.emit(EventChannel.KNOWLEDGE_DELETED, {
            "agent_id": agent_id,
            "knowledge_id": knowledge_id
        })

        # Note: Database deletion would need to be implemented in DAL
        logger.info(f"Deleted knowledge {knowledge_id} for agent {agent_id}")
        return True

    def bulk_ingest(self, request: BulkIngestionRequest) -> Dict[str, Any]:
        """Bulk ingest knowledge items"""
        results = []
        success_count = 0
        error_count = 0

        for item in request.items:
            try:
                result = self.store_knowledge(item)
                results.append({"id": item.id, "status": "success"})
                success_count += 1
            except Exception as e:
                results.append({"id": item.id, "status": "error", "error": str(e)})
                error_count += 1

        return {
            "total": len(request.items),
            "successful": success_count,
            "failed": error_count,
            "results": results
        }

    def get_context_for_task(self, agent_id: str, task_description: str, limit: int = 5) -> List[Dict]:
        """Get relevant knowledge context for a task"""
        # Sanitize user input
        sanitized_agent_id = self.sanitizer.sanitize_text(agent_id.strip())
        sanitized_task_description = self.sanitizer.sanitize_text(task_description.strip())

        query = KnowledgeQuery(
            query=sanitized_task_description,
            agent_id=sanitized_agent_id,
            limit=limit,
            min_relevance=0.1
        )

        return self.search_knowledge(query)

# Initialize service
knowledge_service = KnowledgeManagerService()

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health = dal.health_check()
        metrics_collector.increment_request("health", "GET", "200")
        return {
            "status": "healthy" if all(health.values()) else "degraded",
            "service": "knowledge-manager",
            "components": health,
            "embedding_model": embedding_model is not None
        }
    except Exception as e:
        metrics_collector.increment_error("health", type(e).__name__)
        raise

@app.post("/knowledge")
async def store_knowledge(item: KnowledgeItem):
    """Store a knowledge item"""
    try:
        result = knowledge_service.store_knowledge(item)
        metrics_collector.increment_request("store_knowledge", "POST", "200")
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Failed to store knowledge: {e}")
        metrics_collector.increment_error("store_knowledge", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/{knowledge_id}")
async def get_knowledge(knowledge_id: str, agent_id: Optional[str] = Query(None)):
    """Retrieve a knowledge item"""
    try:
        result = knowledge_service.get_knowledge(knowledge_id, agent_id)
        if not result:
            metrics_collector.increment_request("get_knowledge", "GET", "404")
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        metrics_collector.increment_request("get_knowledge", "GET", "200")
        return result
    except HTTPException:
        raise
    except Exception as e:
        metrics_collector.increment_error("get_knowledge", type(e).__name__)
        raise

@app.put("/knowledge/{knowledge_id}")
async def update_knowledge(knowledge_id: str, agent_id: str, updates: KnowledgeUpdate):
    """Update a knowledge item"""
    try:
        result = knowledge_service.update_knowledge(knowledge_id, agent_id, updates)
        if not result:
            metrics_collector.increment_request("update_knowledge", "PUT", "404")
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        metrics_collector.increment_request("update_knowledge", "PUT", "200")
        return {"status": "success", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        metrics_collector.increment_error("update_knowledge", type(e).__name__)
        raise

@app.delete("/knowledge/{knowledge_id}")
async def delete_knowledge(knowledge_id: str, agent_id: str):
    """Delete a knowledge item"""
    try:
        success = knowledge_service.delete_knowledge(knowledge_id, agent_id)
        if not success:
            metrics_collector.increment_request("delete_knowledge", "DELETE", "404")
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        metrics_collector.increment_request("delete_knowledge", "DELETE", "200")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        metrics_collector.increment_error("delete_knowledge", type(e).__name__)
        raise

@app.post("/knowledge/search")
async def search_knowledge(query: KnowledgeQuery):
    """Search knowledge using semantic similarity"""
    try:
        results = knowledge_service.search_knowledge(query)
        metrics_collector.increment_request("search_knowledge", "POST", "200")
        return {"status": "success", "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        metrics_collector.increment_error("search_knowledge", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/bulk")
async def bulk_ingest(request: BulkIngestionRequest, background_tasks: BackgroundTasks):
    """Bulk ingest knowledge items"""
    try:
        # Run in background for large ingests
        if len(request.items) > 10:
            background_tasks.add_task(knowledge_service.bulk_ingest, request)
            metrics_collector.increment_request("bulk_ingest", "POST", "200")
            return {"status": "processing", "message": "Bulk ingestion started in background"}
        else:
            result = knowledge_service.bulk_ingest(request)
            metrics_collector.increment_request("bulk_ingest", "POST", "200")
            return {"status": "completed", "data": result}
    except Exception as e:
        logger.error(f"Bulk ingestion failed: {e}")
        metrics_collector.increment_error("bulk_ingest", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge")
async def list_knowledge(
    agent_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, le=200)
):
    """List knowledge items"""
    try:
        if agent_id:
            results = dal.get_agent_knowledge(agent_id, category)
            metrics_collector.increment_request("list_knowledge", "GET", "200")
            return {"status": "success", "results": results[:limit], "count": len(results)}
        else:
            # Simplified - in production you'd need proper cross-agent listing
            metrics_collector.increment_request("list_knowledge", "GET", "200")
            return {"status": "success", "results": [], "count": 0, "message": "Agent ID required"}
    except Exception as e:
        logger.error(f"List knowledge failed: {e}")
        metrics_collector.increment_error("list_knowledge", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context/task")
async def get_task_context(request: TaskContextRequest):
    """Get relevant knowledge context for a task"""
    try:
        context = knowledge_service.get_context_for_task(
            request.agent_id,
            request.task_description,
            request.limit
        )
        metrics_collector.increment_request("get_task_context", "POST", "200")
        return {"status": "success", "context": context, "count": len(context)}
    except Exception as e:
        logger.error(f"Task context retrieval failed: {e}")
        metrics_collector.increment_error("get_task_context", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start knowledge training for an agent"""
    try:
        # This would integrate with the RL training system
        background_tasks.add_task(simulate_training, request.agent_id, request.knowledge_items, request.epochs)

        session_id = f"training_{request.agent_id}_{uuid.uuid4().hex[:8]}"
        metrics_collector.increment_request("start_training", "POST", "200")
        return {
            "status": "started",
            "session_id": session_id,
            "message": f"Training started for agent {request.agent_id}"
        }
    except Exception as e:
        logger.error(f"Training start failed: {e}")
        metrics_collector.increment_error("start_training", type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/{session_id}")
async def get_training_status(session_id: str):
    """Get training status (simplified)"""
    try:
        # In production, this would track actual training sessions
        metrics_collector.increment_request("get_training_status", "GET", "200")
        return {
            "session_id": session_id,
            "status": "completed",
            "progress": 100,
            "message": "Training completed successfully"
        }
    except Exception as e:
        metrics_collector.increment_error("get_training_status", type(e).__name__)
        raise

async def simulate_training(agent_id: str, knowledge_items: List[str], epochs: int):
    """Simulate knowledge training process"""
    logger.info(f"Starting training for agent {agent_id} with {len(knowledge_items)} items")

    # Simulate training progress
    for epoch in range(epochs):
        logger.info(f"Training epoch {epoch + 1}/{epochs} for agent {agent_id}")
        await asyncio.sleep(2)  # Simulate training time

    logger.info(f"Training completed for agent {agent_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8014)