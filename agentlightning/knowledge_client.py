#!/usr/bin/env python3
"""
Knowledge Client for Agent Lightning
Provides Python client for agents to interact with the Knowledge Manager Service
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeItem:
    """Knowledge item data structure"""
    id: Optional[str] = None
    agent_id: str = ""
    content: str = ""
    category: str = ""
    source: Optional[str] = None
    metadata: Dict[str, Any] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []

@dataclass
class KnowledgeQuery:
    """Query parameters for knowledge search"""
    query: str = ""
    agent_id: Optional[str] = None
    category: Optional[str] = None
    limit: int = 10
    min_relevance: float = 0.0

class KnowledgeClient:
    """
    Client for interacting with the Knowledge Manager Service
    """

    def __init__(self, base_url: str = "http://localhost:8014", timeout: int = 30):
        """
        Initialize the knowledge client

        Args:
            base_url: Base URL of the knowledge service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the service"""
        url = urljoin(self.base_url + '/', endpoint.lstrip('/'))

        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {method} {url} - {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        return self._make_request("GET", "/health")

    def store_knowledge(self, item: KnowledgeItem) -> Dict[str, Any]:
        """Store a knowledge item"""
        data = {
            "id": item.id,
            "agent_id": item.agent_id,
            "content": item.content,
            "category": item.category,
            "source": item.source,
            "metadata": item.metadata,
            "tags": item.tags
        }
        return self._make_request("POST", "/knowledge", json=data)

    def get_knowledge(self, knowledge_id: str, agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a knowledge item"""
        params = {}
        if agent_id:
            params["agent_id"] = agent_id

        try:
            return self._make_request("GET", f"/knowledge/{knowledge_id}", params=params)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def update_knowledge(self, knowledge_id: str, agent_id: str,
                        content: Optional[str] = None,
                        category: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Update a knowledge item"""
        data = {}
        if content is not None:
            data["content"] = content
        if category is not None:
            data["category"] = category
        if metadata is not None:
            data["metadata"] = metadata
        if tags is not None:
            data["tags"] = tags

        return self._make_request("PUT", f"/knowledge/{knowledge_id}?agent_id={agent_id}", json=data)

    def delete_knowledge(self, knowledge_id: str, agent_id: str) -> bool:
        """Delete a knowledge item"""
        try:
            self._make_request("DELETE", f"/knowledge/{knowledge_id}?agent_id={agent_id}")
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return False
            raise

    def search_knowledge(self, query: KnowledgeQuery) -> List[Dict[str, Any]]:
        """Search knowledge using semantic similarity"""
        data = {
            "query": query.query,
            "agent_id": query.agent_id,
            "category": query.category,
            "limit": query.limit,
            "min_relevance": query.min_relevance
        }
        response = self._make_request("POST", "/knowledge/search", json=data)
        return response.get("results", [])

    def list_knowledge(self, agent_id: Optional[str] = None,
                      category: Optional[str] = None,
                      limit: int = 50) -> List[Dict[str, Any]]:
        """List knowledge items"""
        params = {}
        if agent_id:
            params["agent_id"] = agent_id
        if category:
            params["category"] = category
        if limit != 50:
            params["limit"] = limit

        response = self._make_request("GET", "/knowledge", params=params)
        return response.get("results", [])

    def bulk_ingest(self, agent_id: str, items: List[KnowledgeItem],
                   process_embeddings: bool = True) -> Dict[str, Any]:
        """Bulk ingest knowledge items"""
        data = {
            "agent_id": agent_id,
            "items": [
                {
                    "id": item.id,
                    "agent_id": item.agent_id,
                    "content": item.content,
                    "category": item.category,
                    "source": item.source,
                    "metadata": item.metadata,
                    "tags": item.tags
                }
                for item in items
            ],
            "process_embeddings": process_embeddings
        }
        return self._make_request("POST", "/knowledge/bulk", json=data)

    def get_context_for_task(self, agent_id: str, task_description: str,
                           limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant knowledge context for a task"""
        data = {
            "agent_id": agent_id,
            "task_description": task_description,
            "limit": limit
        }
        response = self._make_request("POST", "/context/task", json=data)
        return response.get("context", [])

    def start_training(self, agent_id: str, knowledge_items: List[str],
                      training_type: str = "incremental", epochs: int = 3) -> Dict[str, Any]:
        """Start knowledge training for an agent"""
        data = {
            "agent_id": agent_id,
            "knowledge_items": knowledge_items,
            "training_type": training_type,
            "epochs": epochs
        }
        return self._make_request("POST", "/training/start", json=data)

    def get_training_status(self, session_id: str) -> Dict[str, Any]:
        """Get training status"""
        return self._make_request("GET", f"/training/{session_id}")

# Convenience functions for easy usage
def create_knowledge_item(agent_id: str, content: str, category: str,
                         source: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None) -> KnowledgeItem:
    """Create a KnowledgeItem instance"""
    return KnowledgeItem(
        agent_id=agent_id,
        content=content,
        category=category,
        source=source,
        metadata=metadata or {},
        tags=tags or []
    )

def create_knowledge_query(query: str, agent_id: Optional[str] = None,
                          category: Optional[str] = None,
                          limit: int = 10,
                          min_relevance: float = 0.0) -> KnowledgeQuery:
    """Create a KnowledgeQuery instance"""
    return KnowledgeQuery(
        query=query,
        agent_id=agent_id,
        category=category,
        limit=limit,
        min_relevance=min_relevance
    )

# Global client instance
_default_client = None

def get_knowledge_client(base_url: str = "http://localhost:8014") -> KnowledgeClient:
    """Get or create default knowledge client"""
    global _default_client
    if _default_client is None or _default_client.base_url != base_url:
        _default_client = KnowledgeClient(base_url)
    return _default_client

# Quick access functions using default client
def quick_store(agent_id: str, content: str, category: str,
               source: Optional[str] = None) -> Dict[str, Any]:
    """Quickly store knowledge using default client"""
    client = get_knowledge_client()
    item = create_knowledge_item(agent_id, content, category, source)
    return client.store_knowledge(item)

def quick_search(query: str, agent_id: Optional[str] = None,
                limit: int = 10) -> List[Dict[str, Any]]:
    """Quickly search knowledge using default client"""
    client = get_knowledge_client()
    query_obj = create_knowledge_query(query, agent_id, limit=limit)
    return client.search_knowledge(query_obj)

def quick_context(agent_id: str, task_description: str,
                 limit: int = 5) -> List[Dict[str, Any]]:
    """Quickly get context for a task using default client"""
    client = get_knowledge_client()
    return client.get_context_for_task(agent_id, task_description, limit)