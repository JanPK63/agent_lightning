#!/usr/bin/env python3
"""
Comprehensive tests for the Knowledge Manager Service
"""

import pytest
from agentlightning.knowledge_client import (
    KnowledgeClient,
    KnowledgeItem,
    KnowledgeQuery,
    create_knowledge_item,
    create_knowledge_query,
    quick_store,
    quick_search,
    quick_context
)


class TestKnowledgeClient:
    """Test the KnowledgeClient functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        # Use a mock/test URL that won't actually connect
        self.client = KnowledgeClient(base_url="http://test-server:8014")

    def test_create_knowledge_item(self):
        """Test creating knowledge items"""
        item = create_knowledge_item(
            agent_id="test_agent",
            content="This is test knowledge",
            category="test",
            source="test_source",
            tags=["tag1", "tag2"]
        )

        assert item.agent_id == "test_agent"
        assert item.content == "This is test knowledge"
        assert item.category == "test"
        assert item.source == "test_source"
        assert item.tags == ["tag1", "tag2"]
        assert item.metadata == {}

    def test_create_knowledge_query(self):
        """Test creating knowledge queries"""
        query = create_knowledge_query(
            query="test query",
            agent_id="test_agent",
            category="test_category",
            limit=10,
            min_relevance=0.5
        )

        assert query.query == "test query"
        assert query.agent_id == "test_agent"
        assert query.category == "test_category"
        assert query.limit == 10
        assert query.min_relevance == 0.5

    def test_knowledge_item_dataclass(self):
        """Test KnowledgeItem dataclass functionality"""
        item = KnowledgeItem(
            id="test_id",
            agent_id="test_agent",
            content="Test content",
            category="test",
            source="test_source",
            metadata={"key": "value"},
            tags=["tag1"]
        )

        assert item.id == "test_id"
        assert item.agent_id == "test_agent"
        assert item.content == "Test content"
        assert item.category == "test"
        assert item.source == "test_source"
        assert item.metadata == {"key": "value"}
        assert item.tags == ["tag1"]


class TestKnowledgeServiceIntegration:
    """Integration tests for knowledge service (requires running service)"""

    def test_store_knowledge(self):
        """Test storing knowledge via client"""
        client = KnowledgeClient()

        item = create_knowledge_item(
            agent_id="test_agent",
            content="Integration test knowledge",
            category="integration_test"
        )

        result = client.store_knowledge(item)
        assert "status" in result
        assert result["status"] == "success"

    def test_search_knowledge(self):
        """Test searching knowledge via client"""
        client = KnowledgeClient()

        query = create_knowledge_query(
            query="integration test",
            agent_id="test_agent",
            limit=5
        )

        results = client.search_knowledge(query)
        assert isinstance(results, list)

    def test_get_context_for_task(self):
        """Test getting context for a task"""
        client = KnowledgeClient()

        context = client.get_context_for_task(
            agent_id="test_agent",
            task_description="Test task description",
            limit=3
        )

        assert isinstance(context, list)

    def test_quick_functions(self):
        """Test quick access functions"""
        # Test quick_store
        result = quick_store(
            agent_id="test_agent",
            content="Quick test content",
            category="quick_test"
        )
        assert "status" in result

        # Test quick_search
        results = quick_search("quick test", agent_id="test_agent")
        assert isinstance(results, list)

        # Test quick_context
        context = quick_context("test_agent", "Test task")
        assert isinstance(context, list)


class TestKnowledgeServiceMock:
    """Mock tests that don't require running service"""

    def test_client_initialization(self):
        """Test client initialization with different parameters"""
        # Default initialization
        client1 = KnowledgeClient()
        assert client1.base_url == "http://localhost:8014"
        assert client1.timeout == 30

        # Custom initialization
        client2 = KnowledgeClient(
            base_url="http://custom-server:9000",
            timeout=60
        )
        assert client2.base_url == "http://custom-server:9000"
        assert client2.timeout == 60

    def test_knowledge_item_creation_edge_cases(self):
        """Test knowledge item creation with edge cases"""
        # Empty content
        item1 = KnowledgeItem(agent_id="test", content="", category="test")
        assert item1.content == ""

        # None values
        item2 = KnowledgeItem(
            agent_id="test",
            content="test",
            category="test",
            source=None,
            metadata=None,
            tags=None
        )
        assert item2.source is None
        assert item2.metadata == {}
        assert item2.tags == []

        # Large content
        large_content = "x" * 10000
        item3 = KnowledgeItem(
            agent_id="test",
            content=large_content,
            category="test"
        )
        assert len(item3.content) == 10000

    def test_query_creation_edge_cases(self):
        """Test query creation with edge cases"""
        # Empty query
        query1 = KnowledgeQuery(query="")
        assert query1.query == ""

        # Zero limit
        query2 = KnowledgeQuery(query="test", limit=0)
        assert query2.limit == 0

        # Negative relevance
        query3 = KnowledgeQuery(query="test", min_relevance=-1.0)
        assert query3.min_relevance == -1.0

        # Very high relevance
        query4 = KnowledgeQuery(query="test", min_relevance=2.0)
        assert query4.min_relevance == 2.0


if __name__ == "__main__":
    pytest.main([__file__])