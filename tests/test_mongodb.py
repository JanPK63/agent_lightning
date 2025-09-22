"""
Unit tests for MongoDB operations in Agent Lightning
"""

import pytest
from unittest.mock import patch, MagicMock

# Test MongoDB functionality
try:
    from shared.mongodb import MongoDBManager, DocumentStorage
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


@pytest.mark.skipif(not MONGODB_AVAILABLE, reason="MongoDB dependencies not available")
class TestMongoDBManager:
    """Test MongoDB manager functionality"""

    def test_mongodb_manager_init(self):
        """Test MongoDB manager initialization"""
        manager = MongoDBManager()
        assert manager.database_url is not None
        assert manager.database_name is not None
        assert manager.client is None
        assert manager.database is None

    @patch('shared.mongodb.MongoClient')
    def test_connect_success(self, mock_mongo_client):
        """Test successful MongoDB connection"""
        # Mock the client and database
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.admin.command.return_value = {"ok": 1}
        mock_client.__getitem__.return_value = mock_db
        mock_mongo_client.return_value = mock_client

        manager = MongoDBManager()
        result = manager.connect()

        assert result is True
        assert manager.client is not None
        assert manager.database is not None
        mock_mongo_client.assert_called_once()
        mock_client.admin.command.assert_called_once_with('ping')

    @patch('shared.mongodb.MongoClient')
    def test_connect_failure(self, mock_mongo_client):
        """Test MongoDB connection failure"""
        mock_mongo_client.side_effect = Exception("Connection failed")

        manager = MongoDBManager()
        result = manager.connect()

        assert result is False
        assert manager.client is None
        assert manager.database is None

    def test_get_collection_without_connection(self):
        """Test getting collection without connection"""
        manager = MongoDBManager()
        collection = manager.get_collection("test")
        assert collection is None

    @patch('shared.mongodb.MongoClient')
    def test_get_collection_with_connection(self, mock_mongo_client):
        """Test getting collection with active connection"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_client.admin.command.return_value = {"ok": 1}
        mock_db.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_db
        mock_mongo_client.return_value = mock_client

        manager = MongoDBManager()
        manager.connect()

        collection = manager.get_collection("test")
        assert collection is not None
        mock_db.__getitem__.assert_called_with("test")

    def test_health_check_without_connection(self):
        """Test health check without connection"""
        manager = MongoDBManager()
        result = manager.health_check()
        assert result is False

    @patch('shared.mongodb.MongoClient')
    def test_health_check_with_connection(self, mock_mongo_client):
        """Test health check with active connection"""
        mock_client = MagicMock()
        mock_client.admin.command.return_value = {"ok": 1}
        mock_mongo_client.return_value = mock_client

        manager = MongoDBManager()
        manager.connect()

        result = manager.health_check()
        assert result is True


@pytest.mark.skipif(
    not MONGODB_AVAILABLE,
    reason="MongoDB dependencies not available"
)
class TestDocumentStorage:
    """Test document storage operations"""

    def test_document_storage_init(self):
        """Test document storage initialization"""
        manager = MongoDBManager()
        storage = DocumentStorage(manager)
        assert storage.manager is manager

    def test_insert_document_without_connection(self):
        """Test inserting document without connection"""
        manager = MongoDBManager()
        storage = DocumentStorage(manager)

        result = storage.insert_document("test", {"key": "value"})
        assert result is None

    @patch('shared.mongodb.MongoClient')
    def test_insert_document_success(self, mock_mongo_client):
        """Test successful document insertion"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.inserted_id = "test_id"

        mock_client.admin.command.return_value = {"ok": 1}
        mock_db.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_db
        mock_collection.insert_one.return_value = mock_result
        mock_mongo_client.return_value = mock_client

        manager = MongoDBManager()
        manager.connect()
        storage = DocumentStorage(manager)

        result = storage.insert_document("test", {"key": "value"})
        assert result == "test_id"
        mock_collection.insert_one.assert_called_once_with({"key": "value"})

    @patch('shared.mongodb.MongoClient')
    def test_find_documents(self, mock_mongo_client):
        """Test finding documents"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = [{"id": "1"}, {"id": "2"}]

        mock_client.admin.command.return_value = {"ok": 1}
        mock_db.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_db
        mock_collection.find.return_value = mock_cursor
        mock_mongo_client.return_value = mock_client

        manager = MongoDBManager()
        manager.connect()
        storage = DocumentStorage(manager)

        result = storage.find_documents("test", {"status": "active"})
        assert len(result) == 2
        mock_collection.find.assert_called_once_with(
            {"status": "active"}, limit=100
        )

    @patch('shared.mongodb.MongoClient')
    def test_find_document(self, mock_mongo_client):
        """Test finding a single document"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()

        mock_client.admin.command.return_value = {"ok": 1}
        mock_db.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_db
        mock_collection.find_one.return_value = {"id": "1", "name": "test"}
        mock_mongo_client.return_value = mock_client

        manager = MongoDBManager()
        manager.connect()
        storage = DocumentStorage(manager)

        result = storage.find_document("test", {"id": "1"})
        assert result == {"id": "1", "name": "test"}
        mock_collection.find_one.assert_called_once_with({"id": "1"})

    @patch('shared.mongodb.MongoClient')
    def test_update_document_success(self, mock_mongo_client):
        """Test successful document update"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.modified_count = 1

        mock_client.admin.command.return_value = {"ok": 1}
        mock_db.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_db
        mock_collection.update_one.return_value = mock_result
        mock_mongo_client.return_value = mock_client

        manager = MongoDBManager()
        manager.connect()
        storage = DocumentStorage(manager)

        result = storage.update_document(
            "test", {"id": "1"}, {"status": "updated"}
        )
        assert result is True
        mock_collection.update_one.assert_called_once_with(
            {"id": "1"}, {"$set": {"status": "updated"}}
        )

    @patch('shared.mongodb.MongoClient')
    def test_delete_document_success(self, mock_mongo_client):
        """Test successful document deletion"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.deleted_count = 1

        mock_client.admin.command.return_value = {"ok": 1}
        mock_db.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_db
        mock_collection.delete_one.return_value = mock_result
        mock_mongo_client.return_value = mock_client

        manager = MongoDBManager()
        manager.connect()
        storage = DocumentStorage(manager)

        result = storage.delete_document("test", {"id": "1"})
        assert result is True
        mock_collection.delete_one.assert_called_once_with({"id": "1"})

    @patch('shared.mongodb.MongoClient')
    def test_create_index(self, mock_mongo_client):
        """Test index creation"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()

        mock_client.admin.command.return_value = {"ok": 1}
        mock_db.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_db
        mock_collection.create_index.return_value = "index_name"
        mock_mongo_client.return_value = mock_client

        manager = MongoDBManager()
        manager.connect()
        storage = DocumentStorage(manager)

        result = storage.create_index("test", ["field1", "field2"])
        assert result is True
        mock_collection.create_index.assert_called_once_with(
            [("field1", 1), ("field2", 1)], unique=False
        )


@pytest.mark.skipif(not MONGODB_AVAILABLE, reason="MongoDB dependencies not available")
class TestMongoDBIntegration:
    """Test MongoDB integration with data access layer"""

    @patch('shared.mongodb_dal.MongoDBDataAccessLayer')
    @patch('shared.data_access.DataAccessLayer')
    def test_create_data_access_layer_sql(self, mock_sql_dal, mock_mongo_dal):
        """Test creating SQL data access layer"""
        from shared.data_access import create_data_access_layer

        with patch.dict('os.environ', {'DATABASE_URL': 'sqlite:///test.db'}):
            result = create_data_access_layer("test")
            mock_sql_dal.assert_called_once_with("test")
            assert result is not None

    @patch('shared.mongodb_dal.MongoDBDataAccessLayer')
    @patch('shared.data_access.DataAccessLayer')
    def test_create_data_access_layer_mongodb(
        self, mock_sql_dal, mock_mongo_dal
    ):
        """Test creating MongoDB data access layer"""
        from shared.data_access import create_data_access_layer

        with patch.dict(
            'os.environ',
            {'DATABASE_URL': 'mongodb://localhost:27017/test'}
        ):
            result = create_data_access_layer("test")
            mock_mongo_dal.assert_called_once_with("test")
            assert result is not None

    def test_is_mongodb_available(self):
        """Test MongoDB availability check"""
        from shared.mongodb import is_mongodb_available
        # This will depend on whether pymongo is installed
        result = is_mongodb_available()
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__])