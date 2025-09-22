"""
MongoDB connection and document storage for Agent Lightning
Provides MongoDB support alongside SQL databases for document storage
"""

import os
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

try:
    from pymongo import MongoClient
    from pymongo.database import Database
    from pymongo.collection import Collection
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    from motor.motor_asyncio import (
        AsyncIOMotorClient,
        AsyncIOMotorDatabase,
        AsyncIOMotorCollection
    )
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    MongoClient = None
    AsyncIOMotorClient = None

logger = logging.getLogger(__name__)


class MongoDBManager:
    """MongoDB connection manager for document storage"""

    def __init__(self):
        if not MONGODB_AVAILABLE:
            raise ImportError(
                "MongoDB dependencies not installed. "
                "Install with: pip install pymongo motor"
            )

        self.database_url = os.getenv(
            "MONGODB_URL",
            "mongodb://localhost:27017/agent_lightning"
        )
        self.database_name = os.getenv("MONGODB_DATABASE", "agent_lightning")

        # Parse database name from URL if provided
        parsed = urlparse(self.database_url)
        if parsed.path and len(parsed.path) > 1:
            self.database_name = parsed.path.lstrip('/')

        self.client: Optional[MongoClient] = None
        self.async_client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[Database] = None
        self.async_database: Optional[AsyncIOMotorDatabase] = None

    def connect(self) -> bool:
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(
                self.database_url,
                serverSelectionTimeoutMS=5000
            )
            # Test the connection
            self.client.admin.command('ping')
            self.database = self.client[self.database_name]
            logger.info(
                f"✅ Connected to MongoDB database: {self.database_name}"
            )
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False

    async def connect_async(self) -> bool:
        """Establish async MongoDB connection"""
        try:
            self.async_client = AsyncIOMotorClient(
                self.database_url,
                serverSelectionTimeoutMS=5000
            )
            # Test the connection
            await self.async_client.admin.command('ping')
            self.async_database = self.async_client[self.database_name]
            logger.info(
                f"✅ Connected to async MongoDB database: {self.database_name}"
            )
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ Async MongoDB connection failed: {e}")
            return False

    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
            logger.info("✅ MongoDB connection closed")

    async def disconnect_async(self):
        """Close async MongoDB connection"""
        if self.async_client:
            self.async_client.close()
            self.async_client = None
            self.async_database = None
            logger.info("✅ Async MongoDB connection closed")

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """Get a MongoDB collection"""
        if not self.database:
            logger.error("MongoDB not connected")
            return None
        return self.database[collection_name]

    def get_async_collection(
        self, collection_name: str
    ) -> Optional[AsyncIOMotorCollection]:
        """Get an async MongoDB collection"""
        if not self.async_database:
            logger.error("Async MongoDB not connected")
            return None
        return self.async_database[collection_name]

    def health_check(self) -> bool:
        """Check MongoDB health"""
        if not self.client:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return False

    async def health_check_async(self) -> bool:
        """Check async MongoDB health"""
        if not self.async_client:
            return False
        try:
            await self.async_client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"Async MongoDB health check failed: {e}")
            return False


class DocumentStorage:
    """Document storage operations for MongoDB"""

    def __init__(self, manager: MongoDBManager):
        self.manager = manager

    def insert_document(
        self, collection_name: str, document: Dict[str, Any]
    ) -> Optional[str]:
        """Insert a document into MongoDB collection"""
        collection = self.manager.get_collection(collection_name)
        if not collection:
            return None

        try:
            result = collection.insert_one(document)
            logger.info(f"Document inserted with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            return None

    def find_documents(
        self,
        collection_name: str,
        query: Dict[str, Any] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find documents in MongoDB collection"""
        collection = self.manager.get_collection(collection_name)
        if not collection:
            return []

        try:
            cursor = collection.find(query or {}, limit=limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"Failed to find documents: {e}")
            return []

    def find_document(
        self, collection_name: str, query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find a single document in MongoDB collection"""
        collection = self.manager.get_collection(collection_name)
        if not collection:
            return None

        try:
            return collection.find_one(query)
        except Exception as e:
            logger.error(f"Failed to find document: {e}")
            return None

    def update_document(
        self,
        collection_name: str,
        query: Dict[str, Any],
        update_data: Dict[str, Any]
    ) -> bool:
        """Update a document in MongoDB collection"""
        collection = self.manager.get_collection(collection_name)
        if not collection:
            return False

        try:
            result = collection.update_one(query, {"$set": update_data})
            success = result.modified_count > 0
            if success:
                logger.info(f"Document updated in {collection_name}")
            return success
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            return False

    def delete_document(
        self, collection_name: str, query: Dict[str, Any]
    ) -> bool:
        """Delete a document from MongoDB collection"""
        collection = self.manager.get_collection(collection_name)
        if not collection:
            return False

        try:
            result = collection.delete_one(query)
            success = result.deleted_count > 0
            if success:
                logger.info(f"Document deleted from {collection_name}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    def create_index(
        self,
        collection_name: str,
        keys: List[str],
        unique: bool = False
    ) -> bool:
        """Create an index on a MongoDB collection"""
        collection = self.manager.get_collection(collection_name)
        if not collection:
            return False

        try:
            index_keys = [(key, 1) for key in keys]
            collection.create_index(index_keys, unique=unique)
            logger.info(f"Index created on {collection_name} for keys: {keys}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False


# Global instances
mongodb_manager = MongoDBManager()
document_storage = DocumentStorage(mongodb_manager)


def init_mongodb() -> bool:
    """Initialize MongoDB connection"""
    return mongodb_manager.connect()


async def init_mongodb_async() -> bool:
    """Initialize async MongoDB connection"""
    return await mongodb_manager.connect_async()


def test_mongodb_connection() -> bool:
    """Test MongoDB connection"""
    return mongodb_manager.health_check()


def is_mongodb_available() -> bool:
    """Check if MongoDB dependencies are available"""
    return MONGODB_AVAILABLE