"""
Database connection management for Agent Lightning
Handles PostgreSQL connections with connection pooling

DEPRECATED: This module is deprecated. Use shared.database instead,
which supports both SQLite and PostgreSQL based on DATABASE_URL environment variable.
"""

import os
import warnings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging

# Issue deprecation warning
warnings.warn(
    "database_connection module is deprecated. Use shared.database instead, "
    "which supports both SQLite and PostgreSQL based on DATABASE_URL.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv(
            "DATABASE_URL", 
            "postgresql://localhost/agent_lightning"
        )
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def health_check(self):
        """Check database connectivity"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()

# Convenience function for getting sessions
def get_db_session():
    """Get database session - use with context manager"""
    return db_manager.get_session()