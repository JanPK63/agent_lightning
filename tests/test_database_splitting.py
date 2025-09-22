"""
Tests for read/write database splitting functionality
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from shared.database import (
    get_read_session,
    get_write_session,
    db_manager,
    ENABLE_DB_SPLITTING,
    READ_DATABASE_URL,
    WRITE_DATABASE_URL
)


class TestDatabaseSplitting:
    """Test read/write database splitting functionality"""

    def test_default_splitting_disabled(self):
        """Test that database splitting is disabled by default"""
        assert ENABLE_DB_SPLITTING is False

    @patch.dict(os.environ, {
        'ENABLE_DB_SPLITTING': 'true',
        'READ_DATABASE_URL': 'sqlite:///./read.db',
        'WRITE_DATABASE_URL': 'sqlite:///./write.db'
    }, clear=True)
    def test_splitting_enabled_via_env(self):
        """Test enabling database splitting via environment variables"""
        # Reload module to get updated environment
        import importlib
        import shared.database
        importlib.reload(shared.database)

        assert shared.database.ENABLE_DB_SPLITTING is True
        assert shared.database.READ_DATABASE_URL == 'sqlite:///./read.db'
        assert shared.database.WRITE_DATABASE_URL == 'sqlite:///./write.db'

    @patch.dict(os.environ, {
        'ENABLE_DB_SPLITTING': 'true',
        'DATABASE_URL': 'postgresql://user:pass@localhost/main',
        'READ_DATABASE_URL': 'postgresql://user:pass@read-replica:5432/main',
        'WRITE_DATABASE_URL': 'postgresql://user:pass@write-primary:5432/main'
    }, clear=True)
    def test_postgresql_splitting_config(self):
        """Test PostgreSQL read/write splitting configuration"""
        import importlib
        import shared.database
        importlib.reload(shared.database)

        from shared.database import get_database_config_for_url

        read_config = get_database_config_for_url(shared.database.READ_DATABASE_URL)
        write_config = get_database_config_for_url(shared.database.WRITE_DATABASE_URL)

        assert read_config['poolclass'].__name__ == 'QueuePool'
        assert write_config['poolclass'].__name__ == 'QueuePool'
        assert read_config['pool_size'] == 10  # Default POOL_SIZE
        assert write_config['pool_size'] == 10

    @patch.dict(os.environ, {
        'ENABLE_DB_SPLITTING': 'true',
        'READ_DATABASE_URL': 'sqlite:///./read.db',
        'WRITE_DATABASE_URL': 'sqlite:///./write.db'
    }, clear=True)
    def test_session_creation_with_splitting(self):
        """Test that separate sessions are created for read/write when splitting is enabled"""
        import importlib
        import shared.database
        importlib.reload(shared.database)

        from shared.database import get_read_session, get_write_session

        # Should be able to create sessions without errors
        read_session = get_read_session()
        write_session = get_write_session()

        assert read_session is not None
        assert write_session is not None
        assert read_session is not write_session  # Different session instances

    @patch.dict(os.environ, {
        'ENABLE_DB_SPLITTING': 'false',
        'DATABASE_URL': 'sqlite:///./test.db'
    }, clear=True)
    def test_single_database_fallback(self):
        """Test that single database works when splitting is disabled"""
        import importlib
        import shared.database
        importlib.reload(shared.database)

        from shared.database import get_read_session, get_write_session, get_db_session, ReadSessionLocal, WriteSessionLocal, SessionLocal

        # All should return valid sessions when splitting is disabled
        read_session = get_read_session()
        write_session = get_write_session()
        default_session = get_db_session()

        assert read_session is not None
        assert write_session is not None
        assert default_session is not None

        # In single DB mode, all session factories should be the same
        assert ReadSessionLocal is WriteSessionLocal
        assert WriteSessionLocal is SessionLocal

    @patch.dict(os.environ, {
        'ENABLE_DB_SPLITTING': 'true',
        'READ_DATABASE_URL': 'sqlite:///./read.db',
        'WRITE_DATABASE_URL': 'sqlite:///./write.db'
    }, clear=True)
    def test_db_manager_splitting_detection(self):
        """Test that database manager correctly detects splitting status"""
        import importlib
        import shared.database
        importlib.reload(shared.database)

        from shared.database import db_manager

        assert db_manager.is_db_splitting_enabled() is True

    @patch.dict(os.environ, {
        'ENABLE_DB_SPLITTING': 'false',
        'DATABASE_URL': 'sqlite:///./test.db'
    }, clear=True)
    def test_db_manager_no_splitting_detection(self):
        """Test that database manager correctly detects no splitting"""
        import importlib
        import shared.database
        importlib.reload(shared.database)

        from shared.database import db_manager

        assert db_manager.is_db_splitting_enabled() is False

    @patch.dict(os.environ, {
        'ENABLE_DB_SPLITTING': 'true',
        'READ_DATABASE_URL': 'sqlite:///./read.db',
        'WRITE_DATABASE_URL': 'sqlite:///./write.db'
    }, clear=True)
    def test_connection_info_with_splitting(self):
        """Test connection info shows read/write database details when splitting is enabled"""
        import importlib
        import shared.database
        importlib.reload(shared.database)

        from shared.database import db_manager

        info = db_manager.get_connection_info()

        assert info['type'] == 'sql'
        assert info['db_splitting_enabled'] is True
        assert 'read_database' in info
        assert 'write_database' in info
        assert info['read_database']['url'] == 'sqlite:///./read.db'
        assert info['write_database']['url'] == 'sqlite:///./write.db'

    @patch.dict(os.environ, {
        'ENABLE_DB_SPLITTING': 'false',
        'DATABASE_URL': 'sqlite:///./test.db'
    }, clear=True)
    def test_connection_info_without_splitting(self):
        """Test connection info shows single database details when splitting is disabled"""
        import importlib
        import shared.database
        importlib.reload(shared.database)

        from shared.database import db_manager

        info = db_manager.get_connection_info()

        assert info['type'] == 'sql'
        assert info['db_splitting_enabled'] is False
        assert 'read_database' not in info
        assert 'write_database' not in info
        assert 'pool_class' in info
        assert 'pool_size' in info

    @patch.dict(os.environ, {
        'ENABLE_DB_SPLITTING': 'true',
        'READ_DATABASE_URL': 'sqlite:///./read.db',
        'WRITE_DATABASE_URL': 'sqlite:///./write.db'
    }, clear=True)
    def test_pool_stats_with_splitting(self):
        """Test pool stats for both read and write databases when splitting is enabled"""
        import importlib
        import shared.database
        importlib.reload(shared.database)

        from shared.database import db_manager

        read_stats = db_manager.get_pool_stats("read")
        write_stats = db_manager.get_pool_stats("write")

        assert read_stats['database'] == 'read'
        assert write_stats['database'] == 'write'
        assert 'pool_size' in read_stats
        assert 'pool_size' in write_stats

    @patch.dict(os.environ, {
        'ENABLE_DB_SPLITTING': 'true',
        'READ_DATABASE_URL': 'mongodb://localhost/read',
        'WRITE_DATABASE_URL': 'sqlite:///./write.db'
    }, clear=True)
    def test_mixed_database_types(self):
        """Test handling of mixed database types (MongoDB read, SQL write)"""
        import importlib
        import shared.database
        importlib.reload(shared.database)

        from shared.database import db_manager

        # Read should be MongoDB (no SQLAlchemy engine)
        read_stats = db_manager.get_pool_stats("read")
        assert read_stats['type'] == 'mongodb'

        # Write should be SQL
        write_stats = db_manager.get_pool_stats("write")
        assert write_stats['database'] == 'write'
        assert 'pool_size' in write_stats


if __name__ == "__main__":
    pytest.main([__file__])