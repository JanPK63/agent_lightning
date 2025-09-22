"""
Tests for SQLAlchemy connection pooling functionality
"""

import os
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from shared.database import (
    get_database_config,
    db_manager,
    POOL_SIZE,
    MAX_OVERFLOW,
    POOL_TIMEOUT,
    POOL_RECYCLE,
    POOL_PRE_PING
)


class TestConnectionPooling:
    """Test connection pooling configuration and monitoring"""

    def test_default_pool_configuration(self):
        """Test default pool configuration values"""
        assert POOL_SIZE == 10
        assert MAX_OVERFLOW == 20
        assert POOL_TIMEOUT == 30
        assert POOL_RECYCLE == 3600
        assert POOL_PRE_PING is True

    @patch.dict(os.environ, {
        'DB_POOL_SIZE': '5',
        'DB_MAX_OVERFLOW': '10',
        'DB_POOL_TIMEOUT': '15',
        'DB_POOL_RECYCLE': '1800',
        'DB_POOL_PRE_PING': 'false'
    })
    def test_custom_pool_configuration(self):
        """Test custom pool configuration via environment variables"""
        # Import fresh to get updated values
        import importlib
        import shared.database
        importlib.reload(shared.database)

        assert shared.database.POOL_SIZE == 5
        assert shared.database.MAX_OVERFLOW == 10
        assert shared.database.POOL_TIMEOUT == 15
        assert shared.database.POOL_RECYCLE == 1800
        assert shared.database.POOL_PRE_PING is False

    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///test.db'}, clear=True)
    def test_sqlite_pool_config(self):
        """Test SQLite pool configuration"""
        # Reload module to ensure default constants are used
        import importlib
        import shared.database
        importlib.reload(shared.database)
        from shared.database import get_database_config

        config = get_database_config()

        assert config['poolclass'].__name__ == 'StaticPool'
        assert config['pool_pre_ping'] is True  # Should be True by default
        assert 'connect_args' in config
        assert config['connect_args']['check_same_thread'] is False

    @patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost/test'}, clear=True)
    def test_postgresql_pool_config(self):
        """Test PostgreSQL pool configuration"""
        # Reload module to ensure default constants are used
        import importlib
        import shared.database
        importlib.reload(shared.database)
        from shared.database import get_database_config

        config = get_database_config()

        assert config['poolclass'].__name__ == 'QueuePool'
        assert config['pool_pre_ping'] is True  # Should be True by default
        assert config['pool_size'] == 10  # Default POOL_SIZE
        assert config['max_overflow'] == 20  # Default MAX_OVERFLOW
        assert config['pool_timeout'] == 30  # Default POOL_TIMEOUT
        assert config['pool_recycle'] == 3600  # Default POOL_RECYCLE

    @patch.dict(os.environ, {'DATABASE_URL': 'mongodb://localhost/test'})
    def test_mongodb_pool_config(self):
        """Test MongoDB pool configuration (should return None)"""
        config = get_database_config()

        assert config is None

    def test_pool_stats_mongodb(self):
        """Test pool stats for MongoDB configuration"""
        # Clear any existing DATABASE_URL
        with patch.dict(os.environ, {'DATABASE_URL': 'mongodb://localhost/test'}, clear=True):
            with patch('shared.database.engine', None):
                stats = db_manager.get_pool_stats()

                assert 'type' in stats
                assert stats['type'] == 'mongodb'
                assert 'message' in stats

    def test_connection_info_sqlite(self):
        """Test connection info for SQLite"""
        with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///test.db'}, clear=True):
            info = db_manager.get_connection_info()

            assert info['type'] == 'sql'
            assert 'pool_class' in info
            assert 'pool_size' in info
            assert 'max_overflow' in info
            assert 'pool_timeout' in info
            assert 'pool_recycle' in info
            assert 'pool_pre_ping' in info

    def test_connection_info_postgresql(self):
        """Test connection info for PostgreSQL"""
        # Reload module to ensure default constants are used
        import importlib
        import shared.database
        importlib.reload(shared.database)
        from shared.database import db_manager

        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost/test'}, clear=True):
            info = db_manager.get_connection_info()

            assert info['type'] == 'sql'
            assert info['pool_class'] == 'QueuePool'
            assert info['pool_size'] == 10  # Default POOL_SIZE
            assert info['max_overflow'] == 20  # Default MAX_OVERFLOW

    def test_connection_info_mongodb(self):
        """Test connection info for MongoDB"""
        with patch.dict(os.environ, {'DATABASE_URL': 'mongodb://user:pass@localhost/test'}, clear=True):
            info = db_manager.get_connection_info()

            assert info['type'] == 'mongodb'
            assert 'pooling' in info

    @patch('shared.database.engine')
    def test_pool_stats_sql_success(self, mock_engine):
        """Test successful pool stats retrieval for SQL database"""
        # Mock the engine and pool
        mock_pool = MagicMock()
        mock_pool.size = 10
        mock_pool.checkedin = 8
        mock_pool.checkedout = 2
        mock_pool.invalid = 0
        mock_pool.overflow = 1
        mock_pool.timeout = 0
        mock_pool.recycle = 3600

        mock_engine.pool = mock_pool

        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost/test'}, clear=True):
            stats = db_manager.get_pool_stats()

            assert 'pool_size' in stats
            assert 'checkedin' in stats
            assert 'checkedout' in stats
            assert 'connections_in_pool' in stats
            assert 'connections_checked_out' in stats

    @patch('shared.database.engine', None)
    def test_pool_stats_no_engine(self):
        """Test pool stats when no engine is available"""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost/test'}, clear=True):
            stats = db_manager.get_pool_stats()

            # Should return MongoDB message since engine is None
            assert 'type' in stats
            assert stats['type'] == 'mongodb'

    def test_pool_stats_exception_handling(self):
        """Test pool stats exception handling"""
        with patch('shared.database.engine') as mock_engine:
            # Make pool attribute access raise an exception
            type(mock_engine).pool = PropertyMock(side_effect=Exception("Test error"))

            with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost/test'}, clear=True):
                stats = db_manager.get_pool_stats()

                assert 'error' in stats
                assert 'Test error' in stats['error']


class TestDatabaseManagerIntegration:
    """Integration tests for database manager"""

    def test_health_check_sqlite(self):
        """Test health check with SQLite"""
        with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///./test.db'}):
            # This would normally test actual database connection
            # For now, just ensure the method exists and doesn't crash
            result = db_manager.health_check()
            assert isinstance(result, bool)

    def test_is_mongodb_configured(self):
        """Test MongoDB configuration detection"""
        with patch.dict(os.environ, {'DATABASE_URL': 'mongodb://localhost/test'}, clear=True):
            assert db_manager.is_mongodb_configured() is True

        with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///test.db'}, clear=True):
            assert db_manager.is_mongodb_configured() is False

        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost/test'}, clear=True):
            assert db_manager.is_mongodb_configured() is False


if __name__ == "__main__":
    pytest.main([__file__])