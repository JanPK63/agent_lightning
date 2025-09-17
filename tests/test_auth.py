"""
Tests for authentication and authorization
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from agentlightning.auth import AuthManager, RateLimiter
from shared.models import ApiKey


class TestAuthManager:
    """Test the AuthManager class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.dal = Mock()
        self.auth_manager = AuthManager(self.dal)

    def test_hash_api_key(self):
        """Test API key hashing"""
        key = "test-api-key-123"
        hash1 = self.auth_manager.hash_api_key(key)
        hash2 = self.auth_manager.hash_api_key(key)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
        assert hash1 != key

    def test_generate_api_key(self):
        """Test API key generation"""
        key1 = self.auth_manager.generate_api_key()
        key2 = self.auth_manager.generate_api_key()

        assert key1 != key2
        assert len(key1) >= 32  # Should be reasonably long
        assert isinstance(key1, str)

    @patch('agentlightning.auth.datetime')
    def test_validate_api_key_valid(self, mock_datetime):
        """Test validating a valid API key"""
        mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 12, 0, 0)

        # Mock database session and key
        mock_session = Mock()
        mock_key = Mock()
        mock_key.id = "key-123"
        mock_key.name = "Test Key"
        mock_key.user_id = "user-123"
        mock_key.permissions = ["read", "write"]
        mock_key.is_expired.return_value = False
        mock_key.rate_limit_requests = 100
        mock_key.rate_limit_window = 60
        mock_key.usage_count = 0  # Initialize as integer

        # Mock the context manager using patch
        with patch.object(self.dal.db, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_session
            mock_get_db.return_value.__exit__.return_value = None

            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_key

            result = self.auth_manager.validate_api_key("test-key")

            assert result is not None
            assert result['id'] == "key-123"
            assert result['name'] == "Test Key"
            assert result['permissions'] == ["read", "write"]

            # Verify usage was updated
            assert mock_key.last_used_at is not None
            assert mock_key.usage_count == 1
            mock_session.commit.assert_called_once()

    def test_validate_api_key_invalid(self):
        """Test validating an invalid API key"""
        mock_session = Mock()

        # Mock the context manager using patch
        with patch.object(self.dal.db, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_session
            mock_get_db.return_value.__exit__.return_value = None

            mock_session.query.return_value.filter_by.return_value.first.return_value = None

            result = self.auth_manager.validate_api_key("invalid-key")

            assert result is None

    def test_validate_api_key_expired(self):
        """Test validating an expired API key"""
        mock_session = Mock()
        mock_key = Mock()
        mock_key.is_expired.return_value = True

        # Mock the context manager using patch
        with patch.object(self.dal.db, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_session
            mock_get_db.return_value.__exit__.return_value = None

            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_key

            result = self.auth_manager.validate_api_key("expired-key")

            assert result is None

    @patch('agentlightning.auth.datetime')
    @patch('agentlightning.auth.ApiKey')
    @patch.object(AuthManager, 'generate_api_key')
    def test_create_api_key(self, mock_generate_key, mock_api_key_class, mock_datetime):
        """Test creating a new API key"""
        mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        # Mock the generated key and database key
        mock_generate_key.return_value = "test-generated-key"
        mock_key = Mock()
        mock_key.id = "new-key-123"
        mock_key.permissions = ["read"]
        mock_api_key_class.return_value = mock_key

        mock_session = Mock()

        # Mock the context manager using patch
        with patch.object(self.dal.db, 'get_db') as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = mock_session
            mock_get_db.return_value.__exit__.return_value = None

            mock_session.add.return_value = None
            mock_session.commit.return_value = None

            result = self.auth_manager.create_api_key(
                name="Test Key",
                user_id="user-123",
                permissions=["read"],
                expires_in_days=30
            )

            assert 'key' in result
            assert result['key'] == "test-generated-key"
            assert result['name'] == "Test Key"
            assert result['user_id'] == "user-123"
            assert result['permissions'] == ["read"]

            # Verify ApiKey was created with correct parameters
            mock_api_key_class.assert_called_once()
            call_args = mock_api_key_class.call_args
            assert call_args[1]['name'] == "Test Key"
            assert call_args[1]['user_id'] == "user-123"
            assert call_args[1]['permissions'] == ["read"]


class TestRateLimiter:
    """Test the RateLimiter class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.limiter = RateLimiter()

    @patch('agentlightning.auth.time')
    def test_is_allowed_under_limit(self, mock_time):
        """Test rate limiting when under the limit"""
        mock_time.time.return_value = 1000

        # Should allow first request
        assert self.limiter.is_allowed("user1", 2, 60) == True

        # Should allow second request
        assert self.limiter.is_allowed("user1", 2, 60) == True

        # Should deny third request
        assert self.limiter.is_allowed("user1", 2, 60) == False

    @patch('agentlightning.auth.time')
    def test_is_allowed_window_expiry(self, mock_time):
        """Test that old requests expire from the window"""
        # First request at time 1000
        mock_time.time.return_value = 1000
        assert self.limiter.is_allowed("user1", 1, 60) == True

        # Second request at time 1061 (61 seconds later, window has moved)
        mock_time.time.return_value = 1061
        assert self.limiter.is_allowed("user1", 1, 60) == True

    def test_is_allowed_different_keys(self):
        """Test that different keys are tracked separately"""
        # User1 should be allowed
        assert self.limiter.is_allowed("user1", 1, 60) == True

        # User2 should also be allowed (separate tracking)
        assert self.limiter.is_allowed("user2", 1, 60) == True

        # User1 should be denied second request
        assert self.limiter.is_allowed("user1", 1, 60) == False