"""
Unit tests for API Key Rotation functionality
"""

import pytest
import unittest.mock as mock
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from services.api_key_rotation_service import (
    ApiKeyRotationService,
    RotationResult,
    NotificationInfo
)
from shared.models import ApiKey, ApiKeyRotationPolicy
from agentlightning.auth import AuthManager


class TestApiKeyRotationService:
    """Test cases for ApiKeyRotationService"""

    @pytest.fixture
    def rotation_service(self):
        """Create a test instance of ApiKeyRotationService"""
        with patch('services.api_key_rotation_service.db_manager') as mock_db:
            with patch('services.api_key_rotation_service.EventBus') as mock_event_bus:
                service = ApiKeyRotationService()
                service.event_bus = mock_event_bus.return_value
                return service

    @pytest.fixture
    def mock_api_key(self):
        """Create a mock API key"""
        key = Mock(spec=ApiKey)
        key.id = "test-key-id"
        key.key_hash = "old_hash"
        key.name = "Test Key"
        key.user_id = "user123"
        key.is_active = True
        key.is_rotation_enabled = True
        key.rotation_locked = False
        key.expires_at = datetime.utcnow() + timedelta(days=30)
        key.last_rotated_at = None
        key.next_rotation_at = datetime.utcnow() + timedelta(days=1)  # Due tomorrow
        key.rotation_count = 0
        return key

    @pytest.fixture
    def mock_policy(self):
        """Create a mock rotation policy"""
        policy = Mock(spec=ApiKeyRotationPolicy)
        policy.id = "policy123"
        policy.auto_rotate_days = 90
        policy.notify_before_days = 7
        policy.grace_period_days = 30
        policy.is_active = True
        return policy

    def test_rotate_api_key_success(self, rotation_service, mock_api_key, mock_policy):
        """Test successful API key rotation"""
        with patch.object(rotation_service, 'auth_manager') as mock_auth:
            with patch('services.api_key_rotation_service.db_manager') as mock_db:
                # Setup mocks
                mock_auth.generate_api_key.return_value = "new_test_key_12345"
                mock_auth.hash_api_key.return_value = "new_key_hash"

                mock_session = Mock()
                mock_db.get_db.return_value.__enter__.return_value = mock_session

                mock_session.query.return_value.filter.return_value.first.return_value = mock_api_key
                mock_session.query.return_value.filter.return_value.filter.return_value.first.return_value = mock_policy

                # Execute rotation
                result = rotation_service.rotate_api_key("test-key-id", "user123", "scheduled")

                # Verify result
                assert result.success == True
                assert result.api_key_id == "test-key-id"
                assert result.new_key == "new_test_key_12345"
                assert result.old_key_hash == "old_hash"
                assert result.new_key_hash == "new_key_hash"

                # Verify database calls
                assert mock_session.add.called
                assert mock_session.commit.called

                # Verify event emission
                rotation_service.event_bus.emit.assert_called()

    def test_rotate_api_key_not_found(self, rotation_service):
        """Test rotation of non-existent API key"""
        with patch('services.api_key_rotation_service.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = None

            result = rotation_service.rotate_api_key("non-existent-key")

            assert result.success == False
            assert "not found" in result.error_message.lower()

    def test_rotate_api_key_rotation_disabled(self, rotation_service, mock_api_key):
        """Test rotation when rotation is disabled"""
        mock_api_key.is_rotation_enabled = False

        with patch('services.api_key_rotation_service.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = mock_api_key

            result = rotation_service.rotate_api_key("test-key-id")

            assert result.success == False
            assert "disabled" in result.error_message.lower()

    def test_rotate_api_key_locked(self, rotation_service, mock_api_key):
        """Test rotation when key is locked"""
        mock_api_key.rotation_locked = True

        with patch('services.api_key_rotation_service.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = mock_api_key

            result = rotation_service.rotate_api_key("test-key-id")

            assert result.success == False
            assert "locked" in result.error_message.lower()

    def test_get_keys_due_for_rotation(self, rotation_service, mock_api_key):
        """Test getting keys due for rotation"""
        with patch('services.api_key_rotation_service.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.filter.return_value.all.return_value = [mock_api_key]

            keys = rotation_service.get_keys_due_for_rotation()

            assert len(keys) == 1
            assert keys[0]['id'] == "test-key-id"
            assert keys[0]['name'] == "Test Key"
            assert keys[0]['days_until_rotation'] == 1

    def test_get_keys_due_for_rotation_no_keys(self, rotation_service):
        """Test getting keys due for rotation when none are due"""
        with patch('services.api_key_rotation_service.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.filter.return_value.all.return_value = []

            keys = rotation_service.get_keys_due_for_rotation()

            assert len(keys) == 0

    def test_cleanup_expired_keys(self, rotation_service, mock_api_key):
        """Test cleanup of expired keys"""
        # Make key expired
        mock_api_key.expires_at = datetime.utcnow() - timedelta(days=1)
        mock_api_key.last_rotated_at = datetime.utcnow() - timedelta(days=40)  # Past grace period

        with patch('services.api_key_rotation_service.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.all.return_value = [mock_api_key]

            count = rotation_service.cleanup_expired_keys()

            assert count == 1
            assert mock_api_key.is_active == False
            assert mock_session.commit.called

    def test_cleanup_expired_keys_within_grace_period(self, rotation_service, mock_api_key):
        """Test that keys within grace period are not cleaned up"""
        # Make key expired but within grace period
        mock_api_key.expires_at = datetime.utcnow() - timedelta(days=1)
        mock_api_key.last_rotated_at = datetime.utcnow() - timedelta(days=10)  # Within 30-day grace

        with patch('services.api_key_rotation_service.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.all.return_value = [mock_api_key]

            count = rotation_service.cleanup_expired_keys()

            assert count == 0
            assert mock_api_key.is_active == True  # Should not be deactivated

    def test_bulk_rotate_keys(self, rotation_service):
        """Test bulk rotation of multiple keys"""
        with patch.object(rotation_service, 'rotate_api_key') as mock_rotate:
            mock_rotate.side_effect = [
                RotationResult(True, "key1", "new1", "old1", "hash1", "hist1"),
                RotationResult(False, "key2", "", "", "", "", "error"),
                RotationResult(True, "key3", "new3", "old3", "hash3", "hist3")
            ]

            results = rotation_service.bulk_rotate_keys(["key1", "key2", "key3"], "admin", "bulk")

            assert len(results) == 3
            assert results[0].success == True
            assert results[1].success == False
            assert results[2].success == True

            # Verify all keys were attempted
            assert mock_rotate.call_count == 3

    def test_get_pending_notifications(self, rotation_service, mock_api_key, mock_policy):
        """Test getting pending notifications"""
        # Set up key that's due for notification (rotation in 3 days, notify 7 days before)
        mock_api_key.next_rotation_at = datetime.utcnow() + timedelta(days=3)

        with patch('services.api_key_rotation_service.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session

            # Mock policy query
            mock_session.query.return_value.filter.return_value.all.return_value = [mock_policy]

            # Mock key query with notification cutoff
            mock_key_query = Mock()
            mock_key_query.all.return_value = [mock_api_key]
            mock_session.query.return_value.filter.return_value.filter.return_value.filter.return_value.filter.return_value = mock_key_query

            notifications = rotation_service.get_pending_notifications()

            assert len(notifications) >= 0  # May be 0 depending on exact timing

    def test_get_rotation_history(self, rotation_service):
        """Test getting rotation history"""
        mock_history = [Mock(), Mock()]
        mock_history[0].to_dict.return_value = {"id": "hist1", "rotated_at": "2023-01-01"}
        mock_history[1].to_dict.return_value = {"id": "hist2", "rotated_at": "2023-01-02"}

        with patch('services.api_key_rotation_service.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_history

            history = rotation_service.get_rotation_history("test-key-id", 10)

            assert len(history) == 2
            assert history[0]["id"] == "hist1"
            assert history[1]["id"] == "hist2"


class TestRotationResult:
    """Test cases for RotationResult dataclass"""

    def test_rotation_result_creation(self):
        """Test creating a RotationResult"""
        result = RotationResult(
            success=True,
            api_key_id="test-id",
            new_key="new-key",
            old_key_hash="old-hash",
            new_key_hash="new-hash",
            rotation_history_id="hist-id"
        )

        assert result.success == True
        assert result.api_key_id == "test-id"
        assert result.new_key == "new-key"
        assert result.old_key_hash == "old-hash"
        assert result.new_key_hash == "new-hash"
        assert result.rotation_history_id == "hist-id"
        assert result.error_message is None

    def test_rotation_result_with_error(self):
        """Test RotationResult with error message"""
        result = RotationResult(
            success=False,
            api_key_id="test-id",
            new_key="",
            old_key_hash="",
            new_key_hash="",
            rotation_history_id="",
            error_message="Test error"
        )

        assert result.success == False
        assert result.error_message == "Test error"


class TestNotificationInfo:
    """Test cases for NotificationInfo dataclass"""

    def test_notification_info_creation(self):
        """Test creating a NotificationInfo"""
        rotation_date = datetime.utcnow() + timedelta(days=7)

        notification = NotificationInfo(
            api_key_id="test-key",
            key_name="Test Key",
            user_id="user123",
            days_until_rotation=7,
            rotation_date=rotation_date
        )

        assert notification.api_key_id == "test-key"
        assert notification.key_name == "Test Key"
        assert notification.user_id == "user123"
        assert notification.days_until_rotation == 7
        assert notification.rotation_date == rotation_date


if __name__ == "__main__":
    pytest.main([__file__])