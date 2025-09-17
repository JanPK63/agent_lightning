"""
Comprehensive tests for the database encryption system
Tests key management, encryption middleware, and encrypted fields
"""

import pytest
import unittest.mock as mock
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from services.key_management_service import KeyManagementService, KeyRotationResult
from shared.encryption_middleware import EncryptionMiddleware
from shared.encrypted_fields import (
    EncryptedString, EncryptedText, EncryptedJSON,
    encrypt_value, decrypt_value, encryption_metrics
)
from shared.models import (
    EncryptionKey, KeyUsageLog, KeyRotationHistory, KeyAccessAudit,
    User, Conversation, Agent, Workflow
)
from agentlightning.auth import AuthManager


class TestKeyManagementService:
    """Test cases for KeyManagementService"""

    @pytest.fixture
    def key_service(self):
        """Create a test instance of KeyManagementService"""
        with patch('services.key_management_service.db_manager'):
            with patch('services.key_management_service.EventBus'):
                service = KeyManagementService()
                return service

    def test_generate_master_key(self, key_service):
        """Test master key generation"""
        with patch.object(key_service, '_encrypt_with_passphrase') as mock_encrypt:
            with patch('services.key_management_service.db_manager') as mock_db:
                mock_encrypt.return_value = b'encrypted_key_data'

                mock_session = Mock()
                mock_db.get_db.return_value.__enter__.return_value = mock_session

                # Mock no existing master key
                mock_session.query.return_value.filter.return_value.first.return_value = None

                result = key_service.generate_master_key()

                assert result is not None
                assert mock_encrypt.called
                assert mock_session.add.called
                assert mock_session.commit.called

    def test_get_master_key(self, key_service):
        """Test master key retrieval"""
        with patch.object(key_service, '_decrypt_with_passphrase') as mock_decrypt:
            with patch('services.key_management_service.db_manager') as mock_db:
                mock_decrypt.return_value = b'master_key_32_bytes'

                mock_session = Mock()
                mock_db.get_db.return_value.__enter__.return_value = mock_session

                mock_key_record = Mock()
                mock_key_record.encrypted_key = b'encrypted_data'
                mock_key_record.key_hash = 'correct_hash'
                mock_session.query.return_value.filter.return_value.first.return_value = mock_key_record

                with patch.object(key_service, '_calculate_key_hash', return_value='correct_hash'):
                    result = key_service.get_master_key()

                    assert result == b'master_key_32_bytes'
                    assert mock_decrypt.called

    def test_generate_data_key(self, key_service):
        """Test data key generation"""
        with patch.object(key_service, 'get_master_key', return_value=b'master_key_32_bytes'):
            with patch.object(key_service, '_encrypt_data') as mock_encrypt:
                with patch('services.key_management_service.db_manager') as mock_db:
                    mock_encrypt.return_value = b'encrypted_data_key'

                    mock_session = Mock()
                    mock_db.get_db.return_value.__enter__.return_value = mock_session

                    result = key_service.generate_data_key('users')

                    assert result.startswith('data_users_')
                    assert mock_encrypt.called
                    assert mock_session.add.called

    def test_rotate_key_success(self, key_service):
        """Test successful key rotation"""
        with patch('services.key_management_service.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session

            # Mock existing key
            mock_key = Mock()
            mock_key.key_id = 'test_key'
            mock_key.key_type = 'data'
            mock_key.status = 'active'
            mock_key.expires_at = datetime.utcnow() + timedelta(days=30)
            mock_key.next_rotation_at = datetime.utcnow() + timedelta(days=30)

            mock_session.query.return_value.filter.return_value.first.return_value = mock_key

            with patch.object(key_service, 'get_master_key', return_value=b'master_key'):
                with patch.object(key_service, '_encrypt_data') as mock_encrypt:
                    mock_encrypt.return_value = b'new_encrypted_key'

                    result = key_service.rotate_key('test_key', 'scheduled')

                    assert result.success == True
                    assert result.old_key_id == 'test_key'
                    assert result.new_key_id.startswith('test_key_rotated_')

    def test_get_keys_due_for_rotation(self, key_service):
        """Test getting keys due for rotation"""
        with patch('services.key_management_service.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session

            mock_key = Mock()
            mock_key.id = '123'
            mock_key.key_id = 'test_key'
            mock_key.key_type = 'data'
            mock_key.name = 'Test Key'
            mock_key.next_rotation_at = datetime.utcnow() - timedelta(days=1)  # Overdue
            mock_key.rotation_count = 5

            mock_session.query.return_value.filter.return_value.all.return_value = [mock_key]

            keys = key_service.get_keys_due_for_rotation()

            assert len(keys) == 1
            assert keys[0]['key_id'] == 'test_key'
            assert keys[0]['days_until_rotation'] < 0  # Negative for overdue


class TestEncryptionMiddleware:
    """Test cases for EncryptionMiddleware"""

    @pytest.fixture
    def middleware(self):
        """Create a test instance of EncryptionMiddleware"""
        with patch('shared.encryption_middleware.db_manager'):
            with patch('shared.encryption_middleware.EventBus'):
                mw = EncryptionMiddleware()
                return mw

    def test_middleware_initialization(self, middleware):
        """Test middleware initialization"""
        assert middleware.encryption_enabled == True
        assert 'User' in middleware.models_with_encryption
        assert 'Conversation' in middleware.models_with_encryption

    def test_encrypt_instance_fields(self, middleware):
        """Test encrypting instance fields"""
        # Create mock user instance
        mock_user = Mock()
        mock_user.email = 'test@example.com'
        mock_user.password_hash = 'hashed_password'
        mock_user.email_encrypted = None
        mock_user.password_hash_encrypted = None

        with patch.object(middleware, '_encrypt_field_value') as mock_encrypt:
            mock_encrypt.side_effect = [b'encrypted_email', b'encrypted_password']

            middleware._encrypt_instance_fields(mock_user, None)

            assert mock_encrypt.call_count == 2
            # Verify the instance was updated
            assert mock_user.email_encrypted == b'encrypted_email'
            assert mock_user.password_hash_encrypted == b'encrypted_password'

    def test_decrypt_instance_fields(self, middleware):
        """Test decrypting instance fields"""
        # Create mock user instance
        mock_user = Mock()
        mock_user.email_encrypted = b'encrypted_email_data'
        mock_user.password_hash_encrypted = b'encrypted_password_data'

        with patch.object(middleware, '_decrypt_field_value') as mock_decrypt:
            mock_decrypt.side_effect = ['decrypted@example.com', 'decrypted_hash']

            middleware._decrypt_instance_fields(mock_user, None)

            assert mock_decrypt.call_count == 2
            # Verify the instance was updated
            assert mock_user.email_encrypted == 'decrypted@example.com'
            assert mock_user.password_hash_encrypted == 'decrypted_hash'

    def test_get_key_id_for_field(self, middleware):
        """Test key ID generation for fields"""
        from shared.models import User, Conversation

        # Test user email field
        key_id = middleware._get_key_id_for_field(User, 'email_encrypted')
        assert key_id == 'user_email_key'

        # Test conversation query field
        key_id = middleware._get_key_id_for_field(Conversation, 'user_query_encrypted')
        assert key_id == 'conversation_query_key'


class TestEncryptedFields:
    """Test cases for encrypted field types"""

    @pytest.fixture
    def encrypted_string_field(self):
        """Create an encrypted string field"""
        return EncryptedString('test_key')

    def test_encrypted_string_creation(self, encrypted_string_field):
        """Test creating an encrypted string field"""
        assert encrypted_string_field.key_id == 'test_key'
        assert encrypted_string_field.plaintext_type.__name__ == 'String'

    def test_encrypt_value_function(self):
        """Test the encrypt_value utility function"""
        test_data = "sensitive information"
        test_key = "test_encryption_key"

        with patch('shared.encrypted_fields.key_management_service') as mock_kms:
            mock_kms._encrypt_data.return_value = b'encrypted_data'

            result = encrypt_value(test_data, test_key)

            assert result == b'encrypted_data'
            mock_kms._encrypt_data.assert_called_once()

    def test_decrypt_value_function(self):
        """Test the decrypt_value utility function"""
        encrypted_data = b'encrypted_data'
        test_key = "test_encryption_key"

        with patch('shared.encrypted_fields.key_management_service') as mock_kms:
            mock_kms._decrypt_data.return_value = b'decrypted_data'

            result = decrypt_value(encrypted_data, test_key)

            assert result == b'decrypted_data'
            mock_kms._decrypt_data.assert_called_once()


class TestAuthManagerEncryption:
    """Test cases for AuthManager with encrypted data"""

    @pytest.fixture
    def auth_manager(self):
        """Create a test instance of AuthManager"""
        with patch('agentlightning.auth.DataAccessLayer'):
            manager = AuthManager()
            return manager

    def test_authenticate_user_success(self, auth_manager):
        """Test successful user authentication with encrypted data"""
        with patch('agentlightning.auth.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session

            # Mock user with encrypted data
            mock_user = Mock()
            mock_user.id = '123'
            mock_user.username = 'testuser'
            mock_user.email = 'test@example.com'  # Would be decrypted by middleware
            mock_user.password_hash_encrypted = 'hashed_password'
            mock_user.role = 'user'
            mock_user.is_active = True
            mock_user.created_at = datetime.utcnow()
            mock_user.updated_at = datetime.utcnow()

            mock_session.query.return_value.filter.return_value.first.return_value = mock_user

            with patch.object(auth_manager, '_verify_password', return_value=True):
                result = auth_manager.authenticate_user('testuser', 'password123')

                assert result is not None
                assert result['username'] == 'testuser'
                assert result['email'] == 'test@example.com'

    def test_create_user_with_encryption(self, auth_manager):
        """Test creating a user with encrypted data"""
        with patch('agentlightning.auth.db_manager') as mock_db:
            mock_session = Mock()
            mock_db.get_db.return_value.__enter__.return_value = mock_session

            # Mock no existing user
            mock_session.query.return_value.filter.return_value.first.return_value = None

            # Mock user creation
            mock_user = Mock()
            mock_user.id = '123'
            mock_user.username = 'newuser'
            mock_user.email = 'new@example.com'
            mock_user.role = 'user'
            mock_user.is_active = True
            mock_user.created_at = datetime.utcnow()

            with patch('agentlightning.auth.User', return_value=mock_user):
                with patch.object(auth_manager, '_hash_password', return_value='hashed_password'):
                    result = auth_manager.create_user('newuser', 'new@example.com', 'password123')

                    assert result is not None
                    assert result['username'] == 'newuser'
                    assert mock_session.add.called
                    assert mock_session.commit.called


class TestEncryptionMetrics:
    """Test cases for encryption metrics"""

    def test_metrics_tracking(self):
        """Test that metrics are properly tracked"""
        from shared.encrypted_fields import encryption_metrics

        initial_count = encryption_metrics.operation_count

        # Record some operations
        encryption_metrics.record_operation('encrypt', 50, True)
        encryption_metrics.record_operation('decrypt', 30, True)
        encryption_metrics.record_operation('encrypt', 40, False)

        assert encryption_metrics.operation_count == initial_count + 3
        assert encryption_metrics.get_average_time() > 0
        assert encryption_metrics.get_success_rate() == 2/3


class TestEncryptionIntegration:
    """Integration tests for the complete encryption system"""

    def test_full_encryption_workflow(self):
        """Test a complete encryption workflow"""
        # This would be an integration test that:
        # 1. Creates encryption keys
        # 2. Encrypts data using the middleware
        # 3. Stores data in database
        # 4. Retrieves and decrypts data
        # 5. Verifies data integrity

        # For now, we'll just verify the components can be imported and initialized
        try:
            from services.key_management_service import key_management_service
            from shared.encryption_middleware import encryption_middleware
            from shared.encrypted_fields import EncryptedString

            # Verify services are available
            assert key_management_service is not None
            assert encryption_middleware is not None

            # Verify encrypted field can be created
            field = EncryptedString('test_key')
            assert field.key_id == 'test_key'

        except ImportError as e:
            pytest.skip(f"Integration test skipped due to missing dependencies: {e}")


# Performance tests
class TestEncryptionPerformance:
    """Performance tests for encryption operations"""

    def test_encryption_speed(self):
        """Test that encryption operations are reasonably fast"""
        import time

        test_data = "This is test data for performance measurement"
        test_key = "performance_test_key"

        with patch('shared.encrypted_fields.key_management_service') as mock_kms:
            mock_kms._encrypt_data.return_value = b'encrypted_data'

            start_time = time.time()
            for _ in range(100):
                encrypt_value(test_data, test_key)
            end_time = time.time()

            total_time = end_time - start_time
            avg_time = total_time / 100

            # Should be reasonably fast (< 1ms per operation)
            assert avg_time < 0.001, f"Encryption too slow: {avg_time:.4f}s per operation"


if __name__ == "__main__":
    pytest.main([__file__])