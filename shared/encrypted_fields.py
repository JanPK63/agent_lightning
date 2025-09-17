"""
Encrypted Field Types and Utilities for SQLAlchemy
Provides transparent encryption/decryption for database fields
"""

import time
from typing import Any, Optional, Union
from sqlalchemy import String, Text, LargeBinary
from sqlalchemy.types import TypeDecorator
from sqlalchemy.engine import Dialect
import logging

# Lazy import to avoid circular dependency
_key_management_service = None

def _get_key_management_service():
    """Lazy import of key management service to avoid circular imports"""
    global _key_management_service
    if _key_management_service is None:
        from services.key_management_service import key_management_service
        _key_management_service = key_management_service
    return _key_management_service

logger = logging.getLogger(__name__)


class EncryptedField(TypeDecorator):
    """
    Base class for encrypted database fields
    Provides transparent encryption/decryption using AES-256-GCM
    """
    impl = LargeBinary

    def __init__(self, key_id: str, plaintext_type: Any = String, length: Optional[int] = None, **kwargs):
        """
        Initialize encrypted field

        Args:
            key_id: ID of the encryption key to use
            plaintext_type: SQLAlchemy type for the plaintext data
            length: Length for string types
            **kwargs: Additional arguments for the plaintext type
        """
        self.key_id = key_id
        self.plaintext_type = plaintext_type(length=length, **kwargs) if length else plaintext_type(**kwargs)
        super().__init__()

    def load_dialect_impl(self, dialect: Dialect):
        """Return the implementation for the given dialect"""
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value: Any, dialect: Dialect) -> Optional[bytes]:
        """
        Encrypt value before storing in database

        Args:
            value: Plaintext value to encrypt
            dialect: SQLAlchemy dialect

        Returns:
            Encrypted bytes or None
        """
        if value is None:
            return None

        try:
            start_time = time.time()

            # Convert to string if needed
            if not isinstance(value, str):
                value = str(value)

            # Get the encryption key
            encryption_key = self._get_encryption_key()

            # Encrypt the value
            encrypted_data = _get_key_management_service()._encrypt_data(
                value.encode('utf-8'),
                encryption_key
            )

            # Log the encryption operation
            duration_ms = int((time.time() - start_time) * 1000)
            logger.debug(f"Encrypted field using key {self.key_id} in {duration_ms}ms")

            return encrypted_data

        except Exception as e:
            logger.error(f"Error encrypting field with key {self.key_id}: {e}")
            # In production, you might want to raise an exception or return a special marker
            # For now, we'll log and return None to avoid breaking the application
            return None

    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        """
        Decrypt value when retrieving from database

        Args:
            value: Encrypted bytes from database
            dialect: SQLAlchemy dialect

        Returns:
            Decrypted plaintext value or None
        """
        if value is None:
            return None

        try:
            start_time = time.time()

            # Get the encryption key
            encryption_key = self._get_encryption_key()

            # Decrypt the value
            decrypted_bytes = _get_key_management_service()._decrypt_data(value, encryption_key)
            decrypted_value = decrypted_bytes.decode('utf-8')

            # Try to convert back to original type if it was converted to string
            if isinstance(self.plaintext_type, String):
                # Keep as string
                pass
            elif hasattr(self.plaintext_type, 'python_type'):
                try:
                    # Try to convert back to the original Python type
                    if self.plaintext_type.python_type == int:
                        decrypted_value = int(decrypted_value)
                    elif self.plaintext_type.python_type == float:
                        decrypted_value = float(decrypted_value)
                    elif self.plaintext_type.python_type == bool:
                        decrypted_value = decrypted_value.lower() in ('true', '1', 'yes')
                except (ValueError, AttributeError):
                    # If conversion fails, keep as string
                    pass

            # Log the decryption operation
            duration_ms = int((time.time() - start_time) * 1000)
            logger.debug(f"Decrypted field using key {self.key_id} in {duration_ms}ms")

            return decrypted_value

        except Exception as e:
            logger.error(f"Error decrypting field with key {self.key_id}: {e}")
            # Return a placeholder or raise an exception based on your error handling policy
            return "[ENCRYPTION_ERROR]"

    def _get_encryption_key(self) -> bytes:
        """
        Get the encryption key for this field

        Returns:
            Encryption key bytes

        Raises:
            ValueError: If key cannot be retrieved
        """
        try:
            # Try to get as field key first, then as data key
            try:
                return _get_key_management_service().get_field_key(self.key_id)
            except ValueError:
                return _get_key_management_service().get_data_key(self.key_id)
        except Exception as e:
            logger.error(f"Failed to retrieve encryption key {self.key_id}: {e}")
            raise ValueError(f"Encryption key {self.key_id} not available") from e


class EncryptedString(EncryptedField):
    """Encrypted string field"""

    def __init__(self, key_id: str, length: Optional[int] = None, **kwargs):
        super().__init__(key_id, String, length, **kwargs)


class EncryptedText(EncryptedField):
    """Encrypted text field (for longer text)"""

    def __init__(self, key_id: str, **kwargs):
        super().__init__(key_id, Text, **kwargs)


class EncryptedJSON(TypeDecorator):
    """Encrypted JSON field with automatic JSON serialization/deserialization"""
    impl = LargeBinary

    def __init__(self, key_id: str, **kwargs):
        self.key_id = key_id
        super().__init__(**kwargs)

    def load_dialect_impl(self, dialect: Dialect):
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value: Any, dialect: Dialect) -> Optional[bytes]:
        """Serialize and encrypt JSON data"""
        if value is None:
            return None

        try:
            import json
            json_str = json.dumps(value, default=str)
            return _get_key_management_service()._encrypt_data(
                json_str.encode('utf-8'),
                self._get_encryption_key()
            )
        except Exception as e:
            logger.error(f"Error encrypting JSON field: {e}")
            return None

    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        """Decrypt and deserialize JSON data"""
        if value is None:
            return None

        try:
            import json
            decrypted_bytes = _get_key_management_service()._decrypt_data(
                value,
                self._get_encryption_key()
            )
            json_str = decrypted_bytes.decode('utf-8')
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error decrypting JSON field: {e}")
            return {}

    def _get_encryption_key(self) -> bytes:
        """Get encryption key"""
        try:
            return _get_key_management_service().get_field_key(self.key_id)
        except ValueError:
            return _get_key_management_service().get_data_key(self.key_id)


# Convenience functions for creating encrypted fields
def encrypted_string(key_id: str, length: Optional[int] = None, **kwargs) -> EncryptedString:
    """Create an encrypted string field"""
    return EncryptedString(key_id, length, **kwargs)


def encrypted_text(key_id: str, **kwargs) -> EncryptedText:
    """Create an encrypted text field"""
    return EncryptedText(key_id, **kwargs)


def encrypted_json(key_id: str, **kwargs) -> EncryptedJSON:
    """Create an encrypted JSON field"""
    return EncryptedJSON(key_id, **kwargs)


# Decorator for encrypting specific model fields
def encrypted_field(key_id: str):
    """
    Decorator to mark a model field as encrypted

    Usage:
        class MyModel(Base):
            sensitive_data = Column(String(255))

            # Apply encryption
            sensitive_data = encrypted_field('my_field_key')(sensitive_data)
    """
    def decorator(field):
        # This is a conceptual decorator - in practice, you'd modify
        # the column definition to use EncryptedString instead
        field.encryption_key_id = key_id
        return field
    return decorator


# Utility functions for manual encryption/decryption
def encrypt_value(value: Any, key_id: str) -> bytes:
    """
    Manually encrypt a value

    Args:
        value: Value to encrypt
        key_id: Encryption key ID

    Returns:
        Encrypted bytes
    """
    try:
        if isinstance(value, str):
            value_bytes = value.encode('utf-8')
        elif isinstance(value, (int, float, bool)):
            value_bytes = str(value).encode('utf-8')
        else:
            import json
            value_bytes = json.dumps(value, default=str).encode('utf-8')

        encryption_key = _get_key_management_service().get_field_key(key_id)
        return _get_key_management_service()._encrypt_data(value_bytes, encryption_key)

    except ValueError:
        # Try data key if field key fails
        encryption_key = _get_key_management_service().get_data_key(key_id)
        return _get_key_management_service()._encrypt_data(value_bytes, encryption_key)


def decrypt_value(encrypted_data: bytes, key_id: str) -> Any:
    """
    Manually decrypt a value

    Args:
        encrypted_data: Encrypted bytes
        key_id: Encryption key ID

    Returns:
        Decrypted value
    """
    try:
        encryption_key = _get_key_management_service().get_field_key(key_id)
        decrypted_bytes = _get_key_management_service()._decrypt_data(encrypted_data, encryption_key)

        # Try to decode as UTF-8 string
        try:
            return decrypted_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If it's not a string, try JSON
            import json
            json_str = decrypted_bytes.decode('utf-8')
            return json.loads(json_str)

    except ValueError:
        # Try data key if field key fails
        encryption_key = _get_key_management_service().get_data_key(key_id)
        decrypted_bytes = _get_key_management_service()._decrypt_data(encrypted_data, encryption_key)

        try:
            return decrypted_bytes.decode('utf-8')
        except UnicodeDecodeError:
            import json
            json_str = decrypted_bytes.decode('utf-8')
            return json.loads(json_str)


# Performance monitoring utilities
class EncryptionMetrics:
    """Track encryption/decryption performance metrics"""

    def __init__(self):
        self.operations = []
        self.total_time = 0
        self.operation_count = 0

    def record_operation(self, operation: str, duration_ms: int, success: bool):
        """Record an encryption operation"""
        self.operations.append({
            'operation': operation,
            'duration_ms': duration_ms,
            'success': success,
            'timestamp': time.time()
        })
        self.total_time += duration_ms
        self.operation_count += 1

    def get_average_time(self) -> float:
        """Get average operation time"""
        return self.total_time / max(self.operation_count, 1)

    def get_success_rate(self) -> float:
        """Get success rate of operations"""
        if not self.operations:
            return 1.0
        successful = sum(1 for op in self.operations if op['success'])
        return successful / len(self.operations)


# Global metrics instance
encryption_metrics = EncryptionMetrics()


# Context manager for batch encryption operations
class EncryptionContext:
    """Context manager for batch encryption operations with metrics"""

    def __init__(self, operation_name: str = "batch"):
        self.operation_name = operation_name
        self.start_time = None
        self.operations = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = int((time.time() - self.start_time) * 1000)
        success = exc_type is None

        encryption_metrics.record_operation(
            self.operation_name,
            duration_ms,
            success
        )

        if not success:
            logger.error(f"Encryption context {self.operation_name} failed: {exc_val}")

    def record_operation(self, operation: str, success: bool, duration_ms: Optional[int] = None):
        """Record a sub-operation within the context"""
        if duration_ms is None:
            duration_ms = 0
        self.operations.append({
            'operation': operation,
            'success': success,
            'duration_ms': duration_ms
        })


# Example usage in SQLAlchemy models:
"""
Example of how to use encrypted fields in SQLAlchemy models:

from shared.encrypted_fields import EncryptedString, EncryptedJSON

class User(Base):
    __tablename__ = 'users'

    id = Column(String(36), primary_key=True)
    username = Column(String(50), unique=True)

    # Encrypted sensitive fields
    email = Column(EncryptedString('user_email_key', 255))
    profile_data = Column(EncryptedJSON('user_profile_key'))

    # Regular fields (not encrypted)
    created_at = Column(DateTime)
    is_active = Column(Boolean)
"""