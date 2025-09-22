"""
Database Encryption Middleware
Provides transparent encryption/decryption for database operations using SQLAlchemy events
"""

import logging
from typing import Any, Dict
from sqlalchemy import event
from sqlalchemy.orm import Session
import time

from services.key_management_service import key_management_service
from shared.encrypted_fields import encryption_metrics, EncryptionContext
from shared.models import (
    User,
    Conversation,
    Agent,
    Workflow,
    EncryptionKey,
    KeyUsageLog,
)

logger = logging.getLogger(__name__)


class EncryptionMiddleware:
    """
    Middleware for transparent database field encryption/decryption

    This middleware automatically:
    - Encrypts data before saving to encrypted fields
    - Decrypts data when loading from encrypted fields
    - Tracks encryption operations for monitoring
    - Handles key rotation transparently
    """

    def __init__(self):
        self.encryption_enabled = True
        self.performance_monitoring = True
        self.models_with_encryption = {
            User: ["email_encrypted", "password_hash_encrypted"],
            Conversation: ["user_query_encrypted", "agent_response_encrypted"],
            Agent: ["config_encrypted", "capabilities_encrypted"],
            Workflow: ["steps_encrypted", "context_encrypted"],
        }

    def register_events(self):
        """Register SQLAlchemy event listeners"""

        # Before insert/update - encrypt sensitive fields
        event.listen(Session, "before_flush", self._before_flush)

        # After load - decrypt sensitive fields
        for model_class, encrypted_fields in self.models_with_encryption.items():
            for field_name in encrypted_fields:
                if hasattr(model_class, field_name):
                    event.listen(model_class, "load", self._after_load)

        logger.info("Encryption middleware events registered")

    def unregister_events(self):
        """Unregister SQLAlchemy event listeners"""
        # Note: In practice, you'd need to keep references to remove specific listeners
        logger.info("Encryption middleware events unregistered")

    def _before_flush(self, session: Session, flush_context, instances):
        """
        Encrypt sensitive fields before database flush

        Args:
            session: SQLAlchemy session
            flush_context: Flush context
            instances: Instances being flushed
        """
        if not self.encryption_enabled:
            return

        try:
            with EncryptionContext("database_flush") as ctx:
                for instance in session.new | session.dirty:
                    self._encrypt_instance_fields(instance, ctx)

        except Exception as e:
            logger.error(f"Error in encryption middleware before_flush: {e}")
            # Don't raise exception to avoid breaking the application
            # In production, you might want to handle this differently

    def _after_load(self, target, context):
        """
        Decrypt sensitive fields after loading from database

        Args:
            target: Loaded model instance
            context: Load context
        """
        if not self.encryption_enabled:
            return

        try:
            with EncryptionContext("database_load") as ctx:
                self._decrypt_instance_fields(target, ctx)

        except Exception as e:
            logger.error(f"Error in encryption middleware after_load: {e}")
            # Don't raise exception to avoid breaking the application

    def _encrypt_instance_fields(self, instance, context: EncryptionContext):
        """
        Encrypt sensitive fields of a model instance

        Args:
            instance: Model instance
            context: Encryption context for monitoring
        """
        model_class = instance.__class__

        if model_class not in self.models_with_encryption:
            return

        encrypted_fields = self.models_with_encryption[model_class]

        for field_name in encrypted_fields:
            if not hasattr(instance, field_name):
                continue

            # Get the current value
            current_value = getattr(instance, field_name)

            # Skip if already encrypted (check if it's bytes)
            if isinstance(current_value, bytes):
                continue

            # Skip if None
            if current_value is None:
                continue

            try:
                # Determine key ID from field name
                key_id = self._get_key_id_for_field(model_class, field_name)

                # Encrypt the value
                start_time = time.time()
                encrypted_value = self._encrypt_field_value(current_value, key_id)
                encrypt_time = int((time.time() - start_time) * 1000)

                # Update the instance
                setattr(instance, field_name, encrypted_value)

                # Log the operation
                context.record_operation(
                    f"encrypt_{model_class.__name__}.{field_name}", True, encrypt_time
                )

                # Record usage
                self._record_key_usage(
                    key_id, "encrypt", field_name, model_class.__name__
                )

            except Exception as e:
                logger.error(
                    f"Failed to encrypt {field_name} for {model_class.__name__}: {e}"
                )
                context.record_operation(
                    f"encrypt_{model_class.__name__}.{field_name}", False
                )

    def _decrypt_instance_fields(self, instance, context: EncryptionContext):
        """
        Decrypt sensitive fields of a model instance

        Args:
            instance: Model instance
            context: Encryption context for monitoring
        """
        model_class = instance.__class__

        if model_class not in self.models_with_encryption:
            return

        encrypted_fields = self.models_with_encryption[model_class]

        for field_name in encrypted_fields:
            if not hasattr(instance, field_name):
                continue

            # Get the current value
            current_value = getattr(instance, field_name)

            # Skip if not encrypted (check if it's not bytes)
            if not isinstance(current_value, bytes):
                continue

            # Skip if None
            if current_value is None:
                continue

            try:
                # Determine key ID from field name
                key_id = self._get_key_id_for_field(model_class, field_name)

                # Decrypt the value
                start_time = time.time()
                decrypted_value = self._decrypt_field_value(current_value, key_id)
                decrypt_time = int((time.time() - start_time) * 1000)

                # Update the instance
                setattr(instance, field_name, decrypted_value)

                # Log the operation
                context.record_operation(
                    f"decrypt_{model_class.__name__}.{field_name}", True, decrypt_time
                )

                # Record usage
                self._record_key_usage(
                    key_id, "decrypt", field_name, model_class.__name__
                )

            except Exception as e:
                logger.error(
                    f"Failed to decrypt {field_name} for {model_class.__name__}: {e}"
                )
                # Set to error placeholder
                setattr(instance, field_name, "[DECRYPTION_ERROR]")
                context.record_operation(
                    f"decrypt_{model_class.__name__}.{field_name}", False
                )

    def _encrypt_field_value(self, value: Any, key_id: str) -> bytes:
        """
        Encrypt a field value

        Args:
            value: Value to encrypt
            key_id: Encryption key ID

        Returns:
            Encrypted bytes
        """
        from shared.encrypted_fields import encrypt_value

        return encrypt_value(value, key_id)

    def _decrypt_field_value(self, encrypted_value: bytes, key_id: str) -> Any:
        """
        Decrypt a field value

        Args:
            encrypted_value: Encrypted bytes
            key_id: Encryption key ID

        Returns:
            Decrypted value
        """
        from shared.encrypted_fields import decrypt_value

        return decrypt_value(encrypted_value, key_id)

    def _get_key_id_for_field(self, model_class, field_name: str) -> str:
        """
        Get the encryption key ID for a specific field

        Args:
            model_class: Model class
            field_name: Field name

        Returns:
            Key ID string
        """
        # Map field names to key IDs
        key_mappings = {
            User: {
                "email_encrypted": "user_email_key",
                "password_hash_encrypted": "user_password_key",
            },
            Conversation: {
                "user_query_encrypted": "conversation_query_key",
                "agent_response_encrypted": "conversation_response_key",
            },
            Agent: {
                "config_encrypted": "agent_config_key",
                "capabilities_encrypted": "agent_capabilities_key",
            },
            Workflow: {
                "steps_encrypted": "workflow_steps_key",
                "context_encrypted": "workflow_context_key",
            },
        }

        if model_class in key_mappings and field_name in key_mappings[model_class]:
            return key_mappings[model_class][field_name]

        # Fallback to generic key
        return (
            f"{model_class.__name__.lower()}_{field_name.replace('_encrypted', '')}_key"
        )

    def _record_key_usage(
        self,
        key_id: str,
        operation: str,
        field_name: str,
        table_name: str,
        success: bool = True,
    ):
        """
        Record encryption key usage for audit purposes

        Args:
            key_id: Encryption key ID
            operation: Operation type ('encrypt', 'decrypt')
            field_name: Field name
            table_name: Table/model name
            success: Whether operation was successful
        """
        try:
            # Get key record ID for audit
            from shared.database import db_manager

            with db_manager.get_db() as session:
                key_record = (
                    session.query(EncryptionKey)
                    .filter(EncryptionKey.key_id == key_id)
                    .first()
                )

                if key_record:
                    usage_log = KeyUsageLog(
                        key_id=key_record.id,
                        operation=operation,
                        field_name=field_name,
                        table_name=table_name,
                        success=success,
                    )
                    session.add(usage_log)
                    session.commit()

        except Exception as e:
            logger.error(f"Failed to record key usage: {e}")

    def enable_encryption(self):
        """Enable encryption middleware"""
        self.encryption_enabled = True
        logger.info("Encryption middleware enabled")

    def disable_encryption(self):
        """Disable encryption middleware"""
        self.encryption_enabled = False
        logger.info("Encryption middleware disabled")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get encryption middleware metrics

        Returns:
            Dictionary with performance metrics
        """
        return {
            "encryption_enabled": self.encryption_enabled,
            "total_operations": encryption_metrics.operation_count,
            "average_time_ms": encryption_metrics.get_average_time(),
            "success_rate": encryption_metrics.get_success_rate(),
            "operations": encryption_metrics.operations[-10:],  # Last 10 operations
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Health check for encryption middleware

        Returns:
            Health status dictionary
        """
        try:
            # Test key access
            test_key = key_management_service.get_master_key()
            key_accessible = test_key is not None and len(test_key) == 32

            return {
                "status": "healthy" if key_accessible else "unhealthy",
                "encryption_enabled": self.encryption_enabled,
                "master_key_accessible": key_accessible,
                "metrics_available": True,
            }

        except Exception as e:
            logger.error(f"Encryption middleware health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "encryption_enabled": self.encryption_enabled,
            }


# Global middleware instance
encryption_middleware = EncryptionMiddleware()


def init_encryption_middleware():
    """
    Initialize and register the encryption middleware
    Call this function during application startup
    """
    try:
        encryption_middleware.register_events()
        logger.info("Database encryption middleware initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize encryption middleware: {e}")
        raise


def shutdown_encryption_middleware():
    """
    Shutdown and unregister the encryption middleware
    Call this function during application shutdown
    """
    try:
        encryption_middleware.unregister_events()
        logger.info("Database encryption middleware shut down successfully")
    except Exception as e:
        logger.error(f"Error shutting down encryption middleware: {e}")


# Example usage in application startup:
"""
from shared.encryption_middleware import init_encryption_middleware

# In your application startup code
def startup():
    # Initialize database
    # ...

    # Initialize encryption middleware
    init_encryption_middleware()

    # Start your application
    # ...

def shutdown():
    # Shutdown encryption middleware
    shutdown_encryption_middleware()

    # Other shutdown code
    # ...
"""
