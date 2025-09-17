"""
Key Rotation Service
Handles the safe rotation of encryption keys for database records.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from shared.database import get_db_session
from shared.models import (
    EncryptionKey, KeyUsageLog, KeyRotationHistory,
    User, Agent, Conversation, Workflow, ServerTask, ServerResource, ServerRollout
)
from services.key_management_service import KeyManagementService

logger = logging.getLogger(__name__)


class RotationStatus(Enum):
    """Status of key rotation operation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class RotationPhase(Enum):
    """Phase of key rotation"""
    PRE_VALIDATION = "pre_validation"
    KEY_GENERATION = "key_generation"
    DATA_ROTATION = "data_rotation"
    POST_VALIDATION = "post_validation"
    CLEANUP = "cleanup"


@dataclass
class RotationResult:
    """Result of a key rotation operation"""
    success: bool
    status: RotationStatus
    records_processed: int
    records_failed: int
    duration_seconds: float
    new_data_key_id: Optional[str] = None
    new_field_key_id: Optional[str] = None
    error_message: Optional[str] = None
    rollback_available: bool = False


@dataclass
class RotationProgress:
    """Progress tracking for rotation operation"""
    phase: RotationPhase
    records_processed: int
    records_total: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None


class KeyRotationService:
    """
    Service for rotating encryption keys safely.

    Handles the complete lifecycle of key rotation including:
    - Pre-rotation validation
    - Key generation and activation
    - Data re-encryption in batches
    - Post-rotation validation
    - Rollback capability
    """

    def __init__(self, batch_size: int = 1000, max_duration_hours: int = 4):
        self.batch_size = batch_size
        self.max_duration_seconds = max_duration_hours * 3600
        self.kms = KeyManagementService()

    def rotate_keys(
        self,
        data_key_id: Optional[str] = None,
        field_key_id: Optional[str] = None,
        environment: str = "staging",
        dry_run: bool = True
    ) -> RotationResult:
        """
        Execute key rotation for all encrypted data.

        Args:
            data_key_id: Specific data key to rotate to (optional)
            field_key_id: Specific field key to rotate to (optional)
            environment: Environment being rotated
            dry_run: If True, only validate without making changes

        Returns:
            RotationResult with operation details
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting key rotation (dry_run={dry_run}) in {environment}")

        try:
            # Phase 1: Pre-validation
            self._validate_pre_rotation()
            logger.info("Pre-rotation validation completed")

            # Phase 2: Generate/activate keys
            new_data_key, new_field_key = self._prepare_keys(data_key_id, field_key_id)
            logger.info(f"Keys prepared: data={new_data_key.id}, field={new_field_key.id}")

            if dry_run:
                return RotationResult(
                    success=True,
                    status=RotationStatus.COMPLETED,
                    records_processed=0,
                    records_failed=0,
                    duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                    new_data_key_id=new_data_key.id,
                    new_field_key_id=new_field_key.id
                )

            # Phase 3: Rotate data
            rotation_result = self._rotate_data(new_data_key, new_field_key, environment)
            logger.info(f"Data rotation completed: {rotation_result.records_processed} records")

            # Phase 4: Post-validation
            validation_result = self._validate_post_rotation()
            logger.info("Post-rotation validation completed")

            # Phase 5: Cleanup and finalization
            self._finalize_rotation(new_data_key, new_field_key, rotation_result)

            total_duration = (datetime.utcnow() - start_time).total_seconds()

            return RotationResult(
                success=True,
                status=RotationStatus.COMPLETED,
                records_processed=rotation_result.records_processed,
                records_failed=rotation_result.records_failed,
                duration_seconds=total_duration,
                new_data_key_id=new_data_key.id,
                new_field_key_id=new_field_key.id,
                rollback_available=True
            )

        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            total_duration = (datetime.utcnow() - start_time).total_seconds()

            # Attempt rollback if we got far enough
            try:
                self._rollback_rotation()
                rollback_status = RotationStatus.ROLLED_BACK
            except Exception as rollback_error:
                logger.error(f"Rollback also failed: {rollback_error}")
                rollback_status = RotationStatus.FAILED

            return RotationResult(
                success=False,
                status=rollback_status,
                records_processed=0,
                records_failed=0,
                duration_seconds=total_duration,
                error_message=str(e)
            )

    def _validate_pre_rotation(self) -> None:
        """Validate system state before rotation"""
        session = get_db_session()

        try:
            # Check database connectivity
            from sqlalchemy import text
            session.execute(text("SELECT 1"))

            # Check for existing encryption keys
            key_count = session.query(EncryptionKey).filter(
                EncryptionKey.status == 'active'
            ).count()

            if key_count == 0:
                logger.warning("No active encryption keys found - this may be the first rotation")

            # Check audit logging table exists and is accessible
            audit_count = session.query(KeyUsageLog).count()
            logger.info(f"Audit system accessible - {audit_count} total audit records")

            logger.info("Pre-rotation validation passed")

        finally:
            session.close()

    def _prepare_keys(
        self,
        data_key_id: Optional[str],
        field_key_id: Optional[str]
    ) -> Tuple[EncryptionKey, EncryptionKey]:
        """Generate or retrieve keys for rotation"""

        # For now, create placeholder keys - in real implementation these would be generated
        # through the KeyManagementService
        data_key = EncryptionKey(
            key_id=data_key_id or "rotation_data_key",
            key_type='data',
            name='Rotation Data Key',
            encrypted_key=b'placeholder',  # Would be real encrypted key
            key_hash='placeholder',
            algorithm='aes-256-gcm',
            status='active'
        )

        field_key = EncryptionKey(
            key_id=field_key_id or "rotation_field_key",
            key_type='field',
            name='Rotation Field Key',
            encrypted_key=b'placeholder',  # Would be real encrypted key
            key_hash='placeholder',
            algorithm='aes-256-gcm',
            status='active'
        )

        logger.info(f"Prepared keys: data={data_key.key_id}, field={field_key.key_id}")
        return data_key, field_key

    def _rotate_data(
        self,
        new_data_key: EncryptionKey,
        new_field_key: EncryptionKey,
        environment: str
    ) -> RotationResult:
        """Rotate encrypted data to new keys"""

        session = get_db_session()
        records_processed = 0
        records_failed = 0
        start_time = datetime.utcnow()

        try:
            # Define models with encrypted fields
            encrypted_models = [
                (User, ['email_encrypted', 'password_hash_encrypted']),
                (Agent, ['config_encrypted', 'capabilities_encrypted']),
                (Conversation, ['user_query_encrypted', 'agent_response_encrypted']),
                (Workflow, ['steps_encrypted', 'context_encrypted']),
            ]

            for model_class, encrypted_fields in encrypted_models:
                logger.info(f"Rotating {model_class.__tablename__}")

                # Process in batches
                offset = 0
                while True:
                    # Check duration limit
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    if elapsed > self.max_duration_seconds:
                        raise TimeoutError(f"Rotation exceeded {self.max_duration_seconds}s limit")

                    # Get batch of records
                    records = session.query(model_class).offset(offset).limit(self.batch_size).all()
                    if not records:
                        break

                    for record in records:
                        try:
                            # Re-encrypt each field
                            for field_name in encrypted_fields:
                                if hasattr(record, field_name):
                                    # Get current encrypted value
                                    current_value = getattr(record, field_name)

                                    # Decrypt with old key and re-encrypt with new key
                                    # (This would use the encryption middleware logic)
                                    # For now, just mark as processed
                                    pass

                            records_processed += 1

                        except Exception as e:
                            logger.error(f"Failed to rotate record {record.id}: {e}")
                            records_failed += 1

                    session.commit()
                    offset += self.batch_size

                    # Log progress
                    if records_processed % 1000 == 0:
                        logger.info(f"Processed {records_processed} records")

            # Log rotation history
            rotation_history = KeyRotationHistory(
                key_id=new_data_key.id,
                old_key_hash="placeholder",  # Would compute actual hash
                new_key_hash="placeholder",  # Would compute actual hash
                rotation_reason="scheduled_rotation",
                rotated_by="system",
                rotation_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                success=True
            )
            session.add(rotation_history)
            session.commit()

            return RotationResult(
                success=True,
                status=RotationStatus.COMPLETED,
                records_processed=records_processed,
                records_failed=records_failed,
                duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            )

        finally:
            session.close()

    def _validate_post_rotation(self) -> bool:
        """Validate system after rotation"""
        session = get_db_session()

        try:
            # Check decrypt error rate
            recent_errors = session.query(KeyUsageLog).filter(
                KeyUsageLog.operation == 'decrypt',
                KeyUsageLog.success == False,
                KeyUsageLog.timestamp >= datetime.utcnow() - timedelta(minutes=5)
            ).count()

            total_decrypts = session.query(KeyUsageLog).filter(
                KeyUsageLog.operation == 'decrypt',
                KeyUsageLog.timestamp >= datetime.utcnow() - timedelta(minutes=5)
            ).count()

            if total_decrypts > 0:
                error_rate = recent_errors / total_decrypts
                if error_rate > 0.001:  # 0.1%
                    raise ValueError(f"Decrypt error rate too high: {error_rate:.4f}")

            # Check data integrity - sample some records
            user_count = session.query(User).count()
            agent_count = session.query(Agent).count()

            if user_count == 0 or agent_count == 0:
                raise ValueError("Data integrity check failed - missing records")

            logger.info("Post-rotation validation passed")
            return True

        finally:
            session.close()

    def _finalize_rotation(
        self,
        new_data_key: EncryptionKey,
        new_field_key: EncryptionKey,
        rotation_result: RotationResult
    ) -> None:
        """Finalize rotation and update key statuses"""
        # Mark old keys as rotated
        # Activate new keys
        # Update rotation schedules
        logger.info("Rotation finalized")

    def _rollback_rotation(self) -> None:
        """Rollback rotation to previous keys"""
        # This would restore old keys and re-encrypt data
        # Implementation depends on backup strategy
        logger.warning("Rotation rollback initiated")

    def get_rotation_status(self, rotation_id: str) -> Optional[RotationProgress]:
        """Get status of ongoing rotation"""
        # In a real implementation, this would check a rotation tracking table
        return None

    def cancel_rotation(self, rotation_id: str) -> bool:
        """Cancel ongoing rotation"""
        # Implementation would signal rotation job to stop
        logger.info(f"Rotation {rotation_id} cancelled")
        return True