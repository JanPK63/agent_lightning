"""
API Key Rotation Service
Handles automatic rotation, expiration, and notification management for API keys
"""

import uuid
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from shared.database import db_manager
from shared.models import ApiKey, ApiKeyRotationPolicy, ApiKeyRotationHistory
from agentlightning.auth import AuthManager
from shared.events import EventBus, EventChannel

logger = logging.getLogger(__name__)


@dataclass
class RotationResult:
    """Result of a key rotation operation"""
    success: bool
    api_key_id: str
    new_key: str
    old_key_hash: str
    new_key_hash: str
    rotation_history_id: str
    error_message: Optional[str] = None


@dataclass
class NotificationInfo:
    """Information for rotation notifications"""
    api_key_id: str
    key_name: str
    user_id: str
    days_until_rotation: int
    rotation_date: datetime


class ApiKeyRotationService:
    """Service for managing API key rotation operations"""

    def __init__(self, auth_manager: Optional[AuthManager] = None):
        """Initialize the rotation service

        Args:
            auth_manager: AuthManager instance (optional, will create if not provided)
        """
        self.auth_manager = auth_manager or AuthManager()
        self.event_bus = EventBus("api_key_rotation")
        self.event_bus.start()

    def rotate_api_key(self, api_key_id: str, user_id: str = None,
                      reason: str = "scheduled", notes: str = None) -> RotationResult:
        """
        Rotate a single API key

        Args:
            api_key_id: ID of the API key to rotate
            user_id: User performing the rotation (optional)
            reason: Reason for rotation
            notes: Additional notes

        Returns:
            RotationResult with operation details
        """
        try:
            with db_manager.get_db() as session:
                # Get current API key
                api_key = session.query(ApiKey).filter(
                    ApiKey.id == api_key_id,
                    ApiKey.is_active == True
                ).first()

                if not api_key:
                    return RotationResult(
                        success=False,
                        api_key_id=api_key_id,
                        new_key="",
                        old_key_hash="",
                        new_key_hash="",
                        rotation_history_id="",
                        error_message="API key not found or not active"
                    )

                # Check if rotation is allowed
                if not api_key.is_rotation_enabled:
                    return RotationResult(
                        success=False,
                        api_key_id=api_key_id,
                        new_key="",
                        old_key_hash="",
                        new_key_hash="",
                        rotation_history_id="",
                        error_message="Rotation is disabled for this key"
                    )

                if api_key.rotation_locked:
                    return RotationResult(
                        success=False,
                        api_key_id=api_key_id,
                        new_key="",
                        old_key_hash="",
                        new_key_hash="",
                        rotation_history_id="",
                        error_message="API key is locked from rotation"
                    )

                # Generate new key
                new_key = self.auth_manager.generate_api_key()
                new_key_hash = self.auth_manager.hash_api_key(new_key)

                # Get rotation policy
                policy = None
                if api_key.rotation_policy_id:
                    policy = session.query(ApiKeyRotationPolicy).filter(
                        ApiKeyRotationPolicy.id == api_key.rotation_policy_id,
                        ApiKeyRotationPolicy.is_active == True
                    ).first()

                if not policy:
                    # Use default policy
                    policy = session.query(ApiKeyRotationPolicy).filter(
                        ApiKeyRotationPolicy.is_default == True,
                        ApiKeyRotationPolicy.is_active == True
                    ).first()

                # Calculate new expiration date
                new_expires_at = None
                if policy and policy.grace_period_days:
                    new_expires_at = datetime.utcnow() + timedelta(days=policy.grace_period_days)

                # Store old values for history
                old_key_hash = api_key.key_hash
                old_expires_at = api_key.expires_at

                # Update API key
                api_key.key_hash = new_key_hash
                api_key.expires_at = new_expires_at
                api_key.last_rotated_at = datetime.utcnow()
                api_key.rotation_count += 1

                # Calculate next rotation date
                if policy:
                    api_key.next_rotation_at = datetime.utcnow() + timedelta(days=policy.auto_rotate_days)
                else:
                    api_key.next_rotation_at = datetime.utcnow() + timedelta(days=90)  # Default 90 days

                api_key.updated_at = datetime.utcnow()

                # Create rotation history record
                history = ApiKeyRotationHistory(
                    api_key_id=api_key_id,
                    old_key_hash=old_key_hash,
                    new_key_hash=new_key_hash,
                    old_expires_at=old_expires_at,
                    new_expires_at=new_expires_at,
                    rotated_by=user_id,
                    rotation_reason=reason,
                    rotation_policy_id=policy.id if policy else None,
                    notes=notes
                )

                session.add(history)
                session.commit()

                # Emit rotation event
                self.event_bus.emit(
                    EventChannel.SYSTEM_ALERT,
                    {
                        "type": "api_key_rotated",
                        "api_key_id": api_key_id,
                        "user_id": api_key.user_id,
                        "rotation_history_id": str(history.id),
                        "reason": reason
                    }
                )

                logger.info(f"Successfully rotated API key {api_key_id} for user {api_key.user_id}")

                return RotationResult(
                    success=True,
                    api_key_id=api_key_id,
                    new_key=new_key,
                    old_key_hash=old_key_hash,
                    new_key_hash=new_key_hash,
                    rotation_history_id=str(history.id)
                )

        except Exception as e:
            logger.error(f"Error rotating API key {api_key_id}: {e}")
            return RotationResult(
                success=False,
                api_key_id=api_key_id,
                new_key="",
                old_key_hash="",
                new_key_hash="",
                rotation_history_id="",
                error_message=str(e)
            )

    def get_keys_due_for_rotation(self, days_ahead: int = 0) -> List[Dict[str, Any]]:
        """
        Get API keys that are due for rotation

        Args:
            days_ahead: Look ahead this many days (0 = only overdue keys)

        Returns:
            List of keys due for rotation with metadata
        """
        try:
            with db_manager.get_db() as session:
                query = session.query(ApiKey).filter(
                    ApiKey.is_active == True,
                    ApiKey.is_rotation_enabled == True,
                    ApiKey.rotation_locked == False,
                    ApiKey.next_rotation_at.isnot(None)
                )

                if days_ahead > 0:
                    cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
                    query = query.filter(ApiKey.next_rotation_at <= cutoff_date)
                else:
                    query = query.filter(ApiKey.next_rotation_at <= datetime.utcnow())

                keys = query.all()

                result = []
                for key in keys:
                    days_until = 0
                    if key.next_rotation_at:
                        days_until = (key.next_rotation_at - datetime.utcnow()).days

                    result.append({
                        'id': str(key.id),
                        'name': key.name,
                        'user_id': key.user_id,
                        'next_rotation_at': key.next_rotation_at.isoformat() if key.next_rotation_at else None,
                        'days_until_rotation': max(0, days_until),
                        'rotation_count': key.rotation_count,
                        'last_rotated_at': key.last_rotated_at.isoformat() if key.last_rotated_at else None
                    })

                return result

        except Exception as e:
            logger.error(f"Error getting keys due for rotation: {e}")
            return []

    def get_pending_notifications(self) -> List[NotificationInfo]:
        """
        Get API keys that need rotation notifications

        Returns:
            List of NotificationInfo objects
        """
        try:
            with db_manager.get_db() as session:
                # Get all active rotation policies
                policies = session.query(ApiKeyRotationPolicy).filter(
                    ApiKeyRotationPolicy.is_active == True
                ).all()

                notifications = []

                for policy in policies:
                    # Find keys using this policy that need notification
                    notify_cutoff = datetime.utcnow() + timedelta(days=policy.notify_before_days)

                    keys = session.query(ApiKey).filter(
                        ApiKey.rotation_policy_id == policy.id,
                        ApiKey.is_active == True,
                        ApiKey.is_rotation_enabled == True,
                        ApiKey.rotation_locked == False,
                        ApiKey.next_rotation_at <= notify_cutoff,
                        ApiKey.next_rotation_at > datetime.utcnow()
                    ).all()

                    for key in keys:
                        days_until = (key.next_rotation_at - datetime.utcnow()).days
                        notifications.append(NotificationInfo(
                            api_key_id=str(key.id),
                            key_name=key.name,
                            user_id=key.user_id,
                            days_until_rotation=days_until,
                            rotation_date=key.next_rotation_at
                        ))

                return notifications

        except Exception as e:
            logger.error(f"Error getting pending notifications: {e}")
            return []

    def cleanup_expired_keys(self) -> int:
        """
        Clean up expired API keys that are past their grace period

        Returns:
            Number of keys cleaned up
        """
        try:
            with db_manager.get_db() as session:
                # Find keys that are expired and past grace period
                expired_keys = session.query(ApiKey).filter(
                    ApiKey.is_active == True,
                    ApiKey.expires_at.isnot(None),
                    ApiKey.expires_at <= datetime.utcnow()
                ).all()

                cleaned_count = 0
                for key in expired_keys:
                    # Check if we should deactivate (past grace period)
                    policy = None
                    if key.rotation_policy_id:
                        policy = session.query(ApiKeyRotationPolicy).filter(
                            ApiKeyRotationPolicy.id == key.rotation_policy_id
                        ).first()

                    grace_period_days = policy.grace_period_days if policy else 30
                    grace_cutoff = key.last_rotated_at + timedelta(days=grace_period_days) if key.last_rotated_at else datetime.utcnow()

                    if datetime.utcnow() > grace_cutoff:
                        key.is_active = False
                        key.updated_at = datetime.utcnow()
                        cleaned_count += 1

                        logger.info(f"Deactivated expired API key {key.id}")

                session.commit()

                # Emit cleanup event
                if cleaned_count > 0:
                    self.event_bus.emit(
                        EventChannel.SYSTEM_ALERT,
                        {
                            "type": "expired_keys_cleaned",
                            "count": cleaned_count
                        }
                    )

                return cleaned_count

        except Exception as e:
            logger.error(f"Error cleaning up expired keys: {e}")
            return 0

    def bulk_rotate_keys(self, key_ids: List[str], user_id: str = None,
                        reason: str = "bulk_rotation") -> List[RotationResult]:
        """
        Rotate multiple API keys in bulk

        Args:
            key_ids: List of API key IDs to rotate
            user_id: User performing the rotation
            reason: Reason for rotation

        Returns:
            List of RotationResult objects
        """
        results = []

        for key_id in key_ids:
            result = self.rotate_api_key(
                api_key_id=key_id,
                user_id=user_id,
                reason=reason
            )
            results.append(result)

        successful = sum(1 for r in results if r.success)
        logger.info(f"Bulk rotation completed: {successful}/{len(results)} successful")

        return results

    def get_rotation_history(self, api_key_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get rotation history for an API key

        Args:
            api_key_id: API key ID
            limit: Maximum number of records to return

        Returns:
            List of rotation history records
        """
        try:
            with db_manager.get_db() as session:
                history = session.query(ApiKeyRotationHistory).filter(
                    ApiKeyRotationHistory.api_key_id == api_key_id
                ).order_by(
                    ApiKeyRotationHistory.rotated_at.desc()
                ).limit(limit).all()

                return [h.to_dict() for h in history]

        except Exception as e:
            logger.error(f"Error getting rotation history for key {api_key_id}: {e}")
            return []

    def __del__(self):
        """Cleanup event bus on destruction"""
        try:
            if hasattr(self, 'event_bus'):
                self.event_bus.stop()
        except:
            pass


# Global instance
api_key_rotation_service = ApiKeyRotationService()