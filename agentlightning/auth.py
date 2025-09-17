"""
Authentication and authorization utilities for Agent Lightning
"""

import hashlib
import secrets
import time
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from shared.data_access import DataAccessLayer
from shared.models import ApiKey
from .rbac import rbac_manager, Permission
from services.api_key_rotation_service import api_key_rotation_service

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


class AuthManager:
    """Manages API key authentication and authorization"""

    def __init__(self, dal: Optional[DataAccessLayer] = None):
        self.dal = dal or DataAccessLayer("auth")

    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def generate_api_key(self) -> str:
        """Generate a new random API key"""
        return secrets.token_urlsafe(32)

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key and return key info if valid

        Args:
            api_key: The API key to validate

        Returns:
            Key info dict if valid, None if invalid
        """
        if not api_key:
            return None

        key_hash = self.hash_api_key(api_key)

        try:
            with self.dal.db.get_db() as session:
                db_key = session.query(ApiKey).filter_by(
                    key_hash=key_hash,
                    is_active=True
                ).first()

                if not db_key:
                    return None

                # Check expiration
                if db_key.is_expired():
                    logger.warning(f"API key {db_key.id} has expired")
                    return None

                # Update usage stats
                db_key.last_used_at = datetime.utcnow()
                db_key.usage_count += 1
                session.commit()

                return {
                    'id': str(db_key.id),
                    'name': db_key.name,
                    'user_id': db_key.user_id,
                    'permissions': db_key.permissions,
                    'rate_limit_requests': db_key.rate_limit_requests,
                    'rate_limit_window': db_key.rate_limit_window
                }

        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None

    def create_api_key(self, name: str, user_id: str = None,
                      permissions: list = None, expires_in_days: int = 365) -> Dict[str, Any]:
        """
        Create a new API key

        Args:
            name: Human-readable name for the key
            user_id: Associated user ID
            permissions: List of permissions
            expires_in_days: Days until expiration

        Returns:
            Dict with key info and the actual key
        """
        api_key = self.generate_api_key()
        key_hash = self.hash_api_key(api_key)
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        try:
            with self.dal.db.get_db() as session:
                db_key = ApiKey(
                    key_hash=key_hash,
                    name=name,
                    user_id=user_id,
                    permissions=permissions or ['read'],
                    expires_at=expires_at
                )
                session.add(db_key)
                session.commit()

                return {
                    'id': str(db_key.id),
                    'key': api_key,  # Only returned once!
                    'name': name,
                    'user_id': user_id,
                    'permissions': db_key.permissions,
                    'expires_at': expires_at.isoformat()
                }

        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            raise HTTPException(status_code=500, detail="Failed to create API key")

    def rotate_api_key(self, api_key_id: str, user_id: str = None,
                      reason: str = "manual") -> Dict[str, Any]:
        """
        Rotate an API key

        Args:
            api_key_id: ID of the API key to rotate
            user_id: User performing the rotation
            reason: Reason for rotation

        Returns:
            Dict with rotation result and new key info
        """
        result = api_key_rotation_service.rotate_api_key(
            api_key_id=api_key_id,
            user_id=user_id,
            reason=reason
        )

        if result.success:
            return {
                'success': True,
                'api_key_id': result.api_key_id,
                'new_key': result.new_key,
                'rotation_history_id': result.rotation_history_id,
                'message': 'API key rotated successfully'
            }
        else:
            return {
                'success': False,
                'api_key_id': result.api_key_id,
                'error': result.error_message,
                'message': 'Failed to rotate API key'
            }

    def get_api_key_rotation_status(self, api_key_id: str) -> Optional[Dict[str, Any]]:
        """
        Get rotation status for an API key

        Args:
            api_key_id: API key ID

        Returns:
            Dict with rotation status information
        """
        try:
            with self.dal.db.get_db() as session:
                api_key = session.query(ApiKey).filter(
                    ApiKey.id == api_key_id
                ).first()

                if not api_key:
                    return None

                return {
                    'api_key_id': str(api_key.id),
                    'name': api_key.name,
                    'is_rotation_enabled': api_key.is_rotation_enabled,
                    'rotation_locked': api_key.rotation_locked,
                    'last_rotated_at': api_key.last_rotated_at.isoformat() if api_key.last_rotated_at else None,
                    'next_rotation_at': api_key.next_rotation_at.isoformat() if api_key.next_rotation_at else None,
                    'rotation_count': api_key.rotation_count,
                    'is_due_for_rotation': api_key.is_due_for_rotation(),
                    'days_until_rotation': (api_key.next_rotation_at - datetime.utcnow()).days if api_key.next_rotation_at else None
                }

        except Exception as e:
            logger.error(f"Error getting rotation status for key {api_key_id}: {e}")
            return None

    def get_rotation_history(self, api_key_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get rotation history for an API key

        Args:
            api_key_id: API key ID
            limit: Maximum number of records to return

        Returns:
            List of rotation history records
        """
        return api_key_rotation_service.get_rotation_history(api_key_id, limit)

    def enable_rotation(self, api_key_id: str, policy_id: str = None) -> bool:
        """
        Enable automatic rotation for an API key

        Args:
            api_key_id: API key ID
            policy_id: Rotation policy ID (optional, uses default if not provided)

        Returns:
            True if enabled successfully
        """
        try:
            with self.dal.db.get_db() as session:
                api_key = session.query(ApiKey).filter(
                    ApiKey.id == api_key_id
                ).first()

                if not api_key:
                    return False

                api_key.is_rotation_enabled = True
                api_key.rotation_locked = False
                if policy_id:
                    api_key.rotation_policy_id = policy_id

                # Calculate next rotation date
                if not api_key.next_rotation_at:
                    # Set initial rotation date (e.g., 90 days from now)
                    api_key.next_rotation_at = datetime.utcnow() + timedelta(days=90)

                session.commit()

                logger.info(f"Enabled rotation for API key {api_key_id}")
                return True

        except Exception as e:
            logger.error(f"Error enabling rotation for key {api_key_id}: {e}")
            return False

    def disable_rotation(self, api_key_id: str) -> bool:
        """
        Disable automatic rotation for an API key

        Args:
            api_key_id: API key ID

        Returns:
            True if disabled successfully
        """
        try:
            with self.dal.db.get_db() as session:
                api_key = session.query(ApiKey).filter(
                    ApiKey.id == api_key_id
                ).first()

                if not api_key:
                    return False

                api_key.is_rotation_enabled = False
                session.commit()

                logger.info(f"Disabled rotation for API key {api_key_id}")
                return True

        except Exception as e:
            logger.error(f"Error disabling rotation for key {api_key_id}: {e}")
            return False

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with username and password
        Handles encrypted user data automatically via middleware

        Args:
            username: User's username
            password: User's password (plaintext)

        Returns:
            User info dict if authentication successful, None otherwise
        """
        try:
            from shared.models import User
            from shared.encrypted_fields import decrypt_value

            with self.dal.db.get_db() as session:
                # Query user - encryption middleware will handle decryption
                user = session.query(User).filter(
                    User.username == username,
                    User.is_active == True
                ).first()

                if not user:
                    logger.warning(f"Authentication failed: user {username} not found")
                    return None

                # Get password hash - will be automatically decrypted by middleware
                stored_hash = user.password_hash_encrypted or user.password_hash

                if not stored_hash:
                    logger.error(f"No password hash found for user {username}")
                    return None

                # Verify password
                if not self._verify_password(password, stored_hash):
                    logger.warning(f"Authentication failed: invalid password for user {username}")
                    return None

                # Update last login
                user.updated_at = datetime.utcnow()
                session.commit()

                logger.info(f"User {username} authenticated successfully")

                return {
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,  # Will be automatically decrypted by middleware
                    'role': user.role,
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'last_login': user.updated_at.isoformat() if user.updated_at else None
                }

        except Exception as e:
            logger.error(f"Error authenticating user {username}: {e}")
            return None

    def create_user(self, username: str, email: str, password: str,
                   role: str = 'user') -> Optional[Dict[str, Any]]:
        """
        Create a new user with encrypted data
        Handles encryption automatically via middleware

        Args:
            username: Desired username
            email: User's email address
            password: Plaintext password
            role: User role

        Returns:
            User info dict if creation successful, None otherwise
        """
        try:
            from shared.models import User

            # Hash the password
            password_hash = self._hash_password(password)

            with self.dal.db.get_db() as session:
                # Check if user already exists
                existing = session.query(User).filter(
                    (User.username == username) | (User.email == email)
                ).first()

                if existing:
                    logger.warning(f"User creation failed: {username} or {email} already exists")
                    return None

                # Create new user - encryption middleware will handle encryption
                user = User(
                    username=username,
                    email=email,  # Will be automatically encrypted by middleware
                    password_hash=password_hash,  # Will be automatically encrypted by middleware
                    role=role,
                    is_active=True
                )

                session.add(user)
                session.commit()

                logger.info(f"User {username} created successfully")

                return {
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat() if user.created_at else None
                }

        except Exception as e:
            logger.error(f"Error creating user {username}: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email address
        Handles encrypted email data automatically via middleware

        Args:
            email: User's email address

        Returns:
            User info dict if found, None otherwise
        """
        try:
            from shared.models import User

            with self.dal.db.get_db() as session:
                # Query by email - encryption middleware handles decryption
                user = session.query(User).filter(
                    User.email == email,  # Middleware will decrypt for comparison
                    User.is_active == True
                ).first()

                if not user:
                    return None

                return {
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,  # Will be automatically decrypted by middleware
                    'role': user.role,
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat() if user.created_at else None
                }

        except Exception as e:
            logger.error(f"Error getting user by email {email}: {e}")
            return None

    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        try:
            import bcrypt
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        except ImportError:
            # Fallback to hashlib if bcrypt not available
            import hashlib
            return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        try:
            import bcrypt
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except ImportError:
            # Fallback to hashlib if bcrypt not available
            import hashlib
            return hashlib.sha256(password.encode('utf-8')).hexdigest() == hashed_password


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self):
        self.requests = {}  # key -> list of timestamps

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """
        Check if request is allowed under rate limit

        Args:
            key: Rate limit key (e.g., API key ID)
            max_requests: Maximum requests in window
            window_seconds: Time window in seconds

        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        window_start = now - window_seconds

        if key not in self.requests:
            self.requests[key] = []

        # Remove old requests outside the window
        self.requests[key] = [t for t in self.requests[key] if t > window_start]

        # Check if under limit
        if len(self.requests[key]) < max_requests:
            self.requests[key].append(now)
            return True

        return False


# Global instances
auth_manager = AuthManager()
rate_limiter = RateLimiter()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user/key info

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        User/key info dict

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    api_key = credentials.credentials
    key_info = auth_manager.validate_api_key(api_key)

    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Check rate limit
    rate_key = f"api_key:{key_info['id']}"
    if not rate_limiter.is_allowed(
        rate_key,
        key_info['rate_limit_requests'],
        key_info['rate_limit_window']
    ):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return key_info


def require_permission(permission: str):
    """
    Create a dependency that requires a specific permission (legacy API key based)

    Args:
        permission: Required permission string

    Returns:
        Dependency function
    """
    async def permission_checker(user: Dict = Depends(get_current_user)):
        if permission not in user.get('permissions', []):
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return user

    return permission_checker


def require_rbac_permission(permission: Permission):
    """
    Create a dependency that requires a specific RBAC permission

    Args:
        permission: Required RBAC permission

    Returns:
        Dependency function
    """
    return rbac_manager.require_permission(permission)


# Convenience dependencies (legacy API key based)
require_read = require_permission('read')
require_write = require_permission('write')
require_admin = require_permission('admin')

# RBAC-based convenience dependencies
require_server_read = rbac_manager.require_permission(Permission.SERVER_READ)
require_server_write = rbac_manager.require_permission(Permission.SERVER_WRITE)
require_server_admin = rbac_manager.require_permission(Permission.SERVER_ADMIN)

require_task_read = rbac_manager.require_permission(Permission.TASK_READ)
require_task_write = rbac_manager.require_permission(Permission.TASK_WRITE)
require_task_admin = rbac_manager.require_permission(Permission.TASK_ADMIN)

require_resource_read = rbac_manager.require_permission(Permission.RESOURCE_READ)
require_resource_write = rbac_manager.require_permission(Permission.RESOURCE_WRITE)

require_rollout_read = rbac_manager.require_permission(Permission.ROLLOUT_READ)
require_rollout_write = rbac_manager.require_permission(Permission.ROLLOUT_WRITE)