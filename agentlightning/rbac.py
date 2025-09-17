"""
Role-Based Access Control (RBAC) system for Agent Lightning
"""

import os
import logging
from typing import Dict, List, Set, Optional, Any, Callable
from enum import Enum
from functools import wraps

from fastapi import HTTPException, Depends

logger = logging.getLogger(__name__)


class Role(Enum):
    """System roles with hierarchical permissions"""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    AGENT = "agent"
    GUEST = "guest"


class Permission(Enum):
    """System permissions"""
    # Server management
    SERVER_READ = "server:read"
    SERVER_WRITE = "server:write"
    SERVER_ADMIN = "server:admin"

    # Task management
    TASK_READ = "task:read"
    TASK_WRITE = "task:write"
    TASK_ASSIGN = "task:assign"
    TASK_ADMIN = "task:admin"

    # Resource management
    RESOURCE_READ = "resource:read"
    RESOURCE_WRITE = "resource:write"
    RESOURCE_ADMIN = "resource:admin"

    # Rollout management
    ROLLOUT_READ = "rollout:read"
    ROLLOUT_WRITE = "rollout:write"
    ROLLOUT_ADMIN = "rollout:admin"

    # User management
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_ADMIN = "user:admin"

    # System management
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"


# Role-permission mappings
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        # Admin has all permissions
        Permission.SERVER_ADMIN, Permission.SERVER_READ, Permission.SERVER_WRITE,
        Permission.TASK_ADMIN, Permission.TASK_READ, Permission.TASK_WRITE, Permission.TASK_ASSIGN,
        Permission.RESOURCE_ADMIN, Permission.RESOURCE_READ, Permission.RESOURCE_WRITE,
        Permission.ROLLOUT_ADMIN, Permission.ROLLOUT_READ, Permission.ROLLOUT_WRITE,
        Permission.USER_ADMIN, Permission.USER_READ, Permission.USER_WRITE,
        Permission.SYSTEM_ADMIN, Permission.SYSTEM_READ, Permission.SYSTEM_WRITE,
    },
    Role.MANAGER: {
        # Manager can manage tasks, resources, and rollouts
        Permission.SERVER_READ,
        Permission.TASK_ADMIN, Permission.TASK_READ, Permission.TASK_WRITE, Permission.TASK_ASSIGN,
        Permission.RESOURCE_ADMIN, Permission.RESOURCE_READ, Permission.RESOURCE_WRITE,
        Permission.ROLLOUT_ADMIN, Permission.ROLLOUT_READ, Permission.ROLLOUT_WRITE,
        Permission.USER_READ,
        Permission.SYSTEM_READ,
    },
    Role.USER: {
        # Regular user can read/write tasks and resources, read rollouts
        Permission.SERVER_READ,
        Permission.TASK_READ, Permission.TASK_WRITE,
        Permission.RESOURCE_READ, Permission.RESOURCE_WRITE,
        Permission.ROLLOUT_READ, Permission.ROLLOUT_WRITE,
        Permission.SYSTEM_READ,
    },
    Role.AGENT: {
        # Agent can read tasks/resources and write rollouts
        Permission.SERVER_READ,
        Permission.TASK_READ, Permission.TASK_ASSIGN,
        Permission.RESOURCE_READ,
        Permission.ROLLOUT_READ, Permission.ROLLOUT_WRITE,
    },
    Role.GUEST: {
        # Guest can only read basic information
        Permission.SERVER_READ,
        Permission.TASK_READ,
        Permission.RESOURCE_READ,
        Permission.ROLLOUT_READ,
        Permission.SYSTEM_READ,
    },
}


class RBACManager:
    """Role-Based Access Control Manager"""

    def __init__(self):
        # Default role mappings - can be extended via config
        self.role_mappings = {
            'admin': Role.ADMIN,
            'manager': Role.MANAGER,
            'user': Role.USER,
            'agent': Role.AGENT,
            'guest': Role.GUEST,
        }

        # Load custom role mappings from environment
        custom_mappings = os.getenv('RBAC_ROLE_MAPPINGS', '')
        if custom_mappings:
            try:
                import json
                self.role_mappings.update(json.loads(custom_mappings))
                logger.info(f"Loaded custom role mappings: {custom_mappings}")
            except Exception as e:
                logger.error(f"Failed to parse custom role mappings: {e}")

    def get_role_from_user(self, user_info: Dict[str, Any]) -> Role:
        """
        Extract role from user information

        Args:
            user_info: User information dict from auth system

        Returns:
            Role enum value
        """
        # Check OAuth roles first (from JWT token)
        oauth_roles = user_info.get('roles', [])
        if oauth_roles:
            # Use the highest privilege role
            for role_name in ['admin', 'manager', 'user', 'agent']:
                if role_name in oauth_roles:
                    return self.role_mappings.get(role_name, Role.GUEST)

        # Check API key permissions for role inference
        api_permissions = user_info.get('permissions', [])
        if 'admin' in api_permissions:
            return Role.ADMIN
        elif 'write' in api_permissions:
            return Role.USER
        elif 'read' in api_permissions:
            return Role.GUEST

        # Default to guest role
        return Role.GUEST

    def has_permission(self, user_info: Dict[str, Any], permission: Permission) -> bool:
        """
        Check if user has a specific permission

        Args:
            user_info: User information dict
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        user_role = self.get_role_from_user(user_info)
        role_permissions = ROLE_PERMISSIONS.get(user_role, set())

        return permission in role_permissions

    def has_any_permission(self, user_info: Dict[str, Any], permissions: List[Permission]) -> bool:
        """
        Check if user has any of the specified permissions

        Args:
            user_info: User information dict
            permissions: List of permissions to check

        Returns:
            True if user has any of the permissions, False otherwise
        """
        return any(self.has_permission(user_info, perm) for perm in permissions)

    def has_all_permissions(self, user_info: Dict[str, Any], permissions: List[Permission]) -> bool:
        """
        Check if user has all of the specified permissions

        Args:
            user_info: User information dict
            permissions: List of permissions to check

        Returns:
            True if user has all permissions, False otherwise
        """
        return all(self.has_permission(user_info, perm) for perm in permissions)

    def require_permission(self, permission: Permission):
        """
        Create a FastAPI dependency that requires a specific permission

        Args:
            permission: Required permission

        Returns:
            Dependency function
        """
        def dependency(user: Dict = Depends(self.get_current_user_with_rbac)):
            if not self.has_permission(user, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission.value}' required"
                )
            return user
        return dependency

    def require_any_permission(self, permissions: List[Permission]):
        """
        Create a FastAPI dependency that requires any of the specified permissions

        Args:
            permissions: List of permissions

        Returns:
            Dependency function
        """
        def dependency(user: Dict = Depends(self.get_current_user_with_rbac)):
            if not self.has_any_permission(user, permissions):
                perm_names = [p.value for p in permissions]
                raise HTTPException(
                    status_code=403,
                    detail=f"Any of these permissions required: {perm_names}"
                )
            return user
        return dependency

    def require_role(self, role: Role):
        """
        Create a FastAPI dependency that requires a specific role

        Args:
            role: Required role

        Returns:
            Dependency function
        """
        def dependency(user: Dict = Depends(self.get_current_user_with_rbac)):
            user_role = self.get_role_from_user(user)
            if user_role != role:
                raise HTTPException(
                    status_code=403,
                    detail=f"Role '{role.value}' required"
                )
            return user
        return dependency

    def require_role_or_higher(self, min_role: Role):
        """
        Create a FastAPI dependency that requires a role at or above the specified level

        Args:
            min_role: Minimum required role

        Returns:
            Dependency function
        """
        # Define role hierarchy (higher index = more permissions)
        role_hierarchy = [Role.GUEST, Role.AGENT, Role.USER, Role.MANAGER, Role.ADMIN]

        def dependency(user: Dict = Depends(self.get_current_user_with_rbac)):
            user_role = self.get_role_from_user(user)
            user_level = role_hierarchy.index(user_role)
            min_level = role_hierarchy.index(min_role)

            if user_level < min_level:
                raise HTTPException(
                    status_code=403,
                    detail=f"Role '{min_role.value}' or higher required"
                )
            return user
        return dependency

    async def get_current_user_with_rbac(self, request: Optional[Any] = None) -> Dict[str, Any]:
        """
        Get current user with RBAC information
        This should be integrated with your auth system
        """
        # This is a placeholder - integrate with your actual auth system
        # For now, return a guest user
        return {
            'user_id': 'guest',
            'roles': ['guest'],
            'permissions': ['read'],
            'role': Role.GUEST.value
        }


# Global RBAC manager instance
rbac_manager = RBACManager()


# Convenience functions for common permission checks
def require_server_read():
    """Require server read permission"""
    return rbac_manager.require_permission(Permission.SERVER_READ)

def require_server_write():
    """Require server write permission"""
    return rbac_manager.require_permission(Permission.SERVER_WRITE)

def require_server_admin():
    """Require server admin permission"""
    return rbac_manager.require_permission(Permission.SERVER_ADMIN)

def require_task_read():
    """Require task read permission"""
    return rbac_manager.require_permission(Permission.TASK_READ)

def require_task_write():
    """Require task write permission"""
    return rbac_manager.require_permission(Permission.TASK_WRITE)

def require_task_admin():
    """Require task admin permission"""
    return rbac_manager.require_permission(Permission.TASK_ADMIN)

def require_resource_read():
    """Require resource read permission"""
    return rbac_manager.require_permission(Permission.RESOURCE_READ)

def require_resource_write():
    """Require resource write permission"""
    return rbac_manager.require_permission(Permission.RESOURCE_WRITE)

def require_rollout_read():
    """Require rollout read permission"""
    return rbac_manager.require_permission(Permission.ROLLOUT_READ)

def require_rollout_write():
    """Require rollout write permission"""
    return rbac_manager.require_permission(Permission.ROLLOUT_WRITE)

def require_admin_role():
    """Require admin role"""
    return rbac_manager.require_role(Role.ADMIN)

def require_manager_or_higher():
    """Require manager role or higher"""
    return rbac_manager.require_role_or_higher(Role.MANAGER)


# Integration helpers
def get_user_permissions(user_info: Dict[str, Any]) -> List[str]:
    """
    Get list of permission strings for a user

    Args:
        user_info: User information dict

    Returns:
        List of permission strings
    """
    user_role = rbac_manager.get_role_from_user(user_info)
    role_permissions = ROLE_PERMISSIONS.get(user_role, set())

    return [perm.value for perm in role_permissions]


def get_user_role(user_info: Dict[str, Any]) -> str:
    """
    Get user's role as string

    Args:
        user_info: User information dict

    Returns:
        Role name string
    """
    user_role = rbac_manager.get_role_from_user(user_info)
    return user_role.value