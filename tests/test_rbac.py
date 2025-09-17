"""
Tests for Role-Based Access Control (RBAC) system.
"""
import pytest
from agentlightning.rbac import (
    RBACManager, Role, Permission, rbac_manager,
    ROLE_PERMISSIONS, get_user_permissions, get_user_role
)


class TestRBACManager:
    """Test RBAC manager functionality."""

    @pytest.fixture
    def rbac(self):
        """Create RBAC manager instance."""
        return RBACManager()

    def test_init_default(self, rbac):
        """Test RBAC manager initialization with defaults."""
        assert rbac.role_mappings['admin'] == Role.ADMIN
        assert rbac.role_mappings['user'] == Role.USER

    def test_get_role_from_user_guest(self, rbac):
        """Test role extraction for guest user."""
        user_info = {'permissions': ['read']}
        role = rbac.get_role_from_user(user_info)
        assert role == Role.GUEST

    def test_get_role_from_user_user(self, rbac):
        """Test role extraction for regular user."""
        user_info = {'permissions': ['read', 'write']}
        role = rbac.get_role_from_user(user_info)
        assert role == Role.USER

    def test_get_role_from_user_admin(self, rbac):
        """Test role extraction for admin user."""
        user_info = {'permissions': ['admin']}
        role = rbac.get_role_from_user(user_info)
        assert role == Role.ADMIN

    def test_get_role_from_oauth_roles(self, rbac):
        """Test role extraction from OAuth roles."""
        user_info = {'roles': ['admin']}
        role = rbac.get_role_from_user(user_info)
        assert role == Role.ADMIN

    def test_has_permission_admin(self, rbac):
        """Test permission checking for admin role."""
        user_info = {'permissions': ['admin']}
        assert rbac.has_permission(user_info, Permission.SERVER_ADMIN)
        assert rbac.has_permission(user_info, Permission.TASK_READ)
        assert rbac.has_permission(user_info, Permission.USER_ADMIN)

    def test_has_permission_user(self, rbac):
        """Test permission checking for user role."""
        user_info = {'permissions': ['read', 'write']}
        assert rbac.has_permission(user_info, Permission.TASK_READ)
        assert rbac.has_permission(user_info, Permission.RESOURCE_WRITE)
        assert not rbac.has_permission(user_info, Permission.SERVER_ADMIN)
        assert not rbac.has_permission(user_info, Permission.USER_ADMIN)

    def test_has_permission_guest(self, rbac):
        """Test permission checking for guest role."""
        user_info = {'permissions': ['read']}
        assert rbac.has_permission(user_info, Permission.TASK_READ)
        assert rbac.has_permission(user_info, Permission.RESOURCE_READ)
        assert not rbac.has_permission(user_info, Permission.TASK_WRITE)
        assert not rbac.has_permission(user_info, Permission.SERVER_ADMIN)

    def test_has_any_permission(self, rbac):
        """Test checking for any of multiple permissions."""
        user_info = {'permissions': ['read', 'write']}
        permissions = [Permission.SERVER_ADMIN, Permission.TASK_READ]
        assert rbac.has_any_permission(user_info, permissions)

        permissions = [Permission.SERVER_ADMIN, Permission.USER_ADMIN]
        assert not rbac.has_any_permission(user_info, permissions)

    def test_has_all_permissions(self, rbac):
        """Test checking for all of multiple permissions."""
        user_info = {'permissions': ['read', 'write']}
        permissions = [Permission.TASK_READ, Permission.RESOURCE_READ]
        assert rbac.has_all_permissions(user_info, permissions)

        permissions = [Permission.TASK_READ, Permission.SERVER_ADMIN]
        assert not rbac.has_all_permissions(user_info, permissions)


class TestRolePermissions:
    """Test role-permission mappings."""

    def test_admin_permissions(self):
        """Test admin role has all permissions."""
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.SERVER_ADMIN in admin_perms
        assert Permission.TASK_ADMIN in admin_perms
        assert Permission.RESOURCE_ADMIN in admin_perms
        assert Permission.ROLLOUT_ADMIN in admin_perms
        assert Permission.USER_ADMIN in admin_perms
        assert Permission.SYSTEM_ADMIN in admin_perms

    def test_manager_permissions(self):
        """Test manager role permissions."""
        manager_perms = ROLE_PERMISSIONS[Role.MANAGER]
        assert Permission.TASK_ADMIN in manager_perms
        assert Permission.RESOURCE_ADMIN in manager_perms
        assert Permission.ROLLOUT_ADMIN in manager_perms
        assert Permission.USER_READ in manager_perms
        assert Permission.SYSTEM_READ in manager_perms

        # Manager should not have user admin or system admin
        assert Permission.USER_ADMIN not in manager_perms
        assert Permission.SYSTEM_ADMIN not in manager_perms

    def test_user_permissions(self):
        """Test user role permissions."""
        user_perms = ROLE_PERMISSIONS[Role.USER]
        assert Permission.TASK_READ in user_perms
        assert Permission.TASK_WRITE in user_perms
        assert Permission.RESOURCE_READ in user_perms
        assert Permission.RESOURCE_WRITE in user_perms
        assert Permission.ROLLOUT_READ in user_perms
        assert Permission.ROLLOUT_WRITE in user_perms
        assert Permission.SYSTEM_READ in user_perms

        # User should not have admin permissions
        assert Permission.SERVER_ADMIN not in user_perms
        assert Permission.TASK_ADMIN not in user_perms
        assert Permission.USER_ADMIN not in user_perms

    def test_agent_permissions(self):
        """Test agent role permissions."""
        agent_perms = ROLE_PERMISSIONS[Role.AGENT]
        assert Permission.TASK_READ in agent_perms
        assert Permission.TASK_ASSIGN in agent_perms
        assert Permission.RESOURCE_READ in agent_perms
        assert Permission.ROLLOUT_READ in agent_perms
        assert Permission.ROLLOUT_WRITE in agent_perms

        # Agent should not have write permissions for tasks/resources
        assert Permission.TASK_WRITE not in agent_perms
        assert Permission.RESOURCE_WRITE not in agent_perms

    def test_guest_permissions(self):
        """Test guest role permissions."""
        guest_perms = ROLE_PERMISSIONS[Role.GUEST]
        assert Permission.TASK_READ in guest_perms
        assert Permission.RESOURCE_READ in guest_perms
        assert Permission.ROLLOUT_READ in guest_perms
        assert Permission.SYSTEM_READ in guest_perms

        # Guest should not have any write permissions
        assert Permission.TASK_WRITE not in guest_perms
        assert Permission.RESOURCE_WRITE not in guest_perms
        assert Permission.ROLLOUT_WRITE not in guest_perms


class TestPermissionEnums:
    """Test permission enum values."""

    def test_permission_values(self):
        """Test permission enum string values."""
        assert Permission.SERVER_READ.value == "server:read"
        assert Permission.TASK_ADMIN.value == "task:admin"
        assert Permission.RESOURCE_WRITE.value == "resource:write"
        assert Permission.ROLLOUT_READ.value == "rollout:read"
        assert Permission.USER_ADMIN.value == "user:admin"
        assert Permission.SYSTEM_ADMIN.value == "system:admin"

    def test_role_values(self):
        """Test role enum string values."""
        assert Role.ADMIN.value == "admin"
        assert Role.MANAGER.value == "manager"
        assert Role.USER.value == "user"
        assert Role.AGENT.value == "agent"
        assert Role.GUEST.value == "guest"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_user_permissions_admin(self):
        """Test getting permissions for admin user."""
        user_info = {'permissions': ['admin']}
        perms = get_user_permissions(user_info)
        assert 'server:admin' in perms
        assert 'task:admin' in perms
        assert 'user:admin' in perms

    def test_get_user_permissions_user(self):
        """Test getting permissions for regular user."""
        user_info = {'permissions': ['read', 'write']}
        perms = get_user_permissions(user_info)
        assert 'task:read' in perms
        assert 'resource:write' in perms
        assert 'server:admin' not in perms

    def test_get_user_role_admin(self):
        """Test getting role string for admin user."""
        user_info = {'permissions': ['admin']}
        role = get_user_role(user_info)
        assert role == 'admin'

    def test_get_user_role_user(self):
        """Test getting role string for regular user."""
        user_info = {'permissions': ['read', 'write']}
        role = get_user_role(user_info)
        assert role == 'user'

    def test_get_user_role_guest(self):
        """Test getting role string for guest user."""
        user_info = {'permissions': ['read']}
        role = get_user_role(user_info)
        assert role == 'guest'


class TestRBACIntegration:
    """Integration tests for RBAC system."""

    def test_role_hierarchy(self):
        """Test that roles have appropriate permission hierarchies."""
        # Admin should have all permissions
        admin_perms = len(ROLE_PERMISSIONS[Role.ADMIN])
        manager_perms = len(ROLE_PERMISSIONS[Role.MANAGER])
        user_perms = len(ROLE_PERMISSIONS[Role.USER])
        agent_perms = len(ROLE_PERMISSIONS[Role.AGENT])
        guest_perms = len(ROLE_PERMISSIONS[Role.GUEST])

        # Admin should have the most permissions
        assert admin_perms > manager_perms
        assert manager_perms > user_perms
        assert user_perms > agent_perms
        assert agent_perms > guest_perms

    def test_permission_categories(self):
        """Test that permissions are properly categorized."""
        all_perms = set()
        for role_perms in ROLE_PERMISSIONS.values():
            all_perms.update(role_perms)

        # Check that we have permissions for all categories
        categories = ['server', 'task', 'resource', 'rollout', 'user', 'system']
        for category in categories:
            category_perms = [p for p in all_perms if p.value.startswith(f'{category}:')]
            assert len(category_perms) >= 2, f"Category {category} should have at least read/write permissions"

    def test_no_duplicate_permissions(self):
        """Test that there are no duplicate permissions in role mappings."""
        for role, perms in ROLE_PERMISSIONS.items():
            perm_list = list(perms)
            assert len(perm_list) == len(set(perm_list)), f"Role {role.value} has duplicate permissions"