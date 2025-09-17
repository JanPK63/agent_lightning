"""
API Key Management API
FastAPI endpoints for managing API key rotation and lifecycle
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
import logging

from agentlightning.auth import auth_manager, get_current_user
from services.api_key_rotation_service import api_key_rotation_service
from shared.models import ApiKeyRotationPolicy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api-keys", tags=["API Keys"])


# Pydantic models for request/response
class CreateApiKeyRequest(BaseModel):
    name: str
    description: Optional[str] = None
    permissions: Optional[List[str]] = None
    expires_in_days: int = 365
    enable_rotation: bool = True
    rotation_policy_id: Optional[str] = None


class RotateApiKeyRequest(BaseModel):
    reason: str = "manual"
    notes: Optional[str] = None


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    user_id: str
    permissions: List[str]
    is_active: bool
    expires_at: Optional[str]
    last_used_at: Optional[str]
    usage_count: int
    created_at: str
    updated_at: str
    # Rotation fields
    is_rotation_enabled: bool
    rotation_locked: bool
    last_rotated_at: Optional[str]
    next_rotation_at: Optional[str]
    rotation_count: int


class RotationHistoryResponse(BaseModel):
    id: str
    api_key_id: str
    old_key_hash: Optional[str]
    new_key_hash: Optional[str]
    old_expires_at: Optional[str]
    new_expires_at: Optional[str]
    rotated_at: str
    rotated_by: Optional[str]
    rotation_reason: str
    rotation_policy_id: Optional[str]
    notes: Optional[str]


class RotationStatusResponse(BaseModel):
    api_key_id: str
    name: str
    is_rotation_enabled: bool
    rotation_locked: bool
    last_rotated_at: Optional[str]
    next_rotation_at: Optional[str]
    rotation_count: int
    is_due_for_rotation: bool
    days_until_rotation: Optional[int]


@router.post("/", response_model=dict)
async def create_api_key(
    request: CreateApiKeyRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new API key"""
    try:
        # Create the API key
        key_info = auth_manager.create_api_key(
            name=request.name,
            user_id=current_user['id'],
            permissions=request.permissions,
            expires_in_days=request.expires_in_days
        )

        # Enable rotation if requested
        if request.enable_rotation:
            success = auth_manager.enable_rotation(
                api_key_id=key_info['id'],
                policy_id=request.rotation_policy_id
            )
            if not success:
                logger.warning(f"Failed to enable rotation for newly created key {key_info['id']}")

        return {
            "success": True,
            "api_key": key_info,
            "message": "API key created successfully"
        }

    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to create API key")


@router.get("/", response_model=List[ApiKeyResponse])
async def list_api_keys(
    current_user: dict = Depends(get_current_user)
):
    """List all API keys for the current user"""
    try:
        # Get user's API keys from database
        from shared.database import db_manager

        with db_manager.get_db() as session:
            from shared.models import ApiKey
            api_keys = session.query(ApiKey).filter(
                ApiKey.user_id == current_user['id']
            ).all()

            return [ApiKeyResponse(**key.to_dict()) for key in api_keys]

    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        raise HTTPException(status_code=500, detail="Failed to list API keys")


@router.get("/{api_key_id}", response_model=ApiKeyResponse)
async def get_api_key(
    api_key_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get details of a specific API key"""
    try:
        from shared.database import db_manager

        with db_manager.get_db() as session:
            from shared.models import ApiKey
            api_key = session.query(ApiKey).filter(
                ApiKey.id == api_key_id,
                ApiKey.user_id == current_user['id']
            ).first()

            if not api_key:
                raise HTTPException(status_code=404, detail="API key not found")

            return ApiKeyResponse(**api_key.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting API key {api_key_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get API key")


@router.post("/{api_key_id}/rotate", response_model=dict)
async def rotate_api_key(
    api_key_id: str,
    request: RotateApiKeyRequest,
    current_user: dict = Depends(get_current_user)
):
    """Manually rotate an API key"""
    try:
        # Verify ownership
        from shared.database import db_manager

        with db_manager.get_db() as session:
            from shared.models import ApiKey
            api_key = session.query(ApiKey).filter(
                ApiKey.id == api_key_id,
                ApiKey.user_id == current_user['id']
            ).first()

            if not api_key:
                raise HTTPException(status_code=404, detail="API key not found")

        # Perform rotation
        result = auth_manager.rotate_api_key(
            api_key_id=api_key_id,
            user_id=current_user['id'],
            reason=request.reason
        )

        if result['success']:
            return {
                "success": True,
                "new_key": result['new_key'],
                "rotation_history_id": result['rotation_history_id'],
                "message": "API key rotated successfully"
            }
        else:
            raise HTTPException(status_code=400, detail=result['error'])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rotating API key {api_key_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to rotate API key")


@router.get("/{api_key_id}/rotation-status", response_model=RotationStatusResponse)
async def get_rotation_status(
    api_key_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get rotation status for an API key"""
    try:
        # Verify ownership
        from shared.database import db_manager

        with db_manager.get_db() as session:
            from shared.models import ApiKey
            api_key = session.query(ApiKey).filter(
                ApiKey.id == api_key_id,
                ApiKey.user_id == current_user['id']
            ).first()

            if not api_key:
                raise HTTPException(status_code=404, detail="API key not found")

        status = auth_manager.get_api_key_rotation_status(api_key_id)
        if not status:
            raise HTTPException(status_code=404, detail="Rotation status not found")

        return RotationStatusResponse(**status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting rotation status for key {api_key_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get rotation status")


@router.get("/{api_key_id}/rotation-history", response_model=List[RotationHistoryResponse])
async def get_rotation_history(
    api_key_id: str,
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Get rotation history for an API key"""
    try:
        # Verify ownership
        from shared.database import db_manager

        with db_manager.get_db() as session:
            from shared.models import ApiKey
            api_key = session.query(ApiKey).filter(
                ApiKey.id == api_key_id,
                ApiKey.user_id == current_user['id']
            ).first()

            if not api_key:
                raise HTTPException(status_code=404, detail="API key not found")

        history = auth_manager.get_rotation_history(api_key_id, limit)
        return [RotationHistoryResponse(**record) for record in history]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting rotation history for key {api_key_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get rotation history")


@router.post("/{api_key_id}/enable-rotation", response_model=dict)
async def enable_rotation(
    api_key_id: str,
    rotation_policy_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Enable automatic rotation for an API key"""
    try:
        # Verify ownership
        from shared.database import db_manager

        with db_manager.get_db() as session:
            from shared.models import ApiKey
            api_key = session.query(ApiKey).filter(
                ApiKey.id == api_key_id,
                ApiKey.user_id == current_user['id']
            ).first()

            if not api_key:
                raise HTTPException(status_code=404, detail="API key not found")

        success = auth_manager.enable_rotation(api_key_id, rotation_policy_id)

        if success:
            return {
                "success": True,
                "message": "Automatic rotation enabled successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to enable rotation")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling rotation for key {api_key_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to enable rotation")


@router.post("/{api_key_id}/disable-rotation", response_model=dict)
async def disable_rotation(
    api_key_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Disable automatic rotation for an API key"""
    try:
        # Verify ownership
        from shared.database import db_manager

        with db_manager.get_db() as session:
            from shared.models import ApiKey
            api_key = session.query(ApiKey).filter(
                ApiKey.id == api_key_id,
                ApiKey.user_id == current_user['id']
            ).first()

            if not api_key:
                raise HTTPException(status_code=404, detail="API key not found")

        success = auth_manager.disable_rotation(api_key_id)

        if success:
            return {
                "success": True,
                "message": "Automatic rotation disabled successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to disable rotation")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling rotation for key {api_key_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to disable rotation")


@router.delete("/{api_key_id}", response_model=dict)
async def delete_api_key(
    api_key_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete an API key"""
    try:
        # Verify ownership and delete
        from shared.database import db_manager

        with db_manager.get_db() as session:
            from shared.models import ApiKey
            api_key = session.query(ApiKey).filter(
                ApiKey.id == api_key_id,
                ApiKey.user_id == current_user['id']
            ).first()

            if not api_key:
                raise HTTPException(status_code=404, detail="API key not found")

            # Soft delete by deactivating
            api_key.is_active = False
            session.commit()

        return {
            "success": True,
            "message": "API key deactivated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting API key {api_key_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete API key")


# Admin endpoints (require admin permission)
@router.get("/admin/rotation-policies", response_model=List[dict])
async def list_rotation_policies(
    current_user: dict = Depends(get_current_user)
):
    """List all rotation policies (admin only)"""
    # TODO: Add admin permission check
    try:
        from shared.database import db_manager

        with db_manager.get_db() as session:
            policies = session.query(ApiKeyRotationPolicy).filter(
                ApiKeyRotationPolicy.is_active == True
            ).all()

            return [policy.to_dict() for policy in policies]

    except Exception as e:
        logger.error(f"Error listing rotation policies: {e}")
        raise HTTPException(status_code=500, detail="Failed to list rotation policies")


@router.get("/admin/due-for-rotation", response_model=List[dict])
async def get_keys_due_for_rotation(
    days_ahead: int = Query(7, ge=0, le=365),
    current_user: dict = Depends(get_current_user)
):
    """Get all keys due for rotation (admin only)"""
    # TODO: Add admin permission check
    try:
        due_keys = api_key_rotation_service.get_keys_due_for_rotation(days_ahead)
        return due_keys

    except Exception as e:
        logger.error(f"Error getting keys due for rotation: {e}")
        raise HTTPException(status_code=500, detail="Failed to get keys due for rotation")


@router.post("/admin/bulk-rotate", response_model=dict)
async def bulk_rotate_keys(
    key_ids: List[str],
    reason: str = "bulk_admin",
    current_user: dict = Depends(get_current_user)
):
    """Bulk rotate multiple API keys (admin only)"""
    # TODO: Add admin permission check
    try:
        results = api_key_rotation_service.bulk_rotate_keys(
            key_ids=key_ids,
            user_id=current_user['id'],
            reason=reason
        )

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        return {
            "success": True,
            "total_keys": len(results),
            "successful_rotations": successful,
            "failed_rotations": failed,
            "results": [
                {
                    "api_key_id": r.api_key_id,
                    "success": r.success,
                    "error": r.error_message if not r.success else None
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.error(f"Error in bulk rotation: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform bulk rotation")