#!/usr/bin/env python3
"""
OAuth2/OIDC Integration for Agent Lightning
Enterprise-grade authentication with SSO support
"""

import os
import secrets
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import jwt
import hashlib
import base64

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
from .rbac import rbac_manager

logger = logging.getLogger(__name__)


class OAuth2Config:
    """OAuth2/OIDC Configuration"""

    def __init__(self):
        # OAuth2 Provider Settings
        self.provider_url = os.getenv('OAUTH2_PROVIDER_URL', '')
        self.client_id = os.getenv('OAUTH2_CLIENT_ID', '')
        self.client_secret = os.getenv('OAUTH2_CLIENT_SECRET', '')
        self.redirect_uri = os.getenv('OAUTH2_REDIRECT_URI', 'http://localhost:8000/auth/callback')

        # OIDC Settings
        self.issuer = os.getenv('OIDC_ISSUER', '')
        self.jwks_uri = os.getenv('OIDC_JWKS_URI', '')

        # Token Settings
        self.token_url = os.getenv('OAUTH2_TOKEN_URL', '')
        self.authorization_url = os.getenv('OAUTH2_AUTHORIZATION_URL', '')
        self.userinfo_url = os.getenv('OAUTH2_USERINFO_URL', '')

        # Local JWT Settings
        self.jwt_secret = os.getenv('JWT_SECRET', secrets.token_urlsafe(32))
        self.jwt_algorithm = os.getenv('JWT_ALGORITHM', 'HS256')
        self.jwt_expiration = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))

        # Scopes
        self.scopes = os.getenv('OAUTH2_SCOPES', 'openid profile email').split()

        # Cache settings
        self.jwks_cache_ttl = 3600  # 1 hour

    def is_configured(self) -> bool:
        """Check if OAuth2 is properly configured"""
        return bool(
            self.provider_url and
            self.client_id and
            self.client_secret
        )


class JWKSManager:
    """JSON Web Key Set Manager for OIDC"""

    def __init__(self, config: OAuth2Config):
        self.config = config
        self.keys_cache: Dict[str, Any] = {}
        self.cache_timestamp = 0

    async def get_signing_key(self, kid: str) -> Optional[Dict[str, Any]]:
        """Get signing key by Key ID"""
        await self._refresh_keys_if_needed()

        return self.keys_cache.get(kid)

    async def _refresh_keys_if_needed(self):
        """Refresh JWKS if cache is stale"""
        current_time = time.time()

        if current_time - self.cache_timestamp > self.config.jwks_cache_ttl:
            await self._fetch_jwks()

    async def _fetch_jwks(self):
        """Fetch JWKS from provider"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.config.jwks_uri)
                response.raise_for_status()

                jwks = response.json()
                self.keys_cache = {}

                for key in jwks.get('keys', []):
                    kid = key.get('kid')
                    if kid:
                        self.keys_cache[kid] = key

                self.cache_timestamp = time.time()
                logger.info(f"Refreshed JWKS with {len(self.keys_cache)} keys")

        except Exception as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            raise


class OAuth2Client:
    """OAuth2/OIDC Client"""

    def __init__(self, config: OAuth2Config):
        self.config = config
        self.jwks_manager = JWKSManager(config)

    def get_authorization_url(self, state: str = None) -> str:
        """Generate OAuth2 authorization URL"""
        if not state:
            state = secrets.token_urlsafe(32)

        params = {
            'client_id': self.config.client_id,
            'response_type': 'code',
            'scope': ' '.join(self.config.scopes),
            'redirect_uri': self.config.redirect_uri,
            'state': state
        }

        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.config.authorization_url}?{query_string}"

    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.token_url,
                    data={
                        'grant_type': 'authorization_code',
                        'code': code,
                        'redirect_uri': self.config.redirect_uri,
                        'client_id': self.config.client_id,
                        'client_secret': self.config.client_secret
                    },
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                )
                response.raise_for_status()

                token_data = response.json()
                logger.info("Successfully exchanged code for token")
                return token_data

        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            raise HTTPException(status_code=400, detail="Token exchange failed")

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from OAuth2 provider"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.config.userinfo_url,
                    headers={'Authorization': f'Bearer {access_token}'}
                )
                response.raise_for_status()

                user_info = response.json()
                logger.info(f"Retrieved user info for: {user_info.get('email', 'unknown')}")
                return user_info

        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            raise HTTPException(status_code=400, detail="Failed to get user information")

    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return claims"""
        try:
            # Decode header to get key ID
            header = jwt.get_unverified_header(token)
            kid = header.get('kid')

            if not kid:
                raise HTTPException(status_code=401, detail="Invalid token: missing key ID")

            # Get signing key
            signing_key = await self.jwks_manager.get_signing_key(kid)
            if not signing_key:
                raise HTTPException(status_code=401, detail="Invalid token: unknown key ID")

            # Validate token
            try:
                payload = jwt.decode(
                    token,
                    signing_key,
                    algorithms=[signing_key.get('alg', 'RS256')],
                    audience=self.config.client_id,
                    issuer=self.config.issuer
                )

                # Check expiration
                if payload.get('exp', 0) < time.time():
                    raise HTTPException(status_code=401, detail="Token expired")

                return payload

            except jwt.ExpiredSignatureError:
                raise HTTPException(status_code=401, detail="Token expired")
            except jwt.InvalidTokenError as e:
                raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise HTTPException(status_code=401, detail="Token validation failed")


class LocalJWTManager:
    """Local JWT token management for development/testing"""

    def __init__(self, config: OAuth2Config):
        self.config = config

    def create_token(self, user_info: Dict[str, Any]) -> str:
        """Create JWT token with user information"""
        payload = {
            'sub': user_info.get('sub', user_info.get('id')),
            'email': user_info.get('email'),
            'name': user_info.get('name'),
            'preferred_username': user_info.get('preferred_username'),
            'groups': user_info.get('groups', []),
            'roles': user_info.get('roles', []),
            'iat': int(time.time()),
            'exp': int((datetime.now() + timedelta(hours=self.config.jwt_expiration)).timestamp()),
            'iss': 'agent-lightning',
            'aud': 'agent-lightning-api'
        }

        token = jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
        return token

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate local JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
                audience='agent-lightning-api',
                issuer='agent-lightning'
            )

            # Check expiration
            if payload.get('exp', 0) < time.time():
                raise HTTPException(status_code=401, detail="Token expired")

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


class OAuth2Manager:
    """Main OAuth2/OIDC Manager"""

    def __init__(self):
        self.config = OAuth2Config()
        self.oauth_client = OAuth2Client(self.config) if self.config.is_configured() else None
        self.local_jwt = LocalJWTManager(self.config)

        if self.config.is_configured():
            logger.info("✅ OAuth2/OIDC integration enabled")
        else:
            logger.info("⚠️ OAuth2/OIDC not configured, using local JWT only")

    def is_oauth_enabled(self) -> bool:
        """Check if OAuth2 is enabled"""
        return self.oauth_client is not None

    def get_authorization_url(self, state: str = None) -> str:
        """Get OAuth2 authorization URL"""
        if not self.oauth_client:
            raise HTTPException(status_code=503, detail="OAuth2 not configured")

        return self.oauth_client.get_authorization_url(state)

    async def process_callback(self, code: str, state: str) -> Dict[str, Any]:
        """Process OAuth2 callback and return user session"""
        if not self.oauth_client:
            raise HTTPException(status_code=503, detail="OAuth2 not configured")

        # Exchange code for token
        token_data = await self.oauth_client.exchange_code_for_token(code)

        # Get user info
        user_info = await self.oauth_client.get_user_info(token_data['access_token'])

        # Create local JWT token
        jwt_token = self.local_jwt.create_token(user_info)

        return {
            'access_token': jwt_token,
            'token_type': 'Bearer',
            'expires_in': self.config.jwt_expiration * 3600,
            'user_info': user_info
        }

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate token (OAuth2 or local JWT)"""
        if self.oauth_client:
            try:
                return self.oauth_client.validate_token(token)
            except:
                # Fall back to local JWT validation
                pass

        return self.local_jwt.validate_token(token)

    def create_local_token(self, user_info: Dict[str, Any]) -> str:
        """Create local JWT token for development/testing"""
        return self.local_jwt.create_token(user_info)


# Global OAuth2 manager instance
oauth_manager = OAuth2Manager()


# FastAPI dependencies
oauth2_scheme = HTTPBearer(auto_error=False)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Get current authenticated user from OAuth2/Local JWT token with RBAC information"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    token = credentials.credentials
    user_info = oauth_manager.validate_token(token)

    # Add RBAC information
    user_info['role'] = rbac_manager.get_user_role(user_info)
    user_info['permissions'] = rbac_manager.get_user_permissions(user_info)

    return user_info


def require_scope(scope: str):
    """Create dependency that requires specific OAuth2 scope"""
    async def scope_checker(user: Dict = Depends(get_current_user)):
        token_scopes = user.get('scope', '').split()
        if scope not in token_scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions: {scope} scope required"
            )
        return user
    return scope_checker


def require_role(role: str):
    """Create dependency that requires specific role"""
    async def role_checker(user: Dict = Depends(get_current_user)):
        user_roles = user.get('roles', [])
        if role not in user_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions: {role} role required"
            )
        return user
    return role_checker


# Convenience dependencies
require_read = require_scope('read')
require_write = require_scope('write')
require_admin = require_role('admin')