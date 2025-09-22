#!/usr/bin/env python3
"""
Auth Service Microservice - Integrated with Shared Database
Centralized authentication and authorization with JWT, sessions, and RBAC
Using shared PostgreSQL and Redis for persistence and caching
Based on SA-007: Authentication & Authorization Service
"""

import os
import sys
import json
import asyncio
import uuid
import hashlib
import hmac
import secrets
import bcrypt
import pyotp
from jose import JWTError, jwt
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
import logging
import base64
from urllib.parse import urlparse, urlencode, parse_qs
import time

from fastapi import FastAPI, HTTPException, Depends, Request, Response, Header, Query, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, EmailStr

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared data access layer
from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

# Import input sanitization utilities
from shared.sanitization import InputSanitizer, sanitize_user_input

# Import Prometheus metrics
from monitoring.metrics import get_metrics_collector, Timer

# Import OAuth2/OIDC system
try:
    from agentlightning.oauth import oauth_manager, OAuth2Manager
    from agentlightning.oidc_providers import OIDCProviderConfig, get_provider_setup_instructions
    OAUTH_ENABLED = True
except ImportError:
    logger.warning("OAuth2/OIDC system not available")
    oauth_manager = None
    OIDCProviderConfig = None
    get_provider_setup_instructions = None
    OAUTH_ENABLED = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenType(str, Enum):
    """Types of tokens"""
    ACCESS = "access"
    REFRESH = "refresh"
    ID = "id"
    API_KEY = "api_key"


class AuthMethod(str, Enum):
    """Authentication methods"""
    PASSWORD = "password"
    OAUTH2 = "oauth2"
    OIDC = "oidc"
    SAML = "saml"
    API_KEY = "api_key"
    MFA = "mfa"


class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    OPERATOR = "operator"
    VIEWER = "viewer"
    GUEST = "guest"


class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    PENDING = "pending"


# Constants
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 30
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 900  # 15 minutes


# Pydantic Models
class UserCreate(BaseModel):
    """User registration model"""
    email: EmailStr = Field(description="User email")
    password: str = Field(description="User password", min_length=8)
    full_name: str = Field(description="Full name")
    role: Optional[UserRole] = Field(default=UserRole.VIEWER, description="User role")


class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr = Field(description="User email")
    password: str = Field(description="User password")
    mfa_code: Optional[str] = Field(default=None, description="MFA code if enabled")


class TokenRefresh(BaseModel):
    """Token refresh request"""
    refresh_token: str = Field(description="Refresh token")


class PasswordChange(BaseModel):
    """Password change request"""
    current_password: str = Field(description="Current password")
    new_password: str = Field(description="New password", min_length=8)


class APIKeyCreate(BaseModel):
    """API key creation request"""
    name: str = Field(description="API key name")
    service_name: Optional[str] = Field(default=None, description="Service name")
    permissions: List[str] = Field(default_factory=list, description="Permissions")
    expires_in_days: Optional[int] = Field(default=None, description="Expiration in days")


class JWTManager:
    """Manages JWT token generation and validation"""
    
    def __init__(self, secret_key: str = SECRET_KEY, algorithm: str = ALGORITHM):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_ttl = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        self.refresh_token_ttl = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    def generate_tokens(self, user_id: str, email: str, roles: List[str]) -> dict:
        """Generate access and refresh tokens"""
        now = datetime.utcnow()
        
        # Access token payload
        access_payload = {
            "sub": user_id,
            "email": email,
            "roles": roles,
            "iat": now,
            "exp": now + self.access_token_ttl,
            "type": TokenType.ACCESS.value
        }
        
        # Refresh token payload
        refresh_payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + self.refresh_token_ttl,
            "type": TokenType.REFRESH.value,
            "token_id": str(uuid.uuid4())  # For tracking/revocation
        }
        
        return {
            "access_token": jwt.encode(access_payload, self.secret_key, self.algorithm),
            "refresh_token": jwt.encode(refresh_payload, self.secret_key, self.algorithm),
            "token_type": "bearer",
            "expires_in": int(self.access_token_ttl.total_seconds())
        }
    
    def verify_token(self, token: str, token_type: TokenType = TokenType.ACCESS) -> dict:
        """Verify and decode token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != token_type.value:
                raise ValueError(f"Invalid token type: expected {token_type.value}")
            
            return payload
        except JWTError as e:
            raise ValueError(f"Invalid token: {e}")
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return f"sk_{secrets.token_urlsafe(32)}"


class SessionManager:
    """Manages user sessions with Redis"""
    
    def __init__(self, cache):
        self.cache = cache
        self.session_ttl = 3600  # 1 hour
    
    def create_session(self, user_id: str, metadata: dict) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "metadata": metadata
        }
        
        # Store in Redis with TTL
        cache_key = f"session:{session_id}"
        self.cache.set(cache_key, session_data, ttl=self.session_ttl)
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[dict]:
        """Validate and refresh session"""
        cache_key = f"session:{session_id}"
        session = self.cache.get(cache_key)
        
        if session:
            # Update last activity
            session['last_activity'] = datetime.utcnow().isoformat()
            # Refresh TTL
            self.cache.set(cache_key, session, ttl=self.session_ttl)
            return session
        
        return None
    
    def destroy_session(self, session_id: str):
        """Destroy session"""
        cache_key = f"session:{session_id}"
        self.cache.delete(cache_key)
        logger.info(f"Destroyed session {session_id}")


class PasswordManager:
    """Manages password hashing and verification"""
    
    def hash_password(self, password: str) -> str:
        """Hash password with bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hash.encode('utf-8'))
    
    def check_password_strength(self, password: str) -> dict:
        """Check password strength"""
        issues = []
        
        if len(password) < 8:
            issues.append("Too short (min 8 characters)")
        if not any(c.isupper() for c in password):
            issues.append("No uppercase letters")
        if not any(c.islower() for c in password):
            issues.append("No lowercase letters")
        if not any(c.isdigit() for c in password):
            issues.append("No numbers")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("No special characters")
        
        return {
            "strong": len(issues) == 0,
            "score": max(0, 5 - len(issues)) / 5,
            "issues": issues
        }


class MFAManager:
    """Manages Multi-Factor Authentication"""
    
    def generate_secret(self) -> str:
        """Generate TOTP secret"""
        return pyotp.random_base32()
    
    def generate_qr_code_uri(self, user_email: str, secret: str) -> str:
        """Generate QR code URI for authenticator app"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name='Lightning System'
        )
        return totp_uri
    
    def verify_totp(self, secret: str, token: str) -> bool:
        """Verify TOTP token"""
        if not secret or not token:
            return False
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)


class RBACEngine:
    """Role-Based Access Control engine"""
    
    def __init__(self, dal: DataAccessLayer, cache):
        self.dal = dal
        self.cache = cache
        self.permission_cache = {}
        self.init_default_roles()
    
    def init_default_roles(self):
        """Initialize default roles and permissions"""
        self.default_roles = {
            UserRole.ADMIN: ["*:*"],  # All permissions
            UserRole.DEVELOPER: [
                "agents:*", "workflows:*", "models:*", 
                "integrations:read", "metrics:read"
            ],
            UserRole.OPERATOR: [
                "agents:read", "workflows:execute", "models:use",
                "metrics:read"
            ],
            UserRole.VIEWER: [
                "agents:read", "workflows:read", "models:read",
                "metrics:read"
            ],
            UserRole.GUEST: ["metrics:read"]
        }
    
    def check_permission(self, user_roles: List[str], resource: str, action: str) -> bool:
        """Check if user has permission for action on resource"""
        required = f"{resource}:{action}"
        
        for role in user_roles:
            permissions = self.default_roles.get(UserRole(role), [])
            
            # Check exact match
            if required in permissions:
                return True
            
            # Check wildcards
            if f"{resource}:*" in permissions or "*:*" in permissions:
                return True
        
        return False
    
    def get_user_permissions(self, user_roles: List[str]) -> List[str]:
        """Get all permissions for user roles"""
        permissions = set()
        
        for role in user_roles:
            role_perms = self.default_roles.get(UserRole(role), [])
            permissions.update(role_perms)
        
        return list(permissions)


class AuditLogger:
    """Comprehensive audit logging for all user actions"""

    def __init__(self, dal: DataAccessLayer):
        self.dal = dal

    def log_event(self, event_type: str, user_id: str = None, details: dict = None,
                  ip_address: str = None, user_agent: str = None, success: bool = True):
        """Log an audit event"""
        event_data = {
            'event_type': event_type,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat(),
            'ip_address': ip_address,
            'user_agent': user_agent,
            'success': success,
            'details': details or {}
        }

        # Log to console
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"AUDIT: {event_type} - User: {user_id or 'anonymous'} - {status}")

        # Store in database (could be moved to a dedicated audit table)
        try:
            self.dal.record_metric(f"audit_{event_type}", 1, event_data)
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")

    def log_auth_attempt(self, username: str, success: bool, ip_address: str = None,
                        user_agent: str = None, method: str = "password"):
        """Log authentication attempt"""
        self.log_event(
            'auth_attempt',
            user_id=username,
            details={'method': method},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )

    def log_user_action(self, user_id: str, action: str, resource: str = None,
                       details: dict = None, ip_address: str = None, user_agent: str = None):
        """Log user action"""
        self.log_event(
            'user_action',
            user_id=user_id,
            details={
                'action': action,
                'resource': resource,
                **(details or {})
            },
            ip_address=ip_address,
            user_agent=user_agent,
            success=True
        )

    def log_security_event(self, event_type: str, user_id: str = None, details: dict = None,
                          ip_address: str = None, severity: str = "info"):
        """Log security-related event"""
        self.log_event(
            f"security_{event_type}",
            user_id=user_id,
            details={'severity': severity, **(details or {})},
            ip_address=ip_address,
            success=False  # Security events are typically failures
        )


class SecurityManager:
    """Manages security features like rate limiting and brute force protection"""

    def __init__(self, cache, audit_logger: AuditLogger = None):
        self.cache = cache
        self.audit_logger = audit_logger
        self.max_attempts = MAX_LOGIN_ATTEMPTS
        self.lockout_duration = LOCKOUT_DURATION
    
    def check_rate_limit(self, identifier: str, limit: int = 10, window: int = 60) -> bool:
        """Check if request is within rate limit"""
        key = f"rate_limit:{identifier}"
        
        current = self.cache.redis_client.incr(key)
        if current == 1:
            self.cache.redis_client.expire(key, window)
        
        return current <= limit
    
    def record_failed_attempt(self, identifier: str) -> bool:
        """Record failed login attempt"""
        key = f"failed_attempts:{identifier}"
        attempts = self.cache.redis_client.incr(key)
        
        if attempts == 1:
            self.cache.redis_client.expire(key, 3600)  # Reset after 1 hour
        
        if attempts >= self.max_attempts:
            # Lock account
            self.cache.set(f"locked:{identifier}", True, ttl=self.lockout_duration)
            logger.warning(f"Account locked due to too many failed attempts: {identifier}")
            return False
        
        return True
    
    def is_locked(self, identifier: str) -> bool:
        """Check if account is locked"""
        return self.cache.get(f"locked:{identifier}") is not None
    
    def clear_failed_attempts(self, identifier: str):
        """Clear failed attempts after successful login"""
        self.cache.delete(f"failed_attempts:{identifier}")


class AuthService:
    """Main Auth Service - Integrated with shared database"""
    
    def __init__(self):
        self.app = FastAPI(title="Auth Service (Integrated)", version="2.0.0")

        # Initialize components
        self.dal = DataAccessLayer("auth")
        self.cache = get_cache()
        self.jwt_manager = JWTManager()
        self.session_manager = SessionManager(self.cache)
        self.password_manager = PasswordManager()
        self.mfa_manager = MFAManager()
        self.rbac_engine = RBACEngine(self.dal, self.cache)
        self.audit_logger = AuditLogger(self.dal)
        self.security_manager = SecurityManager(self.cache, self.audit_logger)

        # Initialize input sanitizer
        self.sanitizer = InputSanitizer()

        # OAuth2 scheme
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

        # API keys storage (could be moved to database later)
        self.api_keys = {}

        # Initialize default admin user in database if not exists
        self._init_database_users()

        logger.info("✅ Connected to shared database and cache")

        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
    
    def _init_database_users(self):
        """Initialize default users in database if not exist"""
        try:
            # Check if admin user exists
            admin_email = "admin@example.com"
            existing_admin = self.dal.get_user_by_email(admin_email)
            
            if not existing_admin:
                # Create admin user in database
                with self.dal.db.get_db() as session:
                    from shared.models import User
                    admin = User(
                        id=str(uuid.uuid4()),
                        username="admin",
                        email=admin_email,
                        password_hash=self.password_manager.hash_password("admin"),
                        role="admin",
                        is_active=True
                    )
                    session.add(admin)
                    session.commit()
                    logger.info(f"Created default admin user: {admin_email}")
            else:
                logger.info(f"Admin user already exists: {admin_email}")
                
        except Exception as e:
            logger.error(f"Error initializing database users: {e}")
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _get_user_by_email(self, email: str) -> Optional[dict]:
        """Get user by email from database"""
        # Query the database for the user
        user_data = self.dal.get_user_by_email(email)
        if user_data:
            # Convert database user to expected format
            return {
                'id': user_data['id'],
                'email': user_data['email'],
                'username': user_data.get('username', user_data['email']),
                'password_hash': user_data['password_hash'],
                'full_name': user_data.get('username', 'User'),
                'roles': [user_data.get('role', 'user')],
                'status': UserStatus.ACTIVE.value if user_data.get('is_active', True) else UserStatus.INACTIVE.value,
                'is_active': user_data.get('is_active', True),
                'mfa_enabled': False,
                'mfa_secret': None
            }
        return None
    
    def _get_user_by_id(self, user_id: str) -> Optional[dict]:
        """Get user by ID from database"""
        with self.dal.db.get_db() as session:
            from shared.models import User
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                return {
                    'id': str(user.id),
                    'email': user.email,
                    'username': user.username,
                    'password_hash': user.password_hash,
                    'full_name': user.username or 'User',
                    'roles': [user.role or 'user'],
                    'status': UserStatus.ACTIVE.value if user.is_active else UserStatus.INACTIVE.value,
                    'is_active': user.is_active,
                    'mfa_enabled': False,
                    'mfa_secret': None
                }
        return None
    
    def _get_current_user(self, token: str = Depends(OAuth2PasswordBearer(tokenUrl="/auth/token"))) -> dict:
        """Dependency to get current user from token"""
        try:
            payload = self.jwt_manager.verify_token(token)
            user_id = payload.get("sub")
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Get user from database
            user = self._get_user_by_id(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return user
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            health_status = self.dal.health_check()
            return {
                "service": "auth",
                "status": "healthy" if health_status['database'] and health_status['cache'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "api_keys_count": len(self.api_keys),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/auth/register")
        async def register(user: UserCreate, request: Request):
            """Register new user"""
            try:
                # Sanitize user input
                sanitized_email = self.sanitizer.sanitize_text(user.email.lower().strip())
                sanitized_full_name = self.sanitizer.sanitize_text(user.full_name.strip())
                sanitized_password = user.password  # Don't sanitize password, just validate

                # Check rate limit
                if not self.security_manager.check_rate_limit(request.client.host):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")

                # Check if user exists
                if self._get_user_by_email(sanitized_email):
                    self.audit_logger.log_security_event(
                        'registration_attempt_duplicate_email',
                        user_id=sanitized_email,
                        details={'email': sanitized_email},
                        ip_address=request.client.host
                    )
                    raise HTTPException(status_code=400, detail="Email already registered")

                # Check password strength
                strength = self.password_manager.check_password_strength(sanitized_password)
                if not strength['strong']:
                    self.audit_logger.log_security_event(
                        'registration_weak_password',
                        user_id=sanitized_email,
                        details={'issues': strength['issues']},
                        ip_address=request.client.host
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"Weak password: {', '.join(strength['issues'])}"
                    )

                # Create user in database
                user_id = str(uuid.uuid4())
                with self.dal.db.get_db() as session:
                    from shared.models import User
                    new_user = User(
                        id=user_id,
                        username=sanitized_full_name,
                        email=sanitized_email,
                        password_hash=self.password_manager.hash_password(sanitized_password),
                        role=user.role.value,
                        is_active=False  # Require email verification
                    )
                    session.add(new_user)
                    session.commit()

                # Log registration
                self.audit_logger.log_user_action(
                    user_id=user_id,
                    action='user_registration',
                    details={
                        'email': sanitized_email,
                        'role': user.role.value,
                        'method': 'password'
                    },
                    ip_address=request.client.host,
                    user_agent=request.headers.get("user-agent")
                )

                self.dal.record_metric('user_registration', 1, {
                    'email': sanitized_email,
                    'role': user.role.value
                })

                logger.info(f"User registered: {sanitized_email}")

                return {
                    "message": "User registered successfully",
                    "user_id": user_id,
                    "email": sanitized_email
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/auth/token")
        async def login(form_data: OAuth2PasswordRequestForm = Depends(), request: Request = None):
            """Login and get tokens"""
            try:
                # Sanitize input
                sanitized_username = self.sanitizer.sanitize_text(form_data.username.lower().strip())

                # Check if account is locked
                if self.security_manager.is_locked(sanitized_username):
                    raise HTTPException(
                        status_code=403,
                        detail="Account locked due to too many failed attempts"
                    )

                # Get user
                user = self._get_user_by_email(sanitized_username)
                if not user:
                    self.audit_logger.log_auth_attempt(
                        username=sanitized_username,
                        success=False,
                        ip_address=request.client.host if request else "unknown",
                        user_agent=request.headers.get("user-agent") if request else "unknown",
                        method="password"
                    )
                    self.security_manager.record_failed_attempt(sanitized_username)
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid email or password"
                    )

                # Verify password
                if not self.password_manager.verify_password(form_data.password, user['password_hash']):
                    self.audit_logger.log_auth_attempt(
                        username=sanitized_username,
                        success=False,
                        ip_address=request.client.host if request else "unknown",
                        user_agent=request.headers.get("user-agent") if request else "unknown",
                        method="password"
                    )
                    self.security_manager.record_failed_attempt(sanitized_username)
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid email or password"
                    )

                # Check account status
                if user['status'] != UserStatus.ACTIVE.value:
                    self.audit_logger.log_security_event(
                        'login_inactive_account',
                        user_id=user['id'],
                        details={'status': user['status']},
                        ip_address=request.client.host if request else "unknown"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail=f"Account is {user['status']}"
                    )

                # Clear failed attempts
                self.security_manager.clear_failed_attempts(sanitized_username)

                # Generate tokens
                tokens = self.jwt_manager.generate_tokens(
                    user['id'],
                    user['email'],
                    user['roles']
                )

                # Create session
                session_id = self.session_manager.create_session(user['id'], {
                    "ip": request.client.host if request else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown") if request else "unknown"
                })

                # Log successful login
                self.audit_logger.log_auth_attempt(
                    username=sanitized_username,
                    success=True,
                    ip_address=request.client.host if request else "unknown",
                    user_agent=request.headers.get("user-agent") if request else "unknown",
                    method="password"
                )

                self.audit_logger.log_user_action(
                    user_id=user['id'],
                    action='login',
                    details={'method': 'password'},
                    ip_address=request.client.host if request else "unknown",
                    user_agent=request.headers.get("user-agent") if request else "unknown"
                )

                self.dal.record_metric('user_login', 1, {
                    'user_id': user['id'],
                    'email': user['email']
                })

                logger.info(f"User logged in: {user['email']}")

                return {
                    **tokens,
                    "session_id": session_id
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Login failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/auth/refresh")
        async def refresh_token(refresh: TokenRefresh):
            """Refresh access token"""
            try:
                # Verify refresh token
                payload = self.jwt_manager.verify_token(refresh.refresh_token, TokenType.REFRESH)
                user_id = payload.get("sub")
                
                if not user_id:
                    raise HTTPException(status_code=401, detail="Invalid refresh token")
                
                # Get user from database
                user = self._get_user_by_id(user_id)
                if not user:
                    raise HTTPException(status_code=401, detail="User not found")
                
                # Generate new tokens
                tokens = self.jwt_manager.generate_tokens(
                    user['id'],
                    user['email'],
                    user['roles']
                )
                
                return tokens
                
            except ValueError as e:
                raise HTTPException(status_code=401, detail=str(e))
            except Exception as e:
                logger.error(f"Token refresh failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/auth/logout")
        async def logout(
            session_id: Optional[str] = Header(None),
            current_user: dict = Depends(self._get_current_user)
        ):
            """Logout user"""
            try:
                # Destroy session if provided
                if session_id:
                    self.session_manager.destroy_session(session_id)

                # In production, would also blacklist the token

                logger.info(f"User logged out: {current_user['email']}")

                return {"message": "Logged out successfully"}

            except Exception as e:
                logger.error(f"Logout failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # OAuth2/OIDC endpoints
        if OAUTH_ENABLED:
            @self.app.get("/auth/oauth/login")
            async def oauth_login(state: str = None):
                """Initiate OAuth2/OIDC login"""
                try:
                    if not oauth_manager or not oauth_manager.is_oauth_enabled():
                        raise HTTPException(status_code=503, detail="OAuth2 not configured")

                    authorization_url = oauth_manager.get_authorization_url(state)
                    return {"authorization_url": authorization_url}

                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"OAuth login failed: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @self.app.get("/auth/oauth/callback")
            async def oauth_callback(code: str, state: str = None):
                """Handle OAuth2/OIDC callback"""
                try:
                    if not oauth_manager or not oauth_manager.is_oauth_enabled():
                        raise HTTPException(status_code=503, detail="OAuth2 not configured")

                    # Process the callback
                    session_data = await oauth_manager.process_callback(code, state)

                    # Create session for the user
                    user_info = session_data['user_info']
                    session_id = self.session_manager.create_session(
                        user_info.get('sub', user_info.get('id')),
                        {
                            "auth_method": "oauth2",
                            "provider": "oidc",
                            "user_info": user_info
                        }
                    )

                    # Log successful OAuth login
                    self.dal.record_metric('oauth_login', 1, {
                        'user_id': user_info.get('sub', user_info.get('id')),
                        'email': user_info.get('email'),
                        'provider': 'oidc'
                    })

                    logger.info(f"OAuth login successful: {user_info.get('email', 'unknown')}")

                    return {
                        **session_data,
                        "session_id": session_id
                    }

                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"OAuth callback failed: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @self.app.get("/auth/oauth/providers")
            async def get_oauth_providers():
                """Get available OAuth2/OIDC providers"""
                providers = {
                    "oauth_enabled": OAUTH_ENABLED and oauth_manager and oauth_manager.is_oauth_enabled()
                }

                if providers["oauth_enabled"]:
                    providers["authorization_url"] = oauth_manager.get_authorization_url()

                # Add popular provider configurations
                if OIDCProviderConfig:
                    providers["available_providers"] = OIDCProviderConfig.get_available_providers()
                    providers["supported_providers"] = list(OIDCProviderConfig.PROVIDERS.keys())

                return providers

            @self.app.get("/auth/oauth/providers/{provider_name}/setup")
            async def get_provider_setup(provider_name: str):
                """Get setup instructions for a specific OIDC provider"""
                if not get_provider_setup_instructions:
                    raise HTTPException(status_code=503, detail="OIDC providers not available")

                if provider_name not in OIDCProviderConfig.PROVIDERS:
                    raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found")

                instructions = get_provider_setup_instructions(provider_name)
                return {
                    "provider": provider_name,
                    "setup_instructions": instructions,
                    "required_env_vars": OIDCProviderConfig.PROVIDERS[provider_name]
                }
        
        @self.app.get("/auth/me")
        async def get_current_user(current_user: dict = Depends(self._get_current_user)):
            """Get current user info"""
            return {
                "id": current_user['id'],
                "email": current_user['email'],
                "full_name": current_user['full_name'],
                "roles": current_user['roles'],
                "status": current_user['status'],
                "permissions": self.rbac_engine.get_user_permissions(current_user['roles'])
            }
        
        @self.app.post("/auth/change-password")
        async def change_password(
            password_change: PasswordChange,
            current_user: dict = Depends(self._get_current_user)
        ):
            """Change user password"""
            try:
                # Verify current password
                if not self.password_manager.verify_password(
                    password_change.current_password,
                    current_user['password_hash']
                ):
                    raise HTTPException(status_code=400, detail="Current password is incorrect")
                
                # Check new password strength
                strength = self.password_manager.check_password_strength(password_change.new_password)
                if not strength['strong']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Weak password: {', '.join(strength['issues'])}"
                    )
                
                # Update password in database
                with self.dal.db.get_db() as session:
                    from shared.models import User
                    user = session.query(User).filter(User.id == current_user['id']).first()
                    if user:
                        user.password_hash = self.password_manager.hash_password(
                            password_change.new_password
                        )
                        session.commit()
                
                logger.info(f"Password changed for user: {current_user['email']}")
                
                return {"message": "Password changed successfully"}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Password change failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/auth/enable-mfa")
        async def enable_mfa(current_user: dict = Depends(self._get_current_user)):
            """Enable MFA for user"""
            try:
                # Generate secret (would be stored in database in production)
                secret = self.mfa_manager.generate_secret()
                # Note: In production, store mfa_secret in database
                
                # Generate QR code URI
                qr_uri = self.mfa_manager.generate_qr_code_uri(current_user['email'], secret)
                
                return {
                    "secret": secret,
                    "qr_code_uri": qr_uri,
                    "message": "Scan QR code with authenticator app and verify"
                }
                
            except Exception as e:
                logger.error(f"MFA enable failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/auth/verify-mfa")
        async def verify_mfa(
            mfa_code: str,
            current_user: dict = Depends(self._get_current_user)
        ):
            """Verify MFA setup"""
            try:
                # Note: In production, retrieve mfa_secret from database
                # For now, MFA is not fully implemented with database storage
                raise HTTPException(status_code=501, detail="MFA verification not fully implemented with database")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"MFA verification failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/auth/api-keys")
        async def create_api_key(
            api_key_req: APIKeyCreate,
            current_user: dict = Depends(self._get_current_user)
        ):
            """Create API key"""
            try:
                # Sanitize input
                sanitized_name = self.sanitizer.sanitize_text(api_key_req.name.strip())
                sanitized_service_name = (
                    self.sanitizer.sanitize_text(api_key_req.service_name.strip())
                    if api_key_req.service_name else None
                )

                # Check permission
                if not self.rbac_engine.check_permission(
                    current_user['roles'], "api_keys", "create"
                ):
                    raise HTTPException(status_code=403, detail="Permission denied")

                # Generate API key
                api_key = self.jwt_manager.generate_api_key()
                key_id = str(uuid.uuid4())

                # Store API key (hash it in production)
                self.api_keys[key_id] = {
                    "id": key_id,
                    "name": sanitized_name,
                    "key_hash": hashlib.sha256(api_key.encode()).hexdigest(),
                    "service_name": sanitized_service_name,
                    "permissions": api_key_req.permissions,
                    "created_by": current_user['id'],
                    "created_at": datetime.utcnow().isoformat(),
                    "expires_at": (
                        datetime.utcnow() + timedelta(days=api_key_req.expires_in_days)
                    ).isoformat() if api_key_req.expires_in_days else None,
                    "active": True
                }

                logger.info(f"API key created: {sanitized_name} by {current_user['email']}")

                return {
                    "api_key": api_key,  # Only return once, user must save it
                    "key_id": key_id,
                    "name": sanitized_name
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"API key creation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/auth/api-keys")
        async def list_api_keys(current_user: dict = Depends(self._get_current_user)):
            """List API keys"""
            try:
                # Filter keys created by user (or all for admin)
                if UserRole.ADMIN.value in current_user['roles']:
                    keys = list(self.api_keys.values())
                else:
                    keys = [
                        k for k in self.api_keys.values()
                        if k['created_by'] == current_user['id']
                    ]
                
                # Don't expose the actual keys
                return [{
                    "id": k['id'],
                    "name": k['name'],
                    "service_name": k['service_name'],
                    "created_at": k['created_at'],
                    "expires_at": k['expires_at'],
                    "active": k['active']
                } for k in keys]
                
            except Exception as e:
                logger.error(f"API key listing failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/auth/api-keys/{key_id}")
        async def revoke_api_key(
            key_id: str,
            current_user: dict = Depends(self._get_current_user)
        ):
            """Revoke API key"""
            try:
                if key_id not in self.api_keys:
                    raise HTTPException(status_code=404, detail="API key not found")
                
                api_key = self.api_keys[key_id]
                
                # Check permission
                if (api_key['created_by'] != current_user['id'] and
                    UserRole.ADMIN.value not in current_user['roles']):
                    raise HTTPException(status_code=403, detail="Permission denied")
                
                # Revoke key
                api_key['active'] = False
                api_key['revoked_at'] = datetime.utcnow().isoformat()
                
                logger.info(f"API key revoked: {api_key['name']} by {current_user['email']}")
                
                return {"message": "API key revoked successfully"}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"API key revocation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""
        
        def on_user_event(event):
            """Handle user-related events"""
            event_type = event.data.get('type')
            user_id = event.data.get('user_id')
            logger.info(f"Received user event {event_type} for {user_id}")
        
        # Register handlers
        self.dal.event_bus.on(EventChannel.SYSTEM_ALERT, on_user_event)
        
        logger.info("Event handlers registered for cross-service communication")
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Auth Service (Integrated) starting up...")
        
        # Verify connections
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        if not health['cache']:
            logger.warning("Cache not available, performance may be degraded")
        
        logger.info("Auth Service ready")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Auth Service shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = AuthService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("AUTH_PORT", 8106))
    logger.info(f"Starting Auth Service (Integrated) on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()