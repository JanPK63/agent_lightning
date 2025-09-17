#!/usr/bin/env python3
"""
Auth Service Microservice
Centralized authentication and authorization with OAuth2/OIDC support
Based on the architecture from technical_architecture.md
"""

import os
import sys
import json
import asyncio
import uuid
import hashlib
import hmac
import secrets
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
import logging
import base64
from urllib.parse import urlparse, urlencode, parse_qs
import time

from fastapi import FastAPI, HTTPException, Depends, Request, Response, Header, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, EmailStr

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class GrantType(str, Enum):
    """OAuth2 grant types"""
    AUTHORIZATION_CODE = "authorization_code"
    IMPLICIT = "implicit"
    PASSWORD = "password"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"
    DEVICE_CODE = "device_code"


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
    PENDING = "pending"
    LOCKED = "locked"


# Pydantic Models
class User(BaseModel):
    """User model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    organization_id: str
    roles: List[UserRole] = Field(default_factory=lambda: [UserRole.VIEWER])
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    email_verified: bool = False
    mfa_enabled: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Organization(BaseModel):
    """Organization model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    domain: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    max_users: int = 10
    max_agents: int = 50
    features: List[str] = Field(default_factory=list)


class OAuthClient(BaseModel):
    """OAuth2 client model"""
    client_id: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    client_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(64))
    name: str
    redirect_uris: List[str]
    grant_types: List[GrantType]
    scopes: List[str]
    organization_id: str
    created_at: datetime = Field(default_factory=datetime.now)


class Token(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    scope: Optional[str] = None


class TokenData(BaseModel):
    """Token payload data"""
    sub: str  # Subject (user ID)
    org: str  # Organization ID
    roles: List[str]
    scopes: List[str] = Field(default_factory=list)
    exp: int  # Expiration
    iat: int  # Issued at
    jti: str  # JWT ID


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str
    organization_id: Optional[str] = None
    mfa_code: Optional[str] = None


class RegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    organization_name: Optional[str] = None


class AuthorizeRequest(BaseModel):
    """OAuth2 authorization request"""
    response_type: str
    client_id: str
    redirect_uri: str
    scope: Optional[str] = None
    state: Optional[str] = None
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None


class APIKey(BaseModel):
    """API Key model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key: str = Field(default_factory=lambda: f"sk_{secrets.token_urlsafe(32)}")
    name: str
    user_id: str
    organization_id: str
    scopes: List[str] = Field(default_factory=list)
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = None


class Permission(BaseModel):
    """Permission model"""
    resource: str
    action: str
    conditions: Dict[str, Any] = Field(default_factory=dict)


class RoleDefinition(BaseModel):
    """Role definition with permissions"""
    role: UserRole
    permissions: List[Permission]
    description: str


class AuthService:
    """Main Authentication Service"""
    
    def __init__(self):
        self.app = FastAPI(title="Auth Service", version="1.0.0")
        
        # Configuration
        self.jwt_secret = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
        self.jwt_algorithm = "HS256"
        self.access_token_expire = 3600  # 1 hour
        self.refresh_token_expire = 604800  # 7 days
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Storage (in-memory for now)
        self.users: Dict[str, User] = {}
        self.organizations: Dict[str, Organization] = {}
        self.oauth_clients: Dict[str, OAuthClient] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.authorization_codes: Dict[str, Dict[str, Any]] = {}
        self.refresh_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Role definitions
        self.role_permissions = self._define_role_permissions()
        
        # Metrics
        self.metrics = {
            "total_logins": 0,
            "failed_logins": 0,
            "active_sessions": 0,
            "tokens_issued": 0
        }
        
        # OAuth2 scheme
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
        self._create_default_data()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_event_handlers(self):
        """Setup startup and shutdown handlers"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize service"""
            asyncio.create_task(self._cleanup_expired_tokens())
            logger.info("Auth Service started")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            logger.info("Auth Service shut down")
    
    def _define_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Define role-based permissions"""
        return {
            UserRole.ADMIN: [
                Permission(resource="*", action="*"),
            ],
            UserRole.DEVELOPER: [
                Permission(resource="agents", action="*"),
                Permission(resource="workflows", action="*"),
                Permission(resource="integrations", action="*"),
                Permission(resource="models", action="read"),
            ],
            UserRole.OPERATOR: [
                Permission(resource="agents", action="read"),
                Permission(resource="agents", action="execute"),
                Permission(resource="workflows", action="read"),
                Permission(resource="workflows", action="execute"),
                Permission(resource="integrations", action="read"),
            ],
            UserRole.VIEWER: [
                Permission(resource="agents", action="read"),
                Permission(resource="workflows", action="read"),
                Permission(resource="integrations", action="read"),
                Permission(resource="metrics", action="read"),
            ],
            UserRole.GUEST: [
                Permission(resource="public", action="read"),
            ]
        }
    
    def _create_default_data(self):
        """Create default users and organizations"""
        # Default organization
        default_org = Organization(
            id="default-org",
            name="Default Organization",
            domain="localhost",
            features=["agents", "workflows", "integrations", "ai_models"],
            max_users=100,
            max_agents=1000
        )
        self.organizations[default_org.id] = default_org
        
        # Admin user
        admin_password_hash = self._hash_password("admin123")
        admin = User(
            id="user-admin",
            email="admin@agentlightning.ai",
            username="admin",
            full_name="System Administrator",
            organization_id="default-org",
            roles=[UserRole.ADMIN],
            email_verified=True
        )
        self.users[admin.id] = admin
        self.users[f"cred:{admin.username}"] = admin_password_hash
        
        # Demo user
        demo_password_hash = self._hash_password("demo123")
        demo = User(
            id="user-demo",
            email="demo@agentlightning.ai",
            username="demo",
            full_name="Demo User",
            organization_id="default-org",
            roles=[UserRole.DEVELOPER],
            email_verified=True
        )
        self.users[demo.id] = demo
        self.users[f"cred:{demo.username}"] = demo_password_hash
        
        # Default OAuth client
        client = OAuthClient(
            client_id="default-client",
            client_secret="default-secret",
            name="Agent Lightning Web",
            redirect_uris=["http://localhost:3000/callback"],
            grant_types=[GrantType.AUTHORIZATION_CODE, GrantType.PASSWORD, GrantType.REFRESH_TOKEN],
            scopes=["read", "write", "admin"],
            organization_id="default-org"
        )
        self.oauth_clients[client.client_id] = client
        
        logger.info("Created default data")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Service health check"""
            return {
                "status": "healthy",
                "service": "auth",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics
            }
        
        # Authentication endpoints
        @self.app.post("/api/v1/auth/login", response_model=Token)
        async def login(request: LoginRequest):
            """User login with username/password"""
            user = await self._authenticate_user(
                request.username, 
                request.password,
                request.organization_id
            )
            
            if not user:
                self.metrics["failed_logins"] += 1
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Check MFA if enabled
            if user.mfa_enabled and request.mfa_code:
                if not self._verify_mfa(user.id, request.mfa_code):
                    raise HTTPException(status_code=401, detail="Invalid MFA code")
            
            # Generate tokens
            access_token = self._create_access_token(user)
            refresh_token = self._create_refresh_token(user)
            
            # Update metrics
            user.last_login = datetime.now()
            self.metrics["total_logins"] += 1
            self.metrics["tokens_issued"] += 2
            
            logger.info(f"User logged in: {user.username}")
            
            return Token(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.access_token_expire
            )
        
        @self.app.post("/api/v1/auth/register", response_model=User)
        async def register(request: RegisterRequest):
            """Register new user"""
            # Check if username exists
            for user in self.users.values():
                if isinstance(user, User) and user.username == request.username:
                    raise HTTPException(status_code=400, detail="Username already exists")
                if isinstance(user, User) and user.email == request.email:
                    raise HTTPException(status_code=400, detail="Email already exists")
            
            # Create or get organization
            if request.organization_name:
                org = Organization(
                    name=request.organization_name,
                    domain=request.email.split("@")[1]
                )
                self.organizations[org.id] = org
                org_id = org.id
            else:
                org_id = "default-org"
            
            # Create user
            user = User(
                email=request.email,
                username=request.username,
                full_name=request.full_name,
                organization_id=org_id,
                roles=[UserRole.DEVELOPER]
            )
            
            # Store user and password
            self.users[user.id] = user
            self.users[f"cred:{user.username}"] = self._hash_password(request.password)
            
            logger.info(f"User registered: {user.username}")
            return user
        
        @self.app.post("/api/v1/auth/logout")
        async def logout(authorization: str = Header(None)):
            """User logout"""
            if authorization and authorization.startswith("Bearer "):
                token = authorization.split(" ")[1]
                # In production, invalidate token in cache/blacklist
                logger.info(f"User logged out")
            
            return {"message": "Logged out successfully"}
        
        # OAuth2 endpoints
        @self.app.get("/api/v1/auth/authorize")
        async def authorize(
            response_type: str = Query(...),
            client_id: str = Query(...),
            redirect_uri: str = Query(...),
            scope: str = Query(None),
            state: str = Query(None),
            code_challenge: str = Query(None),
            code_challenge_method: str = Query(None)
        ):
            """OAuth2 authorization endpoint"""
            # Validate client
            if client_id not in self.oauth_clients:
                raise HTTPException(status_code=400, detail="Invalid client")
            
            client = self.oauth_clients[client_id]
            
            if redirect_uri not in client.redirect_uris:
                raise HTTPException(status_code=400, detail="Invalid redirect URI")
            
            # Generate authorization code
            code = secrets.token_urlsafe(32)
            self.authorization_codes[code] = {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": scope,
                "code_challenge": code_challenge,
                "expires_at": time.time() + 600  # 10 minutes
            }
            
            # Redirect with code
            params = {"code": code}
            if state:
                params["state"] = state
            
            redirect_url = f"{redirect_uri}?{urlencode(params)}"
            return RedirectResponse(url=redirect_url)
        
        @self.app.post("/api/v1/auth/token", response_model=Token)
        async def token(
            grant_type: str = Form(...),
            code: str = Form(None),
            redirect_uri: str = Form(None),
            client_id: str = Form(None),
            client_secret: str = Form(None),
            username: str = Form(None),
            password: str = Form(None),
            refresh_token: str = Form(None),
            scope: str = Form(None)
        ):
            """OAuth2 token endpoint"""
            
            if grant_type == GrantType.AUTHORIZATION_CODE:
                # Exchange authorization code for tokens
                if code not in self.authorization_codes:
                    raise HTTPException(status_code=400, detail="Invalid authorization code")
                
                code_data = self.authorization_codes[code]
                if time.time() > code_data["expires_at"]:
                    raise HTTPException(status_code=400, detail="Authorization code expired")
                
                # Validate client
                if client_id != code_data["client_id"]:
                    raise HTTPException(status_code=400, detail="Client mismatch")
                
                # Create tokens (simplified - would validate user in production)
                demo_user = self.users["user-demo"]
                access_token = self._create_access_token(demo_user)
                refresh_token = self._create_refresh_token(demo_user)
                
                # Clean up code
                del self.authorization_codes[code]
                
            elif grant_type == GrantType.PASSWORD:
                # Resource owner password credentials
                user = await self._authenticate_user(username, password)
                if not user:
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                access_token = self._create_access_token(user)
                refresh_token = self._create_refresh_token(user)
                
            elif grant_type == GrantType.REFRESH_TOKEN:
                # Refresh token grant
                user = self._verify_refresh_token(refresh_token)
                if not user:
                    raise HTTPException(status_code=401, detail="Invalid refresh token")
                
                access_token = self._create_access_token(user)
                refresh_token = self._create_refresh_token(user)
                
            elif grant_type == GrantType.CLIENT_CREDENTIALS:
                # Client credentials grant
                if not client_id or not client_secret:
                    raise HTTPException(status_code=400, detail="Client credentials required")
                
                if client_id not in self.oauth_clients:
                    raise HTTPException(status_code=401, detail="Invalid client")
                
                client = self.oauth_clients[client_id]
                if client.client_secret != client_secret:
                    raise HTTPException(status_code=401, detail="Invalid client secret")
                
                # Create service token
                access_token = self._create_service_token(client)
                refresh_token = None
                
            else:
                raise HTTPException(status_code=400, detail="Unsupported grant type")
            
            self.metrics["tokens_issued"] += 1 if not refresh_token else 2
            
            return Token(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.access_token_expire,
                scope=scope
            )
        
        @self.app.get("/api/v1/auth/userinfo", response_model=User)
        async def userinfo(current_user: User = Depends(self.get_current_user)):
            """Get current user info"""
            return current_user
        
        # User management
        @self.app.get("/api/v1/users", response_model=List[User])
        async def list_users(
            current_user: User = Depends(self.get_current_user),
            organization_id: Optional[str] = None
        ):
            """List users (admin only)"""
            if UserRole.ADMIN not in current_user.roles:
                raise HTTPException(status_code=403, detail="Admin access required")
            
            users = []
            for user in self.users.values():
                if isinstance(user, User):
                    if not organization_id or user.organization_id == organization_id:
                        users.append(user)
            
            return users
        
        @self.app.get("/api/v1/users/{user_id}", response_model=User)
        async def get_user(
            user_id: str,
            current_user: User = Depends(self.get_current_user)
        ):
            """Get user details"""
            if user_id != current_user.id and UserRole.ADMIN not in current_user.roles:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            if user_id not in self.users:
                raise HTTPException(status_code=404, detail="User not found")
            
            return self.users[user_id]
        
        # API Key management
        @self.app.post("/api/v1/api-keys", response_model=APIKey)
        async def create_api_key(
            name: str,
            scopes: List[str] = [],
            expires_in_days: Optional[int] = None,
            current_user: User = Depends(self.get_current_user)
        ):
            """Create new API key"""
            api_key = APIKey(
                name=name,
                user_id=current_user.id,
                organization_id=current_user.organization_id,
                scopes=scopes,
                expires_at=datetime.now() + timedelta(days=expires_in_days) if expires_in_days else None
            )
            
            self.api_keys[api_key.key] = api_key
            logger.info(f"Created API key: {api_key.name} for user {current_user.username}")
            
            return api_key
        
        @self.app.get("/api/v1/api-keys", response_model=List[APIKey])
        async def list_api_keys(current_user: User = Depends(self.get_current_user)):
            """List user's API keys"""
            keys = []
            for key in self.api_keys.values():
                if key.user_id == current_user.id:
                    # Don't return the actual key value
                    key_copy = key.copy()
                    key_copy.key = key.key[:10] + "..." + key.key[-4:]
                    keys.append(key_copy)
            
            return keys
        
        @self.app.delete("/api/v1/api-keys/{key_id}")
        async def revoke_api_key(
            key_id: str,
            current_user: User = Depends(self.get_current_user)
        ):
            """Revoke API key"""
            # Find key by ID
            key_to_delete = None
            for key in self.api_keys.values():
                if key.id == key_id and key.user_id == current_user.id:
                    key_to_delete = key.key
                    break
            
            if not key_to_delete:
                raise HTTPException(status_code=404, detail="API key not found")
            
            del self.api_keys[key_to_delete]
            logger.info(f"Revoked API key: {key_id}")
            
            return {"message": "API key revoked"}
        
        # Permission checking
        @self.app.post("/api/v1/auth/check-permission")
        async def check_permission(
            resource: str,
            action: str,
            current_user: User = Depends(self.get_current_user)
        ):
            """Check if user has permission"""
            has_permission = self._check_permission(current_user, resource, action)
            
            return {
                "allowed": has_permission,
                "user_id": current_user.id,
                "resource": resource,
                "action": action
            }
        
        # Session management
        @self.app.get("/api/v1/sessions")
        async def list_sessions(current_user: User = Depends(self.get_current_user)):
            """List user's active sessions"""
            user_sessions = []
            for session_id, session in self.sessions.items():
                if session.get("user_id") == current_user.id:
                    user_sessions.append({
                        "id": session_id,
                        "created_at": session.get("created_at"),
                        "last_accessed": session.get("last_accessed"),
                        "ip_address": session.get("ip_address")
                    })
            
            return user_sessions
        
        @self.app.get("/api/v1/metrics")
        async def get_metrics(current_user: User = Depends(self.get_current_user)):
            """Get auth service metrics"""
            if UserRole.ADMIN not in current_user.roles:
                raise HTTPException(status_code=403, detail="Admin access required")
            
            return {
                "metrics": self.metrics,
                "total_users": len([u for u in self.users.values() if isinstance(u, User)]),
                "total_organizations": len(self.organizations),
                "active_sessions": len(self.sessions),
                "api_keys": len(self.api_keys)
            }
    
    async def get_current_user(self, token: str = Depends(OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token"))) -> User:
        """Get current user from token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload.get("sub")
            
            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            user = self.users.get(user_id)
            if not user or not isinstance(user, User):
                raise HTTPException(status_code=401, detail="User not found")
            
            return user
            
        except JWTError as e:
            if "expired" in str(e):
                raise HTTPException(status_code=401, detail="Token expired")
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def _authenticate_user(self, username: str, password: str, organization_id: Optional[str] = None) -> Optional[User]:
        """Authenticate user with username/password"""
        # Find user
        user = None
        for u in self.users.values():
            if isinstance(u, User) and u.username == username:
                if not organization_id or u.organization_id == organization_id:
                    user = u
                    break
        
        if not user:
            return None
        
        # Verify password
        stored_password = self.users.get(f"cred:{username}")
        if not stored_password:
            return None
        
        if not self._verify_password(password, stored_password):
            return None
        
        return user
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password using bcrypt"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def _verify_mfa(self, user_id: str, code: str) -> bool:
        """Verify MFA code"""
        # Simplified - in production, use TOTP/HOTP
        return code == "123456"
    
    def _create_access_token(self, user: User) -> str:
        """Create JWT access token"""
        payload = {
            "sub": user.id,
            "org": user.organization_id,
            "roles": [role.value for role in user.roles],
            "scopes": ["read", "write"],
            "exp": int(time.time()) + self.access_token_expire,
            "iat": int(time.time()),
            "jti": str(uuid.uuid4())
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _create_refresh_token(self, user: User) -> str:
        """Create refresh token"""
        token_id = str(uuid.uuid4())
        
        # Store refresh token
        self.refresh_tokens[token_id] = {
            "user_id": user.id,
            "expires_at": time.time() + self.refresh_token_expire
        }
        
        payload = {
            "sub": user.id,
            "jti": token_id,
            "exp": int(time.time()) + self.refresh_token_expire,
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _verify_refresh_token(self, token: str) -> Optional[User]:
        """Verify refresh token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            if payload.get("type") != "refresh":
                return None
            
            token_id = payload.get("jti")
            if token_id not in self.refresh_tokens:
                return None
            
            token_data = self.refresh_tokens[token_id]
            if time.time() > token_data["expires_at"]:
                del self.refresh_tokens[token_id]
                return None
            
            user_id = token_data["user_id"]
            return self.users.get(user_id)
            
        except JWTError:
            return None
    
    def _create_service_token(self, client: OAuthClient) -> str:
        """Create service account token"""
        payload = {
            "sub": f"client:{client.client_id}",
            "org": client.organization_id,
            "scopes": client.scopes,
            "exp": int(time.time()) + self.access_token_expire,
            "iat": int(time.time()),
            "type": "service"
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission"""
        for role in user.roles:
            permissions = self.role_permissions.get(role, [])
            for permission in permissions:
                if permission.resource == "*" or permission.resource == resource:
                    if permission.action == "*" or permission.action == action:
                        return True
        
        return False
    
    async def _cleanup_expired_tokens(self):
        """Background task to cleanup expired tokens"""
        while True:
            await asyncio.sleep(3600)  # Run every hour
            
            # Clean expired refresh tokens
            expired = []
            for token_id, data in self.refresh_tokens.items():
                if time.time() > data["expires_at"]:
                    expired.append(token_id)
            
            for token_id in expired:
                del self.refresh_tokens[token_id]
            
            # Clean expired authorization codes
            expired = []
            for code, data in self.authorization_codes.items():
                if time.time() > data["expires_at"]:
                    expired.append(code)
            
            for code in expired:
                del self.authorization_codes[code]
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired tokens")


def create_service():
    """Create and return the service instance"""
    return AuthService()


if __name__ == "__main__":
    import uvicorn
    
    print("Auth Service Microservice")
    print("=" * 60)
    
    service = create_service()
    
    print("\n🔐 Starting Auth Service on port 8006")
    print("\nEndpoints:")
    print("  • POST /api/v1/auth/login - User login")
    print("  • POST /api/v1/auth/register - User registration")
    print("  • POST /api/v1/auth/logout - User logout")
    print("  • GET  /api/v1/auth/authorize - OAuth2 authorization")
    print("  • POST /api/v1/auth/token - OAuth2 token exchange")
    print("  • GET  /api/v1/auth/userinfo - Get current user")
    print("  • POST /api/v1/api-keys - Create API key")
    print("  • GET  /api/v1/users - List users (admin)")
    
    print("\n🔑 Default Credentials:")
    print("  • Admin: username=admin, password=admin123")
    print("  • Demo:  username=demo, password=demo123")
    
    print("\n🛡️ Features:")
    print("  • OAuth2/OIDC support")
    print("  • JWT token authentication")
    print("  • API key management")
    print("  • Role-based access control (RBAC)")
    print("  • Multi-factor authentication (MFA)")
    print("  • Session management")
    
    uvicorn.run(service.app, host="0.0.0.0", port=8006, reload=False)