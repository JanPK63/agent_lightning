# Solution Architecture: Authentication & Authorization Service

**Document ID:** SA-007  
**Date:** 2025-09-06  
**Status:** For Review  
**Priority:** Critical  
**Author:** System Architect  
**Dependencies:** SA-001 (Database), SA-002 (Redis), SA-003 (Microservices)  

## Executive Summary

This solution architecture details the integration of the Authentication & Authorization service with the shared PostgreSQL database and Redis cache. The Auth service is critical for securing all system endpoints, managing user sessions, and enforcing role-based access control (RBAC).

## Problem Statement

### Current Issues

- **In-Memory Sessions:** User sessions lost on service restart
- **No Token Persistence:** JWT tokens not tracked or revokable
- **Isolated User Management:** User data not shared across services
- **No Audit Trail:** Authentication events not logged
- **Limited Security:** No rate limiting or brute force protection

### Business Impact

- Security vulnerabilities from untracked sessions
- Cannot revoke compromised tokens
- Poor user experience with lost sessions
- No compliance audit trail
- Vulnerable to authentication attacks

## Proposed Solution

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Auth Service                                  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Authentication Manager                     │   │
│  │                                                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │   JWT    │  │  Session │  │   RBAC   │             │   │
│  │  │ Manager  │  │  Manager │  │  Engine  │             │   │
│  │  └─────┬────┘  └─────┬────┘  └─────┬────┘             │   │
│  │        │             │              │                   │   │
│  └────────┼─────────────┼──────────────┼───────────────────┘   │
│           │             │              │                        │
│  ┌────────▼─────────────▼──────────────▼────────────────────┐   │
│  │              Data Access Layer (DAL)                     │   │
│  │                                                          │   │
│  │  - User management                                       │   │
│  │  - Session persistence                                   │   │
│  │  - Token blacklist                                       │   │
│  │  - Audit logging                                         │   │
│  └────────┬──────────────────────────┬──────────────────────┘   │
│           │                          │                          │
└───────────┼──────────────────────────┼──────────────────────────┘
            │                          │
     ┌──────▼──────┐            ┌──────▼──────┐
     │  PostgreSQL │            │    Redis    │
     │             │            │             │
     │ • Users     │            │ • Sessions  │
     │ • Roles     │            │ • Tokens    │
     │ • Audit     │            │ • Rate Limit│
     └─────────────┘            └─────────────┘
```

### Database Schema Extensions

```sql
-- Extend users table with auth fields
ALTER TABLE users ADD COLUMN IF NOT EXISTS
    password_hash VARCHAR(255),
    email_verified BOOLEAN DEFAULT false,
    two_factor_enabled BOOLEAN DEFAULT false,
    two_factor_secret VARCHAR(255),
    last_login TIMESTAMP WITH TIME ZONE,
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE;

-- Roles table
CREATE TABLE IF NOT EXISTS roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User roles mapping
CREATE TABLE IF NOT EXISTS user_roles (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    granted_by UUID REFERENCES users(id),
    PRIMARY KEY (user_id, role_id)
);

-- Refresh tokens
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    revoked BOOLEAN DEFAULT false,
    revoked_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Authentication audit log
CREATE TABLE IF NOT EXISTS auth_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    event_type VARCHAR(50) NOT NULL, -- 'login', 'logout', 'token_refresh', 'password_change'
    success BOOLEAN NOT NULL,
    ip_address INET,
    user_agent TEXT,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API keys for service authentication
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    service_name VARCHAR(100),
    permissions JSONB DEFAULT '[]',
    last_used TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX idx_auth_audit_user ON auth_audit(user_id);
CREATE INDEX idx_auth_audit_created ON auth_audit(created_at);
```

### Key Components

#### 1. JWT Manager

```python
import jwt
from datetime import datetime, timedelta

class JWTManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_ttl = timedelta(minutes=15)
        self.refresh_token_ttl = timedelta(days=30)
    
    def generate_tokens(self, user_id: str, roles: List[str]) -> dict:
        """Generate access and refresh tokens"""
        now = datetime.utcnow()
        
        # Access token payload
        access_payload = {
            "sub": user_id,
            "roles": roles,
            "iat": now,
            "exp": now + self.access_token_ttl,
            "type": "access"
        }
        
        # Refresh token payload
        refresh_payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + self.refresh_token_ttl,
            "type": "refresh"
        }
        
        return {
            "access_token": jwt.encode(access_payload, self.secret_key, self.algorithm),
            "refresh_token": jwt.encode(refresh_payload, self.secret_key, self.algorithm),
            "expires_in": int(self.access_token_ttl.total_seconds())
        }
    
    def verify_token(self, token: str, token_type: str = "access") -> dict:
        """Verify and decode token"""
        payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        
        if payload.get("type") != token_type:
            raise ValueError(f"Invalid token type: expected {token_type}")
        
        return payload
```

#### 2. Session Manager

```python
class SessionManager:
    def __init__(self, dal: DataAccessLayer, cache):
        self.dal = dal
        self.cache = cache
        self.session_ttl = 3600  # 1 hour
    
    def create_session(self, user_id: str, metadata: dict) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata
        }
        
        # Store in Redis with TTL
        cache_key = f"session:{session_id}"
        self.cache.set(cache_key, session_data, ttl=self.session_ttl)
        
        # Store in database for persistence
        self.dal.create_session({
            "id": session_id,
            "user_id": user_id,
            "expires_at": datetime.utcnow() + timedelta(seconds=self.session_ttl)
        })
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[dict]:
        """Validate and refresh session"""
        cache_key = f"session:{session_id}"
        session = self.cache.get(cache_key)
        
        if session:
            # Refresh TTL
            self.cache.expire(cache_key, self.session_ttl)
            return session
        
        # Check database as fallback
        db_session = self.dal.get_session(session_id)
        if db_session and db_session['expires_at'] > datetime.utcnow():
            # Restore to cache
            self.cache.set(cache_key, db_session, ttl=self.session_ttl)
            return db_session
        
        return None
```

#### 3. RBAC Engine

```python
class RBACEngine:
    def __init__(self, dal: DataAccessLayer):
        self.dal = dal
        self.permission_cache = {}
    
    def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has permission for action on resource"""
        # Get user roles
        user_roles = self.dal.get_user_roles(user_id)
        
        for role in user_roles:
            permissions = role.get('permissions', [])
            
            # Check exact match
            if f"{resource}:{action}" in permissions:
                return True
            
            # Check wildcards
            if f"{resource}:*" in permissions or "*:*" in permissions:
                return True
        
        return False
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for user"""
        cache_key = f"permissions:{user_id}"
        
        # Check cache
        if cache_key in self.permission_cache:
            return self.permission_cache[cache_key]
        
        # Aggregate permissions from all roles
        permissions = set()
        user_roles = self.dal.get_user_roles(user_id)
        
        for role in user_roles:
            permissions.update(role.get('permissions', []))
        
        result = list(permissions)
        self.permission_cache[cache_key] = result
        
        return result
```

#### 4. Security Features

```python
class SecurityManager:
    def __init__(self, cache):
        self.cache = cache
        self.max_attempts = 5
        self.lockout_duration = 900  # 15 minutes
    
    def check_rate_limit(self, identifier: str, limit: int = 10, window: int = 60) -> bool:
        """Check if request is within rate limit"""
        key = f"rate_limit:{identifier}"
        
        current = self.cache.incr(key)
        if current == 1:
            self.cache.expire(key, window)
        
        return current <= limit
    
    def record_failed_attempt(self, user_id: str) -> bool:
        """Record failed login attempt"""
        key = f"failed_attempts:{user_id}"
        attempts = self.cache.incr(key)
        
        if attempts == 1:
            self.cache.expire(key, 3600)  # Reset after 1 hour
        
        if attempts >= self.max_attempts:
            # Lock account
            self.cache.set(f"locked:{user_id}", True, ttl=self.lockout_duration)
            return False
        
        return True
    
    def is_locked(self, user_id: str) -> bool:
        """Check if account is locked"""
        return self.cache.get(f"locked:{user_id}") is not None
```

### Implementation Features

#### 1. Multi-Factor Authentication

```python
import pyotp

class MFAManager:
    def generate_secret(self) -> str:
        """Generate TOTP secret"""
        return pyotp.random_base32()
    
    def generate_qr_code(self, user_email: str, secret: str) -> str:
        """Generate QR code for authenticator app"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name='Lightning System'
        )
        return totp_uri
    
    def verify_totp(self, secret: str, token: str) -> bool:
        """Verify TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
```

#### 2. Password Security

```python
import bcrypt

class PasswordManager:
    def hash_password(self, password: str) -> str:
        """Hash password with bcrypt"""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    def verify_password(self, password: str, hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode(), hash.encode())
    
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
        
        return {
            "strong": len(issues) == 0,
            "issues": issues
        }
```

## Success Metrics

### Technical Metrics

- ✅ 100% session persistence across restarts
- ✅ < 10ms token validation latency
- ✅ Zero unauthorized access incidents
- ✅ Complete audit trail of all auth events
- ✅ 99.99% authentication service availability

### Business Metrics

- ✅ Enhanced security posture
- ✅ Compliance with auth standards (OAuth2, JWT)
- ✅ Reduced security incidents by 90%
- ✅ Complete user activity tracking
- ✅ Support for enterprise SSO

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Token compromise | Low | High | Short TTL, refresh tokens, blacklist |
| Brute force attacks | Medium | Medium | Rate limiting, account lockout |
| Session hijacking | Low | High | Secure cookies, IP validation |
| Database compromise | Low | Critical | Encryption, secure storage |

## Testing Strategy

### Unit Tests

- JWT generation and validation
- Password hashing and verification
- Permission checking logic
- Rate limiting algorithms

### Integration Tests

- End-to-end authentication flow
- Token refresh mechanism
- Session management
- RBAC enforcement

### Security Tests

- Penetration testing
- Token security validation
- SQL injection prevention
- XSS/CSRF protection

## Migration Plan

1. Deploy integrated auth service on port 8106
2. Migrate existing users to new schema
3. Generate API keys for services
4. Enable JWT validation on all endpoints
5. Monitor authentication metrics
6. Decommission old auth service

## Approval

**Review Checklist:**
- [ ] Security measures comprehensive
- [ ] Token management secure
- [ ] Session handling robust
- [ ] RBAC implementation complete
- [ ] Audit logging adequate

**Sign-off Required From:**
- [ ] Security Team
- [ ] DevOps Team
- [ ] Compliance Team
- [ ] Development Team

---

**Next Steps After Approval:**
1. Implement JWT manager
2. Create auth database tables
3. Deploy integrated service
4. Configure service authentication
5. Enable audit logging

**Related Documents:**
- SA-001: Database Persistence Layer
- SA-002: Redis Cache & Event Bus  
- SA-003: Microservices Integration
- SA-004: Workflow Engine Integration
- SA-005: Integration Hub Service
- SA-006: AI Model Service Integration
- Next: SA-008 WebSocket Service Integration