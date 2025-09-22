"""
Enhanced Authentication API with JWT
Provides login, registration, and token management
"""

from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Optional
from jwt_auth import jwt_auth, get_current_user, require_admin
from monitoring.http_metrics_middleware import add_http_metrics_middleware
from shared.database import db_manager
from shared.models import User
import uuid
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Auth API", version="2.0")

# Add HTTP metrics middleware for automatic request/response monitoring
app = add_http_metrics_middleware(app, service_name="auth_api")
app = FastAPI(title="Enhanced Auth API", version="2.0")

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400
    user_info: dict

@app.post("/auth/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    """Register new user"""
    try:
        with db_manager.get_db() as session:
            # Check if user exists
            existing = session.query(User).filter(
                (User.username == request.username) | (User.email == request.email)
            ).first()
            
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username or email already exists"
                )
            
            # Create new user
            user = User(
                id=uuid.uuid4(),
                username=request.username,
                email=request.email,
                password_hash=jwt_auth.hash_password(request.password),
                role=request.role,
                is_active=True
            )
            
            session.add(user)
            session.commit()
            
            # Create token
            user_data = {
                "user_id": str(user.id),
                "username": user.username,
                "role": user.role
            }
            
            token = jwt_auth.create_token(user_data)
            
            return TokenResponse(
                access_token=token,
                user_info=user.to_dict()
            )
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """User login"""
    try:
        with db_manager.get_db() as session:
            user = session.query(User).filter(User.username == request.username).first()
            
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            if not jwt_auth.verify_password(request.password, user.password_hash):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Create token
            user_data = {
                "user_id": str(user.id),
                "username": user.username,
                "role": user.role
            }
            
            token = jwt_auth.create_token(user_data)
            
            return TokenResponse(
                access_token=token,
                user_info=user.to_dict()
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """User logout (blacklist token)"""
    # Note: In a real implementation, you'd get the actual token
    # For now, we'll just return success
    return {"message": "Logged out successfully"}

@app.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {"user": current_user}

@app.get("/auth/users")
async def list_users(current_user: dict = Depends(require_admin)):
    """List all users (admin only)"""
    try:
        with db_manager.get_db() as session:
            users = session.query(User).all()
            return {"users": [user.to_dict() for user in users]}
    except Exception as e:
        logger.error(f"List users error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list users")

@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "service": "enhanced_auth_api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8890)