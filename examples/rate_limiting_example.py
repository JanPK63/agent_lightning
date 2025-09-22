#!/usr/bin/env python3
"""
Rate Limiting Middleware Usage Examples

This file demonstrates how to use the rate limiting middleware
in FastAPI applications and service methods.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, HTTPException

from shared.rate_limiting_middleware import (
    RateLimitingMiddleware,
    RateLimitMiddlewareConfig,
    rate_limited,
    async_rate_limited,
    create_endpoint_rate_limiter,
    setup_rate_limiting
)


# Example 1: Basic FastAPI app with rate limiting middleware
def create_basic_app():
    """Create a basic FastAPI app with rate limiting"""
    app = FastAPI(title="Rate Limited API")

    # Configure rate limiting
    config = RateLimitMiddlewareConfig(
        max_requests=10,  # 10 requests per window
        window_seconds=60,  # 60 second window
        identifier_header="X-API-Key",  # Use API key header
        fallback_to_ip=True,  # Fallback to client IP
        exclude_paths=["/health", "/docs", "/openapi.json"]
    )

    # Add middleware
    app.add_middleware(RateLimitingMiddleware, config=config)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/api/data")
    async def get_data(request: Request):
        return {
            "message": "Data retrieved successfully",
            "client_ip": getattr(request.client, 'host', 'unknown')
        }

    @app.post("/api/data")
    async def create_data(data: dict, request: Request):
        return {
            "message": "Data created successfully",
            "data": data,
            "client_ip": getattr(request.client, 'host', 'unknown')
        }

    return app


# Example 2: Service class with rate limited methods
class DataService:
    """Example service with rate limited methods"""

    @rate_limited(max_requests=5, window_seconds=60)
    def process_user_data(self, user_id: str, data: dict):
        """Process data for a specific user - rate limited per user"""
        print(f"Processing data for user {user_id}")
        return {
            "user_id": user_id,
            "processed_data": data,
            "status": "success"
        }

    @rate_limited(
        max_requests=100,
        window_seconds=300,
        identifier_func=lambda self, **kwargs: "global_limit"
    )
    def get_global_stats(self):
        """Get global statistics - shared limit across all calls"""
        return {
            "total_users": 1000,
            "total_requests": 50000,
            "uptime_hours": 24
        }

    @async_rate_limited(max_requests=20, window_seconds=60)
    async def async_process_data(self, data: dict):
        """Async method with rate limiting"""
        # Simulate async work
        import asyncio
        await asyncio.sleep(0.1)
        return {
            "processed": True,
            "data": data
        }


# Example 3: Advanced configuration with Redis
def create_advanced_app():
    """Create app with advanced rate limiting configuration"""
    app = FastAPI(title="Advanced Rate Limited API")

    # Create rate limiter with Redis (if available)
    try:
        import redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        # Test connection
        redis_client.ping()
        print("Redis connection successful")
    except Exception as e:
        print(f"Redis not available: {e}")
        redis_client = None

    # Create middleware with Redis support
    middleware = create_endpoint_rate_limiter(
        max_requests=50,
        window_seconds=300,  # 5 minutes
        redis_client=redis_client,
        identifier_header="Authorization",
        fallback_to_ip=True
    )

    app.add_middleware(RateLimitingMiddleware, config=middleware.config)

    # Service instance
    data_service = DataService()

    @app.get("/api/stats")
    async def get_stats():
        """Get global statistics"""
        return data_service.get_global_stats()

    @app.post("/api/user/{user_id}/data")
    async def process_user_data(user_id: str, data: dict):
        """Process user data with per-user rate limiting"""
        try:
            result = data_service.process_user_data(user_id, data)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/async/process")
    async def async_process(data: dict):
        """Async data processing"""
        try:
            result = await data_service.async_process_data(data)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Example 4: Using the setup utility function
def create_app_with_setup_utility():
    """Create app using the setup_rate_limiting utility"""
    app = FastAPI(title="Rate Limited API (Setup Utility)")

    # Configure rate limiting
    config = RateLimitMiddlewareConfig(
        max_requests=25,
        window_seconds=120,
        identifier_header="X-API-Key"
    )

    # Setup rate limiting with exception handler
    setup_rate_limiting(app, config, exception_handler=True)

    @app.get("/api/test")
    async def test_endpoint():
        return {"message": "Rate limiting is working!"}

    return app


# Main execution
if __name__ == "__main__":
    print("Rate Limiting Middleware Examples")
    print("=" * 40)

    # Test basic functionality
    print("\n1. Testing basic middleware creation...")
    app = create_basic_app()
    print("✓ Basic app created successfully")

    print("\n2. Testing service methods...")
    service = DataService()

    # Test rate limited method
    try:
        for i in range(3):
            result = service.process_user_data("user123", {"test": "data"})
            print(f"✓ Call {i+1} successful: {result['status']}")
    except Exception as e:
        print(f"Rate limit exceeded: {e}")

    # Test global stats method
    stats = service.get_global_stats()
    print(f"✓ Global stats: {stats['total_users']} users")

    print("\n3. Testing advanced configuration...")
    try:
        advanced_app = create_advanced_app()
        print("✓ Advanced app created successfully")
    except Exception as e:
        print(f"Advanced app creation failed: {e}")

    print("\n4. Testing setup utility...")
    utility_app = create_app_with_setup_utility()
    print("✓ Setup utility app created successfully")

    print("\n" + "=" * 40)
    print("All examples completed successfully!")
    print("\nTo run a FastAPI server, use:")
    print("uvicorn examples.rate_limiting_example:create_basic_app --reload")