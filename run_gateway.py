#!/usr/bin/env python3
"""
Run the API Gateway server
"""
import uvicorn
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting Agent Lightning API Gateway...")
    print("=" * 60)
    
    # Import after path is set
    from api_gateway import APIGateway
    
    # Create gateway instance
    gateway = APIGateway()
    
    # Run server
    uvicorn.run(
        gateway.app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )