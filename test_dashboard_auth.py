#!/usr/bin/env python3
"""
Test Dashboard Authentication
"""

import requests

def test_auth():
    """Test authentication through API Gateway"""
    print("Testing authentication through API Gateway...")
    
    # Test authentication endpoint
    response = requests.post(
        "http://localhost:8000/auth/token",
        data={
            "username": "admin@example.com",
            "password": "admin"
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=5
    )
    
    if response.status_code == 200:
        token_data = response.json()
        print("✅ Authentication successful!")
        print(f"   Access Token: {token_data['access_token'][:50]}...")
        print(f"   Token Type: {token_data['token_type']}")
        print(f"   Expires In: {token_data['expires_in']} seconds")
        
        # Test using the token to access a protected endpoint
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        
        # Test health endpoint with auth
        health_response = requests.get(
            "http://localhost:8000/agents",
            headers=headers,
            timeout=5
        )
        
        if health_response.status_code == 200:
            print("✅ Token authentication works! Can access protected endpoints.")
        else:
            print(f"❌ Token authentication failed: {health_response.status_code}")
            
    else:
        print(f"❌ Authentication failed: {response.status_code}")
        print(f"   Response: {response.text}")

if __name__ == "__main__":
    test_auth()