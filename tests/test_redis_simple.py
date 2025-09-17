#!/usr/bin/env python3
"""
Simple Redis connection test
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set Redis password in environment
os.environ['REDIS_PASSWORD'] = 'redis_secure_password_123'

from shared.cache import CacheManager

def test_redis_connection():
    """Test basic Redis connection"""
    print("Testing Redis connection...")
    
    try:
        # Create cache manager with password
        cache = CacheManager(password='redis_secure_password_123')
        
        # Test ping
        if cache.health_check():
            print("✅ Redis connection successful!")
            
            # Test basic operations
            cache.set("test:key", "Hello Redis!", ttl=60)
            value = cache.get("test:key")
            
            if value == "Hello Redis!":
                print("✅ Set/Get operations working!")
            else:
                print(f"❌ Get returned: {value}")
            
            # Clean up
            cache.delete("test:key")
            print("✅ Delete operation working!")
            
            # Get Redis info
            info = cache.get_info()
            print(f"\n📊 Redis Info:")
            print(f"  Version: {info.get('redis_version', 'N/A')}")
            print(f"  Memory Used: {info.get('used_memory_human', 'N/A')}")
            print(f"  Connected Clients: {info.get('connected_clients', 0)}")
            
            return True
        else:
            print("❌ Redis health check failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_redis_connection()
    sys.exit(0 if success else 1)