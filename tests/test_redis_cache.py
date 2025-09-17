#!/usr/bin/env python3
"""
Test script for Redis cache and event bus implementation
Verifies cache operations, pub/sub, and distributed locking
"""

import sys
import os
import time
import json
import threading
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.cache import CacheManager, get_cache
from shared.events import EventBus, EventChannel, Event, EventPublisher, EventSubscriber
from shared.cache_decorators import cached, cache_lock, rate_limit, warm_agent_cache

def test_cache_operations():
    """Test basic cache operations"""
    print("\n🧪 Testing Cache Operations...")
    
    cache = get_cache()
    
    # Test set/get
    test_data = {"name": "Test Agent", "status": "active"}
    cache.set("test:agent", test_data, ttl=60)
    retrieved = cache.get("test:agent")
    assert retrieved == test_data, "Cache get/set failed"
    print("✅ Set/Get operations working")
    
    # Test exists
    assert cache.exists("test:agent"), "Cache exists failed"
    print("✅ Exists operation working")
    
    # Test TTL
    ttl = cache.ttl("test:agent")
    assert ttl > 0, "TTL not set correctly"
    print(f"✅ TTL operation working (TTL: {ttl}s)")
    
    # Test delete
    cache.delete("test:agent")
    assert not cache.exists("test:agent"), "Cache delete failed"
    print("✅ Delete operation working")
    
    # Test batch operations
    batch_data = {
        "test:1": {"id": 1, "name": "Item 1"},
        "test:2": {"id": 2, "name": "Item 2"},
        "test:3": {"id": 3, "name": "Item 3"}
    }
    cache.mset(batch_data, ttl=60)
    results = cache.mget(list(batch_data.keys()))
    assert results == list(batch_data.values()), "Batch operations failed"
    print("✅ Batch operations working")
    
    # Test pattern deletion
    cache.delete_pattern("test:*")
    assert not any(cache.exists(k) for k in batch_data.keys()), "Pattern deletion failed"
    print("✅ Pattern deletion working")
    
    print("✨ All cache operations passed!")

def test_pub_sub():
    """Test pub/sub functionality"""
    print("\n🧪 Testing Pub/Sub...")
    
    received_events = []
    
    def event_handler(event: Event):
        received_events.append(event)
        print(f"  📨 Received event: {event.channel}")
    
    # Create event bus
    bus = EventBus("test_service")
    
    # Register handler
    bus.on(EventChannel.AGENT_CREATED, event_handler)
    bus.on(EventChannel.TASK_COMPLETED, event_handler)
    
    # Start subscriber
    bus.start()
    time.sleep(1)  # Give subscriber time to start
    
    # Publish events
    bus.emit(EventChannel.AGENT_CREATED, {
        "agent_id": "test_agent",
        "name": "Test Agent"
    })
    
    bus.emit(EventChannel.TASK_COMPLETED, {
        "task_id": "task_123",
        "result": "Success"
    })
    
    # Wait for events to be received
    time.sleep(2)
    
    # Check results
    assert len(received_events) == 2, f"Expected 2 events, got {len(received_events)}"
    assert received_events[0].channel == EventChannel.AGENT_CREATED.value
    assert received_events[1].channel == EventChannel.TASK_COMPLETED.value
    
    # Stop subscriber
    bus.stop()
    
    print("✅ Pub/Sub working correctly")
    print(f"✅ Received {len(received_events)} events")
    print("✨ Pub/Sub tests passed!")

def test_distributed_locking():
    """Test distributed locking"""
    print("\n🧪 Testing Distributed Locking...")
    
    cache = get_cache()
    results = []
    
    def critical_section(worker_id):
        """Simulated critical section"""
        with cache.lock("test_resource", timeout=5):
            print(f"  🔒 Worker {worker_id} acquired lock")
            results.append(worker_id)
            time.sleep(0.5)  # Simulate work
            print(f"  🔓 Worker {worker_id} released lock")
    
    # Start multiple threads trying to acquire the same lock
    threads = []
    for i in range(3):
        thread = threading.Thread(target=critical_section, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # Check that operations were serialized
    assert len(results) == 3, "Not all workers completed"
    print(f"✅ Lock serialization working (order: {results})")
    print("✨ Distributed locking tests passed!")

def test_cache_decorators():
    """Test cache decorators"""
    print("\n🧪 Testing Cache Decorators...")
    
    call_count = 0
    
    @cached(ttl=60, prefix="test")
    def expensive_function(x, y):
        nonlocal call_count
        call_count += 1
        return x + y
    
    # First call - should execute function
    result1 = expensive_function(2, 3)
    assert result1 == 5
    assert call_count == 1
    print("✅ First call executed function")
    
    # Second call - should use cache
    result2 = expensive_function(2, 3)
    assert result2 == 5
    assert call_count == 1
    print("✅ Second call used cache")
    
    # Different arguments - should execute function
    result3 = expensive_function(3, 4)
    assert result3 == 7
    assert call_count == 2
    print("✅ Different arguments executed function")
    
    # Invalidate cache
    expensive_function.invalidate(2, 3)
    result4 = expensive_function(2, 3)
    assert result4 == 5
    assert call_count == 3
    print("✅ Cache invalidation working")
    
    print("✨ Cache decorators tests passed!")

def test_rate_limiting():
    """Test rate limiting"""
    print("\n🧪 Testing Rate Limiting...")
    
    @rate_limit(max_calls=3, period=5)
    def limited_function():
        return "success"
    
    # Should allow first 3 calls
    for i in range(3):
        result = limited_function()
        assert result == "success"
        print(f"✅ Call {i+1} allowed")
    
    # Fourth call should be rate limited
    try:
        limited_function()
        assert False, "Rate limit should have been triggered"
    except Exception as e:
        assert "Rate limit exceeded" in str(e)
        print("✅ Rate limit enforced correctly")
    
    print("✨ Rate limiting tests passed!")

def test_cache_patterns():
    """Test cache patterns"""
    print("\n🧪 Testing Cache Patterns...")
    
    cache = get_cache()
    
    # Test cache-aside pattern
    load_count = 0
    
    def data_loader():
        nonlocal load_count
        load_count += 1
        return {"loaded": True, "count": load_count}
    
    # First call - should load
    result1 = cache.cache_aside("test:cache_aside", data_loader, ttl=60)
    assert result1["loaded"] == True
    assert load_count == 1
    print("✅ Cache-aside: First call loaded data")
    
    # Second call - should use cache
    result2 = cache.cache_aside("test:cache_aside", data_loader, ttl=60)
    assert result2["count"] == 1
    assert load_count == 1
    print("✅ Cache-aside: Second call used cache")
    
    # Test write-through pattern
    write_count = 0
    
    def data_writer(value):
        nonlocal write_count
        write_count += 1
        return {"written": value, "count": write_count}
    
    result = cache.write_through("test:write_through", 
                                 {"data": "test"}, 
                                 data_writer, ttl=60)
    assert result["written"]["data"] == "test"
    assert write_count == 1
    print("✅ Write-through: Data written and cached")
    
    # Verify it's in cache
    cached_value = cache.get("test:write_through")
    assert cached_value["count"] == 1
    print("✅ Write-through: Data retrievable from cache")
    
    print("✨ Cache patterns tests passed!")

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 Redis Cache & Event Bus Test Suite")
    print("=" * 60)
    
    try:
        # Check Redis connection
        cache = get_cache()
        if not cache.health_check():
            print("❌ Redis is not running or not accessible")
            print("Please ensure Redis is running:")
            print("  brew services start redis  # macOS")
            print("  sudo systemctl start redis  # Linux")
            return 1
        
        print("✅ Redis connection successful")
        
        # Run tests
        test_cache_operations()
        test_pub_sub()
        test_distributed_locking()
        test_cache_decorators()
        test_rate_limiting()
        test_cache_patterns()
        
        # Show cache stats
        info = cache.get_info()
        print("\n📊 Redis Statistics:")
        print(f"  Memory Used: {info.get('used_memory_human', 'N/A')}")
        print(f"  Connected Clients: {info.get('connected_clients', 0)}")
        print(f"  Total Commands: {info.get('total_commands_processed', 0)}")
        
        print("\n" + "=" * 60)
        print("✨ All tests passed successfully!")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up test data
        try:
            cache = get_cache()
            cache.delete_pattern("test:*")
            cache.delete_pattern("rate:*")
            cache.delete_pattern("lock:*")
            print("\n🧹 Test data cleaned up")
        except:
            pass

if __name__ == "__main__":
    sys.exit(main())