#!/usr/bin/env python3
"""
Test script for event bus backend failover between Redis and RabbitMQ
Demonstrates switching between event bus backends dynamically
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.events import EventBus, EventChannel

def test_backend_switching():
    """Test switching between Redis and RabbitMQ backends"""
    print("🧪 Testing Event Bus Backend Switching...")

    received_events = []

    def event_handler(event):
        received_events.append(event)
        print(f"  📨 Received event on {event.service} backend: {event.channel}")

    # Test Redis backend
    print("\n🔴 Testing Redis Backend...")
    try:
        bus_redis = EventBus("test_service", backend="redis")

        # Register handler
        bus_redis.on(EventChannel.AGENT_CREATED, event_handler)
        bus_redis.on(EventChannel.TASK_COMPLETED, event_handler)

        # Start subscriber
        bus_redis.start()
        time.sleep(1)

        # Publish events
        bus_redis.emit(EventChannel.AGENT_CREATED, {"agent_id": "redis_test", "backend": "redis"})
        bus_redis.emit(EventChannel.TASK_COMPLETED, {"task_id": "redis_task", "backend": "redis"})

        # Wait for events
        time.sleep(2)

        # Check results
        redis_events = [e for e in received_events if e.data.get('backend') == 'redis']
        assert len(redis_events) == 2, f"Expected 2 Redis events, got {len(redis_events)}"
        print("✅ Redis backend working correctly")

        # Stop Redis bus
        bus_redis.stop()

    except Exception as e:
        print(f"❌ Redis backend test failed: {e}")
        print("💡 Ensure Redis is running on localhost:6379")

    # Clear received events
    received_events.clear()

    # Test RabbitMQ backend
    print("\n🐰 Testing RabbitMQ Backend...")
    try:
        import pika
    except ImportError:
        print("❌ RabbitMQ dependencies not available, skipping RabbitMQ tests")
        return

    try:
        bus_rabbitmq = EventBus("test_service", backend="rabbitmq")

        # Register handler
        bus_rabbitmq.on(EventChannel.AGENT_CREATED, event_handler)
        bus_rabbitmq.on(EventChannel.TASK_COMPLETED, event_handler)

        # Start subscriber
        bus_rabbitmq.start()
        time.sleep(2)

        # Publish events
        bus_rabbitmq.emit(EventChannel.AGENT_CREATED, {"agent_id": "rabbitmq_test", "backend": "rabbitmq"})
        bus_rabbitmq.emit(EventChannel.TASK_COMPLETED, {"task_id": "rabbitmq_task", "backend": "rabbitmq"})

        # Wait for events
        time.sleep(3)

        # Check results
        rabbitmq_events = [e for e in received_events if e.data.get('backend') == 'rabbitmq']
        assert len(rabbitmq_events) == 2, f"Expected 2 RabbitMQ events, got {len(rabbitmq_events)}"
        print("✅ RabbitMQ backend working correctly")

        # Stop RabbitMQ bus
        bus_rabbitmq.stop()

    except Exception as e:
        print(f"❌ RabbitMQ backend test failed: {e}")
        print("💡 Ensure RabbitMQ is running on localhost:5672")

    print("\n✨ Backend switching tests completed!")

def test_event_history_redis_only():
    """Test that event history only works with Redis"""
    print("\n📚 Testing Event History (Redis-only feature)...")

    try:
        # Test with Redis
        bus_redis = EventBus("test_service", backend="redis")
        bus_redis.emit(EventChannel.AGENT_CREATED, {"test": "history_redis"})

        # Get event history using the proper function
        from shared.events import get_event_history
        history = get_event_history(EventChannel.AGENT_CREATED, limit=5)
        if history and len(history) > 0:
            print(f"✅ Event history available with Redis backend ({len(history)} events)")
            print(f"   Latest event: {history[0].data}")
        else:
            print("⚠️ Event history not available")

    except Exception as e:
        print(f"❌ Redis event history test failed: {e}")

    # Test with RabbitMQ (should not have history)
    try:
        import pika
        from shared.events import get_event_history
        # Switch to RabbitMQ backend temporarily
        import os
        old_backend = os.environ.get('EVENT_BUS_BACKEND')
        os.environ['EVENT_BUS_BACKEND'] = 'rabbitmq'

        try:
            history = get_event_history(EventChannel.AGENT_CREATED, limit=5)
            if len(history) == 0:
                print("✅ RabbitMQ correctly doesn't support event history")
            else:
                print(f"⚠️ RabbitMQ unexpectedly returned history: {len(history)} events")
        finally:
            # Restore original backend
            if old_backend:
                os.environ['EVENT_BUS_BACKEND'] = old_backend
            elif 'EVENT_BUS_BACKEND' in os.environ:
                del os.environ['EVENT_BUS_BACKEND']

    except ImportError:
        print("ℹ️ RabbitMQ not available for history test")
    except Exception as e:
        print(f"❌ RabbitMQ history test failed: {e}")

def main():
    """Run all failover tests"""
    print("=" * 60)
    print("🚀 Event Bus Backend Failover Test Suite")
    print("=" * 60)

    try:
        test_backend_switching()
        test_event_history_redis_only()

        print("\n" + "=" * 60)
        print("✨ All failover tests completed!")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())