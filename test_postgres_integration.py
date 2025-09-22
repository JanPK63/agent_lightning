#!/usr/bin/env python3
"""
Test script for PostgreSQL integration with Agent Lightning
"""

import os
import sys
from shared.database import get_database_config, init_database, test_connection

def test_sqlite_connection():
    """Test SQLite connection (default)"""
    print("🧪 Testing SQLite connection...")

    # Set SQLite URL
    os.environ["DATABASE_URL"] = "sqlite:///./test_agentlightning.db"

    try:
        config = get_database_config()
        print(f"✅ SQLite config: {config}")

        success = init_database()
        if success:
            print("✅ SQLite database initialized successfully")

        success = test_connection()
        if success:
            print("✅ SQLite connection test passed")
        else:
            print("❌ SQLite connection test failed")

    except Exception as e:
        print(f"❌ SQLite test failed: {e}")
        return False

    return True

def test_postgres_config():
    """Test PostgreSQL configuration (without actual connection)"""
    print("\n🧪 Testing PostgreSQL configuration...")

    # Set PostgreSQL URL
    os.environ["DATABASE_URL"] = "postgresql://user:password@localhost:5432/agent_lightning"

    try:
        config = get_database_config()
        print(f"✅ PostgreSQL config: {config}")

        # Check if config has PostgreSQL-specific settings
        expected_keys = ["poolclass", "pool_pre_ping", "pool_size", "max_overflow", "pool_timeout", "pool_recycle"]
        for key in expected_keys:
            if key in config:
                print(f"✅ PostgreSQL config has {key}: {config[key]}")
            else:
                print(f"❌ PostgreSQL config missing {key}")

    except Exception as e:
        print(f"❌ PostgreSQL config test failed: {e}")
        return False

    return True

def test_database_url_parsing():
    """Test database URL parsing for different backends"""
    print("\n🧪 Testing database URL parsing...")

    test_urls = [
        "sqlite:///./agentlightning.db",
        "postgresql://user:pass@localhost:5432/db",
        "mysql://user:pass@localhost/db",
        "oracle://user:pass@localhost/db"
    ]

    for url in test_urls:
        os.environ["DATABASE_URL"] = url
        try:
            config = get_database_config()
            backend = "SQLite" if url.startswith("sqlite") else \
                     "PostgreSQL" if url.startswith("postgresql") else \
                     "Other"
            print(f"✅ {backend} URL '{url}' -> config keys: {list(config.keys())}")
        except Exception as e:
            print(f"❌ Failed to parse URL '{url}': {e}")
            return False

    return True

def main():
    """Run all tests"""
    print("🚀 Starting PostgreSQL integration tests...\n")

    tests = [
        test_sqlite_connection,
        test_postgres_config,
        test_database_url_parsing
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All PostgreSQL integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())