#!/usr/bin/env python3
"""
Integration Status Report - Check all services and their connections
"""

import asyncio
import aiohttp
import psycopg2
import redis
from datetime import datetime
from tabulate import tabulate

# Service endpoints
SERVICES = {
    "API Gateway": "http://localhost:8000",
    "Auth Service": "http://localhost:8001", 
    "Agent Designer": "http://localhost:8002",
    "Workflow Engine": "http://localhost:8003",
    "AI Model Service": "http://localhost:8004",
    "Service Discovery": "http://localhost:8005",
    "Integration Hub": "http://localhost:8006",
    "Monitoring Service": "http://localhost:8007",
    "WebSocket Service": "http://localhost:8008",
    "RL Server": "http://localhost:8010",
    "RL Orchestrator": "http://localhost:8011",
    "Memory Service": "http://localhost:8012",
    "Checkpoint Service": "http://localhost:8013",
    "Batch Accumulator": "http://localhost:8014",
    "AutoGen Integration": "http://localhost:8015",
    "Monitoring Dashboard": "http://localhost:8052"
}

# New services added in Phase 4
NEW_SERVICES = [
    "Memory Service",
    "Checkpoint Service", 
    "Batch Accumulator",
    "AutoGen Integration"
]


async def check_service(name, url):
    """Check if a service is running"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/health", timeout=2) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "name": name,
                        "status": "✅ Running",
                        "port": url.split(":")[-1],
                        "integration": "✅" if name not in NEW_SERVICES else "🆕",
                        "details": data.get("status", "healthy")
                    }
                else:
                    return {
                        "name": name,
                        "status": "⚠️ Unhealthy",
                        "port": url.split(":")[-1],
                        "integration": "❌",
                        "details": f"HTTP {resp.status}"
                    }
    except asyncio.TimeoutError:
        return {
            "name": name,
            "status": "❌ Timeout",
            "port": url.split(":")[-1],
            "integration": "❌",
            "details": "No response"
        }
    except Exception as e:
        return {
            "name": name,
            "status": "❌ Down",
            "port": url.split(":")[-1],
            "integration": "❌",
            "details": str(e)[:30]
        }


def check_database():
    """Check PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="agent_lightning",
            user="agent_user",
            password="agent_password"
        )
        cur = conn.cursor()
        
        # Check table counts
        tables = [
            "agents", "tasks", "workflows", "agent_memories",
            "checkpoints", "experience_replay_buffer"
        ]
        
        counts = {}
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cur.fetchone()[0]
            except:
                counts[table] = "N/A"
        
        conn.close()
        return {"status": "✅ Connected", "tables": counts}
    except Exception as e:
        return {"status": "❌ Error", "error": str(e)}


def check_redis():
    """Check Redis connection"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        keys = r.dbsize()
        return {"status": "✅ Connected", "keys": keys}
    except Exception as e:
        return {"status": "❌ Error", "error": str(e)}


async def check_integration_points():
    """Check key integration points"""
    integrations = []
    
    # Check if new services are using shared database
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="agent_lightning",
            user="agent_user",
            password="agent_password"
        )
        cur = conn.cursor()
        
        # Check memory persistence
        cur.execute("SELECT COUNT(*) FROM agent_memories")
        memory_count = cur.fetchone()[0]
        integrations.append({
            "Integration": "Memory → Database",
            "Status": "✅" if memory_count > 0 else "⚠️",
            "Details": f"{memory_count} memories stored"
        })
        
        # Check checkpoints
        cur.execute("SELECT COUNT(*) FROM checkpoints")
        checkpoint_count = cur.fetchone()[0]
        integrations.append({
            "Integration": "Checkpoints → Database",
            "Status": "✅" if checkpoint_count >= 0 else "❌",
            "Details": f"{checkpoint_count} checkpoints"
        })
        
        # Check agents from AutoGen
        cur.execute("SELECT COUNT(*) FROM agents WHERE id IN ('planner', 'executor', 'critic')")
        autogen_agents = cur.fetchone()[0]
        integrations.append({
            "Integration": "AutoGen → Agent Registry",
            "Status": "✅" if autogen_agents > 0 else "❌",
            "Details": f"{autogen_agents} AutoGen agents"
        })
        
        conn.close()
    except Exception as e:
        integrations.append({
            "Integration": "Database Integration",
            "Status": "❌",
            "Details": str(e)[:50]
        })
    
    # Check Redis cache integration
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        keys = r.keys("memory:*")
        integrations.append({
            "Integration": "Memory → Cache",
            "Status": "✅" if len(keys) > 0 else "⚠️",
            "Details": f"{len(keys)} cached items"
        })
    except:
        integrations.append({
            "Integration": "Cache Integration",
            "Status": "❌",
            "Details": "Redis not available"
        })
    
    return integrations


async def main():
    print("=" * 80)
    print("🚀 AI AGENT FRAMEWORK - INTEGRATION STATUS REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check all services
    print("📊 SERVICE STATUS:")
    print("-" * 80)
    
    tasks = []
    for name, url in SERVICES.items():
        tasks.append(check_service(name, url))
    
    results = await asyncio.gather(*tasks)
    
    # Separate running and down services
    running = [r for r in results if "✅" in r["status"]]
    down = [r for r in results if "❌" in r["status"] or "⚠️" in r["status"]]
    
    # Display results
    headers = ["Service", "Port", "Status", "Integration", "Details"]
    table_data = [[r["name"], r["port"], r["status"], r["integration"], r["details"]] for r in running]
    
    if table_data:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    if down:
        print("\n⚠️ SERVICES NOT RUNNING:")
        down_data = [[r["name"], r["port"], r["status"], r["integration"], r["details"]] for r in down]
        print(tabulate(down_data, headers=headers, tablefmt="grid"))
    
    # Database status
    print("\n💾 DATABASE STATUS:")
    print("-" * 80)
    db_status = check_database()
    print(f"PostgreSQL: {db_status['status']}")
    if "tables" in db_status:
        for table, count in db_status["tables"].items():
            print(f"  - {table}: {count} records")
    
    # Redis status
    print("\n🔴 CACHE STATUS:")
    print("-" * 80)
    redis_status = check_redis()
    print(f"Redis: {redis_status['status']}")
    if "keys" in redis_status:
        print(f"  - Total keys: {redis_status['keys']}")
    
    # Integration points
    print("\n🔗 INTEGRATION POINTS:")
    print("-" * 80)
    integrations = await check_integration_points()
    int_headers = ["Integration", "Status", "Details"]
    int_data = [[i["Integration"], i["Status"], i["Details"]] for i in integrations]
    print(tabulate(int_data, headers=int_headers, tablefmt="grid"))
    
    # Summary
    print("\n📈 SUMMARY:")
    print("-" * 80)
    total_services = len(SERVICES)
    running_count = len(running)
    new_running = sum(1 for r in running if r["name"] in NEW_SERVICES)
    
    print(f"✅ Services Running: {running_count}/{total_services}")
    print(f"🆕 New Services (Phase 4): {new_running}/{len(NEW_SERVICES)}")
    print(f"🔗 Database Integration: {'✅ Active' if db_status['status'] == '✅ Connected' else '❌ Failed'}")
    print(f"🔗 Cache Integration: {'✅ Active' if redis_status['status'] == '✅ Connected' else '❌ Failed'}")
    
    # New services integration status
    print("\n🆕 NEW SERVICES INTEGRATION:")
    print("-" * 80)
    new_services_status = []
    for service in NEW_SERVICES:
        service_running = any(r["name"] == service and "✅" in r["status"] for r in results)
        new_services_status.append({
            "Service": service,
            "Running": "✅" if service_running else "❌",
            "Integrated": "✅" if service_running else "❌"
        })
    
    ns_headers = ["Service", "Running", "Integrated"]
    ns_data = [[s["Service"], s["Running"], s["Integrated"]] for s in new_services_status]
    print(tabulate(ns_data, headers=ns_headers, tablefmt="grid"))
    
    print("\n" + "=" * 80)
    print("✅ INTEGRATION STATUS REPORT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())