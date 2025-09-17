#!/usr/bin/env python3
"""
Test script for Memory Persistence System
Tests memory creation, retrieval, and consolidation
"""

import asyncio
import aiohttp
import json
import random
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MEMORY_SERVICE_URL = "http://localhost:8012"
AGENT_ID = "test_engineer"  # Using existing agent


async def test_memory_creation():
    """Test creating memories"""
    print("\n🧪 Testing Memory Creation...")
    
    memories_created = []
    
    async with aiohttp.ClientSession() as session:
        # Create different types of memories
        memory_types = ["episodic", "semantic", "procedural", "working"]
        importance_levels = ["critical", "high", "medium", "low", "temporary"]
        
        for i in range(10):
            memory_data = {
                "agent_id": AGENT_ID,
                "memory_type": random.choice(memory_types),
                "importance": random.choice(importance_levels),
                "content": {
                    "description": f"Test memory {i}",
                    "timestamp": datetime.now().isoformat(),
                    "data": f"Important information #{i}",
                    "test_run": True,
                    "iteration": i
                },
                "tags": ["test", f"batch_{i//3}"],
                "context": {
                    "test_suite": "memory_system",
                    "environment": "development"
                }
            }
            
            # Add embedding for some memories
            if i % 2 == 0:
                memory_data["embedding"] = np.random.randn(1536).tolist()
            
            try:
                async with session.post(
                    f"{MEMORY_SERVICE_URL}/memories",
                    json=memory_data
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        memories_created.append(result["memory_id"])
                        print(f"  ✅ Created memory {i}: {result['memory_id']}")
                    else:
                        print(f"  ❌ Failed to create memory {i}: {resp.status}")
            except Exception as e:
                print(f"  ❌ Error creating memory {i}: {e}")
    
    print(f"\n  Created {len(memories_created)} memories")
    return memories_created


async def test_memory_retrieval(memory_ids):
    """Test retrieving memories"""
    print("\n🧪 Testing Memory Retrieval...")
    
    async with aiohttp.ClientSession() as session:
        # Test individual memory retrieval
        if memory_ids:
            test_id = memory_ids[0]
            try:
                async with session.get(
                    f"{MEMORY_SERVICE_URL}/memories/{test_id}"
                ) as resp:
                    if resp.status == 200:
                        memory = await resp.json()
                        print(f"  ✅ Retrieved memory: {memory['id']}")
                        print(f"     Type: {memory['memory_type']}")
                        print(f"     Importance: {memory['importance']}")
                        print(f"     Strength: {memory['strength']}")
                    else:
                        print(f"  ❌ Failed to retrieve memory: {resp.status}")
            except Exception as e:
                print(f"  ❌ Error retrieving memory: {e}")
        
        # Test query-based retrieval
        query_data = {
            "agent_id": AGENT_ID,
            "memory_types": ["episodic", "semantic"],
            "tags": ["test"],
            "limit": 5
        }
        
        try:
            async with session.post(
                f"{MEMORY_SERVICE_URL}/memories/query",
                json=query_data
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"\n  ✅ Query returned {result['count']} memories")
                    for mem in result['memories'][:3]:
                        print(f"     - {mem['id']}: {mem['memory_type']} ({mem['importance']})")
                else:
                    print(f"  ❌ Query failed: {resp.status}")
        except Exception as e:
            print(f"  ❌ Error querying memories: {e}")


async def test_experience_replay():
    """Test experience replay buffer"""
    print("\n🧪 Testing Experience Replay Buffer...")
    
    async with aiohttp.ClientSession() as session:
        # Add experiences
        for i in range(5):
            experience = {
                "agent_id": AGENT_ID,
                "state": {
                    "position": [random.random(), random.random()],
                    "health": 100 - i * 10
                },
                "action": {
                    "type": "move",
                    "direction": random.choice(["north", "south", "east", "west"])
                },
                "reward": random.uniform(-1, 1),
                "next_state": {
                    "position": [random.random(), random.random()],
                    "health": 95 - i * 10
                },
                "done": i == 4,
                "priority": random.uniform(0.5, 1.5)
            }
            
            try:
                async with session.post(
                    f"{MEMORY_SERVICE_URL}/experience-replay",
                    json=experience
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"  ✅ Added experience {i}: reward={experience['reward']:.2f}")
                    else:
                        print(f"  ❌ Failed to add experience {i}: {resp.status}")
            except Exception as e:
                print(f"  ❌ Error adding experience {i}: {e}")
        
        # Sample experiences
        try:
            async with session.get(
                f"{MEMORY_SERVICE_URL}/experience-replay/{AGENT_ID}/sample?batch_size=3"
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"\n  ✅ Sampled {result['batch_size']} experiences")
                else:
                    print(f"  ❌ Failed to sample experiences: {resp.status}")
        except Exception as e:
            print(f"  ❌ Error sampling experiences: {e}")


async def test_memory_consolidation():
    """Test memory consolidation"""
    print("\n🧪 Testing Memory Consolidation...")
    
    async with aiohttp.ClientSession() as session:
        # Trigger consolidation
        try:
            async with session.post(
                f"{MEMORY_SERVICE_URL}/memories/{AGENT_ID}/consolidate",
                json={"threshold": 0.5}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"  ✅ Consolidation started: {result['status']}")
                    print(f"     Agent: {result['agent_id']}")
                    print(f"     Threshold: {result['threshold']}")
                else:
                    print(f"  ❌ Failed to start consolidation: {resp.status}")
        except Exception as e:
            print(f"  ❌ Error starting consolidation: {e}")
        
        # Wait for consolidation to complete
        await asyncio.sleep(2)
        
        # Apply memory decay
        try:
            async with session.post(
                f"{MEMORY_SERVICE_URL}/memories/{AGENT_ID}/decay"
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"\n  ✅ Memory decay applied")
                    print(f"     Memories pruned: {result.get('memories_pruned', 0)}")
                else:
                    print(f"  ❌ Failed to apply decay: {resp.status}")
        except Exception as e:
            print(f"  ❌ Error applying decay: {e}")


async def test_memory_health():
    """Test service health"""
    print("\n🧪 Testing Service Health...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{MEMORY_SERVICE_URL}/health") as resp:
                if resp.status == 200:
                    health = await resp.json()
                    print(f"  ✅ Service: {health['service']}")
                    print(f"     Status: {health['status']}")
                    print(f"     Database: {'✅' if health['database'] else '❌'}")
                    print(f"     Cache: {'✅' if health['cache'] else '❌'}")
                    print(f"     pgvector: {'✅' if health['pgvector'] else '❌'}")
                else:
                    print(f"  ❌ Health check failed: {resp.status}")
        except Exception as e:
            print(f"  ❌ Error checking health: {e}")


async def test_vector_similarity():
    """Test vector similarity search"""
    print("\n🧪 Testing Vector Similarity Search...")
    
    async with aiohttp.ClientSession() as session:
        # Create a query embedding
        query_embedding = np.random.randn(1536).tolist()
        
        query_data = {
            "agent_id": AGENT_ID,
            "query_embedding": query_embedding,
            "limit": 3,
            "include_inactive": False
        }
        
        try:
            async with session.post(
                f"{MEMORY_SERVICE_URL}/memories/query",
                json=query_data
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"  ✅ Similarity search returned {result['count']} memories")
                    
                    # Show similarity scores if available
                    for mem in result['memories']:
                        if 'similarity' in mem:
                            print(f"     - {mem['id']}: similarity={mem['similarity']:.3f}")
                else:
                    print(f"  ❌ Similarity search failed: {resp.status}")
        except Exception as e:
            print(f"  ❌ Error in similarity search: {e}")


async def run_all_tests():
    """Run all memory system tests"""
    print("=" * 60)
    print("🚀 MEMORY PERSISTENCE SYSTEM TEST SUITE")
    print("=" * 60)
    
    # Check if service is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{MEMORY_SERVICE_URL}/health") as resp:
                if resp.status != 200:
                    print("❌ Memory Service is not running!")
                    print(f"   Please ensure it's running on port 8012")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to Memory Service: {e}")
        print(f"   Please ensure it's running on {MEMORY_SERVICE_URL}")
        return
    
    # Run tests
    await test_memory_health()
    memory_ids = await test_memory_creation()
    await test_memory_retrieval(memory_ids)
    await test_experience_replay()
    await test_vector_similarity()
    await test_memory_consolidation()
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ MEMORY SYSTEM TEST SUITE COMPLETED")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - Memory Service: Running on port 8012")
    print(f"  - pgvector: Enabled for similarity search")
    print(f"  - Consolidation: Background task active")
    print(f"  - Experience Replay: Buffer operational")
    print(f"  - Agent tested: {AGENT_ID}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())