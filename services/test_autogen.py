#!/usr/bin/env python3
"""
Test script for AutoGen Integration
"""

import asyncio
import aiohttp
import json
from datetime import datetime

AUTOGEN_SERVICE_URL = "http://localhost:8015"


async def test_health():
    """Test service health"""
    print("\n🧪 Testing AutoGen Service Health...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{AUTOGEN_SERVICE_URL}/health") as resp:
                if resp.status == 200:
                    health = await resp.json()
                    print(f"  ✅ Service: {health['service']}")
                    print(f"     Status: {health['status']}")
                    print(f"     Agents: {health['agents']}")
                    print(f"     Active Conversations: {health['active_conversations']}")
                    return True
                else:
                    print(f"  ❌ Health check failed: {resp.status}")
                    return False
        except Exception as e:
            print(f"  ❌ Error checking health: {e}")
            return False


async def test_list_agents():
    """Test listing agents"""
    print("\n🧪 Testing Agent Listing...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{AUTOGEN_SERVICE_URL}/agents") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"  ✅ Found {len(data['agents'])} agents:")
                    for agent in data['agents']:
                        print(f"     - {agent['name']} ({agent['role']})")
                        print(f"       {agent['system_message']}")
                else:
                    print(f"  ❌ Failed to list agents: {resp.status}")
        except Exception as e:
            print(f"  ❌ Error listing agents: {e}")


async def test_create_agent():
    """Test creating a new agent"""
    print("\n🧪 Testing Agent Creation...")
    
    async with aiohttp.ClientSession() as session:
        agent_data = {
            "name": "test_researcher",
            "role": "researcher",
            "system_message": "You are a research specialist. Find and analyze information.",
            "model": "gpt-4"
        }
        
        try:
            async with session.post(
                f"{AUTOGEN_SERVICE_URL}/agents/create",
                params=agent_data
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"  ✅ Created agent: {result['agent_name']}")
                    print(f"     Role: {result['role']}")
                    print(f"     Status: {result['status']}")
                else:
                    print(f"  ❌ Failed to create agent: {resp.status}")
                    text = await resp.text()
                    print(f"     Error: {text}")
        except Exception as e:
            print(f"  ❌ Error creating agent: {e}")


async def test_two_agent_conversation():
    """Test two-agent conversation"""
    print("\n🧪 Testing Two-Agent Conversation...")
    
    async with aiohttp.ClientSession() as session:
        conv_data = {
            "conversation_type": "two_agent",
            "agent_names": ["planner", "executor"],
            "initial_message": "Create a plan to build a simple todo app",
            "max_round": 5
        }
        
        try:
            async with session.post(
                f"{AUTOGEN_SERVICE_URL}/conversations/start",
                json=conv_data
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"  ✅ Started conversation: {result['conversation_id']}")
                    print(f"     Type: {result['type']}")
                    print(f"     Agents: {result['agents']}")
                    
                    # Get conversation details
                    await asyncio.sleep(2)  # Wait for conversation to process
                    
                    async with session.get(
                        f"{AUTOGEN_SERVICE_URL}/conversations/{result['conversation_id']}"
                    ) as detail_resp:
                        if detail_resp.status == 200:
                            details = await detail_resp.json()
                            print(f"     Result: {details.get('result', {}).get('summary', 'Processing...')}")
                else:
                    print(f"  ❌ Failed to start conversation: {resp.status}")
                    text = await resp.text()
                    print(f"     Error: {text}")
        except Exception as e:
            print(f"  ❌ Error in conversation: {e}")


async def test_group_chat():
    """Test group chat"""
    print("\n🧪 Testing Group Chat...")
    
    async with aiohttp.ClientSession() as session:
        chat_data = {
            "agent_names": ["planner", "executor", "critic"],
            "topic": "What's the best architecture for a microservices system?",
            "max_round": 5,
            "selection_method": "auto"
        }
        
        try:
            async with session.post(
                f"{AUTOGEN_SERVICE_URL}/conversations/group",
                json=chat_data
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"  ✅ Started group chat: {result['conversation_id']}")
                    print(f"     Agents: {result['agents']}")
                    print(f"     Status: {result['status']}")
                else:
                    print(f"  ❌ Failed to start group chat: {resp.status}")
        except Exception as e:
            print(f"  ❌ Error in group chat: {e}")


async def test_collaborative_solve():
    """Test collaborative problem solving"""
    print("\n🧪 Testing Collaborative Problem Solving...")
    
    async with aiohttp.ClientSession() as session:
        problem_data = {
            "problem": "Design a scalable real-time chat application",
            "agent_names": ["planner", "executor", "critic"]
        }
        
        try:
            async with session.post(
                f"{AUTOGEN_SERVICE_URL}/conversations/solve",
                json=problem_data
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"  ✅ Started collaborative solving: {result['conversation_id']}")
                    print(f"     Problem type: {result['type']}")
                    print(f"     Agents involved: {result['agents']}")
                else:
                    print(f"  ❌ Failed to start collaborative solving: {resp.status}")
        except Exception as e:
            print(f"  ❌ Error in collaborative solving: {e}")


async def run_all_tests():
    """Run all AutoGen tests"""
    print("=" * 60)
    print("🚀 AUTOGEN INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Check if service is running
    if not await test_health():
        print("❌ AutoGen Service is not running!")
        print(f"   Please ensure it's running on port 8015")
        return
    
    # Run tests
    await test_list_agents()
    await test_create_agent()
    await test_two_agent_conversation()
    await test_group_chat()
    await test_collaborative_solve()
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ AUTOGEN INTEGRATION TEST SUITE COMPLETED")
    print("=" * 60)
    print("\nNote: The service is using mock AutoGen classes for testing")
    print("To use real AutoGen functionality, ensure AutoGen is properly installed")
    print("and OpenAI API keys are configured.")


if __name__ == "__main__":
    asyncio.run(run_all_tests())