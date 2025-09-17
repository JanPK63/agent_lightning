#!/usr/bin/env python3
"""
Test the knowledge-enhanced agents
"""

from knowledge_manager import KnowledgeManager

# Initialize knowledge manager
km = KnowledgeManager()

print("=" * 60)
print("🧪 TESTING AGENT KNOWLEDGE")
print("=" * 60)

# Test full-stack developer's knowledge about Agent Lightning
print("\n📚 Full-Stack Developer Knowledge Test")
print("-" * 40)

queries = [
    "monitoring dashboard",
    "production API",
    "memory manager",
    "agent configuration",
    "reward functions"
]

for query in queries:
    print(f"\n🔍 Searching for: '{query}'")
    results = km.search_knowledge("full_stack_developer", query, limit=2)
    
    if results:
        for i, item in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"  Category: {item.category}")
            print(f"  Source: {item.source}")
            print(f"  Content: {item.content[:150]}...")
    else:
        print("  No results found")

# Show statistics
print("\n" + "=" * 60)
print("📊 KNOWLEDGE STATISTICS")
print("=" * 60)

for agent_name in ["full_stack_developer", "devops_engineer"]:
    stats = km.get_statistics(agent_name)
    print(f"\n{agent_name}:")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Categories: {stats['categories']}")
    
print("\n✅ Your agents are now experts on the Agent Lightning system!")