#!/usr/bin/env python3
"""
Unified Agent Registry - Aggregates all agents from all services
"""

import asyncio
import httpx
from typing import Dict, List, Any
from datetime import datetime

class UnifiedAgentRegistry:
    """Aggregates agents from all services"""
    
    def __init__(self):
        self.service_endpoints = {
            'agent-coordinator': 'http://agent-coordinator:8011/agents',
            'workflow-api': 'http://workflow-api:8004/agents', 
            'workflow-engine': 'http://workflow-engine:8013/agents',
            'knowledge-manager': 'http://knowledge-manager:8014/agents',
            'memory-manager': 'http://memory-manager:8012/agents'
        }
        
        # Local agents (fallback)
        self.local_agents = {
            'full_stack_developer': {
                'name': 'Full Stack Developer',
                'specialization': 'Complete web application development',
                'model': 'claude-3-sonnet',
                'capabilities': ['frontend', 'backend', 'api-design', 'full-stack'],
                'service': 'local'
            },
            'data_scientist': {
                'name': 'Data Scientist', 
                'specialization': 'Data analysis and machine learning',
                'model': 'claude-3-sonnet',
                'capabilities': ['data-analysis', 'machine-learning', 'visualization', 'statistics'],
                'service': 'local'
            },
            'security_expert': {
                'name': 'Security Expert',
                'specialization': 'Cybersecurity and secure coding', 
                'model': 'claude-3-opus',
                'capabilities': ['security-analysis', 'vulnerability-assessment', 'compliance'],
                'service': 'local'
            },
            'devops_engineer': {
                'name': 'DevOps Engineer',
                'specialization': 'Infrastructure and deployment automation',
                'model': 'claude-3-haiku', 
                'capabilities': ['infrastructure', 'deployment', 'monitoring', 'automation'],
                'service': 'local'
            },
            'system_architect': {
                'name': 'System Architect',
                'specialization': 'Software architecture and system design',
                'model': 'gpt-4o',
                'capabilities': ['architecture-design', 'system-design', 'scalability', 'integration'],
                'service': 'local'
            }
        }
    
    async def get_all_agents(self) -> Dict[str, Any]:
        """Get agents from all services"""
        all_agents = {}
        
        # Add local agents first
        for agent_id, config in self.local_agents.items():
            all_agents[agent_id] = {
                'id': agent_id,
                'status': 'available',
                **config
            }
        
        # Query all services
        async with httpx.AsyncClient(timeout=2.0) as client:
            for service_name, endpoint in self.service_endpoints.items():
                try:
                    response = await client.get(endpoint)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Handle different response formats
                        agents_data = []
                        if 'agents' in data:
                            if isinstance(data['agents'], list):
                                if data['agents'] and isinstance(data['agents'][0], str):
                                    # List of agent IDs
                                    agents_data = [{'id': aid, 'service': service_name} for aid in data['agents']]
                                else:
                                    # List of agent objects
                                    agents_data = data['agents']
                        
                        # Add service agents
                        for agent in agents_data:
                            if isinstance(agent, dict):
                                agent_id = agent.get('id', f"{service_name}_{len(all_agents)}")
                                all_agents[agent_id] = {
                                    'id': agent_id,
                                    'name': agent.get('name', agent_id.replace('_', ' ').title()),
                                    'service': service_name,
                                    'status': agent.get('status', 'available'),
                                    'capabilities': agent.get('capabilities', []),
                                    **agent
                                }
                
                except Exception as e:
                    print(f"Failed to get agents from {service_name}: {e}")
        
        return {
            'agents': list(all_agents.values()),
            'count': len(all_agents),
            'services_checked': list(self.service_endpoints.keys()),
            'timestamp': datetime.now().isoformat()
        }