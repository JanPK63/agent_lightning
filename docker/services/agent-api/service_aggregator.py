#!/usr/bin/env python3
"""
Service Aggregator - API Backbone for Agent Lightning
Connects all microservices into one unified system
"""

import asyncio
import httpx
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ServiceAggregator:
    """Aggregates all microservices into unified API"""
    
    def __init__(self):
        self.services = {
            'agent-coordinator': 'http://agent-coordinator:8011',
            'memory-manager': 'http://memory-manager:8012', 
            'workflow-engine': 'http://workflow-engine:8013',
            'knowledge-manager': 'http://knowledge-manager:8014',
            'rl-training-server': 'http://rl-training-server:8010',
            'workflow-api': 'http://workflow-api:8004',
            'rl-api': 'http://rl-api:8003'
        }
        
    async def get_all_agents(self) -> Dict[str, Any]:
        """Aggregate agents from all services"""
        all_agents = []
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Get agents from each service
            for service_name, base_url in self.services.items():
                try:
                    response = await client.get(f"{base_url}/agents")
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Handle different response formats
                        if isinstance(data, dict):
                            if 'agents' in data:
                                agents = data['agents']
                            elif 'data' in data:
                                agents = data['data']
                            else:
                                agents = [data] if data else []
                        else:
                            agents = data if isinstance(data, list) else []
                        
                        # Add service info to each agent
                        for agent in agents:
                            if isinstance(agent, dict):
                                agent['service'] = service_name
                                agent['service_url'] = base_url
                                all_agents.append(agent)
                            elif isinstance(agent, str):
                                all_agents.append({
                                    'id': agent,
                                    'name': agent.replace('_', ' ').title(),
                                    'service': service_name,
                                    'service_url': base_url,
                                    'status': 'available'
                                })
                                
                except Exception as e:
                    logger.warning(f"Failed to get agents from {service_name}: {e}")
        
        return {
            'agents': all_agents,
            'count': len(all_agents),
            'services': list(self.services.keys()),
            'timestamp': datetime.now().isoformat()
        }
    
    async def execute_task(self, agent_id: str, task: str, **kwargs) -> Dict[str, Any]:
        """Execute task by routing to appropriate service"""
        
        # Find which service has this agent
        agents_data = await self.get_all_agents()
        target_service = None
        
        for agent in agents_data['agents']:
            if agent.get('id') == agent_id:
                target_service = agent.get('service')
                break
        
        if not target_service:
            return {'error': f'Agent {agent_id} not found'}
        
        service_url = self.services.get(target_service)
        if not service_url:
            return {'error': f'Service {target_service} not available'}
        
        # Route to appropriate service
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                payload = {
                    'agent_id': agent_id,
                    'task': task,
                    **kwargs
                }
                
                response = await client.post(f"{service_url}/execute", json=payload)
                if response.status_code == 200:
                    return response.json()
                else:
                    return {'error': f'Service returned {response.status_code}'}
                    
            except Exception as e:
                return {'error': f'Failed to execute task: {e}'}
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Check health of all services"""
        health_status = {}
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for service_name, base_url in self.services.items():
                try:
                    response = await client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        health_status[service_name] = {
                            'status': 'healthy',
                            'url': base_url,
                            'response_time': response.elapsed.total_seconds()
                        }
                    else:
                        health_status[service_name] = {
                            'status': 'unhealthy',
                            'url': base_url,
                            'error': f'HTTP {response.status_code}'
                        }
                except Exception as e:
                    health_status[service_name] = {
                        'status': 'unreachable',
                        'url': base_url,
                        'error': str(e)
                    }
        
        return {
            'services': health_status,
            'total_services': len(self.services),
            'healthy_services': len([s for s in health_status.values() if s['status'] == 'healthy']),
            'timestamp': datetime.now().isoformat()
        }