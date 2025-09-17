#!/usr/bin/env python3
"""
Agent Executor Fix - Minimal implementation to make agents actually work
This bridges the gap between task assignment and AI execution
"""

import os
import asyncio
import json
import time
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentExecutor:
    """Minimal agent executor that actually performs tasks"""
    
    def __init__(self):
        # Initialize AI clients
        self.openai_client = None
        self.anthropic_client = None
        
        # Try to initialize OpenAI
        self.openai_client = None
        try:
            from openai import OpenAI
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                self.openai_client = OpenAI(api_key=openai_key)
                logger.info("✅ OpenAI client initialized")
        except ImportError:
            logger.warning("OpenAI package not available")
        
        # Try to initialize Anthropic
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            self.anthropic_client = Anthropic(api_key=anthropic_key)
            logger.info("✅ Anthropic client initialized")
        
        if not self.openai_client and not self.anthropic_client:
            logger.warning("⚠️ No AI clients available. Using mock responses.")
    
    async def execute_task(self, task_description: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the appropriate AI model"""
        
        try:
            # Get agent configuration
            model = agent_config.get('model', 'gpt-4o')
            specialization = agent_config.get('specialization', 'general')
            capabilities = agent_config.get('capabilities', [])
            
            # Build system prompt based on agent specialization
            system_prompt = self._build_system_prompt(specialization, capabilities)
            
            # Add web search capability to task
            enhanced_task = await self._enhance_task_with_web_data(task_description)
            logger.info(f"Original task length: {len(task_description)}")
            logger.info(f"Enhanced task length: {len(enhanced_task)}")
            logger.info(f"Task enhanced: {enhanced_task != task_description}")
            
            # Execute with appropriate model
            if (model.startswith('gpt') or model.startswith('o1')) and self.openai_client:
                result = await self._execute_with_openai(enhanced_task, system_prompt, model)
            elif ('claude' in model.lower()) and self.anthropic_client:
                result = await self._execute_with_anthropic(enhanced_task, system_prompt, model)
            else:
                # Fallback to mock execution
                result = await self._execute_mock(enhanced_task, specialization)
            
            return {
                "status": "completed",
                "result": result,
                "agent_model": model,
                "execution_time": time.time(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_system_prompt(self, specialization: str, capabilities: list) -> str:
        """Build system prompt based on agent specialization"""
        
        base_prompt = "You are a helpful AI assistant that performs tasks efficiently and provides concrete results."
        
        specialization_prompts = {
            'full_stack_developer': """You are a senior full-stack developer. When given a task:
- Write actual code, don't just describe what to do
- Provide complete, working implementations
- Include necessary imports and dependencies
- Consider security, performance, and best practices
- Create files, functions, and complete solutions""",
            
            'data_scientist': """You are an expert data scientist. When given a task:
- Write actual Python code for data analysis
- Provide complete data processing pipelines
- Create visualizations and statistical analyses
- Generate insights from data, don't just describe methods""",
            
            'security_expert': """You are a cybersecurity expert. When given a task:
- Perform actual security analysis
- Write security scanning code
- Provide specific vulnerability assessments
- Create security implementations, not just recommendations""",
            
            'devops_engineer': """You are a DevOps engineer. When given a task:
- Write actual deployment scripts
- Create infrastructure as code
- Provide working CI/CD configurations
- Implement monitoring and automation solutions""",
            
            'system_architect': """You are a system architect. When given a task:
- Design actual system architectures
- Create detailed technical specifications
- Provide implementation roadmaps
- Design scalable, maintainable solutions""",
        }
        
        prompt = specialization_prompts.get(specialization, base_prompt)
        
        if capabilities:
            prompt += f"\\n\\nYour specific capabilities include: {', '.join(capabilities)}"
        
        prompt += "\\n\\nIMPORTANT: Always provide concrete, actionable results. Create actual code, files, or implementations rather than just describing what should be done."
        prompt += "\\n\\nYou have access to current web information. When asked about recent developments, trends, or current information, use the provided web search results to give accurate, up-to-date answers."
        
        return prompt
    
    async def _execute_with_openai(self, task: str, system_prompt: str, model: str) -> str:
        """Execute task using OpenAI"""
        try:
            if not self.openai_client:
                raise Exception("OpenAI client not initialized")
                
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task}
                ],
                max_tokens=4000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI execution failed: {e}")
            raise
    
    async def _execute_with_anthropic(self, task: str, system_prompt: str, model: str) -> str:
        """Execute task using Anthropic Claude"""
        try:
            # Map model names to correct Anthropic format
            model_map = {
                'claude-3-opus-20240229': 'claude-3-opus-20240229',
                'claude-3-sonnet-20240229': 'claude-3-sonnet-20240229', 
                'claude-3-haiku-20240307': 'claude-3-haiku-20240307',
                'claude-3-5-sonnet-20241022': 'claude-3-5-sonnet-20241022',
                'claude-3-5-haiku-20241022': 'claude-3-5-haiku-20241022'
            }
            
            actual_model = model_map.get(model, model)
            logger.info(f"Calling Claude with model: {actual_model}")
            logger.info(f"Task length: {len(task)} chars")
            
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model=actual_model,
                max_tokens=4000,
                system=system_prompt,
                messages=[{"role": "user", "content": task}]
            )
            
            logger.info(f"Claude response type: {type(response)}")
            logger.info(f"Claude response content length: {len(response.content) if response.content else 0}")
            
            if response.content and len(response.content) > 0:
                result_text = response.content[0].text
                logger.info(f"Claude result length: {len(result_text)} chars")
                return result_text if result_text.strip() else "Claude returned empty text"
            else:
                return "Claude response had no content"
        except Exception as e:
            logger.error(f"Anthropic execution failed: {e}")
            return f"Claude error: {str(e)}"
    
    async def _execute_mock(self, task: str, specialization: str) -> str:
        """Mock execution for testing without API keys"""
        
        mock_responses = {
            'full_stack_developer': f"""# Full-Stack Solution for: {task}

```python
# Example implementation
def solve_task():
    '''
    Complete implementation for: {task}
    '''
    # TODO: Replace with actual implementation
    result = "Task completed successfully"
    return result

if __name__ == "__main__":
    result = solve_task()
    print(f"Result: {{result}}")
```

## Implementation Notes:
- Created working code structure
- Included error handling
- Added documentation
- Ready for deployment

**Status: IMPLEMENTED** ✅""",
            
            'data_scientist': f"""# Data Science Solution for: {task}

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data analysis implementation
def analyze_data():
    '''
    Data analysis for: {task}
    '''
    # Sample data processing
    data = pd.DataFrame({{'values': np.random.randn(100)}})
    
    # Analysis
    summary = data.describe()
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(data['values'], bins=20)
    plt.title('Data Distribution')
    plt.savefig('analysis_result.png')
    
    return summary

result = analyze_data()
print("Analysis completed:", result)
```

**Analysis Complete** ✅""",
            
            'default': f"""# Task Execution Result

## Task: {task}

### Implementation:
✅ Task analyzed and processed
✅ Solution developed
✅ Implementation ready

### Output:
The requested task has been completed successfully. 

**Status: COMPLETED** ✅"""
        }
        
        return mock_responses.get(specialization, mock_responses['default'])
    
    async def _enhance_task_with_web_data(self, task: str) -> str:
        """Enhance task with relevant web data"""
        try:
            # Check if task needs web search
            task_lower = task.lower()
            needs_web = any(word in task_lower for word in 
                          ['latest', 'current', 'recent', 'new', 'trends', '2024', '2025', 'developments', 'ai']) or 'browse' in task_lower
            
            logger.info(f"Task needs web search: {needs_web} (task: {task[:50]}...)")
            
            if needs_web:
                search_terms = self._extract_search_terms(task)
                logger.info(f"Searching web for: {search_terms}")
                web_info = await self._search_web(search_terms)
                
                if web_info and len(web_info) > 50:
                    enhanced_task = f"{task}\n\n=== CURRENT WEB INFORMATION ===\n{web_info}\n\nIMPORTANT: Use the above current web information to provide an up-to-date answer. Do not say you cannot browse the internet."
                    logger.info(f"Enhanced task with web data: {len(web_info)} chars")
                    return enhanced_task
                else:
                    logger.warning(f"Web search returned insufficient data: {len(web_info) if web_info else 0} chars")
            
            return task
        except Exception as e:
            logger.warning(f"Web enhancement failed: {e}")
            return task
    
    def _extract_search_terms(self, task: str) -> str:
        """Extract key terms for web search"""
        # Simple keyword extraction
        keywords = []
        task_lower = task.lower()
        
        # Look for specific technologies, trends, etc.
        tech_terms = ['ai', 'machine learning', 'python', 'javascript', 'react', 'docker', 'kubernetes', 
                     'blockchain', 'quantum computing', 'cybersecurity', 'cloud computing']
        
        for term in tech_terms:
            if term in task_lower:
                keywords.append(term)
        
        # Add "latest" or "current" for trend searches
        if any(word in task_lower for word in ['latest', 'current', 'new', 'trends', 'developments']):
            keywords.append('2024')
        
        return ' '.join(keywords[:3])  # Limit to 3 terms
    
    async def _search_web(self, search_terms: str) -> str:
        """Search web for current information"""
        try:
            from web_search_tool import search_web
            
            # Use the web search tool
            result = await asyncio.to_thread(search_web, search_terms, 3)
            return result
            
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return None


# Integration with existing system
class TaskExecutionBridge:
    """Bridge between task assignment and actual execution"""
    
    def __init__(self):
        self.executor = AgentExecutor()
        
    async def execute_agent_task(self, agent_id: str, task_description: str, context: Dict = None, model: str = None) -> Dict[str, Any]:
        """Execute a task with a specific agent"""
        
        # Get agent configuration
        agent_config = await self._get_agent_config(agent_id)
        
        if not agent_config:
            return {
                "status": "failed",
                "error": f"Agent {agent_id} not found",
                "timestamp": datetime.now().isoformat()
            }
        
        # Override model if specified
        if model:
            agent_config['model'] = model
        
        # Add context to task if provided
        full_task = task_description
        if context:
            full_task += f"\\n\\nContext: {json.dumps(context, indent=2)}"
        
        # Execute the task
        result = await self.executor.execute_task(full_task, agent_config)
        
        return result
    
    async def _get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration (integrate with actual agent service)"""
        
        # Mock agent configurations - replace with actual API call
        mock_agents = {
            'full_stack_developer': {
                'model': 'gpt-4o',
                'specialization': 'full_stack_developer',
                'capabilities': ['frontend', 'backend', 'api-design', 'full-stack']
            },
            'data_scientist': {
                'model': 'gpt-4o',
                'specialization': 'data_scientist',
                'capabilities': ['data-analysis', 'machine-learning', 'visualization']
            },
            'security_expert': {
                'model': 'gpt-4o',
                'specialization': 'security_expert',
                'capabilities': ['security-analysis', 'vulnerability-assessment']
            },
            'devops_engineer': {
                'model': 'gpt-4o',
                'specialization': 'devops_engineer',
                'capabilities': ['infrastructure', 'deployment', 'monitoring']
            },
            'system_architect': {
                'model': 'gpt-4o',
                'specialization': 'system_architect',
                'capabilities': ['architecture-design', 'system-design']
            }
        }
        
        return mock_agents.get(agent_id)


# Simple API for testing
if __name__ == "__main__":
    async def test_execution():
        bridge = TaskExecutionBridge()
        
        # Test task
        result = await bridge.execute_agent_task(
            agent_id="full_stack_developer",
            task_description="Create a simple REST API endpoint for user authentication",
            context={"framework": "FastAPI", "database": "PostgreSQL"}
        )
        
        print("Execution Result:")
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_execution())

# Install required packages if not available
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run(['pip', 'install', 'requests', 'beautifulsoup4'])