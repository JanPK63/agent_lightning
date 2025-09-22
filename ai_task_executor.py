#!/usr/bin/env python3
"""
AI Task Executor Module
Handles task execution using AI models and integrates with the agent system
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import shared components
from shared.data_access import DataAccessLayer
from agentlightning.llm_providers import generate_with_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AITaskExecutor:
    """
    AI Task Executor - Handles task execution using AI models
    Integrates with the agent system and shared database
    """

    def __init__(self):
        """Initialize the AI Task Executor"""
        self.dal = DataAccessLayer("ai_task_executor")
        logger.info("✅ AI Task Executor initialized")

    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a task using AI models

        Args:
            task_id: The ID of the task to execute

        Returns:
            Dict containing execution results
        """
        try:
            # Get task details from database
            task = self.dal.get_task(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")

            logger.info(f"Executing task {task_id}: {task.get('description', 'No description')}")

            # Update task status to started
            self.dal.update_task_status(task_id, "started")

            # Get agent details
            agent_id = task.get('agent_id')
            if agent_id:
                agent = self.dal.get_agent(agent_id)
                agent_model = agent.get('model', 'claude-3-haiku') if agent else 'claude-3-haiku'
            else:
                agent_model = 'claude-3-haiku'

            # Prepare execution context
            context = task.get('context', {})
            task_description = task.get('description', '')

            # Execute the task using AI
            result = await self._execute_with_ai(task_description, agent_model, context)

            # Update task with results
            update_data = {
                "status": "completed",
                "result": result,
                "completed_at": datetime.utcnow().isoformat()
            }
            self.dal.update_task(task_id, update_data)

            # Update agent status back to idle
            if agent_id:
                self.dal.update_agent_status(agent_id, "idle")

            logger.info(f"Task {task_id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Task execution failed for {task_id}: {e}")

            # Update task status to failed
            error_data = {
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat()
            }
            self.dal.update_task(task_id, error_data)

            # Update agent status to error
            agent_id = task.get('agent_id') if 'task' in locals() else None
            if agent_id:
                self.dal.update_agent_status(agent_id, "error")

            raise

    async def _execute_with_ai(self, task_description: str, model: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task using AI model

        Args:
            task_description: Description of the task
            model: AI model to use
            context: Additional context for execution

        Returns:
            Dict containing execution results
        """
        try:
            # Prepare the prompt for task execution
            system_prompt = """You are an AI assistant executing tasks for an agent system.
Your role is to analyze the task description and provide a helpful, accurate response.
Focus on being practical and providing actionable results."""

            user_prompt = f"""Task: {task_description}

Context: {json.dumps(context, indent=2)}

Please execute this task and provide a detailed response with your analysis and results."""

            # Get AI response
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Use the LLM provider to get response
            response = await generate_with_model(
                model_name=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )

            # Parse and structure the response
            result = {
                "task_description": task_description,
                "model_used": model,
                "response": response.get("content", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "context": context,
                "success": True
            }

            return result

        except Exception as e:
            logger.error(f"AI execution failed: {e}")
            return {
                "task_description": task_description,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "success": False
            }

    async def analyze_code(self, code: str, task_description: str) -> Dict[str, Any]:
        """
        Analyze code as part of task execution

        Args:
            code: Code to analyze
            task_description: Description of the analysis task

        Returns:
            Dict containing analysis results
        """
        try:
            system_prompt = """You are a code analysis expert. Analyze the provided code and provide insights,
suggestions for improvements, potential issues, and best practices."""

            user_prompt = f"""Task: {task_description}

Code to analyze:
```python
{code}
```

Please provide a comprehensive analysis including:
1. Code quality assessment
2. Potential issues or bugs
3. Suggestions for improvements
4. Best practices recommendations"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = await generate_with_model(
                model_name="claude-3-haiku",
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )

            return {
                "analysis_type": "code_analysis",
                "code_length": len(code),
                "analysis": response.get("content", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "success": True
            }

        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {
                "analysis_type": "code_analysis",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "success": False
            }

    async def generate_documentation(self, content: str, doc_type: str = "general") -> Dict[str, Any]:
        """
        Generate documentation for given content

        Args:
            content: Content to document
            doc_type: Type of documentation (api, code, general)

        Returns:
            Dict containing generated documentation
        """
        try:
            system_prompt = f"""You are a technical documentation expert specializing in {doc_type} documentation.
Generate clear, comprehensive documentation that follows best practices."""

            user_prompt = f"""Generate {doc_type} documentation for the following content:

{content}

Please provide:
1. Overview/summary
2. Key components or sections
3. Usage examples (if applicable)
4. Important notes or considerations"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = await self.llm_provider.generate_response(
                messages=messages,
                model="claude-3-haiku",
                temperature=0.2,
                max_tokens=1500
            )

            return {
                "documentation_type": doc_type,
                "content_length": len(content),
                "documentation": response.get("content", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "success": True
            }

        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return {
                "documentation_type": doc_type,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "success": False
            }


# Test function for development
async def test_executor():
    """Test the AI Task Executor"""
    executor = AITaskExecutor()

    # Test basic task execution
    test_task = {
        "id": "test_task_001",
        "description": "Analyze the current project structure",
        "agent_id": "test_agent",
        "context": {"source": "test"}
    }

    # Create test task in database
    executor.dal.create_task(test_task)

    # Execute the task
    result = await executor.execute_task("test_task_001")
    print(f"Test execution result: {result}")

    return result


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_executor())