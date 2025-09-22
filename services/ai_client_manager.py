#!/usr/bin/env python3
"""
AI Client Manager - Enhanced multi-provider AI client support
Supports OpenAI, Anthropic, and Grok/xAI for task analysis and confidence scoring
"""

import os
import asyncio
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AIClientManager:
    """Enhanced AI client manager supporting multiple providers"""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.grok_client = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize all available AI clients"""
        # OpenAI
        try:
            from openai import OpenAI
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                self.openai_client = OpenAI(api_key=openai_key)
                logger.info("✅ OpenAI client initialized")
        except ImportError:
            logger.warning("OpenAI package not available")

        # Anthropic
        try:
            from anthropic import Anthropic
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if anthropic_key:
                self.anthropic_client = Anthropic(api_key=anthropic_key)
                logger.info("✅ Anthropic client initialized")
        except ImportError:
            logger.warning("Anthropic package not available")

        # Grok (xAI)
        try:
            import requests
            grok_key = os.getenv('GROK_API_KEY') or os.getenv('XAI_API_KEY')
            if grok_key:
                self.grok_client = {"api_key": grok_key, "base_url": "https://api.x.ai/v1"}
                logger.info("✅ Grok client initialized")
        except ImportError:
            logger.warning("Grok/xAI client not available")

    def get_available_clients(self) -> Dict[str, bool]:
        """Get status of all AI clients"""
        return {
            "openai": self.openai_client is not None,
            "anthropic": self.anthropic_client is not None,
            "grok": self.grok_client is not None
        }

    def has_any_client(self) -> bool:
        """Check if any AI client is available"""
        return any(self.get_available_clients().values())

    async def analyze_task_complexity(self, task_description: str) -> Dict[str, Any]:
        """Use AI to analyze task complexity and requirements"""
        if not self.has_any_client():
            return {"complexity": "medium", "confidence": 0.5, "analysis": "No AI clients available"}

        try:
            # Use OpenAI if available
            if self.openai_client:
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "system",
                        "content": "Analyze the task complexity and categorize it. Return JSON with: complexity (low/medium/high), confidence (0-1), categories (array of relevant categories), and analysis (brief explanation)."
                    }, {
                        "role": "user",
                        "content": f"Task: {task_description}"
                    }],
                    max_tokens=200,
                    temperature=0.3
                )

                result_text = response.choices[0].message.content
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    return {"complexity": "medium", "confidence": 0.7, "analysis": result_text}

            # Use Anthropic if available
            elif self.anthropic_client:
                response = await asyncio.to_thread(
                    self.anthropic_client.messages.create,
                    model="claude-3-haiku-20240307",
                    max_tokens=200,
                    system="Analyze the task complexity and categorize it. Return JSON with: complexity (low/medium/high), confidence (0-1), categories (array of relevant categories), and analysis (brief explanation).",
                    messages=[{"role": "user", "content": f"Task: {task_description}"}]
                )

                result_text = response.content[0].text
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    return {"complexity": "medium", "confidence": 0.7, "analysis": result_text}

            # Use Grok if available
            elif self.grok_client:
                import requests
                headers = {
                    "Authorization": f"Bearer {self.grok_client['api_key']}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "grok-beta",
                    "messages": [{
                        "role": "system",
                        "content": "Analyze the task complexity and categorize it. Return JSON with: complexity (low/medium/high), confidence (0-1), categories (array of relevant categories), and analysis (brief explanation)."
                    }, {
                        "role": "user",
                        "content": f"Task: {task_description}"
                    }],
                    "max_tokens": 200,
                    "temperature": 0.3
                }

                response = await asyncio.to_thread(
                    requests.post,
                    f"{self.grok_client['base_url']}/chat/completions",
                    headers=headers,
                    json=data
                )

                if response.status_code == 200:
                    result = response.json()
                    result_text = result["choices"][0]["message"]["content"]
                    try:
                        return json.loads(result_text)
                    except json.JSONDecodeError:
                        return {"complexity": "medium", "confidence": 0.7, "analysis": result_text}

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")

        return {"complexity": "medium", "confidence": 0.5, "analysis": "Fallback analysis"}

    async def get_task_suggestions(self, task_description: str) -> Dict[str, Any]:
        """Get AI-powered suggestions for task execution"""
        if not self.has_any_client():
            return {"suggestions": [], "confidence": 0.5, "analysis": "No AI clients available"}

        try:
            # Use OpenAI if available
            if self.openai_client:
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "system",
                        "content": "Provide specific suggestions for executing this task. Return JSON with: suggestions (array of actionable steps), confidence (0-1), and analysis (brief explanation)."
                    }, {
                        "role": "user",
                        "content": f"Task: {task_description}"
                    }],
                    max_tokens=300,
                    temperature=0.4
                )

                result_text = response.choices[0].message.content
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    return {"suggestions": [], "confidence": 0.6, "analysis": result_text}

        except Exception as e:
            logger.error(f"AI suggestions failed: {e}")

        return {"suggestions": [], "confidence": 0.5, "analysis": "Fallback suggestions"}