#!/usr/bin/env python3
"""
LLM Provider Integrations for Agent Lightning
Supports OpenAI, Anthropic, and Grok models
"""

import os
import asyncio
import logging
import base64
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from io import BytesIO

from .types import MultiModalContent, TextContent, ImageContent, AudioContent, VideoContent, ContentType

logger = logging.getLogger(__name__)


def convert_multimodal_to_messages(content: Union[str, List[MultiModalContent], MultiModalContent]) -> List[Dict[str, Any]]:
    """
    Convert multi-modal content to OpenAI/Anthropic message format.

    Args:
        content: Text string, single MultiModalContent, or list of MultiModalContent

    Returns:
        List of message dictionaries compatible with LLM APIs
    """
    messages = []

    if isinstance(content, str):
        # Simple text input
        messages.append({"role": "user", "content": content})
    elif isinstance(content, MultiModalContent):
        # Single multi-modal content
        content_list = [content]
    elif isinstance(content, list):
        # List of multi-modal content
        content_list = content
    else:
        raise ValueError(f"Unsupported content type: {type(content)}")

    if 'content_list' in locals():
        # Build multi-modal message
        message_content = []

        for item in content_list:
            if isinstance(item, TextContent):
                message_content.append({"type": "text", "text": item.text})
            elif isinstance(item, ImageContent):
                # Handle image data
                if isinstance(item.image_data, str):
                    # Assume it's base64 or URL
                    if item.image_data.startswith('data:'):
                        # Already data URL
                        image_url = item.image_data
                    elif item.image_data.startswith('http'):
                        # URL
                        image_url = item.image_data
                    else:
                        # Assume base64
                        format_str = f"data:image/{item.format or 'png'};base64,"
                        image_url = format_str + item.image_data
                else:
                    # Bytes - convert to base64
                    import base64
                    b64_data = base64.b64encode(item.image_data).decode('utf-8')
                    format_str = f"data:image/{item.format or 'png'};base64,"
                    image_url = format_str + b64_data

                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            elif isinstance(item, AudioContent):
                # For now, convert audio to text description
                # TODO: Implement proper audio handling when supported
                message_content.append({
                    "type": "text",
                    "text": f"[Audio content: {item.format or 'unknown format'}, duration: {item.duration or 'unknown'}s]"
                })
            elif isinstance(item, VideoContent):
                # For now, convert video to text description
                # TODO: Implement proper video handling when supported
                message_content.append({
                    "type": "text",
                    "text": f"[Video content: {item.format or 'unknown format'}, duration: {item.duration or 'unknown'}s]"
                })

        messages.append({"role": "user", "content": message_content})

    return messages


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._get_api_key()
        self.client = None
        self._initialize_client()

    @abstractmethod
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment"""
        pass

    @abstractmethod
    def _initialize_client(self):
        """Initialize the LLM client"""
        pass

    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate response from LLM"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        return self.client is not None


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""

    def _get_api_key(self) -> Optional[str]:
        return os.getenv('OPENAI_API_KEY')

    def _initialize_client(self):
        if self.api_key:
            try:
                import openai
                self.client = openai.AsyncOpenAI(api_key=self.api_key)
                logger.info("✅ OpenAI provider initialized")
            except ImportError:
                logger.warning("OpenAI package not installed")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        if not self.client:
            raise Exception("OpenAI client not initialized")

        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get('model', 'gpt-4o'),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000),
                **kwargs
            )

            return {
                'content': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def is_available(self) -> bool:
        return self.client is not None


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider"""

    def _get_api_key(self) -> Optional[str]:
        return os.getenv('ANTHROPIC_API_KEY')

    def _initialize_client(self):
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
                logger.info("✅ Anthropic provider initialized")
            except ImportError:
                logger.warning("Anthropic package not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        if not self.client:
            raise Exception("Anthropic client not initialized")

        try:
            # Convert OpenAI format to Anthropic format
            system_message = ""
            anthropic_messages = []

            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    anthropic_messages.append(msg)

            response = await self.client.messages.create(
                model=kwargs.get('model', 'claude-3-5-sonnet-20241022'),
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                system=system_message,
                messages=anthropic_messages
            )

            return {
                'content': response.content[0].text,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                },
                'model': response.model,
                'finish_reason': response.stop_reason
            }
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

    def is_available(self) -> bool:
        return self.client is not None


class GrokProvider(LLMProvider):
    """Grok LLM provider (xAI)"""

    def _get_api_key(self) -> Optional[str]:
        return os.getenv('GROK_API_KEY') or os.getenv('XAI_API_KEY')

    def _initialize_client(self):
        if self.api_key:
            try:
                import openai
                # Grok uses OpenAI-compatible API
                self.client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1"
                )
                logger.info("✅ Grok provider initialized")
            except ImportError:
                logger.warning("OpenAI package not installed (required for Grok)")
            except Exception as e:
                logger.error(f"Failed to initialize Grok client: {e}")

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        if not self.client:
            raise Exception("Grok client not initialized")

        try:
            # Map grok-code-fast-1 to grok-beta if needed
            model = kwargs.get('model', 'grok-code-fast-1')
            if model == 'grok-code-fast-1':
                model = 'grok-beta'  # Use the available model

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000),
                **kwargs
            )

            return {
                'content': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'total_tokens': response.usage.total_tokens if response.usage else 0
                },
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }
        except Exception as e:
            logger.error(f"Grok generation failed: {e}")
            raise

    def is_available(self) -> bool:
        return self.client is not None


class LLMProviderManager:
    """Manager for multiple LLM providers"""

    def __init__(self):
        self.providers = {
            'openai': OpenAIProvider(),
            'anthropic': AnthropicProvider(),
            'grok': GrokProvider()
        }
        logger.info("LLM Provider Manager initialized")

    def get_provider(self, model_name: str) -> Optional[LLMProvider]:
        """Get appropriate provider based on model name"""
        if model_name.startswith('gpt') or model_name.startswith('o1'):
            return self.providers.get('openai')
        elif 'claude' in model_name.lower():
            return self.providers.get('anthropic')
        elif 'grok' in model_name.lower():
            return self.providers.get('grok')
        else:
            # Try all providers to find one that supports the model
            for provider in self.providers.values():
                if provider.is_available():
                    return provider
            return None

    def get_available_providers(self) -> Dict[str, bool]:
        """Get availability status of all providers"""
        return {
            name: provider.is_available()
            for name, provider in self.providers.items()
        }

    async def generate(self, model_name: str, messages: Union[List[Dict[str, Any]], str, List[MultiModalContent], MultiModalContent], **kwargs) -> Dict[str, Any]:
        """Generate response using appropriate provider"""
        provider = self.get_provider(model_name)
        if not provider:
            raise Exception(f"No available provider for model: {model_name}")

        # Convert multi-modal content to message format if needed
        if isinstance(messages, (str, list)) and (not messages or (isinstance(messages, list) and not isinstance(messages[0], dict))):
            messages = convert_multimodal_to_messages(messages)

        # Update kwargs with model name
        kwargs['model'] = model_name
        return await provider.generate(messages, **kwargs)

    def list_supported_models(self) -> Dict[str, List[str]]:
        """List supported models by provider"""
        return {
            'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
            'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'],
            'grok': ['grok-code-fast-1', 'grok-beta', 'grok-2-vision']
        }

    def list_vision_supported_models(self) -> Dict[str, List[str]]:
        """List vision-capable models by provider"""
        return {
            'openai': ['gpt-4o', 'gpt-4-turbo'],
            'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'],
            'grok': ['grok-2-vision']
        }


# Global instance
llm_manager = LLMProviderManager()


async def generate_with_model(model_name: str, messages: Union[List[Dict[str, Any]], str, List[MultiModalContent], MultiModalContent], **kwargs) -> Dict[str, Any]:
    """Convenience function to generate with any supported model"""
    return await llm_manager.generate(model_name, messages, **kwargs)


def get_available_providers() -> Dict[str, bool]:
    """Get availability status of all providers"""
    return llm_manager.get_available_providers()


def list_supported_models() -> Dict[str, List[str]]:
    """List all supported models"""
    return llm_manager.list_supported_models()


# Backward compatibility functions
async def generate_openai(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Generate using OpenAI (backward compatibility)"""
    return await llm_manager.providers['openai'].generate(messages, **kwargs)


async def generate_anthropic(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Generate using Anthropic (backward compatibility)"""
    return await llm_manager.providers['anthropic'].generate(messages, **kwargs)


async def generate_grok(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Generate using Grok (new functionality)"""
    return await llm_manager.providers['grok'].generate(messages, **kwargs)