#!/usr/bin/env python3
"""
Tests for LLM Provider integrations
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from agentlightning.llm_providers import (
    LLMProviderManager,
    OpenAIProvider,
    AnthropicProvider,
    GrokProvider,
    generate_with_model,
    get_available_providers,
    list_supported_models
)


class TestLLMProviders:
    """Test LLM provider functionality"""

    def test_provider_manager_initialization(self):
        """Test that provider manager initializes correctly"""
        manager = LLMProviderManager()
        assert manager.providers is not None
        assert 'openai' in manager.providers
        assert 'anthropic' in manager.providers
        assert 'grok' in manager.providers

    def test_get_provider_by_model(self):
        """Test provider selection by model name"""
        manager = LLMProviderManager()

        # Test OpenAI models
        assert manager.get_provider('gpt-4o').__class__.__name__ == 'OpenAIProvider'
        assert manager.get_provider('gpt-3.5-turbo').__class__.__name__ == 'OpenAIProvider'

        # Test Anthropic models
        assert manager.get_provider('claude-3-5-sonnet-20241022').__class__.__name__ == 'AnthropicProvider'
        assert manager.get_provider('claude-3-opus-20240229').__class__.__name__ == 'AnthropicProvider'

        # Test Grok models
        assert manager.get_provider('grok-code-fast-1').__class__.__name__ == 'GrokProvider'
        assert manager.get_provider('grok-beta').__class__.__name__ == 'GrokProvider'

    def test_list_supported_models(self):
        """Test listing supported models"""
        models = list_supported_models()
        assert 'openai' in models
        assert 'anthropic' in models
        assert 'grok' in models

        # Check Grok models are included
        assert 'grok-code-fast-1' in models['grok']
        assert 'grok-beta' in models['grok']

    def test_get_available_providers(self):
        """Test getting provider availability"""
        availability = get_available_providers()
        assert isinstance(availability, dict)
        assert 'openai' in availability
        assert 'anthropic' in availability
        assert 'grok' in availability

    @patch.dict('os.environ', {'GROK_API_KEY': 'test-key'})
    def test_grok_provider_initialization(self):
        """Test Grok provider initialization"""
        with patch('openai.AsyncOpenAI') as mock_openai:
            provider = GrokProvider()
            assert provider.api_key == 'test-key'
            # Provider should attempt to initialize client
            mock_openai.assert_called_once()

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_openai_provider_initialization(self):
        """Test OpenAI provider initialization"""
        with patch('openai.AsyncOpenAI') as mock_openai:
            provider = OpenAIProvider()
            assert provider.api_key == 'test-key'
            mock_openai.assert_called_once()

    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    def test_anthropic_provider_initialization(self):
        """Test Anthropic provider initialization"""
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            provider = AnthropicProvider()
            assert provider.api_key == 'test-key'
            mock_anthropic.assert_called_once()

    @pytest.mark.asyncio
    async def test_grok_provider_generate(self):
        """Test Grok provider generate method"""
        with patch.dict('os.environ', {'GROK_API_KEY': 'test-key'}):
            with patch('openai.AsyncOpenAI') as mock_openai_class:
                # Mock the OpenAI client
                mock_client = Mock()
                mock_openai_class.return_value = mock_client

                # Mock the response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "Test response"
                mock_response.choices[0].finish_reason = "stop"
                mock_response.model = "grok-beta"
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 20
                mock_response.usage.total_tokens = 30

                mock_client.chat.completions.create = Mock(return_value=mock_response)

                provider = GrokProvider()
                messages = [{"role": "user", "content": "Hello"}]
                result = await provider.generate(messages, model="grok-code-fast-1")

                assert result['content'] == "Test response"
                assert result['model'] == "grok-beta"
                assert result['usage']['total_tokens'] == 30

    def test_provider_fallback(self):
        """Test provider fallback when no specific provider matches"""
        manager = LLMProviderManager()

        # Mock providers to be unavailable
        with patch.object(manager.providers['openai'], 'is_available', return_value=False):
            with patch.object(manager.providers['anthropic'], 'is_available', return_value=False):
                with patch.object(manager.providers['grok'], 'is_available', return_value=True):
                    provider = manager.get_provider('unknown-model')
                    assert provider.__class__.__name__ == 'GrokProvider'


if __name__ == "__main__":
    pytest.main([__file__])