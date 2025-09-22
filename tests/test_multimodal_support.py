#!/usr/bin/env python3
"""
Tests for multi-modal support in Agent Lightning
"""

import pytest
from agentlightning.types import TextContent, ImageContent, AudioContent, VideoContent, ContentType
from agentlightning.llm_providers import convert_multimodal_to_messages


class TestMultiModalSupport:
    """Test multi-modal content handling"""

    def test_text_content_creation(self):
        """Test creating text content"""
        text_content = TextContent(text="Hello world")
        assert text_content.content_type == ContentType.TEXT
        assert text_content.text == "Hello world"

    def test_image_content_creation(self):
        """Test creating image content"""
        image_data = b"fake_image_data"
        image_content = ImageContent(image_data=image_data, format="png")
        assert image_content.content_type == ContentType.IMAGE
        assert image_content.image_data == image_data
        assert image_content.format == "png"

    def test_audio_content_creation(self):
        """Test creating audio content"""
        audio_data = b"fake_audio_data"
        audio_content = AudioContent(audio_data=audio_data, format="mp3", duration=10.5)
        assert audio_content.content_type == ContentType.AUDIO
        assert audio_content.audio_data == audio_data
        assert audio_content.format == "mp3"
        assert audio_content.duration == 10.5

    def test_video_content_creation(self):
        """Test creating video content"""
        video_data = b"fake_video_data"
        video_content = VideoContent(video_data=video_data, format="mp4", duration=30.0)
        assert video_content.content_type == ContentType.VIDEO
        assert video_content.video_data == video_data
        assert video_content.format == "mp4"
        assert video_content.duration == 30.0

    def test_convert_text_to_messages(self):
        """Test converting text to messages"""
        messages = convert_multimodal_to_messages("Hello world")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello world"

    def test_convert_text_content_to_messages(self):
        """Test converting TextContent to messages"""
        text_content = TextContent(text="Hello from TextContent")
        messages = convert_multimodal_to_messages(text_content)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "Hello from TextContent"

    def test_convert_image_content_to_messages(self):
        """Test converting ImageContent to messages"""
        import base64
        image_data = b"fake_image_data"
        image_content = ImageContent(image_data=image_data, format="png")
        messages = convert_multimodal_to_messages(image_content)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 1
        content_item = messages[0]["content"][0]
        assert content_item["type"] == "image_url"
        assert "data:image/png;base64," in content_item["image_url"]["url"]

    def test_convert_mixed_content_to_messages(self):
        """Test converting mixed multi-modal content to messages"""
        text_content = TextContent(text="Describe this image:")
        image_content = ImageContent(image_data=b"fake_image", format="jpeg")
        audio_content = AudioContent(audio_data=b"fake_audio", format="wav", duration=5.0)

        messages = convert_multimodal_to_messages([text_content, image_content, audio_content])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert len(content) == 3

        # Check text content
        assert content[0]["type"] == "text"
        assert "Describe this image:" in content[0]["text"]

        # Check image content
        assert content[1]["type"] == "image_url"

        # Check audio content (converted to text description)
        assert content[2]["type"] == "text"
        assert "[Audio content:" in content[2]["text"]
        assert "wav" in content[2]["text"]
        assert "5.0s" in content[2]["text"]

    @pytest.mark.asyncio
    async def test_llm_provider_with_multimodal(self):
        """Test LLM provider can handle multi-modal input"""
        from agentlightning.llm_providers import llm_manager

        # This test would require actual API keys and models
        # For now, just test that the conversion works
        text_content = TextContent(text="Hello")
        image_content = ImageContent(image_data=b"fake", format="png")

        # Test that the manager accepts multi-modal content
        # (This will fail without API keys, but tests the interface)
        try:
            await llm_manager.generate("gpt-4o", [text_content, image_content])
        except Exception as e:
            # Expected to fail without API keys, but should not fail due to type issues
            assert "No available provider" not in str(e)  # Should find the provider
            assert "API key" in str(e) or "client not initialized" in str(e)


if __name__ == "__main__":
    pytest.main([__file__])