#!/usr/bin/env python3
"""
Comprehensive tests for input sanitization utilities
Tests all sanitization functions, security threat detection, and edge cases
"""

import pytest
from shared.sanitization import (
    InputSanitizer,
    sanitize_user_input,
    detect_security_threats
)


class TestInputSanitizer:
    """Test the InputSanitizer class"""

    def setup_method(self):
        """Setup sanitizer instance for each test"""
        self.sanitizer = InputSanitizer()

    def test_sanitize_html_basic(self):
        """Test basic HTML sanitization"""
        html = "<script>alert('xss')</script><p>Safe content</p>"
        result = self.sanitizer.sanitize_html(html)
        # Bleach strips script tags but may leave content
        assert "<script>" not in result
        assert "<p>Safe content</p>" in result

    def test_sanitize_html_with_links(self):
        """Test HTML sanitization with links allowed"""
        html = '<a href="http://example.com">Link</a><img src="x">'
        result = self.sanitizer.sanitize_html(
            html, allow_links=True, allow_images=False
        )
        assert 'href="http://example.com"' in result
        assert "<img" not in result

    def test_sanitize_html_with_images(self):
        """Test HTML sanitization with images allowed"""
        html = '<img src="safe.jpg" alt="test"><script>evil</script>'
        result = self.sanitizer.sanitize_html(
            html, allow_links=False, allow_images=True
        )
        assert 'src="safe.jpg"' in result
        assert 'alt="test"' in result
        assert "<script>" not in result

    def test_sanitize_text_basic(self):
        """Test basic text sanitization"""
        text = "Hello\x00World\x01Test"
        result = self.sanitizer.sanitize_text(text)
        assert result == "HelloWorldTest"
        assert "\x00" not in result

    def test_sanitize_text_with_length_limit(self):
        """Test text sanitization with length limit"""
        text = "This is a very long text that should be truncated"
        result = self.sanitizer.sanitize_text(text, max_length=20)
        # rstrip() removes trailing whitespace, so actual length may be less
        assert len(result) <= 20
        assert result in text[:20]

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization"""
        filename = "../../../etc/passwd"
        result = self.sanitizer.sanitize_filename(filename)
        assert result == "etcpasswd"
        assert ".." not in result
        assert "/" not in result

    def test_sanitize_filename_invalid(self):
        """Test filename sanitization with invalid input"""
        assert self.sanitizer.sanitize_filename("") == "unnamed_file"
        assert self.sanitizer.sanitize_filename(None) == "unnamed_file"

    def test_sanitize_filename_long(self):
        """Test filename sanitization with long names"""
        long_name = "a" * 300 + ".txt"
        result = self.sanitizer.sanitize_filename(long_name)
        assert len(result) <= 255
        assert result.endswith(".txt")

    def test_sanitize_url_valid(self):
        """Test URL sanitization with valid URLs"""
        url = "http://example.com/path?param=value#fragment"
        result = self.sanitizer.sanitize_url(url)
        assert result == "http://example.com/path?param=value#fragment"

    def test_sanitize_url_invalid_scheme(self):
        """Test URL sanitization with invalid schemes"""
        assert self.sanitizer.sanitize_url("javascript:alert('xss')") is None
        assert self.sanitizer.sanitize_url("ftp://example.com") is None

    def test_sanitize_url_malformed(self):
        """Test URL sanitization with malformed URLs"""
        assert self.sanitizer.sanitize_url("not-a-url") is None
        assert self.sanitizer.sanitize_url("") is None

    def test_detect_sql_injection(self):
        """Test SQL injection detection"""
        # Test patterns that should match the regex
        assert self.sanitizer.detect_sql_injection("; SELECT * FROM users")
        assert self.sanitizer.detect_sql_injection("UNION SELECT password")
        assert self.sanitizer.detect_sql_injection(
            "<script>alert('xss')</script>"
        )
        # Test safe content
        assert not self.sanitizer.detect_sql_injection("Hello World")

    def test_sanitize_dict_basic(self):
        """Test dictionary sanitization"""
        data = {
            "name": "<b>John</b>",
            "description": "Safe text",
            "nested": {"html": "<script>evil</script>"}
        }
        result = self.sanitizer.sanitize_dict(data, ["html"])
        # HTML field gets HTML sanitized (tags should be preserved if allowed)
        assert result["name"] == "<b>John</b>"
        assert result["description"] == "Safe text"
        # Nested dict with html field should have script content removed
        assert "<script>" not in result["nested"]["html"]

    def test_sanitize_list_basic(self):
        """Test list sanitization"""
        data = ["<b>Item 1</b>", "<script>evil</script>", "Safe item"]
        result = self.sanitizer.sanitize_list(data)
        # List sanitization only applies text sanitization to strings
        assert result[0] == "<b>Item 1</b>"  # HTML tags preserved
        # Text sanitization doesn't remove HTML tags, only control chars
        assert result[1] == "<script>evil</script>"
        assert result[2] == "Safe item"

    def test_validate_email_format(self):
        """Test email format validation"""
        assert self.sanitizer.validate_email_format("user@example.com")
        assert not self.sanitizer.validate_email_format("invalid-email")
        assert not self.sanitizer.validate_email_format("")

    def test_validate_phone_format(self):
        """Test phone number format validation"""
        assert self.sanitizer.validate_phone_format("1234567890")
        assert self.sanitizer.validate_phone_format("+123456789012")
        assert not self.sanitizer.validate_phone_format("123")  # Too short


class TestSanitizeUserInput:
    """Test the sanitize_user_input convenience function"""

    def test_sanitize_text_input(self):
        """Test text input sanitization"""
        result = sanitize_user_input("Hello\x00World", "text")
        assert result == "HelloWorld"

    def test_sanitize_html_input(self):
        """Test HTML input sanitization"""
        result = sanitize_user_input("<script>evil</script>Safe", "html")
        # Bleach strips script tags but may leave content
        assert "<script>" not in result
        assert "Safe" in result

    def test_sanitize_email_input(self):
        """Test email input sanitization"""
        result = sanitize_user_input("USER@EXAMPLE.COM", "email")
        assert result == "user@example.com"

    def test_sanitize_url_input(self):
        """Test URL input sanitization"""
        result = sanitize_user_input("http://example.com", "url")
        assert result == "http://example.com"

    def test_sanitize_filename_input(self):
        """Test filename input sanitization"""
        result = sanitize_user_input("../../../file.txt", "filename")
        assert result == "file.txt"

    def test_sanitize_dict_input(self):
        """Test dictionary input sanitization"""
        data = {"name": "<b>John</b>", "email": "john@example.com"}
        result = sanitize_user_input(data, "text")
        # Dict sanitization applies text sanitization to all string values
        assert result["name"] == "<b>John</b>"  # HTML preserved in text mode
        assert result["email"] == "john@example.com"

    def test_sanitize_list_input(self):
        """Test list input sanitization"""
        data = ["<b>Item</b>", "Safe"]
        result = sanitize_user_input(data, "html")
        assert result == ["<b>Item</b>", "Safe"]

    def test_sanitize_invalid_input(self):
        """Test sanitization with invalid input types"""
        assert sanitize_user_input(123, "text") == 123
        assert sanitize_user_input(None, "text") is None


class TestDetectSecurityThreats:
    """Test security threat detection"""

    def test_detect_sql_injection_threats(self):
        """Test SQL injection threat detection"""
        threats = detect_security_threats("; SELECT * FROM users")
        assert threats["sql_injection"]

    def test_detect_xss_threats(self):
        """Test XSS threat detection"""
        threats = detect_security_threats("<script>alert('xss')</script>")
        assert threats["xss_patterns"]

        threats = detect_security_threats("javascript:evil()")
        assert threats["xss_patterns"]

    def test_detect_path_traversal_threats(self):
        """Test path traversal threat detection"""
        threats = detect_security_threats("../../../etc/passwd")
        assert threats["path_traversal"]

    def test_detect_command_injection_threats(self):
        """Test command injection threat detection"""
        threats = detect_security_threats("; rm -rf /")
        # The pattern requires semicolon + space + command
        assert threats["command_injection"]

    def test_detect_no_threats(self):
        """Test detection with safe content"""
        threats = detect_security_threats("Hello World")
        assert all(not threat for threat in threats.values())

    def test_detect_multiple_threats(self):
        """Test detection of multiple threat types"""
        content = ("../../../etc/passwd; SELECT * FROM users; "
                   "<script>evil</script>")
        threats = detect_security_threats(content)
        assert threats["path_traversal"]
        assert threats["sql_injection"]
        assert threats["xss_patterns"]
        # No command injection since no rm/ls/cat/echo/eval commands
        assert not threats["command_injection"]


class TestEdgeCases:
    """Test edge cases and error handling"""

    def setup_method(self):
        """Setup sanitizer instance"""
        self.sanitizer = InputSanitizer()

    def test_empty_inputs(self):
        """Test handling of empty inputs"""
        assert self.sanitizer.sanitize_html("") == ""
        assert self.sanitizer.sanitize_text("") == ""
        assert self.sanitizer.sanitize_url("") is None
        assert self.sanitizer.sanitize_filename("") == "unnamed_file"

    def test_none_inputs(self):
        """Test handling of None inputs"""
        assert self.sanitizer.sanitize_html(None) == ""
        assert self.sanitizer.sanitize_text(None) == ""
        assert self.sanitizer.sanitize_url(None) is None
        assert self.sanitizer.sanitize_filename(None) == "unnamed_file"

    def test_large_inputs(self):
        """Test handling of very large inputs"""
        large_text = "A" * 10000
        result = self.sanitizer.sanitize_text(large_text, max_length=100)
        assert len(result) == 100

    def test_unicode_inputs(self):
        """Test handling of Unicode inputs"""
        unicode_text = "Hello 世界 🌍"
        result = self.sanitizer.sanitize_text(unicode_text)
        assert result == unicode_text

    def test_special_characters(self):
        """Test handling of special characters"""
        special = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        result = self.sanitizer.sanitize_text(special)
        assert result == special

    def test_nested_structures(self):
        """Test sanitization of deeply nested structures"""
        nested = {
            "level1": {
                "level2": {
                    "level3": ["<script>evil</script>", "safe"]
                }
            }
        }
        result = self.sanitizer.sanitize_dict(nested)
        # List sanitization applies text sanitization (no HTML removal)
        # HTML tags are preserved in text mode
        script_item = result["level1"]["level2"]["level3"][0]
        assert script_item == "<script>evil</script>"
        assert result["level1"]["level2"]["level3"][1] == "safe"


if __name__ == "__main__":
    pytest.main([__file__])