"""
Unit tests for PII Masking Utility

Tests the mask_pii function and its components for masking emails,
phone numbers, names, and IP addresses.
"""

import io
import logging
from shared.pii_masker import (
    mask_pii,
    mask_email,
    mask_phone,
    mask_name,
    mask_ip,
    PIIMaskingFilter
)


class TestMaskEmail:
    """Test email masking functionality."""

    def test_simple_email(self):
        """Test masking a simple email address."""
        email = "test@example.com"
        expected = "t**t@e*****e.com"
        result = mask_email(type('Match', (), {'group': lambda self: email})())
        assert result == expected

    def test_short_username(self):
        """Test masking email with short username."""
        email = "a@b.com"
        expected = "a@b.com"
        result = mask_email(type('Match', (), {'group': lambda self: email})())
        assert result == expected

    def test_long_domain(self):
        """Test masking email with long domain."""
        email = "user@verylongdomainname.com"
        expected = "u**r@v****************e.com"
        result = mask_email(type('Match', (), {'group': lambda self: email})())
        assert result == expected


class TestMaskPhone:
    """Test phone number masking functionality."""

    def test_us_phone_with_dashes(self):
        """Test masking US phone number with dashes."""
        phone = "123-456-7890"
        expected = "***-***-****"
        result = mask_phone(type('Match', (), {'group': lambda self: phone})())
        assert result == expected

    def test_us_phone_with_dots(self):
        """Test masking US phone number with dots."""
        phone = "123.456.7890"
        expected = "***.***.****"
        result = mask_phone(type('Match', (), {'group': lambda self: phone})())
        assert result == expected


class TestMaskName:
    """Test name masking functionality."""

    def test_three_letter_name(self):
        """Test masking a three-letter name."""
        name = "John"
        expected = "J**n"
        result = mask_name(type('Match', (), {'group': lambda self: name})())
        assert result == expected

    def test_two_letter_name(self):
        """Test masking a two-letter name."""
        name = "Jo"
        expected = "J*"
        result = mask_name(type('Match', (), {'group': lambda self: name})())
        assert result == expected

    def test_single_letter_name(self):
        """Test masking a single-letter name."""
        name = "J"
        expected = "J"
        result = mask_name(type('Match', (), {'group': lambda self: name})())
        assert result == expected


class TestMaskIP:
    """Test IP address masking functionality."""

    def test_standard_ip(self):
        """Test masking a standard IP address."""
        ip = "192.168.1.1"
        expected = "192.***.*.1"
        result = mask_ip(type('Match', (), {'group': lambda self: ip})())
        assert result == expected

    def test_ip_with_single_digits(self):
        """Test masking IP with single digit octets."""
        ip = "1.2.3.4"
        expected = "1.*.*.4"
        result = mask_ip(type('Match', (), {'group': lambda self: ip})())
        assert result == expected


class TestMaskPII:
    """Test the main mask_pii function."""

    def test_mask_all_pii(self):
        """Test masking all types of PII in text."""
        text = ("Contact John at john@example.com or call 123-456-7890. "
                "His IP is 192.168.1.1.")
        expected = ("C*****t J**n at j**n@e*****e.com or call ***-***-****. "
                    "H*s IP is 192.***.*.1.")
        result = mask_pii(text)
        assert result == expected

    def test_mask_only_email(self):
        """Test masking only emails."""
        text = "Email: test@example.com and phone: 123-456-7890"
        expected = "Email: t**t@e*****e.com and phone: 123-456-7890"
        config = {'email': True, 'phone': False, 'name': False, 'ip': False}
        result = mask_pii(text, config)
        assert result == expected

    def test_mask_only_phone(self):
        """Test masking only phone numbers."""
        text = "Call 123-456-7890 or email test@example.com"
        expected = "Call ***-***-**** or email test@example.com"
        config = {'email': False, 'phone': True, 'name': False, 'ip': False}
        result = mask_pii(text, config)
        assert result == expected

    def test_mask_only_name(self):
        """Test masking only names."""
        text = "Hello John, your email is john@example.com"
        expected = "H***o J**n, your email is john@example.com"
        config = {'email': False, 'phone': False, 'name': True, 'ip': False}
        result = mask_pii(text, config)
        assert result == expected

    def test_mask_only_ip(self):
        """Test masking only IP addresses."""
        text = "Server at 192.168.1.1 and email test@example.com"
        expected = "Server at 192.***.*.1 and email test@example.com"
        config = {'email': False, 'phone': False, 'name': False, 'ip': True}
        result = mask_pii(text, config)
        assert result == expected

    def test_no_pii(self):
        """Test text with no PII."""
        text = "This is a normal sentence without any PII."
        expected = "T**s is a normal sentence without any PII."
        result = mask_pii(text)
        assert result == expected

    def test_empty_config(self):
        """Test with empty config (should mask all)."""
        text = "Email: test@example.com"
        expected = "E***l: t**t@e*****e.com"
        result = mask_pii(text, {})
        assert result == expected

    def test_none_config(self):
        """Test with None config (should mask all)."""
        text = "Email: test@example.com"
        expected = "E***l: t**t@e*****e.com"
        result = mask_pii(text, None)
        assert result == expected

    def test_case_insensitive_email(self):
        """Test email masking is case insensitive."""
        text = "Email: Test@Example.Com"
        expected = "E***l: T**t@E*****e.C*m"
        result = mask_pii(text)
        assert result == expected


class TestPIIIntegration:
    """Integration tests for PII masking in the logging system."""

    def test_pii_masking_in_logger(self):
        """Test PII masking in logger output using PIIMaskingFilter."""
        # Create a logger and handler
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)

        # Create StringIO to capture log output
        string_io = io.StringIO()
        handler = logging.StreamHandler(string_io)
        handler.setLevel(logging.INFO)

        # Add PII masking filter to the handler
        pii_filter = PIIMaskingFilter()
        handler.addFilter(pii_filter)

        # Set a simple formatter
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

        # Log a message containing various types of PII
        test_message = ("User John logged in from 192.168.1.1 with email "
                        "john@example.com and phone 123-456-7890")
        logger.info(test_message)

        # Get the captured log output
        log_output = string_io.getvalue()

        # Verify that PII has been masked in the output
        # Name: John -> J**n
        assert "J**n" in log_output
        # IP: 192.168.1.1 -> 192.***.*.1
        assert "192.***.*.1" in log_output
        # Email: john@example.com -> j**n@e*****e.com
        assert "j**n@e*****e.com" in log_output
        # Phone: 123-456-7890 -> ***-***-****
        assert "***-***-****" in log_output

        # Ensure the original PII is not present in the output
        assert "John" not in log_output
        assert "192.168.1.1" not in log_output
        assert "john@example.com" not in log_output
        assert "123-456-7890" not in log_output