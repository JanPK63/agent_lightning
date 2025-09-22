"""
PII Masking Utility

This module provides a configurable function to mask personally identifiable
information (PII) in text strings using regex patterns. Supported PII types
include emails, phone numbers, names, and IP addresses.

The masking is designed to be partial to maintain readability while protecting
sensitive data.
"""

import logging
import os
import re
from typing import Dict, Optional

# Regex patterns for PII detection
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_PATTERN = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'  # Basic US phone format
NAME_PATTERN = r'\b[A-Z][a-z]+\b'  # Simple capitalized word (potential name)
IP_PATTERN = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'


def mask_email(match: re.Match) -> str:
    """Mask an email address by replacing middle characters of username
    and domain with asterisks."""
    email = match.group()
    if '@' not in email:
        return email
    user, domain = email.split('@', 1)
    if len(user) > 2:
        masked_user = user[0] + '*' * (len(user) - 2) + user[-1]
    else:
        masked_user = (user[0] + '*' * (len(user) - 1)
                       if len(user) > 1 else user)

    domain_parts = domain.split('.')
    if len(domain_parts) >= 2:
        main_domain = domain_parts[0]
        if len(main_domain) > 2:
            masked_domain = (main_domain[0] +
                             '*' * (len(main_domain) - 2) +
                             main_domain[-1])
        else:
            masked_domain = (main_domain[0] + '*' * (len(main_domain) - 1)
                             if len(main_domain) > 1 else main_domain)
        domain_parts[0] = masked_domain
        masked_domain_full = '.'.join(domain_parts)
    else:
        masked_domain_full = domain

    return masked_user + '@' + masked_domain_full


def mask_phone(match: re.Match) -> str:
    """Mask a phone number by replacing all digits with asterisks."""
    phone = match.group()
    return re.sub(r'\d', '*', phone)


def mask_name(match: re.Match) -> str:
    """Mask a name by replacing middle characters with asterisks."""
    name = match.group()
    if len(name) > 2:
        return name[0] + '*' * (len(name) - 2) + name[-1]
    elif len(name) == 2:
        return name[0] + '*'
    else:
        return name


def mask_ip(match: re.Match) -> str:
    """Mask an IP address by replacing middle octets with asterisks."""
    ip = match.group()
    parts = ip.split('.')
    if len(parts) == 4:
        parts[1] = '*' * len(parts[1])  # Mask second octet
        parts[2] = '*' * len(parts[2])  # Mask third octet
    return '.'.join(parts)


def mask_pii(text: str, config: Optional[Dict[str, bool]] = None) -> str:
    """
    Mask PII in the given text based on the provided configuration.

    Args:
        text: The input text to mask.
        config: Dictionary to enable/disable masking for each PII type.
                Keys: 'email', 'phone', 'name', 'ip'. Default: all True.

    Returns:
        The text with PII masked according to the config.
    """
    if config is None:
        config = {'email': True, 'phone': True, 'name': True, 'ip': True}

    # Apply masking in order: email, phone, name, ip
    if config.get('email', True):
        text = re.sub(EMAIL_PATTERN, mask_email, text, flags=re.IGNORECASE)

    if config.get('phone', True):
        text = re.sub(PHONE_PATTERN, mask_phone, text)

    if config.get('name', True):
        text = re.sub(NAME_PATTERN, mask_name, text)

    if config.get('ip', True):
        text = re.sub(IP_PATTERN, mask_ip, text)

    return text


class PIIMaskingFilter(logging.Filter):
    """
    Logging filter that masks PII in log messages.

    This filter can be added to logging handlers to automatically mask
    personally identifiable information in log messages before they are
    output.
    """

    def __init__(self, config: Optional[Dict[str, bool]] = None,
                 enabled: Optional[bool] = None):
        """
        Initialize the PII masking filter.

        Args:
            config: Dictionary to enable/disable masking for each PII type.
                   Keys: 'email', 'phone', 'name', 'ip'. Default: all True.
            enabled: Whether to enable PII masking. If None, checks
                    AGENT_LIGHTNING_PII_MASKING_ENABLED environment variable.
                    Default: True if env var not set.
        """
        super().__init__()
        self.config = config or {'email': True, 'phone': True,
                                 'name': True, 'ip': True}
        if enabled is None:
            env_value = os.getenv('AGENT_LIGHTNING_PII_MASKING_ENABLED',
                                  'true').lower()
            self.enabled = env_value in ('true', '1', 'yes', 'on')
        else:
            self.enabled = enabled

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter the log record, masking PII in the message if enabled.

        Args:
            record: The log record to filter.

        Returns:
            Always returns True to allow the record to be logged.
        """
        if (self.enabled and hasattr(record, 'msg') and
                isinstance(record.msg, str)):
            record.msg = mask_pii(record.msg, self.config)
        return True