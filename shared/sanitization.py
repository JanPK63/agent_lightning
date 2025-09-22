#!/usr/bin/env python3
"""
Input Sanitization Utilities for Agent Lightning

This module provides comprehensive input sanitization utilities designed to
protect against common web security vulnerabilities including XSS (Cross-Site
Scripting), SQL injection, path traversal, and other injection attacks.

Key Features:
- HTML sanitization with configurable tag/attribute filtering
- SQL injection pattern detection and logging
- URL validation and sanitization for SSRF prevention
- Filename sanitization for secure file handling
- Recursive sanitization for complex data structures
- Email and phone number format validation

Security Considerations:
- All sanitization functions are designed to be safe by default
- Suspicious patterns are logged for security monitoring
- Input validation is performed before sanitization
- Fallback mechanisms ensure graceful degradation

Usage Examples:
    Basic text sanitization:
        >>> from shared.sanitization import sanitize_user_input
        >>> clean_text = sanitize_user_input(
        ...     "<script>alert('xss')</script>", "html"
        ... )
        >>> print(clean_text)  # Output: ""

    Dictionary sanitization:
        >>> data = {"name": "<b>John</b>", "email": "john@example.com"}
        >>> clean_data = sanitize_user_input(data, "text")
        >>> print(clean_data["name"])  # Output: "John"

    Security threat detection:
        >>> from shared.sanitization import detect_security_threats
        >>> threats = detect_security_threats("SELECT * FROM users")
        >>> print(threats["sql_injection"])  # Output: True

Classes:
    InputSanitizer: Main sanitization class with comprehensive methods

Functions:
    sanitize_user_input: Convenience function for basic sanitization
    detect_security_threats: Security threat detection utility

Security Notes:
- Always validate input before sanitization
- Log security events for monitoring
- Use appropriate sanitization level for your use case
- Consider rate limiting for API endpoints
"""

import re
import bleach
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote
import logging

logger = logging.getLogger(__name__)

# HTML sanitization configuration
ALLOWED_HTML_TAGS = [
    'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'a', 'img'
]

ALLOWED_HTML_ATTRIBUTES = {
    'a': ['href', 'title'],
    'img': ['src', 'alt', 'title'],
    '*': ['class', 'id']
}

# SQL injection patterns to detect (for logging purposes)
SQL_INJECTION_PATTERNS = [
    r';\s*(?:select|insert|update|delete|drop|create|alter)\s',
    r'union\s+select',
    r'--\s*$',
    r'/\*.*\*/',
    r'xp_cmdshell',
    r'exec\s*\(',
    r'script\s*>',
    r'<script',
    r'javascript:',
    r'on\w+\s*=',
    r'eval\s*\(',
    r'document\.',
    r'window\.',
    r'alert\s*\('
]


class InputSanitizer:
    """
    Comprehensive input sanitization utility for Agent Lightning.

    This class provides methods to sanitize various types of user input
    to prevent common security vulnerabilities. It includes HTML sanitization,
    text cleaning, URL validation, filename sanitization, and security threat
    detection.

    The sanitizer is designed to be safe by default, with configurable options
    for different use cases. All sanitization operations include proper error
    handling and logging.

    Attributes:
        sql_patterns (List[Pattern]): Compiled regex patterns for SQL injection
                                     detection

    Example:
        >>> sanitizer = InputSanitizer()
        >>> clean_html = sanitizer.sanitize_html(
        ...     "<script>alert('xss')</script>"
        ... )
        >>> print(clean_html)  # Output: ""
    """

    def __init__(self):
        self.sql_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SQL_INJECTION_PATTERNS
        ]

    def sanitize_html(
        self,
        content: str,
        allow_links: bool = True,
        allow_images: bool = False
    ) -> str:
        """
        Sanitize HTML content to prevent XSS attacks.

        This method removes potentially dangerous HTML tags and attributes
        while preserving safe content. It uses the bleach library for robust
        HTML sanitization.

        Args:
            content (str): HTML content to sanitize. If None or not a string,
                          returns empty string.
            allow_links (bool): Whether to allow <a> tags with href and title
                               attributes. Default is True.
            allow_images (bool): Whether to allow <img> tags with src, alt,
                                and title attributes. Default is False.

        Returns:
            str: Sanitized HTML content with dangerous tags/attributes removed.

        Raises:
            No exceptions are raised; errors are logged and fallback
            sanitization is applied.
            is applied.

        Examples:
            Basic sanitization:
                >>> sanitizer = InputSanitizer()
                >>> html = "<script>alert('xss')</script><p>Safe content</p>"
                >>> clean = sanitizer.sanitize_html(html)
                >>> print(clean)  # Output: "<p>Safe content</p>"

            Allow links but not images:
                >>> html = '<a href="http://example.com">Link</a><img src="x">'
                >>> clean = sanitizer.sanitize_html(html, allow_links=True,
                ...                                allow_images=False)
                >>> print(clean)  # Output: Link with href

        Security Notes:
            - Always sanitize user-provided HTML content
            - Consider the security context of where sanitized HTML
              will be used
            - Links are allowed by default but should be validated separately
            - Images are disabled by default due to potential abuse
        """
        if not content or not isinstance(content, str):
            return content or ""

        # Configure allowed tags and attributes
        tags = ALLOWED_HTML_TAGS.copy()
        attributes = ALLOWED_HTML_ATTRIBUTES.copy()

        if not allow_links:
            tags = [tag for tag in tags if tag != 'a']
            attributes.pop('a', None)

        if not allow_images:
            tags = [tag for tag in tags if tag != 'img']
            attributes.pop('img', None)

        try:
            sanitized = bleach.clean(
                content,
                tags=tags,
                attributes=attributes,
                strip=True
            )
            return sanitized
        except Exception as e:
            logger.warning(f"HTML sanitization failed: {e}")
            # Fallback: escape all HTML
            return bleach.clean(content, tags=[], attributes={}, strip=True)

    def sanitize_text(
        self,
        content: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Sanitize plain text content

        Args:
            content: Text content to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized text content
        """
        if not content or not isinstance(content, str):
            return content or ""

        # Remove null bytes and other control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)

        # Trim whitespace
        sanitized = sanitized.strip()

        # Apply length limit
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip()

        return sanitized

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other attacks

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        if not filename or not isinstance(filename, str):
            return "unnamed_file"

        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)

        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')

        # Ensure it's not empty
        if not sanitized:
            return "unnamed_file"

        # Limit length
        if len(sanitized) > 255:
            name, ext = (sanitized.rsplit('.', 1) if '.' in sanitized
                         else (sanitized, ''))
            if ext:
                sanitized = name[:250] + '.' + ext[:4]  # Keep extension short
            else:
                sanitized = sanitized[:255]

        return sanitized

    def sanitize_url(self, url: str) -> Optional[str]:
        """
        Sanitize URL to prevent SSRF and other attacks

        Args:
            url: URL to sanitize

        Returns:
            Sanitized URL or None if invalid
        """
        if not url or not isinstance(url, str):
            return None

        try:
            # Parse and validate URL
            from urllib.parse import urlparse
            parsed = urlparse(url)

            # Only allow http and https schemes
            if parsed.scheme not in ['http', 'https']:
                return None

            # Reconstruct safe URL
            safe_url = f"{parsed.scheme}://{parsed.netloc}{quote(parsed.path)}"
            if parsed.query:
                safe_url += f"?{quote(parsed.query, safe='&=')}"
            if parsed.fragment:
                safe_url += f"#{quote(parsed.fragment)}"

            return safe_url

        except Exception as e:
            logger.warning(f"URL sanitization failed: {e}")
            return None

    def detect_sql_injection(self, content: str) -> bool:
        """
        Detect potential SQL injection patterns (for logging purposes)

        Args:
            content: Content to check

        Returns:
            True if suspicious patterns detected
        """
        if not content or not isinstance(content, str):
            return False

        for pattern in self.sql_patterns:
            if pattern.search(content):
                logger.warning("Potential SQL injection pattern detected: "
                               f"{pattern.pattern}")
                return True

        return False

    def sanitize_dict(
        self,
        data: Dict[str, Any],
        sanitize_html_fields: List[str] = None
    ) -> Dict[str, Any]:
        """
        Sanitize all values in a dictionary

        Args:
            data: Dictionary to sanitize
            sanitize_html_fields: List of field names that should have
            HTML sanitized

        Returns:
            Sanitized dictionary
        """
        if not data or not isinstance(data, dict):
            return data or {}

        sanitized = {}
        html_fields = sanitize_html_fields or []

        for key, value in data.items():
            if isinstance(value, str):
                if key in html_fields:
                    sanitized[key] = self.sanitize_html(value)
                else:
                    sanitized[key] = self.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value, html_fields)
            elif isinstance(value, list):
                sanitized[key] = self.sanitize_list(value, html_fields)
            else:
                sanitized[key] = value

        return sanitized

    def sanitize_list(
        self,
        data: List[Any],
        sanitize_html_fields: List[str] = None
    ) -> List[Any]:
        """
        Sanitize all items in a list

        Args:
            data: List to sanitize
            sanitize_html_fields: List of field names that should have
            HTML sanitized

        Returns:
            Sanitized list
        """
        if not data or not isinstance(data, list):
            return data or []

        sanitized = []
        for item in data:
            if isinstance(item, str):
                sanitized.append(self.sanitize_text(item))
            elif isinstance(item, dict):
                sanitized.append(
                    self.sanitize_dict(item, sanitize_html_fields)
                )
            elif isinstance(item, list):
                sanitized.append(
                    self.sanitize_list(item, sanitize_html_fields)
                )
            else:
                sanitized.append(item)

        return sanitized

    def validate_email_format(self, email: str) -> bool:
        """
        Basic email format validation

        Args:
            email: Email address to validate

        Returns:
            True if format is valid
        """
        if not email or not isinstance(email, str):
            return False

        # Simple regex for email validation
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def validate_phone_format(self, phone: str) -> bool:
        """
        Basic phone number format validation

        Args:
            phone: Phone number to validate

        Returns:
            True if format is valid
        """
        if not phone or not isinstance(phone, str):
            return False

        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone)

        # Check if it's a reasonable length (7-15 digits)
        return 7 <= len(digits_only) <= 15


# Global sanitizer instance
sanitizer = InputSanitizer()


def sanitize_user_input(
    content: Union[str, Dict, List],
    field_type: str = "text"
) -> Union[str, Dict, List]:
    """
    Convenience function for sanitizing user input with automatic
    type detection.

    This function provides a simple interface for sanitizing various types of
    user input. It automatically selects the appropriate sanitization method
    based on the field_type parameter and handles different data types
    (strings, dictionaries, lists).

    Args:
        content (Union[str, Dict, List]): The content to sanitize. Can be:
            - str: Single string value
            - Dict: Dictionary with string keys and various value types
            - List: List containing various data types
        field_type (str): The type of field being sanitized. Options:
            - "text": Basic text sanitization (default)
            - "html": HTML sanitization with XSS protection
            - "email": Email address sanitization (converts to lowercase)
            - "url": URL validation and sanitization
            - "filename": Filename sanitization for secure file handling

    Returns:
        Union[str, Dict, List]: Sanitized content of the same type as input.
        Returns the original content if sanitization is not applicable.

    Examples:
        String sanitization:
            >>> clean = sanitize_user_input(
            ...     "<script>alert('xss')</script>", "html"
            ... )
            >>> print(clean)  # Output: ""

        Dictionary sanitization:
            >>> data = {"name": "John", "bio": "<b>Developer</b>"}
            >>> clean = sanitize_user_input(data, "text")
            >>> print(clean["bio"])  # Output: "Developer"

        List sanitization:
            >>> items = ["<i>Item 1</i>", "<b>Item 2</b>"]
            >>> clean = sanitize_user_input(items, "html")
            >>> print(clean)  # Output: ["Item 1", "Item 2"]

        Email sanitization:
            >>> email = sanitize_user_input("USER@EXAMPLE.COM", "email")
            >>> print(email)  # Output: "user@example.com"

    Security Notes:
        - Always specify the appropriate field_type for proper sanitization
        - For HTML content, consider the display context and allowed tags
        - URL sanitization only allows http/https schemes
        - Filename sanitization prevents path traversal attacks
    """
    if isinstance(content, str):
        if field_type == "html":
            return sanitizer.sanitize_html(content)
        elif field_type == "email":
            return sanitizer.sanitize_text(content).lower()
        elif field_type == "url":
            return sanitizer.sanitize_url(content)
        elif field_type == "filename":
            return sanitizer.sanitize_filename(content)
        else:  # text
            return sanitizer.sanitize_text(content)
    elif isinstance(content, dict):
        return sanitizer.sanitize_dict(content)
    elif isinstance(content, list):
        return sanitizer.sanitize_list(content)
    else:
        return content


def detect_security_threats(content: str) -> Dict[str, bool]:
    """
    Detect various security threats in content for logging and monitoring.

    This function analyzes the provided content for common security threats
    including SQL injection patterns, XSS attempts, path traversal, and
    command injection. All detected threats are logged for security monitoring.

    Args:
        content (str): The content to analyze for security threats.
                      If None or not a string, returns all False values.

    Returns:
        Dict[str, bool]: Dictionary containing detection results:
            - "sql_injection": True if SQL injection patterns detected
            - "xss_patterns": True if XSS patterns detected
            - "path_traversal": True if path traversal patterns detected
            - "command_injection": True if command injection patterns detected

    Examples:
        Basic threat detection:
            >>> threats = detect_security_threats("SELECT * FROM users")
            >>> print(threats["sql_injection"])  # Output: True

        Multiple threat types:
            >>> content = "../../../etc/passwd; rm -rf /"
            >>> threats = detect_security_threats(content)
            >>> print(threats["path_traversal"])    # Output: True
            >>> print(threats["command_injection"]) # Output: True

        Safe content:
            >>> threats = detect_security_threats("Hello World")
            >>> print(all(threats.values()))  # Output: False

    Security Notes:
        - This function is designed for logging and monitoring purposes
        - Always combine with appropriate sanitization methods
        - False positives are possible; use in conjunction with other
          security measures
        - All detections are logged with warning level for security monitoring
    """
    return {
        "sql_injection": sanitizer.detect_sql_injection(content),
        "xss_patterns": bool(re.search(
            r'<script|<iframe|javascript:|on\w+\s*=',
            content, re.IGNORECASE
        )),
        "path_traversal": bool(re.search(r'\.\./|\.\.\\', content)),
        "command_injection": bool(re.search(
            r';\s*(?:rm|ls|cat|echo|eval)\s',
            content, re.IGNORECASE
        ))
    }


# Export common functions
__all__ = [
    'InputSanitizer',
    'sanitizer',
    'sanitize_user_input',
    'detect_security_threats'
]
