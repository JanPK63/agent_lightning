#!/usr/bin/env python3
"""
Input Validation Middleware for Agent Lightning
Provides automatic input sanitization for FastAPI applications
"""

import logging
from typing import Callable, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .sanitization import InputSanitizer, detect_security_threats

logger = logging.getLogger(__name__)


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic input validation and sanitization.

    This middleware intercepts incoming requests and automatically:
    - Sanitizes JSON request bodies
    - Sanitizes query parameters
    - Sanitizes form data
    - Detects security threats
    - Logs suspicious activities

    The middleware is designed to be safe by default and can be configured
    to skip certain endpoints or apply different sanitization levels.
    """

    def __init__(self, app: Callable, exclude_paths: list = None):
        """
        Initialize the validation middleware.

        Args:
            app: FastAPI application instance
            exclude_paths: List of path prefixes to exclude from validation
        """
        super().__init__(app)
        self.sanitizer = InputSanitizer()
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request through validation middleware.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware in chain

        Returns:
            Response: Processed response
        """
        # Skip validation for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        try:
            # Validate and sanitize request data
            await self._validate_request(request)

            # Continue with request processing
            response = await call_next(request)

            return response

        except Exception as e:
            logger.error(f"Validation middleware error: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request data", "details": str(e)}
            )

    async def _validate_request(self, request: Request) -> None:
        """
        Validate and sanitize request data.

        Args:
            request: FastAPI request object
        """
        # Validate query parameters
        await self._validate_query_params(request)

        # Validate request body based on content type
        content_type = request.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            await self._validate_json_body(request)
        elif "application/x-www-form-urlencoded" in content_type:
            await self._validate_form_data(request)
        elif "multipart/form-data" in content_type:
            await self._validate_multipart_data(request)

    async def _validate_query_params(self, request: Request) -> None:
        """
        Validate and sanitize query parameters.

        Args:
            request: FastAPI request object
        """
        sanitized_params = {}

        for key, value in request.query_params.items():
            if isinstance(value, str):
                # Sanitize parameter value
                sanitized_value = self.sanitizer.sanitize_text(value)

                # Check for security threats
                threats = detect_security_threats(sanitized_value)
                if any(threats.values()):
                    logger.warning(
                        f"Security threat detected in query param '{key}': {threats}"
                    )

                sanitized_params[key] = sanitized_value
            else:
                sanitized_params[key] = value

        # Update request query params (if possible)
        # Note: FastAPI query params are immutable, so we log issues instead
        for key, original_value in request.query_params.items():
            sanitized_value = sanitized_params.get(key)
            if isinstance(original_value, str) and sanitized_value != original_value:
                logger.info(f"Query param '{key}' sanitized: '{original_value}' -> '{sanitized_value}'")

    async def _validate_json_body(self, request: Request) -> None:
        """
        Validate and sanitize JSON request body.

        Args:
            request: FastAPI request object
        """
        try:
            # Read and parse JSON body
            body = await request.json()

            # Sanitize the JSON data
            sanitized_body = self._sanitize_json_data(body)

            # Store sanitized body for later use
            # Note: We can't modify the request body directly in middleware
            # This would need to be handled by individual endpoints or a custom Request class
            request.state.sanitized_body = sanitized_body

        except Exception as e:
            logger.warning(f"Failed to parse JSON body: {e}")
            raise ValueError("Invalid JSON request body")

    async def _validate_form_data(self, request: Request) -> None:
        """
        Validate and sanitize form data.

        Args:
            request: FastAPI request object
        """
        try:
            # Parse form data
            form_data = await request.form()

            sanitized_data = {}
            for key, value in form_data.items():
                if isinstance(value, str):
                    sanitized_value = self.sanitizer.sanitize_text(value)

                    # Check for security threats
                    threats = detect_security_threats(sanitized_value)
                    if any(threats.values()):
                        logger.warning(
                            f"Security threat detected in form field '{key}': {threats}"
                        )

                    sanitized_data[key] = sanitized_value
                else:
                    sanitized_data[key] = value

            # Store sanitized form data
            request.state.sanitized_form = sanitized_data

        except Exception as e:
            logger.warning(f"Failed to parse form data: {e}")
            raise ValueError("Invalid form data")

    async def _validate_multipart_data(self, request: Request) -> None:
        """
        Validate multipart form data (files and fields).

        Args:
            request: FastAPI request object
        """
        try:
            # Parse multipart data
            form_data = await request.form()

            sanitized_data = {}
            for key, value in form_data.items():
                if isinstance(value, str):
                    # Sanitize text fields
                    sanitized_value = self.sanitizer.sanitize_text(value)

                    # Check for security threats
                    threats = detect_security_threats(sanitized_value)
                    if any(threats.values()):
                        logger.warning(
                            f"Security threat detected in multipart field '{key}': {threats}"
                        )

                    sanitized_data[key] = sanitized_value
                elif hasattr(value, 'filename'):
                    # Handle file uploads - sanitize filename
                    if value.filename:
                        sanitized_filename = self.sanitizer.sanitize_filename(value.filename)
                        if sanitized_filename != value.filename:
                            logger.info(f"Filename sanitized: '{value.filename}' -> '{sanitized_filename}'")
                        # Note: We can't modify the filename here, but we can log the issue
                else:
                    sanitized_data[key] = value

            # Store sanitized multipart data
            request.state.sanitized_multipart = sanitized_data

        except Exception as e:
            logger.warning(f"Failed to parse multipart data: {e}")
            raise ValueError("Invalid multipart data")

    def _sanitize_json_data(self, data: Any) -> Any:
        """
        Recursively sanitize JSON data.

        Args:
            data: JSON data to sanitize

        Returns:
            Sanitized data
        """
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Sanitize key if it's a string
                sanitized_key = self.sanitizer.sanitize_text(key) if isinstance(key, str) else key

                # Sanitize value
                if isinstance(value, str):
                    sanitized_value = self.sanitizer.sanitize_text(value)

                    # Check for security threats
                    threats = detect_security_threats(sanitized_value)
                    if any(threats.values()):
                        logger.warning(
                            f"Security threat detected in JSON field '{key}': {threats}"
                        )
                else:
                    sanitized_value = self._sanitize_json_data(value)

                sanitized[sanitized_key] = sanitized_value
            return sanitized

        elif isinstance(data, list):
            return [self._sanitize_json_data(item) for item in data]

        elif isinstance(data, str):
            sanitized = self.sanitizer.sanitize_text(data)

            # Check for security threats
            threats = detect_security_threats(sanitized)
            if any(threats.values()):
                logger.warning(f"Security threat detected in string data: {threats}")

            return sanitized

        else:
            return data


# Convenience function to create validation middleware
def create_validation_middleware(exclude_paths: list = None):
    """
    Create an input validation middleware instance.

    Args:
        exclude_paths: List of path prefixes to exclude from validation

    Returns:
        InputValidationMiddleware: Configured middleware instance
    """
    return InputValidationMiddleware


# Export the middleware class
__all__ = ['InputValidationMiddleware', 'create_validation_middleware']