#!/usr/bin/env python3
"""
AgentOps OpenTelemetry Import Patch

This module patches the incorrect OpenTelemetry import in agentops package
to fix the "cannot import name 'metrics' from 'opentelemetry'" error.

The agentops package uses the old import pattern:
    from opentelemetry import metrics, trace

This should be:
    from opentelemetry import trace
    from opentelemetry.metrics import set_meter_provider, get_meter_provider, get_meter

This patch monkey-patches the opentelemetry module to provide the old interface.
"""

import sys
import importlib
from unittest.mock import MagicMock

def patch_opentelemetry_metrics():
    """
    Monkey-patch opentelemetry to provide the old metrics import interface
    that agentops expects.
    """
    try:
        # Import the correct modules
        from opentelemetry.metrics import set_meter_provider, get_meter_provider, get_meter
        from opentelemetry import trace

        # Get the opentelemetry module
        opentelemetry_module = sys.modules.get('opentelemetry')
        if opentelemetry_module is None:
            # Force import if not already loaded
            import opentelemetry
            opentelemetry_module = opentelemetry

        # Create a mock metrics module that provides the expected interface
        mock_metrics = MagicMock()
        mock_metrics.set_meter_provider = set_meter_provider
        mock_metrics.get_meter_provider = get_meter_provider
        mock_metrics.get_meter = get_meter

        # Add the metrics attribute to the opentelemetry module
        opentelemetry_module.metrics = mock_metrics

        print("✅ AgentOps OpenTelemetry patch applied successfully")
        return True

    except ImportError as e:
        print(f"❌ Failed to apply AgentOps patch: {e}")
        return False

# Apply the patch when this module is imported
patch_opentelemetry_metrics()