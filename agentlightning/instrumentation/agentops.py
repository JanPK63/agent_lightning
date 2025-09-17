"""
Minimal no-op Instrumentation shim for tests.
"""

from typing import Optional

import logging

logger = logging.getLogger(__name__)


def instrument_agentops():
    """No-op instrumentation for test environments."""
    logger.debug("instrument_agentops: no-op patching")
    return False


def uninstrument_agentops():
    pass


def agentops_local_server():
    class _StubServer:
        def run(self, **kwargs):
            pass
    return _StubServer()


def _patch_new_agentops():
    return False


def _unpatch_new_agentops():
    pass


def _patch_old_agentops():
    return False


def _unpatch_old_agentops():
    pass

class AgentOpsServerManager:
    def __init__(self, daemon: bool = True, port: Optional[int] = None):
        self.server_port = port
        self.daemon = daemon
        self._started = False

    def start(self):
        # Lightweight stub for tests; no external server started.
        self._started = True

    def is_alive(self) -> bool:
        return self._started

    def stop(self):
        self._started = False

    def get_port(self) -> Optional[int]:
        return self.server_port
