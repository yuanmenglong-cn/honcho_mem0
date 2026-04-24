"""mem0 integration module.

Provides mem0 memory storage/retrieval capabilities to honcho agents.
Configuration-driven via settings.MEM0 flags.
"""

from src.mem0_integration.client import (
    Mem0Client,
    close_mem0_client,
    get_mem0_client,
)

__all__ = ["Mem0Client", "get_mem0_client", "close_mem0_client"]
