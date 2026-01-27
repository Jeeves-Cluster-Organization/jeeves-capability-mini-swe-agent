"""Mini-SWE-Agent Pipeline Components.

This module provides agent components for the jeeves-core pipeline:
- SWEPostProcessor: Completion detection and error handling
"""

from minisweagent.capability.agents.swe_post_processor import (
    SWEPostProcessor,
    create_post_processor,
    COMPLETION_MARKERS,
)

__all__ = [
    "SWEPostProcessor",
    "create_post_processor",
    "COMPLETION_MARKERS",
]
