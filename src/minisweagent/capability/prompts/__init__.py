"""Prompt registry for mini-swe-agent capability.

This module provides the prompt registry that manages all prompt templates
for the SWE agent pipeline stages.
"""

from minisweagent.capability.prompts.registry import (
    MiniSWEPromptRegistry,
    create_prompt_registry,
)

__all__ = [
    "MiniSWEPromptRegistry",
    "create_prompt_registry",
]
