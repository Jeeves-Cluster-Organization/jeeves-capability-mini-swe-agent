"""Mini-SWE-Agent Capability Configuration.

This module provides configuration for the capability including
pipeline definitions and agent profiles.
"""

from minisweagent.capability.config.pipeline import (
    create_swe_pipeline_config,
    PipelineMode,
)

__all__ = [
    "create_swe_pipeline_config",
    "PipelineMode",
]
