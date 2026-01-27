"""Pipeline Configuration for Mini-SWE-Agent.

This module defines pipeline configurations for different execution modes:
1. SINGLE_AGENT: Legacy single-agent loop
2. SEQUENTIAL: Multi-stage sequential pipeline
3. PARALLEL: Multi-stage with parallel analysis

Pipeline Architecture:
```
                    ┌─> [code_searcher] ──┐
[task_parser] ─────>├─> [file_analyzer] ──├──> [planner] ──> [executor] ──> [verifier]
                    └─> [test_discovery] ─┘
```

The parallel stages (code_searcher, file_analyzer, test_discovery) run
concurrently via Go's goroutines when using jeeves-core's parallel mode.
"""

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

# Add jeeves-core to path
_jeeves_core_path = Path(__file__).parent.parent.parent.parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from protocols.config import (
    AgentConfig,
    PipelineConfig,
    RoutingRule,
    TokenStreamMode,
    AgentOutputMode,
)


class PipelineMode(str, Enum):
    """Pipeline execution modes."""
    SINGLE_AGENT = "single_agent"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


# =============================================================================
# ROUTING RULES
# =============================================================================

# Verifier routing rules
VERIFIER_ROUTING_RULES = [
    RoutingRule(condition="verdict", value="success", target="end"),
    RoutingRule(condition="verdict", value="loop_back", target="executor"),
    RoutingRule(condition="verdict", value="needs_more_info", target="code_searcher"),
]

# Executor routing rules
EXECUTOR_ROUTING_RULES = [
    RoutingRule(condition="needs_retry", value=True, target="planner"),
    RoutingRule(condition="error", value=True, target="verifier"),
]


# =============================================================================
# AGENT CONFIGURATIONS
# =============================================================================

def _create_task_parser_config() -> AgentConfig:
    """Create configuration for the task parser agent."""
    return AgentConfig(
        name="task_parser",
        output_key="task_info",
        has_llm=True,
        has_tools=False,
        model_role="task_parser",
        default_next="code_searcher",
        temperature=0.1,
        max_tokens=1000,
        prompt_key="mini_swe.task_parser",
    )


def _create_code_searcher_config(parallel: bool = False) -> AgentConfig:
    """Create configuration for the code searcher agent."""
    return AgentConfig(
        name="code_searcher",
        output_key="search_results",
        has_llm=True,
        has_tools=True,
        model_role="code_searcher",
        allowed_tools=["bash_execute", "find_files", "grep_search"],
        default_next="planner" if parallel else "file_analyzer",
        requires=["task_parser"] if parallel else None,
        temperature=0.2,
        max_tokens=2000,
        prompt_key="mini_swe.code_searcher",
    )


def _create_file_analyzer_config(parallel: bool = False) -> AgentConfig:
    """Create configuration for the file analyzer agent."""
    return AgentConfig(
        name="file_analyzer",
        output_key="analysis_results",
        has_llm=True,
        has_tools=True,
        model_role="file_analyzer",
        allowed_tools=["read_file", "bash_execute"],
        default_next="planner",
        requires=["task_parser"] if parallel else ["code_searcher"],
        temperature=0.3,
        max_tokens=3000,
        prompt_key="mini_swe.file_analyzer",
    )


def _create_test_discovery_config() -> AgentConfig:
    """Create configuration for the test discovery agent (parallel only)."""
    return AgentConfig(
        name="test_discovery",
        output_key="test_info",
        has_llm=True,
        has_tools=True,
        model_role="file_analyzer",  # Reuse file_analyzer model
        allowed_tools=["bash_execute", "find_files", "grep_search"],
        default_next="planner",
        requires=["task_parser"],  # Parallel with code_searcher and file_analyzer
        temperature=0.2,
        max_tokens=2000,
        prompt_key="mini_swe.test_discovery",
    )


def _create_planner_config(parallel: bool = False) -> AgentConfig:
    """Create configuration for the planner agent."""
    requires = (
        ["code_searcher", "file_analyzer", "test_discovery"]
        if parallel
        else ["file_analyzer"]
    )
    return AgentConfig(
        name="planner",
        output_key="plan",
        has_llm=True,
        has_tools=False,
        model_role="planner",
        default_next="executor",
        requires=requires,
        join_strategy="all",  # Wait for all dependencies
        temperature=0.3,
        max_tokens=4000,
        prompt_key="mini_swe.planner",
    )


def _create_executor_config() -> AgentConfig:
    """Create configuration for the executor agent."""
    return AgentConfig(
        name="executor",
        output_key="execution",
        has_llm=True,
        has_tools=True,
        model_role="executor",
        allowed_tools=["bash_execute", "write_file", "edit_file"],
        default_next="verifier",
        requires=["planner"],
        temperature=0.1,
        max_tokens=2000,
        routing_rules=EXECUTOR_ROUTING_RULES,
        prompt_key="mini_swe.executor",
    )


def _create_verifier_config() -> AgentConfig:
    """Create configuration for the verifier agent."""
    return AgentConfig(
        name="verifier",
        output_key="verification",
        has_llm=True,
        has_tools=True,
        model_role="verifier",
        allowed_tools=["bash_execute", "run_tests", "read_file"],
        default_next="end",
        requires=["executor"],
        temperature=0.2,
        max_tokens=2000,
        routing_rules=VERIFIER_ROUTING_RULES,
        prompt_key="mini_swe.verifier",
        # Enable token streaming for final response
        token_stream=TokenStreamMode.AUTHORITATIVE,
        output_mode=AgentOutputMode.TEXT,
    )


def _create_swe_agent_config() -> AgentConfig:
    """Create configuration for single-agent mode (legacy)."""
    return AgentConfig(
        name="swe_agent",
        output_key="result",
        has_llm=True,
        has_tools=True,
        model_role="swe_agent",
        allowed_tools=["bash_execute"],
        tool_access="all",
        default_next="end",
        temperature=0.3,
        max_tokens=4000,
        prompt_key="mini_swe.swe_agent",
    )


# =============================================================================
# PIPELINE CONFIGURATIONS
# =============================================================================

def create_swe_pipeline_config(
    mode: PipelineMode = PipelineMode.PARALLEL,
    max_iterations: int = 50,
    max_llm_calls: int = 100,
    max_agent_hops: int = 200,
) -> PipelineConfig:
    """Create pipeline configuration for SWE execution.

    Args:
        mode: Pipeline execution mode
        max_iterations: Maximum pipeline iterations
        max_llm_calls: Maximum LLM calls across all agents
        max_agent_hops: Maximum agent transitions

    Returns:
        PipelineConfig for the specified mode
    """
    if mode == PipelineMode.SINGLE_AGENT:
        return create_single_agent_config(max_iterations, max_llm_calls, max_agent_hops)

    is_parallel = mode == PipelineMode.PARALLEL

    agents: List[AgentConfig] = [
        _create_task_parser_config(),
        _create_code_searcher_config(parallel=is_parallel),
        _create_file_analyzer_config(parallel=is_parallel),
    ]

    # Add test_discovery only in parallel mode
    if is_parallel:
        agents.append(_create_test_discovery_config())

    agents.extend([
        _create_planner_config(parallel=is_parallel),
        _create_executor_config(),
        _create_verifier_config(),
    ])

    return PipelineConfig(
        name=f"mini_swe_{mode.value}",
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        max_agent_hops=max_agent_hops,
        agents=agents,
        clarification_resume_stage="task_parser",
        confirmation_resume_stage="executor",
    )


def create_single_agent_config(
    max_iterations: int = 50,
    max_llm_calls: int = 100,
    max_agent_hops: int = 200,
) -> PipelineConfig:
    """Create single-agent pipeline configuration (legacy mode).

    This configuration creates a pipeline with a single SWE agent
    that handles all tasks, matching the original mini-swe-agent behavior.

    Args:
        max_iterations: Maximum pipeline iterations
        max_llm_calls: Maximum LLM calls
        max_agent_hops: Maximum agent transitions

    Returns:
        PipelineConfig for single-agent mode
    """
    return PipelineConfig(
        name="mini_swe_single_agent",
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        max_agent_hops=max_agent_hops,
        agents=[_create_swe_agent_config()],
        clarification_resume_stage="swe_agent",
        confirmation_resume_stage="swe_agent",
    )


__all__ = [
    "PipelineMode",
    "create_swe_pipeline_config",
    "create_single_agent_config",
]
