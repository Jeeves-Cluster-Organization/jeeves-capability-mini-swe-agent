"""Capability Registration for Mini-SWE-Agent.

This module registers mini-swe-agent as a capability with jeeves-core.
All capability resources (tools, agents, services) are registered here.

Constitutional Reference:
- CONTRACT.md: Capabilities MUST implement a registration function
- Avionics R3: No Domain Logic - infrastructure provides transport, not business logic
- Capability Constitution R6: Domain Config Ownership

Usage:
    from minisweagent.capability.wiring import register_capability

    # At application startup (before using runtime services)
    register_capability()
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add jeeves-core to path for imports
_jeeves_core_path = Path(__file__).parent.parent.parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from protocols.capability import (
    get_capability_resource_registry,
    DomainModeConfig,
    DomainServiceConfig,
    DomainAgentConfig,
    CapabilityToolsConfig,
    CapabilityOrchestratorConfig,
)
from protocols import AgentLLMConfig

if TYPE_CHECKING:
    from protocols.capability import CapabilityResourceRegistry

# =============================================================================
# CAPABILITY CONSTANTS
# =============================================================================

CAPABILITY_ID = "mini-swe-agent"
CAPABILITY_DESCRIPTION = "Software engineering agent for code modifications"


# =============================================================================
# AGENT LLM CONFIGURATIONS
# =============================================================================

# Agent configurations for different roles in the pipeline
AGENT_LLM_CONFIGS = {
    # Task parser - fast model for initial analysis
    "task_parser": AgentLLMConfig(
        agent_name="task_parser",
        model="qwen2.5-7b-instruct-q4_k_m",
        temperature=0.1,
        max_tokens=1000,
        context_window=8192,
        timeout_seconds=60,
    ),
    # Code searcher - medium model for code search
    "code_searcher": AgentLLMConfig(
        agent_name="code_searcher",
        model="qwen2.5-7b-instruct-q4_k_m",
        temperature=0.2,
        max_tokens=2000,
        context_window=16384,
        timeout_seconds=120,
    ),
    # File analyzer - medium model for analysis (can run in parallel)
    "file_analyzer": AgentLLMConfig(
        agent_name="file_analyzer",
        model="qwen2.5-14b-instruct-q4_k_m",
        temperature=0.3,
        max_tokens=3000,
        context_window=16384,
        timeout_seconds=120,
    ),
    # Planner - larger model for planning changes
    "planner": AgentLLMConfig(
        agent_name="planner",
        model="qwen2.5-32b-instruct-q4_k_m",
        temperature=0.3,
        max_tokens=4000,
        context_window=32768,
        timeout_seconds=180,
    ),
    # Executor - fast model for executing changes
    "executor": AgentLLMConfig(
        agent_name="executor",
        model="qwen2.5-7b-instruct-q4_k_m",
        temperature=0.1,
        max_tokens=2000,
        context_window=16384,
        timeout_seconds=120,
    ),
    # Verifier - medium model for verification
    "verifier": AgentLLMConfig(
        agent_name="verifier",
        model="qwen2.5-14b-instruct-q4_k_m",
        temperature=0.2,
        max_tokens=2000,
        context_window=16384,
        timeout_seconds=120,
    ),
}


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

AGENT_DEFINITIONS = [
    DomainAgentConfig(
        name="task_parser",
        description="Parses user task and extracts key information",
        layer="perception",
        tools=[],
    ),
    DomainAgentConfig(
        name="code_searcher",
        description="Searches codebase for relevant files and symbols",
        layer="execution",
        tools=["bash_execute", "find_files", "grep_search"],
    ),
    DomainAgentConfig(
        name="file_analyzer",
        description="Analyzes file contents to understand code structure",
        layer="execution",
        tools=["read_file", "bash_execute"],
    ),
    DomainAgentConfig(
        name="planner",
        description="Plans code changes based on analysis",
        layer="planning",
        tools=[],
    ),
    DomainAgentConfig(
        name="executor",
        description="Executes planned code changes",
        layer="execution",
        tools=["bash_execute", "write_file", "edit_file"],
    ),
    DomainAgentConfig(
        name="verifier",
        description="Verifies changes work correctly",
        layer="validation",
        tools=["bash_execute", "run_tests"],
    ),
]


# =============================================================================
# REGISTRATION FUNCTIONS
# =============================================================================

def _create_tool_catalog():
    """Lazy initializer for tool catalog.

    Returns the tool catalog when infrastructure needs it.
    This avoids circular imports and allows lazy loading.
    """
    from minisweagent.capability.tools.catalog import create_tool_catalog
    return create_tool_catalog()


def _create_orchestrator_service(
    llm_factory,
    tool_executor,
    log,
    persistence,
    control_tower=None,
):
    """Factory function to create the orchestrator service.

    Called by infrastructure to create the service that handles requests.
    """
    from minisweagent.capability.orchestrator import create_swe_orchestrator
    return create_swe_orchestrator(
        llm_factory=llm_factory,
        tool_executor=tool_executor,
        log=log,
        persistence=persistence,
        control_tower=control_tower,
    )


def register_capability() -> None:
    """Register mini-swe-agent capability with jeeves-core.

    This function must be called at startup before using runtime services.
    It registers all capability resources with the infrastructure.
    """
    registry = get_capability_resource_registry()

    # 1. Register capability mode
    registry.register_mode(
        CAPABILITY_ID,
        DomainModeConfig(
            mode_id=CAPABILITY_ID,
            response_fields=["status", "output", "trajectory"],
            requires_repo_path=True,
        ),
    )

    # 2. Register service configuration
    registry.register_service(
        CAPABILITY_ID,
        DomainServiceConfig(
            service_id=f"{CAPABILITY_ID}_service",
            service_type="flow",
            capabilities=["code_modification", "bug_fixing", "code_analysis"],
            max_concurrent=5,
            is_default=True,
            is_readonly=False,  # SWE agent modifies code
            requires_confirmation=True,  # Confirm before executing commands
            default_session_title="SWE Session",
            pipeline_stages=[
                "task_parser",
                "code_searcher",
                "file_analyzer",
                "planner",
                "executor",
                "verifier",
            ],
        ),
    )

    # 3. Register agent definitions
    registry.register_agents(CAPABILITY_ID, AGENT_DEFINITIONS)

    # 4. Register tools configuration (lazy initialization)
    registry.register_tools(
        CAPABILITY_ID,
        CapabilityToolsConfig(
            tool_ids=[
                "bash_execute",
                "read_file",
                "write_file",
                "edit_file",
                "find_files",
                "grep_search",
                "run_tests",
            ],
            initializer=_create_tool_catalog,
        ),
    )

    # 5. Register orchestrator factory
    registry.register_orchestrator(
        CAPABILITY_ID,
        CapabilityOrchestratorConfig(
            factory=_create_orchestrator_service,
        ),
    )

    # 6. Register LLM configurations with capability registry
    try:
        from avionics.capability_registry import get_capability_registry

        llm_registry = get_capability_registry()
        for agent_name, config in AGENT_LLM_CONFIGS.items():
            llm_registry.register(
                capability_id=CAPABILITY_ID,
                agent_name=agent_name,
                config=config,
            )
    except ImportError:
        # avionics not available - running standalone
        pass


def get_agent_config(agent_name: str) -> AgentLLMConfig:
    """Get LLM configuration for a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        AgentLLMConfig for the agent

    Raises:
        KeyError: If agent not found
    """
    if agent_name not in AGENT_LLM_CONFIGS:
        raise KeyError(f"Unknown agent: {agent_name}")
    return AGENT_LLM_CONFIGS[agent_name]


__all__ = [
    "register_capability",
    "get_agent_config",
    "CAPABILITY_ID",
    "AGENT_LLM_CONFIGS",
    "AGENT_DEFINITIONS",
]
