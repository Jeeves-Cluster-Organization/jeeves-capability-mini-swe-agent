"""Capability Registration for Mini-SWE-Agent.

This module registers mini-swe-agent as a capability with jeeves-infra.
All capability resources (tools, agents, services) are registered here.

Jeeves-Core Components Wired (via gRPC):
- KernelClient: gRPC client to Go kernel for process lifecycle, resource quotas
- CommBusClient: gRPC client for pub/sub messaging via Go kernel

Constitutional Reference:
- CONTRACT.md: Capabilities MUST implement a registration function
- Micro-OS Architecture: Go kernel is REQUIRED, not optional
- Capability Constitution R6: Domain Config Ownership

Usage:
    from minisweagent.capability.wiring import register_capability, create_jeeves_context

    # At application startup (before using runtime services)
    register_capability()

    # Create jeeves context with Go kernel connection (REQUIRED)
    context = create_jeeves_context()
    orchestrator = create_swe_orchestrator(kernel_client=context.kernel_client, ...)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

# jeeves-core is now a proper package - install with: pip install -e ./jeeves-core
# No sys.path manipulation needed

from jeeves_infra.protocols import (
    get_capability_resource_registry,
    DomainModeConfig,
    DomainServiceConfig,
    DomainAgentConfig,
    CapabilityToolsConfig,
    CapabilityOrchestratorConfig,
)
from jeeves_infra.protocols import AgentLLMConfig

if TYPE_CHECKING:
    from jeeves_infra.protocols import CapabilityResourceRegistry
    from jeeves_infra.kernel_client import KernelClient

logger = logging.getLogger(__name__)

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
    kernel_client,
):
    """Factory function to create the orchestrator service.

    Called by infrastructure to create the service that handles requests.

    Args:
        kernel_client: gRPC client to Go kernel (REQUIRED)
        llm_factory: Factory for LLM providers
        tool_executor: Tool executor instance
        log: Logger instance
        persistence: Persistence adapter

    Raises:
        RuntimeError: If kernel_client is None
    """
    if kernel_client is None:
        raise RuntimeError(
            "Go kernel is required. Infrastructure must provide kernel_client."
        )
    from minisweagent.capability.orchestrator import create_swe_orchestrator
    return create_swe_orchestrator(
        kernel_client=kernel_client,
        llm_factory=llm_factory,
        tool_executor=tool_executor,
        log=log,
        persistence=persistence,
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
        from jeeves_infra.capability_registry import get_capability_registry

        llm_registry = get_capability_registry()
        for agent_name, config in AGENT_LLM_CONFIGS.items():
            llm_registry.register(
                capability_id=CAPABILITY_ID,
                agent_name=agent_name,
                config=config,
            )
    except ImportError:
        # jeeves_infra not available - running standalone
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


# =============================================================================
# JEEVES-CORE CONTEXT
# =============================================================================


@dataclass
class JeevesContext:
    """Context containing wired Go kernel gRPC client (REQUIRED).

    The Go kernel is mandatory per the micro-OS architecture.
    Python capabilities communicate with the Go kernel via gRPC.

    Usage:
        context = create_jeeves_context()
        await context.kernel_client.record_llm_call(pid, tokens_in=100, tokens_out=50)
    """
    kernel_client: "KernelClient"
    db: Optional[Any] = None
    _kernel_address: str = field(default="localhost:50051", repr=False)


def create_jeeves_context(
    db: Optional[Any] = None,
    kernel_address: Optional[str] = None,
) -> JeevesContext:
    """Create a JeevesContext with wired Go kernel gRPC client.

    The Go kernel is REQUIRED. This function will fail if the kernel
    client cannot be created.

    This wires up:
    1. KernelClient - gRPC client to Go kernel for lifecycle, resources, events

    Args:
        db: Optional database connection for persistent features
        kernel_address: gRPC address of Go kernel (default: KERNEL_GRPC_ADDRESS env or localhost:50051)

    Returns:
        JeevesContext with wired kernel client

    Raises:
        RuntimeError: If kernel client cannot be created

    Example:
        context = create_jeeves_context()
        orchestrator = create_swe_orchestrator(kernel_client=context.kernel_client, ...)
    """
    import os
    from grpc import aio as grpc_aio

    # Determine kernel address
    if kernel_address is None:
        kernel_address = os.getenv("KERNEL_GRPC_ADDRESS", "localhost:50051")

    try:
        from jeeves_infra.kernel_client import KernelClient

        # Create gRPC channel (lazy - doesn't connect until first call)
        channel = grpc_aio.insecure_channel(kernel_address)
        kernel_client = KernelClient(channel)
        logger.info(f"KernelClient initialized (target: {kernel_address})")

        return JeevesContext(
            kernel_client=kernel_client,
            db=db,
            _kernel_address=kernel_address,
        )

    except ImportError as e:
        raise RuntimeError(
            f"Go kernel client not available: {e}\n"
            "Install jeeves-infra: pip install -e ./jeeves-infra"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to create kernel client for {kernel_address}: {e}\n"
            "Ensure the Go kernel is running:\n"
            "  cd jeeves-core && go run ./cmd/kernel --grpc-port 50051"
        ) from e


# Global context (lazily initialized)
_global_context: Optional[JeevesContext] = None


def get_jeeves_context() -> JeevesContext:
    """Get the global JeevesContext, creating if needed.

    This provides a singleton context for use across the application.
    The Go kernel MUST be running for this to succeed.

    Returns:
        The global JeevesContext instance

    Raises:
        RuntimeError: If Go kernel is not available
    """
    global _global_context
    if _global_context is None:
        _global_context = create_jeeves_context()
    return _global_context


def reset_jeeves_context() -> None:
    """Reset the global JeevesContext (for testing)."""
    global _global_context
    _global_context = None


__all__ = [
    "register_capability",
    "get_agent_config",
    "CAPABILITY_ID",
    "AGENT_LLM_CONFIGS",
    "AGENT_DEFINITIONS",
    # Jeeves-core integration
    "JeevesContext",
    "create_jeeves_context",
    "get_jeeves_context",
    "reset_jeeves_context",
]
