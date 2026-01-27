"""Capability Registration for Mini-SWE-Agent.

This module registers mini-swe-agent as a capability with jeeves-core.
All capability resources (tools, agents, services) are registered here.

Jeeves-Core Components Wired:
- ControlTower: Kernel for lifecycle, resources, IPC, and events
- InMemoryCommBus: Message bus for inter-service communication
- Memory Handlers: Session state, entity tracking, memory search

Constitutional Reference:
- CONTRACT.md: Capabilities MUST implement a registration function
- Avionics R3: No Domain Logic - infrastructure provides transport, not business logic
- Capability Constitution R6: Domain Config Ownership

Usage:
    from minisweagent.capability.wiring import register_capability, create_jeeves_context

    # At application startup (before using runtime services)
    register_capability()

    # Create full jeeves-core context for enhanced features
    context = create_jeeves_context()
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

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
    from control_tower.kernel import ControlTower
    from control_tower.ipc.commbus import InMemoryCommBus

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


# =============================================================================
# JEEVES-CORE CONTEXT
# =============================================================================


@dataclass
class JeevesContext:
    """Context containing all wired jeeves-core components.

    This provides access to:
    - ControlTower: Kernel for lifecycle, resources, IPC, and events
    - CommBus: Message bus for inter-service communication
    - Logger: Structured logging

    Usage:
        context = create_jeeves_context()

        # Use ControlTower for resource tracking
        context.control_tower.record_llm_call(pid, tokens_in=100, tokens_out=50)

        # Use CommBus for memory operations
        await context.commbus.query(GetSessionState(session_id="..."))
    """
    control_tower: Optional["ControlTower"] = None
    commbus: Optional["InMemoryCommBus"] = None
    db: Optional[Any] = None  # Database connection for persistence
    _memory_handlers_registered: bool = field(default=False, repr=False)

    def is_wired(self) -> bool:
        """Check if core components are wired."""
        return self.control_tower is not None and self.commbus is not None


def _create_simple_logger() -> Any:
    """Create a simple logger adapter for ControlTower.

    ControlTower expects a LoggerProtocol with bind() method.
    """
    class SimpleLoggerAdapter:
        """Adapter to make Python logger compatible with LoggerProtocol."""

        def __init__(self, name: str = "jeeves"):
            self._logger = logging.getLogger(name)
            self._context: dict = {}

        def info(self, event: str, **kwargs) -> None:
            self._logger.info(f"{event}: {kwargs}")

        def warn(self, event: str, **kwargs) -> None:
            self._logger.warning(f"{event}: {kwargs}")

        def warning(self, event: str, **kwargs) -> None:
            self._logger.warning(f"{event}: {kwargs}")

        def error(self, event: str, **kwargs) -> None:
            self._logger.error(f"{event}: {kwargs}")

        def debug(self, event: str, **kwargs) -> None:
            self._logger.debug(f"{event}: {kwargs}")

        def bind(self, **kwargs) -> "SimpleLoggerAdapter":
            """Return self with updated context."""
            adapter = SimpleLoggerAdapter(self._logger.name)
            adapter._context = {**self._context, **kwargs}
            return adapter

    return SimpleLoggerAdapter("mini-swe-agent")


def create_jeeves_context(
    db: Optional[Any] = None,
    register_memory: bool = True,
) -> JeevesContext:
    """Create a JeevesContext with wired jeeves-core components.

    This wires up:
    1. ControlTower - Kernel for lifecycle, resources, IPC, and events
    2. InMemoryCommBus - Message bus for inter-service communication
    3. Memory handlers - Session state, entity tracking, memory search

    Args:
        db: Optional database connection for persistent features
        register_memory: Whether to register memory handlers (default True)

    Returns:
        JeevesContext with wired components

    Example:
        # Basic usage
        context = create_jeeves_context()

        # With database for persistent sessions
        context = create_jeeves_context(db=async_pool)

        # Use in orchestrator
        orchestrator = create_swe_orchestrator(
            control_tower=context.control_tower,
            ...
        )
    """
    context = JeevesContext()

    try:
        # Import jeeves-core components
        from control_tower.kernel import ControlTower
        from control_tower.ipc.commbus import InMemoryCommBus
        from control_tower.types import ResourceQuota

        # Create logger adapter
        log = _create_simple_logger()

        # Create default resource quota
        default_quota = ResourceQuota(
            max_llm_calls=100,
            max_tool_calls=200,
            max_agent_hops=200,
            max_iterations=50,
            max_input_tokens=8192,
            max_output_tokens=4096,
            max_context_tokens=32768,
            timeout_seconds=300,
        )

        # Create ControlTower kernel
        context.control_tower = ControlTower(
            logger=log,
            default_quota=default_quota,
            default_service=f"{CAPABILITY_ID}_service",
            db=db,
        )
        logger.info("ControlTower kernel initialized")

        # Create CommBus
        context.commbus = InMemoryCommBus(
            query_timeout=30.0,
            logger=log,
        )
        logger.info("InMemoryCommBus initialized")

        # Register memory handlers if requested
        if register_memory:
            _register_memory_handlers(context, db, log)

        context.db = db

    except ImportError as e:
        logger.warning(f"jeeves-core components not available: {e}")
    except Exception as e:
        logger.error(f"Failed to create JeevesContext: {e}")

    return context


def _register_memory_handlers(
    context: JeevesContext,
    db: Optional[Any],
    log: Any,
) -> None:
    """Register memory handlers with the CommBus.

    Wires up handlers for:
    - GetSessionState, GetRecentEntities, SearchMemory
    - ClearSession, UpdateFocus, AddEntityReference
    """
    if context.commbus is None:
        return

    try:
        from memory_module.handlers import register_memory_handlers

        register_memory_handlers(
            commbus=context.commbus,
            session_state_service=None,  # Lazy init from db
            db=db,
            logger=log,
        )
        context._memory_handlers_registered = True
        logger.info("Memory handlers registered with CommBus")

    except ImportError as e:
        logger.debug(f"Memory module not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to register memory handlers: {e}")


# Global context (lazily initialized)
_global_context: Optional[JeevesContext] = None


def get_jeeves_context() -> JeevesContext:
    """Get the global JeevesContext, creating if needed.

    This provides a singleton context for use across the application.

    Returns:
        The global JeevesContext instance
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
