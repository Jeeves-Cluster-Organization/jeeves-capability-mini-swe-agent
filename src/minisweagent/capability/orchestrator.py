"""Orchestrator for Mini-SWE-Agent Capability.

This module creates the orchestrator service that handles SWE requests.
All execution flows through jeeves-core's PipelineRunner.

Pipeline Modes:
- unified: Single-stage pipeline with self-routing (mimics original agent loop)
- parallel: Multi-stage pipeline with parallel analysis

Constitutional Reference:
- Capability owns orchestration logic
- Infrastructure provides runtime (LLM, tools, persistence)
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

# Add jeeves-core to path
_jeeves_core_path = Path(__file__).parent.parent.parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from protocols.agents import (
    Agent,
    PipelineRunner,
    create_pipeline_runner,
    create_envelope,
    LLMProvider,
    ToolExecutor,
    Logger,
    Persistence,
    PromptRegistry,
)
from protocols.config import AgentConfig, PipelineConfig, RoutingRule
from protocols import RequestContext

if TYPE_CHECKING:
    pass


# =============================================================================
# ORCHESTRATOR MODES
# =============================================================================

class OrchestratorMode:
    """Orchestrator execution modes."""
    PIPELINE = "pipeline"  # Single unified mode - all execution via PipelineRunner


# =============================================================================
# SWE ORCHESTRATOR
# =============================================================================

@dataclass
class SWEOrchestratorConfig:
    """Configuration for the SWE orchestrator."""
    max_iterations: int = 50
    max_llm_calls: int = 100
    max_agent_hops: int = 200
    step_limit: int = 0  # 0 = disabled
    cost_limit: float = 3.0
    timeout: int = 30
    confirm_commands: bool = True
    pipeline_mode: str = "unified"  # "unified" (single-stage) or "parallel" (multi-stage)


class SWEOrchestrator:
    """Orchestrator for SWE agent execution.

    All execution flows through jeeves-core's PipelineRunner:
    - unified: Single-stage pipeline with self-routing (mimics original agent loop)
    - parallel: Multi-stage pipeline with parallel analysis

    Constitutional Reference:
    - Capability owns orchestration logic
    - Infrastructure provides runtime (LLM, tools, persistence)
    """

    def __init__(
        self,
        config: SWEOrchestratorConfig,
        llm_factory: Optional[Callable[[str], LLMProvider]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        logger: Optional[Logger] = None,
        persistence: Optional[Persistence] = None,
        prompt_registry: Optional[PromptRegistry] = None,
        control_tower: Optional[Any] = None,
    ):
        self.config = config
        self.llm_factory = llm_factory
        self.tool_executor = tool_executor
        self.logger = logger or _NullLogger()
        self.persistence = persistence
        self.prompt_registry = prompt_registry
        self.control_tower = control_tower
        self._pipeline_runner: Optional[PipelineRunner] = None

    def _get_pipeline_runner(self) -> PipelineRunner:
        """Get or create the pipeline runner for parallel execution."""
        if self._pipeline_runner is None:
            pipeline_config = self._create_pipeline_config()
            self._pipeline_runner = create_pipeline_runner(
                config=pipeline_config,
                llm_provider_factory=self.llm_factory,
                tool_executor=self.tool_executor,
                logger=self.logger,
                persistence=self.persistence,
                prompt_registry=self.prompt_registry,
            )
        return self._pipeline_runner

    def _create_pipeline_config(self) -> PipelineConfig:
        """Create pipeline configuration based on pipeline_mode.

        Modes:
        - unified: Single-stage with self-routing loop (mimics original agent behavior)
        - parallel: Multi-stage with parallel analysis
        """
        if self.config.pipeline_mode == "unified":
            return self._create_unified_pipeline_config()
        else:
            return self._create_parallel_pipeline_config()

    def _create_unified_pipeline_config(self) -> PipelineConfig:
        """Create single-stage pipeline with self-routing loop.

        This mimics the original mini-swe-agent behavior:
        - Single agent with LLM and all tools
        - Loops back to itself until completion marker detected
        - Exits on completion or limits exceeded
        """
        return PipelineConfig(
            name="mini_swe_unified",
            max_iterations=self.config.max_iterations,
            max_llm_calls=self.config.max_llm_calls,
            max_agent_hops=self.config.max_agent_hops,
            agents=[
                AgentConfig(
                    name="swe_agent",
                    output_key="result",
                    has_llm=True,
                    has_tools=True,
                    model_role="swe_agent",
                    allowed_tools=[
                        "bash_execute", "read_file", "write_file",
                        "edit_file", "find_files", "grep_search", "run_tests"
                    ],
                    default_next="swe_agent",  # Loop back by default
                    temperature=0.3,
                    max_tokens=4000,
                    prompt_key="mini_swe.swe_agent",
                    routing_rules=[
                        RoutingRule(condition="completed", value=True, target="end"),
                        RoutingRule(condition="limits_exceeded", value=True, target="end"),
                    ],
                ),
            ],
            clarification_resume_stage="swe_agent",
            confirmation_resume_stage="swe_agent",
        )

    def _create_parallel_pipeline_config(self) -> PipelineConfig:
        """Create multi-stage pipeline with parallel analysis."""
        return PipelineConfig(
            name="mini_swe_parallel",
            max_iterations=self.config.max_iterations,
            max_llm_calls=self.config.max_llm_calls,
            max_agent_hops=self.config.max_agent_hops,
            agents=[
                # Stage 1: Parse task
                AgentConfig(
                    name="task_parser",
                    output_key="task_info",
                    has_llm=True,
                    model_role="task_parser",
                    default_next="parallel_analysis",
                    temperature=0.1,
                    max_tokens=1000,
                    prompt_key="mini_swe.task_parser",
                ),
                # Stage 2: Parallel analysis (fan-out)
                AgentConfig(
                    name="code_searcher",
                    output_key="search_results",
                    has_llm=True,
                    has_tools=True,
                    model_role="code_searcher",
                    allowed_tools=["bash_execute", "find_files", "grep_search"],
                    default_next="planner",
                    requires=["task_parser"],
                    temperature=0.2,
                    max_tokens=2000,
                    prompt_key="mini_swe.code_searcher",
                ),
                AgentConfig(
                    name="file_analyzer",
                    output_key="analysis_results",
                    has_llm=True,
                    has_tools=True,
                    model_role="file_analyzer",
                    allowed_tools=["read_file", "bash_execute"],
                    default_next="planner",
                    requires=["task_parser"],
                    temperature=0.3,
                    max_tokens=3000,
                    prompt_key="mini_swe.file_analyzer",
                ),
                # Stage 3: Planning (fan-in)
                AgentConfig(
                    name="planner",
                    output_key="plan",
                    has_llm=True,
                    model_role="planner",
                    default_next="executor",
                    requires=["code_searcher", "file_analyzer"],
                    join_strategy="all",
                    temperature=0.3,
                    max_tokens=4000,
                    prompt_key="mini_swe.planner",
                ),
                # Stage 4: Execution
                AgentConfig(
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
                    prompt_key="mini_swe.executor",
                    routing_rules=[
                        RoutingRule(condition="needs_retry", value=True, target="planner"),
                    ],
                ),
                # Stage 5: Verification
                AgentConfig(
                    name="verifier",
                    output_key="verification",
                    has_llm=True,
                    has_tools=True,
                    model_role="verifier",
                    allowed_tools=["bash_execute", "run_tests"],
                    default_next="end",
                    requires=["executor"],
                    temperature=0.2,
                    max_tokens=2000,
                    prompt_key="mini_swe.verifier",
                    routing_rules=[
                        RoutingRule(condition="verdict", value="loop_back", target="executor"),
                        RoutingRule(condition="verdict", value="success", target="end"),
                    ],
                ),
            ],
            clarification_resume_stage="task_parser",
            confirmation_resume_stage="executor",
        )

    async def run(
        self,
        task: str,
        user_id: str = "anonymous",
        session_id: str = "",
        thread_id: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute an SWE task via pipeline.

        Args:
            task: The task description
            user_id: User identifier
            session_id: Session identifier
            thread_id: Thread ID for state persistence
            **kwargs: Additional parameters

        Returns:
            Dict with execution results
        """
        return await self._run_pipeline(task, user_id, session_id, thread_id, **kwargs)

    async def _run_pipeline(
        self,
        task: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run task via jeeves-core PipelineRunner.

        Uses either unified (single-stage) or parallel (multi-stage) pipeline
        based on config.pipeline_mode.
        """
        self.logger.info("swe_pipeline_started", task=task[:100])

        runner = self._get_pipeline_runner()

        # Create request context
        import uuid
        request_context = RequestContext(
            request_id=str(uuid.uuid4()),
            capability="mini-swe-agent",
            user_id=user_id,
            session_id=session_id,
        )

        # Create envelope
        envelope = create_envelope(
            raw_input=task,
            request_context=request_context,
            metadata=kwargs,
        )

        try:
            # Run pipeline
            result_envelope = await runner.run(envelope, thread_id=thread_id)

            return {
                "status": "completed" if not result_envelope.terminated else result_envelope.terminal_reason,
                "output": result_envelope.outputs.get("verification", {}).get("response", ""),
                "outputs": result_envelope.outputs,
                "llm_calls": result_envelope.llm_call_count,
                "agent_hops": result_envelope.agent_hop_count,
            }
        except Exception as e:
            self.logger.error("swe_pipeline_error", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }

    async def run_streaming(
        self,
        task: str,
        user_id: str = "anonymous",
        session_id: str = "",
        thread_id: str = "",
        **kwargs,
    ):
        """Execute with streaming outputs.

        Yields stage outputs as they complete.
        """
        runner = self._get_pipeline_runner()

        import uuid
        request_context = RequestContext(
            request_id=str(uuid.uuid4()),
            capability="mini-swe-agent",
            user_id=user_id,
            session_id=session_id,
        )

        envelope = create_envelope(
            raw_input=task,
            request_context=request_context,
            metadata=kwargs,
        )

        async for stage_name, output in runner.run_streaming(envelope, thread_id=thread_id):
            yield (stage_name, output)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_swe_orchestrator(
    llm_factory: Optional[Callable[[str], LLMProvider]] = None,
    tool_executor: Optional[ToolExecutor] = None,
    logger: Optional[Logger] = None,
    persistence: Optional[Persistence] = None,
    prompt_registry: Optional[PromptRegistry] = None,
    control_tower: Optional[Any] = None,
    **config_kwargs,
) -> SWEOrchestrator:
    """Factory function to create an SWE orchestrator.

    Called by infrastructure to create the orchestrator service.

    Args:
        llm_factory: Factory to create LLM providers by role
        tool_executor: Tool executor instance
        logger: Logger instance
        persistence: State persistence
        prompt_registry: Prompt template registry
        control_tower: Control tower for lifecycle management
        **config_kwargs: Additional config options (pipeline_mode, step_limit, etc.)

    Returns:
        SWEOrchestrator instance
    """
    config = SWEOrchestratorConfig(**config_kwargs)

    return SWEOrchestrator(
        config=config,
        llm_factory=llm_factory,
        tool_executor=tool_executor,
        logger=logger,
        persistence=persistence,
        prompt_registry=prompt_registry,
        control_tower=control_tower,
    )


# =============================================================================
# NULL LOGGER
# =============================================================================

class _NullLogger:
    """Null logger for when no logger is provided."""
    def info(self, event: str, **kwargs) -> None: pass
    def warn(self, event: str, **kwargs) -> None: pass
    def error(self, event: str, **kwargs) -> None: pass
    def debug(self, event: str, **kwargs) -> None: pass
    def bind(self, **kwargs) -> "_NullLogger": return self


__all__ = [
    "SWEOrchestrator",
    "SWEOrchestratorConfig",
    "OrchestratorMode",
    "create_swe_orchestrator",
]
