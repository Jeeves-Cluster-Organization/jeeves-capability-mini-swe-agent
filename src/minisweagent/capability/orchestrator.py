"""Orchestrator for Mini-SWE-Agent Capability (v2.0).

This module creates the orchestrator service that handles SWE requests.
All execution flows through jeeves-core's PipelineRunner.

Pipeline Modes:
- unified: Single-stage pipeline with self-routing (mimics original agent loop)
- parallel: Multi-stage pipeline with parallel analysis

v2.0 Features:
- L4 Working Memory (session persistence)
- L7 Tool Health Monitoring
- Event Streaming
- Checkpointing
- Prometheus Metrics
- Event Logging

Constitutional Reference:
- Capability owns orchestration logic
- Infrastructure provides runtime (LLM, tools, persistence)
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
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

# Optional: CommBus for memory operations
try:
    from control_tower.ipc.commbus import InMemoryCommBus, CommBusProtocol
    HAS_COMMBUS = True
except ImportError:
    HAS_COMMBUS = False
    InMemoryCommBus = None
    CommBusProtocol = None

# v2.0 Services
from minisweagent.capability.services import (
    WorkingMemoryService,
    ToolHealthService,
    EventStreamService,
    CheckpointService,
    EventLogService,
)
from minisweagent.capability.services.working_memory_service import (
    WorkingMemory,
    Finding,
    FocusState,
)
from minisweagent.capability.services.event_stream_service import EventCategory
from minisweagent.capability.observability.metrics import MetricsExporter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
    """Configuration for the SWE orchestrator (v2.0)."""

    # Execution limits
    max_iterations: int = 50
    max_llm_calls: int = 100
    max_agent_hops: int = 200
    step_limit: int = 0  # 0 = disabled
    cost_limit: float = 3.0
    timeout: int = 30
    confirm_commands: bool = True
    pipeline_mode: str = "unified"  # "unified" (single-stage) or "parallel" (multi-stage)

    # v2.0: Database configuration
    database_url: Optional[str] = field(
        default_factory=lambda: os.getenv("MSWEA_DATABASE_URL")
    )

    # v2.0: Session configuration
    enable_sessions: bool = True
    session_ttl_seconds: int = 86400  # 24 hours

    # v2.0: Observability
    enable_metrics: bool = False
    metrics_port: int = 9090
    enable_event_streaming: bool = True
    enable_checkpoints: bool = True
    enable_event_log: bool = True


class SWEOrchestrator:
    """Orchestrator for SWE agent execution (v2.0).

    All execution flows through jeeves-core's PipelineRunner:
    - unified: Single-stage pipeline with self-routing (mimics original agent loop)
    - parallel: Multi-stage pipeline with parallel analysis

    v2.0 Features:
    - Session persistence via WorkingMemoryService
    - Tool health monitoring via ToolHealthService
    - Real-time event streaming via EventStreamService
    - Pipeline checkpointing via CheckpointService
    - Audit logging via EventLogService
    - Prometheus metrics via MetricsExporter

    Constitutional Reference:
    - Capability owns orchestration logic
    - Infrastructure provides runtime (LLM, tools, persistence)
    """

    def __init__(
        self,
        config: SWEOrchestratorConfig,
        llm_factory: Optional[Callable[[str], LLMProvider]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        log: Optional[Logger] = None,
        persistence: Optional[Persistence] = None,
        prompt_registry: Optional[PromptRegistry] = None,
        control_tower: Optional[Any] = None,
        commbus: Optional[Any] = None,
    ):
        self.config = config
        self.llm_factory = llm_factory
        self.tool_executor = tool_executor
        self.log = log or _NullLogger()
        self.persistence = persistence
        self.prompt_registry = prompt_registry
        self.control_tower = control_tower
        self.commbus = commbus
        self._pipeline_runner: Optional[PipelineRunner] = None

        # v2.0: Database and services
        self._db_pool = None
        self._services_initialized = False

        # v2.0: Service instances (initialized lazily)
        self.working_memory_service: Optional[WorkingMemoryService] = None
        self.tool_health_service: Optional[ToolHealthService] = None
        self.event_stream_service: Optional[EventStreamService] = None
        self.checkpoint_service: Optional[CheckpointService] = None
        self.event_log_service: Optional[EventLogService] = None
        self.metrics_exporter: Optional[MetricsExporter] = None

        # Initialize event stream (in-memory, no DB required)
        if config.enable_event_streaming:
            self.event_stream_service = EventStreamService()

        # Initialize metrics exporter
        if config.enable_metrics:
            self.metrics_exporter = MetricsExporter(
                port=config.metrics_port,
                enabled=True
            )

    async def _init_database(self):
        """Initialize database connection pool (v2.0)."""
        if self._db_pool is not None:
            return self._db_pool

        if not self.config.database_url:
            logger.warning("No database URL configured. v2.0 features disabled.")
            return None

        try:
            import asyncpg
            self._db_pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=2,
                max_size=10
            )
            logger.info("Database connection pool initialized")
            return self._db_pool
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None

    async def _init_services(self):
        """Initialize v2.0 services."""
        if self._services_initialized:
            return

        db = await self._init_database()
        if db is None:
            self._services_initialized = True
            return

        # Initialize database-backed services
        if self.config.enable_sessions:
            self.working_memory_service = WorkingMemoryService(db)
            logger.info("WorkingMemoryService initialized")

        self.tool_health_service = ToolHealthService(db)
        logger.info("ToolHealthService initialized")

        if self.config.enable_checkpoints:
            self.checkpoint_service = CheckpointService(db)
            logger.info("CheckpointService initialized")

        if self.config.enable_event_log:
            self.event_log_service = EventLogService(db)
            logger.info("EventLogService initialized")

        # Start metrics server if enabled
        if self.metrics_exporter:
            self.metrics_exporter.start_server()

        self._services_initialized = True
        logger.info("All v2.0 services initialized")

    # =========================================================================
    # CONTROL TOWER INTEGRATION
    # =========================================================================

    def _record_resource_usage(
        self,
        pid: str,
        llm_calls: int = 0,
        tool_calls: int = 0,
        agent_hops: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> Optional[str]:
        """Record resource usage via ControlTower.

        Args:
            pid: Process ID (envelope_id)
            llm_calls: Number of LLM calls to record
            tool_calls: Number of tool calls to record
            agent_hops: Number of agent hops to record
            tokens_in: Input tokens used
            tokens_out: Output tokens used

        Returns:
            Quota exceeded reason if any, None otherwise
        """
        if self.control_tower is None:
            return None

        quota_exceeded = None

        # Record LLM calls
        for _ in range(llm_calls):
            result = self.control_tower.record_llm_call(
                pid, tokens_in=tokens_in // max(llm_calls, 1),
                tokens_out=tokens_out // max(llm_calls, 1)
            )
            if result:
                quota_exceeded = result

        # Record tool calls
        for _ in range(tool_calls):
            result = self.control_tower.record_tool_call(pid)
            if result:
                quota_exceeded = result

        # Record agent hops
        for _ in range(agent_hops):
            result = self.control_tower.record_agent_hop(pid)
            if result:
                quota_exceeded = result

        return quota_exceeded

    async def _query_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Query session state via CommBus if available.

        Args:
            session_id: Session ID to query

        Returns:
            Session state dict or None if unavailable
        """
        if self.commbus is None or not HAS_COMMBUS:
            return None

        try:
            from memory_module.messages import GetSessionState
            result = await self.commbus.query(GetSessionState(session_id=session_id))
            return result
        except Exception as e:
            logger.debug(f"Failed to query session state: {e}")
            return None

    async def _update_session_focus(
        self,
        session_id: str,
        focus_type: str,
        focus_id: str,
        focus_label: str = "",
    ) -> None:
        """Update session focus via CommBus if available.

        Args:
            session_id: Session ID
            focus_type: Type of focus (e.g., "file", "function")
            focus_id: ID of the focused item
            focus_label: Human-readable label
        """
        if self.commbus is None or not HAS_COMMBUS:
            return

        try:
            from memory_module.messages import UpdateFocus
            await self.commbus.send(UpdateFocus(
                session_id=session_id,
                focus_type=focus_type,
                focus_id=focus_id,
                focus_label=focus_label,
            ))
        except Exception as e:
            logger.debug(f"Failed to update session focus: {e}")

    async def close(self):
        """Clean up resources."""
        if self._db_pool:
            await self._db_pool.close()
            self._db_pool = None
            logger.info("Database connection pool closed")

    def _get_pipeline_runner(self) -> PipelineRunner:
        """Get or create the pipeline runner for parallel execution."""
        if self._pipeline_runner is None:
            pipeline_config = self._create_pipeline_config()
            self._pipeline_runner = create_pipeline_runner(
                config=pipeline_config,
                llm_provider_factory=self.llm_factory,
                tool_executor=self.tool_executor,
                logger=self.log,
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
        resume_checkpoint: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute an SWE task via pipeline (v2.0).

        Args:
            task: The task description
            user_id: User identifier
            session_id: Session identifier for working memory persistence
            thread_id: Thread ID for state persistence
            resume_checkpoint: Checkpoint ID to resume from (v2.0)
            **kwargs: Additional parameters

        Returns:
            Dict with execution results including:
            - status: Execution status
            - output: Final output
            - outputs: All stage outputs
            - llm_calls: Number of LLM calls
            - agent_hops: Number of agent transitions
            - session_id: Session ID (if session persistence enabled)
            - findings_count: Number of findings (if sessions enabled)
        """
        # v2.0: Initialize services
        await self._init_services()

        # v2.0: Generate session ID if not provided but sessions enabled
        if self.config.enable_sessions and not session_id:
            session_id = f"session_{int(time.time())}"

        return await self._run_pipeline(
            task, user_id, session_id, thread_id,
            resume_checkpoint=resume_checkpoint,
            **kwargs
        )

    async def _run_pipeline(
        self,
        task: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        resume_checkpoint: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run task via jeeves-core PipelineRunner (v2.0).

        Uses either unified (single-stage) or parallel (multi-stage) pipeline
        based on config.pipeline_mode.

        v2.0 enhancements:
        - Loads working memory from previous session
        - Emits events during execution
        - Creates checkpoints at each stage
        - Records metrics
        - Saves working memory after completion
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        self.log.info("swe_pipeline_started", task=task[:100], session_id=session_id)

        # v2.0: Emit pipeline start event
        if self.event_stream_service:
            await self.event_stream_service.emit(
                EventCategory.INFO,
                f"Pipeline started: {task[:50]}...",
                metadata={"session_id": session_id, "pipeline_mode": self.config.pipeline_mode}
            )

        # v2.0: Log event
        if self.event_log_service:
            await self.event_log_service.log_event(
                session_id=session_id,
                event_category="pipeline",
                event_type="started",
                payload={"task": task, "pipeline_mode": self.config.pipeline_mode}
            )

        # v2.0: Load working memory from session
        working_memory = None
        if self.working_memory_service and session_id:
            working_memory = await self.working_memory_service.load_session(session_id)
            if working_memory:
                logger.info(f"Loaded session {session_id}: {len(working_memory.findings)} findings")

        # v2.0: Check for checkpoint resume
        checkpoint_envelope_state = None
        if resume_checkpoint and self.checkpoint_service:
            checkpoint = await self.checkpoint_service.load_checkpoint(resume_checkpoint)
            if checkpoint:
                checkpoint_envelope_state = checkpoint.envelope_state
                logger.info(f"Resuming from checkpoint: {checkpoint.agent_name}")

        runner = self._get_pipeline_runner()

        # Create request context
        request_context = RequestContext(
            request_id=request_id,
            capability="mini-swe-agent",
            user_id=user_id,
            session_id=session_id,
        )

        # Build metadata with working memory context
        metadata = dict(kwargs)
        if working_memory:
            metadata["working_memory"] = {
                "session_id": working_memory.session_id,
                "findings_count": len(working_memory.findings),
                "previous_findings": [
                    {"id": f.id, "content": f.content, "source": f.source}
                    for f in working_memory.findings[-10:]  # Last 10 findings
                ],
                "entity_refs_count": len(working_memory.entity_refs),
            }
            if working_memory.focus_state:
                metadata["focus_state"] = {
                    "current_file": working_memory.focus_state.current_file,
                    "current_function": working_memory.focus_state.current_function,
                    "current_task": working_memory.focus_state.current_task,
                }

        # Create envelope
        envelope = create_envelope(
            raw_input=task,
            request_context=request_context,
            metadata=metadata,
        )

        # Restore from checkpoint if available
        if checkpoint_envelope_state:
            envelope.outputs = checkpoint_envelope_state.get("outputs", {})

        try:
            # v2.0: Emit agent start event
            if self.event_stream_service:
                await self.event_stream_service.emit(
                    EventCategory.AGENT_STARTED,
                    f"Starting {self.config.pipeline_mode} pipeline",
                    agent_name=self.config.pipeline_mode
                )

            # Run pipeline
            result_envelope = await runner.run(envelope, thread_id=thread_id)

            # v2.0: Emit completion event
            if self.event_stream_service:
                await self.event_stream_service.emit(
                    EventCategory.AGENT_COMPLETED,
                    "Pipeline completed",
                    agent_name=self.config.pipeline_mode,
                    metadata={"llm_calls": result_envelope.llm_call_count}
                )

            # Record resource usage via ControlTower
            if self.control_tower:
                quota_exceeded = self._record_resource_usage(
                    pid=envelope.envelope_id,
                    llm_calls=result_envelope.llm_call_count,
                    agent_hops=result_envelope.agent_hop_count,
                )
                if quota_exceeded:
                    logger.warning(f"Resource quota exceeded: {quota_exceeded}")

            # v2.0: Save checkpoint
            if self.checkpoint_service:
                await self.checkpoint_service.save_checkpoint(
                    checkpoint_id=f"{session_id}_{request_id}",
                    session_id=session_id,
                    agent_name="completed",
                    envelope_state={"outputs": result_envelope.outputs},
                    next_agent=None
                )

            # v2.0: Extract and save findings to working memory
            if self.working_memory_service and session_id:
                new_findings = self._extract_findings(result_envelope, task)
                if working_memory:
                    working_memory.findings.extend(new_findings)
                    working_memory.metadata["last_task"] = task
                else:
                    working_memory = WorkingMemory(
                        session_id=session_id,
                        findings=new_findings,
                        metadata={"last_task": task}
                    )
                await self.working_memory_service.save_session(
                    working_memory,
                    ttl_seconds=self.config.session_ttl_seconds
                )
                logger.info(f"Saved session {session_id}: {len(working_memory.findings)} findings")

            # v2.0: Record metrics
            duration = time.time() - start_time
            if self.metrics_exporter:
                status = "success" if not result_envelope.terminated else "terminated"
                self.metrics_exporter.record_pipeline_execution(
                    pipeline_mode=self.config.pipeline_mode,
                    status=status,
                    duration=duration
                )

            # v2.0: Log completion event
            if self.event_log_service:
                await self.event_log_service.log_event(
                    session_id=session_id,
                    event_category="pipeline",
                    event_type="completed",
                    payload={
                        "duration": duration,
                        "llm_calls": result_envelope.llm_call_count,
                        "agent_hops": result_envelope.agent_hop_count,
                    }
                )

            return {
                "status": "completed" if not result_envelope.terminated else result_envelope.terminal_reason,
                "output": result_envelope.outputs.get("verification", {}).get("response", ""),
                "outputs": result_envelope.outputs,
                "llm_calls": result_envelope.llm_call_count,
                "agent_hops": result_envelope.agent_hop_count,
                "duration": duration,
                "session_id": session_id,
                "findings_count": len(working_memory.findings) if working_memory else 0,
            }

        except Exception as e:
            self.log.error("swe_pipeline_error", error=str(e))

            # v2.0: Emit error event
            if self.event_stream_service:
                await self.event_stream_service.emit(
                    EventCategory.ERROR,
                    f"Pipeline error: {str(e)}",
                    agent_name=self.config.pipeline_mode
                )

            # v2.0: Record error metrics
            duration = time.time() - start_time
            if self.metrics_exporter:
                self.metrics_exporter.record_pipeline_execution(
                    pipeline_mode=self.config.pipeline_mode,
                    status="error",
                    duration=duration
                )

            # v2.0: Log error event
            if self.event_log_service:
                await self.event_log_service.log_event(
                    session_id=session_id,
                    event_category="pipeline",
                    event_type="error",
                    payload={"error": str(e), "duration": duration}
                )

            return {
                "status": "error",
                "error": str(e),
                "duration": duration,
                "session_id": session_id,
            }

    def _extract_findings(self, envelope, task: str) -> List[Finding]:
        """Extract findings from pipeline result (v2.0).

        Analyzes pipeline outputs to extract discovered facts.
        """
        findings = []
        finding_id = 0

        # Extract from search results
        search_results = envelope.outputs.get("search_results", {})
        if isinstance(search_results, dict):
            files_found = search_results.get("files", [])
            for file_path in files_found[:5]:  # Limit to 5
                finding_id += 1
                findings.append(Finding(
                    id=f"f_{int(time.time())}_{finding_id}",
                    content=f"Found relevant file: {file_path}",
                    source="code_searcher",
                    confidence=0.8,
                ))

        # Extract from analysis results
        analysis = envelope.outputs.get("analysis_results", {})
        if isinstance(analysis, dict):
            for key, value in list(analysis.items())[:3]:
                finding_id += 1
                findings.append(Finding(
                    id=f"f_{int(time.time())}_{finding_id}",
                    content=f"{key}: {str(value)[:100]}",
                    source="file_analyzer",
                    confidence=0.7,
                ))

        # Extract from plan
        plan = envelope.outputs.get("plan", {})
        if isinstance(plan, dict) and "steps" in plan:
            finding_id += 1
            findings.append(Finding(
                id=f"f_{int(time.time())}_{finding_id}",
                content=f"Plan created with {len(plan.get('steps', []))} steps",
                source="planner",
                confidence=0.9,
            ))

        # Basic finding for task completion
        if not findings:
            finding_id += 1
            findings.append(Finding(
                id=f"f_{int(time.time())}_{finding_id}",
                content=f"Task processed: {task[:100]}",
                source="pipeline",
                confidence=0.5,
            ))

        return findings

    async def run_streaming(
        self,
        task: str,
        user_id: str = "anonymous",
        session_id: str = "",
        thread_id: str = "",
        **kwargs,
    ):
        """Execute with streaming outputs (v2.0).

        Yields stage outputs as they complete.
        Also emits events and records metrics.
        """
        # v2.0: Initialize services
        await self._init_services()

        # v2.0: Generate session ID if needed
        if self.config.enable_sessions and not session_id:
            session_id = f"session_{int(time.time())}"

        runner = self._get_pipeline_runner()

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

        # v2.0: Emit start event
        if self.event_stream_service:
            await self.event_stream_service.emit(
                EventCategory.INFO,
                f"Streaming pipeline started: {task[:50]}..."
            )

        async for stage_name, output in runner.run_streaming(envelope, thread_id=thread_id):
            # v2.0: Emit stage event
            if self.event_stream_service:
                await self.event_stream_service.emit(
                    EventCategory.AGENT_COMPLETED,
                    f"Stage {stage_name} completed",
                    agent_name=stage_name
                )
            yield (stage_name, output)

    # =========================================================================
    # v2.0 SESSION MANAGEMENT HELPERS
    # =========================================================================

    async def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List active sessions (v2.0).

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries
        """
        await self._init_services()
        if not self.working_memory_service:
            return []
        return await self.working_memory_service.list_sessions(limit)

    async def delete_session(self, session_id: str):
        """Delete a session (v2.0).

        Args:
            session_id: Session to delete
        """
        await self._init_services()
        if self.working_memory_service:
            await self.working_memory_service.delete_session(session_id)
        if self.checkpoint_service:
            await self.checkpoint_service.delete_session_checkpoints(session_id)

    async def get_session(self, session_id: str) -> Optional[WorkingMemory]:
        """Get session working memory (v2.0).

        Args:
            session_id: Session ID

        Returns:
            WorkingMemory if found, None otherwise
        """
        await self._init_services()
        if not self.working_memory_service:
            return None
        return await self.working_memory_service.load_session(session_id)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_swe_orchestrator(
    llm_factory: Optional[Callable[[str], LLMProvider]] = None,
    tool_executor: Optional[ToolExecutor] = None,
    log: Optional[Logger] = None,
    persistence: Optional[Persistence] = None,
    prompt_registry: Optional[PromptRegistry] = None,
    control_tower: Optional[Any] = None,
    commbus: Optional[Any] = None,
    **config_kwargs,
) -> SWEOrchestrator:
    """Factory function to create an SWE orchestrator (v2.0).

    Called by infrastructure to create the orchestrator service.

    Args:
        llm_factory: Factory to create LLM providers by role
        tool_executor: Tool executor instance
        log: Logger instance
        persistence: State persistence
        prompt_registry: Prompt template registry
        control_tower: Control tower for lifecycle management (jeeves-core kernel)
        commbus: CommBus for inter-service communication (memory queries, events)
        **config_kwargs: Additional config options including v2.0:
            - database_url: PostgreSQL connection URL
            - enable_sessions: Enable session persistence
            - enable_metrics: Enable Prometheus metrics
            - metrics_port: Prometheus metrics port
            - enable_event_streaming: Enable real-time events
            - enable_checkpoints: Enable pipeline checkpointing
            - enable_event_log: Enable audit logging

    Returns:
        SWEOrchestrator instance

    Example:
        # With jeeves-core context
        from minisweagent.capability.wiring import create_jeeves_context

        context = create_jeeves_context()
        orchestrator = create_swe_orchestrator(
            control_tower=context.control_tower,
            commbus=context.commbus,
            llm_factory=my_factory,
        )
    """
    config = SWEOrchestratorConfig(**config_kwargs)

    return SWEOrchestrator(
        config=config,
        llm_factory=llm_factory,
        tool_executor=tool_executor,
        log=log,
        persistence=persistence,
        prompt_registry=prompt_registry,
        control_tower=control_tower,
        commbus=commbus,
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
