# Jeeves-Core Capability Wiring Plan (v2.0)

**Date**: 2026-01-27
**Status**: Planning Phase
**Target**: Wire all latent jeeves-core capabilities
**Compatibility**: **Breaking changes allowed** (v2.0 major version bump)

---

## Executive Summary

This document provides a comprehensive implementation plan to wire the **85% of jeeves-core infrastructure** currently unused by mini-swe-agent. The plan assumes **no backward compatibility constraints** and targets a **v2.0 release** with breaking changes.

**Scope**: 10 major capability integrations over 3 phases (6-8 weeks total)

**Breaking Changes**:
- New database schema (PostgreSQL required)
- New CLI arguments and configuration format
- New tool signatures (async-only)
- New agent protocol (working memory injection)
- New session management (session IDs required for persistence)

---

## Unwired Capabilities (Audit Summary)

| Capability | Readiness | Impact | Effort | Phase |
|-----------|-----------|--------|--------|-------|
| **L4 Working Memory** | 100% | ⭐⭐⭐ | 1 day | 1 |
| **L7 Tool Health** | 100% | ⭐⭐ | 1 day | 1 |
| **Event Streaming** | 90% | ⭐⭐ | 1 day | 1 |
| **L3 Semantic Search** | 100% | ⭐⭐⭐ | 2 days | 2 |
| **L5 Graph Storage** | 100% | ⭐⭐⭐ | 3 days | 2 |
| **Clarification Interrupts** | 100% | ⭐⭐ | 2 days | 2 |
| **Prometheus Metrics** | 100% | ⭐⭐ | 1 day | 3 |
| **NLI Verification** | 100% | ⭐ | 1 day | 3 |
| **Checkpointing** | 80% | ⭐ | 3 days | 3 |
| **L2 Event Log** | 100% | ⭐ | 1 day | 3 |

**Total**: 16 days effort across 3 phases

---

## Phase 1: Foundation (Week 1)
**Goal**: Wire core memory and observability infrastructure
**Duration**: 3 days
**Dependencies**: PostgreSQL database, jeeves-core configured

### 1.1 L4 Working Memory (Session State)

#### **What It Does**
Persist session state across queries, enabling:
- Resume previous analysis sessions
- Share context between pipeline runs
- Store findings, focus state, entity references
- Incremental codebase understanding

#### **Breaking Changes**
1. **Database Required**: PostgreSQL database now mandatory (no in-memory fallback)
2. **New CLI Args**: `--session <id>` required for persistence
3. **New Config**: `database_url` in config file
4. **Agent Protocol**: Agents receive `working_memory` in context

#### **Implementation**

##### **Step 1: Add Database Configuration**
**File**: `src/minisweagent/config/mini.yaml`

```yaml
# NEW SECTION: Database configuration (BREAKING)
database:
  url: "postgresql://localhost:5432/mini_swe_agent"
  pool_size: 10
  max_overflow: 20
  echo: false  # SQL logging

# NEW SECTION: Session configuration
session:
  enable_persistence: true
  default_ttl: 86400  # 24 hours
  max_session_size_mb: 100
```

**Migration**:
```bash
# Users must set up PostgreSQL
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15
export MSWEA_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/mini_swe_agent"
```

##### **Step 2: Initialize Database Schema**
**File**: `src/minisweagent/capability/db/migrations/001_working_memory.sql` (NEW)

```sql
-- Session state table
CREATE TABLE IF NOT EXISTS session_state (
    session_id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    focus_state JSONB,  -- Current attention focus
    findings JSONB,     -- Discovered facts
    entity_refs JSONB,  -- Code entities (files, functions)
    metadata JSONB,     -- Additional context
    ttl_seconds INTEGER DEFAULT 86400
);

CREATE INDEX idx_session_updated ON session_state(updated_at);
CREATE INDEX idx_session_findings ON session_state USING gin(findings);

-- Automatic cleanup trigger
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    DELETE FROM session_state
    WHERE updated_at < NOW() - INTERVAL '1 second' * ttl_seconds;
END;
$$ LANGUAGE plpgsql;
```

**Migration Command**:
```bash
# NEW CLI: mini-jeeves db migrate
mini-jeeves db migrate --apply
```

##### **Step 3: Wire Session Service into Orchestrator**
**File**: `src/minisweagent/capability/orchestrator.py`

**BREAKING CHANGE**: New constructor signature

```python
# BEFORE (old)
class SWEOrchestrator:
    def __init__(self, config: SWEOrchestratorConfig):
        self.config = config
        self.runner = PipelineRunner(...)

# AFTER (new - BREAKING)
from mission_system.adapters import create_session_state_service, create_database_client

class SWEOrchestrator:
    def __init__(self, config: SWEOrchestratorConfig, database_url: str):
        self.config = config
        self.db = create_database_client(database_url)
        self.session_service = create_session_state_service(self.db)  # NEW
        self.runner = PipelineRunner(...)

    async def run(
        self,
        task: str,
        session_id: Optional[str] = None,  # NEW parameter
    ) -> PipelineResult:
        # Load previous session if provided
        working_memory = None
        if session_id:
            working_memory = await self.session_service.load_session(session_id)
            logger.info(f"Loaded session {session_id}: {len(working_memory.findings)} findings")

        # Inject working memory into envelope
        envelope = Envelope(
            request_id=generate_id(),
            task=task,
            context={
                "working_memory": working_memory,  # NEW
                "session_id": session_id,
            },
        )

        # Run pipeline
        result = await self.runner.execute(envelope)

        # Save session state
        if session_id and result.status == "success":
            updated_memory = self._extract_working_memory(envelope)
            await self.session_service.save_session(session_id, updated_memory)
            logger.info(f"Saved session {session_id}: {len(updated_memory.findings)} findings")

        return result

    def _extract_working_memory(self, envelope: Envelope) -> WorkingMemory:
        """Extract working memory from envelope context."""
        return WorkingMemory(
            session_id=envelope.context["session_id"],
            focus_state=envelope.context.get("focus_state"),
            findings=envelope.context.get("findings", []),
            entity_refs=envelope.context.get("entity_refs", []),
            metadata=envelope.context.get("metadata", {}),
        )
```

##### **Step 4: Update Agents to Use Working Memory**
**File**: `src/minisweagent/capability/prompts/registry.py`

**BREAKING CHANGE**: New prompt variables

```python
class MiniSWEPromptRegistry:
    SYSTEM_PROMPTS = {
        "code_searcher": """You are a code searcher agent.

## Previous Findings (from working memory)
{previous_findings}

## Current Task
{task}

## Instructions
1. Review previous findings to avoid duplicate work
2. Search for new relevant code
3. Add discoveries to findings list
""",

        "planner": """You are a planning agent.

## Session Context
{session_context}

## Previous Analysis
Files analyzed: {analyzed_files}
Functions discovered: {discovered_functions}

## Current Task
{task}

## Instructions
1. Leverage previous analysis to create an efficient plan
2. Focus on gaps in understanding
3. Build on previous findings
""",
    }
```

**Agent Update**:
```python
# File: src/minisweagent/capability/orchestrator.py (agent execution)

async def execute_agent(agent_name: str, envelope: Envelope):
    # Extract working memory
    working_memory = envelope.context.get("working_memory")

    # Build prompt with memory context
    prompt_vars = {
        "task": envelope.task,
        "previous_findings": format_findings(working_memory.findings) if working_memory else "None",
        "session_context": format_context(working_memory) if working_memory else "New session",
        "analyzed_files": extract_files(working_memory) if working_memory else [],
    }

    # LLM call with memory-enhanced prompt
    response = await llm.complete(
        system=registry.get_prompt(agent_name).format(**prompt_vars),
        messages=envelope.messages,
    )

    # Update working memory with new findings
    if response.findings:
        envelope.context.setdefault("findings", []).extend(response.findings)
```

##### **Step 5: Update CLI to Support Sessions**
**File**: `src/minisweagent/run/mini_jeeves.py`

**BREAKING CHANGE**: New required arguments

```python
@app.command()
def run(
    task: str = typer.Option(..., "--task", "-t", help="Task to execute"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID for persistence"),  # NEW
    new_session: bool = typer.Option(False, "--new-session", help="Create new session"),  # NEW
    list_sessions: bool = typer.Option(False, "--list-sessions", help="List active sessions"),  # NEW
    database_url: Optional[str] = typer.Option(None, "--database-url", envvar="MSWEA_DATABASE_URL"),  # NEW
    # ... existing args ...
):
    """Run mini-jeeves with session persistence."""

    # List sessions
    if list_sessions:
        sessions = await orchestrator.session_service.list_sessions()
        table = Table(title="Active Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Created", style="green")
        table.add_column("Findings", style="yellow")
        for session in sessions:
            table.add_row(session.id, session.created_at, str(len(session.findings)))
        console.print(table)
        return

    # Generate session ID if needed
    if new_session:
        session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        console.print(f"[green]Created new session: {session}[/green]")

    # Run with session
    orchestrator = SWEOrchestrator(config, database_url)
    result = await orchestrator.run(task, session_id=session)

    if session:
        console.print(f"[blue]Session {session} saved. Resume with: --session {session}[/blue]")
```

##### **Usage Examples**

```bash
# Create new session
mini-jeeves -t "Find authentication code" --new-session
# Output: Created new session: session_20260127_143022
# Output: Session session_20260127_143022 saved. Resume with: --session session_20260127_143022

# Resume session (10x faster, reuses findings)
mini-jeeves -t "Fix the auth bug" --session session_20260127_143022

# List active sessions
mini-jeeves --list-sessions
# Output:
# ┌─────────────────────────┬────────────┬──────────┐
# │ Session ID              │ Created    │ Findings │
# ├─────────────────────────┼────────────┼──────────┤
# │ session_20260127_143022 │ 2 hrs ago  │ 47       │
# │ session_20260127_091234 │ 1 day ago  │ 23       │
# └─────────────────────────┴────────────┴──────────┘
```

##### **Testing**

**File**: `tests/capability/test_working_memory.py` (NEW)

```python
import pytest
from minisweagent.capability.orchestrator import SWEOrchestrator

@pytest.mark.asyncio
async def test_session_persistence(db_url):
    orchestrator = SWEOrchestrator(config, db_url)

    # First query: Analyze codebase
    result1 = await orchestrator.run(
        task="Find authentication code",
        session_id="test_session_001",
    )
    assert result1.status == "success"

    # Load session
    session = await orchestrator.session_service.load_session("test_session_001")
    assert len(session.findings) > 0

    # Second query: Should reuse findings (faster)
    start = time.time()
    result2 = await orchestrator.run(
        task="Fix authentication bug",
        session_id="test_session_001",
    )
    duration = time.time() - start

    assert result2.status == "success"
    assert duration < 5.0  # Should be fast (reused findings)

@pytest.mark.asyncio
async def test_session_isolation(db_url):
    """Verify sessions don't leak data."""
    orchestrator = SWEOrchestrator(config, db_url)

    # Session A
    await orchestrator.run("Task A", session_id="session_a")
    session_a = await orchestrator.session_service.load_session("session_a")

    # Session B
    await orchestrator.run("Task B", session_id="session_b")
    session_b = await orchestrator.session_service.load_session("session_b")

    # Verify isolation
    assert session_a.findings != session_b.findings
```

---

### 1.2 L7 Tool Health Monitoring

#### **What It Does**
Track tool invocation metrics and auto-quarantine failing tools:
- Record success/failure rates per tool
- Detect degraded tools (10-50% error rate)
- Quarantine failing tools (>50% error rate)
- Provide fallback strategies

#### **Breaking Changes**
1. **Database Required**: Tool metrics stored in PostgreSQL
2. **Tool Executor**: New async wrapper with health checks
3. **Tool Signatures**: All tools must be async
4. **Error Handling**: Tools return Result[T, Error] instead of raising

#### **Implementation**

##### **Step 1: Database Schema**
**File**: `src/minisweagent/capability/db/migrations/002_tool_health.sql` (NEW)

```sql
-- Tool health metrics table
CREATE TABLE IF NOT EXISTS tool_health (
    tool_name VARCHAR(255) PRIMARY KEY,
    invocation_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    total_latency_ms BIGINT DEFAULT 0,
    last_invocation TIMESTAMP,
    status VARCHAR(50) DEFAULT 'healthy',  -- healthy, degraded, quarantined
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Invocation history (for debugging)
CREATE TABLE IF NOT EXISTS tool_invocations (
    id SERIAL PRIMARY KEY,
    tool_name VARCHAR(255) NOT NULL,
    success BOOLEAN NOT NULL,
    latency_ms INTEGER NOT NULL,
    error_message TEXT,
    invoked_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_tool_invocations_tool ON tool_invocations(tool_name);
CREATE INDEX idx_tool_invocations_time ON tool_invocations(invoked_at);

-- Automatic status calculation trigger
CREATE OR REPLACE FUNCTION update_tool_status()
RETURNS TRIGGER AS $$
DECLARE
    error_rate FLOAT;
BEGIN
    -- Calculate error rate
    IF NEW.invocation_count > 0 THEN
        error_rate := NEW.failure_count::FLOAT / NEW.invocation_count;

        -- Update status
        IF error_rate >= 0.5 THEN
            NEW.status := 'quarantined';
        ELSIF error_rate >= 0.1 THEN
            NEW.status := 'degraded';
        ELSE
            NEW.status := 'healthy';
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tool_health_status_trigger
    BEFORE UPDATE ON tool_health
    FOR EACH ROW
    EXECUTE FUNCTION update_tool_status();
```

##### **Step 2: Wrap Tool Executor with Health Monitoring**
**File**: `src/minisweagent/capability/tools/confirming_executor.py`

**BREAKING CHANGE**: New constructor, async-only

```python
# BEFORE (old)
class ConfirmingToolExecutor:
    def execute(self, tool_name: str, params: Dict) -> Any:
        # Sync execution
        return self.tools[tool_name](**params)

# AFTER (new - BREAKING)
from mission_system.adapters import create_tool_health_service
from typing import Result, Ok, Err

class ConfirmingToolExecutor:
    def __init__(self, tools: Dict, db_client, confirmation_handler):
        self.tools = tools
        self.confirmation_handler = confirmation_handler
        self.health_service = create_tool_health_service(db_client)  # NEW

    async def execute(self, tool_name: str, params: Dict) -> Result[Any, str]:  # BREAKING: async + Result
        """Execute tool with health monitoring and confirmation."""

        # 1. Check tool health status
        status = await self.health_service.get_tool_status(tool_name)
        if status == "quarantined":
            logger.warning(f"Tool {tool_name} is quarantined (error rate >50%)")
            return Err(f"Tool {tool_name} unavailable due to high failure rate")

        if status == "degraded":
            logger.warning(f"Tool {tool_name} is degraded (error rate 10-50%)")
            # Continue but warn user

        # 2. Check if confirmation needed
        if self.tools[tool_name].risk_level == RiskLevel.HIGH:
            approved = await self.confirmation_handler.confirm(tool_name, params)
            if not approved:
                return Err(f"Tool {tool_name} execution denied by user")

        # 3. Execute with timing
        start_time = time.time()
        try:
            result = await self.tools[tool_name].execute(**params)  # Must be async
            latency_ms = int((time.time() - start_time) * 1000)

            # Record success
            await self.health_service.record_invocation(
                tool_name=tool_name,
                success=True,
                latency_ms=latency_ms,
            )

            return Ok(result)

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)

            # Record failure
            await self.health_service.record_invocation(
                tool_name=tool_name,
                success=False,
                latency_ms=latency_ms,
                error_message=str(e),
            )

            logger.error(f"Tool {tool_name} failed: {e}")
            return Err(str(e))
```

##### **Step 3: Convert All Tools to Async**
**File**: `src/minisweagent/capability/tools/catalog.py`

**BREAKING CHANGE**: All tools must be async

```python
# BEFORE (sync)
def bash_execute(command: str, timeout: int = 30) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, timeout=timeout)
    return result.stdout.decode()

# AFTER (async - BREAKING)
async def bash_execute(command: str, timeout: int = 30) -> str:
    """Execute bash command with timeout."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout.decode()
    except asyncio.TimeoutError:
        proc.kill()
        raise TimeoutError(f"Command timed out after {timeout}s")

# BEFORE (sync)
def read_file(file_path: str) -> str:
    with open(file_path) as f:
        return f.read()

# AFTER (async - BREAKING)
async def read_file(file_path: str) -> str:
    """Read file contents asynchronously."""
    async with aiofiles.open(file_path, 'r') as f:
        return await f.read()
```

##### **Step 4: Add Tool Health CLI Commands**
**File**: `src/minisweagent/run/mini_jeeves.py`

**NEW COMMANDS**:

```python
@app.command()
def tool_health(
    database_url: str = typer.Option(..., "--database-url", envvar="MSWEA_DATABASE_URL"),
):
    """Display tool health metrics."""
    health_service = create_tool_health_service(create_database_client(database_url))
    metrics = await health_service.get_all_metrics()

    table = Table(title="Tool Health Metrics")
    table.add_column("Tool", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Success Rate", style="green")
    table.add_column("Avg Latency", style="yellow")
    table.add_column("Total Calls", style="blue")

    for metric in metrics:
        status_color = {
            "healthy": "green",
            "degraded": "yellow",
            "quarantined": "red",
        }[metric.status]

        success_rate = f"{metric.success_rate:.1%}"
        avg_latency = f"{metric.avg_latency_ms:.0f}ms"

        table.add_row(
            metric.tool_name,
            f"[{status_color}]{metric.status}[/{status_color}]",
            success_rate,
            avg_latency,
            str(metric.invocation_count),
        )

    console.print(table)

@app.command()
def tool_reset(
    tool_name: str = typer.Argument(..., help="Tool name to reset"),
    database_url: str = typer.Option(..., "--database-url", envvar="MSWEA_DATABASE_URL"),
):
    """Reset tool health metrics (unquarantine)."""
    health_service = create_tool_health_service(create_database_client(database_url))
    await health_service.reset_metrics(tool_name)
    console.print(f"[green]Reset metrics for {tool_name}[/green]")
```

##### **Usage Examples**

```bash
# View tool health
mini-jeeves tool-health
# Output:
# ┌──────────────┬────────────┬──────────────┬─────────────┬────────────┐
# │ Tool         │ Status     │ Success Rate │ Avg Latency │ Total Call │
# ├──────────────┼────────────┼──────────────┼─────────────┼────────────┤
# │ bash_execute │ healthy    │ 95.2%        │ 234ms       │ 1,234      │
# │ read_file    │ healthy    │ 99.8%        │ 12ms        │ 5,678      │
# │ write_file   │ degraded   │ 87.3%        │ 45ms        │ 890        │
# │ run_tests    │ quarantine │ 42.1%        │ 12,345ms    │ 123        │
# └──────────────┴────────────┴──────────────┴─────────────┴────────────┘

# Reset quarantined tool
mini-jeeves tool-reset run_tests
```

##### **Testing**

```python
@pytest.mark.asyncio
async def test_tool_health_quarantine(db_url):
    """Verify tools are quarantined after repeated failures."""
    executor = ConfirmingToolExecutor(tools, db, handler)

    # Simulate 10 failures
    for _ in range(10):
        result = await executor.execute("failing_tool", {})
        assert result.is_err()

    # Check status
    status = await executor.health_service.get_tool_status("failing_tool")
    assert status == "quarantined"

    # Verify next execution is blocked
    result = await executor.execute("failing_tool", {})
    assert result.is_err()
    assert "quarantined" in result.unwrap_err()
```

---

### 1.3 Event Streaming to CLI

#### **What It Does**
Stream pipeline events to CLI in real-time:
- Agent started/completed events
- Tool execution events
- Progress indicators
- Error notifications

#### **Breaking Changes**
1. **Async CLI**: CLI runner must be async
2. **Event Loop**: New event loop for streaming
3. **Output Format**: Structured events instead of raw stdout

#### **Implementation**

##### **Step 1: Wire Event Aggregator**
**File**: `src/minisweagent/run/mini_jeeves.py`

**BREAKING CHANGE**: CLI runner is now async

```python
# BEFORE (sync)
def main(task: str):
    orchestrator = SWEOrchestrator(config)
    result = orchestrator.run(task)
    print(result)

# AFTER (async - BREAKING)
from control_tower.services.event_aggregator import EventAggregator
from protocols import EventCategory

async def main(task: str, stream: bool = True):
    """Run with event streaming."""
    orchestrator = SWEOrchestrator(config, db_url)

    if stream:
        # Stream events in real-time
        aggregator = EventAggregator()

        # Start pipeline in background
        pipeline_task = asyncio.create_task(
            orchestrator.run(task, session_id=session)
        )

        # Stream events to console
        with Live(console=console, refresh_per_second=4) as live:
            status_panel = create_status_panel()

            async for event in aggregator.stream_events():
                update_status_panel(status_panel, event)
                live.update(status_panel)

                # Check if pipeline done
                if pipeline_task.done():
                    break

        result = await pipeline_task
    else:
        # Non-streaming mode
        result = await orchestrator.run(task, session_id=session)

    return result

def create_status_panel() -> Panel:
    """Create rich panel for status display."""
    return Panel(
        Group(
            Text("Pipeline Status", style="bold cyan"),
            Table.grid(padding=(0, 2)),
        ),
        title="Mini-Jeeves",
        border_style="blue",
    )

def update_status_panel(panel: Panel, event: AgentEvent):
    """Update panel with new event."""
    if event.category == EventCategory.AGENT_STARTED:
        panel.renderable.renderables.append(
            Text(f"🤖 Starting: {event.agent_name}", style="green")
        )
    elif event.category == EventCategory.TOOL_EXECUTED:
        tool_name = event.metadata.get("tool_name")
        panel.renderable.renderables.append(
            Text(f"🔧 Executed: {tool_name}", style="yellow")
        )
    elif event.category == EventCategory.AGENT_COMPLETED:
        panel.renderable.renderables.append(
            Text(f"✅ Completed: {event.agent_name}", style="green")
        )
    elif event.category == EventCategory.ERROR:
        panel.renderable.renderables.append(
            Text(f"❌ Error: {event.message}", style="red bold")
        )
```

##### **Step 2: Add CLI Flag**

```python
@app.command()
def run(
    task: str = typer.Option(..., "-t", "--task"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable event streaming"),  # NEW
    # ... other args ...
):
    """Run with optional event streaming."""
    asyncio.run(main(task, stream=not no_stream))
```

##### **Usage Examples**

```bash
# With streaming (default)
mini-jeeves -t "Fix authentication bug"
# Output (live updating):
# ┌─────────────────────────────────────────┐
# │ Pipeline Status                         │
# ├─────────────────────────────────────────┤
# │ 🤖 Starting: task_parser                │
# │ ✅ Completed: task_parser               │
# │ 🤖 Starting: code_searcher              │
# │ 🔧 Executed: grep_search                │
# │ 🔧 Executed: read_file                  │
# │ ✅ Completed: code_searcher             │
# │ 🤖 Starting: planner                    │
# └─────────────────────────────────────────┘

# Without streaming
mini-jeeves -t "Fix bug" --no-stream
# Output (batch at end)
```

---

## Phase 2: Advanced Memory (Week 2-3)
**Goal**: Wire L3 semantic search and L5 graph storage
**Duration**: 7 days
**Dependencies**: Phase 1 complete, pgvector extension installed

### 2.1 L3 Semantic Code Search

#### **What It Does**
Natural language code search using embeddings:
- Index codebase with text embeddings
- Search by conceptual similarity
- Retrieve relevant code blocks
- 10x better than grep for conceptual queries

#### **Breaking Changes**
1. **PostgreSQL Extension**: Requires pgvector extension
2. **Indexing Step**: One-time codebase indexing required
3. **New Tool**: `semantic_search` tool added
4. **Embedding Model**: Downloads ~100MB model on first use

#### **Implementation**

##### **Step 1: Database Schema**
**File**: `src/minisweagent/capability/db/migrations/003_semantic_search.sql` (NEW)

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Semantic chunks table
CREATE TABLE IF NOT EXISTS semantic_chunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    source_file VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),  -- 384-dim embeddings (all-MiniLM-L6-v2)
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for similarity search
CREATE INDEX idx_chunks_embedding ON semantic_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_chunks_file ON semantic_chunks(source_file);

-- Full-text search index (fallback)
CREATE INDEX idx_chunks_content ON semantic_chunks USING gin(to_tsvector('english', content));
```

##### **Step 2: Add Semantic Search Tool**
**File**: `src/minisweagent/capability/tools/catalog.py`

**NEW TOOL**:

```python
from mission_system.adapters import create_code_indexer

@tool(
    name="semantic_search",
    description="Search codebase using natural language query",
    risk_level=RiskLevel.READ_ONLY,
    category=ToolCategory.READ,
)
async def semantic_search(
    query: str,
    limit: int = 5,
    min_score: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Search codebase using semantic similarity.

    Args:
        query: Natural language description of code to find
        limit: Maximum number of results
        min_score: Minimum similarity score (0-1)

    Returns:
        List of matching code blocks with scores

    Example:
        results = await semantic_search("password validation logic", limit=5)
    """
    indexer = create_code_indexer(db)
    results = await indexer.search(query, limit=limit, min_score=min_score)

    return [
        {
            "file": r.source_file,
            "content": r.content,
            "score": r.score,
            "line_start": r.metadata.get("line_start"),
            "line_end": r.metadata.get("line_end"),
        }
        for r in results
    ]
```

##### **Step 3: Add Indexing CLI Command**
**File**: `src/minisweagent/run/mini_jeeves.py`

**NEW COMMAND**:

```python
@app.command()
def index(
    repo_path: str = typer.Argument(".", help="Repository path to index"),
    file_pattern: str = typer.Option("**/*.py", "--pattern", help="File glob pattern"),
    database_url: str = typer.Option(..., "--database-url", envvar="MSWEA_DATABASE_URL"),
    chunk_size: int = typer.Option(512, "--chunk-size", help="Tokens per chunk"),
    batch_size: int = typer.Option(100, "--batch-size", help="Files per batch"),
):
    """Index codebase for semantic search."""
    indexer = create_code_indexer(create_database_client(database_url))

    # Find files
    files = glob.glob(f"{repo_path}/{file_pattern}", recursive=True)
    console.print(f"[cyan]Found {len(files)} files to index[/cyan]")

    # Index in batches with progress bar
    with Progress() as progress:
        task = progress.add_task("[green]Indexing...", total=len(files))

        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]

            for file_path in batch:
                try:
                    content = read_file(file_path)
                    await indexer.index_file(file_path, content, chunk_size=chunk_size)
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[red]Failed to index {file_path}: {e}[/red]")

    console.print(f"[green]✓ Indexed {len(files)} files[/green]")

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
    database_url: str = typer.Option(..., "--database-url", envvar="MSWEA_DATABASE_URL"),
):
    """Search codebase semantically (for testing)."""
    indexer = create_code_indexer(create_database_client(database_url))
    results = await indexer.search(query, limit=limit)

    for i, result in enumerate(results, 1):
        console.print(Panel(
            Syntax(result.content, "python", line_numbers=True),
            title=f"[cyan]{i}. {result.source_file}[/cyan] (score: {result.score:.2f})",
            border_style="green" if result.score > 0.9 else "yellow",
        ))
```

##### **Step 4: Update code_searcher Agent**
**File**: `src/minisweagent/capability/wiring.py`

**BREAKING CHANGE**: Add semantic_search to code_searcher tools

```python
AGENT_DEFINITIONS = [
    DomainAgentConfig(
        name="code_searcher",
        description="Searches codebase for relevant files and symbols",
        layer="execution",
        tools=["bash_execute", "find_files", "grep_search", "semantic_search"],  # NEW TOOL
    ),
    # ... other agents
]
```

##### **Step 5: Update Prompts**
**File**: `src/minisweagent/capability/prompts/registry.py`

```python
SYSTEM_PROMPTS = {
    "code_searcher": """You are a code search agent with access to:

1. **semantic_search**: Best for conceptual queries
   - Use for: "find authentication logic", "locate database models"
   - Returns top-N most relevant code blocks

2. **grep_search**: Best for exact pattern matching
   - Use for: specific function names, import statements
   - Returns all exact matches

3. **find_files**: Best for file path patterns
   - Use for: "find all test files", "locate config files"

## Strategy
1. Start with semantic_search for broad understanding
2. Use grep_search for specific symbols
3. Use find_files for path-based queries

## Examples
Task: "Find password validation"
→ semantic_search("password validation logic")

Task: "Find where User class is defined"
→ grep_search("class User", type="py")

Task: "Find all migration files"
→ find_files("**/migrations/*.py")
""",
}
```

##### **Usage Examples**

```bash
# Index codebase (one-time setup)
mini-jeeves index . --pattern "**/*.py"
# Output: Found 1,234 files to index
# Output: ✓ Indexed 1,234 files

# Test semantic search
mini-jeeves search "password validation logic" --limit 3
# Output:
# ┌────────────────────────────────────────────────────┐
# │ 1. src/auth/validators.py (score: 0.95)           │
# ├────────────────────────────────────────────────────┤
# │ def validate_password(password: str) -> bool:      │
# │     if len(password) < 8:                          │
# │         return False                               │
# │     if not any(c.isupper() for c in password):     │
# │         return False                               │
# │     return True                                    │
# └────────────────────────────────────────────────────┘

# Use in agent task
mini-jeeves -t "Fix the password validator to require special characters"
# Agent uses semantic_search internally:
# → semantic_search("password validation")
# → Returns validate_password() function
# → Agent modifies and tests
```

---

### 2.2 L5 Graph Storage (Entity Relationships)

#### **What It Does**
Model codebase as entity relationship graph:
- Track imports, calls, inheritance
- Query dependencies ("what depends on X?")
- Detect circular dependencies
- Impact analysis ("what breaks if I change X?")

#### **Breaking Changes**
1. **Graph Schema**: New tables for nodes/edges
2. **Post-Processing**: Automatic AST parsing after file operations
3. **New Tools**: `graph_query` tool added
4. **CLI Commands**: Graph visualization commands

#### **Implementation**

##### **Step 1: Database Schema**
**File**: `src/minisweagent/capability/db/migrations/004_graph_storage.sql` (NEW)

```sql
-- Graph nodes (code entities)
CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(50) NOT NULL,  -- file, class, function, variable
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_nodes_type ON graph_nodes(node_type);

-- Graph edges (relationships)
CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id SERIAL PRIMARY KEY,
    source_id VARCHAR(255) NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    target_id VARCHAR(255) NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL,  -- imports, calls, inherits, defines
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(source_id, target_id, edge_type)
);

CREATE INDEX idx_edges_source ON graph_edges(source_id);
CREATE INDEX idx_edges_target ON graph_edges(target_id);
CREATE INDEX idx_edges_type ON graph_edges(edge_type);

-- Materialized view for common queries
CREATE MATERIALIZED VIEW IF NOT EXISTS dependency_graph AS
SELECT
    source_id,
    target_id,
    edge_type,
    n1.metadata->>'file' as source_file,
    n2.metadata->>'file' as target_file
FROM graph_edges e
JOIN graph_nodes n1 ON e.source_id = n1.node_id
JOIN graph_nodes n2 ON e.target_id = n2.node_id
WHERE edge_type IN ('imports', 'calls', 'inherits');

CREATE INDEX idx_dep_graph_source ON dependency_graph(source_id);
CREATE INDEX idx_dep_graph_target ON dependency_graph(target_id);
```

##### **Step 2: Add Graph Extraction Post-Processor**
**File**: `src/minisweagent/capability/agents/swe_post_processor.py`

**BREAKING CHANGE**: Automatic graph extraction after file operations

```python
from mission_system.adapters import create_graph_storage
import ast

class GraphExtractor:
    """Extract code entities and relationships from Python files."""

    def __init__(self, graph_storage):
        self.graph = graph_storage

    async def extract_from_file(self, file_path: str, content: str):
        """Extract entities and relationships from file."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.warning(f"Failed to parse {file_path}")
            return

        # Add file node
        await self.graph.add_node(
            f"file:{file_path}",
            "file",
            {"path": file_path, "lines": len(content.splitlines())}
        )

        # Extract entities
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Add class node
                class_id = f"class:{file_path}:{node.name}"
                await self.graph.add_node(
                    class_id,
                    "class",
                    {"name": node.name, "file": file_path, "line": node.lineno}
                )

                # Link file → class
                await self.graph.add_edge(
                    f"file:{file_path}",
                    class_id,
                    "defines"
                )

                # Extract base classes (inheritance)
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_id = f"class:{file_path}:{base.id}"  # Simplified
                        await self.graph.add_edge(
                            class_id,
                            base_id,
                            "inherits"
                        )

            elif isinstance(node, ast.FunctionDef):
                # Add function node
                func_id = f"function:{file_path}:{node.name}"
                await self.graph.add_node(
                    func_id,
                    "function",
                    {"name": node.name, "file": file_path, "line": node.lineno}
                )

                # Link file → function
                await self.graph.add_edge(
                    f"file:{file_path}",
                    func_id,
                    "defines"
                )

            elif isinstance(node, ast.Import):
                # Add import edges
                for alias in node.names:
                    module = alias.name
                    await self.graph.add_edge(
                        f"file:{file_path}",
                        f"file:{module.replace('.', '/')}.py",
                        "imports"
                    )

            elif isinstance(node, ast.ImportFrom):
                # Add from-import edges
                if node.module:
                    await self.graph.add_edge(
                        f"file:{file_path}",
                        f"file:{node.module.replace('.', '/')}.py",
                        "imports"
                    )

# Hook into file operations
async def post_process_file_write(file_path: str, content: str):
    """Automatically extract graph after file write."""
    extractor = GraphExtractor(create_graph_storage(db))
    await extractor.extract_from_file(file_path, content)
```

##### **Step 3: Add Graph Query Tool**
**File**: `src/minisweagent/capability/tools/catalog.py`

**NEW TOOL**:

```python
@tool(
    name="graph_query",
    description="Query code dependency graph",
    risk_level=RiskLevel.READ_ONLY,
    category=ToolCategory.READ,
)
async def graph_query(
    query_type: str,  # "depends_on", "used_by", "path", "circular"
    node_id: str,
    max_depth: int = 3,
) -> List[Dict[str, Any]]:
    """
    Query code dependency graph.

    Args:
        query_type: Type of query
            - "depends_on": What does node_id depend on?
            - "used_by": What depends on node_id?
            - "path": Find path from node_id to target
            - "circular": Find circular dependencies from node_id
        node_id: Starting node (e.g., "file:auth.py")
        max_depth: Maximum depth to traverse

    Returns:
        List of related nodes with relationship info

    Examples:
        # What depends on auth.py?
        graph_query("used_by", "file:auth.py")

        # Find circular dependencies
        graph_query("circular", "file:models.py")
    """
    graph = create_graph_storage(db)

    if query_type == "depends_on":
        # Find dependencies (outgoing edges)
        neighbors = await graph.query_neighbors(
            node_id,
            edge_type="imports",
            direction="outgoing",
            max_depth=max_depth
        )

    elif query_type == "used_by":
        # Find dependents (incoming edges)
        neighbors = await graph.query_neighbors(
            node_id,
            edge_type="imports",
            direction="incoming",
            max_depth=max_depth
        )

    elif query_type == "circular":
        # Find cycles
        cycles = await graph.find_cycles(node_id, max_depth=max_depth)
        return [{"cycle": cycle} for cycle in cycles]

    return [
        {
            "node_id": n.node_id,
            "node_type": n.node_type,
            "metadata": n.metadata,
        }
        for n in neighbors
    ]
```

##### **Step 4: Add Graph CLI Commands**
**File**: `src/minisweagent/run/mini_jeeves.py`

**NEW COMMANDS**:

```python
@app.command()
def graph_build(
    repo_path: str = typer.Argument(".", help="Repository path"),
    database_url: str = typer.Option(..., "--database-url", envvar="MSWEA_DATABASE_URL"),
):
    """Build code dependency graph."""
    graph = create_graph_storage(create_database_client(database_url))
    extractor = GraphExtractor(graph)

    files = glob.glob(f"{repo_path}/**/*.py", recursive=True)

    with Progress() as progress:
        task = progress.add_task("[green]Building graph...", total=len(files))

        for file_path in files:
            content = read_file(file_path)
            await extractor.extract_from_file(file_path, content)
            progress.update(task, advance=1)

    console.print(f"[green]✓ Built graph for {len(files)} files[/green]")

@app.command()
def graph_deps(
    file_path: str = typer.Argument(..., help="File to analyze"),
    direction: str = typer.Option("depends_on", "--direction", help="depends_on or used_by"),
    database_url: str = typer.Option(..., "--database-url", envvar="MSWEA_DATABASE_URL"),
):
    """Show file dependencies."""
    graph = create_graph_storage(create_database_client(database_url))
    node_id = f"file:{file_path}"

    if direction == "depends_on":
        neighbors = await graph.query_neighbors(node_id, "imports", "outgoing")
        console.print(f"[cyan]{file_path} depends on:[/cyan]")
    else:
        neighbors = await graph.query_neighbors(node_id, "imports", "incoming")
        console.print(f"[cyan]{file_path} is used by:[/cyan]")

    for neighbor in neighbors:
        file = neighbor.metadata.get("path", neighbor.node_id)
        console.print(f"  - {file}")
```

##### **Usage Examples**

```bash
# Build graph (one-time setup)
mini-jeeves graph-build .
# Output: ✓ Built graph for 1,234 files

# Check what depends on a file
mini-jeeves graph-deps src/auth/models.py --direction used_by
# Output:
# src/auth/models.py is used by:
#   - src/auth/views.py
#   - src/api/endpoints.py
#   - src/admin/forms.py

# Use in agent task
mini-jeeves -t "Refactor auth.py without breaking dependencies"
# Agent uses graph_query internally:
# → graph_query("used_by", "file:auth.py")
# → Returns [api.py, routes.py, middleware.py]
# → Agent ensures changes don't break dependents
```

---

### 2.3 Clarification Interrupts

#### **What It Does**
Request clarification from user on ambiguous tasks:
- Detect ambiguous requests
- Pause pipeline execution
- Prompt user for clarification
- Resume with clarified intent

#### **Breaking Changes**
1. **Interrupt Protocol**: Agents can request clarifications
2. **CLI Prompts**: Interactive prompts during execution
3. **Webhook Support**: Optional webhook notifications

#### **Implementation**

##### **Step 1: Extend Interrupt Handler**
**File**: `src/minisweagent/capability/interrupts/clarification_handler.py` (NEW)

```python
from protocols import FlowInterrupt, InterruptKind

class ClarificationHandler:
    """Handle clarification requests from agents."""

    def __init__(self, interrupt_service, cli_service):
        self.interrupt_service = interrupt_service
        self.cli_service = cli_service

    async def request_clarification(
        self,
        question: str,
        options: List[str],
        context: Dict[str, Any],
    ) -> str:
        """
        Request clarification from user.

        Args:
            question: Question to ask
            options: List of possible answers
            context: Additional context

        Returns:
            Selected option
        """
        # Create interrupt
        interrupt = FlowInterrupt(
            kind=InterruptKind.CLARIFICATION,
            question=question,
            options=options,
            metadata=context,
        )

        # Persist interrupt
        interrupt_id = await self.interrupt_service.create_interrupt(interrupt)

        # Prompt user (CLI or webhook)
        if self.cli_service:
            response = await self.cli_service.prompt_choice(question, options)
        else:
            # Wait for webhook response
            response = await self.interrupt_service.wait_for_response(
                interrupt_id,
                timeout=300  # 5 minutes
            )

        return response
```

##### **Step 2: Update task_parser Agent**
**File**: `src/minisweagent/capability/prompts/registry.py`

```python
SYSTEM_PROMPTS = {
    "task_parser": """You are a task parser. Extract key information and detect ambiguity.

## Output Format (JSON)
{
    "objective": "Main goal",
    "files": ["mentioned files"],
    "commands": ["commands to run"],
    "ambiguous": true/false,
    "clarification_question": "Question if ambiguous",
    "options": ["Option A", "Option B"]
}

## Ambiguity Detection
Detect if task is ambiguous:
- Multiple interpretations possible
- Missing critical information
- Unclear scope

If ambiguous, set "ambiguous": true and provide clarification question.

## Examples
Task: "Fix the login bug"
Output:
{
    "objective": "Fix bug in login functionality",
    "ambiguous": true,
    "clarification_question": "Which login issue should I fix?",
    "options": [
        "SQL injection vulnerability in login form",
        "Password reset token expiration",
        "Session timeout too short"
    ]
}
""",
}
```

##### **Step 3: Wire Clarification Handler into Orchestrator**
**File**: `src/minisweagent/capability/orchestrator.py`

```python
async def execute_task_parser(self, envelope: Envelope):
    """Execute task parser with clarification support."""

    # Run LLM
    response = await self.llm.complete(
        system=self.prompts.get("task_parser"),
        messages=[{"role": "user", "content": envelope.task}]
    )

    # Parse response
    result = json.loads(response.content)

    # Check for ambiguity
    if result.get("ambiguous"):
        # Request clarification
        selected = await self.clarification_handler.request_clarification(
            question=result["clarification_question"],
            options=result["options"],
            context={"original_task": envelope.task},
        )

        # Update task with clarification
        envelope.task = f"{envelope.task}\n\nClarification: {selected}"
        logger.info(f"User selected: {selected}")

    # Continue pipeline
    return result
```

##### **Usage Examples**

```bash
# Ambiguous task triggers clarification
mini-jeeves -t "Fix the login bug"

# Output (interactive prompt):
# ┌────────────────────────────────────────────────────┐
# │ Clarification Needed                               │
# ├────────────────────────────────────────────────────┤
# │ Which login issue should I fix?                    │
# │                                                    │
# │ 1. SQL injection vulnerability in login form      │
# │ 2. Password reset token expiration                │
# │ 3. Session timeout too short                      │
# │                                                    │
# │ Select option [1-3]: _                             │
# └────────────────────────────────────────────────────┘

# User selects: 1

# Output:
# ✓ Proceeding with: SQL injection vulnerability in login form
# 🤖 Starting: code_searcher
# ...
```

---

## Phase 3: Production Readiness (Week 4-5)
**Goal**: Add observability, verification, and advanced features
**Duration**: 6 days
**Dependencies**: Phase 1 & 2 complete

### 3.1 Prometheus Metrics

#### **What It Does**
Export metrics to Prometheus for monitoring:
- Pipeline execution metrics (duration, success rate)
- Agent metrics (per-agent latency)
- LLM metrics (token usage, cost)
- Tool metrics (execution count, latency)

#### **Breaking Changes**
1. **Metrics Port**: HTTP server on port 9090 for metrics
2. **New Dependency**: prometheus_client library required
3. **Environment Variable**: `PROMETHEUS_PORT` for custom port

#### **Implementation**

**File**: `src/minisweagent/capability/observability/metrics.py` (NEW)

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Pipeline metrics
pipeline_executions = Counter(
    'mini_swe_pipeline_executions_total',
    'Total pipeline executions',
    ['pipeline_mode', 'status']
)

pipeline_duration = Histogram(
    'mini_swe_pipeline_duration_seconds',
    'Pipeline execution duration',
    ['pipeline_mode'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]
)

# Agent metrics
agent_calls = Counter(
    'mini_swe_agent_calls_total',
    'Total agent calls',
    ['agent_name', 'status']
)

agent_latency = Histogram(
    'mini_swe_agent_latency_seconds',
    'Agent latency',
    ['agent_name'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

# LLM metrics
llm_tokens = Counter(
    'mini_swe_llm_tokens_total',
    'Total LLM tokens',
    ['model', 'type']  # type: input/output
)

llm_cost = Counter(
    'mini_swe_llm_cost_usd_total',
    'Total LLM cost in USD',
    ['model']
)

# Tool metrics
tool_executions = Counter(
    'mini_swe_tool_executions_total',
    'Total tool executions',
    ['tool_name', 'status']
)

tool_latency = Histogram(
    'mini_swe_tool_latency_seconds',
    'Tool execution latency',
    ['tool_name'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30]
)

# Active sessions
active_sessions = Gauge(
    'mini_swe_active_sessions',
    'Number of active sessions'
)

def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics HTTP server."""
    start_http_server(port)
    logger.info(f"Prometheus metrics available at http://localhost:{port}/metrics")
```

**Usage**:
```bash
# Start with metrics
mini-jeeves -t "Fix bug" --enable-metrics --metrics-port 9090

# View metrics
curl http://localhost:9090/metrics

# Prometheus config
scrape_configs:
  - job_name: 'mini-swe-agent'
    static_configs:
      - targets: ['localhost:9090']
```

---

### 3.2 NLI Verification (Anti-Hallucination)

#### **What It Does**
Verify LLM claims are supported by evidence:
- Check if plan descriptions match actual code
- Detect hallucinations before presenting results
- Improve answer accuracy

#### **Implementation**

**File**: `src/minisweagent/capability/agents/swe_post_processor.py`

```python
from mission_system.adapters import create_nli_service

class VerificationPostProcessor:
    """Verify agent outputs against evidence."""

    def __init__(self, nli_service):
        self.nli = nli_service

    async def verify_plan(self, plan: Dict, codebase_context: str) -> Dict:
        """Verify plan claims against codebase."""
        verified_claims = []

        for claim in plan.get("claims", []):
            result = await self.nli.verify_claim(
                claim=claim["description"],
                evidence=codebase_context,
            )

            verified_claims.append({
                "claim": claim,
                "verified": result.label == "entailment",
                "confidence": result.score,
            })

        # Flag hallucinations
        hallucinations = [c for c in verified_claims if not c["verified"]]
        if hallucinations:
            logger.warning(f"Detected {len(hallucinations)} potential hallucinations")

        return {
            **plan,
            "verified_claims": verified_claims,
            "hallucination_rate": len(hallucinations) / len(verified_claims),
        }
```

---

### 3.3 Checkpointing

#### **What It Does**
Save pipeline state for resume after interruptions:
- Checkpoint at each agent completion
- Resume from checkpoint on failure
- Time-travel debugging

#### **Implementation**

**File**: `src/minisweagent/capability/orchestrator.py`

```python
async def run_with_checkpointing(self, task: str, session_id: str):
    """Run pipeline with automatic checkpointing."""

    # Try to resume from checkpoint
    checkpoint = await self.checkpoint_service.load_checkpoint(session_id)
    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint.agent_name}")
        envelope = checkpoint.envelope
        start_agent = checkpoint.next_agent
    else:
        envelope = Envelope(task=task, ...)
        start_agent = "task_parser"

    # Run pipeline with checkpointing
    for agent_name in self.pipeline.get_agents_from(start_agent):
        try:
            result = await self.execute_agent(agent_name, envelope)

            # Save checkpoint
            await self.checkpoint_service.save_checkpoint(
                session_id=session_id,
                agent_name=agent_name,
                envelope=envelope,
                next_agent=self.pipeline.get_next(agent_name),
            )

        except Exception as e:
            logger.error(f"Agent {agent_name} failed, checkpoint saved")
            raise
```

---

### 3.4 L2 Event Log (Audit Trail)

#### **What It Does**
Persistent append-only log of all events:
- Full audit trail
- Compliance/debugging
- Event replay

#### **Implementation**

**File**: `src/minisweagent/capability/db/migrations/005_event_log.sql` (NEW)

```sql
CREATE TABLE IF NOT EXISTS event_log (
    event_id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT NOW(),
    event_category VARCHAR(50) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    agent_name VARCHAR(100),
    payload JSONB,
    metadata JSONB
);

CREATE INDEX idx_event_log_session ON event_log(session_id);
CREATE INDEX idx_event_log_timestamp ON event_log(timestamp);
CREATE INDEX idx_event_log_category ON event_log(event_category);
```

---

## Implementation Roadmap

### Week 1: Foundation
- [ ] Day 1: L4 Working Memory
  - [ ] Database schema
  - [ ] Wire session service
  - [ ] Update CLI
  - [ ] Tests
- [ ] Day 2: L7 Tool Health
  - [ ] Database schema
  - [ ] Wrap executor
  - [ ] Convert tools to async
  - [ ] Tests
- [ ] Day 3: Event Streaming
  - [ ] Wire event aggregator
  - [ ] Update CLI
  - [ ] Rich UI
  - [ ] Tests

### Week 2: Semantic Search
- [ ] Day 4-5: L3 Semantic Search
  - [ ] Database schema (pgvector)
  - [ ] Indexing CLI
  - [ ] Semantic search tool
  - [ ] Update code_searcher agent
  - [ ] Tests

### Week 3: Graph Storage
- [ ] Day 6-8: L5 Graph Storage
  - [ ] Database schema
  - [ ] AST extraction
  - [ ] Graph query tool
  - [ ] Graph CLI commands
  - [ ] Tests
- [ ] Day 9-10: Clarification Interrupts
  - [ ] Clarification handler
  - [ ] Update task_parser
  - [ ] CLI prompts
  - [ ] Tests

### Week 4: Observability
- [ ] Day 11: Prometheus Metrics
  - [ ] Metrics definitions
  - [ ] Wire into pipeline
  - [ ] HTTP server
  - [ ] Tests
- [ ] Day 12: NLI Verification
  - [ ] Verification post-processor
  - [ ] Wire into verifier agent
  - [ ] Tests

### Week 5: Advanced Features
- [ ] Day 13-15: Checkpointing
  - [ ] Database schema
  - [ ] Checkpoint service
  - [ ] Resume logic
  - [ ] CLI commands
  - [ ] Tests
- [ ] Day 16: L2 Event Log
  - [ ] Database schema
  - [ ] Event persistence
  - [ ] Query APIs
  - [ ] Tests

---

## Breaking Changes Summary

### Configuration

**OLD** (`src/minisweagent/config/mini.yaml`):
```yaml
model:
  model_name: "gpt-4"
  temperature: 0.7
```

**NEW** (v2.0):
```yaml
model:
  model_name: "gpt-4"
  temperature: 0.7

# NEW: Database required
database:
  url: "postgresql://localhost:5432/mini_swe_agent"
  pool_size: 10

# NEW: Session management
session:
  enable_persistence: true
  default_ttl: 86400

# NEW: Observability
observability:
  enable_metrics: true
  metrics_port: 9090
  enable_event_log: true
```

### CLI

**OLD**:
```bash
mini -t "Fix bug" --config mini.yaml
```

**NEW** (v2.0):
```bash
# Database URL required
export MSWEA_DATABASE_URL="postgresql://localhost:5432/mini_swe_agent"

# With session
mini-jeeves -t "Fix bug" --session my-project

# With metrics
mini-jeeves -t "Fix bug" --enable-metrics --metrics-port 9090

# New commands
mini-jeeves db migrate
mini-jeeves index .
mini-jeeves graph-build .
mini-jeeves tool-health
mini-jeeves --list-sessions
```

### Tool Signatures

**OLD** (sync):
```python
def bash_execute(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True)
    return result.stdout.decode()
```

**NEW** (async + Result):
```python
async def bash_execute(command: str) -> str:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode()
```

### Agent Protocol

**OLD**:
```python
# Agents receive only task
async def execute_agent(agent_name: str, task: str):
    response = await llm.complete(task)
    return response
```

**NEW**:
```python
# Agents receive working memory
async def execute_agent(agent_name: str, envelope: Envelope):
    working_memory = envelope.context.get("working_memory")
    prompt_vars = {
        "task": envelope.task,
        "previous_findings": working_memory.findings if working_memory else [],
    }
    response = await llm.complete(prompt.format(**prompt_vars))
    return response
```

---

## Testing Strategy

### Unit Tests
```python
# Test each feature in isolation
tests/capability/test_working_memory.py
tests/capability/test_tool_health.py
tests/capability/test_semantic_search.py
tests/capability/test_graph_storage.py
tests/capability/test_clarification.py
tests/capability/test_metrics.py
tests/capability/test_checkpointing.py
```

### Integration Tests
```python
# Test end-to-end workflows
tests/integration/test_session_persistence.py
tests/integration/test_semantic_workflow.py
tests/integration/test_graph_workflow.py
```

### Performance Tests
```python
# Benchmark new features
tests/performance/test_semantic_search_speed.py
tests/performance/test_graph_query_speed.py
```

---

## Migration Guide (for Users)

### Step 1: Install PostgreSQL
```bash
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=mini_swe_agent \
  postgres:15-alpine

# Install pgvector extension
docker exec -it <container> psql -U postgres -d mini_swe_agent -c "CREATE EXTENSION vector;"
```

### Step 2: Update Configuration
```bash
export MSWEA_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/mini_swe_agent"
```

### Step 3: Run Migrations
```bash
mini-jeeves db migrate --apply
```

### Step 4: Index Codebase (Optional)
```bash
mini-jeeves index . --pattern "**/*.py"
mini-jeeves graph-build .
```

### Step 5: Use New Features
```bash
# Create session
mini-jeeves -t "Find auth code" --new-session

# Resume session
mini-jeeves -t "Fix the bug" --session session_20260127_143022

# View tool health
mini-jeeves tool-health

# Check dependencies
mini-jeeves graph-deps src/auth.py --direction used_by
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Session Persistence** | 10x faster follow-up queries | Benchmark before/after |
| **Tool Health** | <5% quarantined tools | Monitor quarantine rate |
| **Semantic Search** | >80% relevance for top-5 | User study |
| **Graph Queries** | <100ms for dependencies | Query latency |
| **Event Streaming** | <200ms event delivery | Stream latency |
| **Clarifications** | <30% tasks need clarification | Request rate |
| **Metrics Coverage** | 100% of agents instrumented | Code coverage |
| **Uptime** | >99.9% with checkpointing | Downtime tracking |

---

## Rollout Plan

### Alpha (Internal)
- Week 1-2: Phase 1 features
- Test on internal projects
- Gather feedback

### Beta (Early Adopters)
- Week 3-4: Phase 2 features
- Open to beta testers
- Fix bugs

### GA (v2.0)
- Week 5-6: Phase 3 features
- Public release
- Documentation

---

## Documentation Updates Needed

1. **New User Guide**: `docs/v2-user-guide.md`
2. **Migration Guide**: `docs/migration-v1-to-v2.md`
3. **Database Setup**: `docs/database-setup.md`
4. **Semantic Search Guide**: `docs/semantic-search.md`
5. **Graph Queries Guide**: `docs/graph-queries.md`
6. **Metrics Guide**: `docs/observability.md`
7. **API Reference**: Update for all new tools

---

**END OF WIRING PLAN**

---

*This wiring plan provides a comprehensive roadmap to integrate all latent jeeves-core capabilities into mini-swe-agent v2.0. Implementation assumes no backward compatibility constraints and targets a 6-8 week development cycle.*
