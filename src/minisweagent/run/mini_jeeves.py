#!/usr/bin/env python3
"""Run mini-SWE-agent with jeeves-core orchestration (v2.0).

All execution flows through jeeves-core's PipelineRunner:
- unified: Single-stage pipeline with self-routing (default, mimics original agent)
- parallel: Multi-stage pipeline with parallel analysis

v2.0 Features:
- Session persistence (--session, --new-session)
- Database management (db migrate, db status)
- Semantic search (index, search)
- Graph queries (graph-build, graph-deps)
- Tool health monitoring (tool-health, tool-reset)
- Prometheus metrics (--enable-metrics)

Usage:
    mini-jeeves -t "Fix the bug in auth.py"
    mini-jeeves -t "Add logging" --pipeline parallel
    mini-jeeves -t "Quick fix" --yolo  # Skip confirmations
    mini-jeeves -t "Continue work" --session my-project
    mini-jeeves db migrate
    mini-jeeves tool-health
"""

import asyncio
import glob as globlib
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add jeeves-core to path
_jeeves_core_path = Path(__file__).parent.parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from minisweagent import global_config_dir
from minisweagent.capability import register_capability
from minisweagent.capability.orchestrator import (
    SWEOrchestrator,
    SWEOrchestratorConfig,
    create_swe_orchestrator,
)
from minisweagent.capability.tools.catalog import configure_tools
from minisweagent.capability.prompts import create_prompt_registry
from minisweagent.config import builtin_config_dir, get_config_path

logger = logging.getLogger(__name__)
console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich", help="Mini-SWE-Agent v2.0 - AI-powered software engineering")

DEFAULT_CONFIG = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT = global_config_dir / "last_mini_jeeves_run.traj.json"

# =============================================================================
# v2.0: Subcommand groups
# =============================================================================

db_app = typer.Typer(help="Database management commands")
app.add_typer(db_app, name="db")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_database_url() -> Optional[str]:
    """Get database URL from environment."""
    return os.getenv("MSWEA_DATABASE_URL")


def _create_llm_factory(provider: str, base_url: str, model: Optional[str] = None):
    """Create LLM provider factory."""
    def factory(agent_role: str):
        try:
            from avionics.llm.factory import create_llm_provider
            from avionics.settings import Settings

            settings = Settings(
                llm_provider=provider,
                llamaserver_host=base_url.rstrip("/v1"),
                default_model=model or "default",
            )
            return create_llm_provider(settings, agent_name=agent_role)
        except ImportError:
            if provider == "openai_http":
                from avionics.llm.providers.openai_http_provider import OpenAIHTTPProvider
                return OpenAIHTTPProvider(
                    api_base=base_url,
                    model=model or "default",
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")

    return factory


def _create_tool_executor(mode: str = "confirm", whitelist_patterns: Optional[list] = None):
    """Create tool executor with confirmation support."""
    from minisweagent.capability.tools.catalog import get_tool_catalog
    from minisweagent.capability.tools.confirming_executor import create_confirming_executor

    catalog = get_tool_catalog()

    class CapabilityToolExecutor:
        async def execute(self, tool_name: str, params: dict) -> dict:
            tool = catalog.get_tool(tool_name)
            if tool is None:
                return {"status": "error", "error": f"Unknown tool: {tool_name}"}
            return await tool.function(**params)

        def get_available_tools(self):
            return catalog.list_tools()

    inner_executor = CapabilityToolExecutor()

    if mode == "yolo":
        return inner_executor
    else:
        return create_confirming_executor(
            inner_executor=inner_executor,
            mode=mode,
            whitelist_patterns=whitelist_patterns,
        )


# =============================================================================
# v2.0: DATABASE COMMANDS
# =============================================================================

@db_app.command("migrate")
def db_migrate(
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="MSWEA_DATABASE_URL",
        help="PostgreSQL connection URL"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show pending migrations without applying"),
):
    """Run database migrations."""
    if not database_url:
        console.print("[red]Error: Database URL not provided. Set MSWEA_DATABASE_URL or use --database-url[/red]")
        raise typer.Exit(1)

    from minisweagent.capability.db.migrator import DatabaseMigrator

    async def run():
        migrator = DatabaseMigrator(database_url)
        if dry_run:
            console.print("[yellow]Dry run - showing pending migrations:[/yellow]")
        await migrator.migrate(dry_run=dry_run)
        if not dry_run:
            console.print("[green]✓ Migrations applied successfully[/green]")

    asyncio.run(run())


@db_app.command("status")
def db_status(
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="MSWEA_DATABASE_URL",
        help="PostgreSQL connection URL"
    ),
):
    """Show migration status."""
    if not database_url:
        console.print("[red]Error: Database URL not provided. Set MSWEA_DATABASE_URL or use --database-url[/red]")
        raise typer.Exit(1)

    from minisweagent.capability.db.migrator import DatabaseMigrator

    async def run():
        migrator = DatabaseMigrator(database_url)
        await migrator.status()

    asyncio.run(run())


# =============================================================================
# v2.0: SESSION COMMANDS
# =============================================================================

@app.command("list-sessions")
def list_sessions(
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="MSWEA_DATABASE_URL",
        help="PostgreSQL connection URL"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum sessions to show"),
):
    """List active sessions."""
    if not database_url:
        console.print("[red]Error: Database URL not provided[/red]")
        raise typer.Exit(1)

    async def run():
        orchestrator = create_swe_orchestrator(database_url=database_url)
        sessions = await orchestrator.list_sessions(limit=limit)
        await orchestrator.close()

        if not sessions:
            console.print("[yellow]No active sessions found[/yellow]")
            return

        table = Table(title="Active Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Created", style="green")
        table.add_column("Updated", style="green")
        table.add_column("Findings", style="yellow")
        table.add_column("Entities", style="blue")

        for s in sessions:
            created = s['created_at'].strftime("%Y-%m-%d %H:%M") if s.get('created_at') else "N/A"
            updated = s['updated_at'].strftime("%Y-%m-%d %H:%M") if s.get('updated_at') else "N/A"
            table.add_row(
                s['session_id'],
                created,
                updated,
                str(s.get('finding_count', 0)),
                str(s.get('entity_count', 0)),
            )

        console.print(table)

    asyncio.run(run())


@app.command("session-delete")
def session_delete(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="MSWEA_DATABASE_URL",
        help="PostgreSQL connection URL"
    ),
):
    """Delete a session."""
    if not database_url:
        console.print("[red]Error: Database URL not provided[/red]")
        raise typer.Exit(1)

    async def run():
        orchestrator = create_swe_orchestrator(database_url=database_url)
        await orchestrator.delete_session(session_id)
        await orchestrator.close()
        console.print(f"[green]✓ Deleted session: {session_id}[/green]")

    asyncio.run(run())


# =============================================================================
# v2.0: INDEXING COMMANDS
# =============================================================================

@app.command("index")
def index_codebase(
    path: str = typer.Argument(".", help="Repository path to index"),
    pattern: str = typer.Option("**/*.py", "--pattern", "-p", help="File glob pattern"),
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="MSWEA_DATABASE_URL",
        help="PostgreSQL connection URL"
    ),
    chunk_size: int = typer.Option(512, "--chunk-size", help="Tokens per chunk"),
):
    """Index codebase for semantic search."""
    if not database_url:
        console.print("[red]Error: Database URL not provided[/red]")
        raise typer.Exit(1)

    from minisweagent.capability.services import CodeIndexerService

    async def run():
        import asyncpg
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        indexer = CodeIndexerService(pool)

        # Find files
        full_pattern = os.path.join(path, pattern)
        files = globlib.glob(full_pattern, recursive=True)
        console.print(f"[cyan]Found {len(files)} files to index[/cyan]")

        if not files:
            console.print("[yellow]No files found matching pattern[/yellow]")
            return

        # Index files
        from rich.progress import Progress
        with Progress() as progress:
            task = progress.add_task("[green]Indexing...", total=len(files))

            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    await indexer.index_file(file_path, content, chunk_size=chunk_size)
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[red]Failed to index {file_path}: {e}[/red]")

        await pool.close()
        console.print(f"[green]✓ Indexed {len(files)} files[/green]")

    asyncio.run(run())


@app.command("search")
def search_code(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="MSWEA_DATABASE_URL",
        help="PostgreSQL connection URL"
    ),
):
    """Search codebase semantically."""
    if not database_url:
        console.print("[red]Error: Database URL not provided[/red]")
        raise typer.Exit(1)

    from minisweagent.capability.services import CodeIndexerService

    async def run():
        import asyncpg
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        indexer = CodeIndexerService(pool)

        results = await indexer.search(query, limit=limit)
        await pool.close()

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            file_path = result.get('source_file', 'unknown')
            content = result.get('content', '')[:500]

            color = "green" if score > 0.8 else "yellow" if score > 0.6 else "white"
            panel = Panel(
                content,
                title=f"[{color}]{i}. {file_path}[/{color}] (score: {score:.2f})",
                border_style=color,
            )
            console.print(panel)

    asyncio.run(run())


# =============================================================================
# v2.0: GRAPH COMMANDS
# =============================================================================

@app.command("graph-build")
def graph_build(
    path: str = typer.Argument(".", help="Repository path"),
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="MSWEA_DATABASE_URL",
        help="PostgreSQL connection URL"
    ),
):
    """Build code dependency graph."""
    if not database_url:
        console.print("[red]Error: Database URL not provided[/red]")
        raise typer.Exit(1)

    from minisweagent.capability.services import GraphService
    from minisweagent.capability.agents.graph_extractor import GraphExtractor

    async def run():
        import asyncpg
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        graph = GraphService(pool)
        extractor = GraphExtractor(graph)

        # Find Python files
        files = globlib.glob(os.path.join(path, "**/*.py"), recursive=True)
        console.print(f"[cyan]Found {len(files)} Python files[/cyan]")

        from rich.progress import Progress
        with Progress() as progress:
            task = progress.add_task("[green]Building graph...", total=len(files))

            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    await extractor.extract_from_file(file_path, content)
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[red]Failed to process {file_path}: {e}[/red]")

        await pool.close()
        console.print(f"[green]✓ Built graph for {len(files)} files[/green]")

    asyncio.run(run())


@app.command("graph-deps")
def graph_deps(
    file_path: str = typer.Argument(..., help="File to analyze"),
    direction: str = typer.Option("depends_on", "--direction", "-d", help="depends_on or used_by"),
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="MSWEA_DATABASE_URL",
        help="PostgreSQL connection URL"
    ),
):
    """Show file dependencies."""
    if not database_url:
        console.print("[red]Error: Database URL not provided[/red]")
        raise typer.Exit(1)

    from minisweagent.capability.services import GraphService

    async def run():
        import asyncpg
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        graph = GraphService(pool)

        node_id = f"file:{file_path}"
        edge_direction = "outgoing" if direction == "depends_on" else "incoming"

        neighbors = await graph.query_neighbors(node_id, "imports", edge_direction)
        await pool.close()

        if not neighbors:
            console.print(f"[yellow]No dependencies found for {file_path}[/yellow]")
            return

        label = "depends on" if direction == "depends_on" else "is used by"
        console.print(f"[cyan]{file_path} {label}:[/cyan]")

        for neighbor in neighbors:
            path = neighbor.get('metadata', {}).get('path', neighbor.get('node_id', 'unknown'))
            console.print(f"  - {path}")

    asyncio.run(run())


# =============================================================================
# v2.0: TOOL HEALTH COMMANDS
# =============================================================================

@app.command("tool-health")
def tool_health(
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="MSWEA_DATABASE_URL",
        help="PostgreSQL connection URL"
    ),
):
    """Display tool health metrics."""
    if not database_url:
        console.print("[red]Error: Database URL not provided[/red]")
        raise typer.Exit(1)

    from minisweagent.capability.services import ToolHealthService

    async def run():
        import asyncpg
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        health_service = ToolHealthService(pool)

        metrics = await health_service.get_all_metrics()
        await pool.close()

        if not metrics:
            console.print("[yellow]No tool metrics found[/yellow]")
            return

        table = Table(title="Tool Health Metrics")
        table.add_column("Tool", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Success Rate", style="green")
        table.add_column("Avg Latency", style="yellow")
        table.add_column("Total Calls", style="blue")

        for metric in metrics:
            status = metric.get('status', 'unknown')
            status_color = {
                "healthy": "green",
                "degraded": "yellow",
                "quarantined": "red",
            }.get(status, "white")

            invocations = metric.get('invocation_count', 0)
            success = metric.get('success_count', 0)
            success_rate = f"{success / invocations:.1%}" if invocations > 0 else "N/A"

            total_latency = metric.get('total_latency_ms', 0)
            avg_latency = f"{total_latency / invocations:.0f}ms" if invocations > 0 else "N/A"

            table.add_row(
                metric.get('tool_name', 'unknown'),
                f"[{status_color}]{status}[/{status_color}]",
                success_rate,
                avg_latency,
                str(invocations),
            )

        console.print(table)

    asyncio.run(run())


@app.command("tool-reset")
def tool_reset(
    tool_name: str = typer.Argument(..., help="Tool name to reset"),
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="MSWEA_DATABASE_URL",
        help="PostgreSQL connection URL"
    ),
):
    """Reset tool health metrics (unquarantine)."""
    if not database_url:
        console.print("[red]Error: Database URL not provided[/red]")
        raise typer.Exit(1)

    from minisweagent.capability.services import ToolHealthService

    async def run():
        import asyncpg
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        health_service = ToolHealthService(pool)

        await health_service.reset_metrics(tool_name)
        await pool.close()
        console.print(f"[green]✓ Reset metrics for {tool_name}[/green]")

    asyncio.run(run())


# =============================================================================
# MAIN RUN COMMAND (Updated for v2.0)
# =============================================================================

async def _run_async(
    task: str,
    pipeline_mode: str,
    execution_mode: str,
    llm_provider: str,
    llm_url: str,
    llm_model: Optional[str],
    cost_limit: float,
    step_limit: int,
    timeout: int,
    streaming: bool,
    config: dict,
    # v2.0 parameters
    session_id: Optional[str] = None,
    database_url: Optional[str] = None,
    enable_metrics: bool = False,
    metrics_port: int = 9090,
    resume_checkpoint: Optional[str] = None,
) -> dict:
    """Run the orchestrator asynchronously (v2.0)."""
    # Configure tools
    configure_tools(cwd=os.getcwd(), timeout=timeout)

    # Create components
    try:
        llm_factory = _create_llm_factory(llm_provider, llm_url, llm_model)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not create LLM factory: {e}[/yellow]")
        llm_factory = None

    # Create tool executor with confirmation support
    whitelist = config.get("agent", {}).get("whitelist_actions", [])
    tool_executor = _create_tool_executor(mode=execution_mode, whitelist_patterns=whitelist)

    # Create prompt registry
    prompt_registry = create_prompt_registry()

    # v2.0: Create orchestrator with new options
    orchestrator = create_swe_orchestrator(
        llm_factory=llm_factory,
        tool_executor=tool_executor,
        prompt_registry=prompt_registry,
        pipeline_mode=pipeline_mode,
        cost_limit=cost_limit,
        step_limit=step_limit,
        timeout=timeout,
        # v2.0 config
        database_url=database_url,
        enable_sessions=session_id is not None or database_url is not None,
        enable_metrics=enable_metrics,
        metrics_port=metrics_port,
    )

    try:
        # Run with interactive handling
        if streaming:
            result = {}
            async for stage_name, output in orchestrator.run_streaming(
                task, session_id=session_id or ""
            ):
                if stage_name == "__end__":
                    result = output
                else:
                    console.print(f"\n[bold blue]{stage_name}[/bold blue] completed")
                    if isinstance(output, dict) and "response" in output:
                        console.print(output["response"][:500])
            return result
        else:
            return await orchestrator.run(
                task,
                session_id=session_id or "",
                resume_checkpoint=resume_checkpoint,
            )
    finally:
        await orchestrator.close()


_HELP_TEXT = """Run mini-SWE-agent with jeeves-core orchestration (v2.0).

[not dim]
Pipeline modes:

[bold green]--pipeline unified[/bold green]  Single-agent loop (default)
[bold green]--pipeline parallel[/bold green] Multi-stage with parallel analysis

Execution modes:

[bold green]--yolo[/bold green]     Execute commands without confirmation
[bold green]--confirm[/bold green]  Ask before executing (default)

v2.0 Session persistence:

[bold green]--session <id>[/bold green]  Resume a session
[bold green]--new-session[/bold green]   Create a new session

v2.0 Observability:

[bold green]--enable-metrics[/bold green]  Enable Prometheus metrics
[bold green]--metrics-port[/bold green]    Metrics server port (default: 9090)

More commands: mini-jeeves --help
[/not dim]
"""


# fmt: off
@app.command(help=_HELP_TEXT)
def run(
    task: str = typer.Option(None, "-t", "--task", help="Task/problem statement"),
    pipeline: str = typer.Option("unified", "-p", "--pipeline", help="Pipeline mode: unified or parallel"),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Skip command confirmations"),
    llm_provider: str = typer.Option(
        os.getenv("JEEVES_LLM_ADAPTER", "openai_http"),
        "--llm-provider",
        help="LLM provider: openai_http, litellm",
    ),
    llm_url: str = typer.Option(
        os.getenv("JEEVES_LLM_BASE_URL", "http://localhost:8080/v1"),
        "--llm-url",
        help="LLM server URL",
    ),
    llm_model: Optional[str] = typer.Option(
        os.getenv("JEEVES_LLM_MODEL"),
        "--llm-model",
        help="Model name",
    ),
    cost_limit: float = typer.Option(3.0, "-l", "--cost-limit", help="Cost limit"),
    step_limit: int = typer.Option(0, "--step-limit", help="Step limit (0 = disabled)"),
    timeout: int = typer.Option(30, "--timeout", help="Command timeout in seconds"),
    streaming: bool = typer.Option(False, "-s", "--streaming", help="Stream stage outputs"),
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Config file"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
    # v2.0 options
    session: Optional[str] = typer.Option(None, "--session", help="Session ID for persistence"),
    new_session: bool = typer.Option(False, "--new-session", help="Create a new session"),
    database_url: Optional[str] = typer.Option(
        None, "--database-url", envvar="MSWEA_DATABASE_URL",
        help="PostgreSQL connection URL"
    ),
    enable_metrics: bool = typer.Option(False, "--enable-metrics", help="Enable Prometheus metrics"),
    metrics_port: int = typer.Option(9090, "--metrics-port", help="Prometheus metrics port"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume from checkpoint ID"),
) -> Any:
    # fmt: on
    """Run mini-jeeves on a task."""

    # Register capability with jeeves-core
    try:
        register_capability()
        console.print("[green]Capability registered with jeeves-core[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not register capability: {e}[/yellow]")

    # Load config
    config_path = get_config_path(config_spec)
    console.print(f"Loading config from [bold green]'{config_path}'[/bold green]")
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}

    # v2.0: Handle session ID
    session_id = session
    if new_session:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        console.print(f"[green]Created new session: {session_id}[/green]")

    # Get task
    if not task:
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.shortcuts import PromptSession

        prompt_session = PromptSession(history=FileHistory(global_config_dir / "mini_task_history.txt"))
        console.print("[bold yellow]What do you want to do?")
        exec_mode = "yolo" if yolo else "confirm"
        task = prompt_session.prompt(
            "",
            multiline=True,
            bottom_toolbar=HTML(
                f"Submit: <b fg='yellow'>Esc+Enter</b> | "
                f"Pipeline: <b fg='cyan'>{pipeline}</b> | "
                f"Mode: <b fg='cyan'>{exec_mode}</b>" +
                (f" | Session: <b fg='green'>{session_id}</b>" if session_id else "")
            ),
        )
        console.print("[bold green]Got it![/bold green]")

    execution_mode = "yolo" if yolo else "confirm"
    console.print(f"\nPipeline: [bold cyan]{pipeline}[/bold cyan]")
    console.print(f"Execution: [bold cyan]{execution_mode}[/bold cyan]")
    console.print(f"LLM: [bold cyan]{llm_provider}[/bold cyan] @ {llm_url}")
    if session_id:
        console.print(f"Session: [bold green]{session_id}[/bold green]")
    if enable_metrics:
        console.print(f"Metrics: [bold green]http://localhost:{metrics_port}/metrics[/bold green]")
    console.print()

    # Run
    exit_status, result = None, None
    try:
        result = asyncio.run(_run_async(
            task=task,
            pipeline_mode=pipeline,
            execution_mode=execution_mode,
            llm_provider=llm_provider,
            llm_url=llm_url,
            llm_model=llm_model,
            cost_limit=cost_limit,
            step_limit=step_limit,
            timeout=timeout,
            streaming=streaming,
            config=config,
            # v2.0
            session_id=session_id,
            database_url=database_url,
            enable_metrics=enable_metrics,
            metrics_port=metrics_port,
            resume_checkpoint=resume,
        ))
        exit_status = result.get("status", "unknown")

        console.print(f"\n[bold green]Status:[/bold green] {exit_status}")
        if "output" in result:
            console.print(f"\n[bold green]Output:[/bold green]\n{result['output']}")
        if "error" in result:
            console.print(f"\n[bold red]Error:[/bold red] {result['error']}")

        # v2.0: Show session info
        if result.get("session_id"):
            console.print(f"\n[bold blue]Session:[/bold blue] {result['session_id']}")
            console.print(f"[blue]Findings:[/blue] {result.get('findings_count', 0)}")
            console.print(f"[blue]Resume with:[/blue] --session {result['session_id']}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        exit_status = "interrupted"
        result = {"status": "interrupted"}

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        exit_status = "error"
        result = {"error": str(e), "traceback": traceback.format_exc()}

    # Save trajectory
    if output and result:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "task": task,
                "pipeline_mode": pipeline,
                "execution_mode": execution_mode,
                "session_id": session_id,
                "result": result,
                "exit_status": exit_status,
            }, indent=2, default=str))
            console.print(f"\nTrajectory saved to [bold green]{output}[/bold green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save trajectory: {e}[/yellow]")

    return result


if __name__ == "__main__":
    app()
