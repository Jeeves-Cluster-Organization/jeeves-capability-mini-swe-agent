#!/usr/bin/env python3
"""Run mini-SWE-agent with jeeves-core orchestration.

All execution flows through jeeves-core's PipelineRunner:
- unified: Single-stage pipeline with self-routing (default, mimics original agent)
- parallel: Multi-stage pipeline with parallel analysis

Usage:
    mini-jeeves -t "Fix the bug in auth.py"
    mini-jeeves -t "Add logging" --pipeline parallel
    mini-jeeves -t "Quick fix" --yolo  # Skip confirmations
"""

import asyncio
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from rich.console import Console

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

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")

DEFAULT_CONFIG = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT = global_config_dir / "last_mini_jeeves_run.traj.json"

_HELP_TEXT = """Run mini-SWE-agent with jeeves-core orchestration.

[not dim]
Pipeline modes:

[bold green]--pipeline unified[/bold green]  Single-agent loop (default)
[bold green]--pipeline parallel[/bold green] Multi-stage with parallel analysis

Execution modes:

[bold green]--yolo[/bold green]     Execute commands without confirmation
[bold green]--confirm[/bold green]  Ask before executing (default)

LLM Provider options:

[bold green]--llm-provider openai_http[/bold green]  Direct HTTP to OpenAI-compatible server
[bold green]--llm-provider litellm[/bold green]      Use litellm for unified API
[bold green]--llm-url http://localhost:8080/v1[/bold green]  LLM server URL

More information: [bold green]https://mini-swe-agent.com/latest/usage/mini/[/bold green]
[/not dim]
"""


def _create_llm_factory(provider: str, base_url: str, model: Optional[str] = None):
    """Create LLM provider factory.

    Args:
        provider: Provider type (openai_http, litellm)
        base_url: Base URL for the LLM server
        model: Model name (optional)

    Returns:
        Factory function that creates LLM providers
    """
    def factory(agent_role: str):
        """Create LLM provider for an agent role."""
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
            # Fall back to direct provider creation
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
    """Create tool executor with confirmation support.

    Args:
        mode: Execution mode (yolo, confirm, human)
        whitelist_patterns: Patterns for commands that skip confirmation

    Returns:
        Tool executor with confirmation support
    """
    from minisweagent.capability.tools.catalog import get_tool_catalog
    from minisweagent.capability.tools.confirming_executor import create_confirming_executor

    catalog = get_tool_catalog()

    class CapabilityToolExecutor:
        """Tool executor backed by capability tool catalog."""

        async def execute(self, tool_name: str, params: dict) -> dict:
            tool = catalog.get_tool(tool_name)
            if tool is None:
                return {"status": "error", "error": f"Unknown tool: {tool_name}"}
            return await tool.function(**params)

        def get_available_tools(self):
            return catalog.list_tools()

    inner_executor = CapabilityToolExecutor()

    # Wrap with confirmation if not yolo mode
    if mode == "yolo":
        return inner_executor
    else:
        return create_confirming_executor(
            inner_executor=inner_executor,
            mode=mode,
            whitelist_patterns=whitelist_patterns,
        )


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
) -> dict:
    """Run the orchestrator asynchronously.

    Args:
        task: Task description
        pipeline_mode: Pipeline mode (unified or parallel)
        execution_mode: Execution mode (yolo or confirm)
        llm_provider: LLM provider type
        llm_url: LLM server URL
        llm_model: Model name
        cost_limit: Cost limit
        step_limit: Step limit
        timeout: Command timeout
        streaming: Whether to stream output
        config: Additional config

    Returns:
        Result dict
    """
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

    # Create orchestrator
    orchestrator = create_swe_orchestrator(
        llm_factory=llm_factory,
        tool_executor=tool_executor,
        prompt_registry=prompt_registry,
        pipeline_mode=pipeline_mode,
        cost_limit=cost_limit,
        step_limit=step_limit,
        timeout=timeout,
    )

    # Run with interactive handling
    if streaming:
        result = {}
        async for stage_name, output in orchestrator.run_streaming(task):
            if stage_name == "__end__":
                result = output
            else:
                console.print(f"\n[bold blue]{stage_name}[/bold blue] completed")
                if isinstance(output, dict) and "response" in output:
                    console.print(output["response"][:500])
        return result
    else:
        return await orchestrator.run(task)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
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
) -> Any:
    # fmt: on
    """Main entry point for mini-jeeves."""

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
                f"Mode: <b fg='cyan'>{exec_mode}</b>"
            ),
        )
        console.print("[bold green]Got it![/bold green]")

    execution_mode = "yolo" if yolo else "confirm"
    console.print(f"\nPipeline: [bold cyan]{pipeline}[/bold cyan]")
    console.print(f"Execution: [bold cyan]{execution_mode}[/bold cyan]")
    console.print(f"LLM: [bold cyan]{llm_provider}[/bold cyan] @ {llm_url}\n")

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
        ))
        exit_status = result.get("status", "unknown")

        console.print(f"\n[bold green]Status:[/bold green] {exit_status}")
        if "output" in result:
            console.print(f"\n[bold green]Output:[/bold green]\n{result['output']}")
        if "error" in result:
            console.print(f"\n[bold red]Error:[/bold red] {result['error']}")

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
            import json
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "task": task,
                "pipeline_mode": pipeline,
                "execution_mode": execution_mode,
                "result": result,
                "exit_status": exit_status,
            }, indent=2))
            console.print(f"\nTrajectory saved to [bold green]{output}[/bold green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save trajectory: {e}[/yellow]")

    return result


if __name__ == "__main__":
    app()
