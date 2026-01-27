"""Prompt registry implementation for mini-swe-agent.

This module provides the PromptRegistry implementation that loads templates
from mini.yaml and renders them with Jinja2.
"""

import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from jinja2 import Template, StrictUndefined

# Add jeeves-core to path
_jeeves_core_path = Path(__file__).parent.parent.parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from protocols.agents import PromptRegistry


# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "mini.yaml"


class MiniSWEPromptRegistry(PromptRegistry):
    """Prompt registry for mini-swe-agent templates.

    Loads templates from mini.yaml config and provides them via the
    PromptRegistry protocol. Templates are rendered with Jinja2 using
    StrictUndefined to catch missing variables.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the prompt registry.

        Args:
            config_path: Path to the config file (default: mini.yaml)
        """
        self._templates: Dict[str, str] = {}
        self._config: Dict[str, Any] = {}
        self._load_config(config_path or DEFAULT_CONFIG_PATH)
        self._register_templates()

    def _load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to the config file
        """
        if config_path.exists():
            self._config = yaml.safe_load(config_path.read_text())
        else:
            self._config = {}

    def _register_templates(self) -> None:
        """Register all templates from config."""
        agent_config = self._config.get("agent", {})

        # Core templates from mini.yaml
        self._templates["mini_swe.system"] = agent_config.get("system_template", "")
        self._templates["mini_swe.instance"] = agent_config.get("instance_template", "")
        self._templates["mini_swe.action_observation"] = agent_config.get("action_observation_template", "")
        self._templates["mini_swe.format_error"] = agent_config.get("format_error_template", "")
        self._templates["mini_swe.timeout"] = agent_config.get("timeout_template", "")

        # Combined prompt for single-agent mode (system + instance)
        self._templates["mini_swe.swe_agent"] = self._combine_templates(
            agent_config.get("system_template", ""),
            agent_config.get("instance_template", ""),
        )

        # Pipeline stage prompts
        self._templates["mini_swe.task_parser"] = self._create_task_parser_prompt()
        self._templates["mini_swe.code_searcher"] = self._create_code_searcher_prompt()
        self._templates["mini_swe.file_analyzer"] = self._create_file_analyzer_prompt()
        self._templates["mini_swe.planner"] = self._create_planner_prompt()
        self._templates["mini_swe.executor"] = self._create_executor_prompt()
        self._templates["mini_swe.verifier"] = self._create_verifier_prompt()

    def _combine_templates(self, system: str, instance: str) -> str:
        """Combine system and instance templates for single-agent mode.

        Args:
            system: System template
            instance: Instance template

        Returns:
            Combined template
        """
        return f"{system}\n\n---\n\n{instance}"

    def _create_task_parser_prompt(self) -> str:
        """Create prompt for task parser stage."""
        return """You are a task analysis assistant. Your job is to understand and break down the given task.

Given the following task:
{{task}}

Analyze the task and provide:
1. A clear summary of what needs to be done
2. Key files or areas of code that might be involved
3. Any potential challenges or considerations

Format your response as JSON:
{
  "summary": "Brief summary of the task",
  "likely_files": ["list", "of", "likely", "files"],
  "considerations": ["list", "of", "considerations"]
}"""

    def _create_code_searcher_prompt(self) -> str:
        """Create prompt for code searcher stage."""
        return """You are a code search specialist. Your job is to find relevant code for the task.

Task summary: {{task_info.summary}}
Likely files: {{task_info.likely_files}}

Use the available tools (find_files, grep_search, bash_execute) to:
1. Locate relevant source files
2. Find related function/class definitions
3. Identify dependencies and imports

Report your findings as a structured summary of relevant code locations."""

    def _create_file_analyzer_prompt(self) -> str:
        """Create prompt for file analyzer stage."""
        return """You are a file analysis specialist. Your job is to understand the structure of relevant files.

Task summary: {{task_info.summary}}
Likely files: {{task_info.likely_files}}

Use the available tools (read_file, bash_execute) to:
1. Read and understand key file contents
2. Identify patterns and coding conventions
3. Understand the relationships between components

Report your analysis of the file structure and content."""

    def _create_planner_prompt(self) -> str:
        """Create prompt for planner stage."""
        return """You are a planning specialist. Your job is to create an implementation plan.

Task summary: {{task_info.summary}}
Search results: {{search_results}}
Analysis results: {{analysis_results}}

Create a detailed plan that includes:
1. Step-by-step implementation approach
2. Specific files to modify
3. Code changes to make
4. Testing strategy

Format your plan clearly with numbered steps."""

    def _create_executor_prompt(self) -> str:
        """Create prompt for executor stage."""
        return """You are an implementation specialist. Your job is to execute the plan.

Plan: {{plan}}

Use the available tools (bash_execute, write_file, edit_file) to:
1. Make the necessary code changes
2. Follow the plan step by step
3. Handle any issues that arise

Report the changes you made and any issues encountered."""

    def _create_verifier_prompt(self) -> str:
        """Create prompt for verifier stage."""
        return """You are a verification specialist. Your job is to verify the implementation.

Execution results: {{execution}}

Use the available tools (bash_execute, run_tests) to:
1. Run relevant tests
2. Verify the changes work correctly
3. Check for any regressions

Report your verification results with verdict:
- "success" if all tests pass and implementation is correct
- "loop_back" if changes need revision"""

    def get(self, key: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Get a rendered prompt template.

        Args:
            key: Template key (e.g., "mini_swe.system")
            context: Context dictionary for template rendering
            **kwargs: Additional context variables

        Returns:
            Rendered prompt string
        """
        template_str = self._templates.get(key)

        if template_str is None:
            # Fall back to using the key as the template itself
            template_str = key

        # Build context with defaults
        all_context = self._get_default_context()
        if context:
            all_context.update(context)
        all_context.update(kwargs)

        # Render template
        try:
            template = Template(template_str, undefined=StrictUndefined)
            return template.render(**all_context)
        except Exception as e:
            # Return template with error message if rendering fails
            return f"Template rendering error for {key}: {e}\n\n{template_str}"

    def _get_default_context(self) -> Dict[str, Any]:
        """Get default context variables for template rendering.

        Returns:
            Dictionary with default context variables
        """
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        }

    def register(self, key: str, template: str) -> None:
        """Register a new template.

        Args:
            key: Template key
            template: Template string
        """
        self._templates[key] = template

    def list_keys(self) -> list:
        """List all registered template keys.

        Returns:
            List of template keys
        """
        return list(self._templates.keys())


def create_prompt_registry(
    config_path: Optional[Path] = None,
) -> MiniSWEPromptRegistry:
    """Factory function to create a prompt registry.

    Args:
        config_path: Path to config file (optional)

    Returns:
        MiniSWEPromptRegistry instance
    """
    return MiniSWEPromptRegistry(config_path=config_path)


__all__ = [
    "MiniSWEPromptRegistry",
    "create_prompt_registry",
    "DEFAULT_CONFIG_PATH",
]
