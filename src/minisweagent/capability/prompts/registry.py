"""Prompt registry implementation for mini-swe-agent (v2.0).

This module provides the PromptRegistry implementation that loads templates
from mini.yaml and renders them with Jinja2.

v2.0 Features:
- Session context variables (previous_findings, focus_state, session_id)
- Semantic search usage examples
- Graph query usage examples
- Ambiguity detection support
"""

import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Template, StrictUndefined

# jeeves-core is now a proper package - no sys.path manipulation needed

from jeeves_infra.runtime import PromptRegistry


# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "mini.yaml"


# =============================================================================
# v2.0: SESSION CONTEXT TEMPLATES
# =============================================================================

SESSION_CONTEXT_TEMPLATE = """
{% if working_memory %}
## Session Context (v2.0)
Session ID: {{working_memory.session_id}}
Previous findings: {{working_memory.findings_count}} discoveries from earlier in this session

{% if working_memory.previous_findings %}
### Previous Findings
{% for finding in working_memory.previous_findings %}
- [{{finding.source}}] {{finding.content}}
{% endfor %}
{% endif %}
{% endif %}

{% if focus_state %}
### Current Focus
{% if focus_state.current_file %}Current file: {{focus_state.current_file}}{% endif %}
{% if focus_state.current_function %}Current function: {{focus_state.current_function}}{% endif %}
{% if focus_state.current_task %}Current task: {{focus_state.current_task}}{% endif %}
{% endif %}
"""

SEMANTIC_SEARCH_USAGE = """
## Semantic Search (v2.0)
Use `semantic_search` for conceptual queries:
- "password validation logic" → finds related code by meaning
- "authentication handlers" → finds auth-related functions
- Better than grep for understanding-based searches

Example:
```
semantic_search("database connection pooling", limit=5)
```
"""

GRAPH_QUERY_USAGE = """
## Graph Queries (v2.0)
Use `graph_query` for dependency analysis:
- "depends_on" → what does this file/function use?
- "used_by" → what depends on this file/function?
- "circular" → find circular dependencies

Examples:
```
graph_query("used_by", "file:auth.py")  # Who uses auth.py?
graph_query("depends_on", "file:models.py")  # What does models.py need?
```
"""


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
        """Create prompt for task parser stage (v2.0 with session context)."""
        return """You are a task analysis assistant. Your job is to understand and break down the given task.
""" + SESSION_CONTEXT_TEMPLATE + """
Given the following task:
{{task}}

Analyze the task and provide:
1. A clear summary of what needs to be done
2. Key files or areas of code that might be involved
3. Any potential challenges or considerations
4. Whether the task is ambiguous and needs clarification

Format your response as JSON:
{
  "summary": "Brief summary of the task",
  "likely_files": ["list", "of", "likely", "files"],
  "considerations": ["list", "of", "considerations"],
  "ambiguous": false,
  "clarification_question": null,
  "options": []
}

If the task is ambiguous, set "ambiguous": true and provide a clarification question with options.
Example for ambiguous task:
{
  "summary": "Fix bug in login",
  "likely_files": ["auth.py", "login.py"],
  "considerations": ["Multiple possible bugs"],
  "ambiguous": true,
  "clarification_question": "Which login issue should I address?",
  "options": ["Password validation error", "Session timeout issue", "OAuth callback failure"]
}"""

    def _create_code_searcher_prompt(self) -> str:
        """Create prompt for code searcher stage (v2.0 with semantic search)."""
        return """You are a code search specialist. Your job is to find relevant code for the task.
""" + SESSION_CONTEXT_TEMPLATE + """
Task summary: {{task_info.summary}}
Likely files: {{task_info.likely_files}}
""" + SEMANTIC_SEARCH_USAGE + """
Use the available tools to:
1. Use `semantic_search` for conceptual queries (e.g., "authentication logic")
2. Use `find_files` and `grep_search` for exact matches
3. Find related function/class definitions
4. Identify dependencies and imports

Report your findings as a structured summary of relevant code locations."""

    def _create_file_analyzer_prompt(self) -> str:
        """Create prompt for file analyzer stage (v2.0 with graph queries)."""
        return """You are a file analysis specialist. Your job is to understand the structure of relevant files.
""" + SESSION_CONTEXT_TEMPLATE + """
Task summary: {{task_info.summary}}
Likely files: {{task_info.likely_files}}
""" + GRAPH_QUERY_USAGE + """
Use the available tools to:
1. Read and understand key file contents using `read_file`
2. Use `graph_query` to understand dependencies between files
3. Identify patterns and coding conventions
4. Understand the relationships between components

Report your analysis of the file structure, dependencies, and content."""

    def _create_planner_prompt(self) -> str:
        """Create prompt for planner stage (v2.0 with session context)."""
        return """You are a planning specialist. Your job is to create an implementation plan.
""" + SESSION_CONTEXT_TEMPLATE + """
Task summary: {{task_info.summary}}
Search results: {{search_results}}
Analysis results: {{analysis_results}}

Create a detailed plan that includes:
1. Step-by-step implementation approach
2. Specific files to modify (prioritize based on dependency graph)
3. Code changes to make
4. Testing strategy
5. Rollback considerations

Format your plan as JSON:
{
  "steps": [
    {
      "step": 1,
      "action": "Modify auth.py",
      "details": "Add password validation",
      "files": ["auth.py"],
      "risk": "low"
    }
  ],
  "test_strategy": "Run unit tests and integration tests",
  "estimated_impact": ["auth.py", "login.py"]
}"""

    def _create_executor_prompt(self) -> str:
        """Create prompt for executor stage (v2.0 with confirmation support)."""
        return """You are an implementation specialist. Your job is to execute the plan.
""" + SESSION_CONTEXT_TEMPLATE + """
Plan: {{plan}}

Use the available tools to:
1. Make the necessary code changes using `edit_file` or `write_file`
2. Follow the plan step by step
3. Handle any issues that arise
4. Report changes for confirmation if required

IMPORTANT: For destructive operations (delete, overwrite):
- The tool will ask for user confirmation
- Wait for confirmation before proceeding

Report the changes you made and any issues encountered.
Format:
{
  "files_modified": ["file1.py", "file2.py"],
  "changes_made": [
    {"file": "file1.py", "change": "Added validation function"}
  ],
  "issues": [],
  "needs_retry": false
}"""

    def _create_verifier_prompt(self) -> str:
        """Create prompt for verifier stage (v2.0 with comprehensive checks)."""
        return """You are a verification specialist. Your job is to verify the implementation.
""" + SESSION_CONTEXT_TEMPLATE + """
Execution results: {{execution}}

Use the available tools to:
1. Run relevant tests using `run_tests` or `bash_execute pytest`
2. Verify the changes work correctly
3. Check for any regressions
4. Use `graph_query` to verify no unexpected dependencies broken

Verification checklist:
- [ ] All tests pass
- [ ] Changes match the plan
- [ ] No new linting errors
- [ ] Dependencies intact

Report your verification results:
{
  "verdict": "success|loop_back",
  "tests_run": 10,
  "tests_passed": 10,
  "tests_failed": 0,
  "issues": [],
  "response": "All tests pass. Implementation verified successfully."
}

Verdicts:
- "success": All tests pass and implementation is correct
- "loop_back": Changes need revision (provide specific issues)"""

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
