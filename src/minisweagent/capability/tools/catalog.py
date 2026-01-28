"""Tool Catalog for Mini-SWE-Agent.

This module defines capability-owned tools for code modification tasks.
Tools are registered with the CapabilityToolCatalog from jeeves-core.

Constitutional Reference (Contract 10):
- ToolId enums are CAPABILITY-OWNED, not defined in avionics or mission_system
- This ensures layer violations are avoided (L5 importing L2 for tool identifiers)
- Allows capabilities to define their own tool sets independently

Usage:
    catalog = create_tool_catalog()
    tool = catalog.get_tool("bash_execute")
    result = await tool.function({"command": "ls -la"})
"""

import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

# jeeves-core is now a proper package - no sys.path manipulation needed

from protocols.capability import CapabilityToolCatalog
from jeeves_infra.protocols import RiskLevel, ToolCategory

# =============================================================================
# TOOL ID ENUM (CAPABILITY-OWNED per Contract 10)
# =============================================================================


class ToolId(str, Enum):
    """Capability-owned tool identifiers for mini-swe-agent.

    These are the tools available to the SWE agent for code modification tasks.
    Risk levels indicate required confirmation:
    - LOW: No confirmation needed
    - MEDIUM: Log but don't block
    - HIGH: Require user confirmation
    """

    # Execution tools
    BASH_EXECUTE = "bash_execute"

    # File operations
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EDIT_FILE = "edit_file"

    # Search tools
    FIND_FILES = "find_files"
    GREP_SEARCH = "grep_search"

    # Test tools
    RUN_TESTS = "run_tests"

    # v2.0 tools
    SEMANTIC_SEARCH = "semantic_search"
    GRAPH_QUERY = "graph_query"


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

# Global configuration
_TOOL_CONFIG = {
    "timeout": int(os.getenv("MSWEA_TOOL_TIMEOUT", "30")),
    "cwd": os.getenv("MSWEA_WORKING_DIR", os.getcwd()),
    "env": {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
    },
}


async def bash_execute(
    command: str,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a bash command in a subprocess.

    This is the primary tool for the SWE agent. It executes commands
    in a subprocess and returns the output.

    Args:
        command: The bash command to execute
        cwd: Working directory (defaults to configured cwd)
        timeout: Command timeout in seconds (defaults to configured timeout)

    Returns:
        Dict with 'output', 'returncode', and 'status' keys
    """
    working_dir = cwd or _TOOL_CONFIG["cwd"]
    cmd_timeout = timeout or _TOOL_CONFIG["timeout"]

    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            cwd=working_dir,
            env=os.environ | _TOOL_CONFIG["env"],
            timeout=cmd_timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {
            "status": "success",
            "output": result.stdout,
            "returncode": result.returncode,
            "command": command,
        }
    except subprocess.TimeoutExpired as e:
        output = e.output.decode("utf-8", errors="replace") if e.output else ""
        return {
            "status": "timeout",
            "output": output,
            "returncode": -1,
            "error": f"Command timed out after {cmd_timeout}s",
            "command": command,
        }
    except Exception as e:
        return {
            "status": "error",
            "output": "",
            "returncode": -1,
            "error": str(e),
            "command": command,
        }


async def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> Dict[str, Any]:
    """Read a file and return its contents.

    Args:
        path: Path to the file (absolute or relative to cwd)
        start_line: Optional starting line number (1-indexed)
        end_line: Optional ending line number (1-indexed)

    Returns:
        Dict with 'content', 'path', and 'status' keys
    """
    try:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(_TOOL_CONFIG["cwd"]) / file_path

        if not file_path.exists():
            return {
                "status": "not_found",
                "error": f"File not found: {path}",
                "path": str(file_path),
            }

        content = file_path.read_text(encoding="utf-8", errors="replace")

        # Apply line filtering if specified
        if start_line is not None or end_line is not None:
            lines = content.splitlines(keepends=True)
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else len(lines)
            content = "".join(lines[start_idx:end_idx])

        return {
            "status": "success",
            "content": content,
            "path": str(file_path),
            "size": len(content),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "path": path,
        }


async def write_file(
    path: str,
    content: str,
    create_dirs: bool = True,
) -> Dict[str, Any]:
    """Write content to a file.

    Args:
        path: Path to the file
        content: Content to write
        create_dirs: Whether to create parent directories if they don't exist

    Returns:
        Dict with 'path', 'size', and 'status' keys
    """
    try:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(_TOOL_CONFIG["cwd"]) / file_path

        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(content, encoding="utf-8")

        return {
            "status": "success",
            "path": str(file_path),
            "size": len(content),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "path": path,
        }


async def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> Dict[str, Any]:
    """Edit a file by replacing text.

    Args:
        path: Path to the file
        old_text: Text to find and replace
        new_text: Replacement text
        replace_all: If True, replace all occurrences; otherwise just first

    Returns:
        Dict with 'path', 'replacements', and 'status' keys
    """
    try:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(_TOOL_CONFIG["cwd"]) / file_path

        if not file_path.exists():
            return {
                "status": "not_found",
                "error": f"File not found: {path}",
                "path": str(file_path),
            }

        content = file_path.read_text(encoding="utf-8", errors="replace")

        # Count occurrences
        count = content.count(old_text)
        if count == 0:
            return {
                "status": "not_found",
                "error": f"Text not found in file: {old_text[:50]}...",
                "path": str(file_path),
                "replacements": 0,
            }

        # Perform replacement
        if replace_all:
            new_content = content.replace(old_text, new_text)
            replacements = count
        else:
            new_content = content.replace(old_text, new_text, 1)
            replacements = 1

        file_path.write_text(new_content, encoding="utf-8")

        return {
            "status": "success",
            "path": str(file_path),
            "replacements": replacements,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "path": path,
        }


async def find_files(
    pattern: str,
    path: Optional[str] = None,
    max_results: int = 100,
) -> Dict[str, Any]:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern to match (e.g., "**/*.py")
        path: Starting directory (defaults to cwd)
        max_results: Maximum number of results to return

    Returns:
        Dict with 'files', 'count', and 'status' keys
    """
    try:
        search_path = Path(path) if path else Path(_TOOL_CONFIG["cwd"])
        if not search_path.is_absolute():
            search_path = Path(_TOOL_CONFIG["cwd"]) / search_path

        files = []
        for match in search_path.glob(pattern):
            if match.is_file():
                files.append(str(match.relative_to(search_path)))
                if len(files) >= max_results:
                    break

        return {
            "status": "success",
            "files": files,
            "count": len(files),
            "truncated": len(files) >= max_results,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "pattern": pattern,
        }


async def grep_search(
    pattern: str,
    path: Optional[str] = None,
    file_pattern: str = "*",
    max_results: int = 100,
) -> Dict[str, Any]:
    """Search for a pattern in files using grep.

    Args:
        pattern: Regex pattern to search for
        path: Directory to search in (defaults to cwd)
        file_pattern: Glob pattern for files to search
        max_results: Maximum number of matches to return

    Returns:
        Dict with 'matches', 'count', and 'status' keys
    """
    search_path = path or _TOOL_CONFIG["cwd"]

    # Use grep command for efficiency
    cmd = f"grep -rn --include='{file_pattern}' -E '{pattern}' . | head -n {max_results}"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            cwd=search_path,
            timeout=_TOOL_CONFIG["timeout"],
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        matches = []
        for line in result.stdout.strip().split("\n"):
            if line:
                # Parse grep output: ./file:line:content
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    matches.append({
                        "file": parts[0].lstrip("./"),
                        "line": int(parts[1]) if parts[1].isdigit() else 0,
                        "content": parts[2].strip(),
                    })

        return {
            "status": "success",
            "matches": matches,
            "count": len(matches),
            "truncated": len(matches) >= max_results,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": "Search timed out",
            "matches": [],
            "count": 0,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "pattern": pattern,
        }


async def semantic_search(
    query: str,
    limit: int = 5,
    min_score: float = 0.7,
) -> Dict[str, Any]:
    """Search codebase using natural language query (L3 semantic search).

    This tool uses embeddings to find code semantically similar to the query.
    Much better than grep for conceptual searches.

    Args:
        query: Natural language description of code to find
        limit: Maximum number of results
        min_score: Minimum similarity score (0-1)

    Returns:
        Dict with 'results', 'count', and 'status' keys

    Example:
        semantic_search("password validation logic", limit=5)
        semantic_search("database connection setup", limit=3, min_score=0.8)
    """
    try:
        # Import service (lazy load)
        from minisweagent.capability.services import CodeIndexerService
        import asyncpg

        # Get database connection
        db_url = os.getenv("MSWEA_DATABASE_URL")
        if not db_url:
            return {
                "status": "error",
                "error": "MSWEA_DATABASE_URL not set. Semantic search requires database.",
                "results": [],
                "count": 0,
            }

        # Connect and search
        conn = await asyncpg.connect(db_url)
        try:
            indexer = CodeIndexerService(conn)
            chunks = await indexer.search(query, limit=limit, min_score=min_score)

            results = [
                {
                    "file": chunk.source_file,
                    "content": chunk.content,
                    "score": chunk.score,
                    "line_start": chunk.metadata.get("line_start"),
                    "line_end": chunk.metadata.get("line_end"),
                }
                for chunk in chunks
            ]

            return {
                "status": "success",
                "results": results,
                "count": len(results),
                "query": query,
            }
        finally:
            await conn.close()

    except ImportError as e:
        return {
            "status": "error",
            "error": f"Missing dependency: {e}. Install with: pip install sentence-transformers",
            "results": [],
            "count": 0,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "results": [],
            "count": 0,
        }


async def graph_query(
    query_type: str,
    node_id: str,
    max_depth: int = 3,
) -> Dict[str, Any]:
    """Query code dependency graph (L5 graph storage).

    This tool queries the entity relationship graph to find dependencies,
    dependents, and circular dependencies.

    Args:
        query_type: Type of query:
            - "depends_on": What does node_id depend on?
            - "used_by": What depends on node_id?
            - "circular": Find circular dependencies from node_id
        node_id: Node identifier (e.g., "file:auth.py" or "function:login")
        max_depth: Maximum traversal depth

    Returns:
        Dict with 'nodes', 'count', and 'status' keys

    Examples:
        graph_query("used_by", "file:auth.py")
        graph_query("depends_on", "class:User", max_depth=2)
        graph_query("circular", "file:models.py")
    """
    try:
        # Import service (lazy load)
        from minisweagent.capability.services import GraphService
        import asyncpg

        # Get database connection
        db_url = os.getenv("MSWEA_DATABASE_URL")
        if not db_url:
            return {
                "status": "error",
                "error": "MSWEA_DATABASE_URL not set. Graph queries require database.",
                "nodes": [],
                "count": 0,
            }

        # Connect and query
        conn = await asyncpg.connect(db_url)
        try:
            graph = GraphService(conn)

            if query_type == "depends_on":
                neighbors = await graph.query_neighbors(
                    node_id, edge_type="imports", direction="outgoing", max_depth=max_depth
                )
                results = [
                    {
                        "node_id": n.node_id,
                        "node_type": n.node_type,
                        "metadata": n.metadata,
                    }
                    for n in neighbors
                ]

            elif query_type == "used_by":
                neighbors = await graph.query_neighbors(
                    node_id, edge_type="imports", direction="incoming", max_depth=max_depth
                )
                results = [
                    {
                        "node_id": n.node_id,
                        "node_type": n.node_type,
                        "metadata": n.metadata,
                    }
                    for n in neighbors
                ]

            elif query_type == "circular":
                cycles = await graph.find_cycles(node_id, max_depth=max_depth)
                results = [{"cycle": cycle} for cycle in cycles]

            else:
                return {
                    "status": "error",
                    "error": f"Unknown query type: {query_type}. Use 'depends_on', 'used_by', or 'circular'.",
                    "nodes": [],
                    "count": 0,
                }

            return {
                "status": "success",
                "nodes": results,
                "count": len(results),
                "query_type": query_type,
                "node_id": node_id,
            }
        finally:
            await conn.close()

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "nodes": [],
            "count": 0,
        }


async def run_tests(
    test_cmd: Optional[str] = None,
    test_path: Optional[str] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Run tests using the project's test command.

    Args:
        test_cmd: Custom test command (auto-detects if not provided)
        test_path: Specific test file or directory
        timeout: Test timeout in seconds

    Returns:
        Dict with 'output', 'returncode', and 'status' keys
    """
    cwd = _TOOL_CONFIG["cwd"]

    # Auto-detect test command if not provided
    if not test_cmd:
        if (Path(cwd) / "pytest.ini").exists() or (Path(cwd) / "pyproject.toml").exists():
            test_cmd = "pytest"
        elif (Path(cwd) / "package.json").exists():
            test_cmd = "npm test"
        elif (Path(cwd) / "Makefile").exists():
            test_cmd = "make test"
        else:
            test_cmd = "pytest"  # Default to pytest

    if test_path:
        test_cmd = f"{test_cmd} {test_path}"

    return await bash_execute(test_cmd, timeout=timeout)


# =============================================================================
# CATALOG CREATION
# =============================================================================

_catalog_instance: Optional[CapabilityToolCatalog] = None


def create_tool_catalog() -> CapabilityToolCatalog:
    """Create and populate the tool catalog for mini-swe-agent.

    Returns:
        CapabilityToolCatalog with all tools registered
    """
    global _catalog_instance
    if _catalog_instance is not None:
        return _catalog_instance

    catalog = CapabilityToolCatalog("mini-swe-agent")

    # Register bash_execute (HIGH risk - executes arbitrary commands)
    catalog.register(
        tool_id=ToolId.BASH_EXECUTE.value,
        func=bash_execute,
        description="Execute a bash command in a subprocess",
        parameters={
            "command": "string",
            "cwd": "string?",
            "timeout": "integer?",
        },
        category=ToolCategory.EXECUTE.value,
        risk_level=RiskLevel.HIGH.value,
    )

    # Register read_file (READ_ONLY - safe)
    catalog.register(
        tool_id=ToolId.READ_FILE.value,
        func=read_file,
        description="Read contents of a file",
        parameters={
            "path": "string",
            "start_line": "integer?",
            "end_line": "integer?",
        },
        category=ToolCategory.READ.value,
        risk_level=RiskLevel.READ_ONLY.value,
    )

    # Register write_file (WRITE risk - creates/overwrites files)
    catalog.register(
        tool_id=ToolId.WRITE_FILE.value,
        func=write_file,
        description="Write content to a file",
        parameters={
            "path": "string",
            "content": "string",
            "create_dirs": "boolean?",
        },
        category=ToolCategory.WRITE.value,
        risk_level=RiskLevel.WRITE.value,
    )

    # Register edit_file (WRITE risk - modifies files)
    catalog.register(
        tool_id=ToolId.EDIT_FILE.value,
        func=edit_file,
        description="Edit a file by replacing text",
        parameters={
            "path": "string",
            "old_text": "string",
            "new_text": "string",
            "replace_all": "boolean?",
        },
        category=ToolCategory.WRITE.value,
        risk_level=RiskLevel.WRITE.value,
    )

    # Register find_files (READ_ONLY - safe)
    catalog.register(
        tool_id=ToolId.FIND_FILES.value,
        func=find_files,
        description="Find files matching a glob pattern",
        parameters={
            "pattern": "string",
            "path": "string?",
            "max_results": "integer?",
        },
        category=ToolCategory.READ.value,
        risk_level=RiskLevel.READ_ONLY.value,
    )

    # Register grep_search (READ_ONLY - safe)
    catalog.register(
        tool_id=ToolId.GREP_SEARCH.value,
        func=grep_search,
        description="Search for a pattern in files",
        parameters={
            "pattern": "string",
            "path": "string?",
            "file_pattern": "string?",
            "max_results": "integer?",
        },
        category=ToolCategory.READ.value,
        risk_level=RiskLevel.READ_ONLY.value,
    )

    # Register run_tests (MEDIUM risk - executes test commands)
    catalog.register(
        tool_id=ToolId.RUN_TESTS.value,
        func=run_tests,
        description="Run project tests",
        parameters={
            "test_cmd": "string?",
            "test_path": "string?",
            "timeout": "integer?",
        },
        category=ToolCategory.EXECUTE.value,
        risk_level=RiskLevel.MEDIUM.value,
    )

    # Register semantic_search (READ_ONLY - safe, requires database)
    catalog.register(
        tool_id=ToolId.SEMANTIC_SEARCH.value,
        func=semantic_search,
        description="Search codebase using natural language (semantic search)",
        parameters={
            "query": "string",
            "limit": "integer?",
            "min_score": "number?",
        },
        category=ToolCategory.READ.value,
        risk_level=RiskLevel.READ_ONLY.value,
    )

    # Register graph_query (READ_ONLY - safe, requires database)
    catalog.register(
        tool_id=ToolId.GRAPH_QUERY.value,
        func=graph_query,
        description="Query code dependency graph",
        parameters={
            "query_type": "string",
            "node_id": "string",
            "max_depth": "integer?",
        },
        category=ToolCategory.READ.value,
        risk_level=RiskLevel.READ_ONLY.value,
    )

    _catalog_instance = catalog
    return catalog


def get_tool_catalog() -> CapabilityToolCatalog:
    """Get the singleton tool catalog instance.

    Returns:
        CapabilityToolCatalog instance
    """
    if _catalog_instance is None:
        return create_tool_catalog()
    return _catalog_instance


def configure_tools(
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    """Configure tool execution settings.

    Args:
        cwd: Working directory for command execution
        timeout: Default timeout for commands
        env: Environment variables to set
    """
    if cwd is not None:
        _TOOL_CONFIG["cwd"] = cwd
    if timeout is not None:
        _TOOL_CONFIG["timeout"] = timeout
    if env is not None:
        _TOOL_CONFIG["env"].update(env)


__all__ = [
    "ToolId",
    "create_tool_catalog",
    "get_tool_catalog",
    "configure_tools",
    # Tool functions (for direct use)
    "bash_execute",
    "read_file",
    "write_file",
    "edit_file",
    "find_files",
    "grep_search",
    "run_tests",
    "semantic_search",
    "graph_query",
    "grep_search",
    "run_tests",
]
