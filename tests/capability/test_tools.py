"""Tests for capability tools (catalog.py)."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add jeeves-core to path
_jeeves_core_path = Path(__file__).parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))


class TestToolId:
    """Tests for the ToolId enum."""

    def test_tool_id_values(self):
        """Test that ToolId has expected values."""
        from minisweagent.capability.tools.catalog import ToolId

        assert ToolId.BASH_EXECUTE.value == "bash_execute"
        assert ToolId.READ_FILE.value == "read_file"
        assert ToolId.WRITE_FILE.value == "write_file"
        assert ToolId.EDIT_FILE.value == "edit_file"
        assert ToolId.FIND_FILES.value == "find_files"
        assert ToolId.GREP_SEARCH.value == "grep_search"
        assert ToolId.RUN_TESTS.value == "run_tests"


class TestToolCatalog:
    """Tests for tool catalog creation and registration."""

    def test_create_tool_catalog_registers_all_tools(self):
        """Test that create_tool_catalog registers all expected tools."""
        from minisweagent.capability.tools.catalog import create_tool_catalog, ToolId

        catalog = create_tool_catalog()

        # All ToolId values should be registered
        for tool_id in ToolId:
            assert catalog.has_tool(tool_id.value), f"Tool {tool_id.value} not registered"

    def test_get_tool_returns_definition(self):
        """Test that get_tool returns a valid tool definition."""
        from minisweagent.capability.tools.catalog import create_tool_catalog

        catalog = create_tool_catalog()
        tool = catalog.get_tool("bash_execute")

        assert tool is not None
        assert tool.name == "bash_execute"
        assert "command" in tool.parameters
        assert callable(tool.function)

    def test_catalog_generates_prompt_section(self):
        """Test that catalog can generate prompt section."""
        from minisweagent.capability.tools.catalog import create_tool_catalog

        catalog = create_tool_catalog()
        prompt = catalog.generate_prompt_section()

        assert "mini-swe-agent" in prompt
        assert "bash_execute" in prompt
        assert "read_file" in prompt


class TestBashExecute:
    """Tests for bash_execute tool."""

    @pytest.mark.asyncio
    async def test_bash_execute_simple_command(self):
        """Test bash_execute with a simple command."""
        from minisweagent.capability.tools.catalog import bash_execute

        result = await bash_execute("echo hello")

        assert result["status"] == "success"
        assert "hello" in result["output"]
        assert result["returncode"] == 0

    @pytest.mark.asyncio
    async def test_bash_execute_with_cwd(self):
        """Test bash_execute with custom working directory."""
        from minisweagent.capability.tools.catalog import bash_execute

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await bash_execute("pwd", cwd=tmpdir)

            assert result["status"] == "success"
            # Normalize paths for comparison
            assert Path(result["output"].strip()).resolve() == Path(tmpdir).resolve()

    @pytest.mark.asyncio
    async def test_bash_execute_timeout(self):
        """Test bash_execute timeout handling."""
        from minisweagent.capability.tools.catalog import bash_execute

        # This should timeout with a 1 second timeout
        result = await bash_execute("sleep 10", timeout=1)

        assert result["status"] == "timeout"
        assert "timed out" in result.get("error", "").lower()


class TestReadFile:
    """Tests for read_file tool."""

    @pytest.mark.asyncio
    async def test_read_file_success(self):
        """Test reading an existing file."""
        from minisweagent.capability.tools.catalog import read_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content\nline 2\nline 3")
            f.flush()
            temp_path = f.name

        try:
            result = await read_file(temp_path)

            assert result["status"] == "success"
            assert "test content" in result["content"]
            assert result["size"] > 0
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_read_file_with_line_range(self):
        """Test reading specific lines from a file."""
        from minisweagent.capability.tools.catalog import read_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5")
            f.flush()
            temp_path = f.name

        try:
            result = await read_file(temp_path, start_line=2, end_line=4)

            assert result["status"] == "success"
            assert "line 2" in result["content"]
            assert "line 3" in result["content"]
            assert "line 1" not in result["content"]
            assert "line 5" not in result["content"]
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_read_file_not_found(self):
        """Test reading a non-existent file."""
        from minisweagent.capability.tools.catalog import read_file

        result = await read_file("/nonexistent/path/file.txt")

        assert result["status"] == "not_found"
        assert "not found" in result.get("error", "").lower()


class TestWriteFile:
    """Tests for write_file tool."""

    @pytest.mark.asyncio
    async def test_write_file_creates_new_file(self):
        """Test writing a new file."""
        from minisweagent.capability.tools.catalog import write_file

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")
            result = await write_file(file_path, "test content")

            assert result["status"] == "success"
            assert os.path.exists(file_path)
            assert Path(file_path).read_text() == "test content"

    @pytest.mark.asyncio
    async def test_write_file_creates_directories(self):
        """Test that write_file creates parent directories."""
        from minisweagent.capability.tools.catalog import write_file

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "nested", "dir", "test.txt")
            result = await write_file(file_path, "content", create_dirs=True)

            assert result["status"] == "success"
            assert os.path.exists(file_path)


class TestEditFile:
    """Tests for edit_file tool."""

    @pytest.mark.asyncio
    async def test_edit_file_replaces_text(self):
        """Test replacing text in a file."""
        from minisweagent.capability.tools.catalog import edit_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\nhello again")
            f.flush()
            temp_path = f.name

        try:
            result = await edit_file(temp_path, "hello", "goodbye")

            assert result["status"] == "success"
            assert result["replacements"] == 1  # Only first occurrence

            content = Path(temp_path).read_text()
            assert "goodbye world" in content
            assert "hello again" in content  # Second occurrence unchanged
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_edit_file_replace_all(self):
        """Test replacing all occurrences."""
        from minisweagent.capability.tools.catalog import edit_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\nhello again")
            f.flush()
            temp_path = f.name

        try:
            result = await edit_file(temp_path, "hello", "goodbye", replace_all=True)

            assert result["status"] == "success"
            assert result["replacements"] == 2

            content = Path(temp_path).read_text()
            assert "hello" not in content
            assert content.count("goodbye") == 2
        finally:
            os.unlink(temp_path)


class TestFindFiles:
    """Tests for find_files tool."""

    @pytest.mark.asyncio
    async def test_find_files_by_pattern(self):
        """Test finding files by glob pattern."""
        from minisweagent.capability.tools.catalog import find_files

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "test1.py").write_text("# test")
            (Path(tmpdir) / "test2.py").write_text("# test")
            (Path(tmpdir) / "test.txt").write_text("test")

            result = await find_files("*.py", path=tmpdir)

            assert result["status"] == "success"
            assert result["count"] == 2
            assert "test1.py" in result["files"]
            assert "test2.py" in result["files"]
            assert "test.txt" not in result["files"]


class TestGrepSearch:
    """Tests for grep_search tool."""

    @pytest.mark.asyncio
    async def test_grep_search_finds_matches(self):
        """Test searching for pattern in files."""
        from minisweagent.capability.tools.catalog import grep_search

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            (Path(tmpdir) / "test.py").write_text("def hello():\n    return 'world'")

            result = await grep_search("hello", path=tmpdir)

            assert result["status"] == "success"
            assert result["count"] > 0
            assert any("hello" in m.get("content", "") for m in result["matches"])
