"""Tests for GraphExtractor."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import tempfile
from pathlib import Path

from minisweagent.capability.agents.graph_extractor import GraphExtractor


class MockGraphService:
    """Mock graph service for testing."""

    def __init__(self):
        self.nodes = {}
        self.edges = []

    async def add_node(self, node_id: str, node_type: str, properties: dict):
        """Mock add_node."""
        self.nodes[node_id] = {"type": node_type, "properties": properties}

    async def add_edge(self, from_id: str, to_id: str, edge_type: str, properties: dict = None):
        """Mock add_edge."""
        self.edges.append({
            "from": from_id,
            "to": to_id,
            "type": edge_type,
            "properties": properties or {}
        })


class TestGraphExtractorBasic:
    """Basic tests for GraphExtractor."""

    @pytest.fixture
    def graph_service(self):
        """Create mock graph service."""
        return MockGraphService()

    @pytest.fixture
    def extractor(self, graph_service):
        """Create extractor with mock service."""
        return GraphExtractor(graph_service)

    @pytest.mark.asyncio
    async def test_extract_file_node(self, extractor, graph_service):
        """Test file node extraction."""
        content = "# Empty file\n"

        await extractor.extract_from_file("test.py", content)

        assert "file:test.py" in graph_service.nodes
        assert graph_service.nodes["file:test.py"]["type"] == "file"
        assert graph_service.nodes["file:test.py"]["properties"]["path"] == "test.py"

    @pytest.mark.asyncio
    async def test_extract_handles_syntax_error(self, extractor, graph_service):
        """Test graceful handling of syntax errors."""
        content = "def broken(\n"  # Invalid syntax

        # Should not raise
        await extractor.extract_from_file("broken.py", content)

        # File node should not be added on parse failure
        assert "file:broken.py" not in graph_service.nodes


class TestClassExtraction:
    """Tests for class extraction."""

    @pytest.fixture
    def graph_service(self):
        return MockGraphService()

    @pytest.fixture
    def extractor(self, graph_service):
        return GraphExtractor(graph_service)

    @pytest.mark.asyncio
    async def test_extract_class(self, extractor, graph_service):
        """Test class node extraction."""
        content = '''
class MyClass:
    pass
'''
        await extractor.extract_from_file("test.py", content)

        assert "class:test.py:MyClass" in graph_service.nodes
        node = graph_service.nodes["class:test.py:MyClass"]
        assert node["type"] == "class"
        assert node["properties"]["name"] == "MyClass"

    @pytest.mark.asyncio
    async def test_extract_class_with_inheritance(self, extractor, graph_service):
        """Test class inheritance edge extraction."""
        content = '''
class Parent:
    pass

class Child(Parent):
    pass
'''
        await extractor.extract_from_file("test.py", content)

        # Find inheritance edge
        inherit_edges = [e for e in graph_service.edges if e["type"] == "inherits"]
        assert len(inherit_edges) == 1
        assert inherit_edges[0]["from"] == "class:test.py:Child"
        assert inherit_edges[0]["to"] == "class:test.py:Parent"

    @pytest.mark.asyncio
    async def test_file_defines_class(self, extractor, graph_service):
        """Test file -> class 'defines' edge."""
        content = '''
class MyClass:
    pass
'''
        await extractor.extract_from_file("test.py", content)

        defines_edges = [e for e in graph_service.edges if e["type"] == "defines"]
        assert any(
            e["from"] == "file:test.py" and e["to"] == "class:test.py:MyClass"
            for e in defines_edges
        )


class TestFunctionExtraction:
    """Tests for function extraction."""

    @pytest.fixture
    def graph_service(self):
        return MockGraphService()

    @pytest.fixture
    def extractor(self, graph_service):
        return GraphExtractor(graph_service)

    @pytest.mark.asyncio
    async def test_extract_function(self, extractor, graph_service):
        """Test function node extraction."""
        content = '''
def my_function():
    pass
'''
        await extractor.extract_from_file("test.py", content)

        assert "function:test.py:my_function" in graph_service.nodes
        node = graph_service.nodes["function:test.py:my_function"]
        assert node["type"] == "function"
        assert node["properties"]["name"] == "my_function"
        assert node["properties"]["async"] is False

    @pytest.mark.asyncio
    async def test_extract_async_function(self, extractor, graph_service):
        """Test async function extraction."""
        content = '''
async def async_func():
    pass
'''
        await extractor.extract_from_file("test.py", content)

        assert "function:test.py:async_func" in graph_service.nodes
        node = graph_service.nodes["function:test.py:async_func"]
        assert node["properties"]["async"] is True

    @pytest.mark.asyncio
    async def test_extract_function_calls(self, extractor, graph_service):
        """Test function call edge extraction."""
        content = '''
def helper():
    pass

def main():
    helper()
'''
        await extractor.extract_from_file("test.py", content)

        call_edges = [e for e in graph_service.edges if e["type"] == "calls"]
        assert any(
            e["from"] == "function:test.py:main" and e["to"] == "function:test.py:helper"
            for e in call_edges
        )

    @pytest.mark.asyncio
    async def test_extract_method_in_class(self, extractor, graph_service):
        """Test method extraction within class."""
        content = '''
class MyClass:
    def my_method(self):
        pass
'''
        await extractor.extract_from_file("test.py", content)

        assert "function:test.py:my_method" in graph_service.nodes

        # Class should define the method
        defines_edges = [e for e in graph_service.edges if e["type"] == "defines"]
        assert any(
            e["from"] == "class:test.py:MyClass" and e["to"] == "function:test.py:my_method"
            for e in defines_edges
        )


class TestImportExtraction:
    """Tests for import extraction."""

    @pytest.fixture
    def graph_service(self):
        return MockGraphService()

    @pytest.fixture
    def extractor(self, graph_service):
        return GraphExtractor(graph_service)

    @pytest.mark.asyncio
    async def test_extract_import(self, extractor, graph_service):
        """Test import edge extraction."""
        content = '''
import os
'''
        await extractor.extract_from_file("test.py", content)

        import_edges = [e for e in graph_service.edges if e["type"] == "imports"]
        assert any(
            e["from"] == "file:test.py" and "os.py" in e["to"]
            for e in import_edges
        )

    @pytest.mark.asyncio
    async def test_extract_from_import(self, extractor, graph_service):
        """Test from-import edge extraction."""
        content = '''
from pathlib import Path
'''
        await extractor.extract_from_file("test.py", content)

        import_edges = [e for e in graph_service.edges if e["type"] == "imports"]
        assert any(
            e["from"] == "file:test.py" and "pathlib" in e["to"]
            for e in import_edges
        )
        # Check that imported name is in properties
        path_import = [e for e in import_edges if "pathlib" in e["to"]][0]
        assert path_import["properties"]["name"] == "Path"

    @pytest.mark.asyncio
    async def test_extract_import_with_alias(self, extractor, graph_service):
        """Test import with alias."""
        content = '''
import numpy as np
'''
        await extractor.extract_from_file("test.py", content)

        import_edges = [e for e in graph_service.edges if e["type"] == "imports"]
        np_import = [e for e in import_edges if "numpy" in e["to"]][0]
        assert np_import["properties"]["alias"] == "np"


class TestDirectoryExtraction:
    """Tests for directory extraction."""

    @pytest.fixture
    def graph_service(self):
        return MockGraphService()

    @pytest.fixture
    def extractor(self, graph_service):
        return GraphExtractor(graph_service)

    @pytest.mark.asyncio
    async def test_extract_from_directory(self, extractor, graph_service):
        """Test extraction from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = Path(tmpdir) / "file1.py"
            file1.write_text("def func1(): pass")

            file2 = Path(tmpdir) / "file2.py"
            file2.write_text("def func2(): pass")

            await extractor.extract_from_directory(tmpdir)

            # Both files should be extracted
            file_nodes = [k for k in graph_service.nodes.keys() if k.startswith("file:")]
            assert len(file_nodes) == 2

    @pytest.mark.asyncio
    async def test_extract_from_directory_with_pattern(self, extractor, graph_service):
        """Test extraction with specific pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            py_file = Path(tmpdir) / "code.py"
            py_file.write_text("def func(): pass")

            txt_file = Path(tmpdir) / "notes.txt"
            txt_file.write_text("Not a Python file")

            await extractor.extract_from_directory(tmpdir, pattern="*.py")

            # Only Python file should be extracted
            file_nodes = [k for k in graph_service.nodes.keys() if k.startswith("file:")]
            assert len(file_nodes) == 1
            assert "code.py" in file_nodes[0]

    @pytest.mark.asyncio
    async def test_extract_from_directory_handles_errors(self, extractor, graph_service):
        """Test graceful error handling during directory extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid file
            valid_file = Path(tmpdir) / "valid.py"
            valid_file.write_text("def func(): pass")

            # Create a file that will fail to parse
            invalid_file = Path(tmpdir) / "invalid.py"
            invalid_file.write_text("def broken(\n")  # Syntax error

            # Should not raise
            await extractor.extract_from_directory(tmpdir)

            # Valid file should still be extracted
            assert any("valid.py" in k for k in graph_service.nodes.keys())
