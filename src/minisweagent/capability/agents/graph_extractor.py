"""Graph Extractor - Extract code entities and relationships from Python files."""

import ast
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GraphExtractor:
    """Extract code entities and relationships from Python files."""

    def __init__(self, graph_service):
        """Initialize graph extractor.

        Args:
            graph_service: GraphService instance
        """
        self.graph = graph_service

    async def extract_from_file(self, file_path: str, content: str):
        """Extract entities and relationships from a Python file.

        Args:
            file_path: Path to file
            content: File content
        """
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return

        # Add file node
        await self.graph.add_node(
            f"file:{file_path}",
            "file",
            {"path": file_path, "lines": len(content.splitlines())}
        )

        # Extract top-level entities
        for node in ast.iter_child_nodes(tree):
            await self._extract_node(node, file_path, parent_id=f"file:{file_path}")

        logger.info(f"Extracted graph entities from {file_path}")

    async def _extract_node(self, node: ast.AST, file_path: str, parent_id: Optional[str] = None):
        """Extract entity from AST node.

        Args:
            node: AST node
            file_path: Source file path
            parent_id: Parent node ID (optional)
        """
        if isinstance(node, ast.ClassDef):
            # Add class node
            class_id = f"class:{file_path}:{node.name}"
            await self.graph.add_node(
                class_id,
                "class",
                {"name": node.name, "file": file_path, "line": node.lineno}
            )

            # Link parent -> class
            if parent_id:
                await self.graph.add_edge(parent_id, class_id, "defines")

            # Extract base classes (inheritance)
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_id = f"class:{file_path}:{base.id}"
                    await self.graph.add_edge(class_id, base_id, "inherits")

            # Extract class members
            for child in ast.iter_child_nodes(node):
                await self._extract_node(child, file_path, parent_id=class_id)

        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Add function node
            func_id = f"function:{file_path}:{node.name}"
            await self.graph.add_node(
                func_id,
                "function",
                {"name": node.name, "file": file_path, "line": node.lineno, "async": isinstance(node, ast.AsyncFunctionDef)}
            )

            # Link parent -> function
            if parent_id:
                await self.graph.add_edge(parent_id, func_id, "defines")

            # Extract function calls
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        called_id = f"function:{file_path}:{child.func.id}"
                        await self.graph.add_edge(func_id, called_id, "calls")
                    elif isinstance(child.func, ast.Attribute):
                        # Handle method calls (simplified)
                        if isinstance(child.func.value, ast.Name):
                            called_id = f"function:{file_path}:{child.func.attr}"
                            await self.graph.add_edge(func_id, called_id, "calls")

        elif isinstance(node, ast.Import):
            # Add import edges
            for alias in node.names:
                module = alias.name
                module_path = module.replace('.', '/') + '.py'
                await self.graph.add_edge(
                    f"file:{file_path}",
                    f"file:{module_path}",
                    "imports",
                    {"alias": alias.asname}
                )

        elif isinstance(node, ast.ImportFrom):
            # Add from-import edges
            if node.module:
                module_path = node.module.replace('.', '/') + '.py'
                for alias in node.names:
                    await self.graph.add_edge(
                        f"file:{file_path}",
                        f"file:{module_path}",
                        "imports",
                        {"name": alias.name, "alias": alias.asname}
                    )

    async def extract_from_directory(self, directory: str, pattern: str = "**/*.py"):
        """Extract entities from all Python files in a directory.

        Args:
            directory: Directory path
            pattern: File pattern (glob)
        """
        from pathlib import Path

        dir_path = Path(directory)
        files = list(dir_path.glob(pattern))

        logger.info(f"Extracting from {len(files)} files in {directory}")

        for file_path in files:
            try:
                content = file_path.read_text()
                await self.extract_from_file(str(file_path), content)
            except Exception as e:
                logger.error(f"Failed to extract from {file_path}: {e}")

        logger.info(f"Completed graph extraction from {len(files)} files")
