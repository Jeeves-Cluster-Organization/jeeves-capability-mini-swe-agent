"""Graph Service (L5) - Code Entity Relationship Graph."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Graph node representing a code entity."""

    node_id: str
    node_type: str  # file, class, function, variable
    metadata: Dict[str, Any]


@dataclass
class GraphEdge:
    """Graph edge representing a relationship."""

    source_id: str
    target_id: str
    edge_type: str  # imports, calls, inherits, defines
    metadata: Dict[str, Any]


class GraphService:
    """Service for managing code entity relationship graph (L5)."""

    def __init__(self, db_client):
        """Initialize graph service.

        Args:
            db_client: Database client (asyncpg connection or pool)
        """
        self.db = db_client

    async def add_node(self, node_id: str, node_type: str, metadata: Dict[str, Any]):
        """Add a node to the graph.

        Args:
            node_id: Unique node identifier
            node_type: Type of entity (file, class, function, etc.)
            metadata: Node metadata
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            await conn.execute("""
                INSERT INTO graph_nodes (node_id, node_type, metadata)
                VALUES ($1, $2, $3)
                ON CONFLICT (node_id) DO UPDATE SET
                    node_type = EXCLUDED.node_type,
                    metadata = EXCLUDED.metadata
            """, node_id, node_type, metadata)

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def add_edge(self, source_id: str, target_id: str, edge_type: str,
                       metadata: Optional[Dict[str, Any]] = None):
        """Add an edge to the graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            metadata: Edge metadata
        """
        if metadata is None:
            metadata = {}

        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            await conn.execute("""
                INSERT INTO graph_edges (source_id, target_id, edge_type, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                    metadata = EXCLUDED.metadata
            """, source_id, target_id, edge_type, metadata)

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID.

        Args:
            node_id: Node identifier

        Returns:
            GraphNode if found, None otherwise
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            row = await conn.fetchrow(
                "SELECT * FROM graph_nodes WHERE node_id = $1",
                node_id
            )

            if not row:
                return None

            return GraphNode(
                node_id=row['node_id'],
                node_type=row['node_type'],
                metadata=row['metadata'] or {},
            )

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def query_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "outgoing",
        max_depth: int = 1
    ) -> List[GraphNode]:
        """Query neighboring nodes.

        Args:
            node_id: Starting node
            edge_type: Filter by edge type (optional)
            direction: 'outgoing', 'incoming', or 'both'
            max_depth: Maximum traversal depth

        Returns:
            List of neighboring nodes
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            if direction == "outgoing":
                query = """
                    SELECT DISTINCT n.node_id, n.node_type, n.metadata
                    FROM graph_edges e
                    JOIN graph_nodes n ON e.target_id = n.node_id
                    WHERE e.source_id = $1
                """
                params = [node_id]

                if edge_type:
                    query += " AND e.edge_type = $2"
                    params.append(edge_type)

            elif direction == "incoming":
                query = """
                    SELECT DISTINCT n.node_id, n.node_type, n.metadata
                    FROM graph_edges e
                    JOIN graph_nodes n ON e.source_id = n.node_id
                    WHERE e.target_id = $1
                """
                params = [node_id]

                if edge_type:
                    query += " AND e.edge_type = $2"
                    params.append(edge_type)

            else:  # both
                query = """
                    SELECT DISTINCT n.node_id, n.node_type, n.metadata
                    FROM graph_edges e
                    JOIN graph_nodes n ON (e.source_id = n.node_id OR e.target_id = n.node_id)
                    WHERE (e.source_id = $1 OR e.target_id = $1) AND n.node_id != $1
                """
                params = [node_id]

                if edge_type:
                    query += " AND e.edge_type = $2"
                    params.append(edge_type)

            rows = await conn.fetch(query, *params)

            return [
                GraphNode(
                    node_id=row['node_id'],
                    node_type=row['node_type'],
                    metadata=row['metadata'] or {},
                )
                for row in rows
            ]

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def find_cycles(self, node_id: str, max_depth: int = 5) -> List[List[str]]:
        """Find circular dependencies from a node.

        Args:
            node_id: Starting node
            max_depth: Maximum cycle depth

        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        # Simple DFS-based cycle detection
        cycles = []
        visited = set()
        path = []

        async def dfs(current_id: str, depth: int):
            if depth > max_depth:
                return

            if current_id in path:
                # Found a cycle
                cycle_start = path.index(current_id)
                cycle = path[cycle_start:] + [current_id]
                cycles.append(cycle)
                return

            if current_id in visited:
                return

            visited.add(current_id)
            path.append(current_id)

            # Get outgoing edges
            neighbors = await self.query_neighbors(current_id, direction="outgoing")

            for neighbor in neighbors:
                await dfs(neighbor.node_id, depth + 1)

            path.pop()

        await dfs(node_id, 0)
        return cycles

    async def delete_node(self, node_id: str):
        """Delete a node and its edges.

        Args:
            node_id: Node to delete
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            # Edges are deleted automatically via CASCADE
            await conn.execute(
                "DELETE FROM graph_nodes WHERE node_id = $1",
                node_id
            )

            logger.info(f"Deleted node: {node_id}")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def clear_graph(self):
        """Clear the entire graph."""
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            await conn.execute("DELETE FROM graph_edges")
            await conn.execute("DELETE FROM graph_nodes")

            logger.info("Cleared graph")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)
