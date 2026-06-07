"""Abstract base class for graph stores used by mem0's optional graph memory layer.

A graph store holds (entity, relationship, entity) triples extracted from memory
text. Concrete providers (Apache AGE, Neo4j, Memgraph, Kuzu, ...) implement the
contract defined here. The orchestration layer in ``mem0.memory.main.Memory``
talks to any graph store through this interface, so adding a new backend is a
matter of writing a new subclass plus a factory entry.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class GraphStoreBase(ABC):
    """Provider-agnostic interface for a property graph store.

    All methods raise ``NotImplementedError`` in the base class — subclasses
    must implement every abstract method. Optional helpers have sensible
    default implementations that derived classes may override.
    """

    @abstractmethod
    def add_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create a node with ``label`` and ``properties``.

        Args:
            label: The node label (e.g. ``"Person"``, ``"Project"``).
            properties: A flat dict of node properties. Values must be JSON-
                serializable (str, int, float, bool, list, dict, None).

        Returns:
            The backend-assigned node identifier. The format is provider-
            specific (e.g. AGE returns the ``id()`` from Cypher, Neo4j returns
            its element id).
        """
        pass

    @abstractmethod
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a directed edge from ``source_id`` to ``target_id``.

        Args:
            source_id: The source node id (as returned by ``add_node``).
            target_id: The target node id.
            rel_type: The relationship type (e.g. ``"WORKS_ON"``).
            properties: Optional edge properties.

        Returns:
            The backend-assigned edge identifier.
        """
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return ``{label, properties}`` for a node, or ``None`` if missing."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for nodes whose properties match ``query``.

        Implementations decide how ``query`` is interpreted — full-text match
        on property values is a common baseline. ``filters`` follow the same
        conventions as vector store filters (``{"user_id": "alice"}``).

        Returns:
            A list of result dicts. Each dict must contain at least ``id``,
            ``label``, ``score`` and ``payload`` keys for symmetry with
            ``VectorStoreBase.search``.
        """
        pass

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        rel_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Traverse the graph from ``node_id`` up to ``depth`` hops.

        Args:
            node_id: Starting node.
            depth: Maximum hop count (1 = direct neighbors only).
            rel_types: If given, restrict traversal to these relationship
                types.

        Returns:
            A list of neighbor node dicts in the same shape as ``get_node``,
            with an additional ``distance`` field.
        """
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its incident edges."""
        pass

    @abstractmethod
    def delete_all(self) -> None:
        """Delete every node and edge in the graph (irreversible)."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Drop and recreate the underlying graph namespace/table."""
        pass

    # ------------------------------------------------------------------ helpers

    def add_triple(
        self,
        source: tuple,  # (label, properties)
        target: tuple,  # (label, properties)
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Convenience wrapper that creates two nodes and the edge between them.

        Returns a dict with ``source_id``, ``target_id`` and ``edge_id``.
        Subclasses may override for atomicity (e.g. single Cypher MERGE).
        """
        src_label, src_props = source
        tgt_label, tgt_props = target
        source_id = self.add_node(src_label, src_props)
        target_id = self.add_node(tgt_label, tgt_props)
        edge_id = self.add_edge(source_id, target_id, rel_type, properties)
        return {"source_id": source_id, "target_id": target_id, "edge_id": edge_id}
