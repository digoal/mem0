"""Apache AGE (PostgreSQL graph extension) implementation of :class:`GraphStoreBase`.

Apache AGE (https://age.apache.org/) is a PostgreSQL extension that adds
property-graph capabilities using a Cypher-like query language. Running on
top of PostgreSQL means mem0 deployments that already use ``pgvector`` for
vector storage can add graph memory on the **same database instance** — and,
when the ``connection_pool`` argument is supplied, the **same connection
pool**, which is exactly the "one PostgreSQL connection for both vector and
graph" use case.

Cypher queries are executed via the ``cypher()`` SQL function::

    SELECT * FROM cypher('graph_name', $$ MATCH (n:Person) RETURN n $$)
    AS r(result agtype);

The ``agtype`` return value is a JSON-like type. We fetch it as text and
parse with ``json.loads``.
"""

import json
import logging
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# Try psycopg3 first, then fall back to psycopg2 — same pattern as pgvector so
# the two backends stay in lockstep.
try:
    from psycopg import sql
    from psycopg_pool import ConnectionPool as _ConnectionPool

    PSYCOPG_VERSION = 3
except ImportError:  # pragma: no cover - exercised only when psycopg3 is absent
    try:
        from psycopg2 import sql  # type: ignore
        from psycopg2.pool import ThreadedConnectionPool as _ConnectionPool  # type: ignore

        PSYCOPG_VERSION = 2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Apache AGE backend requires psycopg[pool] (v3) or psycopg2. "
            "Install with `pip install psycopg[binary,pool]` or "
            "`pip install psycopg2-binary`."
        ) from exc

from mem0.graphs.base import GraphStoreBase

logger = logging.getLogger(__name__)


_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_ident(name: str) -> str:
    """Validate and quote a SQL/Cypher identifier.

    AGE graph names live in the SQL identifier namespace, so we restrict the
    set of legal characters to avoid SQL injection from configuration input.
    """
    if not _SAFE_IDENT.match(name):
        raise ValueError(
            f"Invalid graph identifier: {name!r}. "
            "Must match [A-Za-z_][A-Za-z0-9_]*."
        )
    return f'"{name}"'


class GraphData(BaseModel):
    """A node or edge returned from the graph store.

    Mirrors the shape of :class:`mem0.vector_stores.pgvector.OutputData` so
    callers can treat search results uniformly.
    """

    id: Optional[str] = None
    label: Optional[str] = None
    score: Optional[float] = None
    payload: Optional[dict] = None


class ApacheAGE(GraphStoreBase):
    """Apache AGE backed graph store.

    Args:
        dbname: Database name.
        user: PostgreSQL username.
        password: PostgreSQL password.
        host: PostgreSQL host.
        port: PostgreSQL port.
        graph_name: Name of the AGE graph (created on first use). Must match
            ``^[A-Za-z_][A-Za-z0-9_]*$``.
        minconn / maxconn: Connection pool sizing (ignored when
            ``connection_string`` or ``connection_pool`` is supplied).
        sslmode: Optional PostgreSQL SSL mode.
        connection_string: Full PostgreSQL DSN. Overrides the individual
            ``user``/``password``/``host``/``port``/``dbname`` arguments.
        connection_pool: A pre-built psycopg2/psycopg connection pool. Pass
            the same pool used by ``PGVector`` to share connections between
            the vector and graph stores.
    """

    def __init__(
        self,
        dbname: str = "postgres",
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        graph_name: str = "mem0_graph",
        minconn: int = 1,
        maxconn: int = 5,
        sslmode: Optional[str] = None,
        connection_string: Optional[str] = None,
        connection_pool: Optional[Any] = None,
    ):
        if not _SAFE_IDENT.match(graph_name):
            raise ValueError(
                f"Invalid graph_name: {graph_name!r}. "
                "Must match [A-Za-z_][A-Za-z0-9_]*."
            )
        self.graph_name = graph_name
        self._quoted_graph = _quote_ident(graph_name)

        if connection_pool is not None:
            self.connection_pool = connection_pool
        else:
            if connection_string is None:
                if not (user and password) or not (host and port):
                    raise ValueError(
                        "ApacheAGE requires either `connection_string`, "
                        "`connection_pool`, or the (user, password, host, "
                        "port) tuple."
                    )
                connection_string = (
                    f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
                )
            if sslmode:
                # Append/replace sslmode without rewriting the URL ourselves —
                # psycopg accepts it as a query parameter.
                sep = "&" if "?" in connection_string else "?"
                connection_string = f"{connection_string}{sep}sslmode={sslmode}"

            if PSYCOPG_VERSION == 3:
                self.connection_pool = _ConnectionPool(
                    conninfo=connection_string,
                    min_size=minconn,
                    max_size=maxconn,
                    open=True,
                )
            else:
                self.connection_pool = _ConnectionPool(
                    minconn=minconn,
                    maxconn=maxconn,
                    dsn=connection_string,
                )

        self._ensure_graph_exists()

    # --------------------------------------------------------------- pool

    @contextmanager
    def _get_cursor(self, commit: bool = False):
        """Yield a cursor, returning the connection to the pool on exit."""
        if PSYCOPG_VERSION == 3:
            with self.connection_pool.connection() as conn:
                with conn.cursor() as cur:
                    try:
                        yield cur
                        if commit:
                            conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
        else:
            conn = self.connection_pool.getconn()
            cur = conn.cursor()
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()
                self.connection_pool.putconn(conn)

    # --------------------------------------------------------------- setup

    def _ensure_graph_exists(self) -> None:
        """Create the ``age`` extension and the graph if they don't exist."""
        with self._get_cursor(commit=True) as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS age")
            cur.execute("LOAD 'age'")
            # AGE's CREATE GRAPH is idempotent via a SELECT on ag_graph.
            cur.execute(
                sql.SQL("SELECT 1 FROM ag_graph WHERE graphname = %s"),
                (self.graph_name,),
            )
            if cur.fetchone() is None:
                cur.execute(
                    sql.SQL("SELECT create_graph({})").format(sql.Literal(self.graph_name))
                )

    def _cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Run a Cypher statement via AGE's ``cypher()`` SQL function.

        ``query`` is the Cypher body (without the outer ``$$ ... $$``). The
        ``params`` dict is passed as ``agtype`` via ``age_agtype_in``.
        """
        with self._get_cursor() as cur:
            # age_agtype_in takes a JSON string. Build it carefully.
            if params:
                # Each value must be wrapped as an agtype literal in the Cypher
                # text. For simplicity we use a JSON map and the ::agtype
                # cast, which AGE accepts for parameterized values via
                # agtype_map.
                params_json = json.dumps(_stringify_keys(params))
                cur.execute(
                    "SELECT * FROM cypher(%s, %s, agtype_map(%s))",
                    (self.graph_name, query, params_json),
                )
            else:
                cur.execute(
                    "SELECT * FROM cypher(%s, %s)",
                    (self.graph_name, query),
                )
            return cur.fetchall()

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _parse_agtype(value: Any) -> Any:
        """Convert agtype text (``"{}"``, ``"[]"``, ``"\"x\""`` ...) to Python.

        AGE returns agtype as a textual representation of JSON for object/map
        and array types, and as bare text for scalars. Anything that fails to
        parse is returned as the raw string.
        """
        if value is None:
            return None
        if isinstance(value, (dict, list, int, float, bool)):
            return value
        if not isinstance(value, str):
            return value
        s = value.strip()
        if not s or s in ("null", "true", "false") or s.lstrip("-").isdigit():
            try:
                return json.loads(s) if s in ("null", "true", "false") else int(s) if s.lstrip("-").isdigit() else s
            except (ValueError, TypeError):
                return s
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return s
        # Bare scalar — strip surrounding quotes if present.
        if s.startswith('"') and s.endswith('"'):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return s
        return s

    @staticmethod
    def _node_from_agtype(raw: Any) -> Optional[Dict[str, Any]]:
        """Parse a node from an agtype ``{id, label, properties}`` map."""
        parsed = ApacheAGE._parse_agtype(raw)
        if not isinstance(parsed, dict):
            return None
        return {
            "id": str(parsed.get("id")) if parsed.get("id") is not None else None,
            "label": parsed.get("label"),
            "properties": parsed.get("properties") or {},
        }

    # --------------------------------------------------------------- API

    def add_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create a single node. Returns the AGE ``id()`` of the new node."""
        if not _SAFE_IDENT.match(label):
            raise ValueError(f"Invalid node label: {label!r}")
        # Cypher parameter map: pass properties as a single map.
        props_json = json.dumps(_stringify_keys(properties))
        cypher = (
            f"CREATE (n:{label}) SET n = agtype_map({json.dumps(props_json)}) "
            f"RETURN id(n)"
        )
        rows = self._cypher(cypher, params=None)
        if not rows:
            raise RuntimeError("Apache AGE did not return a node id")
        # First column of the first row holds the id.
        return str(ApacheAGE._parse_agtype(rows[0][0]))

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a directed edge between two existing nodes."""
        if not _SAFE_IDENT.match(rel_type):
            raise ValueError(f"Invalid relationship type: {rel_type!r}")
        props = properties or {}
        props_json = json.dumps(_stringify_keys(props))
        cypher = (
            "MATCH (a), (b) "
            f"WHERE id(a) = {int(source_id)} AND id(b) = {int(target_id)} "
            f"CREATE (a)-[r:{rel_type}]->(b) "
            f"SET r = agtype_map({json.dumps(props_json)}) "
            "RETURN id(r)"
        )
        rows = self._cypher(cypher, params=None)
        if not rows:
            raise RuntimeError(
                f"Apache AGE could not create edge {source_id}-[{rel_type}]->{target_id}"
            )
        return str(ApacheAGE._parse_agtype(rows[0][0]))

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        cypher = (
            f"MATCH (n) WHERE id(n) = {int(node_id)} "
            "RETURN {id: id(n), label: label(n), properties: properties(n)}"
        )
        rows = self._cypher(cypher, params=None)
        if not rows:
            return None
        return ApacheAGE._node_from_agtype(rows[0][0])

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Case-insensitive substring match across all node property values.

        For richer search semantics, override in a subclass or post-filter
        the results. ``filters`` are passed through as a payload predicate:
        every property on the node must equal the corresponding filter value.
        """
        if limit <= 0:
            return []
        # Escape backslash and single-quote for Cypher string literal.
        q = query.replace("\\", "\\\\").replace("'", "\\'")
        cypher = (
            "MATCH (n) "
            f"WHERE any(k IN keys(n) WHERE toString(n[k]) CONTAINS '{q}') "
            "RETURN {id: id(n), label: label(n), properties: properties(n)} "
            f"LIMIT {int(limit)}"
        )
        rows = self._cypher(cypher, params=None)
        results: List[Dict[str, Any]] = []
        for row in rows:
            parsed = ApacheAGE._node_from_agtype(row[0])
            if parsed is None:
                continue
            if filters and not _payload_matches(parsed.get("properties") or {}, filters):
                continue
            results.append(
                {
                    "id": parsed["id"],
                    "label": parsed["label"],
                    "score": 1.0,
                    "payload": parsed["properties"],
                }
            )
        return results

    def get_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        rel_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if depth < 1:
            return []
        rel_clause = ""
        if rel_types:
            cleaned = [r for r in rel_types if _SAFE_IDENT.match(r)]
            if not cleaned:
                return []
            rel_clause = ":" + "|".join(cleaned)
        cypher = (
            f"MATCH (n)-[r{rel_clause}*1..{int(depth)}]-(m) "
            f"WHERE id(n) = {int(node_id)} AND id(m) <> {int(node_id)} "
            "RETURN {id: id(m), label: label(m), properties: properties(m), "
            "distance: length(r)}"
        )
        rows = self._cypher(cypher, params=None)
        results: List[Dict[str, Any]] = []
        for row in rows:
            parsed = ApacheAGE._parse_agtype(row[0])
            if not isinstance(parsed, dict):
                continue
            results.append(
                {
                    "id": str(parsed.get("id")),
                    "label": parsed.get("label"),
                    "properties": parsed.get("properties") or {},
                    "distance": int(parsed.get("distance") or 0),
                }
            )
        return results

    def delete_node(self, node_id: str) -> None:
        cypher = f"MATCH (n) WHERE id(n) = {int(node_id)} DETACH DELETE n"
        self._cypher(cypher, params=None)

    def delete_all(self) -> None:
        cypher = "MATCH (n) DETACH DELETE n"
        self._cypher(cypher, params=None)

    def reset(self) -> None:
        """Drop and recreate the graph namespace."""
        with self._get_cursor(commit=True) as cur:
            cur.execute("LOAD 'age'")
            cur.execute(
                sql.SQL("SELECT drop_graph({}, true)").format(sql.Literal(self.graph_name))
            )
        self._ensure_graph_exists()

    def __del__(self) -> None:
        try:
            if PSYCOPG_VERSION == 3:
                self.connection_pool.close()
            else:
                self.connection_pool.closeall()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


def _stringify_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """AGE / agtype requires string keys; coerce recursively for safety."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        out[str(k)] = v
    return out


def _payload_matches(payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for k, v in filters.items():
        if payload.get(k) != v:
            return False
    return True
