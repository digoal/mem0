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
        raise ValueError(f"Invalid graph identifier: {name!r}. Must match [A-Za-z_][A-Za-z0-9_]*.")
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
            raise ValueError(f"Invalid graph_name: {graph_name!r}. Must match [A-Za-z_][A-Za-z0-9_]*.")
        self.graph_name = graph_name
        self._quoted_graph = _quote_ident(graph_name)
        # Track pool ownership so __del__ doesn't close an externally-managed
        # pool (e.g. one shared with PGVector).
        self._owns_pool = connection_pool is None

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
                connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
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
            # Put ag_catalog on the search_path so unqualified references to
            # ``ag_graph``, ``create_graph``, ``drop_graph`` resolve. AGE's
            # docs require this and the catalog is not on the default path.
            cur.execute('SET search_path = ag_catalog, "$user", public')
            # AGE 1.7 column is `name`, not `graphname`.
            cur.execute(sql.SQL("SELECT 1 FROM ag_graph WHERE name = {}").format(sql.Literal(self.graph_name)))
            if cur.fetchone() is None:
                cur.execute(sql.SQL("SELECT create_graph({})").format(sql.Literal(self.graph_name)))

    def _cypher(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        result_columns: int = 1,
        commit: bool = False,
    ) -> List[Any]:
        """Run a Cypher statement via AGE's ``cypher()`` SQL function.

        ``query`` is the Cypher body (without the outer ``$$ ... $$``). The
        ``params`` argument is reserved for future use — current callers all
        inline values into the Cypher text because AGE 1.7's Cypher parser
        does not expose ``agtype_map()`` to user queries.

        The ``cypher()`` function expects its query argument to be a
        dollar-quoted (``$$ ... $$``) cstring. We therefore inline the query
        body into the SQL; the graph name passes through a placeholder so
        the driver can quote it. ``query`` is built by us and only ever
        embeds values we formatted with :meth:`_cypher_literal`, so it
        cannot contain stray ``$$`` sequences.

        ``result_columns`` is the number of columns the Cypher ``RETURN``
        clause yields. ``cypher()`` returns ``record`` and PostgreSQL requires
        a column-definition list (``AS r(c1 agtype, c2 agtype, ...)``) when
        calling it from a plain ``SELECT *`` context.

        ``commit`` controls whether the underlying transaction is committed
        before the connection is returned to the pool. Write operations
        (``add_node``/``add_edge``/``delete_*``/``reset``) must commit so a
        follow-up read on a different pooled connection sees the change.
        """
        del params  # see docstring
        if "$$" in query:
            raise ValueError("Cypher query body must not contain '$$'")
        cols = ", ".join(f"c{i} agtype" for i in range(result_columns))
        # Compose with ``sql.SQL``/``sql.Literal`` so the graph name is
        # quoted by the driver and we never use psycopg2's ``%s`` placeholder
        # for it. That's important: the cypher body may contain literal ``%``
        # characters (e.g. a user-supplied string with ``100%``), which
        # collide with psycopg2's ``%`` parameter substitution and would
        # raise ``IndexError: tuple index out of range``.
        sql_text = sql.SQL("SELECT * FROM cypher({gname}, $${body}$$) AS r({cols})").format(
            gname=sql.Literal(self.graph_name),
            body=sql.SQL(query),
            cols=sql.SQL(cols),
        )
        with self._get_cursor(commit=commit) as cur:
            # Put ag_catalog on the search_path so the unqualified
            # ``cypher()`` function reference resolves.
            cur.execute('SET search_path = ag_catalog, "$user", public')
            cur.execute(sql_text)
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

    @staticmethod
    def _cypher_literal(value: Any) -> str:
        """Format a Python value as a Cypher literal.

        Property maps in Cypher are written inline as ``{k: v, ...}`` — there
        is no ``agtype_map()`` helper visible inside the Cypher parser, so we
        build the literal text directly. Strings are single-quoted with
        backslash and single-quote escaping; map keys are written as bare
        identifiers when safe and backtick-quoted otherwise. Values must be
        JSON-serializable (str/int/float/bool/None/list/dict).
        """
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace("'", "\\'")
            return f"'{escaped}'"
        if isinstance(value, list):
            return "[" + ", ".join(ApacheAGE._cypher_literal(v) for v in value) + "]"
        if isinstance(value, dict):
            parts = []
            for k, v in value.items():
                key_str = str(k)
                if _SAFE_IDENT.match(key_str):
                    key_rendered = key_str
                else:
                    # Backtick-quote unsafe keys; double any embedded backticks.
                    key_rendered = "`" + key_str.replace("`", "``") + "`"
                parts.append(f"{key_rendered}: {ApacheAGE._cypher_literal(v)}")
            return "{" + ", ".join(parts) + "}"
        raise TypeError(f"Cannot format {type(value).__name__} as a Cypher literal")

    # --------------------------------------------------------------- API

    def add_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create a single node. Returns the AGE ``id()`` of the new node."""
        if not _SAFE_IDENT.match(label):
            raise ValueError(f"Invalid node label: {label!r}")
        # Inline the property map into the CREATE clause — `agtype_map()` is
        # not visible from inside the Cypher parser in AGE 1.7, so we have to
        # write the literal `{k: v, ...}` text directly.
        if properties:
            props_literal = ApacheAGE._cypher_literal(properties)
            cypher = f"CREATE (n:{label} {props_literal}) RETURN id(n)"
        else:
            cypher = f"CREATE (n:{label}) RETURN id(n)"
        rows = self._cypher(cypher, commit=True)
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
        # Same inline-map approach as add_node — `SET r = {k: v, ...}` is
        # visible to the Cypher parser; `agtype_map()` is not.
        if props:
            props_literal = ApacheAGE._cypher_literal(props)
            set_clause = f"SET r = {props_literal} "
        else:
            set_clause = ""
        cypher = (
            "MATCH (a), (b) "
            f"WHERE id(a) = {int(source_id)} AND id(b) = {int(target_id)} "
            f"CREATE (a)-[r:{rel_type}]->(b) "
            f"{set_clause}"
            "RETURN id(r)"
        )
        rows = self._cypher(cypher, commit=True)
        if not rows:
            raise RuntimeError(f"Apache AGE could not create edge {source_id}-[{rel_type}]->{target_id}")
        return str(ApacheAGE._parse_agtype(rows[0][0]))

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        cypher = (
            f"MATCH (n) WHERE id(n) = {int(node_id)} RETURN {{id: id(n), label: label(n), properties: properties(n)}}"
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

        Implementation note: the obvious Cypher pattern

            ``MATCH (n) WHERE any(k IN keys(n) WHERE toString(n[k]) CONTAINS ...)``

        triggers "failed to construct the join relation" in AGE 1.7.0 because
        ``keys(n)`` produces an untyped list. We fall back to a SQL scan of
        the graph's global vertex table (``<graph>._ag_label_vertex``) — the
        same data, but reachable through the relational planner.
        """
        if limit <= 0:
            return []
        # ``ILIKE`` is case-insensitive. Escape LIKE metacharacters so the
        # query behaves as a plain substring match.
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like_pattern = f"%{escaped}%"
        # ``self.graph_name`` is validated against ``_SAFE_IDENT`` and
        # ``self._quoted_graph`` quotes it for SQL, so it's safe to interpolate
        # the schema name here. The like-pattern and limit use placeholders.
        query_sql = sql.SQL(
            "SELECT v.id::text::bigint AS id, "
            "       lbl.name AS label, "
            "       v.properties AS properties "
            "FROM {graph}._ag_label_vertex v "
            "LEFT JOIN ag_catalog.ag_label lbl "
            "  ON lbl.id = (v.id::text::bigint >> 48)::int "
            " AND lbl.graph = (SELECT graphid FROM ag_catalog.ag_graph "
            "                  WHERE name = {gname}) "
            " AND lbl.kind = 'v' "
            "WHERE v.properties::text ILIKE %s "
            "LIMIT {lim}"
        ).format(
            graph=sql.SQL(self._quoted_graph),
            gname=sql.Literal(self.graph_name),
            lim=sql.Literal(int(limit)),
        )
        with self._get_cursor() as cur:
            cur.execute(query_sql, (like_pattern,))
            rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            node_id, label, raw_props = row
            properties = ApacheAGE._parse_agtype(raw_props)
            if not isinstance(properties, dict):
                properties = {}
            if filters and not _payload_matches(properties, filters):
                continue
            results.append(
                {
                    "id": str(node_id),
                    "label": label,
                    "score": 1.0,
                    "payload": properties,
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
        # `size(r)` works for both single-hop and variable-length patterns.
        # `length(r)` errors with "must resolve to a scalar" on multi-hop
        # patterns in AGE 1.7.
        cypher = (
            f"MATCH (n)-[r{rel_clause}*1..{int(depth)}]-(m) "
            f"WHERE id(n) = {int(node_id)} AND id(m) <> {int(node_id)} "
            "RETURN {id: id(m), label: label(m), properties: properties(m), "
            "distance: size(r)}"
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
        self._cypher(cypher, commit=True)

    def delete_all(self) -> None:
        cypher = "MATCH (n) DETACH DELETE n"
        self._cypher(cypher, commit=True)

    def reset(self) -> None:
        """Drop and recreate the graph namespace."""
        with self._get_cursor(commit=True) as cur:
            cur.execute("LOAD 'age'")
            cur.execute(sql.SQL("SELECT drop_graph({}, true)").format(sql.Literal(self.graph_name)))
        self._ensure_graph_exists()

    def __del__(self) -> None:
        # Only close pools we created. A shared pool (e.g. one provided by
        # PGVector) may still be in use by other stores.
        if not getattr(self, "_owns_pool", True):
            return
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
