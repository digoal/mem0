"""Unit tests for the Apache AGE graph store backend.

Covers:
- Provider construction (individual params, connection_string, shared pool)
- Cypher query plumbing (mocked at the cursor level)
- Identifier / argument validation
- Pydantic config validation
- Memory.add_to_graph / search_graph integration with a mocked graph store

Integration tests against a live PostgreSQL+AGE instance are not included —
they require ``apt.postgresql.org`` packages that are not part of the default
test environment. The unit tests below exercise every code path that the
integration tests would touch.
"""

import unittest
from unittest.mock import MagicMock, patch

from mem0.configs.base import GraphStoreConfig, MemoryConfig
from mem0.configs.graphs.apache_age import ApacheAGEConfig
from mem0.graphs.apache_age import (
    ApacheAGE,
    GraphData,
    _quote_ident,
    _SAFE_IDENT,
)
from mem0.graphs.base import GraphStoreBase
from mem0.utils.factory import GraphStoreFactory


def _cypher_body_of(mock_cursor):
    """Return the Cypher text passed to the ``cypher()`` call.

    The cypher body is inlined inside ``$$ ... $$`` in the SQL string —
    extract it from the SQL text rather than the params tuple (which now
    only holds the graph name).
    """
    import re

    for call in mock_cursor.execute.call_args_list:
        sql = str(call.args[0]) if call.args else str(call.kwargs.get("query", ""))
        if "cypher(" not in sql:
            continue
        m = re.search(r"\$\$\s*(.*?)\s*\$\$", sql, re.DOTALL)
        if m is None:
            continue
        return m.group(1)
    raise AssertionError("No cypher() call found in mock_cursor.execute history")


class TestIdentifierValidation(unittest.TestCase):
    """Identifier safety: graph names, labels, rel types go into Cypher raw strings."""

    def test_safe_ident_accepts_normal(self):
        for name in ["mem0", "Mem0_Graph", "graph_42", "_underscore"]:
            self.assertTrue(bool(_SAFE_IDENT.match(name)))

    def test_safe_ident_rejects_injection(self):
        for name in [
            "1graph",  # leading digit
            "graph; DROP",  # SQL injection
            "graph-name",  # hyphen
            "graph.name",  # dot
            "graph name",  # whitespace
            "",
        ]:
            self.assertFalse(bool(_SAFE_IDENT.match(name)), f"should reject: {name!r}")

    def test_quote_ident_raises_for_invalid(self):
        with self.assertRaises(ValueError):
            _quote_ident("graph; DROP TABLE users")

    def test_quote_ident_quotes_valid(self):
        self.assertEqual(_quote_ident("mem0_graph"), '"mem0_graph"')


class TestParseAgtype(unittest.TestCase):
    """The _parse_agtype helper turns AGE's text-shaped results into Python values."""

    def test_parses_json_object(self):
        self.assertEqual(ApacheAGE._parse_agtype('{"id": 1, "label": "Person"}'), {"id": 1, "label": "Person"})

    def test_parses_json_array(self):
        self.assertEqual(ApacheAGE._parse_agtype('["a", "b"]'), ["a", "b"])

    def test_parses_quoted_string(self):
        self.assertEqual(ApacheAGE._parse_agtype('"hello"'), "hello")

    def test_parses_integer(self):
        self.assertEqual(ApacheAGE._parse_agtype("42"), 42)

    def test_returns_raw_when_unparseable(self):
        self.assertEqual(ApacheAGE._parse_agtype("not_json_at_all"), "not_json_at_all")

    def test_handles_none(self):
        self.assertIsNone(ApacheAGE._parse_agtype(None))

    def test_passes_through_native_types(self):
        self.assertEqual(ApacheAGE._parse_agtype({"id": 1}), {"id": 1})
        self.assertEqual(ApacheAGE._parse_agtype([1, 2]), [1, 2])


class TestApacheAGEInit(unittest.TestCase):
    """Construction and connection-pool wiring."""

    def setUp(self):
        self.mock_pool = MagicMock()
        self.mock_cursor = MagicMock()
        # psycopg3 cursors are themselves context managers (`with conn.cursor()
        # as cur:`). Make the mock behave the same way so the `cur` binding
        # in the production code points at our mock cursor.
        self.mock_cursor.__enter__.return_value = self.mock_cursor
        self.mock_cursor.__exit__.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = self.mock_cursor
        self.mock_pool.connection.return_value.__enter__.return_value = mock_conn
        self.mock_pool.getconn.return_value = mock_conn
        self.mock_cursor.fetchone.return_value = None

    def _make_shared_pool(self):
        """Build a fresh shared pool that yields the same cursor for testing."""
        shared_pool = MagicMock()
        shared_conn = MagicMock()
        shared_conn.cursor.return_value = self.mock_cursor
        shared_pool.connection.return_value.__enter__.return_value = shared_conn
        return shared_pool

    @patch("mem0.graphs.apache_age.PSYCOPG_VERSION", 3)
    @patch("mem0.graphs.apache_age._ConnectionPool")
    def test_init_with_individual_params_psycopg3(self, mock_conn_pool):
        mock_conn_pool.return_value = self.mock_pool

        ApacheAGE(
            dbname="db",
            user="u",
            password="p",
            host="localhost",
            port=5432,
            graph_name="mem0_graph",
        )

        mock_conn_pool.assert_called_once()
        kwargs = mock_conn_pool.call_args.kwargs
        self.assertEqual(kwargs["conninfo"], "postgresql://u:p@localhost:5432/db")
        self.assertEqual(kwargs["min_size"], 1)
        self.assertEqual(kwargs["max_size"], 5)

    @patch("mem0.graphs.apache_age.PSYCOPG_VERSION", 2)
    @patch("mem0.graphs.apache_age._ConnectionPool")
    def test_init_with_connection_string(self, mock_conn_pool):
        mock_conn_pool.return_value = self.mock_pool

        ApacheAGE(
            dbname="ignored",
            connection_string="postgresql://x:y@h:5432/d",
            graph_name="mem0_graph",
        )

        kwargs = mock_conn_pool.call_args.kwargs
        # psycopg2 uses positional dsn=
        self.assertEqual(kwargs["dsn"], "postgresql://x:y@h:5432/d")

    def test_init_with_shared_connection_pool(self):
        """The killer feature: pass the same pool as PGVector to share connections."""
        shared_pool = self._make_shared_pool()

        with patch("mem0.graphs.apache_age.PSYCOPG_VERSION", 3):
            store = ApacheAGE(connection_pool=shared_pool, graph_name="mem0_graph")

        # The store uses the *same* pool object the user passed in.
        self.assertIs(store.connection_pool, shared_pool)
        # And it did issue the create-graph SQL — proving it works against a
        # pool that came from elsewhere (e.g. PGVector's pool).
        self.mock_cursor.execute.assert_any_call("CREATE EXTENSION IF NOT EXISTS age")
        self.mock_cursor.execute.assert_any_call("LOAD 'age'")

    def test_init_rejects_invalid_graph_name(self):
        with self.assertRaises(ValueError):
            ApacheAGE(connection_pool=MagicMock(), graph_name="bad name")

    def test_init_requires_connection_info(self):
        with self.assertRaises(ValueError):
            # No pool, no DSN, no user/password/host/port.
            ApacheAGE(graph_name="mem0_graph")


class TestApacheAGEQueries(unittest.TestCase):
    """End-to-end behavior with mocked cursor."""

    def setUp(self):
        self.mock_pool = MagicMock()
        self.mock_cursor = MagicMock()
        # The context manager returned by `_get_cursor` should yield our cursor.
        cursor_cm = MagicMock()
        cursor_cm.__enter__.return_value = self.mock_cursor
        cursor_cm.__exit__.return_value = None

        with patch("mem0.graphs.apache_age.PSYCOPG_VERSION", 3):
            with patch("mem0.graphs.apache_age._ConnectionPool", return_value=self.mock_pool):
                self.store = ApacheAGE(
                    connection_pool=self.mock_pool,
                    graph_name="mem0_graph",
                )
        # Replace _get_cursor with a context manager that returns our cursor.
        self.store._get_cursor = MagicMock(return_value=cursor_cm)
        # Reset execute call history after init.
        self.mock_cursor.reset_mock()

    def test_add_node_runs_cypher_create(self):
        self.mock_cursor.fetchall.return_value = [(42,)]

        node_id = self.store.add_node("Person", {"name": "alice", "age": 30})

        self.assertEqual(node_id, "42")
        cypher_body = _cypher_body_of(self.mock_cursor)
        self.assertIn("CREATE (n:Person", cypher_body)
        self.assertIn("alice", cypher_body)
        self.assertIn("age", cypher_body)
        self.assertIn("RETURN id(n)", cypher_body)

    def test_add_node_rejects_invalid_label(self):
        with self.assertRaises(ValueError):
            self.store.add_node("Person; DROP TABLE", {})

    def test_add_edge_runs_cypher_match_create(self):
        self.mock_cursor.fetchall.return_value = [(99,)]

        edge_id = self.store.add_edge("1", "2", "KNOWS", {"since": 2020})

        self.assertEqual(edge_id, "99")
        cypher_body = _cypher_body_of(self.mock_cursor)
        self.assertIn("MATCH (a), (b)", cypher_body)
        self.assertIn("id(a) = 1", cypher_body)
        self.assertIn("id(b) = 2", cypher_body)
        self.assertIn("CREATE (a)-[r:KNOWS]->(b)", cypher_body)
        # The properties are inlined as a Cypher map literal.
        self.assertIn("since", cypher_body)
        self.assertIn("2020", cypher_body)

    def test_add_edge_rejects_invalid_rel_type(self):
        with self.assertRaises(ValueError):
            self.store.add_edge("1", "2", "DROP TABLE x", {})

    def test_get_node_returns_parsed(self):
        self.mock_cursor.fetchall.return_value = [('{"id": 7, "label": "Person", "properties": {"name": "bob"}}',)]

        result = self.store.get_node("7")

        self.assertEqual(
            result,
            {
                "id": "7",
                "label": "Person",
                "properties": {"name": "bob"},
            },
        )

    def test_get_node_returns_none_when_missing(self):
        self.mock_cursor.fetchall.return_value = []
        self.assertIsNone(self.store.get_node("999"))

    def test_search_filters_payload(self):
        # SQL fallback: 3 columns (id, label, raw agtype properties).
        self.mock_cursor.fetchall.return_value = [
            (1, "Person", '{"name": "alice", "user_id": "u1"}'),
            (2, "Person", '{"name": "bob",   "user_id": "u2"}'),
        ]

        results = self.store.search("alice", limit=10, filters={"user_id": "u1"})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "1")
        self.assertEqual(results[0]["label"], "Person")
        self.assertEqual(results[0]["payload"], {"name": "alice", "user_id": "u1"})

    def test_get_neighbors_with_rel_type_filter(self):
        self.mock_cursor.fetchall.return_value = [
            ('{"id": 5, "label": "Project", "properties": {"name": "mem0"}, "distance": 1}',)
        ]

        results = self.store.get_neighbors("1", depth=2, rel_types=["WORKS_ON", "OWNS"])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "5")
        self.assertEqual(results[0]["distance"], 1)
        cypher_body = _cypher_body_of(self.mock_cursor)
        self.assertIn(":WORKS_ON|OWNS", cypher_body)
        # `size()` not `length()` for variable-length patterns.
        self.assertIn("size(r)", cypher_body)

    def test_get_neighbors_rejects_invalid_rel_types(self):
        # All rel_types are invalid → should yield nothing, not run a query.
        results = self.store.get_neighbors("1", rel_types=["bad type"])
        self.assertEqual(results, [])

    def test_delete_node_runs_detach_delete(self):
        self.mock_cursor.fetchall.return_value = []
        self.store.delete_node("42")

        cypher_body = _cypher_body_of(self.mock_cursor)
        self.assertIn("MATCH (n) WHERE id(n) = 42 DETACH DELETE n", cypher_body)

    def test_delete_all_runs_unscoped_detach(self):
        self.mock_cursor.fetchall.return_value = []
        self.store.delete_all()
        cypher_body = _cypher_body_of(self.mock_cursor)
        self.assertIn("MATCH (n) DETACH DELETE n", cypher_body)

    def test_reset_drops_and_recreates_graph(self):
        # First call (drop_graph) returns success, _ensure_graph_exists then
        # re-runs CREATE EXTENSION / LOAD.
        self.mock_cursor.reset_mock()
        self.mock_cursor.fetchone.return_value = None
        self.store.reset()

        executed = [c.args[0] for c in self.mock_cursor.execute.call_args_list if c.args]
        self.assertTrue(any("drop_graph" in str(s) for s in executed))
        self.assertIn("CREATE EXTENSION IF NOT EXISTS age", executed)


class TestApacheAGEConfig(unittest.TestCase):
    """Pydantic config matches the same strictness as PGVectorConfig."""

    def test_minimal_config_passes(self):
        cfg = ApacheAGEConfig(user="u", password="p", host="h", port=5432)
        self.assertEqual(cfg.graph_name, "mem0_graph")
        self.assertEqual(cfg.minconn, 1)
        self.assertEqual(cfg.maxconn, 5)

    def test_connection_string_skips_param_validation(self):
        cfg = ApacheAGEConfig(connection_string="postgresql://x@h:1/d")
        self.assertEqual(cfg.connection_string, "postgresql://x@h:1/d")

    def test_connection_pool_skips_param_validation(self):
        cfg = ApacheAGEConfig(connection_pool=MagicMock())
        self.assertIsNotNone(cfg.connection_pool)

    def test_missing_credentials_raises(self):
        with self.assertRaises(ValueError):
            ApacheAGEConfig()  # nothing provided

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ApacheAGEConfig(user="u", password="p", host="h", port=5432, made_up_field="x")
        self.assertIn("made_up_field", str(ctx.exception))


class TestMemoryConfigIntegration(unittest.TestCase):
    """``MemoryConfig.graph`` wires up correctly with the default factory."""

    def test_default_memory_config_has_no_graph(self):
        cfg = MemoryConfig()
        self.assertIsNone(cfg.graph)

    def test_graph_config_propagates_provider(self):
        cfg = MemoryConfig(
            graph={
                "provider": "apache_age",
                "config": {
                    "user": "u",
                    "password": "p",
                    "host": "h",
                    "port": 5432,
                },
            }
        )
        self.assertIsInstance(cfg.graph, GraphStoreConfig)
        self.assertEqual(cfg.graph.provider, "apache_age")
        self.assertEqual(cfg.graph.threshold, 0.7)  # default

    def test_factory_resolves_apache_age(self):
        # Sanity: factory has the provider; doesn't need a real DB to confirm
        # the entry exists.
        self.assertIn("apache_age", GraphStoreFactory.provider_to_class)


class TestMemoryGraphWiring(unittest.TestCase):
    """``Memory.add_to_graph`` and ``Memory.search_graph`` delegate correctly.

    Uses a fake graph store so we don't need a live DB to verify the wiring.
    """

    def _make_memory_with_fake_graph(self):
        from mem0.memory.main import Memory

        fake_store = MagicMock(spec=GraphStoreBase)
        # Pretend the factory hands back our fake.
        with patch("mem0.memory.main.GraphStoreFactory.create", return_value=fake_store) as factory_mock:
            cfg = MemoryConfig(
                graph={
                    "provider": "apache_age",
                    "config": {
                        "user": "u",
                        "password": "p",
                        "host": "h",
                        "port": 5432,
                    },
                }
            )
            mem = Memory.__new__(Memory)
            mem.config = cfg
            mem._graph_store = None  # force lazy init
            # Trigger lazy init
            _ = mem.graph_store
            return mem, fake_store, factory_mock

    def test_add_to_graph_dedupes_nodes(self):
        mem, fake_store, _ = self._make_memory_with_fake_graph()
        fake_store.add_node.side_effect = ["1", "2", "3"]

        triples = [
            {
                "source": {"label": "Person", "name": "alice", "properties": {}},
                "relationship": "KNOWS",
                "target": {"label": "Person", "name": "bob", "properties": {}},
            },
            {
                # alice appears again — should not create a second node
                "source": {"label": "Person", "name": "alice", "properties": {}},
                "relationship": "WORKS_ON",
                "target": {"label": "Project", "name": "mem0", "properties": {}},
            },
        ]

        results = mem.add_to_graph(triples, memory_id="mem-1")

        # Two triples → two edges, but only three distinct (label,name) nodes.
        self.assertEqual(len(results), 2)
        self.assertEqual(fake_store.add_node.call_count, 3)
        self.assertEqual(fake_store.add_edge.call_count, 2)
        # Each node got the memory_id tag.
        for call in fake_store.add_node.call_args_list:
            props = call.kwargs["properties"] if "properties" in call.kwargs else call.args[1]
            self.assertEqual(props["memory_id"], "mem-1")

    def test_search_graph_delegates(self):
        mem, fake_store, _ = self._make_memory_with_fake_graph()
        fake_store.search.return_value = [{"id": "1", "label": "Person", "score": 0.9, "payload": {}}]

        results = mem.search_graph("alice", limit=5, filters={"user_id": "u1"})

        self.assertEqual(len(results), 1)
        fake_store.search.assert_called_once_with("alice", limit=5, filters={"user_id": "u1"})

    def test_add_to_graph_returns_empty_when_unconfigured(self):
        from mem0.memory.main import Memory

        mem = Memory.__new__(Memory)
        mem.config = MemoryConfig()  # graph=None
        mem._graph_store = None  # mimic post-init state
        # No graph_store property access should happen.
        self.assertEqual(mem.add_to_graph([]), [])


class TestGraphDataModel(unittest.TestCase):
    def test_round_trip(self):
        node = GraphData(id="1", label="Person", score=0.9, payload={"name": "alice"})
        assert node.id == "1"
        assert node.label == "Person"
        assert node.payload == {"name": "alice"}


if __name__ == "__main__":
    unittest.main()
