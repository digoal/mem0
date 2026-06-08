"""Live integration tests for the Apache AGE graph store backend.

These tests require a running PostgreSQL instance with the ``age`` extension
installed. They are skipped unless ``MEM0_AGE_TEST_DSN`` is set, e.g.::

    MEM0_AGE_TEST_DSN="host=/tmp port=54329 dbname=mem0_age_test user=postgres" \
        pytest tests/graphs/test_apache_age_integration.py -v

The DSN is split into kwargs that match the ``ApacheAGE`` constructor, so the
easiest form is ``connection_string=...``::

    MEM0_AGE_TEST_DSN="postgresql://postgres@/mem0_age_test?host=/tmp&port=54329"
"""

import os
import unittest

import pytest

from mem0.graphs.apache_age import ApacheAGE


PG_ENV = "MEM0_AGE_TEST_DSN"


def _build_store() -> ApacheAGE:
    """Build an ``ApacheAGE`` against the configured DSN. Drops the test graph
    on entry so each test starts from a clean slate."""
    dsn = os.environ[PG_ENV]
    if dsn.startswith("postgresql://") or dsn.startswith("postgres://"):
        store = ApacheAGE(
            connection_string=dsn,
            graph_name="mem0_intg",
        )
    else:
        # DSN as libpq keyword string: "host=... port=... dbname=... user=..."
        # We extract host/port/dbname/user and pass explicitly.
        params = dict(p.split("=", 1) for p in dsn.split() if "=" in p)
        store = ApacheAGE(
            host=params.get("host", "localhost"),
            port=int(params.get("port", "5432")),
            user=params.get("user", "postgres"),
            password=params.get("password"),
            dbname=params.get("dbname", "postgres"),
            graph_name="mem0_intg",
        )
    store.delete_all()
    return store


@pytest.mark.skipif(
    PG_ENV not in os.environ,
    reason=(f"Set {PG_ENV} to a live PostgreSQL+AGE DSN to run integration tests. See module docstring for examples."),
)
class TestApacheAGEIntegration(unittest.TestCase):
    """End-to-end tests against a real PostgreSQL+AGE instance."""

    @classmethod
    def setUpClass(cls):
        cls.store = _build_store()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.store.delete_all()
        except Exception:
            pass

    def setUp(self):
        self.store.delete_all()

    # --- nodes ----------------------------------------------------------

    def test_add_node_with_string_int_float_bool(self):
        node_id = self.store.add_node(
            "Person",
            {
                "name": "alice",
                "age": 30,
                "score": 4.5,
                "active": True,
                "tag": "中文",  # unicode
            },
        )
        # AGE ids are bigints; we return them as str.
        self.assertTrue(node_id.isdigit())

        node = self.store.get_node(node_id)
        self.assertEqual(node["label"], "Person")
        self.assertEqual(node["properties"]["name"], "alice")
        self.assertEqual(node["properties"]["age"], 30)
        self.assertEqual(node["properties"]["active"], True)
        self.assertEqual(node["properties"]["tag"], "中文")

    def test_add_node_with_null(self):
        # AGE 1.7 silently drops ``null`` values from property maps, so
        # ``None`` is not preserved as a stored property — the key simply
        # does not exist after the round-trip. Callers that need to record
        # "this attribute has no value" should omit the key or use a
        # sentinel (e.g. empty string) instead.
        node_id = self.store.add_node("Person", {"name": "bob", "deleted": None})
        node = self.store.get_node(node_id)
        self.assertEqual(node["properties"]["name"], "bob")
        self.assertNotIn("deleted", node["properties"])

    def test_add_node_with_list(self):
        node_id = self.store.add_node("Person", {"tags": ["a", "b", "c"]})
        node = self.store.get_node(node_id)
        self.assertEqual(node["properties"]["tags"], ["a", "b", "c"])

    def test_add_node_with_quote_in_value(self):
        node_id = self.store.add_node("Person", {"name": "O'Brien"})
        node = self.store.get_node(node_id)
        self.assertEqual(node["properties"]["name"], "O'Brien")

    def test_add_node_with_backslash_in_value(self):
        node_id = self.store.add_node("Person", {"path": "C:\\Users\\alice"})
        node = self.store.get_node(node_id)
        self.assertEqual(node["properties"]["path"], "C:\\Users\\alice")

    def test_add_node_empty_properties(self):
        node_id = self.store.add_node("Empty", {})
        node = self.store.get_node(node_id)
        self.assertEqual(node["label"], "Empty")
        self.assertEqual(node["properties"], {})

    def test_get_node_returns_none_for_missing(self):
        # AGE returns 0 for ids that don't exist (id() of no node).
        # We treat missing as ``None`` regardless.
        result = self.store.get_node("999999999999999")
        self.assertIsNone(result)

    # --- edges ----------------------------------------------------------

    def test_add_edge_with_properties(self):
        a = self.store.add_node("Person", {"name": "alice"})
        b = self.store.add_node("Person", {"name": "bob"})
        edge_id = self.store.add_edge(a, b, "KNOWS", {"since": 2020})
        self.assertTrue(edge_id.isdigit())

    def test_add_edge_without_properties(self):
        a = self.store.add_node("Person", {"name": "a"})
        b = self.store.add_node("Person", {"name": "b"})
        edge_id = self.store.add_edge(a, b, "KNOWS")
        self.assertTrue(edge_id.isdigit())

    # --- neighbors ------------------------------------------------------

    def test_get_neighbors_single_hop(self):
        a = self.store.add_node("Person", {"name": "a"})
        b = self.store.add_node("Person", {"name": "b"})
        c = self.store.add_node("Person", {"name": "c"})
        self.store.add_edge(a, b, "KNOWS")
        self.store.add_edge(b, c, "KNOWS")

        neighbors = self.store.get_neighbors(a, depth=1, rel_types=["KNOWS"])
        ids = {n["id"] for n in neighbors}
        self.assertEqual(ids, {b})

    def test_get_neighbors_multi_hop(self):
        # `size(r)` must work for variable-length patterns.
        a = self.store.add_node("Person", {"name": "a"})
        b = self.store.add_node("Person", {"name": "b"})
        c = self.store.add_node("Person", {"name": "c"})
        d = self.store.add_node("Person", {"name": "d"})
        self.store.add_edge(a, b, "KNOWS")
        self.store.add_edge(b, c, "KNOWS")
        self.store.add_edge(c, d, "KNOWS")

        neighbors = self.store.get_neighbors(a, depth=3, rel_types=["KNOWS"])
        by_id = {n["id"]: n for n in neighbors}
        self.assertEqual(set(by_id), {b, c, d})
        # 1-hop: b, 2-hop: c, 3-hop: d.
        self.assertEqual(by_id[b]["distance"], 1)
        self.assertEqual(by_id[c]["distance"], 2)
        self.assertEqual(by_id[d]["distance"], 3)

    def test_get_neighbors_rel_type_filter(self):
        a = self.store.add_node("Person", {"name": "a"})
        b = self.store.add_node("Person", {"name": "b"})
        c = self.store.add_node("Person", {"name": "c"})
        self.store.add_edge(a, b, "KNOWS")
        self.store.add_edge(a, c, "OWNS")

        ks = self.store.get_neighbors(a, depth=1, rel_types=["KNOWS"])
        self.assertEqual({n["id"] for n in ks}, {b})

    # --- search ---------------------------------------------------------

    def test_search_basic_substring(self):
        a = self.store.add_node("Person", {"name": "alice"})
        self.store.add_node("Person", {"name": "bob"})

        results = self.store.search("ali", limit=10)
        self.assertEqual([r["id"] for r in results], [a])

    def test_search_case_insensitive(self):
        a = self.store.add_node("Person", {"name": "Alice"})
        results = self.store.search("alice", limit=10)
        self.assertEqual([r["id"] for r in results], [a])

    def test_search_across_labels(self):
        p = self.store.add_node("Person", {"name": "mem0"})
        prj = self.store.add_node("Project", {"title": "mem0"})

        results = self.store.search("mem0", limit=10)
        ids = {r["id"] for r in results}
        self.assertEqual(ids, {p, prj})
        labels = {r["label"] for r in results}
        self.assertEqual(labels, {"Person", "Project"})

    def test_search_with_filter(self):
        a = self.store.add_node("Person", {"name": "alice", "user_id": "u1"})
        self.store.add_node("Person", {"name": "alfred", "user_id": "u2"})

        results = self.store.search("al", limit=10, filters={"user_id": "u1"})
        self.assertEqual([r["id"] for r in results], [a])

    def test_search_escapes_like_metachars(self):
        # `%` and `_` in the query must be treated as literals, not wildcards.
        a = self.store.add_node("Person", {"name": "100%"})
        b = self.store.add_node("Person", {"name": "100x"})
        c = self.store.add_node("Person", {"name": "999"})

        results = self.store.search("100%", limit=10)
        self.assertEqual([r["id"] for r in results], [a])
        # Sanity: the other two shouldn't be matched.
        self.assertNotIn(b, [r["id"] for r in results])
        self.assertNotIn(c, [r["id"] for r in results])

    def test_search_limit(self):
        for i in range(5):
            self.store.add_node("Person", {"name": f"alice_{i}"})
        results = self.store.search("alice", limit=2)
        self.assertEqual(len(results), 2)

    # --- delete / reset -------------------------------------------------

    def test_delete_node_detaches_edges(self):
        a = self.store.add_node("Person", {"name": "a"})
        b = self.store.add_node("Person", {"name": "b"})
        self.store.add_edge(a, b, "KNOWS")

        self.store.delete_node(a)
        self.assertIsNone(self.store.get_node(a))
        # b still exists; the edge is gone.
        self.assertIsNotNone(self.store.get_node(b))
        # No KNOWS edge should remain.
        self.assertEqual(self.store.get_neighbors(b, depth=1, rel_types=["KNOWS"]), [])

    def test_delete_all(self):
        for i in range(3):
            self.store.add_node("Person", {"name": f"p{i}"})
        self.store.delete_all()
        # No nodes of label Person should remain.
        results = self.store.search("p", limit=10)
        self.assertEqual(results, [])

    def test_reset(self):
        self.store.add_node("Person", {"name": "alice"})
        self.store.reset()
        self.assertEqual(self.store.search("alice", limit=10), [])

    # --- validation -----------------------------------------------------

    def test_invalid_label_raises(self):
        with self.assertRaises(ValueError):
            self.store.add_node("Person; DROP", {})

    def test_invalid_rel_type_raises(self):
        a = self.store.add_node("Person", {"name": "a"})
        b = self.store.add_node("Person", {"name": "b"})
        with self.assertRaises(ValueError):
            self.store.add_edge(a, b, "bad type", {})


if __name__ == "__main__":
    unittest.main()
