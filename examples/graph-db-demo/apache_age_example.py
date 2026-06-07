"""Apache AGE as a graph store for Mem0.

This example shows the simplest way to add relationship-aware memory to a
Mem0 deployment that's already running PostgreSQL (for pgvector).

Prerequisites
-------------
1. PostgreSQL with the AGE extension installed:
   https://age.apache.org/getstarted/
2. psycopg2-binary or psycopg[binary,pool]

Run
---
$ python apache_age_example.py
"""

from mem0 import Memory

config = {
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "user": "postgres",
            "password": "postgres",
            "host": "127.0.0.1",
            "port": "5432",
        },
    },
    "graph": {
        "provider": "apache_age",
        "config": {
            "user": "postgres",
            "password": "postgres",
            "host": "127.0.0.1",
            "port": "5432",
            "graph_name": "mem0_graph",
        },
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4.1-mini"},
    },
}

m = Memory.from_config(config)

# 1. Add some memories through the normal flow.
messages = [
    {"role": "user", "content": "Alice works on the Mem0 project and uses pgvector."},
    {"role": "assistant", "content": "Noted. Alice + Mem0 + pgvector."},
]
m.add(messages, user_id="alice")

# 2. Insert (entity, relationship, entity) triples into the graph.
triples = [
    {
        "source":     {"label": "Person",  "name": "alice",  "properties": {"role": "engineer"}},
        "relationship": "WORKS_ON",
        "target":     {"label": "Project", "name": "mem0",   "properties": {}},
    },
    {
        # Alice appears again — deduped within this call.
        "source":     {"label": "Person",   "name": "alice",    "properties": {}},
        "relationship": "USES",
        "target":     {"label": "Library",  "name": "pgvector", "properties": {}},
    },
]
ids = m.add_to_graph(triples, memory_id="example-1")
print("Inserted triples:", ids)

# 3. Search the graph by name.
results = m.search_graph("alice", limit=10)
for r in results:
    print(r)

# 4. Drop the graph if you want a clean slate.
m.graph_store.reset()
