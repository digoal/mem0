"""Pydantic config for the Apache AGE graph store backend.

Mirrors the shape of :class:`mem0.configs.vector_stores.pgvector.PGVectorConfig`
so users familiar with the pgvector config can switch to the graph store by
changing only the ``provider`` key. The most important shared option is
``connection_pool`` — when supplied, it lets the graph store reuse the
``PGVector`` connection pool for true "one PostgreSQL connection, two stores"
operation.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApacheAGEConfig(BaseModel):
    dbname: str = Field("postgres", description="Default name for the database")
    user: Optional[str] = Field(None, description="Database user")
    password: Optional[str] = Field(None, description="Database password")
    host: Optional[str] = Field(None, description="Database host. Default is localhost")
    port: Optional[int] = Field(None, description="Database port")
    graph_name: str = Field("mem0_graph", description="AGE graph namespace")
    minconn: int = Field(1, description="Minimum number of connections in the pool")
    maxconn: int = Field(5, description="Maximum number of connections in the pool")
    sslmode: Optional[str] = Field(
        None, description="SSL mode for PostgreSQL connection (e.g. 'require', 'prefer', 'disable')"
    )
    connection_string: Optional[str] = Field(
        None,
        description="PostgreSQL DSN. Overrides user/password/host/port/dbname when set.",
    )
    connection_pool: Optional[Any] = Field(
        None,
        description=(
            "Pre-built psycopg2/psycopg connection pool. Pass the same pool "
            "used by PGVector to share a single pool between vector and graph stores."
        ),
    )

    @model_validator(mode="before")
    def check_auth_and_connection(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("connection_pool") is not None:
            return values
        if values.get("connection_string") is not None:
            return values

        user, password = values.get("user"), values.get("password")
        host, port = values.get("host"), values.get("port")
        if not user and not password:
            raise ValueError(
                "Both 'user' and 'password' must be provided when not using connection_string or connection_pool."
            )
        if not host and not port:
            raise ValueError(
                "Both 'host' and 'port' must be provided when not using connection_string or connection_pool."
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. "
                f"Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    model_config = ConfigDict(arbitrary_types_allowed=True)
