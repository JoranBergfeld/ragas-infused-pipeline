"""Writing the ``rag`` schema: ingest and the export import share this (issue #45).

Both run from the laptop as the database owner (``RAG_IMPORT_DATABASE_URL``, the
``rag`` login), never from the serving container, which is read-only. Default
privileges on the database make every table ``rag`` creates readable to
``rag_reader``, so nothing here GRANTs.

Every vector table records which embedding model filled it in
``rag.embedding_model``. Writing with a different model, or a different
dimension, is refused until the caller asks for ``replace`` (drop and reload),
because one table holding vectors from two models would rank garbage without
any error. Serving reads the same record (``recorded_mismatch``) and refuses to
query with a model the table wasn't filled with.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from ragpipe.config import EmbeddingSpec
from ragpipe.pgstore import SCHEMA, TableSpec, create_table_sql, fill_doc_len_sql, vector_literal

EMBEDDING_META = f"{SCHEMA}.embedding_model"
EMBEDDING_META_DDL = (
    f"CREATE TABLE IF NOT EXISTS {EMBEDDING_META} (\n"
    "  table_name text PRIMARY KEY,\n"
    "  model text NOT NULL,\n"
    "  dimensions integer NOT NULL,\n"
    "  query_prompt text NOT NULL DEFAULT '',\n"
    "  document_prompt text NOT NULL DEFAULT '',\n"
    "  updated_at timestamptz NOT NULL DEFAULT now()\n"
    ")"
)
# For a record table created before the prompts were part of the identity.
EMBEDDING_META_MIGRATE = (
    f"ALTER TABLE {EMBEDDING_META} "
    "ADD COLUMN IF NOT EXISTS query_prompt text NOT NULL DEFAULT '', "
    "ADD COLUMN IF NOT EXISTS document_prompt text NOT NULL DEFAULT ''"
)


class EmbeddingMismatch(RuntimeError):
    """A table holds vectors from another model, dimension or prompt pair than
    the setting."""


@dataclass(frozen=True)
class Recorded:
    """What ``rag.embedding_model`` says filled a table."""

    model: str
    dimensions: int
    query_prompt: str = ""
    document_prompt: str = ""

    def differs_from(self, embedding: EmbeddingSpec) -> str | None:
        """The first part of the embedding identity that differs, described, or None."""
        for name, ours, theirs in (
            ("model", self.model, embedding.model),
            ("dimensions", self.dimensions, embedding.dimensions),
            ("query prompt", self.query_prompt, embedding.query_prompt),
            ("document prompt", self.document_prompt, embedding.document_prompt),
        ):
            if ours != theirs:
                return f"{name} {ours!r}, the setting says {theirs!r}"
        return None


def _ensure_meta(conn) -> None:
    conn.execute(EMBEDDING_META_DDL)
    conn.execute(EMBEDDING_META_MIGRATE)


def owner_database_url() -> str:  # pragma: no cover - env
    url = os.environ.get("RAG_IMPORT_DATABASE_URL")
    if not url:
        raise SystemExit(
            "set RAG_IMPORT_DATABASE_URL to the owner login (rag): ingest and "
            "pg_import write the rag schema directly"
        )
    return url


def connect_owner():  # pragma: no cover - needs a database
    """An autocommit connection as the owner, with the schema and the
    embedding-model record in place."""
    import psycopg

    conn = psycopg.connect(owner_database_url(), autocommit=True)
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    _ensure_meta(conn)
    return conn


def _column_dimensions(conn, spec: TableSpec) -> int | None:
    """The ``vector(N)`` size of the table's embedding column, None if no table.
    pgvector stores N as the column's typmod."""
    row = conn.execute(
        "SELECT a.atttypmod FROM pg_attribute a "
        "WHERE a.attrelid = to_regclass(%s) AND a.attname = 'embedding' AND NOT a.attisdropped",
        (spec.qualified,),
    ).fetchone()
    return row[0] if row else None


_RECORD_COLUMNS = "model, dimensions, query_prompt, document_prompt"


def _recorded(conn, spec: TableSpec) -> Recorded | None:
    row = conn.execute(
        f"SELECT {_RECORD_COLUMNS} FROM {EMBEDDING_META} WHERE table_name = %s", (spec.table,)
    ).fetchone()
    return Recorded(*row) if row else None


def mismatch_reason(
    spec: TableSpec,
    embedding: EmbeddingSpec,
    *,
    column_dims: int | None,
    recorded: Recorded | None,
) -> str | None:
    """Why ``spec``'s existing table can't take vectors from ``embedding``, or None.

    No table yet, or a table without a record (filled before records existed,
    so assumed to be the current setting), passes; the column size is checked
    whenever there is a column. The prompts count like the model: vectors made
    with another prefix sit in a shifted space and rank worse without any error.
    """
    if spec.embed_input is None:
        return None
    if column_dims is not None and column_dims != embedding.dimensions:
        return (
            f"{spec.qualified} holds vector({column_dims}), EMBEDDING_DIMENSIONS is "
            f"{embedding.dimensions}"
        )
    if recorded is not None:
        diff = recorded.differs_from(embedding)
        if diff:
            return f"{spec.qualified} was embedded with {diff}"
    return None


def check_writable(conn, spec: TableSpec, embedding: EmbeddingSpec) -> None:
    """Raise EmbeddingMismatch if ``spec``'s table holds another embedding's
    vectors. Ingest calls it before crawling, so a refusal costs nothing."""
    _ensure_meta(conn)
    reason = mismatch_reason(
        spec, embedding, column_dims=_column_dimensions(conn, spec), recorded=_recorded(conn, spec)
    )
    if reason:
        raise EmbeddingMismatch(f"{reason}. Re-embed it: rerun with --replace.")


def prepare_table(conn, spec: TableSpec, embedding: EmbeddingSpec, *, replace: bool = False) -> None:
    """Create ``spec``'s table if needed and claim it for ``embedding``.

    ``replace`` drops the table (and its record) first. Without it, a table
    filled by another model or dimension raises EmbeddingMismatch, before
    anything is written.
    """
    _ensure_meta(conn)
    if replace:
        conn.execute(f"DROP TABLE IF EXISTS {spec.qualified}")
        conn.execute(f"DELETE FROM {EMBEDDING_META} WHERE table_name = %s", (spec.table,))
    check_writable(conn, spec, embedding)
    for stmt in create_table_sql(spec, embedding.dimensions):
        conn.execute(stmt)
    if spec.embed_input is not None:
        conn.execute(
            f"INSERT INTO {EMBEDDING_META} (table_name, {_RECORD_COLUMNS}) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (table_name) DO UPDATE SET model = EXCLUDED.model, "
            "dimensions = EXCLUDED.dimensions, query_prompt = EXCLUDED.query_prompt, "
            "document_prompt = EXCLUDED.document_prompt, updated_at = now()",
            (
                spec.table, embedding.model, embedding.dimensions,
                embedding.query_prompt, embedding.document_prompt,
            ),
        )


def document_input(
    spec: TableSpec, row: dict[str, Any], embedding: EmbeddingSpec | None
) -> str:
    """The exact text a row's vector embeds: the table's ``embed_input`` behind the
    model's document prompt, ``{title}`` filled from the row (graph entities have
    a ``name`` instead). The one definition, shared by ingest and the import.
    ``embedding`` None: no prompt (unit tests and prompt-less models)."""
    assert spec.embed_input is not None
    text = spec.embed_input(row)
    if embedding is None:
        return text
    return embedding.document_text(text, row.get("title") or row.get("name"))


def embed_rows(
    spec: TableSpec,
    rows: list[dict[str, Any]],
    embed_batch: Callable[[list[str]], list[list[float]]],
    embedding: EmbeddingSpec | None = None,
    *,
    batch_size: int = 256,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """``rows`` with an ``embedding`` each, of ``document_input``."""
    if spec.embed_input is None:
        return rows
    out: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = embed_batch([document_input(spec, r, embedding) for r in batch])
        out.extend({**r, "embedding": v} for r, v in zip(batch, vectors))
        log(f"  embedded {len(out)}/{len(rows)} for {spec.qualified}")
    return out


def upsert_sql(spec: TableSpec) -> str:
    """Insert or overwrite one row by id. ``doc_len`` resets so the BM25 length
    is recomputed from the regenerated ``tsv``."""
    columns = list(spec.columns)
    values = ["%s"] * len(columns)
    if spec.embed_input is not None:
        columns.append("embedding")
        values.append("%s::vector")
    updates = [f"{c} = EXCLUDED.{c}" for c in columns if c != "id"]
    if spec.searchable:
        updates.append("doc_len = NULL")
    return (
        f"INSERT INTO {spec.qualified} ({', '.join(columns)}) VALUES ({', '.join(values)}) "
        f"ON CONFLICT (id) DO UPDATE SET {', '.join(updates)}"
    )


def row_params(spec: TableSpec, row: dict[str, Any], dimensions: int) -> list[Any]:
    values = [row[c] for c in spec.columns]
    if spec.embed_input is not None:
        vector = row["embedding"]
        if len(vector) != dimensions:
            raise EmbeddingMismatch(
                f"{spec.qualified} row {row['id']}: {len(vector)}-dim vector, expected {dimensions}"
            )
        values.append(vector_literal(vector))
    return values


@dataclass
class WriteResult:
    table: str
    rows: int
    written: int
    pruned: int


def write_table(
    conn,
    spec: TableSpec,
    rows: list[dict[str, Any]],
    embedding: EmbeddingSpec,
    *,
    prune: bool = True,
    replace: bool = False,
    batch_size: int = 500,
    log: Callable[[str], None] = print,
) -> WriteResult:
    """Upsert ``rows`` (already carrying ``embedding`` where the table has
    vectors) into ``spec``'s table, then, with ``prune``, delete every id they
    don't include. ADR-0007's in-place update with prune, on Postgres.

    ``conn`` is an autocommit connection. Each batch commits on its own, so a
    crash keeps what was written and a rerun overwrites it. Pass ``prune=False``
    for a partial (``limit``) run: pruning against a partial set would empty
    the rest of the table.
    """
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{spec.qualified}: duplicate ids in the rows to write")
    prepare_table(conn, spec, embedding, replace=replace)

    sql = upsert_sql(spec)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        params = [row_params(spec, r, embedding.dimensions) for r in batch]
        with conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(sql, params)
        log(f"  wrote {min(start + batch_size, len(rows))}/{len(rows)} to {spec.qualified}")

    pruned = 0
    if prune:
        cur = conn.execute(f"DELETE FROM {spec.qualified} WHERE NOT (id = ANY(%s))", (ids,))
        pruned = cur.rowcount
    if spec.searchable:
        conn.execute(fill_doc_len_sql(spec))
    conn.execute(f"ANALYZE {spec.qualified}")
    count = conn.execute(f"SELECT count(*) FROM {spec.qualified}").fetchone()[0]
    if prune and count != len(rows):
        raise RuntimeError(f"{spec.qualified}: {count} rows after write, expected {len(rows)}")
    return WriteResult(spec.table, count, len(rows), pruned)


def recorded_mismatch(conn, embedding: EmbeddingSpec) -> str | None:
    """For serving: why the tables can't be queried with ``embedding``, or None.

    No record table (a database filled before records existed) passes, since
    there is nothing to compare against. Rows are read as JSON because the
    service can't migrate a record table that predates the prompt columns."""
    exists = conn.execute("SELECT to_regclass(%s)", (EMBEDDING_META,)).fetchone()[0]
    if exists is None:
        return None
    for (row,) in conn.execute(
        f"SELECT to_jsonb(m) FROM {EMBEDDING_META} m ORDER BY m.table_name"
    ):
        recorded = Recorded(
            model=row["model"],
            dimensions=row["dimensions"],
            query_prompt=row.get("query_prompt", ""),
            document_prompt=row.get("document_prompt", ""),
        )
        diff = recorded.differs_from(embedding)
        if diff:
            return f"{SCHEMA}.{row['table_name']} was embedded with {diff}"
    return None
