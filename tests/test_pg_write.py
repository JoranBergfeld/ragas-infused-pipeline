"""The ingest writer's pure parts. The SQL against a real Postgres is in
test_pgstore.py (needs RAG_TEST_DATABASE_URL)."""
from __future__ import annotations

import pytest

from ragpipe.config import EmbeddingSpec
from ragpipe.pg_write import (
    EmbeddingMismatch,
    embed_rows,
    mismatch_reason,
    row_params,
    upsert_sql,
)
from ragpipe.pgstore import (
    BASELINE,
    GRAPH_RELATIONSHIPS,
    MS_DOCS,
    RAPTOR_SAC,
    create_table_sql,
)

EMB = EmbeddingSpec(model="text-embedding-3-small", dimensions=1536)


def _chunk(**kw) -> dict:
    return {"id": "c0", "title": "T", "url": "u", "chunk_id": 0,
            "content": "body", "context": "T > H", **kw}


def test_the_column_size_comes_from_the_setting():
    ddl = create_table_sql(MS_DOCS, 1024)[0]
    assert "embedding vector(1024) NOT NULL" in ddl
    assert not any("vector(" in stmt for stmt in create_table_sql(GRAPH_RELATIONSHIPS, 1024))


def test_mismatch_reason_passes_a_new_or_unrecorded_table():
    assert mismatch_reason(MS_DOCS, EMB, column_dims=None, recorded=None) is None
    assert mismatch_reason(MS_DOCS, EMB, column_dims=1536, recorded=None) is None
    assert (
        mismatch_reason(MS_DOCS, EMB, column_dims=1536, recorded=("text-embedding-3-small", 1536))
        is None
    )


def test_mismatch_reason_names_the_model_or_dimension_that_differs():
    spark = EmbeddingSpec(model="nemotron-embed", dimensions=1024)
    assert "vector(1536)" in mismatch_reason(MS_DOCS, spark, column_dims=1536, recorded=None)
    same_dims = EmbeddingSpec(model="nemotron-embed", dimensions=1536)
    reason = mismatch_reason(
        MS_DOCS, same_dims, column_dims=1536, recorded=("text-embedding-3-small", 1536)
    )
    assert "text-embedding-3-small" in reason and "nemotron-embed" in reason


def test_tables_without_vectors_never_mismatch():
    spark = EmbeddingSpec(model="nemotron-embed", dimensions=1024)
    assert mismatch_reason(GRAPH_RELATIONSHIPS, spark, column_dims=None, recorded=None) is None


def test_upsert_overwrites_every_column_and_resets_doc_len():
    sql = upsert_sql(MS_DOCS)
    assert "ON CONFLICT (id) DO UPDATE SET" in sql
    assert "content = EXCLUDED.content" in sql and "embedding = EXCLUDED.embedding" in sql
    assert "doc_len = NULL" in sql
    assert "id = EXCLUDED.id" not in sql
    assert "%s::vector" in sql
    # no vector, no searchable text: neither clause
    rel = upsert_sql(GRAPH_RELATIONSHIPS)
    assert "embedding" not in rel and "doc_len" not in rel


def test_row_params_refuse_a_wrong_sized_vector():
    row = _chunk(embedding=[0.0] * 3)
    with pytest.raises(EmbeddingMismatch, match="3-dim vector, expected 4"):
        row_params(MS_DOCS, row, 4)
    assert row_params(MS_DOCS, _chunk(embedding=[0.0] * 4), 4)[-1] == "[0.0,0.0,0.0,0.0]"


def test_embed_rows_embeds_each_tables_own_input():
    seen: list[str] = []

    def batch(texts):
        seen.extend(texts)
        return [[float(i)] for i, _ in enumerate(texts)]

    out = embed_rows(MS_DOCS, [_chunk()], batch, log=lambda _m: None)
    assert seen == ["T > H\n\nbody"] and out[0]["embedding"] == [0.0]

    seen.clear()
    embed_rows(BASELINE, [_chunk(context="")], batch, log=lambda _m: None)
    assert seen == ["body"]

    # RAPTOR: leaves embed like ms_docs, summary nodes (no context) their text.
    seen.clear()
    embed_rows(
        RAPTOR_SAC, [_chunk(level=0), _chunk(id="s", context="", level=1)], batch,
        log=lambda _m: None,
    )
    assert seen == ["T > H\n\nbody", "body"]


def test_embed_rows_leaves_tables_without_vectors_alone():
    def boom(_texts):
        raise AssertionError("relationships aren't embedded")

    rows = [{"id": "rel-0"}]
    assert embed_rows(GRAPH_RELATIONSHIPS, rows, boom) is rows
