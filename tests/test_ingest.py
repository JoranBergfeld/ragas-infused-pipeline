import re

import pytest

from ragpipe.ingest import (
    _graph_extract_workers,
    _parse_args,
    build_documents,
)
from ragpipe.pgstore import MS_DOCS, RAPTOR_SAC


def _fake_batch_embed(texts):
    # one deterministic 2-d vector per input, in order
    return [[float(len(t) % 7), 0.1] for t in texts]


def _no_context(document: str, chunk: str) -> str:
    return ""


def test_build_documents_chunks_embeds_and_decorates():
    pages = [{"url": "http://x", "title": "T", "markdown": "# H\n\n" + "word " * 600}]

    docs = build_documents(
        pages,
        embed_batch_fn=_fake_batch_embed,
        context_fn=lambda doc, chunk: "generated ctx",
        max_chars=1000,
        overlap=100,
        batch_size=2,
    )

    assert len(docs) >= 2
    first = docs[0]
    assert re.fullmatch(r"[A-Za-z0-9_\-=]+", first["id"])
    again = build_documents(
        pages,
        embed_batch_fn=_fake_batch_embed,
        context_fn=lambda doc, chunk: "generated ctx",
        max_chars=1000,
        overlap=100,
        batch_size=2,
    )
    assert again[0]["id"] == first["id"]
    assert docs[0]["id"] != docs[1]["id"]
    assert first["title"] == "T"
    # ADR-0003: content stays clean; decoration lives in `context`
    assert "generated ctx" not in first["content"]
    assert first["context"] == "T > H\ngenerated ctx"
    assert len(first["embedding"]) == 2
    # every column the table has, plus the vector
    assert set(first) == set(MS_DOCS.columns) | {"embedding"}


def test_build_documents_breadcrumb_only_on_empty_context():
    pages = [{"url": "http://x", "title": "T", "markdown": "# H\n\nbody text"}]
    docs = build_documents(
        pages, embed_batch_fn=_fake_batch_embed, context_fn=_no_context
    )
    assert docs[0]["context"] == "T > H"


def test_build_documents_embeds_decorated_text_in_chunk_order():
    pages = [{"url": "http://x", "title": "T", "markdown": "# H\n\n" + "word " * 1500}]
    seen = []

    def batch(texts):
        seen.extend(texts)
        return [[float(i), 0.0] for i, _ in enumerate(texts)]

    docs = build_documents(
        pages, embed_batch_fn=batch, context_fn=_no_context,
        max_chars=500, overlap=50, batch_size=2,
    )
    # the EMBEDDED text is context + "\n\n" + content (ADR-0003)
    assert [f"{d['context']}\n\n{d['content']}" for d in docs] == seen
    assert len(docs) >= 3


def test_raptor_leaves_embed_through_the_raptor_spec():
    pages = [{"url": "http://x", "title": "T", "markdown": "# H\n\nbody"}]
    seen = []

    def batch(texts):
        seen.extend(texts)
        return [[0.0, 0.0] for _ in texts]

    docs = build_documents(pages, embed_batch_fn=batch, context_fn=_no_context, spec=RAPTOR_SAC)
    assert seen == [RAPTOR_SAC.embed_input({**d, "level": 0}) for d in docs]


def test_parse_args_defaults_to_contextual_for_backward_compat():
    # The azure.yaml postprovision hook calls `python -m ragpipe.ingest` (no args)
    # and the README smokes with a bare integer limit; both must stay 'contextual'.
    assert _parse_args([]) == ("contextual", None, False)
    assert _parse_args(["3"]) == ("contextual", 3, False)
    assert _parse_args(["contextual", "5"]) == ("contextual", 5, False)


def test_parse_args_selects_substrate_builders_with_optional_limit():
    assert _parse_args(["baseline"]) == ("baseline", None, False)
    assert _parse_args(["raptor", "3"]) == ("raptor", 3, False)
    assert _parse_args(["graph", "10"]) == ("graph", 10, False)


def test_parse_args_replace_anywhere():
    assert _parse_args(["--replace"]) == ("contextual", None, True)
    assert _parse_args(["graph", "--replace"]) == ("graph", None, True)
    assert _parse_args(["--replace", "baseline", "3"]) == ("baseline", 3, True)


def test_parse_args_rejects_unknown_command():
    with pytest.raises(SystemExit):
        _parse_args(["bogus"])


def test_graph_extract_workers_resolution(monkeypatch):
    # explicit value always wins over the env
    monkeypatch.setenv("RAGPIPE_GRAPH_EXTRACT_WORKERS", "24")
    assert _graph_extract_workers(4) == 4
    # env override applies when unset
    assert _graph_extract_workers() == 24
    # default when neither given
    monkeypatch.delenv("RAGPIPE_GRAPH_EXTRACT_WORKERS", raising=False)
    assert _graph_extract_workers() == 8
    # floored at 1
    assert _graph_extract_workers(0) == 1


def test_every_builder_names_the_tables_it_writes():
    from ragpipe.ingest import _BUILDERS, BUILDER_TABLES
    from ragpipe.pgstore import TABLES

    assert set(BUILDER_TABLES) == set(_BUILDERS)
    assert sorted(s.table for specs in BUILDER_TABLES.values() for s in specs) == sorted(
        s.table for s in TABLES
    )
