"""Ingest: crawl, extract, chunk, decorate, embed, write ``rag`` in Postgres.

    uv run python -m ragpipe.ingest [contextual|baseline|raptor|graph] [LIMIT] [--replace]

Each builder writes its table(s) straight into homelab's Postgres as the owner
login (``RAG_IMPORT_DATABASE_URL``) through ``ragpipe.pg_write``: upsert by id,
then prune ids the fresh corpus no longer has (ADR-0007), skipped on a LIMIT
smoke run. Vectors come from the shared embedding setting (``EMBEDDING_*``,
issue #45); a table filled by another model is refused until ``--replace``
drops and reloads it, which is how an embedding-model switch is done.
"""
from __future__ import annotations

import base64
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from bs4 import BeautifulSoup

from ragpipe.chunking import chunk_markdown
from ragpipe.extraction import html_to_markdown
from ragpipe.pg_write import embed_rows
from ragpipe.pgstore import (
    BASELINE,
    GRAPH_COMMUNITIES,
    GRAPH_ENTITIES,
    GRAPH_RELATIONSHIPS,
    MS_DOCS,
    RAPTOR_SAC,
    TableSpec,
)


def _doc_id(url: str, index: int) -> str:
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    slug = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    return f"{slug}_{index}_{digest}"


def build_documents(
    pages: list[dict[str, Any]],
    embed_batch_fn: Callable[[list[str]], list[list[float]]],
    context_fn: Callable[[str, str], str],
    max_chars: int = 2000,
    overlap: int = 200,
    batch_size: int = 64,
    context_workers: int = 4,
    spec: TableSpec = MS_DOCS,
) -> list[dict[str, Any]]:
    """Turn fetched pages into decorated ``rag.ms_docs`` rows.

    Per chunk: `context` = breadcrumb + generated situating context (ADR-0001),
    `content` = clean chunk text (ADR-0003), `embedding` = the vector of
    ``MS_DOCS.embed_input`` (`context + "\\n\\n" + content`). Context generation
    runs on a small thread pool (the callable is cache-backed and thread-safe);
    embedding is batched. ``spec`` picks whose ``embed_input`` applies (RAPTOR's
    leaves pass ``RAPTOR_SAC``).
    """
    metas: list[dict[str, Any]] = []
    chunks: list[Any] = []  # MarkdownChunk
    page_md: list[str] = []
    for page in pages:
        for i, chunk in enumerate(
            chunk_markdown(page["markdown"], page["title"], max_chars=max_chars, overlap=overlap)
        ):
            metas.append(
                {
                    "id": _doc_id(page["url"], i),
                    "title": page["title"],
                    "url": page["url"],
                    "chunk_id": i,
                }
            )
            chunks.append(chunk)
            page_md.append(page["markdown"])

    total = len(chunks)
    print(f"Generating context for {total} chunks…", flush=True)
    with ThreadPoolExecutor(max_workers=context_workers) as pool:
        generated = list(pool.map(lambda pair: context_fn(*pair), zip(page_md, [c.text for c in chunks])))
    contexts = [
        c.breadcrumb + (f"\n{g}" if g else "") for c, g in zip(chunks, generated)
    ]

    rows = [
        {**meta, "content": chunk.text, "context": ctx}
        for meta, chunk, ctx in zip(metas, chunks, contexts)
    ]
    print(f"Embedding {total} chunks in batches of {batch_size}…", flush=True)
    return embed_rows(spec, rows, embed_batch_fn, batch_size=batch_size, log=_log)


def build_baseline_documents(
    pages: list[dict[str, Any]],
    embed_batch_fn: Callable[[list[str]], list[list[float]]],
    max_chars: int = 2000,
    overlap: int = 200,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Build plain (no contextual decoration) rows for ``rag.baseline``.

    Mirrors build_documents field shapes but skips the ContextGenerator entirely:
    - ``context`` field is empty string
    - ``embedding`` embeds raw ``content`` only (``BASELINE.embed_input``)
    """
    docs: list[dict[str, Any]] = []
    for page in pages:
        for i, chunk in enumerate(
            chunk_markdown(page["markdown"], page["title"], max_chars=max_chars, overlap=overlap)
        ):
            content = chunk.text
            docs.append(
                {
                    "id": _doc_id(page["url"], i),
                    "title": page["title"],
                    "url": page["url"],
                    "chunk_id": i,
                    "content": content,
                    "context": "",
                }
            )
    return embed_rows(BASELINE, docs, embed_batch_fn, batch_size=batch_size, log=_log)


def fetch_pages(
    urls: list[str], max_workers: int = 6, max_retries: int = 4
) -> list[dict[str, Any]]:  # pragma: no cover - network
    """Fetch and extract pages concurrently; retry throttling, skip+log real failures.

    learn.microsoft.com rate-limits aggressive crawls (HTTP 429), so concurrency is
    modest and 429/5xx responses are retried with exponential backoff (honoring a
    Retry-After header when present). Genuine 404s are skipped immediately. Failures
    are logged with the actual status code, not just the exception type, so a high
    skip rate is diagnosable.
    """
    import collections

    import httpx

    client = httpx.Client(
        timeout=30, follow_redirects=True, headers={"User-Agent": "ragpipe-ingest"}
    )
    skip_reasons: collections.Counter[str] = collections.Counter()

    def fetch_one(url: str) -> dict[str, Any] | None:
        for attempt in range(max_retries):
            try:
                resp = client.get(url)
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("retry-after")
                    delay = float(retry_after) if retry_after else min(2**attempt, 30)
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                title = (
                    soup.title.string.strip() if soup.title and soup.title.string else url
                )
                markdown, used_main = html_to_markdown(resp.text)
                if not used_main:
                    skip_reasons["main-content fallback (kept)"] += 1
                if not markdown.strip():
                    return None
                return {"url": url, "title": title, "markdown": markdown}
            except httpx.HTTPStatusError as exc:
                skip_reasons[f"HTTP {exc.response.status_code}"] += 1
                return None
            except Exception as exc:  # noqa: BLE001 - one bad URL must not abort the crawl
                skip_reasons[type(exc).__name__] += 1
                return None
        skip_reasons["throttled (gave up)"] += 1
        return None

    pages: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, page in enumerate(pool.map(fetch_one, urls), 1):
            if page is not None:
                pages.append(page)
            if i % 50 == 0:
                print(f"  fetched {i}/{len(urls)} ({len(pages)} ok)", flush=True)
    client.close()
    print(f"Fetched {len(pages)}/{len(urls)} pages.", flush=True)
    if skip_reasons:
        print(f"  skipped: {dict(skip_reasons)}", flush=True)
    return pages


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write(
    settings: Any,
    spec: TableSpec,
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    replace: bool,
) -> None:  # pragma: no cover - needs a database
    """Write one table's rows to Postgres. A ``limit`` smoke run doesn't prune:
    against a partial fresh set that would empty the rest of the table."""
    from ragpipe.pg_write import connect_owner, write_table

    with connect_owner() as conn:
        result = write_table(
            conn, spec, rows, settings.embedding,
            prune=limit is None, replace=replace, log=_log,
        )
    note = " (limit, prune skipped)" if limit is not None else f", pruned {result.pruned} stale"
    print(
        f"Wrote {result.written} rows to {spec.qualified}: {result.rows} in the table{note}.",
        flush=True,
    )


def _source_urls(limit: int | None) -> list[str]:  # pragma: no cover - file
    import yaml

    with open("data/corpus_sources.yaml") as f:
        urls = yaml.safe_load(f)["sources"]
    return urls[:limit] if limit is not None else urls


def build_contextual(
    settings: Any, limit: int | None = None, replace: bool = False
) -> None:  # pragma: no cover
    """Ingest the decorated ``rag.ms_docs`` table (the contextual mode)."""
    from ragpipe.context_gen import ContextGenerator, build_context_complete_fn
    from ragpipe.embeddings import build_batch_embed_fn

    embed_batch = build_batch_embed_fn(settings)
    context_gen = ContextGenerator(
        build_context_complete_fn(settings), model=settings.foundry_chat_model
    )
    pages = fetch_pages(_source_urls(limit))
    if not pages:
        raise SystemExit("No pages fetched; nothing to ingest.")

    docs = build_documents(pages, embed_batch_fn=embed_batch, context_fn=context_gen.generate)
    if context_gen.fallback_count:
        print(f"  context fallbacks (breadcrumb-only): {context_gen.fallback_count}", flush=True)
    print(f"{len(docs)} chunks from {len(pages)} pages.", flush=True)
    _write(settings, MS_DOCS, docs, limit=limit, replace=replace)


def raptor_summary_documents(nodes: list[Any]) -> list[dict[str, Any]]:
    """Shape RAPTOR summary nodes into the same field layout as leaf docs, plus `level`.

    Summary nodes don't come from a URL or a discrete integer chunk, so `url` is
    empty and `chunk_id` is 0. The `level` field (>= 1) lets callers distinguish
    summaries from leaves in a mixed index.
    """
    return [
        {
            "id": node.id,
            "title": f"RAPTOR summary L{node.level}",
            "url": "",
            "chunk_id": 0,
            "content": node.text,
            "context": "",
            "embedding": node.embedding,
            "level": node.level,
        }
        for node in nodes
    ]


def build_raptor(
    settings: Any, limit: int | None = None, replace: bool = False
) -> None:  # pragma: no cover
    """Ingest ``rag.raptor_sac``: SAC leaf rows with level=0 plus RAPTOR summary nodes.

    Builds the full RAPTOR tree on top of the SAC (situating-context + embedding)
    leaf rows and writes leaves and summaries to one table. Leaf vectors embed
    context + content, summaries their text: ``RAPTOR_SAC.embed_input``.
    """
    import sys

    from ragpipe.context_gen import ContextGenerator, build_context_complete_fn
    from ragpipe.embeddings import build_batch_embed_fn
    from ragpipe.raptor import RaptorNode, build_raptor_tree, pack_passages

    urls = _source_urls(limit)
    embed_batch = build_batch_embed_fn(settings)
    context_gen = ContextGenerator(
        build_context_complete_fn(settings), model=settings.foundry_chat_model
    )

    pages = fetch_pages(urls)
    if not pages:
        raise SystemExit("No pages fetched; nothing to ingest.")

    # 1. Build SAC leaf docs (same decoration as the contextual table)
    leaf_docs = build_documents(
        pages, embed_batch_fn=embed_batch, context_fn=context_gen.generate, spec=RAPTOR_SAC
    )
    if context_gen.fallback_count:
        print(f"  context fallbacks (breadcrumb-only): {context_gen.fallback_count}", flush=True)
    for d in leaf_docs:
        d["level"] = 0

    # 2. Wrap leaf docs as RaptorNode leaves
    leaves = [
        RaptorNode(id=d["id"], text=d["content"], level=0, embedding=d["embedding"])
        for d in leaf_docs
    ]

    # 3. Summarizer: bounded chat-completion, same client factory as context_gen.
    #    Not content-address-cached (ADR-0005 style); would need a separate cache
    #    keyed on cluster membership which changes every ingest. Acceptable for now.
    complete_fn = build_context_complete_fn(settings)

    SUMMARY_PROMPT = (
        "You are a technical writer. Write a concise abstractive summary of the "
        "following related Azure documentation passages. Capture the key concepts "
        "and relationships. Reply with only the summary, no preamble.\n\n{passages}"
    )

    def summarize_fn(texts: list[str]) -> str:
        joined = "\n\n---\n\n".join(pack_passages(texts))
        prompt = SUMMARY_PROMPT.format(passages=joined)
        for attempt in range(settings.max_retries + 1):
            try:
                return complete_fn(prompt).strip()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"summarize_fn attempt {attempt}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        return " ".join(texts[:3])[:500]  # last-resort truncated concat

    # 4. Build RAPTOR tree
    print(f"Building RAPTOR tree from {len(leaves)} leaves (max_levels={settings.raptor_max_levels})…", flush=True)
    summaries = build_raptor_tree(
        leaves,
        summarize_fn=summarize_fn,
        embed_batch_fn=embed_batch,
        max_levels=settings.raptor_max_levels,
    )
    print(f"  RAPTOR produced {len(summaries)} summary nodes.", flush=True)

    # 5. Write leaves + summaries (prune skipped on a partial ingest).
    docs = leaf_docs + raptor_summary_documents(summaries)
    print(f"{len(leaf_docs)} leaves + {len(summaries)} summary nodes.", flush=True)
    _write(settings, RAPTOR_SAC, docs, limit=limit, replace=replace)


def build_baseline(
    settings: Any, limit: int | None = None, replace: bool = False
) -> None:  # pragma: no cover
    """Ingest the plain-chunk ``rag.baseline`` table (no contextual decoration)."""
    from ragpipe.embeddings import build_batch_embed_fn

    embed_batch = build_batch_embed_fn(settings)
    pages = fetch_pages(_source_urls(limit))
    if not pages:
        raise SystemExit("No pages fetched; nothing to ingest.")
    docs = build_baseline_documents(pages, embed_batch_fn=embed_batch)
    print(f"{len(docs)} chunks from {len(pages)} pages.", flush=True)
    _write(settings, BASELINE, docs, limit=limit, replace=replace)


def _graph_extract_workers(value: int | None = None) -> int:
    """Resolve the graph extraction thread-pool size.

    An explicit ``value`` wins; otherwise read ``RAGPIPE_GRAPH_EXTRACT_WORKERS``
    (default 8). Each extraction LLM call is independent and gpt deployments
    typically have ample TPM headroom, so the pool can be widened via env for a
    stack with a larger quota without a code change. Floored at 1.
    """
    if value is None:
        value = int(os.environ.get("RAGPIPE_GRAPH_EXTRACT_WORKERS", "8"))
    return max(1, value)


def _finish_graph(
    settings: Any,
    entities: list[Any],
    relationships: list[Any],
    communities: list[Any],
    community_map: dict[str, int],
    *,
    limit: int | None,
    replace: bool,
) -> None:  # pragma: no cover - network
    """Embed entity and community descriptions and write the three graph tables.
    Shared by both build_graph paths (fresh extraction and cache-resume) so the
    embed/write tail lives in exactly one place. Relationships aren't embedded:
    nothing queries their vectors (ADR-0021)."""
    from ragpipe.embeddings import build_batch_embed_fn
    from ragpipe.graphrag import (
        community_documents,
        entity_documents,
        relationship_documents,
    )

    embed_batch = build_batch_embed_fn(settings)
    # build_batch_embed_fn splits each call into small requests, so a big batch
    # here can't blow the per-request array cap or the TPM limit.
    ent_docs = embed_rows(
        GRAPH_ENTITIES, entity_documents(entities, community=community_map), embed_batch,
        batch_size=2048, log=_log,
    )
    com_docs = embed_rows(
        GRAPH_COMMUNITIES, community_documents(communities), embed_batch,
        batch_size=2048, log=_log,
    )
    rel_docs = relationship_documents(relationships)

    for spec, docs in (
        (GRAPH_ENTITIES, ent_docs),
        (GRAPH_RELATIONSHIPS, rel_docs),
        (GRAPH_COMMUNITIES, com_docs),
    ):
        _write(settings, spec, docs, limit=limit, replace=replace)

    print(
        f"build_graph done: {len(entities)} entities, {len(relationships)} relationships, "
        f"{len(communities)} communities.",
        flush=True,
    )


def build_graph(
    settings: Any,
    limit: int | None = None,
    replace: bool = False,
    extract_workers: int | None = None,
) -> None:  # pragma: no cover
    """Ingest the GraphRAG tables: entities, relationships, and communities.

    Steps:
    1. Fetch pages; chunk into SAC leaf docs (same decoration as main).
    2. For each leaf doc, call an LLM to extract entities + relationships in the
       delimited format parse_extraction expects.
    3. Merge entities, detect communities (Louvain), produce Community objects
       with an LLM-written title + summary per community.
    4. Embed and write the three ``rag.graph_*`` tables.

    Community reports: LLM-generated (title + summary) per community -- NOT a
    shortcut. A separate chat completion writes a short report over the member
    entities' descriptions for each community group.
    """
    extract_workers = _graph_extract_workers(extract_workers)

    import sys

    from ragpipe.context_gen import ContextGenerator, build_context_complete_fn
    from ragpipe.embeddings import build_batch_embed_fn
    from ragpipe.graphrag import (
        Community,
        detect_communities,
        load_graph_state,
        merge_entities,
        merge_relationships,
        parse_extraction,
        save_graph_state,
    )

    urls = _source_urls(limit)
    cache_path = os.environ.get("RAGPIPE_GRAPH_CACHE", ".graph_cache.json")
    cached = load_graph_state(cache_path, limit=limit)
    if cached is not None:
        entities, relationships, communities, community_map = cached
        print(
            f"Loaded graph state cache '{cache_path}': {len(entities)} entities, "
            f"{len(relationships)} relationships, {len(communities)} communities "
            f"(skipping extraction).",
            flush=True,
        )
        _finish_graph(
            settings, entities, relationships, communities, community_map,
            limit=limit, replace=replace,
        )
        return

    embed_batch = build_batch_embed_fn(settings)
    complete_fn = build_context_complete_fn(settings)
    context_gen = ContextGenerator(complete_fn, model=settings.foundry_chat_model)

    pages = fetch_pages(urls)
    if not pages:
        raise SystemExit("No pages fetched; nothing to ingest.")

    # 1. Build SAC leaf docs for chunk text + ids + urls.
    leaf_docs = build_documents(pages, embed_batch_fn=embed_batch, context_fn=context_gen.generate)
    if context_gen.fallback_count:
        print(f"  context fallbacks (breadcrumb-only): {context_gen.fallback_count}", flush=True)

    EXTRACT_PROMPT = (
        "Extract entities and relationships from the following text. "
        "Output records separated by '##'. "
        "Entity format: (\"entity\"<|>NAME<|>type<|>description). "
        "Relationship format: (\"relationship\"<|>SOURCE<|>TARGET<|>description<|>weight). "
        "Use a numeric weight 1-10 for relationships. "
        "Output only the records, no preamble.\n\n{text}"
    )

    def extract_fn(chunk_text: str) -> str:
        prompt = EXTRACT_PROMPT.format(text=chunk_text)
        for attempt in range(settings.max_retries + 1):
            try:
                return complete_fn(prompt).strip()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"extract_fn attempt {attempt}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        return ""

    # 2. Extract entities + relationships from each leaf doc. The LLM call per
    #    chunk is independent, so fan it out over a thread pool (one serial pass
    #    over ~7744 chunks would take hours); parse in stable doc order afterward
    #    so entity/relationship merge order is deterministic.
    print(f"Extracting entities/relationships from {len(leaf_docs)} leaf docs (workers={extract_workers})…", flush=True)
    raws: list[str] = [""] * len(leaf_docs)
    done = 0
    with ThreadPoolExecutor(max_workers=extract_workers) as pool:
        futures = {pool.submit(extract_fn, doc["content"]): i for i, doc in enumerate(leaf_docs)}
        for fut in as_completed(futures):
            raws[futures[fut]] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  extracted {done}/{len(leaf_docs)} docs", flush=True)

    all_entities = []
    all_relationships = []
    for doc, raw in zip(leaf_docs, raws):
        if raw:
            ents, rels = parse_extraction(raw, source_chunk_id=doc["id"], source_url=doc["url"])
            all_entities.extend(ents)
            all_relationships.extend(rels)

    print(f"  raw: {len(all_entities)} entities, {len(all_relationships)} relationships", flush=True)

    # 3. Merge and detect communities.
    entities = merge_entities(all_entities)
    relationships = merge_relationships(all_relationships)
    community_map = detect_communities([e.name for e in entities], relationships)
    print(
        f"  merged: {len(entities)} entities, {len(relationships)} relationships, "
        f"{len(community_map)} entity->community mappings",
        flush=True,
    )

    # Group entities by community id for report generation.
    from collections import defaultdict
    groups: dict[int, list] = defaultdict(list)
    for e in entities:
        cid = community_map.get(e.name, -1)
        groups[cid].append(e)

    # Group relationships by community (both endpoints must be in community).
    rel_groups: dict[int, list] = defaultdict(list)
    for r in relationships:
        src_cid = community_map.get(r.source, -1)
        tgt_cid = community_map.get(r.target, -1)
        if src_cid == tgt_cid and src_cid != -1:
            rel_groups[src_cid].append(r)

    COMMUNITY_REPORT_PROMPT = (
        "You are a technical writer. Write a short report for a cluster of related entities. "
        "First line: a concise title (no label prefix). "
        "Second line: a 2-4 sentence summary of what this cluster covers and why it matters. "
        "Reply with only title and summary, separated by a newline.\n\n"
        "Entities:\n{entities}\n\nRelationships:\n{relationships}"
    )

    print(f"Generating community reports for {len(groups)} communities…", flush=True)

    def make_report(cid: int, members: list) -> Community:
        ent_text = "\n".join(f"- {e.name} ({e.type}): {e.description[:120]}" for e in members[:20])
        rel_text = "\n".join(
            f"- {r.source} -> {r.target}: {r.description[:80]}"
            for r in rel_groups.get(cid, [])[:10]
        )
        prompt = COMMUNITY_REPORT_PROMPT.format(entities=ent_text, relationships=rel_text or "(none)")
        report = ""
        for attempt in range(settings.max_retries + 1):
            try:
                report = complete_fn(prompt).strip()
                break
            except Exception as exc:  # noqa: BLE001
                print(
                    f"community report cid={cid} attempt {attempt}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        lines = report.splitlines()
        title = lines[0].strip() if lines else f"Community {cid}"
        summary = (
            "\n".join(lines[1:]).strip()
            if len(lines) > 1
            else "; ".join(e.description[:80] for e in members[:5])
        )
        # Union the constituent members' (and their edges') source URLs so the
        # deterministic URL-match metric can score the @global stage instead of
        # treating an empty-URL community as a structural 0.0 (ADR-0002).
        source_urls: list[str] = []
        for e in members:
            for u in e.source_urls:
                if u and u not in source_urls:
                    source_urls.append(u)
        for r in rel_groups.get(cid, []):
            for u in r.source_urls:
                if u and u not in source_urls:
                    source_urls.append(u)
        return Community(id=cid, level=0, title=title, summary=summary, source_urls=source_urls)

    # Community reports are independent LLM calls; fan them out over the same
    # pool as extraction (1000+ communities done serially would dominate the
    # run). Results are collected by index so community order stays deterministic.
    ordered_groups = sorted(groups.items())
    communities: list = [None] * len(ordered_groups)
    done = 0
    with ThreadPoolExecutor(max_workers=extract_workers) as pool:
        futures = {
            pool.submit(make_report, cid, members): i
            for i, (cid, members) in enumerate(ordered_groups)
        }
        for fut in as_completed(futures):
            communities[futures[fut]] = fut.result()
            done += 1
            if done % 100 == 0:
                print(f"  generated {done}/{len(ordered_groups)} reports", flush=True)

    print(f"  generated {len(communities)} community reports.", flush=True)

    # Persist the expensive extraction + report output *before* the fragile
    # embed/upload tail, so a transient failure there resumes from cache on the
    # next run instead of re-spending the ~hour-long extraction. Best-effort:
    # a cache-write failure must never abort an otherwise-successful build.
    try:
        save_graph_state(
            cache_path,
            limit=limit,
            entities=entities,
            relationships=relationships,
            communities=communities,
            community_map=community_map,
        )
        print(f"Saved graph state cache '{cache_path}'.", flush=True)
    except Exception as exc:  # noqa: BLE001 - cache is best-effort, never fatal
        print(
            f"  warning: failed to save graph cache: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    _finish_graph(
        settings, entities, relationships, communities, community_map,
        limit=limit, replace=replace,
    )


# Builders dispatchable from the CLI, each taking (settings, limit, replace).
_BUILDERS: dict[str, Callable[..., None]] = {
    "contextual": build_contextual,
    "baseline": build_baseline,
    "raptor": build_raptor,
    "graph": build_graph,
}


def _parse_args(argv: list[str]) -> tuple[str, int | None, bool]:
    """Parse CLI args into (command, limit, replace).

    Backward compatible: no args or a leading integer selects the default
    ``contextual`` build, so ``python -m ragpipe.ingest`` (the azure.yaml
    postprovision hook) and ``python -m ragpipe.ingest 3`` (README smoke) keep
    working. A leading command name selects a builder, with an optional trailing
    limit: ``python -m ragpipe.ingest graph 3``. ``--replace`` anywhere drops and
    reloads the builder's tables (an embedding-model switch).
    """
    replace = "--replace" in argv
    args = [a for a in argv if a != "--replace"]
    command = "contextual"
    rest = args
    if args and not args[0].lstrip("-").isdigit():
        command = args[0]
        rest = args[1:]
    if command not in _BUILDERS:
        raise SystemExit(
            f"unknown command {command!r}; expected one of: {', '.join(_BUILDERS)}"
        )
    limit = int(rest[0]) if rest else None
    return command, limit, replace


# The tables each builder writes, for the preflight check.
BUILDER_TABLES: dict[str, tuple[TableSpec, ...]] = {
    "contextual": (MS_DOCS,),
    "baseline": (BASELINE,),
    "raptor": (RAPTOR_SAC,),
    "graph": (GRAPH_ENTITIES, GRAPH_RELATIONSHIPS, GRAPH_COMMUNITIES),
}


def _preflight(settings: Any, command: str, replace: bool) -> None:  # pragma: no cover - needs a database
    """Fail before crawling or paying for anything: the database must be
    reachable as the owner, and (unless replacing) the builder's tables must
    hold vectors from the configured embedding model."""
    from ragpipe.pg_write import EmbeddingMismatch, check_writable, connect_owner

    with connect_owner() as conn:
        if replace:
            return
        for spec in BUILDER_TABLES[command]:
            try:
                check_writable(conn, spec, settings.embedding)
            except EmbeddingMismatch as err:
                raise SystemExit(f"refusing to ingest: {err}") from None


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - integration entry point
    from ragpipe.config import Settings

    command, limit, replace = _parse_args(sys.argv[1:] if argv is None else argv)
    settings = Settings.from_env()
    _preflight(settings, command, replace)
    _BUILDERS[command](settings, limit=limit, replace=replace)


if __name__ == "__main__":  # pragma: no cover
    main()
