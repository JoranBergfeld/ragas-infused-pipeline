"""Retrieval-only comparison of embedding settings on the real corpus (issue #47).

    uv run python scripts/compare_embeddings.py EXPORT_DIR VARIANTS.json \\
        --db postgresql://postgres:test@localhost:55433 [--tables ms_docs,baseline]

Loads the export's tables into one scratch database per distinct document
embedding (model, dimensions, document prompt, server), through the same
``pg_import.import_table`` path the real import uses. It then runs the
hand-authored test set through the production ``HybridSubstrate`` (dense + BM25
+ RRF) and scores each stage with the eval's own URL hit rate and MRR. No
generation and no judges: the only spend is embeddings.

Never point ``--db`` at homelab's ``rag`` database. Each variant gets its own
database ``embbench_<corpus>`` on that server, created if missing and reused on
rerun (rows already loaded aren't re-embedded).

VARIANTS.json is a list of objects with ``name``, ``model``, ``dimensions`` and
optionally ``base_url``, ``api_key_env`` (the name of the env var holding the
key, so no key sits in the file), ``query_prompt`` and ``document_prompt``. The
first variant is the baseline the paired tests compare against.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

from ragpipe.config import Settings, TestsetMode
from ragpipe.eval.retrieval_metrics import stage_retrieval_metrics
from ragpipe.eval.stats import bootstrap_mean_ci, paired_randomization_test
from ragpipe.eval.testset import load_testset
from ragpipe.pgstore import BASELINE, MS_DOCS, open_pool, PgTable
from ragpipe.pg_import import import_table, read_export
from ragpipe.retrieval.bm25 import BM25Retriever
from ragpipe.retrieval.dense import DenseRetriever
from ragpipe.retrieval.substrate import HybridSubstrate

SPECS = {"ms_docs": MS_DOCS, "baseline": BASELINE}
CUTOFFS = (5,)  # besides the full stage list (candidate_pool deep), as the eval scores it


def _settings(base: Settings, v: dict) -> Settings:
    key = os.environ.get(v["api_key_env"]) if v.get("api_key_env") else None
    if v.get("api_key_env") and not key:
        raise SystemExit(f"{v['name']}: set {v['api_key_env']}")
    return replace(
        base,
        embedding_model=v["model"],
        embedding_dimensions=v["dimensions"],
        embedding_base_url=v.get("base_url"),
        embedding_api_key=key,
        embedding_query_prompt=v.get("query_prompt", ""),
        embedding_document_prompt=v.get("document_prompt", ""),
    )


def _corpus_key(v: dict) -> str:
    """Variants that differ only in the query prompt share a corpus."""
    ident = json.dumps(
        [v["model"], v["dimensions"], v.get("base_url"), v.get("document_prompt", "")]
    )
    slug = re.sub(r"[^a-z0-9]+", "_", f"{v['model'].split('/')[-1]}_{v['dimensions']}".lower())
    return f"{slug}_{hashlib.sha1(ident.encode()).hexdigest()[:6]}"


def _db_url(server: str, name: str) -> str:
    return f"{server.rstrip('/')}/embbench_{name}"


def _ensure_database(server: str, name: str) -> None:  # pragma: no cover - needs a database
    import psycopg

    with psycopg.connect(f"{server.rstrip('/')}/postgres", autocommit=True) as c:
        if not c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (f"embbench_{name}",)).fetchone():
            c.execute(f'CREATE DATABASE "embbench_{name}"')
    with psycopg.connect(_db_url(server, name), autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute("CREATE SCHEMA IF NOT EXISTS rag")


def _load(server: str, name: str, settings: Settings, export: dict, tables: list[str]) -> None:  # pragma: no cover
    import psycopg

    from ragpipe.embeddings import build_batch_embed_fn

    _ensure_database(server, name)
    embed_batch = build_batch_embed_fn(settings)
    with psycopg.connect(_db_url(server, name), autocommit=True) as conn:
        for t in tables:
            spec = SPECS[t]
            import_table(
                conn, spec, export[spec.index], embed_batch, settings.embedding,
                log=lambda m: print(f"  [{name}] {m}", flush=True),
            )


def _urls(chunks) -> list[str]:
    return [c.url for c in chunks]


async def _score(server: str, name: str, settings: Settings, items, tables: list[str]) -> dict:  # pragma: no cover
    from ragpipe.embeddings import build_embed_fn

    embed = build_embed_fn(settings)
    pool = open_pool(_db_url(server, name))
    per_item: dict[str, list[dict]] = {}
    try:
        for t in tables:
            table = PgTable(pool, SPECS[t])
            sub = HybridSubstrate(
                name=t,
                dense=DenseRetriever(table, embed, settings.candidate_pool),
                bm25=BM25Retriever(table, settings.candidate_pool),
                rrf_k=settings.rrf_k,
            )
            rows = []
            for item in items:
                result = await sub.retrieve(item.question, settings.top_k)
                stage_urls: dict[str, list[str]] = {}
                for stage in ("dense", "fused"):
                    urls = _urls(result.stages[stage])
                    stage_urls[stage] = urls
                    for k in CUTOFFS:
                        stage_urls[f"{stage}@{k}"] = urls[:k]
                rows.append(stage_retrieval_metrics(stage_urls, item.ground_truth_context))
            per_item[t] = rows
    finally:
        pool.close()
    return per_item


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - live
    from dotenv import load_dotenv

    p = argparse.ArgumentParser(prog="compare_embeddings")
    p.add_argument("export_dir", type=Path)
    p.add_argument("variants", type=Path)
    p.add_argument("--db", required=True, help="server URL without a database name")
    p.add_argument("--tables", default="ms_docs,baseline")
    p.add_argument("--out", type=Path, default=Path("embedding_comparison.json"))
    args = p.parse_args(argv)
    if re.search(r"/rag/?$", args.db):
        raise SystemExit("--db is a server URL; the bench creates its own embbench_* databases")

    load_dotenv()
    base = Settings.from_env(load=False)
    variants = json.loads(args.variants.read_text(encoding="utf-8"))
    tables = args.tables.split(",")
    items = [i for i in load_testset(TestsetMode.HANDAUTHORED) if i.ground_truth_context]
    export = read_export(args.export_dir.expanduser(), tuple(SPECS[t] for t in tables))
    print(f"{len(items)} questions; tables {tables}; {len(variants)} variants", flush=True)

    loaded: set[str] = set()
    results: dict[str, dict] = {}
    for v in variants:
        s = _settings(base, v)
        corpus = _corpus_key(v)
        if corpus not in loaded:
            print(f"loading corpus {corpus} for {v['name']}…", flush=True)
            _load(args.db, corpus, s, export, tables)
            loaded.add(corpus)
        print(f"scoring {v['name']}…", flush=True)
        results[v["name"]] = asyncio.run(_score(args.db, corpus, s, items, tables))

    baseline_name = variants[0]["name"]
    summary: dict = {}
    for t in tables:
        metrics = list(results[baseline_name][t][0])
        summary[t] = {}
        for name, per_table in results.items():
            rows = per_table[t]
            summary[t][name] = {}
            for m in metrics:
                vals = [r[m] for r in rows]
                entry = {"mean": bootstrap_mean_ci(vals)}
                if name != baseline_name:
                    base_vals = [r[m] for r in results[baseline_name][t]]
                    entry["vs_baseline"] = paired_randomization_test(vals, base_vals)
                summary[t][name][m] = entry
    args.out.write_text(json.dumps({"variants": variants, "summary": summary,
                                    "per_item": results}, indent=1, default=str), encoding="utf-8")

    for t in tables:
        metrics = list(results[baseline_name][t][0])
        print(f"\n== {t} (n={len(items)}) ==")
        print("variant".ljust(24) + "".join(m.replace("hit_rate", "hit").ljust(14) for m in metrics))
        for name in results:
            cells = []
            for m in metrics:
                e = summary[t][name][m]
                cell = f"{e['mean']['mean']:.3f}"
                vb = e.get("vs_baseline")
                if vb and vb["significant"]:
                    cell += "*"
                cells.append(cell.ljust(14))
            print(name.ljust(24) + "".join(cells))
    print(f"\n* = paired randomization test vs {baseline_name}, p < 0.05. Full output: {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
