# 0021 — Retrieval on pgvector in homelab's Postgres

**Status:** Accepted (2026-10-24), amended 2026-10-25 for the reranker (#40) and 2026-11-16
for ingest on Postgres and one embedding setting (#45), and for embedding prompts (#47)

## Context

`ragpipe-search` is an Azure AI Search S1 at about €210 a month, for a demo that barely
runs. personal-website ADR 0002 moves retrieval onto pgvector in homelab's shared Postgres 16
(the `rag` database, homelab #186). The six indexes were exported on 2026-10-24, but without
vectors: every vector field was `retrievable: false`. Issue #37 is the port.

Three things Azure did for us need a replacement:

1. **k-NN on the chunk vectors.** Straightforward in pgvector.
2. **BM25.** Postgres's own `ts_rank` has no IDF: its docs say the ranking functions "do not
   use any global information". The `pg_search`/BM25 extensions want Postgres 17, and the
   shared instance is 16.
3. **The semantic ranker** on `contextual`, `baseline` and `raptor_sac`. It goes with the
   service. Its replacement, a cross-encoder on vLLM on the DGX Spark, wasn't up yet when
   this was written. See the amendment below for what's serving.

## Decision

**One table per former index**, in a `rag` schema, named after the index (`ms_docs`,
`baseline`, `raptor_sac`, `graph_entities`, `graph_communities`, `graph_relationships`).
Same columns as the export, plus `embedding vector(1536)`, a generated `tsv tsvector` over the
fields Azure marked searchable, and `doc_len`. The table names are fixed in
`ragpipe/pgstore.py`; the `*_INDEX` env vars and `SEARCH_ENDPOINT` are gone, replaced by one
`RAG_DATABASE_URL`.

**Dense:** HNSW with `vector_cosine_ops`, approximate. RAG has no clearance filter, so there's
no recall cliff from post-filtering an approximate scan.

**BM25, hand-rolled in SQL.** Okapi BM25 with Azure's defaults (k1 = 1.2, b = 0.75) and
Lucene's IDF, `ln(1 + (N - df + 0.5) / (df + 0.5))`. Query terms are the query's lexemes,
OR'ed, like Azure's `searchMode=any`. Document frequency is counted live from the GIN index,
one count per query term, so there's no stats table to go stale. The `english` text search
config stems, Azure's standard analyzer didn't, and fields are concatenated rather than scored
per field. So it's BM25, but not Azure's BM25 byte for byte. The re-eval is what says whether
that matters.

The first version of the query took ~850 ms when a term is in nearly every chunk ("azure"):
the planner guessed 40 matches instead of 7,195 and spilled a hash join to disk with every row's
content in tow. The shipped query cuts each matching tsvector down to the query's lexemes
(`setweight` + `ts_filter`) before unnesting, sorts only ids and scores, and fetches full rows
for the top k. `idf` is `MATERIALIZED`, because Postgres inlines a CTE used once and would redo
the df counts per row. Same rankings on every test query, ~25 ms.

**Graph hybrid search** (entities, communities) runs both legs and fuses with the existing
RRF, which is what Azure did internally for a query with `search_text` and a vector.
`graph_relationships` gets no vectors: `build_adjacency` only scans the text, so embedding
36k rows nobody queries would be money for nothing.

**The rerank modes are offline, not degraded.** `contextual`, `baseline` and `raptor_sac`
report `available: false, reason: "reranker-offline"` on `/modes`, `/run` refuses them with a
503, and the eval skips them. Running them on the passthrough reranker would work, but their
numbers would then describe a different pipeline under the same name. When the Spark serves the
reranker they come back and get re-evaluated.

**The service connects read-only**: every pooled session starts with
`default_transaction_read_only=on`, like the spatial layer's store, and the login is
`rag_service` (member of `rag_reader`). Only `python -m ragpipe.pg_import` writes, from the
laptop, as `rag`, from `RAG_IMPORT_DATABASE_URL`.

**The import** verifies every export file against `manifest.json` (sha256, count, complete,
unique ids) before it touches the database, then loads table by table in committed batches.
Rows already present are kept and not re-embedded, rows the export lacks are deleted. A rerun
converges on the export, never doubles rows, and a crash halfway resumes without paying twice.
`--replace` drops and reloads, for an export whose rows changed under the same ids.

### Rejected

- **One `chunks` table with an `index` column.** Reads fine, but BM25's N, avgdl and df would
  need a `WHERE index = …` everywhere, and the graph tables have different columns anyway.
- **A precomputed term-statistics table.** Faster to read, but it has to be rebuilt on every
  load and can silently drift. Live GIN counts are a few ms per term at this size.
- **`ts_rank`/`ts_rank_cd`.** No IDF, so "azure" would count as much as "durable".
- **Passthrough rerank for the three rerank modes in the meantime.** See above: it changes what
  their eval numbers mean. If that's wanted it should be its own decision, with its own label.

## Consequences

- Nothing in the running pipeline talks to `*.search.windows.net`. The `DefaultAzureCredential`
  service principal is still needed for Foundry (generation, embeddings, judges).
- Every eval figure from before this describes Azure's ranking. The six runnable modes are
  re-evaluated against Postgres and their `eval_results_<mode>.json` replaced. The three rerank
  modes' Azure-era files are removed rather than left on the card next to Postgres numbers;
  they're in git history.
- `paired_vs_baseline` is empty until `baseline` can run again, since that's the reference mode.
- Ingest still writes Azure AI Search (now with its own `SEARCH_ENDPOINT`). Until it learns to
  write Postgres, a fresh corpus means ingest → export → import. That's a follow-up, and it
  matters once the search service is deleted. *Done 2026-11-16, see the second amendment.*

## Amendment (2026-10-25): the Spark reranker (#40)

The reranker is up, and it isn't the `bge-reranker-v2-m3` named above. Homelab serves
`nvidia/llama-nemotron-rerank-1b-v2` (1B cross-encoder, 8192-token context, commercial-OK
licence) on NVIDIA's `nvcr.io/nvidia/vllm:26.02-py3` image (vLLM 0.15.1). The reason is
homelab's NVIDIA-first rule for NVIDIA hardware, not a bake-off. Nobody benchmarked the two
on this corpus. The re-eval of the three modes is the first real check.

**The three modes come back on it.** `retrieval/cross_encoder.py` posts to `/v1/rerank`
with the fused candidates and keeps vLLM's order and `relevance_score`. Not bare `/rerank`:
vLLM answers there too but skips the API key, and only the firewall keeps it shut. Document text
copies the old semantic configs so the swap is like for like: title + `content` + `context`
for `ms_docs` and `raptor_sac`, title + `content` for `baseline`. The `question:` /
`passage:` template is applied server side, so we send plain text. `Chunk` grew a `context`
field for this. It's retrieval-only, the generator still never sees it (ADR-0003).

**Available means reachable.** `unavailable_reason()` no longer hard-codes
`reranker-offline`. The rerank modes are available when `RERANKER_URL` is set and `/health`
answers 200. The probe has a 2 s timeout and its answer is cached for 60 s, so `/modes` stays
cheap and greys the modes out within a minute of the Spark going down.

**A mid-query failure is `ModeUnavailable`, not passthrough.** Calls get a 15 s timeout and
two retries with backoff on timeouts, connection errors, 429 and 5xx. Anything else, or
running out of retries, raises `ModeUnavailable("reranker-offline")`: `/run` and `/compare`
answer the same 503 as the up-front check, `/run/stream` sends an error frame with the
reason, and the eval drops the mode instead of writing a half-scored file. The failure also
marks the reranker down in the health cache. Falling back to score-sorting would quietly run
a different pipeline under the same name, the argument this ADR already made.

**No usage entry.** The Spark is homelab hardware and nobody bills it, so a rerank call adds
nothing to the run's usage (ADR-0020), same as the passthrough reranker.

**Their re-evaluated numbers won't compare like for like** with the Azure-era files:
different reranker, different first-stage BM25. The `docs/eval-history.md` entry for that run
has to say so. It does (2026-10-25). With `baseline` back, `paired_vs_baseline` fills in again.

### Rejected (amendment)

- **`bge-reranker-v2-m3`**, as planned. Fine model, just not what homelab serves on NVIDIA
  hardware.
- **Probing `/health` on every `/modes` call.** One more network hop per page load for a
  status that changes on the scale of minutes.
- **Retrying forever / long timeouts.** A request stuck on a dead Spark holds a worker and
  the user's spinner. Fail fast and say why.

## Amendment (2026-11-16): ingest writes Postgres, one embedding setting (#45)

`ragpipe-search` was deleted on 2026-11-16. Serving didn't notice: homelab ran `/run` with
`*.search.windows.net` blocked and got cited answers (homelab #186). Ingest did notice,
because it still wrote through `search_index.py`, so a reingest had nowhere to go.

**Ingest writes the `rag` schema directly.** Every builder (`contextual`, `baseline`,
`raptor`, `graph`) writes its tables through `ragpipe.pg_write.write_table`, as the owner
(`RAG_IMPORT_DATABASE_URL`), from the laptop. It uses the same `TableSpec`s and DDL as the
import. ADR-0007's contract carries over unchanged: upsert by id, then delete the ids the
fresh corpus lacks, with no prune on a LIMIT run. An overwritten row gets `doc_len` reset
so BM25 recomputes it from the regenerated `tsv`. Each batch commits on its own, so a crash
keeps what was written. Nothing GRANTs, because the database's default privileges already
make `rag`'s tables readable to `rag_reader`. `search_index.py`, `azure-search-documents`,
`SEARCH_ENDPOINT`/`SEARCH_INDEX` and the Bicep Search service with its role assignments
are gone. `pg_import` stays, as the restore path from the 2026-10-24 export.

**`TableSpec.embed_input` is the one definition of what a table's vectors embed.** Ingest
and the import both embed through it (`pg_write.embed_rows`). Writing that down surfaced a
divergence: the import had embedded `raptor_sac` leaves on `content` alone, while the
ingest that built the Azure index embedded them on `context + content` (the SAC in the
name). The spec now follows the ingest. Summary nodes have no context and embed their text.
The table on homelab still holds the content-only vectors until `raptor` is re-ingested,
and `raptor_sac`'s next eval should mention it.

**One embedding setting for ingest and serving.** `Settings.embedding`
(`config.EmbeddingSpec`) carries the model, the dimension, and optionally an
OpenAI-compatible base URL and key, read from `EMBEDDING_MODEL` (default
`FOUNDRY_EMBEDDING_MODEL`), `EMBEDDING_DIMENSIONS` (default 1536), `EMBEDDING_BASE_URL`
and `EMBEDDING_API_KEY`. The `vector(N)` DDL reads N from it, which removes
`EMBEDDING_DIMENSIONS = 1536` from `pgstore.py`. With no base URL the client is the
Foundry account's Azure OpenAI route with Entra, as before. With one, it's a plain
`openai.OpenAI` client, which is what vLLM on the Spark speaks. A response whose vectors
aren't N long raises at once, rather than as a pgvector error later.

**Each vector table records which model filled it**, in `rag.embedding_model`
(`table_name`, `model`, `dimensions`). If the model or the column size differs, ingest and
the import refuse before writing anything. `--replace` drops the table and reloads it.
Serving reads the same record when it builds a pipeline. On a mismatch every mode answers
`ModeUnavailable("embedding-mismatch")` (a 503, like `reranker-offline`), because two
models' vectors compare without error and rank noise. Only a match is cached, so a
re-ingest takes effect without a restart. A database filled before the record existed
passes, and the first write claims its tables for the configured model.

So moving to Nemotron-3.5-Embed-1B on the Spark (1024 dims, spark:8102) is configuration
plus one re-embed: set the four variables on the service and on the laptop, then run
`python -m ragpipe.ingest <builder> --replace` for each builder. Between the two steps,
serving refuses instead of answering wrong. That switch is a separate decision.

RAGAS `answer_relevancy` stays on `FOUNDRY_EMBEDDING_MODEL`. It's an eval measuring
instrument, not retrieval, and moving it would shift every score in the eval history.

### Rejected (amendment)

- **Keeping the import as the only writer, fed by a local export from ingest.** That's two
  steps and an intermediate format to keep in sync, all to avoid a dozen lines of upsert.
- **Re-embedding only rows whose text changed** (a content hash per row). A full
  re-embed of ~85k rows on `text-embedding-3-small` is well under a dollar, and free on the
  Spark. The cached context generation is the expensive part, and it's already cached.
- **Checking only the dimension.** Two different 1024-dim models fit the same column and
  return garbage without an error. The model name has to be recorded.
- **A separate embedding setting for serving.** That's the drift this amendment removes.

## Amendment (2026-11-16): prompts are part of the embedding identity (#47)

The embedder is moving to EmbeddingGemma 2 on the Spark (`google/embeddinggemma-2`,
spark:8103). The user chose it for its multimodality, not because it won a bake-off
(homelab #220). Like most asymmetric retrieval models it expects one prompt on queries and
another on documents (`task: search result | query: ` / `title: none | text: `). vLLM's
OpenAI-compatible endpoint doesn't add them. If ingest and serving disagree on them,
nothing errors and retrieval just gets worse.

**So the prompts join the setting and the record.** `EMBEDDING_QUERY_PROMPT` and
`EMBEDDING_DOCUMENT_PROMPT` (default empty, so `text-embedding-3-small` is unchanged) live
in `EmbeddingSpec`. `build_embed_fn` adds the query prompt, and
`pg_write.document_input` adds the document prompt in front of the table's
`embed_input`, for ingest and the import alike. `{title}` in the document prompt is
filled from the row's `title` (an entity's `name`), or `none`. RAPTOR's summary vectors
go through `document_input` too, not through the tree builder's embedding. The batch
embedder adds nothing, so there is exactly one place per side.
`rag.embedding_model` gained `query_prompt` and `document_prompt`. Ingest's mismatch
check and serving's `embedding-mismatch` compare all four fields, so a prompt change is
handled like a model change. The owner migrates an older record table in place. Serving,
which is read-only, reads rows as JSON, so it copes with either shape.

**OpenAI-compatible servers are sent `dimensions`.** That gets a Matryoshka model to
return the configured size (Gemma 768 → 512). The Foundry route still sends none, as
before.

**Multimodal later, not now.** The point of Gemma is putting images in the same space.
Text is the only input today. The prompt goes through `EmbeddingSpec.document_text`,
which says it's text-only, so an image document gets its own input builder beside it
rather than a text assumption to unpick.

### Rejected (#47)

- **Prompts inside each `TableSpec.embed_input`.** Those lambdas say *what* a table embeds;
  the prompt belongs to the model, and would have to be repeated in six places.
- **Not recording the prompts.** Same failure as not recording the model, just quieter.

## Sources

- personal-website ADR 0002, *Retrieval on Postgres, not Azure Search*:
  https://github.com/JoranBergfeld/personal-website/blob/main/docs/adr/0002-retrieval-on-postgres-not-azure-search.md
- pgvector README, HNSW and `vector_cosine_ops`: https://github.com/pgvector/pgvector#hnsw
- PostgreSQL 16 docs, *Ranking Search Results* (ranking ignores global information):
  https://www.postgresql.org/docs/16/textsearch-controls.html#TEXTSEARCH-RANKING
- PostgreSQL 16 docs, `MATERIALIZED` CTEs: https://www.postgresql.org/docs/16/queries-with.html#QUERIES-WITH-CTE-MATERIALIZATION
- PostgreSQL 16 docs, `setweight` / `ts_filter`: https://www.postgresql.org/docs/16/functions-textsearch.html
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009).
- Lucene `BM25Similarity` (IDF form, k1/b defaults):
  https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/search/similarities/BM25Similarity.html
- Azure AI Search, *Configure BM25 relevance scoring* (k1 = 1.2, b = 0.75):
  https://learn.microsoft.com/en-us/azure/search/index-ranking-similarity
- Azure AI Search, *Hybrid search scoring (RRF)*:
  https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
- NVIDIA, `llama-nemotron-rerank-1b-v2` model card (vLLM serving, question/passage template):
  https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2
- vLLM docs, Cohere-compatible rerank API (`/rerank`, `/v1/rerank`):
  https://docs.vllm.ai/en/stable/models/pooling_models/scoring/#cohere-rerank-api
- Azure AI Search, *Configure semantic ranker* (title, content and keyword fields, in
  priority order): https://learn.microsoft.com/en-us/azure/search/semantic-how-to-configure
- PostgreSQL 16 docs, `INSERT … ON CONFLICT DO UPDATE`:
  https://www.postgresql.org/docs/16/sql-insert.html#SQL-ON-CONFLICT
- pgvector README, the `vector(n)` type and dimension checks: https://github.com/pgvector/pgvector#storing
- vLLM docs, OpenAI-compatible Embeddings API (`/v1/embeddings`):
  https://docs.vllm.ai/en/stable/serving/openai_compatible_server/#embeddings-api
- homelab #186 (Search deletion and the serving check): https://github.com/JoranBergfeld/homelab/issues/186
- homelab #220 (EmbeddingGemma 2 chosen for multimodality): https://github.com/JoranBergfeld/homelab/issues/220
- EmbeddingGemma (first generation) model card, the same prompt scheme
  (`task: search result | query: `, `title: none | text: `) and Matryoshka sizes:
  https://huggingface.co/google/embeddinggemma-300m. Gemma 2's values come from its own
  `config_sentence_transformers.json`, as read by homelab (#220).
- vLLM docs, `dimensions` on the Embeddings API for Matryoshka models:
  https://docs.vllm.ai/en/stable/models/pooling_models/#matryoshka-embeddings
