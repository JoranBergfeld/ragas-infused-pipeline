from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv


class TestsetMode(str, Enum):
    HANDAUTHORED = "handauthored"
    SYNTHETIC = "synthetic"


class RetrievalMode(str, Enum):
    # Phase 1: the existing decorated index (default) and the new plain index.
    CONTEXTUAL = "contextual"
    BASELINE = "baseline"
    BASELINE_AGENTIC = "baseline_agentic"
    # Later phases register substrates for these; declared up front for a stable
    # API surface and config.
    RAPTOR_SAC = "raptor_sac"
    RAPTOR_SAC_AGENTIC = "raptor_sac_agentic"
    GRAPHRAG = "graphrag"
    GRAPHRAG_AGENTIC = "graphrag_agentic"
    COMBINED = "combined"
    COMBINED_AGENTIC = "combined_agentic"


# The four agentic wrapper modes have no eval coverage (there is no committed
# eval_results_*_agentic.json) so they are exposed as experimental/unevaluated until an
# eval run scores them (issue #11, ADR-0018). The set is derived structurally -- every
# "*_agentic" wrapper -- so any future wrapper is experimental by default until it earns
# committed eval evidence.
EXPERIMENTAL_MODES: frozenset[RetrievalMode] = frozenset(
    m for m in RetrievalMode if m.value.endswith("_agentic")
)


def is_experimental_mode(mode: "RetrievalMode | str") -> bool:
    """True for retrieval modes exposed without eval coverage (the ``*_agentic``
    wrappers). Accepts a ``RetrievalMode`` or its string value."""
    value = mode.value if isinstance(mode, RetrievalMode) else str(mode)
    return any(value == m.value for m in EXPERIMENTAL_MODES)


# Modes whose last retrieval step is a cross-encoder rerank over one table. Azure's
# semantic ranker went with the search service (issue #37, ADR-0021); its
# replacement is a cross-encoder on vLLM on the DGX Spark (issue #40). These run
# only while it answers, and /modes says so. Running them without a reranker would
# change what their eval numbers describe, so they're offline rather than degraded.
RERANK_MODES: frozenset[RetrievalMode] = frozenset(
    {RetrievalMode.CONTEXTUAL, RetrievalMode.BASELINE, RetrievalMode.RAPTOR_SAC}
)
RERANKER_OFFLINE = "reranker-offline"
# The tables were embedded with another model than the service queries with
# (issue #45): every mode is refused until serving's EMBEDDING_* matches or the
# tables are re-ingested.
EMBEDDING_MISMATCH = "embedding-mismatch"


def unavailable_reason(mode: "RetrievalMode | str") -> str | None:
    """Why a mode can't run right now, or None when it can. The rerank modes need
    ``RERANKER_URL`` set and its ``/health`` answering (probed, cached a minute)."""
    if RetrievalMode(mode) in RERANK_MODES:
        from ragpipe.retrieval.cross_encoder import reranker_reachable

        if not reranker_reachable():
            return RERANKER_OFFLINE
    return None


class ModeUnavailable(RuntimeError):
    def __init__(self, mode: "RetrievalMode | str", reason: str) -> None:
        self.mode = RetrievalMode(mode).value
        self.reason = reason
        super().__init__(f"mode {self.mode!r} is unavailable: {reason}")


@dataclass(frozen=True)
class EmbeddingSpec:
    """The one embedding model that ingest writes and serving queries with.

    Vectors from two models don't compare, so both sides read this from the same
    env (ADR-0021, issue #45). ``base_url`` unset: the Foundry account's Azure OpenAI
    route with Entra auth. Set: any OpenAI-compatible ``/v1/embeddings`` server (e.g.
    vLLM on the Spark), with ``api_key`` as a bearer token. Switching model is a
    config change plus ``python -m ragpipe.ingest <builder> --replace``.

    ``query_prompt`` / ``document_prompt`` are the model's asymmetric prefixes
    (EmbeddingGemma: ``task: search result | query: `` and ``title: none | text: ``).
    vLLM doesn't add them, so the caller must, identically at ingest and at query
    time, or retrieval degrades without an error. That's why they're part of the
    recorded embedding identity (issue #47). ``{title}`` in the document prompt is
    filled with the row's title, or ``none`` when the row has none.
    """

    model: str
    dimensions: int = 1536
    base_url: str | None = None
    api_key: str | None = None
    query_prompt: str = ""
    document_prompt: str = ""

    def __post_init__(self) -> None:
        if self.dimensions < 1:
            raise ValueError(f"EMBEDDING_DIMENSIONS must be >= 1, got {self.dimensions}")

    def query_text(self, text: str) -> str:
        """What the model embeds for a search query."""
        return f"{self.query_prompt}{text}"

    def document_text(self, text: str, title: str | None = None) -> str:
        """What the model embeds for a stored text document. Text only: an image
        document would go through its own input builder, not this one."""
        prefix = self.document_prompt.replace("{title}", (title or "").strip() or "none")
        return f"{prefix}{text}"


@dataclass(frozen=True)
class GeneratorSpec:
    """Where answers are generated. ``base_url`` set: an OpenAI-compatible
    /v1/chat/completions server running ``model``. Unset: the Foundry agent
    ``agent_name`` (on ``foundry_model``), as before #52."""

    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None
    agent_name: str | None = None
    agent_version: str | None = None
    foundry_model: str = "unknown"

    def __post_init__(self) -> None:
        if self.base_url and not self.model:
            raise ValueError("GENERATOR_MODEL is required when GENERATOR_BASE_URL is set")
        if not self.base_url and not self.agent_name:
            raise ValueError(
                "Required environment variable GENERATOR_AGENT_NAME is not set "
                "(or set GENERATOR_BASE_URL and GENERATOR_MODEL)"
            )
        if self.temperature is not None and self.temperature < 0:
            raise ValueError(f"GENERATOR_TEMPERATURE must be >= 0, got {self.temperature}")

    @property
    def is_openai_compatible(self) -> bool:
        return bool(self.base_url)

    @property
    def deployment(self) -> str:
        """The model name reported with each generation call's usage."""
        return (self.model if self.base_url else self.foundry_model) or "unknown"


def _optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else None


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required environment variable {name} is not set")
    return value


@dataclass(frozen=True)
class Settings:
    foundry_project_endpoint: str
    foundry_chat_model: str
    foundry_embedding_model: str
    # Foundry agent route (the fallback): required unless generator_base_url is set.
    generator_agent_name: str | None = None
    generator_agent_version: str | None = None
    faithfulness_threshold: float = 0.7
    max_retries: int = 2
    top_k: int = 5
    rrf_k: int = 60
    testset_mode: TestsetMode = TestsetMode.HANDAUTHORED
    # When true, the offline eval also scores context_precision/recall at each
    # retrieval stage (dense/bm25/fused/reranked) — a heavier per-stage sweep.
    per_stage_metrics: bool = False
    # Judge-model split (ADR-0009): the online faithfulness gate is judged by the
    # Anthropic deployment, the offline RAGAS suite by the DeepSeek deployment —
    # neither shares a family with the gpt generator. None = unset; the builders
    # that need them raise rather than silently falling back to the generator.
    judge_model: str | None = None
    offline_judge_model: str | None = None
    # Candidate pool per retrieval leg (dense/bm25) before RRF fusion. Wider than
    # top_k so guardrail retries can widen the rerank window over real candidates.
    candidate_pool: int = 15
    # Retrieval's Postgres (ADR-0021). The service gets a read-only login; the
    # import runs as the owner. Optional so the canary and the judge scripts,
    # which never retrieve, don't need it. Wiring a pipeline refuses without it.
    rag_database_url: str | None = None
    # Route GraphRAG's global community-summary leg by query class instead of
    # always RRF-fusing it (issue #8, ADR-0018). When True, the global leg fires
    # only for sensemaking/breadth queries; factoid queries stay local-only. Set
    # False to restore the legacy always-fuse behavior for A/B evaluation.
    graph_query_routing: bool = True
    # Bounds for later phases (declared now so config is stable).
    agentic_max_iterations: int = 3
    raptor_max_levels: int = 3
    graph_community_levels: int = 1
    # Retrieval embedding (ingest + serving). None: same as foundry_embedding_model,
    # which stays the RAGAS answer_relevancy embedding either way.
    embedding_model: str | None = None
    embedding_dimensions: int = 1536
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_query_prompt: str = ""
    embedding_document_prompt: str = ""
    # Answer generation on an OpenAI-compatible server (the Spark LLM, #52). Unset:
    # the Foundry agent above. Its own client, never the embedder's (#49).
    generator_base_url: str | None = None
    generator_api_key: str | None = None
    generator_model: str | None = None
    # None: the server's own sampling defaults (vLLM uses the model's
    # generation_config, Qwen's recommended non-thinking settings).
    generator_temperature: float | None = None

    @property
    def generator(self) -> GeneratorSpec:
        return GeneratorSpec(
            base_url=self.generator_base_url,
            api_key=self.generator_api_key,
            model=self.generator_model,
            temperature=self.generator_temperature,
            agent_name=self.generator_agent_name,
            agent_version=self.generator_agent_version,
            foundry_model=self.foundry_chat_model,
        )

    @property
    def embedding(self) -> EmbeddingSpec:
        return EmbeddingSpec(
            model=self.embedding_model or self.foundry_embedding_model,
            dimensions=self.embedding_dimensions,
            base_url=self.embedding_base_url,
            api_key=self.embedding_api_key,
            query_prompt=self.embedding_query_prompt,
            document_prompt=self.embedding_document_prompt,
        )

    def __post_init__(self) -> None:
        if not (0.0 <= self.faithfulness_threshold <= 1.0):
            raise ValueError(
                f"FAITHFULNESS_THRESHOLD must be in [0, 1], got {self.faithfulness_threshold}"
            )
        if self.max_retries < 0:
            raise ValueError(f"MAX_RETRIES must be >= 0, got {self.max_retries}")
        if self.top_k < 1:
            raise ValueError(f"TOP_K must be >= 1, got {self.top_k}")
        if self.candidate_pool < self.top_k:
            raise ValueError(
                f"CANDIDATE_POOL ({self.candidate_pool}) must be >= TOP_K ({self.top_k})"
            )
        if self.embedding_dimensions < 1:
            raise ValueError(
                f"EMBEDDING_DIMENSIONS must be >= 1, got {self.embedding_dimensions}"
            )
        # Validates the generator route (GeneratorSpec.__post_init__).
        self.generator

    @classmethod
    def from_env(cls, *, load: bool = True) -> "Settings":
        if load:
            load_dotenv()
        return cls(
            foundry_project_endpoint=_require("FOUNDRY_PROJECT_ENDPOINT"),
            foundry_chat_model=_require("FOUNDRY_CHAT_MODEL"),
            foundry_embedding_model=_require("FOUNDRY_EMBEDDING_MODEL"),
            generator_agent_name=os.environ.get("GENERATOR_AGENT_NAME") or None,
            generator_agent_version=os.environ.get("GENERATOR_AGENT_VERSION") or None,
            faithfulness_threshold=float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.7")),
            max_retries=int(os.environ.get("MAX_RETRIES", "2")),
            top_k=int(os.environ.get("TOP_K", "5")),
            rrf_k=int(os.environ.get("RRF_K", "60")),
            testset_mode=TestsetMode(os.environ.get("TESTSET_MODE", "handauthored")),
            per_stage_metrics=os.environ.get("PER_STAGE_METRICS", "").lower()
            in ("1", "true", "yes"),
            judge_model=os.environ.get("JUDGE_MODEL") or None,
            offline_judge_model=os.environ.get("OFFLINE_JUDGE_MODEL") or None,
            candidate_pool=int(os.environ.get("CANDIDATE_POOL", "15")),
            rag_database_url=os.environ.get("RAG_DATABASE_URL") or None,
            graph_query_routing=os.environ.get("GRAPH_QUERY_ROUTING", "true").lower()
            in ("1", "true", "yes"),
            agentic_max_iterations=int(os.environ.get("AGENTIC_MAX_ITERATIONS", "3")),
            raptor_max_levels=int(os.environ.get("RAPTOR_MAX_LEVELS", "3")),
            graph_community_levels=int(os.environ.get("GRAPH_COMMUNITY_LEVELS", "1")),
            embedding_model=os.environ.get("EMBEDDING_MODEL") or None,
            embedding_dimensions=int(os.environ.get("EMBEDDING_DIMENSIONS", "1536")),
            embedding_base_url=os.environ.get("EMBEDDING_BASE_URL") or None,
            embedding_api_key=os.environ.get("EMBEDDING_API_KEY") or None,
            # Not stripped: the trailing space after "query:" is part of the prompt.
            embedding_query_prompt=os.environ.get("EMBEDDING_QUERY_PROMPT", ""),
            embedding_document_prompt=os.environ.get("EMBEDDING_DOCUMENT_PROMPT", ""),
            generator_base_url=os.environ.get("GENERATOR_BASE_URL") or None,
            generator_api_key=os.environ.get("GENERATOR_API_KEY") or None,
            generator_model=os.environ.get("GENERATOR_MODEL") or None,
            generator_temperature=_optional_float("GENERATOR_TEMPERATURE"),
        )
