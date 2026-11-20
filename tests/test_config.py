import pytest

from ragpipe.config import Settings, TestsetMode


def test_settings_loads_from_env(monkeypatch):
    _base_env(monkeypatch)
    # Override base env with specific values for this test
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://proj.services.ai.azure.com")
    monkeypatch.setenv("FOUNDRY_CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv("RAG_DATABASE_URL", "postgresql://rag_service@db/rag")

    s = Settings.from_env()

    assert s.foundry_chat_model == "gpt-4o"
    assert s.rag_database_url == "postgresql://rag_service@db/rag"
    assert s.faithfulness_threshold == 0.7  # default
    assert s.max_retries == 2  # default
    assert s.top_k == 5  # default
    assert s.rrf_k == 60  # default
    assert s.testset_mode is TestsetMode.HANDAUTHORED  # default


def test_settings_missing_required_raises(monkeypatch):
    monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="FOUNDRY_PROJECT_ENDPOINT"):
        Settings.from_env(load=False)


def _base_env(monkeypatch):
    for k, v in {
        "FOUNDRY_PROJECT_ENDPOINT": "https://acct.services.ai.azure.com/api/projects/p",
        "FOUNDRY_CHAT_MODEL": "gpt-5.4",
        "FOUNDRY_EMBEDDING_MODEL": "text-embedding-3-small",
        "GENERATOR_AGENT_NAME": "gen",
    }.items():
        monkeypatch.setenv(k, v)
    # Isolate test env to ensure defaults are tested, not real environment values
    monkeypatch.delenv("FAITHFULNESS_THRESHOLD", raising=False)
    monkeypatch.delenv("TOP_K", raising=False)
    monkeypatch.delenv("CANDIDATE_POOL", raising=False)
    monkeypatch.delenv("MAX_RETRIES", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("OFFLINE_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("GRAPH_QUERY_ROUTING", raising=False)
    monkeypatch.delenv("RAG_DATABASE_URL", raising=False)
    for k in ("EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY",
              "EMBEDDING_QUERY_PROMPT", "EMBEDDING_DOCUMENT_PROMPT"):
        monkeypatch.delenv(k, raising=False)


def test_judge_models_parsed_from_env(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("JUDGE_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("OFFLINE_JUDGE_MODEL", "DeepSeek-V4-Pro")
    monkeypatch.setenv("CANDIDATE_POOL", "20")
    s = Settings.from_env(load=False)
    assert s.judge_model == "claude-sonnet-4-6"
    assert s.offline_judge_model == "DeepSeek-V4-Pro"
    assert s.candidate_pool == 20


def test_judge_models_default_to_none(monkeypatch):
    _base_env(monkeypatch)
    s = Settings.from_env(load=False)
    assert s.judge_model is None
    assert s.offline_judge_model is None
    assert s.candidate_pool == 15


def test_threshold_out_of_range_rejected(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("FAITHFULNESS_THRESHOLD", "7")  # typo for 0.7
    with pytest.raises(ValueError, match="FAITHFULNESS_THRESHOLD"):
        Settings.from_env(load=False)


def test_candidate_pool_smaller_than_top_k_rejected(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("CANDIDATE_POOL", "3")
    monkeypatch.setenv("TOP_K", "5")
    with pytest.raises(ValueError, match="CANDIDATE_POOL"):
        Settings.from_env(load=False)


def test_retrieval_mode_values():
    from ragpipe.config import RetrievalMode

    assert RetrievalMode.CONTEXTUAL.value == "contextual"
    assert RetrievalMode.BASELINE.value == "baseline"
    # the 8 spec modes exist as names even if not all wired yet
    assert {"baseline", "baseline_agentic", "raptor_sac", "raptor_sac_agentic",
            "graphrag", "graphrag_agentic", "combined", "combined_agentic"} <= {
        m.value for m in RetrievalMode}


def test_settings_has_no_default_mode_field():
    import dataclasses

    from ragpipe.config import Settings

    names = {f.name for f in dataclasses.fields(Settings)}
    assert "default_mode" not in names


def test_graph_query_routing_defaults_true(monkeypatch):
    _base_env(monkeypatch)
    s = Settings.from_env(load=False)
    assert s.graph_query_routing is True


def test_graph_query_routing_disabled_from_env(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("GRAPH_QUERY_ROUTING", "false")
    s = Settings.from_env(load=False)
    assert s.graph_query_routing is False


def test_experimental_modes_are_the_agentic_wrappers():
    from ragpipe.config import EXPERIMENTAL_MODES, RetrievalMode

    assert {m.value for m in EXPERIMENTAL_MODES} == {
        "baseline_agentic",
        "raptor_sac_agentic",
        "graphrag_agentic",
        "combined_agentic",
    }
    # every experimental mode is a real, registry-eligible RetrievalMode
    assert EXPERIMENTAL_MODES <= set(RetrievalMode)


def test_is_experimental_mode_accepts_enum_and_str():
    from ragpipe.config import RetrievalMode, is_experimental_mode

    assert is_experimental_mode(RetrievalMode.COMBINED_AGENTIC) is True
    assert is_experimental_mode("baseline_agentic") is True
    assert is_experimental_mode(RetrievalMode.CONTEXTUAL) is False
    assert is_experimental_mode("contextual") is False
    assert is_experimental_mode("not-a-mode") is False


def test_database_url_is_optional_for_callers_that_never_retrieve(monkeypatch):
    _base_env(monkeypatch)
    assert Settings.from_env(load=False).rag_database_url is None


def test_rerank_modes_are_unavailable_without_a_reranker():
    from ragpipe.config import RetrievalMode, unavailable_reason

    assert unavailable_reason("contextual") == "reranker-offline"
    assert unavailable_reason(RetrievalMode.RAPTOR_SAC) == "reranker-offline"
    assert unavailable_reason("baseline_agentic") is None
    assert unavailable_reason(RetrievalMode.GRAPHRAG) is None


def test_rerank_modes_come_back_when_the_reranker_answers(monkeypatch):
    from ragpipe.config import RetrievalMode, unavailable_reason
    from ragpipe.retrieval import cross_encoder

    monkeypatch.setenv("RERANKER_URL", "http://spark:8100")
    monkeypatch.setattr(cross_encoder, "_probe", lambda *a: True)
    assert unavailable_reason("contextual") is None
    assert unavailable_reason(RetrievalMode.BASELINE) is None
    assert unavailable_reason(RetrievalMode.RAPTOR_SAC) is None


def test_rerank_modes_stay_offline_when_the_probe_fails(monkeypatch):
    from ragpipe.config import unavailable_reason
    from ragpipe.retrieval import cross_encoder

    monkeypatch.setenv("RERANKER_URL", "http://spark:8100")
    monkeypatch.setattr(cross_encoder, "_probe", lambda *a: False)
    assert unavailable_reason("baseline") == "reranker-offline"


def test_embedding_defaults_to_the_foundry_model_at_1536(monkeypatch):
    # Existing deployments set only FOUNDRY_EMBEDDING_MODEL and must keep working.
    _base_env(monkeypatch)
    e = Settings.from_env(load=False).embedding
    assert (e.model, e.dimensions, e.base_url, e.api_key) == (
        "text-embedding-3-small", 1536, None, None
    )


def test_embedding_is_one_setting_for_an_openai_compatible_server(monkeypatch):
    # The Spark switch (issue #45) is env only.
    _base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_MODEL", "nvidia/nemotron-embed")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://spark:8102/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    s = Settings.from_env(load=False)
    assert s.embedding.model == "nvidia/nemotron-embed"
    assert s.embedding.dimensions == 1024
    assert s.embedding.base_url == "http://spark:8102/v1"
    # RAGAS answer_relevancy stays on the Foundry deployment.
    assert s.foundry_embedding_model == "text-embedding-3-small"


def test_embedding_dimensions_must_be_positive(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "0")
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        Settings.from_env(load=False)


def test_embedding_prompts_are_read_verbatim(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_QUERY_PROMPT", "task: search result | query: ")
    monkeypatch.setenv("EMBEDDING_DOCUMENT_PROMPT", "title: none | text: ")
    e = Settings.from_env(load=False).embedding
    # the trailing space is part of the prompt
    assert e.query_prompt == "task: search result | query: "
    assert e.document_prompt == "title: none | text: "


def _clear_generator_env(monkeypatch):
    for k in ("GENERATOR_BASE_URL", "GENERATOR_API_KEY", "GENERATOR_MODEL",
              "GENERATOR_TEMPERATURE", "GENERATOR_AGENT_VERSION"):
        monkeypatch.delenv(k, raising=False)


def test_generator_defaults_to_the_foundry_agent(monkeypatch):
    # Existing deployments set only GENERATOR_AGENT_NAME and must keep working.
    _base_env(monkeypatch)
    _clear_generator_env(monkeypatch)
    g = Settings.from_env(load=False).generator
    assert not g.is_openai_compatible
    assert (g.agent_name, g.deployment) == ("gen", "gpt-5.4")


def test_generator_on_an_openai_compatible_server_is_env_only(monkeypatch):
    # The Spark LLM (#52). Port 8101 is an A/B slot, so the model is a setting.
    _base_env(monkeypatch)
    _clear_generator_env(monkeypatch)
    monkeypatch.delenv("GENERATOR_AGENT_NAME")
    monkeypatch.setenv("GENERATOR_BASE_URL", "http://spark:8101/v1")
    monkeypatch.setenv("GENERATOR_API_KEY", "k")
    monkeypatch.setenv("GENERATOR_MODEL", "nvidia/Qwen3.8-Flash-Next-NVFP4")
    monkeypatch.setenv("GENERATOR_TEMPERATURE", "0.2")
    g = Settings.from_env(load=False).generator
    assert g.is_openai_compatible
    assert (g.base_url, g.api_key, g.model, g.temperature) == (
        "http://spark:8101/v1", "k", "nvidia/Qwen3.8-Flash-Next-NVFP4", 0.2
    )
    assert g.deployment == "nvidia/Qwen3.8-Flash-Next-NVFP4"


def test_generator_base_url_needs_a_model(monkeypatch):
    _base_env(monkeypatch)
    _clear_generator_env(monkeypatch)
    monkeypatch.setenv("GENERATOR_BASE_URL", "http://spark:8101/v1")
    with pytest.raises(ValueError, match="GENERATOR_MODEL"):
        Settings.from_env(load=False)


def test_generator_needs_an_agent_or_a_server(monkeypatch):
    _base_env(monkeypatch)
    _clear_generator_env(monkeypatch)
    monkeypatch.delenv("GENERATOR_AGENT_NAME")
    with pytest.raises(ValueError, match="GENERATOR_AGENT_NAME"):
        Settings.from_env(load=False)


def test_generator_temperature_must_not_be_negative(monkeypatch):
    _base_env(monkeypatch)
    _clear_generator_env(monkeypatch)
    monkeypatch.setenv("GENERATOR_TEMPERATURE", "-0.1")
    with pytest.raises(ValueError, match="GENERATOR_TEMPERATURE"):
        Settings.from_env(load=False)
