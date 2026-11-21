# 0022 — Answer generation on the Spark LLM

**Status:** Accepted (2026-11-18), issue #52

## Context

Answers come from a Foundry Agent Service agent (`ragpipe-generator`, `gpt-5.4`). It has
the Code Interpreter attached (`scripts/setup_agents.py`). The DGX Spark now serves an LLM
next to the reranker and the embedder:

- vLLM, OpenAI-compatible, at `http://192.168.5.6:8101/v1`.
- Model `nvidia/Qwen3.8-Flash-Next-NVFP4`.
- Bearer key, `--max-model-len 32768`, `--max-num-seqs 8`.
- Thinking off by default.

Port 8101 is an A/B slot: `Qwen3.6-35B-A3B-NVFP4` can be swapped in on the same port and
key. Retrieval and embeddings have already left Azure (ADR-0021). Generation is the
biggest per-query Foundry cost left on the request path.

## Decision

**Generation is one setting.** `Settings.generator` / `GENERATOR_*` works the way the
embedding setting does (#45):

- `GENERATOR_BASE_URL` set: answers come from `/v1/chat/completions` on that server.
  - `GENERATOR_MODEL` is required, because the A/B slot means the model is never hard-coded.
  - `GENERATOR_API_KEY` is the bearer key.
  - `GENERATOR_TEMPERATURE` is optional.
- Unset: the Foundry agent (`GENERATOR_AGENT_NAME`), exactly as before. That's the fallback.

`ChatCompletionsAgent` has the agent's `run(prompt)` shape, so `Generator` keeps its
timeout, retry and usage recording for both routes. The chat route:

- **System message:** the agent's instructions (`generate.GENERATOR_INSTRUCTIONS`).
- **User message:** the same grounding prompt as before.
- **Usage:** recorded through `record_openai_chat` under the model name.

**Its own client.** The generator builds an `AsyncOpenAI` for `GENERATOR_BASE_URL` alone. It
never reuses the embedding client or the Foundry chat client. Sharing one is how #49 sent
every planner call to the embedder.

**Sampling is left to the server unless set.** vLLM applies the model's
`generation_config`, which is Qwen's recommended non-thinking settings (temperature 0.7,
top_p 0.8, top_k 20, presence_penalty 1.5). `GENERATOR_TEMPERATURE` overrides only the
temperature. Thinking stays off: it costs latency on the demo's request path, and the
grounding prompt is a reading task.

**The planner and the situating context stay on Foundry for now.** Each has a reason:

- **Situating context.** It is cached per model (ADR-0005). Moving it would invalidate the
  cache, force a re-ingest, and change the corpus under the eval.
- **The agentic planner.** It decides what the agentic modes retrieve. Leaving it alone
  makes generation the only variable in the comparison against the Gemma run (#51).

Either can move later with its own setting and eval.

**Judges stay on Foundry DeepSeek** (homelab #244 covers moving them). The family split of
ADR-0009 still holds:

- Qwen generator (Alibaba).
- Claude online gate (Anthropic).
- DeepSeek offline judge.
- Gemma embedder (Google).

None of these judges its own family.

## Consequences

- **No Code Interpreter on the Spark route.** The agent could compute counts and
  aggregates; the chat route answers from the passages only. The global/multihop cohorts
  (ADR-0019) are where that would show. The full eval measures it.
- **One fewer Foundry asset at runtime.** `scripts/setup_agents.py` skips registration when
  `GENERATOR_BASE_URL` is set. The agent stays registered for the fallback.
- **Concurrency is the Spark's.** `--max-num-seqs 8` is shared by every caller of 8101.
  vLLM queues beyond it, so a busy Spark shows up as latency, not errors. The demo's
  `RAG_TIMEOUT_MS` has to cover generation at p95 plus the guardrail's retries.
- **Swapping the model in the A/B slot is env only**, and so is going back to Foundry.

## Outcome

In the full nine-mode eval against the Gemma run (`docs/eval-history.md`, 2026-11-20), quality held:
- **Faithfulness:** unchanged (pooled +0.012, p = 0.25).
- **Answer relevancy:** pooled −0.027 (p = 0.018).
- **Abstentions:** the graph modes abstain far less (`graphrag` 15 → 4, `graphrag_agentic`
  17 → 6, p ≤ 0.007).

## Sources

- vLLM OpenAI-compatible server, chat completions and `generation_config` defaults:
  https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- Qwen3 recommended sampling (non-thinking mode): the model card's "Best Practices".
- Issue #49: what sharing a client with the embedder does.
- ADR-0009 (family split), ADR-0005 (context cache), ADR-0020 (usage reporting),
  ADR-0021 (retrieval and embeddings on homelab).
