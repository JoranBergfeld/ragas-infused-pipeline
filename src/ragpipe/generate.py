from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ragpipe.models import Chunk
from ragpipe.usage import (
    STAGE_GENERATION,
    record_agent_response,
    record_openai_chat,
    record_tokens,
)

# Without an application-level timeout, a single stalled FoundryAgent call
# (request sent, ACKed, but no response — no keepalive on the socket) blocks the
# whole pipeline indefinitely; this is the eval-harness hang we observed. The
# generator legitimately takes minutes (reasoning + Code Interpreter), so the
# per-attempt budget is generous; a bounded retry recovers from a transient
# stall without unbounded waiting.
DEFAULT_GENERATE_TIMEOUT = 300.0
DEFAULT_GENERATE_MAX_RETRIES = 1


class _Agent(Protocol):
    async def run(self, prompt: str): ...


# The generator's standing instructions. The Foundry agent holds them server-side
# (scripts/setup_agents.py adds a Code Interpreter line); the chat route sends them
# as the system message on every call.
GENERATOR_INSTRUCTIONS = (
    "You are a Microsoft/Azure documentation assistant. Answer the user's question "
    "using only the provided context passages. Cite sources by their [n] index. "
    "If the context is insufficient, say so."
)


@dataclass(frozen=True)
class ChatReply:
    """One chat completion: the answer text plus the raw response for usage."""

    text: str
    response: Any


class ChatCompletionsAgent:
    """`run(prompt)` over an OpenAI-compatible /v1/chat/completions server (the
    Spark LLM, #52), shaped like the Foundry agent so `Generator` drives either.

    `client` is an `openai.AsyncOpenAI` built for the generator alone, never the
    embedding client (#49). No tools: the Code Interpreter is a Foundry-agent
    feature. `temperature` None leaves sampling to the server's defaults.
    """

    def __init__(
        self,
        client,
        model: str,
        *,
        instructions: str = GENERATOR_INSTRUCTIONS,
        temperature: float | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._instructions = instructions
        self._temperature = temperature

    async def run(self, prompt: str) -> ChatReply:
        options: dict[str, Any] = {}
        if self._temperature is not None:
            options["temperature"] = self._temperature
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._instructions},
                {"role": "user", "content": prompt},
            ],
            **options,
        )
        return ChatReply(text=response.choices[0].message.content or "", response=response)


def record_chat_reply(stage: str, deployment: str, reply: ChatReply) -> None:
    record_openai_chat(stage, deployment, reply.response)


CORRECTIVE_INSTRUCTION = (
    "Your previous answer (below) contained claims that could not be verified "
    "against the sources. Write a new answer using ONLY claims directly "
    "supported by the numbered sources. If the sources do not contain the "
    "answer, say you don't know.\n\n"
    "<previous_answer>\n{previous_answer}\n</previous_answer>\n\n"
)


def build_grounding_prompt(
    query: str, chunks: list[Chunk], previous_answer: str | None = None
) -> str:
    sources = "\n\n".join(
        f"[{i + 1}] ({c.url}) {c.content}" for i, c in enumerate(chunks)
    )
    corrective = (
        CORRECTIVE_INSTRUCTION.format(previous_answer=previous_answer)
        if previous_answer
        else ""
    )
    return (
        "Answer the question using ONLY the numbered sources below. "
        "Cite sources inline like [1]. If the sources do not contain the answer, "
        "say you don't know.\n\n"
        f"{corrective}"
        f"Sources:\n{sources}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


class Generator:
    def __init__(
        self,
        agent: _Agent,
        timeout: float = DEFAULT_GENERATE_TIMEOUT,
        max_retries: int = DEFAULT_GENERATE_MAX_RETRIES,
        deployment: str = "unknown",
        record_usage: Callable[[str, str, Any], None] = record_agent_response,
    ) -> None:
        self._agent = agent
        # How this agent's result reports usage: the Foundry agent's
        # AgentResponse, or record_chat_reply for ChatCompletionsAgent.
        self._record_usage = record_usage
        # Model deployment behind the agent, reported with each call's usage.
        self._deployment = deployment
        self._timeout = timeout
        self._max_retries = max_retries

    async def generate(
        self, query: str, chunks: list[Chunk], previous_answer: str | None = None
    ) -> str:
        prompt = build_grounding_prompt(query, chunks, previous_answer)
        last_exc: asyncio.TimeoutError | None = None
        for _ in range(self._max_retries + 1):
            try:
                # Use asyncio.timeout (a context manager) rather than
                # asyncio.wait_for. On Python 3.11 wait_for runs the awaited
                # coroutine as a child Task in a *copied* context, which breaks
                # agent-framework's eager ContextVar set/reset pair ("Token was
                # created in a different Context"; issue #5). asyncio.timeout
                # does not spawn a child Task, so set/reset stay balanced; the
                # TimeoutError semantics are identical.
                async with asyncio.timeout(self._timeout):
                    result = await self._agent.run(prompt)
                self._record_usage(STAGE_GENERATION, self._deployment, result)
                return result.text
            except asyncio.TimeoutError as exc:
                # The request was sent and may be billed, but no usage came back.
                record_tokens(
                    STAGE_GENERATION,
                    self._deployment,
                    reason="call timed out before a response",
                )
                last_exc = exc
        raise last_exc
