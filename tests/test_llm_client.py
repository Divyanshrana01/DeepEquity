from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

import deepequity.agents.cache as cache
import deepequity.agents.llm as llm
from deepequity.agents.routing import AgentRole
from deepequity.agents.schemas import Thesis
from deepequity.core.config import get_settings


class Inner(BaseModel):
    value: str


class Outer(BaseModel):
    name: str
    inner: Inner
    items: list[Inner]


# --- schema conversion ---------------------------------------------------------------


def test_nested_models_are_inlined() -> None:
    # Pydantic emits $defs and $ref for nested models, but strict structured output can't
    # follow references. If these are left in, every call fails at the API with an
    # unhelpful error, so they have to be flattened before sending.
    schema = llm._to_strict_schema(Outer)

    assert "$defs" not in schema
    assert "$ref" not in str(schema)
    assert schema["properties"]["inner"]["properties"]["value"]["type"] == "string"


def test_additional_properties_is_forbidden_everywhere() -> None:
    # Strict mode rejects a schema that doesn't say this, including on nested objects.
    schema = llm._to_strict_schema(Outer)

    assert schema["additionalProperties"] is False
    assert schema["properties"]["inner"]["additionalProperties"] is False
    assert schema["properties"]["items"]["items"]["additionalProperties"] is False


def test_every_property_is_marked_required() -> None:
    # Strict mode wants all keys listed, optional-by-omission isn't allowed.
    schema = llm._to_strict_schema(Outer)

    assert set(schema["required"]) == {"name", "inner", "items"}


def test_a_real_agent_schema_converts_cleanly() -> None:
    # Thesis nests Claim which nests Citation, the deepest structure the agents use.
    schema = llm._to_strict_schema(Thesis)

    assert "$ref" not in str(schema)
    claim = schema["properties"]["claims"]["items"]
    assert claim["properties"]["citations"]["items"]["additionalProperties"] is False


# --- call behaviour ------------------------------------------------------------------


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeUsage:
    def __init__(self, prompt: int = 10, completion: int = 5) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


# Stands in for the Groq client, so none of these tests spend money or need a key.
class FakeCompletions:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(self.replies[min(len(self.calls) - 1, len(self.replies) - 1)])


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.completions = FakeCompletions(replies)
        self.chat = self


@pytest.fixture(autouse=True)
def _settings() -> None:
    get_settings.cache_clear()


async def test_valid_output_is_parsed_and_usage_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(['{"value": "hello"}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)

    result = await llm.complete_structured("sys", "user", Inner)

    assert result.parsed.value == "hello"
    # tokens are tracked from the first call, this is what feeds cost-per-run later
    assert result.usage.total == 15


async def test_bad_output_is_retried_with_the_error_fed_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Telling the model what was wrong with its output works much better than asking the
    # same question again and hoping for a different answer.
    client = FakeClient(["not json at all", '{"value": "fixed"}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)

    result = await llm.complete_structured("sys", "user", Inner)

    assert result.parsed.value == "fixed"
    assert len(client.completions.calls) == 2

    # the retry must actually carry the correction, not just repeat the original ask
    retry_messages = client.completions.calls[1]["messages"]
    assert any("did not match the required schema" in m["content"] for m in retry_messages)


async def test_giving_up_raises_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    get_settings.cache_clear()
    client = FakeClient(["still not json", "nope"])
    monkeypatch.setattr(llm, "get_client", lambda: client)

    with pytest.raises(llm.StructuredOutputError, match="could not produce valid"):
        await llm.complete_structured("sys", "user", Inner)

    assert len(client.completions.calls) == 2


async def test_strict_schema_is_actually_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(['{"value": "x"}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)

    await llm.complete_structured("sys", "user", Inner)

    response_format = client.completions.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


# --- retry timing and server-side rejections -----------------------------------------


def test_the_wait_is_read_from_the_rate_limit_message() -> None:
    # Groq says how long to wait. Retrying immediately, which is what we did before, is
    # guaranteed to fail again and eats more of the allowance on the way.
    exc = Exception(
        "Error code: 429 - Rate limit reached ... Please try again in 12.495s. Need more"
    )

    match = llm._RETRY_AFTER.search(str(exc))

    assert match is not None
    assert float(match.group(1)) == 12.495


async def test_a_rate_limit_waits_before_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    async def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", record_sleep)

    await llm._sleep_before_retry(Exception("please try again in 3.5s"), attempt=1)

    # the server's number plus a small margin, landing exactly on the boundary tends to
    # fail again
    assert slept == [4.5]


async def test_without_a_stated_wait_it_backs_off(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    async def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", record_sleep)

    await llm._sleep_before_retry(Exception("connection reset"), attempt=1)
    await llm._sleep_before_retry(Exception("connection reset"), attempt=3)

    assert slept[0] < slept[1]  # grows with each attempt
    assert slept[1] <= 30.0  # but is capped


def test_the_useful_part_of_a_rejection_is_extracted() -> None:
    # The raw message carries the entire failed generation, thousands of characters of
    # json we do not want to paste back into the next prompt.
    exc = Exception(
        "Error code: 400 - Generated JSON does not match. Error: "
        "jsonschema: '/claims/6' does not validate: expected object, but got string"
    )

    hint = llm._failed_generation_hint(exc)

    assert hint.startswith("jsonschema: '/claims/6'")
    assert len(hint) <= 300


async def test_missing_api_key_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    # A confusing auth error from the SDK is much worse than being told plainly.
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    llm._client = None

    with pytest.raises(RuntimeError, match="console.groq.com"):
        llm.get_client()


# --- cost and cache --------------------------------------------------------------------


async def test_a_real_call_is_priced_and_labelled(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(['{"value": "x"}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)

    response = await llm.complete_structured(
        "sys", "user", Inner, role=AgentRole.SYNTHESIS
    )
    record = response.cost_record()

    assert response.cached is False
    assert record.agent == "synthesis"
    assert record.cost_usd > 0
    assert record.saved_usd == 0.0
    # the role picks the model, no caller names one
    assert response.model == get_settings().llm_strong_model


async def test_a_cache_hit_costs_nothing_and_skips_the_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This is the whole point of the cache and the only thing worth asserting about it
    # here: the api is not called, and the run is not charged, while the tokens the
    # original answer used are still reported so the saving can be measured.
    client = FakeClient(['{"value": "x"}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm, "is_cacheable", lambda role: True)

    async def fake_lookup(*_args: Any, **_kwargs: Any) -> cache.CacheHit:
        return cache.CacheHit(
            raw='{"value": "from cache"}',
            model="openai/gpt-oss-20b",
            prompt_tokens=5000,
            completion_tokens=800,
            similarity=0.99,
        )

    async def unreached_store(**_kwargs: Any) -> None:
        raise AssertionError("a cache hit must not write the entry back")

    monkeypatch.setattr(cache, "lookup", fake_lookup)
    monkeypatch.setattr(cache, "store", unreached_store)

    response = await llm.complete_structured("sys", "user", Inner, role=AgentRole.BULL)
    record = response.cost_record()

    assert response.parsed.value == "from cache"
    assert client.completions.calls == []
    assert record.cost_usd == 0.0
    assert record.saved_usd > 0
    assert record.total_tokens == 5800


async def test_a_stale_cache_entry_becomes_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    # A cached blob outlives the schema that produced it. It should quietly fall through
    # to a real call rather than throwing at an agent that has no idea a cache exists.
    client = FakeClient(['{"value": "fresh"}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm, "is_cacheable", lambda role: True)

    async def stale_lookup(*_args: Any, **_kwargs: Any) -> cache.CacheHit:
        return cache.CacheHit(
            raw='{"wrong_field": 1}',
            model="openai/gpt-oss-20b",
            prompt_tokens=10,
            completion_tokens=10,
            similarity=1.0,
        )

    async def noop_store(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(cache, "lookup", stale_lookup)
    monkeypatch.setattr(cache, "store", noop_store)

    response = await llm.complete_structured("sys", "user", Inner, role=AgentRole.BULL)

    assert response.parsed.value == "fresh"
    assert response.cached is False


async def test_a_broken_cache_does_not_break_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Redis being down should cost us the saving, not the research.
    client = FakeClient(['{"value": "fresh"}'])
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm, "is_cacheable", lambda role: True)

    async def broken(*_args: Any, **_kwargs: Any):
        raise RuntimeError("redis is down")

    monkeypatch.setattr(cache, "lookup", broken)
    monkeypatch.setattr(cache, "store", broken)

    response = await llm.complete_structured("sys", "user", Inner, role=AgentRole.BULL)

    assert response.parsed.value == "fresh"
