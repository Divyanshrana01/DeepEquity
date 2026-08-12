import math

import fakeredis.aioredis
import pytest

from deepequity.agents import cache
from deepequity.core.config import Settings

#fixed vectors keyed by prompt text, so a test can say exactly how alike two prompts are
#instead of hoping the real embedding model puts them where the test needs them
_VECTORS = {
    "what are the margins": [1.0, 0.0, 0.0],
    "what are the margins ": [1.0, 0.0, 0.0],
    #cosine 0.9 against the one above: clearly related, clearly not the same question
    "make the bear case": [0.9, math.sqrt(1 - 0.81), 0.0],
    "something else entirely": [0.0, 0.0, 1.0],
}


@pytest.fixture(autouse=True)
def stub_embeddings_and_redis(monkeypatch: pytest.MonkeyPatch) -> fakeredis.aioredis.FakeRedis:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        return [_VECTORS[text] for text in texts]

    monkeypatch.setattr(cache, "embed_texts", fake_embed)
    monkeypatch.setattr(cache, "get_redis", lambda: redis)
    return redis


def _namespace() -> str:
    return cache.namespace_for("bull", "openai/gpt-oss-20b", "Thesis", "system prompt v1")


async def test_stored_answer_comes_back_for_the_same_question() -> None:
    namespace = _namespace()
    await cache.store(
        namespace=namespace,
        prompt="what are the margins",
        raw='{"answer": 1}',
        model="openai/gpt-oss-20b",
        prompt_tokens=4000,
        completion_tokens=600,
    )

    hit = await cache.lookup(namespace, "what are the margins")

    assert hit is not None
    assert hit.raw == '{"answer": 1}'
    assert hit.prompt_tokens == 4000
    assert hit.similarity == pytest.approx(1.0)


#the one that matters. bull and bear prompts are built from the same evidence about the
#same company, so they sit very close together, and a loose threshold would hand the bear
#agent the bull's thesis. that failure is silent: valid json, right schema, wrong argument.
async def test_a_merely_similar_prompt_is_not_a_hit() -> None:
    namespace = _namespace()
    await cache.store(
        namespace=namespace,
        prompt="what are the margins",
        raw='{"answer": 1}',
        model="openai/gpt-oss-20b",
        prompt_tokens=100,
        completion_tokens=10,
    )

    #0.9 similar, well under the 0.97 default
    assert await cache.lookup(namespace, "make the bear case") is None


#the namespace carries the system prompt, so editing a prompt file retires everything
#cached under the old one. without this a prompt experiment would keep being served
#answers written by the prompt it just replaced.
async def test_changing_the_system_prompt_invalidates_the_cache() -> None:
    old = cache.namespace_for("bull", "openai/gpt-oss-20b", "Thesis", "system prompt v1")
    new = cache.namespace_for("bull", "openai/gpt-oss-20b", "Thesis", "system prompt v2")
    assert old != new

    await cache.store(
        namespace=old,
        prompt="what are the margins",
        raw='{"answer": 1}',
        model="openai/gpt-oss-20b",
        prompt_tokens=100,
        completion_tokens=10,
    )

    assert await cache.lookup(new, "what are the margins") is None


async def test_roles_and_schemas_do_not_share_entries() -> None:
    bull = cache.namespace_for("bull", "openai/gpt-oss-20b", "Thesis", "p")
    bear = cache.namespace_for("bear", "openai/gpt-oss-20b", "Thesis", "p")
    other_schema = cache.namespace_for("bull", "openai/gpt-oss-20b", "ResearchNote", "p")

    await cache.store(
        namespace=bull,
        prompt="what are the margins",
        raw='{"answer": 1}',
        model="openai/gpt-oss-20b",
        prompt_tokens=100,
        completion_tokens=10,
    )

    assert await cache.lookup(bear, "what are the margins") is None
    assert await cache.lookup(other_schema, "what are the margins") is None


async def test_hit_rate_counts_both_sides() -> None:
    await cache.reset_stats()
    namespace = _namespace()

    #a miss on an empty cache
    assert await cache.lookup(namespace, "what are the margins") is None

    await cache.store(
        namespace=namespace,
        prompt="what are the margins",
        raw='{"answer": 1}',
        model="openai/gpt-oss-20b",
        prompt_tokens=4000,
        completion_tokens=600,
    )
    assert await cache.lookup(namespace, "what are the margins") is not None
    assert await cache.lookup(namespace, "something else entirely") is None

    stats = await cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["lookups"] == 3
    assert stats["hit_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert stats["tokens_saved"] == 4600


#entries expire on their own ttl but their ids stay in the index. left alone the index
#fills with ids pointing at nothing and every scan wastes its budget on them, so a lookup
#that finds a dead id should clear it out on the way past.
async def test_expired_entries_are_pruned_from_the_index(
    stub_embeddings_and_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    namespace = _namespace()
    await cache.store(
        namespace=namespace,
        prompt="what are the margins",
        raw='{"answer": 1}',
        model="openai/gpt-oss-20b",
        prompt_tokens=100,
        completion_tokens=10,
    )

    index_key = cache._INDEX.format(namespace=namespace)
    assert await stub_embeddings_and_redis.zcard(index_key) == 1

    #simulate the entry's ttl running out while its id is still indexed
    digest = (await stub_embeddings_and_redis.zrange(index_key, 0, -1))[0]
    await stub_embeddings_and_redis.delete(
        cache._ENTRY.format(namespace=namespace, digest=digest)
    )

    assert await cache.lookup(namespace, "what are the margins") is None
    assert await stub_embeddings_and_redis.zcard(index_key) == 0


async def test_index_is_trimmed_to_the_cap(
    stub_embeddings_and_redis: fakeredis.aioredis.FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache, "get_settings", lambda: Settings(semantic_cache_max_entries=2))
    namespace = _namespace()

    for prompt in ("what are the margins", "make the bear case", "something else entirely"):
        await cache.store(
            namespace=namespace,
            prompt=prompt,
            raw="{}",
            model="openai/gpt-oss-20b",
            prompt_tokens=1,
            completion_tokens=1,
        )

    index_key = cache._INDEX.format(namespace=namespace)
    assert await stub_embeddings_and_redis.zcard(index_key) == 2


def test_cosine_handles_mismatched_lengths() -> None:
    #a stored vector from a different embedding model must score zero, not raise. this is
    #what stops a model swap from crashing every lookup against the old entries.
    assert cache.cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


#regression guard, and the reason this rule exists at all.
#
#the second debate round's prompt is the first round's plus a "here is what you said last
#time" block, which scores above any workable threshold. served the first round's answer,
#the second round appears to happen, costs nothing and changes nothing, which is the exact
#opposite of what a second round is for. a run may reuse an earlier run's work but never
#its own.
async def test_a_run_cannot_be_served_its_own_answer() -> None:
    namespace = _namespace()
    await cache.store(
        namespace=namespace,
        prompt="what are the margins",
        raw='{"answer": "round one"}',
        model="openai/gpt-oss-20b",
        prompt_tokens=100,
        completion_tokens=10,
        origin="run-a",
    )

    assert await cache.lookup(namespace, "what are the margins", exclude_origin="run-a") is None


async def test_a_later_run_still_gets_the_hit() -> None:
    #the exclusion has to be narrow. reusing an earlier run's work is the entire point,
    #and a rule that blocked that too would leave the cache doing nothing at all.
    namespace = _namespace()
    await cache.store(
        namespace=namespace,
        prompt="what are the margins",
        raw='{"answer": "round one"}',
        model="openai/gpt-oss-20b",
        prompt_tokens=100,
        completion_tokens=10,
        origin="run-a",
    )

    hit = await cache.lookup(namespace, "what are the margins", exclude_origin="run-b")

    assert hit is not None
    assert hit.raw == '{"answer": "round one"}'
