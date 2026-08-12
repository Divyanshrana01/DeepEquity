from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime

from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger
from deepequity.core.redis_client import aw, get_redis
from deepequity.ingestion.embeddings import embed_texts

logger = get_logger("deepequity.agents.cache")

#one json blob per cached answer, plus a sorted set per namespace holding the ids so we
#can find them again. the sorted set is scored by timestamp, which gives us "newest
#first" for the scan and a cheap way to trim the oldest entries out.
_ENTRY = "llmcache:entry:{namespace}:{digest}"
_INDEX = "llmcache:index:{namespace}"
_STATS = "llmcache:stats"


#what came back when a lookup succeeded. the tokens and cost are what the original call
#used, carried through so we can report what the hit saved rather than just that it
#happened. a hit that saves nothing is not worth the complexity, and this is how we check.
@dataclass
class CacheHit:
    raw: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    similarity: float


#groups cache entries so unrelated calls can never match each other.
#
#the role, the model, the schema and the system prompt all go in. two agents can be asked
#very similar questions and still need different answers, and an answer generated for one
#schema simply will not parse as another. keeping them apart in the key is cheaper and far
#more reliable than hoping the similarity score notices the difference.
#
#the system prompt is in there as a hash, which quietly does something important: editing
#a prompt invalidates everything cached under the old one. otherwise a Phase 6 prompt
#experiment would keep being served answers written by the prompt it just replaced, and
#the measured "improvement" would be noise.
def namespace_for(role: str, model: str, schema_name: str, system_prompt: str) -> str:
    return f"{role}:{model}:{schema_name}:{_digest(system_prompt)[:12]}"


#cosine similarity between two vectors. the embedding model already returns normalised
#vectors so the denominator is usually 1, but dividing properly costs nothing and means
#this still behaves if the model is ever swapped for one that doesn't normalise.
def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


#looks for an answer we already have to a near enough question.
#
#the threshold is the whole design here and it is set deliberately high. a bull prompt and
#a bear prompt about the same company, built from the same evidence, sit around 0.95
#similar, and serving one as the other would be a silent, confident, wrong answer. we would
#rather miss a real hit than return the wrong note, so the bar is set above where genuinely
#different prompts land, not at where similar ones do.
#
#the scan is capped and runs newest first. brute force over a few hundred short vectors is
#roughly a millisecond, which is nothing against an llm call, and it avoids taking a hard
#dependency on a redis vector index that we would then have to keep working.
#
#exclude_origin is what stops a run reading its own entries, and it is not a nicety. the
#second debate round's prompt is the first round's plus a "here is what you said last
#time" block, which lands above any workable threshold, so without this the second round
#gets served the first round's thesis word for word. the round appears to happen, costs
#nothing, and changes nothing. reusing work from an earlier run is the entire point of the
#cache. reusing your own output inside one run is a loop.
async def lookup(
    namespace: str, prompt: str, exclude_origin: str | None = None
) -> CacheHit | None:
    settings = get_settings()
    vectors = await embed_texts([prompt])
    if not vectors:
        return None
    query_vector = vectors[0]

    redis = get_redis()
    index_key = _INDEX.format(namespace=namespace)
    digests: list[str] = await aw(
        redis.zrevrange(index_key, 0, settings.semantic_cache_scan_limit - 1)
    )
    if not digests:
        await _bump("misses")
        return None

    keys = [_ENTRY.format(namespace=namespace, digest=d) for d in digests]
    blobs: list[str | None] = await aw(redis.mget(keys))

    best: CacheHit | None = None
    expired: list[str] = []

    for digest, blob in zip(digests, blobs, strict=True):
        #the entry key has its own ttl, so it can disappear while its id is still sitting
        #in the index. clean those up as we find them, otherwise the index grows forever
        #and every scan wastes its budget on ids pointing at nothing.
        if blob is None:
            expired.append(digest)
            continue

        entry = json.loads(blob)
        if exclude_origin is not None and entry.get("origin") == exclude_origin:
            continue

        score = cosine(query_vector, entry["vector"])
        if score >= settings.semantic_cache_threshold and (
            best is None or score > best.similarity
        ):
            best = CacheHit(
                raw=entry["raw"],
                model=entry["model"],
                prompt_tokens=int(entry["prompt_tokens"]),
                completion_tokens=int(entry["completion_tokens"]),
                similarity=round(score, 4),
            )

    if expired:
        await aw(redis.zrem(index_key, *expired))

    if best is None:
        await _bump("misses")
        return None

    await _bump("hits")
    await _bump("tokens_saved", best.prompt_tokens + best.completion_tokens)
    logger.info(
        "semantic_cache_hit",
        namespace=namespace,
        similarity=best.similarity,
        tokens_saved=best.prompt_tokens + best.completion_tokens,
    )
    return best


#files an answer away for next time.
#
#the prompt itself is stored truncated, purely so a cache entry can be looked at and
#understood later. the vector is what the matching actually uses.
async def store(
    namespace: str,
    prompt: str,
    raw: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    origin: str | None = None,
) -> None:
    settings = get_settings()
    vectors = await embed_texts([prompt])
    if not vectors:
        return

    digest = _digest(prompt)
    payload = json.dumps(
        {
            "vector": vectors[0],
            "prompt": prompt[:500],
            "raw": raw,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            #which run wrote this. read back on lookup so a run can be stopped from
            #matching its own earlier answers, see the note there.
            "origin": origin,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )

    redis = get_redis()
    entry_key = _ENTRY.format(namespace=namespace, digest=digest)
    index_key = _INDEX.format(namespace=namespace)

    await aw(redis.set(entry_key, payload, ex=settings.semantic_cache_ttl_seconds))
    await aw(redis.zadd(index_key, {digest: datetime.now(UTC).timestamp()}))
    #keep only the newest N ids. without this the index outlives the entries it points at
    #and the scan spends its whole budget on dead ids while real hits sit further down.
    await aw(redis.zremrangebyrank(index_key, 0, -settings.semantic_cache_max_entries - 1))
    #the index has no natural expiry of its own, so give it one well past the entries'
    await aw(redis.expire(index_key, settings.semantic_cache_ttl_seconds * 2))


async def _bump(field: str, amount: int = 1) -> None:
    try:
        await aw(get_redis().hincrby(_STATS, field, amount))
    except Exception as exc:  # noqa: BLE001 - stats are never worth failing a run over
        logger.warning("cache_stats_failed", field=field, error=str(exc))


#the hit rate, which is the number the plan asks for. reported alongside the raw counts
#because a 100% hit rate off two requests means nothing and the denominator is the only
#way to see that.
async def stats() -> dict[str, float | int]:
    raw = await aw(get_redis().hgetall(_STATS))
    hits = int(raw.get("hits", 0))
    misses = int(raw.get("misses", 0))
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "lookups": total,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "tokens_saved": int(raw.get("tokens_saved", 0)),
    }


async def reset_stats() -> None:
    await aw(get_redis().delete(_STATS))
