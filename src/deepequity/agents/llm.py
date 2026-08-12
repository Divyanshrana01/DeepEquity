from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from groq import APIConnectionError, APIStatusError, AsyncGroq, RateLimitError
from pydantic import BaseModel, ValidationError

from deepequity.agents import cache
from deepequity.agents.routing import AgentRole, is_cacheable, model_for
from deepequity.agents.run_context import current_run_id
from deepequity.agents.schemas import AgentCost
from deepequity.core.config import get_settings
from deepequity.core.costs import cost_usd
from deepequity.core.logging import get_logger

logger = get_logger("deepequity.agents.llm")

T = TypeVar("T", bound=BaseModel)

_client: AsyncGroq | None = None


#raised when the model couldn't produce output matching the schema after every retry.
#separate from a network problem because the two want different handling: one is worth
#trying again later, the other means the prompt or the schema needs work.
class StructuredOutputError(Exception):
    pass


#the request exceeded what the model will accept in one go. worth its own type because
#the fix is to send less, not to try again, and the difference matters to the caller.
class PromptTooLargeError(Exception):
    pass


#what a call cost us. tracked per call so a research run can add up what it spent, which
#is the groundwork for the cost-per-run number the project promises to report.
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse[TModel: BaseModel]:
    parsed: TModel
    usage: Usage
    model: str
    #no default on purpose. a cost record labelled with the wrong agent is worse than no
    #breakdown at all, because it quietly points the blame for a big bill at the wrong
    #part of the graph. making it required means it can't be forgotten.
    role: AgentRole
    #served from the semantic cache rather than the api. the tokens are still reported,
    #because what the call would have cost is exactly the interesting number, but the
    #money is zero and the two have to be told apart or the saving becomes invisible.
    cached: bool = False
    similarity: float | None = None

    #what this call actually cost. a cache hit is free, so it prices at zero regardless of
    #how many tokens the original answer took.
    @property
    def cost_usd(self) -> float:
        if self.cached:
            return 0.0
        return cost_usd(self.model, self.usage.prompt_tokens, self.usage.completion_tokens)

    #one line for the run's cost breakdown, so a caller can see which agent spent what
    #without adding up log lines by hand
    def cost_record(self) -> AgentCost:
        return AgentCost(
            agent=self.role.value,
            model=self.model,
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
            cost_usd=self.cost_usd,
            cached=self.cached,
            #what we would have paid without the cache. only meaningful on a hit, and it
            #is the number that answers "did the cache actually save anything".
            saved_usd=(
                cost_usd(self.model, self.usage.prompt_tokens, self.usage.completion_tokens)
                if self.cached
                else 0.0
            ),
        )


def get_client() -> AsyncGroq:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set, the agents can't run without it. "
                "Get a free key at https://console.groq.com/keys and put it in .env"
            )
        _client = AsyncGroq(
            api_key=settings.groq_api_key, timeout=settings.llm_timeout_seconds
        )
    return _client


#groq tells us exactly how long to wait in the 429 message, e.g. "Please try again in
#12.495s". using that beats guessing, and retrying immediately (which is what we did
#before) is guaranteed to fail again and eat more of the allowance on the way.
_RETRY_AFTER = re.compile(r"try again in ([\d.]+)s")


#waits before the next attempt. honours the server's own advice when it gives any,
#otherwise backs off exponentially.
async def _sleep_before_retry(exc: Exception, attempt: int) -> None:
    match = _RETRY_AFTER.search(str(exc))
    if match:
        #a small margin on top, landing exactly on the boundary tends to fail again
        delay = float(match.group(1)) + 1.0
    else:
        delay = min(2.0**attempt, 30.0)

    logger.info("llm_retry_waiting", seconds=round(delay, 1), attempt=attempt)
    await asyncio.sleep(delay)


#pulls the useful part out of a server-side schema rejection. the full message carries the
#entire failed generation, which can be thousands of characters of json we don't want to
#paste back into the next prompt.
def _failed_generation_hint(exc: Exception) -> str:
    text = str(exc)
    marker = "Error: "
    if marker in text:
        return text.split(marker, 1)[1][:300]
    return text[:300]


#pydantic writes json schema with $defs and $ref for nested models, but strict structured
#output wants everything inlined, and it rejects a schema that doesn't forbid extra
#properties. this flattens the references and adds the missing bits.
def _to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                #"#/$defs/Claim" -> the Claim definition itself
                name = node["$ref"].split("/")[-1]
                return inline(definitions[name])
            resolved = {key: inline(value) for key, value in node.items()}
            if resolved.get("type") == "object":
                #strict mode requires this, and requires every property to be listed as
                #required, optional fields aren't allowed to just be absent
                resolved["additionalProperties"] = False
                if "properties" in resolved:
                    resolved["required"] = list(resolved["properties"].keys())
            return resolved
        if isinstance(node, list):
            return [inline(item) for item in node]
        return node

    return dict(inline(schema))


#checks the semantic cache and turns a hit back into a parsed response.
#
#a cached answer is still validated against the schema rather than trusted. it was valid
#when it went in, but the schema can change underneath it, and a stale blob that no longer
#parses should quietly become a cache miss instead of an exception thrown at an agent that
#has no idea a cache exists. cache trouble of any kind degrades to "just call the api".
async def _try_cache[TModel: BaseModel](
    namespace: str,
    user_prompt: str,
    schema: type[TModel],
    model: str,
    role: AgentRole,
) -> LLMResponse[TModel] | None:
    try:
        hit = await cache.lookup(
            namespace, user_prompt, exclude_origin=current_run_id()
        )
    except Exception as exc:  # noqa: BLE001 - a broken cache must never stop a run
        logger.warning("semantic_cache_lookup_failed", role=role.value, error=str(exc))
        return None

    if hit is None:
        return None

    try:
        parsed = schema.model_validate_json(hit.raw)
    except (ValidationError, json.JSONDecodeError):
        logger.warning("semantic_cache_entry_stale", namespace=namespace, schema=schema.__name__)
        return None

    logger.info(
        "llm_call_served_from_cache",
        role=role.value,
        model=hit.model,
        schema=schema.__name__,
        similarity=hit.similarity,
        tokens_saved=hit.prompt_tokens + hit.completion_tokens,
    )
    return LLMResponse(
        parsed=parsed,
        usage=Usage(
            prompt_tokens=hit.prompt_tokens, completion_tokens=hit.completion_tokens
        ),
        model=hit.model,
        role=role,
        cached=True,
        similarity=hit.similarity,
    )


#asks the model a question and insists on getting back something matching the schema.
#
#retries cover two different failures. a rate limit or a dropped connection is worth
#simply trying again. output that doesn't fit the schema is worth trying again too, but
#differently: we hand the model its own broken output and the error, because telling it
#what went wrong works far better than asking the same question again and hoping.
#
#before any of that it checks the semantic cache. a repeat run on the same ticker asks the
#planner a word for word identical question and the debate agents very nearly identical
#ones, so this is where a second run stops costing money.
async def complete_structured[TModel: BaseModel](
    system_prompt: str,
    user_prompt: str,
    schema: type[TModel],
    role: AgentRole = AgentRole.PLANNER,
    model: str | None = None,
    temperature: float | None = None,
    cache_variant: str = "",
) -> LLMResponse[TModel]:
    settings = get_settings()
    #the model comes from the routing table unless a caller names one, which is really
    #only tests and one-off experiments
    model = model or model_for(role)
    temperature = settings.llm_temperature if temperature is None else temperature

    #cache_variant lets a caller say "this call is a different kind of question" when the
    #wording alone doesn't make that obvious. the debate uses it to keep an opening thesis
    #apart from a revision, which read almost identically to a similarity score and are
    #not remotely the same job.
    namespace = cache.namespace_for(
        f"{role.value}{cache_variant}", model, schema.__name__, system_prompt
    )
    if is_cacheable(role):
        hit = await _try_cache(namespace, user_prompt, schema, model, role)
        if hit is not None:
            return hit

    client = get_client()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    usage = Usage()
    last_error = ""

    #the sdk types these tightly (a union of message param shapes, and a specific
    #response format object). we build plain dicts because we're assembling them
    #dynamically across retries, so they're cast at the call rather than fought with.
    response_format: Any = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__.lower(),
            "strict": True,
            "schema": _to_strict_schema(schema),
        },
    }

    for attempt in range(1, settings.llm_max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=settings.llm_max_output_tokens,
                response_format=response_format,
            )
        except APIStatusError as exc:
            #413 means the request itself is too big for the model's per-minute token
            #allowance. sending the identical payload again gets the identical rejection,
            #so retrying just burns time and eats further into the rate limit. fail
            #immediately and say what to do about it.
            if exc.status_code == 413:
                raise PromptTooLargeError(
                    f"request too large for {model}. Reduce max_evidence_chunks or "
                    f"max_evidence_chars_per_chunk. Original error: {exc}"
                ) from exc

            #the server rejected the json against our schema. this is the same class of
            #problem as a local validation failure, so it wants the same handling: show
            #the model what it produced and what was wrong. it was previously falling
            #through to the generic branch, which retried the identical prompt and got
            #the identical broken output.
            if exc.status_code == 400 and "json_validate_failed" in str(exc):
                last_error = str(exc)
                logger.warning(
                    "llm_output_rejected_by_server", attempt=attempt, model=model
                )
                if attempt == settings.llm_max_retries:
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was rejected because it did not "
                            f"match the schema: {_failed_generation_hint(exc)}. Return "
                            "valid JSON matching the schema exactly. Every element of "
                            "every array must be an object of the right type, never a "
                            "bare string."
                        ),
                    }
                )
                continue

            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("llm_call_failed", attempt=attempt, model=model, error=str(exc))
            if attempt == settings.llm_max_retries:
                raise
            await _sleep_before_retry(exc, attempt)
            continue
        except (RateLimitError, APIConnectionError) as exc:
            #the network or the service, not the prompt. nothing to fix in the message,
            #so wait and go round again.
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("llm_call_failed", attempt=attempt, model=model, error=str(exc))
            if attempt == settings.llm_max_retries:
                raise
            await _sleep_before_retry(exc, attempt)
            continue

        if response.usage:
            usage.prompt_tokens += response.usage.prompt_tokens
            usage.completion_tokens += response.usage.completion_tokens

        raw = response.choices[0].message.content or ""

        try:
            parsed = schema.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            logger.warning(
                "llm_output_invalid", attempt=attempt, model=model, error=str(exc)[:300]
            )
            if attempt == settings.llm_max_retries:
                break
            #show the model what it produced and what was wrong with it, so the next
            #attempt is a correction rather than a re-roll
            messages.append({"role": "assistant", "content": raw[:4000]})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response did not match the required schema. "
                        f"The error was: {str(exc)[:800]}. "
                        "Return corrected JSON matching the schema exactly, nothing else."
                    ),
                }
            )
            continue

        #only cached once it has parsed. storing the raw text before validating would let
        #a broken generation be served back to every future run that asks a similar
        #question, which turns one bad answer into a permanent one.
        if is_cacheable(role):
            try:
                await cache.store(
                    namespace=namespace,
                    prompt=user_prompt,
                    raw=raw,
                    model=model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    origin=current_run_id(),
                )
            except Exception as exc:  # noqa: BLE001 - failing to cache is not failing
                logger.warning("semantic_cache_store_failed", role=role.value, error=str(exc))

        spent = cost_usd(model, usage.prompt_tokens, usage.completion_tokens)
        logger.info(
            "llm_call_complete",
            role=role.value,
            model=model,
            schema=schema.__name__,
            attempt=attempt,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=spent,
        )
        return LLMResponse(parsed=parsed, usage=usage, model=model, role=role)

    raise StructuredOutputError(
        f"{model} could not produce valid {schema.__name__} after "
        f"{settings.llm_max_retries} attempts. Last error: {last_error[:500]}"
    )
