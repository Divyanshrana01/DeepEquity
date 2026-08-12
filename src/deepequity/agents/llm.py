from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from groq import APIConnectionError, APIStatusError, AsyncGroq, RateLimitError
from pydantic import BaseModel, ValidationError

from deepequity.core.config import get_settings
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


#asks the model a question and insists on getting back something matching the schema.
#
#retries cover two different failures. a rate limit or a dropped connection is worth
#simply trying again. output that doesn't fit the schema is worth trying again too, but
#differently: we hand the model its own broken output and the error, because telling it
#what went wrong works far better than asking the same question again and hoping.
async def complete_structured[TModel: BaseModel](
    system_prompt: str,
    user_prompt: str,
    schema: type[TModel],
    model: str | None = None,
    temperature: float | None = None,
) -> LLMResponse[TModel]:
    settings = get_settings()
    model = model or settings.llm_fast_model
    temperature = settings.llm_temperature if temperature is None else temperature
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

        logger.info(
            "llm_call_complete",
            model=model,
            schema=schema.__name__,
            attempt=attempt,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        return LLMResponse(parsed=parsed, usage=usage, model=model)

    raise StructuredOutputError(
        f"{model} could not produce valid {schema.__name__} after "
        f"{settings.llm_max_retries} attempts. Last error: {last_error[:500]}"
    )
