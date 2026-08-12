from __future__ import annotations

from dataclasses import dataclass


#what a million tokens costs on each model, in dollars. these are Groq's published
#pay-as-you-go rates.
#
#we run on the free tier, so none of this is actually billed. computing it anyway is the
#point: "that run would have cost $0.004" is a real answer to how the system scales, and
#a raw token count isn't, because input and output are priced differently and a small
#model and a big one produce identical looking counts for very different money.
@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


PRICES: dict[str, ModelPrice] = {
    "openai/gpt-oss-20b": ModelPrice(input_per_million=0.10, output_per_million=0.50),
    "openai/gpt-oss-120b": ModelPrice(input_per_million=0.15, output_per_million=0.75),
    "llama-3.1-8b-instant": ModelPrice(input_per_million=0.05, output_per_million=0.08),
    "llama-3.3-70b-versatile": ModelPrice(input_per_million=0.59, output_per_million=0.79),
}

#used when a model isn't in the table above. priced at the most expensive thing we know
#about rather than at zero, because a silent zero makes an unrecognised model look free,
#and that is the one case where the cost report would actively mislead someone.
_UNKNOWN_MODEL_PRICE = ModelPrice(input_per_million=0.15, output_per_million=0.75)


#looks up what a model charges, falling back rather than raising. a missing price should
#never be the reason a research run dies.
def price_for(model: str) -> ModelPrice:
    return PRICES.get(model, _UNKNOWN_MODEL_PRICE)


#turns a token count into dollars.
#
#rounded to six places because a single cheap call lands around $0.0004, and rounding to
#the usual two would report every call as costing nothing at all.
def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = price_for(model)
    dollars = (
        prompt_tokens / 1_000_000 * price.input_per_million
        + completion_tokens / 1_000_000 * price.output_per_million
    )
    return round(dollars, 6)
