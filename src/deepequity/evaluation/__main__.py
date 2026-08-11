from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from deepequity.core.config import get_settings
from deepequity.core.db import close_pool
from deepequity.core.logging import configure_logging
from deepequity.evaluation.golden_queries import GOLDEN_QUERIES
from deepequity.evaluation.runner import ModeResult, run_evaluation

#where the numbers get written. keeping them in the repo is the point: a later change to
#chunk size or the model can be compared against this instead of against a memory of
#roughly how good it used to be.
RESULTS_PATH = Path("evaluation_baseline.json")


def _print_table(results: dict[str, ModeResult]) -> None:
    print()
    print("Overall, averaged across all queries")
    print(
        f"{'mode':<16} {'P@5':>7} {'R@5':>7} {'R vs max':>9} "
        f"{'MRR':>7} {'nDCG@5':>8} {'p50 ms':>8}"
    )
    print("-" * 66)
    for mode, result in results.items():
        s = result.summary()
        print(
            f"{mode:<16} {s['precision']:>7.3f} {s['recall']:>7.3f} "
            f"{s['recall_vs_ceiling']:>8.0%} {s['mrr']:>8.3f} "
            f"{s['ndcg']:>8.3f} {s['p50_ms']:>8.0f}"
        )
    ceiling = next(iter(results.values())).summary()["recall_ceiling"]
    print(
        f"\n('R vs max' is recall against the best achievable {ceiling:.3f}, "
        "since most questions have far more than 5 right answers)"
    )

    print()
    print("By question type (P@5)")
    categories = sorted({q.query_id.split("-")[0] for q in GOLDEN_QUERIES})
    header = "".join(f"{c:>10}" for c in categories)
    print(f"{'mode':<16}{header}")
    print("-" * (16 + 10 * len(categories)))
    for mode, result in results.items():
        row = result.by_category()
        cells = "".join(f"{row.get(c, {}).get('precision', 0):>10.3f}" for c in categories)
        print(f"{mode:<16}{cells}")


async def main() -> None:
    configure_logging(get_settings().log_level)
    results = await run_evaluation(k=5)

    _print_table(results)

    payload = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "queries": len(GOLDEN_QUERIES),
        "k": 5,
        "embedding_model": get_settings().embedding_model,
        "reranker_model": get_settings().reranker_model,
        "chunk_sizes": {
            "parent": get_settings().parent_chunk_chars,
            "child": get_settings().child_chunk_chars,
            "overlap": get_settings().child_chunk_overlap_chars,
        },
        "modes": {
            mode: {
                "overall": result.summary(),
                "by_category": result.by_category(),
                "per_query": [
                    {
                        "query_id": m.query_id,
                        "precision": m.precision_at_k,
                        "recall": m.recall_at_k,
                        "mrr": m.mrr,
                        "ndcg": m.ndcg_at_k,
                        "total_relevant": m.total_relevant,
                    }
                    for m in result.per_query
                ],
            }
            for mode, result in results.items()
        },
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwritten to {RESULTS_PATH}")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
