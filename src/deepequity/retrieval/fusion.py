from __future__ import annotations

from deepequity.core.config import get_settings
from deepequity.retrieval.models import RetrievedChunk


#merges several ranked lists into one, using Reciprocal Rank Fusion.
#
#the problem it solves: dense search scores are cosine similarities (roughly 0.6 to 0.9
#in practice) and keyword scores are ts_rank values (often 0.01 to 0.5). those numbers
#mean completely different things, so adding or averaging them is meaningless, whichever
#method happens to produce bigger numbers would simply win every time.
#
#RRF throws the scores away and uses only the position in each list. a chunk ranked 1st
#contributes 1/(k+1), ranked 2nd contributes 1/(k+2), and so on, summed across every
#list it appears in. so a passage both methods rank highly beats one that a single
#method loves, which is exactly the behaviour we want from a hybrid.
#
#k defaults to 60, from the original paper. it flattens the curve near the top, so the
#difference between rank 1 and rank 2 isn't dramatic and one over-confident method can't
#run away with the result.
def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], k: int | None = None
) -> list[RetrievedChunk]:
    if k is None:
        k = get_settings().rrf_k

    scores: dict[int, float] = {}
    #keep the first version of each chunk we see, they're identical apart from the score
    #and the method label
    chunks: dict[int, RetrievedChunk] = {}
    #track which methods found each chunk, useful when debugging why something ranked
    #where it did
    methods: dict[int, list[str]] = {}

    for ranking in rankings:
        for position, chunk in enumerate(ranking):
            key = chunk.child_chunk_id
            #position is zero-based, ranks start at 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position + 1)
            chunks.setdefault(key, chunk)
            methods.setdefault(key, []).append(chunk.retrieval_method)

    fused = [
        chunks[key].model_copy(
            update={
                "score": score,
                "retrieval_method": "+".join(sorted(set(methods[key]))),
            }
        )
        for key, score in scores.items()
    ]

    fused.sort(key=lambda chunk: chunk.score, reverse=True)
    return fused
