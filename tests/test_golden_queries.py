from __future__ import annotations

import re

from deepequity.evaluation.golden_queries import GOLDEN_QUERIES, GoldenQuery


def test_semantic_rules_do_not_just_restate_the_question() -> None:
    # The guard that keeps the semantic half of the evaluation honest. These questions
    # exist to test whether search can find the right passage when the wording is
    # nothing like the document's. If the relevance rule reused the question's own
    # words, keyword search would score well by construction and we'd be measuring
    # string overlap instead of understanding.
    #
    # This deliberately does NOT apply to the factual and keyword questions. Asking for
    # "iPhone net sales" is supposed to match text containing "iPhone", that's the whole
    # point of those cases. Their overlap is honest, and the per-category reporting in
    # the runner is what stops it flattering the keyword numbers unnoticed.
    for golden in GOLDEN_QUERIES:
        if not golden.query_id.startswith("sem-"):
            continue

        query_words = set(re.findall(r"[a-z]{4,}", golden.query.lower()))
        for pattern in golden.must_match:
            #strip the regex syntax down to the bare words it looks for
            literal_words = set(re.findall(r"[a-z]{4,}", pattern.lower()))
            overlap = literal_words & query_words
            assert not overlap, (
                f"{golden.query_id}: semantic relevance rule reuses the question's own "
                f"words {overlap}, which would make the measurement circular"
            )


def test_every_query_has_an_intent_written_down() -> None:
    # The intent is what lets someone argue with a rule later instead of taking it on
    # faith, so a rule without one is not reviewable.
    for golden in GOLDEN_QUERIES:
        assert golden.intent, f"{golden.query_id} has no stated intent"


def test_query_ids_are_unique() -> None:
    ids = [golden.query_id for golden in GOLDEN_QUERIES]
    assert len(ids) == len(set(ids))


def test_the_set_covers_a_mix_of_question_styles() -> None:
    # A set of only semantic questions would flatter vector search, a set of only exact
    # terms would flatter keyword search. The mix is what makes the comparison fair.
    prefixes = {golden.query_id.split("-")[0] for golden in GOLDEN_QUERIES}
    assert {"sem", "fact", "kw", "multi"} <= prefixes


def test_relevance_judgement_works_both_ways() -> None:
    rule = GoldenQuery(
        query_id="t",
        query="anything",
        intent="test",
        must_match=[r"suppl(y|ier)", r"manufactur"],
    )

    assert rule.is_relevant("Our supplier and manufacturing partners are concentrated")
    # all-mode needs both terms, not just one
    assert not rule.is_relevant("Our supplier relationships are strong")


def test_any_mode_needs_only_one_term() -> None:
    rule = GoldenQuery(
        query_id="t",
        query="anything",
        intent="test",
        must_match=[r"dividend", r"repurchase"],
        match_mode="any",
    )

    assert rule.is_relevant("the board declared a dividend")
    assert rule.is_relevant("the share repurchase program continued")
    assert not rule.is_relevant("revenue grew in the period")


def test_exclude_overrides_a_match() -> None:
    rule = GoldenQuery(
        query_id="t",
        query="anything",
        intent="test",
        must_match=[r"suppl"],
        exclude=["water supply"],
    )

    assert rule.is_relevant("component supplier concentration")
    assert not rule.is_relevant("water supply in our facilities")


def test_judgement_ignores_capitalisation() -> None:
    rule = GoldenQuery(
        query_id="t", query="anything", intent="test", must_match=[r"effective tax rate"]
    )

    assert rule.is_relevant("The Effective Tax Rate was 15.6%")
