"""Maximal marginal relevance: diversity against near-duplicates."""

from __future__ import annotations



from app.core.retriever import _mmr
from app.core.vector_store import SearchHit


def _mmr_fixture() -> list[SearchHit]:
    # `a` and `b` are near-duplicates of each other; `c` is less relevant but
    # covers different material entirely.
    return [
        SearchHit("a", "a", 0.90, {"title": "A"}, embedding=[1.0, 0.0, 0.0]),
        SearchHit("b", "b", 0.89, {"title": "A"}, embedding=[0.99, 0.14, 0.0]),
        SearchHit("c", "c", 0.45, {"title": "B"}, embedding=[0.0, 0.0, 1.0]),
    ]


def test_mmr_prefers_diverse_chunks_over_near_duplicates():
    selected = _mmr(_mmr_fixture(), k=2, lambda_mult=0.5)

    assert {h.id for h in selected} == {"a", "c"}


def test_mmr_with_lambda_one_is_plain_relevance_ranking():
    selected = _mmr(_mmr_fixture(), k=2, lambda_mult=1.0)

    assert {h.id for h in selected} == {"a", "b"}


def test_mmr_relevance_honours_boosted_scores():
    # A lead-section chunk whose stored score was boosted upstream must be able
    # to win a slot on that boosted score, not its raw similarity.
    hits = _mmr_fixture()
    hits[2].score = 0.95  # as if intro_boost promoted it past the duplicates

    selected = _mmr(hits, k=2, lambda_mult=1.0)

    assert {h.id for h in selected} == {"a", "c"}


def test_mmr_passes_through_when_candidates_fit():
    hits = [SearchHit("a", "a", 0.9, {}, embedding=[1.0, 0.0])]

    assert _mmr(hits, k=5, lambda_mult=0.5) == hits
