"""Holding an answer to the standard its own citations imply.

A sentence carrying `[2]` claims chunk 2 supports it. Nothing upstream checks
that: the marker is in range and the URL is real, so a conflated answer looks
perfectly sourced. The mismatch is between the *claim* and its *source*, and
this module is the only place that looks at both together.
"""

from __future__ import annotations

import re

from .vector_store import SearchHit

CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")


def cited_indices(answer: str, source_count: int) -> list[int]:
    """Citation numbers the model actually used, in order of first appearance."""
    seen: list[int] = []
    for match in CITATION_PATTERN.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= source_count and n not in seen:
            seen.append(n)
    return seen

# Words that carry no subject information, so their presence tells us nothing
# about whether a claim matches the source it cites.
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does for from had
    has have he her him his how however in into is it its more most not of on
    only or she some such than that the their them then there these they this
    those to was were what when where which who whom whose will with would you
    your also been over under between during about
    """.split()
)

_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9']+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LIST_PREFIX = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s*)")

# Fraction of a sentence's content words that must also appear in the chunk it
# cites. Deliberately permissive: this catches a citation pointing at unrelated
# material, and is not meant to police paraphrase or inference.
_MIN_CITATION_OVERLAP = 0.25

def _ordered_terms(text: str) -> list[str]:
    return [
        word.lower()
        for word in _WORD.findall(text)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    ]


def _content_terms(text: str) -> set[str]:
    return set(_ordered_terms(text))


def _term_present(term: str, chunk_terms: set[str]) -> bool:
    """Membership that tolerates the singular/plural split.

    Without it "language" misses a chunk that only ever says "languages", which
    would fail honest sentences far more often than dishonest ones.
    """
    if term in chunk_terms:
        return True
    stem = term.rstrip("s")
    return any(other.rstrip("s") == stem for other in chunk_terms)

# "X is a Y" asserts a category, and the category is precisely what a conflated
# answer gets wrong: *"Java is a programming language …"* cited to the island
# article. Bag-of-words overlap cannot see it — every other word in that
# sentence does come from the chunk.
#
# The pattern is deliberately narrow, because an earlier, looser version broke
# well-grounded answers at a 25% rate. It requires a determiner ("is **a**
# programming language"), which is what separates a category claim from a
# composition statement ("DNA is made of…") or a hedge ("was not a single
# route…") — both of which paraphrase far too freely to check this way. Only
# the first two predicate words are examined: that is the category itself,
# before the qualifiers start.
_DEFINITION = re.compile(
    r"^\s*[\"“']?[A-Z][\w'’\-]*(?:\s+[\w'’\-()]+){0,3}\s+(?:is|was|are|were)\s+"
    r"(?:a|an|the)\s+(?P<predicate>[^,.;:]{3,80})"
)
_NEGATION = re.compile(r"\b(?:not|no|never|rather than|instead of)\b", re.I)
_DEFINITION_TERMS = 2


def _definition_grounded(sentence: str, chunk_terms: set[str]) -> bool:
    match = _DEFINITION.match(sentence)
    if not match or _NEGATION.search(sentence):
        return True
    predicate = _ordered_terms(match.group("predicate"))[:_DEFINITION_TERMS]
    if len(predicate) < _DEFINITION_TERMS:
        return True
    return all(_term_present(term, chunk_terms) for term in predicate)

def _sentence_supported(sentence: str, hits: list[SearchHit]) -> tuple[bool, str]:
    """Whether a sentence's vocabulary matches the chunks it cites."""
    indices = cited_indices(sentence, len(hits))
    if not indices:
        # Uncited text is connective tissue, not a sourced claim. Leaving it
        # alone is the point: only citations get held to a citation's standard.
        return True, ""

    body = CITATION_PATTERN.sub("", sentence)
    terms = _content_terms(body)
    if not terms:
        return True, ""

    best = 0.0
    definition_ok = False
    for n in indices:
        hit = hits[n - 1]
        chunk_terms = _content_terms(
            f"{hit.title} {hit.metadata.get('section_path', '')} "
            f"{hit.metadata.get('body') or hit.text}"
        )
        best = max(best, len(terms & chunk_terms) / len(terms))
        # One cited chunk carrying the definition is enough to justify it.
        definition_ok = definition_ok or _definition_grounded(body, chunk_terms)

    if best < _MIN_CITATION_OVERLAP:
        return False, f"{best:.0%} term overlap with cited chunk"
    if not definition_ok:
        return False, "defining claim absent from cited chunk"
    return True, ""

def verify_citations(
    answer: str, hits: list[SearchHit]
) -> tuple[str, list[tuple[str, str]]]:
    """Drop sentences whose citation points at unrelated material.

    On an ambiguous query the generator occasionally welds two subjects
    together — *"Java is a programming language with four main spoken languages
    on the island of Java [2]"*, where chunk [2] is the island article and says
    nothing about a programming language. The URL is real and the marker is in
    range, so neither :func:`cited_indices` nor :func:`build_sources` can see
    the problem: the mismatch is between the **claim** and its **source**.

    Runs before :func:`build_sources`, so a dropped sentence also drops its
    citation from the source list rather than leaving evidence for a claim that
    is no longer there. An answer emptied by this check leaves the caller with
    no cited sources, which routes it to the ordinary not-covered path.

    Returns the cleaned answer and the (sentence, reason) pairs removed.
    """
    if not hits or not answer.strip():
        return answer, []

    rejected: list[tuple[str, str]] = []
    kept_lines: list[str] = []

    for line in answer.splitlines():
        if not line.strip():
            kept_lines.append(line)
            continue

        # Keep any bullet/number marker attached to the line rather than the
        # sentence, so removing a sentence never leaves a dangling "- ".
        match = _LIST_PREFIX.match(line)
        prefix = match.group(1) if match else ""
        body = line[len(prefix):]

        kept: list[str] = []
        for sentence in _SENTENCE_SPLIT.split(body):
            if not sentence.strip():
                continue
            supported, reason = _sentence_supported(sentence, hits)
            if supported:
                kept.append(sentence)
            else:
                rejected.append((sentence.strip(), reason))

        if kept:
            kept_lines.append(prefix + " ".join(kept))

    return "\n".join(kept_lines).strip(), rejected
