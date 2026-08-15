"""Seed topics for the crawler.

Three sources of titles:

* ``CURATED`` — hand-picked, high-traffic articles that give the index broad,
  reliable coverage plus a few deliberately ambiguous titles (Mercury, Python,
  Java, Mars) so the disambiguation path is exercisable out of the box.
* ``VITAL_PAGES`` — Wikipedia's own *Vital articles* lists, walked in order of
  importance. This is the bulk source, and it exists because a hand-written
  list cannot be big enough: 500-odd curated articles means most real questions
  are about something that was never indexed, which no amount of retrieval or
  reranking can repair. Letting Wikipedia decide what matters also removes the
  authorship bias that made the curated set score well on its own evaluation
  suite and poorly on everything else.
* ``CATEGORIES`` — MediaWiki categories the crawler walks to top the index up to
  the requested article count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from app.core.wikipedia import WikipediaClient

CURATED: dict[str, list[str]] = {
    "science": [
        "Physics", "Quantum mechanics", "Theory of relativity", "Thermodynamics",
        "Photosynthesis", "DNA", "Evolution", "Natural selection", "Genetics",
        "Periodic table", "Chemical element", "Atom", "Black hole", "Big Bang",
        "Solar System", "Milky Way", "Gravity", "Electromagnetism", "Speed of light",
        "Climate change", "Plate tectonics", "Human brain", "Immune system",
        "Antibiotic", "Vaccine", "CRISPR", "Standard Model", "Entropy",
    ],
    "history": [
        "Ancient Egypt", "Roman Empire", "Ancient Greece", "Middle Ages",
        "Renaissance", "Industrial Revolution", "French Revolution",
        "American Revolution", "World War I", "World War II", "Cold War",
        "Silk Road", "Ottoman Empire", "Mughal Empire", "Maurya Empire",
        "Byzantine Empire", "Age of Discovery", "Great Depression",
        "Indian independence movement", "Berlin Wall", "Apollo 11",
    ],
    "technology": [
        "Computer science", "Artificial intelligence", "Machine learning",
        "Deep learning", "Large language model", "Neural network", "Algorithm",
        "Cryptography", "Internet", "World Wide Web", "Operating system",
        "Linux", "Semiconductor", "Transistor", "Quantum computing",
        "Cloud computing", "Blockchain", "Computer virus", "Database",
        "Open-source software", "Python (programming language)",
        "JavaScript", "Rust (programming language)", "HTTP", "Docker (software)",
    ],
    "geography": [
        "Earth", "Continent", "Africa", "Asia", "Europe", "North America",
        "South America", "Antarctica", "Pacific Ocean", "Amazon rainforest",
        "Mount Everest", "Sahara", "Nile", "Himalayas", "Great Barrier Reef",
        "India", "United States", "China", "Japan", "Brazil", "Egypt",
        "United Kingdom", "France", "Germany", "Australia",
    ],
    "arts": [
        "Renaissance art", "Leonardo da Vinci", "Vincent van Gogh", "Pablo Picasso",
        "Michelangelo", "Impressionism", "Classical music", "Ludwig van Beethoven",
        "Wolfgang Amadeus Mozart", "Jazz", "William Shakespeare", "Novel",
        "Poetry", "Cinema of the United States", "Photography", "Architecture",
        "Taj Mahal", "Louvre", "Opera", "Bollywood",
    ],
    "people": [
        "Albert Einstein", "Isaac Newton", "Marie Curie", "Charles Darwin",
        "Alan Turing", "Ada Lovelace", "Nikola Tesla", "Mahatma Gandhi",
        "Nelson Mandela", "Cleopatra", "Julius Caesar", "Genghis Khan",
        "Confucius", "Aristotle", "Plato", "Stephen Hawking", "Rosalind Franklin",
        "Katherine Johnson", "B. R. Ambedkar",
    ],
    # Titles that resolve to disambiguation pages — used to demo that path.
    "ambiguous": [
        "Mercury", "Python", "Java", "Mars", "Apple", "Turkey", "Phoenix",
        "Amazon", "Jupiter",
    ],
}

VITAL_PREFIX = "Vital articles/Level"
# Level 3 is one page of ~1,000 articles — the core of an encyclopedia — so it
# is walked first and a small `--limit` buys the most valuable articles. Level 4
# adds ~10,000 across a dozen subject pages. Level 5 (~50,000) is deliberately
# not used: it reaches well past what an assessment index needs.
VITAL_LEVEL_3 = "Wikipedia:Vital articles/Level 3"
VITAL_LEVEL_4 = "Wikipedia:Vital articles/Level 4/"

# Working pages that sit under the same prefix but are not subject lists.
_VITAL_SKIP_SUFFIXES = ("/Removed", "/draft", "/Draft", "/Article alerts", "/")

# Vital-article pages cite their own scaffolding — other levels, project talk
# archives, and meta articles about Wikipedia itself. Those arrive as
# namespace-0 links like any other. The `List of` / `Index of` / `Outline of`
# families are dropped for a different reason: they are navigation pages whose
# body is a bare list of links, so they chunk into text that matches many
# queries and answers none.
_VITAL_NOISE_PREFIXES = (
    "Wikipedia:",
    "Wikipedia talk:",
    "Portal:",
    "Template:",
    "Category:",
    "Help:",
    "Draft:",
    "List of",
    "Index of",
    "Outline of",
)


async def vital_pages(client: "WikipediaClient", lang: str = "en") -> list[str]:
    """The Vital articles list pages, most important first.

    Only the top-level Level 4 subject pages carry links; their own
    subpages (``…/Physical sciences/Physics``) are transcluded fragments that
    return nothing, so anything deeper than one segment is skipped.
    """
    discovered = await client.subpages(VITAL_PREFIX, lang=lang)

    level_4 = [
        title
        for title in discovered
        if title.startswith(VITAL_LEVEL_4)
        and "/" not in title[len(VITAL_LEVEL_4):]
        and not title.endswith(_VITAL_SKIP_SUFFIXES)
    ]
    return [VITAL_LEVEL_3, *sorted(level_4)]


async def vital_titles(
    client: "WikipediaClient", limit: int, lang: str = "en"
) -> list[str]:
    """Walk the Vital articles lists until ``limit`` titles are collected.

    Stops as soon as the quota is met, so a modest limit costs a couple of API
    calls rather than a crawl of every list.
    """
    titles: list[str] = []
    seen: set[str] = set()

    for page in await vital_pages(client, lang=lang):
        if len(titles) >= limit:
            break
        try:
            links = await client.page_links(page, lang=lang, limit=5000)
        except Exception:  # noqa: BLE001 - one missing list must not stop the crawl
            continue

        for title in links:
            if title in seen or title.startswith(_VITAL_NOISE_PREFIXES):
                continue
            seen.add(title)
            titles.append(title)
            if len(titles) >= limit:
                break

    return titles[:limit]


CATEGORIES: list[str] = [
    "Category:Physics",
    "Category:Astronomy",
    "Category:Machine learning",
    "Category:Computer networking",
    "Category:Ancient Rome",
    "Category:World War II",
    "Category:Renaissance",
    "Category:Human anatomy",
    "Category:Climate",
    "Category:Music genres",
    "Category:Countries in Asia",
    "Category:Nobel laureates in Physics",
]


def curated_titles() -> list[str]:
    """All curated titles, de-duplicated, order preserved."""
    seen: set[str] = set()
    titles: list[str] = []
    for group in CURATED.values():
        for title in group:
            if title not in seen:
                seen.add(title)
                titles.append(title)
    return titles
