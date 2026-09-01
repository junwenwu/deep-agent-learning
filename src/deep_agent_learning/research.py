"""Search a small, versioned corpus of authoritative tax guidance."""

from __future__ import annotations

import json
import re
from datetime import date
from importlib.resources import files
from typing import Any

DEFAULT_CORPUS = files("deep_agent_learning").joinpath("knowledge/tax_sources.json")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _load_corpus() -> list[dict[str, Any]]:
    with DEFAULT_CORPUS.open(encoding="utf-8") as corpus_file:
        records = json.load(corpus_file)
    if not isinstance(records, list):
        raise TypeError("Tax source corpus must contain a JSON array.")
    return records


def _parse_effective_on(effective_on: str | None) -> date | None:
    if effective_on is None:
        return None
    try:
        return date.fromisoformat(effective_on)
    except ValueError as error:
        raise ValueError("effective_on must use YYYY-MM-DD format.") from error


def _is_effective(record: dict[str, Any], effective_on: date | None) -> bool:
    if effective_on is None:
        return True
    effective_from = date.fromisoformat(record["effective_from"])
    effective_to_value = record.get("effective_to")
    effective_to = date.fromisoformat(effective_to_value) if effective_to_value else None
    return effective_from <= effective_on and (
        effective_to is None or effective_on <= effective_to
    )


def _relevance(record: dict[str, Any], query_tokens: set[str]) -> int:
    title_tokens = set(TOKEN_PATTERN.findall(record["title"].lower()))
    section_tokens = set(TOKEN_PATTERN.findall(record["section"].lower()))
    text_tokens = set(TOKEN_PATTERN.findall(record["text"].lower()))
    return (
        4 * len(query_tokens & title_tokens)
        + 3 * len(query_tokens & section_tokens)
        + len(query_tokens & text_tokens)
    )


def search_tax_sources(
    query: str,
    jurisdictions: list[str] | None = None,
    effective_on: str | None = None,
    limit: int = 5,
) -> str:
    """Search authoritative excerpts by text, jurisdiction, and effective date.

    Args:
        query: Terms describing the tax issue to research.
        jurisdictions: Optional jurisdictions, such as ``New York`` or ``California``.
        effective_on: Optional ISO date on which the guidance must be effective.
        limit: Maximum number of excerpts to return, from 1 through 10.
    Returns:
        JSON containing matching excerpts and complete citation metadata.

    Raises:
        ValueError: If the query, date, or result limit is invalid.
    """
    query_tokens = set(TOKEN_PATTERN.findall(query.lower()))
    if not query_tokens:
        raise ValueError("query must contain at least one searchable term.")
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10.")

    requested_jurisdictions = {
        jurisdiction.strip().lower() for jurisdiction in jurisdictions or []
    }
    target_date = _parse_effective_on(effective_on)
    matches: list[tuple[int, dict[str, Any]]] = []
    for record in _load_corpus():
        if requested_jurisdictions and record["jurisdiction"].lower() not in (
            requested_jurisdictions
        ):
            continue
        if not _is_effective(record, target_date):
            continue
        score = _relevance(record, query_tokens)
        if score:
            matches.append((score, record))

    matches.sort(key=lambda item: (-item[0], item[1]["excerpt_id"]))
    payload = {
        "query": query,
        "jurisdictions": jurisdictions or [],
        "effective_on": effective_on,
        "result_count": min(len(matches), limit),
        "results": [record for _, record in matches[:limit]],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def read_tax_source(
    excerpt_id: str,
) -> str:
    """Read one authoritative excerpt by its stable identifier.

    Args:
        excerpt_id: Identifier returned by ``search_tax_sources``.
    Returns:
        JSON containing the excerpt and complete citation metadata, or a structured
        error that directs the caller to search again when the identifier is unknown.
    """
    normalized_id = excerpt_id.strip().lower()
    records = _load_corpus()
    for record in records:
        if record["excerpt_id"].lower() == normalized_id:
            return json.dumps(record, indent=2, sort_keys=True)
    return json.dumps(
        {
            "error": "unknown_excerpt_id",
            "requested_excerpt_id": excerpt_id,
            "recovery": (
                "Call search_tax_sources again and pass an exact excerpt_id from its results."
            ),
            "available_excerpt_ids": sorted(record["excerpt_id"] for record in records),
        },
        indent=2,
        sort_keys=True,
    )