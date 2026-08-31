"""Local tools available to the tax research agent."""

from __future__ import annotations

TAX_CATALOG = {
    "income tax": (
        "Income tax is generally assessed on income earned by individuals or entities. "
        "Collection commonly happens through withholding, estimated payments, and returns."
    ),
    "sales tax": (
        "Sales tax is generally assessed on taxable sales of goods or services. "
        "A seller commonly collects it from the buyer and remits it to a tax authority."
    ),
}


def lookup_tax_topic(topic: str) -> str:
    """Look up a short, educational definition in the local tax catalog.

    Args:
        topic: Tax topic to find, such as ``sales tax`` or ``income tax``.

    Returns:
        A catalog definition or a message listing the known topics.
    """
    normalized_topic = topic.strip().lower()
    if definition := TAX_CATALOG.get(normalized_topic):
        return definition

    known_topics = ", ".join(sorted(TAX_CATALOG))
    return f"No exact match for '{topic}'. Known topics: {known_topics}."