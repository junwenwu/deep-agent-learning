"""Local tools available to the tax research agent."""

from __future__ import annotations

TAX_CATALOG = {
    "income tax": (
        "Income tax is generally assessed on income earned by individuals or entities. "
        "Collection commonly happens through withholding, estimated payments, and returns."
    ),
    "property tax": (
        "Property tax is generally assessed on the value of owned real estate or other "
        "property. The owner commonly pays it to a local tax authority on a recurring basis."
    ),
    "sales tax": (
        "Sales tax is generally assessed on taxable sales of goods or services. "
        "A seller commonly collects it from the buyer and remits it to a tax authority."
    ),
}

JURISDICTION_CATALOG = {
    "federal": (
        "A federal tax jurisdiction applies nationwide under the authority of the national "
        "government. Federal rules can coexist with state and local tax obligations."
    ),
    "local": (
        "A local tax jurisdiction is administered by a city, county, municipality, or similar "
        "authority. Local rules and rates can differ within the same state."
    ),
    "state": (
        "A state tax jurisdiction applies within one state. Each state can define its own tax "
        "bases, rates, filing requirements, and interactions with local taxes."
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


def lookup_tax_jurisdiction(jurisdiction: str) -> str:
    """Look up general guidance about a level of tax jurisdiction.

    Args:
        jurisdiction: Jurisdiction level to find, such as ``federal``, ``state``, or ``local``.

    Returns:
        General jurisdiction guidance or a message listing the known levels.
    """
    normalized_jurisdiction = jurisdiction.strip().lower()
    if guidance := JURISDICTION_CATALOG.get(normalized_jurisdiction):
        return guidance

    known_jurisdictions = ", ".join(sorted(JURISDICTION_CATALOG))
    return (
        f"No exact match for '{jurisdiction}'. "
        f"Known jurisdiction levels: {known_jurisdictions}."
    )