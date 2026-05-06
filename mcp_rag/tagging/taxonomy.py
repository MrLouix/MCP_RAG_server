"""Taxonomy definitions and validation for semantic tagging."""

from typing import Any

DEFAULT_TAXONOMY = {
    "domaine": ["financier", "juridique", "technique", "commercial", "rh", "administratif"],
    "priorite": ["urgent", "normal", "faible"],
    "confidentialite": ["public", "interne", "confidentiel"],
}


def validate_taxonomy(taxonomy: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a taxonomy definition."""
    result = {}
    for key, value in taxonomy.items():
        if isinstance(value, list):
            result[key] = [str(v).lower() for v in value]
        else:
            result[key] = str(value).lower()
    return result


def build_gbnf_schema(taxonomy: dict[str, Any]) -> str:
    """Build a GBNF grammar for constrained JSON output from taxonomy."""
    rules = ["root ::= '{' members '}'", "members ::= pair (',' pair)*"]
    pairs = []
    for key, value in taxonomy.items():
        if isinstance(value, list) and value:
            values_str = " | ".join(f'"{v}"' for v in value)
            pairs.append(f'"{key}"' + " ::= " + values_str + " | null")
        else:
            pairs.append(f'"{key}"' + ' ::= string | null')
    return "\n".join(rules + pairs)
