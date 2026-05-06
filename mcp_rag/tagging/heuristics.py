"""Couche H1 — tags système par heuristiques.

- Chemin, extension, segments de répertoire
- Fichier .ragrules.yaml optionnel
- Constat : exécution en quelques ms, aucun ML.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REGEX_TIMEOUT = 0.2


class HeuristicTagger:
    """Deterministic tag extraction from paths, names, and user rules."""

    def __init__(self, max_rules_bytes: int = 102400) -> None:
        self.max_rules_bytes = max_rules_bytes

    def tag_document(self, path: Path) -> list[str]:
        """Return system tags for a document path."""
        tags: list[str] = []
        tags.append(f"format:{path.suffix.lstrip('.').lower()}")
        tags.extend(self._path_segments(path))
        tags.extend(self._apply_ragrules(path))
        tags.extend(self._dirname_hints(path))
        return list(dict.fromkeys(tags))  # preserve order, deduplicate

    # ------------------------------------------------------------------
    # Segment extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _path_segments(path: Path) -> list[str]:
        """Infer tags from path components like year:2026, client:acme."""
        tags = []
        for part in path.parts:
            lower = part.lower()
            # Date/Year segments
            if re.fullmatch(r"20[0-9]{2}", lower):
                tags.append(f"year:{lower}")
            # Well-known role names
            if lower in ("facture", "devis", "contrat", "compte-rendu", "reunion"):
                tags.append(f"type:{lower}")
        return tags

    @staticmethod
    def _dirname_hints(path: Path) -> list[str]:
        """Extract directory-name hints."""
        tags = []
        dirnames = {p.lower() for p in path.parts}
        domain_map = {
            "juridique": "domaine:juridique",
            "financier": "domaine:financier",
            "finance": "domaine:financier",
            "technique": "domaine:technique",
            "commercial": "domaine:commercial",
            "rh": "domaine:rh",
            "administratif": "domaine:administratif",
        }
        for dirname, tag in domain_map.items():
            if dirname in dirnames:
                tags.append(tag)
        return tags

    # ------------------------------------------------------------------
    # .ragrules.yaml
    # ------------------------------------------------------------------

    def _apply_ragrules(self, path: Path) -> list[str]:
        """Parse .ragrules.yaml and apply matching glob patterns."""
        rules_file = path.parent / ".ragrules.yaml"
        if not rules_file.exists():
            # Walk up to find a rules file at any ancestor
            for parent in path.parents:
                candidate = parent / ".ragrules.yaml"
                if candidate.exists():
                    rules_file = candidate
                    break
            else:
                return []

        try:
            stat = rules_file.stat()
            if stat.st_size > self.max_rules_bytes:
                logger.warning("ragrules_too_large", extra={"path": str(rules_file), "size": stat.st_size})
                return []

            data = yaml.safe_load(rules_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return []
            rules = data.get("rules", [])
            if not isinstance(rules, list):
                return []
        except Exception as exc:
            logger.warning("ragrules_parse_failed", extra={"path": str(rules_file), "error": str(exc)})
            return []

        tags = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            pattern = rule.get("pattern")
            if not pattern:
                continue
            try:
                if _glob_matches(str(path), pattern):
                    rule_tags = rule.get("tags", [])
                    if isinstance(rule_tags, list):
                        tags.extend(rule_tags)
            except Exception:
                pass
        return tags


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _glob_matches(path_str: str, pattern: str) -> bool:
    """Match a path against a glob/regex-like pattern.

    Supports simple globs: *, **, ?
    """
    from fnmatch import fnmatch

    return fnmatch(path_str, pattern)
