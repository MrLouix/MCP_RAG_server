"""Couche H1 — tags système par heuristiques.

- Tokenisation du chemin, nom de fichier, segments de répertoire
- Mapping keyword→tag unifié (type, domaine, priorite, statut)
- Fichier .ragrules.yaml optionnel
- Préfixes de priorité (URGENT_, DRAFT_, etc.)
- Exécution en quelques ms, aucun ML.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REGEX_TIMEOUT = 0.2

# ------------------------------------------------------------------
# Unified keyword → tag mappings  (expand here to enrich auto-tagging)
# ------------------------------------------------------------------

KEYWORD_TO_TAG: dict[str, str] = {
    # --- Document types ---
    "facture": "type:facture",
    "devis": "type:devis",
    "proposition": "type:proposition",
    "offre": "type:offre",
    "contrat": "type:contrat",
    "commande": "type:commande",
    "bon": "type:bon",
    "rapport": "type:rapport",
    "compte-rendu": "type:compte-rendu",
    "cr": "type:compte-rendu",
    "note": "type:note",
    "memo": "type:memo",
    "procédure": "type:procedure",
    "procedure": "type:procedure",
    "manuel": "type:manuel",
    "guide": "type:guide",
    "présentation": "type:presentation",
    "presentation": "type:presentation",
    "formation": "type:formation",
    "spécification": "type:specification",
    "specification": "type:specification",
    "cahier": "type:cahier",
    "cctp": "type:cahier",
    "police": "type:police",
    "planning": "type:planning",
    "emploi": "type:emploi-du-temps",
    "cv": "type:cv",
    "lettre": "type:lettre",
    "attestation": "type:attestation",
    "certificat": "type:certificat",
    "relevé": "type:relevé",
    "releve": "type:relevé",
    "bilan": "type:bilan",
    "budget": "type:budget",
    "compte-rendu": "type:compte-rendu",
    "procès-verbal": "type:pv",
    "proces-verbal": "type:pv",

    # --- Domaines ---
    "commercial": "domaine:commercial",
    "commerciale": "domaine:commercial",
    "commerciaux": "domaine:commercial",
    "ventes": "domaine:commercial",
    "vente": "domaine:commercial",
    "juridique": "domaine:juridique",
    "juridiques": "domaine:juridique",
    "financier": "domaine:financier",
    "financière": "domaine:financier",
    "financiere": "domaine:financier",
    "finance": "domaine:financier",
    "comptabilité": "domaine:comptabilite",
    "comptabilite": "domaine:comptabilite",
    "rh": "domaine:rh",
    "ressources": "domaine:rh",
    "humaines": "domaine:rh",
    "personnel": "domaine:rh",
    "technique": "domaine:technique",
    "techniques": "domaine:technique",
    "informatique": "domaine:technique",
    "administratif": "domaine:administratif",
    "administrative": "domaine:administratif",
    "marketing": "domaine:marketing",
    "communication": "domaine:communication",
    "qualité": "domaine:qualite",
    "qualite": "domaine:qualite",
    "logistique": "domaine:logistique",
    "production": "domaine:production",
    "projet": "domaine:projet",
    "projets": "domaine:projet",
    "sécurité": "domaine:securite",
    "securite": "domaine:securite",
    "santé": "domaine:sante",
    "sante": "domaine:sante",

    # --- Priorité ---
    "urgent": "priorite:urgent",
    "importante": "priorite:important",
    "important": "priorite:important",

    # --- Confidentialité ---
    "confidentiel": "confidentialite:confidentiel",
    "confidentielle": "confidentialite:confidentiel",
    "interne": "confidentialite:interne",
    "public": "confidentialite:public",

    # --- Statut ---
    "brouillon": "statut:brouillon",
    "draft": "statut:brouillon",
    "valide": "statut:valide",
    "validé": "statut:valide",
    "final": "statut:final",
    "signé": "statut:signe",
    "signe": "statut:signe",
    "annexe": "statut:annexe",
    "révision": "statut:revision",
    "revision": "statut:revision",
}

# Filename prefix hints (e.g. URGENT_contract.pdf)
PRIORITY_PREFIXES: dict[str, str] = {
    "urgent": "priorite:urgent",
    "important": "priorite:important",
    "confidentiel": "confidentialite:confidentiel",
    "brouillon": "statut:brouillon",
    "draft": "statut:brouillon",
}


class HeuristicTagger:
    """Deterministic tag extraction from paths and names."""

    def __init__(self, max_rules_bytes: int = 102400) -> None:
        self.max_rules_bytes = max_rules_bytes

    def tag_document(self, path: Path) -> list[str]:
        """Return system tags for a document path."""
        tags: list[str] = []
        tags.append(f"format:{path.suffix.lstrip('.').lower()}")
        tags.extend(self._path_tags(path))
        tags.extend(self._prefix_hints(path))
        tags.extend(self._apply_ragrules(path))
        return list(dict.fromkeys(tags))

    # ------------------------------------------------------------------
    # Unified keyword-based tagging
    # ------------------------------------------------------------------

    def _path_tags(self, path: Path) -> list[str]:
        """Tokenize all path parts and match against KEYWORD_TO_TAG."""
        tags = []
        seen_tags: set[str] = set()
        for part in path.parts:
            if not part or part == "/":
                continue
            lower = part.lower()

            # Year detection (e.g. 2024, 2025, 2026)
            if re.fullmatch(r"20[0-9]{2}", lower):
                tag = f"year:{lower}"
                if tag not in seen_tags:
                    tags.append(tag)
                    seen_tags.add(tag)
                continue

            # Tokenize the part by common separators
            tokens = re.split(r"[_\-\s.]+", lower)

            # Check multi-word keywords first (e.g. "compte-rendu")
            for keyword, tag in KEYWORD_TO_TAG.items():
                if "-" in keyword and keyword in lower and tag not in seen_tags:
                    tags.append(tag)
                    seen_tags.add(tag)

            # Check individual tokens against keyword dict
            for token in tokens:
                if token in KEYWORD_TO_TAG:
                    tag = KEYWORD_TO_TAG[token]
                    if tag not in seen_tags:
                        tags.append(tag)
                        seen_tags.add(tag)

        return tags

    def _prefix_hints(self, path: Path) -> list[str]:
        """Check filename stem for priority/status prefixes."""
        stem = path.stem.lower()
        for prefix, tag in PRIORITY_PREFIXES.items():
            # Check prefix at start (with separator)
            if stem.startswith(prefix + "_") or stem.startswith(prefix + "-"):
                return [tag]
            # Check uppercase prefix (e.g. URGENT_doc.pdf)
            upper = prefix.upper()
            if stem.startswith(upper + "_") or stem.startswith(upper + "-"):
                return [tag]
        return []

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
