"""Tests for H1 heuristic tagger."""

from pathlib import Path

import pytest

from mcp_rag.tagging.heuristics import HeuristicTagger


@pytest.fixture
def tagger():
    return HeuristicTagger()


def test_format_tag(tagger):
    tags = tagger.tag_document(Path("/data/report.pdf"))
    assert "format:pdf" in tags


def test_year_segment(tagger):
    tags = tagger.tag_document(Path("/data/2026/facture.pdf"))
    assert "year:2026" in tags


def test_type_from_name(tagger):
    tags = tagger.tag_document(Path("/data/facture_acme.pdf"))
    assert any(t.startswith("type:") for t in tags)


def test_domaine_from_dirname(tagger):
    tags = tagger.tag_document(Path("/data/juridique/contract.pdf"))
    assert "domaine:juridique" in tags

    tags2 = tagger.tag_document(Path("/data/financier/bilan.xlsx"))
    assert "domaine:financier" in tags2


def test_deduplicate(tagger):
    tags = tagger.tag_document(Path("/data/2026/2026/report.pdf"))
    # Should not have duplicate year tags
    year_tags = [t for t in tags if t.startswith("year:")]
    assert len(year_tags) == 1
