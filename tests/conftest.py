"""Shared fixtures. Also puts the repository root on ``sys.path`` so the tests
run with a plain ``pytest`` from the repo, without any installation step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data"
BANK = DATA / "qasrl-v2"
WIKTIONARY = DATA / "wiktionary"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"corpus file not available: {path}")
    return path


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def bank_dev() -> Path:
    return _require(BANK / "orig" / "dev.jsonl.gz")


@pytest.fixture(scope="session")
def bank_dense_dev() -> Path:
    return _require(BANK / "dense" / "dev.jsonl.gz")


@pytest.fixture(scope="session")
def qa_text_dev() -> Path:
    return _require(DATA / "qa-srl-2.0" / "wiki1.dev.qa")


@pytest.fixture(scope="session")
def inflections_path() -> Path:
    return _require(WIKTIONARY / "en_verb_inflections.txt")


@pytest.fixture(scope="session")
def postags_path() -> Path:
    return _require(WIKTIONARY / "en_postags_withverb.txt")


@pytest.fixture(scope="session")
def verb_phrases_path() -> Path:
    return _require(WIKTIONARY / "verb_phrases.txt")


@pytest.fixture(scope="session")
def lexicon(inflections_path: Path):
    from semantic_corpus.qasrl_core.inflections import load_inflections

    return load_inflections(inflections_path)


@pytest.fixture(scope="session")
def give_forms():
    from semantic_corpus.qasrl_core.models import InflectedForms

    return InflectedForms(
        stem="give",
        present_singular_3rd="gives",
        present_participle="giving",
        past="gave",
        past_participle="given",
    )
