"""One built-in resource pool for curated and corpus-derived generation.

Resource kinds remain distinct: a parsed nominal is not automatically a typed
entity and a dependency tree is not a verified semantic frame. The compiled
bank is read in place; its inventories and occurrence ledger are never copied
into entities.json or loaded wholesale into memory.
"""

from __future__ import annotations

from functools import cached_property
from hashlib import sha256
from pathlib import Path
import re

from .documents import Document
from .semantic_generator import DEFAULT_FRAMES, default_pool
from .semantic_generator.generator import DEFAULT_PARADIGMS
from .template_bank import TemplateBank, fingerprint, read_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_SCOPE = "built_in_v1"
SPLIT_POLICY = "mined-text-sha256-v1-held-out-test"


def _name_pattern(names):
    names = sorted({name for name in names if name}, key=lambda name: (-len(name), name))
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, names)) + r")(?!\w)") if names else None


def _mined_split(text, held_out_pattern):
    if held_out_pattern is not None and held_out_pattern.search(text):
        return "test"
    bucket = int(sha256(text.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "test"


def mined_split(text, held_out_names=()):
    """Seed-independent split; reserve held-out test entity phrases for test."""
    return _mined_split(text, _name_pattern(held_out_names))


class BuiltInPool:
    """Central access to entities, semantic frames and all mined resources.

    A custom entity pool overrides just that component. Mined templates keep
    their attested local fillers; replacing them with arbitrary typed entities
    would lose agreement and semantic compatibility.
    """

    def __init__(self, repo_root=None, entities=None):
        self.repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        self.entities = entities if entities is not None else default_pool()
        self.frames = DEFAULT_FRAMES
        self.paradigms = dict(DEFAULT_PARADIGMS)

    @cached_property
    def bank(self):
        return TemplateBank(self.repo_root / "data" / "eng-base-templates")

    def templates(self, **filters):
        return self.bank.templates(**filters)

    def verbs(self):
        return self.bank.verbs()

    def adjectives(self):
        return self.bank.adjectives()

    def constructions(self):
        return read_rows(self.bank.path / "constructions.jsonl.gz")

    def verb_alternatives(self):
        return read_rows(self.bank.path / "verb_alternatives.jsonl.gz")

    def adjective_alternatives(self):
        return read_rows(self.bank.path / "adjective_alternatives.jsonl.gz")

    def occurrences(self):
        for path in sorted((self.bank.path / "occurrences").glob("*.jsonl.gz")):
            yield from read_rows(path)

    def summary(self):
        """Small monitoring view; no decompression of inventories on page GET."""
        result = dict(entities=len(self.entities), frames=len(self.frames),
                      available=False, status="missing", templates=0, verbs=0,
                      adjectives=0, constructions=0, verb_alternatives=0,
                      adjective_alternatives=0, records=0)
        try:
            bank = self.bank
        except (OSError, ValueError) as error:
            result.update(status="missing" if isinstance(error, FileNotFoundError) else "invalid",
                          error=str(error))
            return result
        counts = bank.manifest.get("counts", {})
        for field, key in (("templates", "templates"), ("verbs", "verb_lemmas"),
                           ("adjectives", "adjective_lemmas"), ("constructions", "constructions"),
                           ("verb_alternatives", "verb_alternative_lemmas"),
                           ("adjective_alternatives", "adjective_alternative_forms"),
                           ("records", "compiled_records")):
            result[field] = counts.get(key, 0)
        result.update(available=True, status="complete", path=str(bank.path),
                      source_sha256=bank.manifest.get("source_sha256"),
                      model=bank.manifest.get("model", {}))
        return result

    def metadata(self, *, include_mined=True):
        # Include every entity attribute affecting realization/selection, not
        # just its text. Dataset version identity must detect pool changes.
        from dataclasses import fields
        entities = [{field.name: (sorted(map(str, entity.labels.labels)) if field.name == "labels"
                                  else getattr(entity, field.name))
                     for field in fields(entity)} for entity in self.entities]
        result = {"resource_scope": RESOURCE_SCOPE, "entities_sha256": fingerprint(entities),
                  "frames": [frame.lemma for frame in self.frames]}
        if include_mined:
            result.update(bank_manifest_sha256=fingerprint(self.bank.manifest),
                          bank_source_sha256=self.bank.manifest.get("source_sha256"),
                          mined_split_policy=SPLIT_POLICY)
        return result

    def mined_documents(self, *, count=1, seed=0, split="train", lemma=None,
                        adjective=None, recombine=False, stats=None, progress=None):
        """Render all selected patterns, then keep the requested stable split.

        The quota applies before split filtering and store deduplication. These
        documents deliberately have no AnnotationRun until semantic annotation.
        """
        if count < 0:
            raise ValueError("examples per mined template cannot be negative")
        if split not in ("train", "dev", "test"):
            raise ValueError(f"unknown split: {split!r}")
        stats = stats if stats is not None else {}
        stats.update(rendered=0, split_excluded=0, eligible=0)
        if count == 0:
            return
        identity = fingerprint(self.bank.manifest)
        held_out = _name_pattern(self.entities.split("test").surface_forms)
        for row in self.bank.generate(count=count, seed=seed, lemma=lemma,
                                      adjective=adjective, recombine=recombine):
            stats["rendered"] += 1
            selected = _mined_split(row["text"], held_out) == split
            if selected:
                stats["eligible"] += 1
            else:
                stats["split_excluded"] += 1
            if progress and stats["rendered"] % 1000 == 0:
                progress(dict(stats))
            if not selected:
                continue
            key = fingerprint([identity, row["template_id"], row["bindings"], recombine])
            yield Document(
                document_id=f"mined-{key}", text=row["text"], source="eng-base-templates", split=split,
                metadata={**{key: value for key, value in row.items() if key != "text"},
                          "resource_pool": "built-in", "bank_manifest_sha256": identity,
                          "bank_source_sha256": self.bank.manifest.get("source_sha256"),
                          "recombined": recombine, "split_policy": SPLIT_POLICY})
        if progress:
            progress(dict(stats))
