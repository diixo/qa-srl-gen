"""One built-in resource pool for curated and corpus-derived generation.

Resource kinds remain distinct: a parsed nominal is not automatically a typed
entity and a dependency tree is not a verified semantic frame. One physical
bank contains every source; readers retain committed per-source byte ranges.
"""

from __future__ import annotations

from functools import cached_property
from hashlib import sha256
from pathlib import Path
import re

from .documents import Document
from .semantic_generator import DEFAULT_FRAMES, default_pool
from .semantic_generator.generator import DEFAULT_PARADIGMS
from .template_bank import TemplateBank, fingerprint
from .controlled_verbs import CONTROLLED_VERBS, VERSION as CONTROLLED_VERSION, controlled_head, excluded_predicates

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
        self.entities = entities if entities is not None else default_pool(
            mined_bank=self.repo_root / 'data/mined-templates')
        self.frames = DEFAULT_FRAMES
        self.paradigms = dict(DEFAULT_PARADIGMS)

    @cached_property
    def bank(self):
        """The single physical bank shared by all generators."""
        return TemplateBank(self.repo_root / 'data' / 'mined-templates')

    @cached_property
    def registered_banks(self):
        return self.bank.components() if self.bank.manifest.get('sources') else ()

    @cached_property
    def banks(self):
        """Stable component snapshot for one job or page request.

        Template identities can occur in several sources with different local
        fillers. Keep every source entry; the corpus store deduplicates texts.
        """
        return self.bank.components()

    def templates(self, **filters):
        self._check_mined_lemma(filters.get('lemma'))
        for bank in self.banks:
            for template in bank.templates(**filters):
                if not excluded_predicates(template.record):
                    yield template

    @staticmethod
    def _check_mined_lemma(lemma):
        if lemma and controlled_head(lemma):
            raise ValueError(f'{lemma!r} is handled by controlled grammar, not mined predicate selection')

    def verbs(self):
        for bank in self.banks:
            for row in bank.verbs():
                if not controlled_head(row['lemma']):
                    yield row

    def adjectives(self):
        for bank in self.banks:
            yield from bank.adjectives()

    def _inventory(self, name, *, optional=False):
        for bank in self.banks:
            if optional and f'{name}.jsonl.gz' not in bank.files:
                continue
            yield from bank.rows(f"{name}.jsonl.gz")

    def constructions(self):
        return (row for row in self._inventory('constructions') if not controlled_head(row['lemma']))

    def verb_alternatives(self):
        return (row for row in self._inventory('verb_alternatives') if not controlled_head(row['lemma']))

    def adjective_alternatives(self):
        return self._inventory("adjective_alternatives")

    def entity_candidates(self):
        return self._inventory('entity_candidates', optional=True)

    def nominals(self):
        return self._inventory('nominals', optional=True)

    def adverbs(self):
        return self._inventory('adverbs', optional=True)

    def occurrences(self):
        for bank in self.banks:
            yield from bank.occurrences()

    def summary(self):
        """Small monitoring view; no decompression of inventories on page GET."""
        result = dict(entities=len(self.entities), frames=len(self.frames),
                      available=False, status="missing", templates=0, verbs=0,
                      adjectives=0, constructions=0, verb_alternatives=0,
                      adjective_alternatives=0, records=0, source_banks=0, banks=[],
                      counts_are_component_entries=True, counts_are_archived=True,
                      controlled_verbs=list(CONTROLLED_VERBS), controlled_policy=CONTROLLED_VERSION)
        try:
            banks = self.banks
        except (OSError, ValueError) as error:
            result.update(status="missing" if isinstance(error, FileNotFoundError) else "invalid",
                          error=str(error))
            return result
        for bank in banks:
            counts = bank.manifest.get("counts", {})
            for field, key in (("templates", "templates"), ("verbs", "verb_lemmas"),
                               ("adjectives", "adjective_lemmas"), ("constructions", "constructions"),
                               ("verb_alternatives", "verb_alternative_lemmas"),
                               ("adjective_alternatives", "adjective_alternative_forms"),
                               ("records", "compiled_records")):
                result[field] += counts.get(key, 0)
            result["banks"].append({"id": bank.source_id, "path": str(bank.path),
                                    "source_sha256": bank.manifest.get("source_sha256"),
                                    "counts": counts})
        result.update(available=True, status="complete", source_banks=len(banks),
                      path=str(self.bank.path), source_sha256=self.bank.manifest.get("source_sha256"),
                      model=self.bank.manifest.get("model", {}))
        return result

    def metadata(self, *, include_mined=True):
        # Include every entity attribute affecting realization/selection, not
        # just its text. Dataset version identity must detect pool changes.
        from dataclasses import fields
        entities = [{field.name: (sorted(map(str, entity.labels.labels)) if field.name == "labels"
                                  else getattr(entity, field.name))
                     for field in fields(entity)} for entity in self.entities]
        result = {"resource_scope": RESOURCE_SCOPE, "entities_sha256": fingerprint(entities),
                  "frames": [frame.lemma for frame in self.frames],
                  "controlled_verbs": list(CONTROLLED_VERBS), "controlled_policy": CONTROLLED_VERSION}
        if include_mined:
            from .template_semantics import VERSION
            result['template_semantics'] = VERSION
            path = self.repo_root / 'data/wiktionary/en_verb_inflections.txt'
            result['semantic_inflections_sha256'] = sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            result.update(bank_manifest_sha256=fingerprint(self.bank.manifest),
                          bank_source_sha256=self.bank.manifest.get("source_sha256"),
                          mined_split_policy=SPLIT_POLICY)
            banks = self.banks
            if len(banks) > 1 or self.registered_banks:
                result["bank_components"] = [
                    {"id": bank.source_id, "manifest_sha256": fingerprint(bank.manifest),
                     "source_sha256": bank.manifest.get("source_sha256")} for bank in banks]
        return result

    def mined_documents(self, *, count=1, seed=0, split="train", lemma=None,
                        adjective=None, recombine=False, stats=None, progress=None):
        """Render all selected patterns, then keep the requested stable split.

        The quota applies before split filtering and store deduplication.
        Use mined_records to additionally derive semantic candidates and QA.
        """
        if count < 0:
            raise ValueError("examples per mined template cannot be negative")
        if split not in ("train", "dev", "test"):
            raise ValueError(f"unknown split: {split!r}")
        stats = stats if stats is not None else {}
        stats.update(rendered=0, split_excluded=0, eligible=0)
        if count == 0:
            return
        self._check_mined_lemma(lemma)
        held_out = _name_pattern(self.entities.split("test").surface_forms)
        for bank in self.banks:
            identity = fingerprint(bank.manifest)
            if adjective is not None and not any(row["lemma"] == adjective for row in bank.adjectives()):
                continue
            try:
                for row in bank.generate(count=count, seed=seed, lemma=lemma,
                                         adjective=adjective, recombine=recombine, exclude_controlled=True):
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
                    key = fingerprint([identity, CONTROLLED_VERSION, row["template_id"], row["bindings"], recombine])
                    source = bank.source_label
                    yield Document(
                        document_id=f"mined-{key}", text=row["text"], source=source, split=split,
                        metadata={**{key: value for key, value in row.items() if key != "text"},
                                  "resource_pool": "built-in", "bank_manifest_sha256": identity,
                                  "controlled_policy": CONTROLLED_VERSION,
                                  "bank_source_sha256": bank.manifest.get("source_sha256"),
                                  "recombined": recombine, "split_policy": SPLIT_POLICY})
            except ValueError as error:
                if str(error) != "no templates match the selection":
                    raise
        if not stats["rendered"]:
            raise ValueError("no templates match the selection")
        if progress:
            progress(dict(stats))

    @cached_property
    def template_annotator(self):
        from .qasrl_core.inflections import load_inflections
        from .template_semantics import TemplateAnnotator
        path = self.repo_root / 'data/wiktionary/en_verb_inflections.txt'
        lexicon = load_inflections(path) if path.is_file() else None
        return TemplateAnnotator(entities=self.entities, frames=self.frames,
                                 paradigms=self.paradigms, lexicon=lexicon)

    def mined_records(self, **options):
        """Render and annotate the same stream, preserving unsupported records."""
        for document in self.mined_documents(**options):
            result = self.template_annotator.annotate(document, seed=options.get('seed', 0))
            yield result.document, result.run
