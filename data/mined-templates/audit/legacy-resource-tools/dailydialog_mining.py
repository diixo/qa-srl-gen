"""Auditable DailyDialog normalization and enrichment of the shared template bank.

The parser proposes entities; only explicit reviewed readings enter EntityPool.
Original turns and ordered text edits are retained in the source snapshot.
"""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

VERSION = "dailydialog-resources-v1"
CURATION = Path(__file__).with_name("dailydialog_curation.json")
APOSTROPHES = "\u2018\u2019\u201b\u02bc\uff07\u0060\u00b4\u2032\u2035\u02bb\u02b9\u275b\u275c"
_CASE_PROTECTED = re.compile(r"(?:https?://|www\s*\.)[^\n\"']*|\b[^\s@]+@[^\s@]+\b|\blower case\b.*", re.I)


def _protected(text, position):
    # Spoken/spaced URLs and dictated lowercase addresses also occur in DailyDialog.
    return any(m.start() <= position < m.end() for m in _CASE_PROTECTED.finditer(text))


def load_curation(path=CURATION):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Normalizer:
    """Apply reviewed replacements only; never guess a spelling by edit distance."""

    def __init__(self, curation):
        self.case_phrases = [(re.compile(r"(?<!\w)" + re.escape(row["before"]) + r"(?!\w)", re.I), row["after"])
                             for row in curation.get("case_phrases", [])]
        self.truecase_forms = {word.casefold(): word for word in curation.get("truecase", [])}
        words = sorted(self.truecase_forms, key=lambda s: (-len(s), s))
        self.truecase_pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, words)) + r")(?!\w)", re.I) if words else None
        self.corrections = []
        seen = set()
        for row in curation.get("corrections", []):
            old, new = row["before"], row["after"]
            if not old or not new or old.casefold() in seen or old == new:
                raise ValueError("invalid or duplicate correction")
            seen.add(old.casefold())
            self.corrections.append((re.compile(r"(?<!\w)" + re.escape(old) + r"(?!\w)", re.I), new))

    def __call__(self, original):
        text, edits = original, []

        def replace(pattern, replacement, kind):
            nonlocal text
            # Right to left: offsets always refer to the text before this edit.
            for match in reversed(list(pattern.finditer(text))):
                if kind != "apostrophe" and _protected(text, match.start()):
                    continue
                old = match.group()
                new = replacement(match) if callable(replacement) else replacement
                if old != new:
                    edits.append(dict(kind=kind, start=match.start(), end=match.end(), before=old, after=new))
                    text = text[:match.start()] + new + text[match.end():]

        replace(re.compile("[" + re.escape(APOSTROPHES) + "]"), "'", "apostrophe")
        # Fix tokenized contractions/possessives, not quoted phrases.
        replace(re.compile(r"\b(\w+)\s*'\s*(s|t|re|ve|ll|d|m)\b(?!')", re.I),
                lambda m: m[1] + "'" + m[2].lower(), "contraction_spacing_case")
        # DailyDialog contains mixed case such as I'Ve and don'T.
        replace(re.compile(r"\b(i)'(m|d|ll|ve)\b", re.I),
                lambda m: "I'" + m[2].lower(), "first_person_contraction")
        replace(re.compile(r"\bi\b"), "I", "first_person_pronoun")
        for pattern, replacement in self.corrections:
            def recase(match):
                old = match.group()
                return replacement.upper() if old.isupper() else (
                    replacement[0].upper() + replacement[1:] if old[0].isupper() else replacement)
            replace(pattern, recase, "reviewed_spelling")
        replace(re.compile(r"\b(\w+)'(s|t|re|ve|ll|d|m)\b(?!')", re.I),
                lambda m: m[1] + "'" + m[2].lower(), "contraction_spacing_case")
        if self.truecase_pattern:
            replace(self.truecase_pattern, lambda m: self.truecase_forms[m.group().casefold()], "reviewed_truecase")
        for pattern, replacement in self.case_phrases:
            replace(pattern, lambda m: replacement[0].upper() + replacement[1:] if m[0][0].isupper() else replacement,
                    "reviewed_context_truecase")
        return text, edits

    def sentence_case(self, doc, edits):
        text = doc.text
        replacements = {}
        for sentence in doc.sents:
            first = next((t for t in sentence if not t.is_punct and not t.is_space), None)
            # Do not turn mixed-case brands (iPhone, eBay) into Iphone/Ebay,
            # abbreviations into new sentences, or lettered list markers into names.
            if first is not None and first.text.islower() and first.text.isalpha():
                if _protected(text, first.idx):
                    continue
                prefix = text[:first.idx].rstrip(' \t\r\n\"\'\u201c\u201d')
                if prefix and not prefix.endswith((".", "?", "!")):
                    continue
                if re.search(r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|St|etc|vs|e\.g|i\.e)|\b[a-zA-Z])\.$", prefix, re.I):
                    continue
                if first.idx not in replacements:
                    replacements[first.idx] = (first.idx + 1, text[first.idx].upper(), "sentence_truecase")
        for start, (end, after, kind) in sorted(replacements.items(), reverse=True):
            before = text[start:end]
            if before != after:
                edits.append(dict(kind=kind, start=start, end=end, before=before, after=after))
                text = text[:start] + after + text[end:]
        return text


def restore_original(text, edits):
    for edit in reversed(edits):
        start = edit["start"]
        end = start + len(edit["after"])
        if text[start:end] != edit["after"]:
            raise ValueError("normalization edit does not match the text")
        text = text[:start] + edit["before"] + text[end:]
    return text


def prepare_source(directory, destination, curation, *, splits=("train", "val", "test"), nlp=None):
    """One record per turn, retaining even repeats and their original split."""
    counts, sources, edit_counts = Counter(), [], Counter()
    normalize = Normalizer(curation)
    rows = []
    if len(set(splits)) != len(splits):
        raise ValueError("splits must not repeat")
    for split in splits:
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown DailyDialog split: {split}")
    with Path(destination).open("x", encoding="utf-8", newline="\n") as sink:
        for split in splits:
            path = Path(directory) / f"dailydialog-{split}.jsonl"
            raw_bytes = path.read_bytes()
            sources.append(dict(file=path.name, split=split, sha256=hashlib.sha256(raw_bytes).hexdigest()))
            for line, raw in enumerate(raw_bytes.decode("utf-8-sig").splitlines(), 1):
                dialogue = json.loads(raw)
                if not isinstance(dialogue, list) or not dialogue:
                    raise ValueError(f"{path}:{line}: expected a nonempty list of turns")
                counts[f"{split}_dialogues"] += 1
                for turn, original in enumerate(dialogue, 1):
                    if not isinstance(original, str) or not original.strip():
                        raise ValueError(f"{path}:{line}:{turn}: expected nonempty text")
                    text, edits = normalize(original)
                    if restore_original(text, edits) != original:
                        raise AssertionError("normalization lost source text")
                    row = dict(idx=counts["turns"], example=text, original=original,
                               origin=dict(corpus="DailyDialog", file=path.name, split=split,
                                           dialogue=line, turn=turn), edits=edits)
                    rows.append(row)
                    counts["turns"] += 1
                    counts[f"{split}_turns"] += 1
        if nlp is not None:
            for processed, (doc, row) in enumerate(nlp.pipe(
                    ((r["example"], r) for r in rows), as_tuples=True, batch_size=256), 1):
                row["example"] = normalize.sentence_case(doc, row["edits"])
                if processed % 10000 == 0:
                    print(f"Truecased {processed:,} / {len(rows):,} turns", flush=True)
        for row in rows:
            if restore_original(row["example"], row["edits"]) != row["original"]:
                raise AssertionError("truecasing lost source text")
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["changed_turns"] += bool(row["edits"])
            edit_counts.update(e["kind"] for e in row["edits"])
    return dict(version=VERSION, sources=sources, counts=dict(counts), edits=dict(edit_counts),
                policy="All requested turns retained; only reviewed spelling changes applied.")


def write_rows(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalized_inputs(source):
    """Reassemble the exact dialogue/turn structure from the normalized snapshot."""
    files = {}
    with Path(source).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            origin = row["origin"]
            dialogues = files.setdefault(origin["file"], [])
            if origin["dialogue"] == len(dialogues) + 1:
                dialogues.append([])
            if origin["dialogue"] != len(dialogues) or origin["turn"] != len(dialogues[-1]) + 1:
                raise ValueError("nonsequential source dialogue or turn")
            dialogues[-1].append(row["example"])
    return {name: ("".join(json.dumps(turns, ensure_ascii=False) + "\n" for turns in dialogues)).encode("utf-8")
            for name, dialogues in files.items()}


def archive_originals(directory, output, preparation):
    """Preserve exact input bytes as audit attachments before any in-place edits."""
    saved = Path(output) / "originals"
    saved.mkdir(exist_ok=False)
    for item in preparation["sources"]:
        content = (Path(directory) / item["file"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("input changed during mining; refuse to archive a different version")
        (saved / item["file"]).write_bytes(content)


def update_inputs(directory, source, preparation, *, published_bank, bank_id):
    """Replace filtered JSONL only after the original bytes are durably published."""
    from zipfile import ZipFile

    revised = normalized_inputs(source)
    staged = []
    with ZipFile(Path(published_bank) / "sources" / f"{bank_id}.zip") as archive:
        for item in preparation["sources"]:
            path = Path(directory) / item["file"]
            original = path.read_bytes()
            if hashlib.sha256(original).hexdigest() != item["sha256"] or original != archive.read("originals/" + item["file"]):
                raise ValueError("original source differs from the published audit copy")
    try:
        for item in preparation["sources"]:
            path = Path(directory) / item["file"]
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".dailydialog-", delete=False) as tmp:
                staged.append((Path(tmp.name), path))
                tmp.write(revised[item["file"]])
                tmp.flush()
                os.fsync(tmp.fileno())
        # Check all originals again before replacing the first file.
        for item in preparation["sources"]:
            if hashlib.sha256((Path(directory) / item["file"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("input changed before replacement")
        for tmp, destination in staged:
            os.replace(tmp, destination)
    finally:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
    return {name: hashlib.sha256(content).hexdigest() for name, content in revised.items()}


class ResourceCollector:
    """Wrap a spaCy pipeline so compilation and entity extraction share a parse."""

    def __init__(self, nlp, curation):
        from .semantic_generator.substitutions import entity_from_json

        self.nlp, self.meta = nlp, nlp.meta
        self.curation = curation
        self.entities, self.candidates, self.nominals, self.adverbs = {}, {}, {}, {}
        self.reviewed = {}
        for entity in curation.get("entities", []):
            parsed = entity_from_json(entity)
            if not parsed.text or not parsed.is_named or parsed.entity_label is None:
                raise ValueError("reviewed entity must be named and typed")
            self.reviewed.setdefault(parsed.text, []).append(entity)
        names = sorted(self.reviewed, key=lambda s: (-len(s), s))
        self.name_pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, names)) + r")(?!\w)") if names else None

    @staticmethod
    def _observe(table, key, fields, text, span, provenance):
        row = table.setdefault(key, {**fields, "count": 0, "examples": []})
        row["count"] += 1
        if len(row["examples"]) < 3:
            row["examples"].append({**provenance, "text": text, "span": span})

    def pipe(self, texts, **kwargs):
        for doc, provenance in self.nlp.pipe(texts, **kwargs):
            for span in doc.ents:
                self._observe(self.candidates, (span.text, span.label_),
                              dict(text=span.text, parser_label=span.label_, review_status="UNREVIEWED"),
                              doc.text, [span.start_char, span.end_char], provenance)
            for token in doc:
                if token.pos_ == "PROPN" and not token.ent_type_:
                    self._observe(self.candidates, (token.text, "PROPN"),
                                  dict(text=token.text, parser_label="PROPN", review_status="UNREVIEWED"),
                                  doc.text, [token.idx, token.idx + len(token)], provenance)
                if token.pos_ == "ADV":
                    self._observe(self.adverbs, (token.lemma_, token.text),
                                  dict(lemma=token.lemma_, text=token.text, review_status="UNREVIEWED"),
                                  doc.text, [token.idx, token.idx + len(token)], provenance)
            for span in doc.noun_chunks:
                self._observe(self.nominals, (span.text, span.root.lemma_, span.root.dep_),
                              dict(text=span.text, head=span.root.lemma_, dependency=span.root.dep_,
                                   review_status="UNREVIEWED"),
                              doc.text, [span.start_char, span.end_char], provenance)
            if self.name_pattern is not None:
                for match in self.name_pattern.finditer(doc.text):
                    for entity in self.reviewed[match.group()]:
                        self._observe(self.entities, (entity["text"], tuple(entity["labels"])),
                                      dict(entity=entity, review_status="REVIEWED", review_method="curated_reading"),
                                      doc.text, list(match.span()), provenance)
            yield doc, provenance

    def save(self, output):
        counts = {}
        for name in ("entities", "entity_candidates", "nominals", "adverbs"):
            table = self.candidates if name == "entity_candidates" else getattr(self, name)
            rows = [row for _, row in sorted(table.items())]
            write_rows(Path(output) / f"{name}.jsonl.gz", rows)
            counts[name] = len(rows)
        return counts


def enrich_bank(source, output, work, nlp, *, wiktionary, wordnet=None, curation=None,
                preparation=None):
    from .template_compiler import build_bank
    from .template_bank import TemplateBank, fingerprint
    from .template_validation import validate_bank

    curation = curation if curation is not None else load_curation()
    collector = ResourceCollector(nlp, curation)
    manifest = build_bank(source, output, work, collector, wiktionary=wiktionary,
                          wordnet=wordnet, batch_size=256)
    output = Path(output)
    manifest["counts"].update(collector.save(output))
    from .template_bank import read_rows
    manifest["resource_inventories"] = {}
    for name in ("entities", "entity_candidates", "nominals", "adverbs"):
        digest = hashlib.sha256()
        for row in read_rows(output / f"{name}.jsonl.gz"):
            digest.update(bytes.fromhex(fingerprint(row)))
        manifest["resource_inventories"][name] = digest.hexdigest()
    manifest["enrichment"] = dict(version=VERSION, curation_sha256=fingerprint(curation),
                                  preparation=preparation, entity_policy="explicit reviewed named readings only")
    for name, value in (("manifest.json", manifest), ("curation.json", curation),
                        ("preparation.json", preparation)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = validate_bank(TemplateBank(output), source)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return manifest
