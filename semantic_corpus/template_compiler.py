"""Compile every input record into an executable, lossless surface template.

All tokens are audited against both the parser and the shipped lexical
resources. Dictionary homographs remain alternatives, never automatic gold.
Every old inventory entry is checked for coverage when --previous is supplied.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
from zipfile import ZipFile

from .qasrl_core.inflections import load_inflections, load_postags
from .syntax_mining import extract
from .template_bank import CorpusTemplate, FORMAT, fingerprint, read_rows


def _write(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


class LexicalAudit:
    def __init__(self, directory, wordnet=None):
        directory = Path(directory)
        self.verbs = load_inflections(directory / "en_verb_inflections.txt")
        self.postags = load_postags(directory / "en_postags_withverb.txt")
        self.cache = {}
        self.wordnet = {"verb": set(), "adj": set()}
        self.exceptions = {"verb": {}, "adj": {}}
        if wordnet is not None:
            with ZipFile(wordnet) as archive:
                for pos in ("verb", "adj"):
                    lines = archive.read(f"wordnet/index.{pos}").decode("utf-8").splitlines()
                    self.wordnet[pos] = {line.split()[0] for line in lines if line and not line[0].isspace()}
                    for line in archive.read(f"wordnet/{pos}.exc").decode("utf-8").splitlines():
                        form, *lemmas = line.split()
                        self.exceptions[pos][form] = lemmas

    def wordnet_lemmas(self, word, pos):
        candidates = [word, *self.exceptions[pos].get(word, ())]
        rules = (("er", ""), ("est", ""), ("er", "e"), ("est", "e")) if pos == "adj" else (
            ("s", ""), ("ies", "y"), ("es", "e"), ("es", ""), ("ed", "e"),
            ("ed", ""), ("ing", "e"), ("ing", ""))
        candidates.extend(word[:-len(old)] + new for old, new in rules if word.endswith(old))
        return [lemma for lemma in dict.fromkeys(candidates) if lemma in self.wordnet[pos]]

    def alternatives(self, surface):
        key = surface.lower()
        if key not in self.cache:
            adjectives = self.wordnet_lemmas(key, "adj")
            if "adj" in self.postags.get(key, ()) and key not in adjectives:
                adjectives.append(key)
            self.cache[key] = {
                "verb_lemmas": list(dict.fromkeys((*self.verbs.lemmas_for(key), *self.wordnet_lemmas(key, "verb")))),
                "adjective_lemmas": adjectives,
            }
        return self.cache[key]

    def recovery(self, doc):
        """Expose missed bare imperatives as an explicit alternative analysis.

        Fires per clause, even if another sentence in the record has a verb.
        It does not pretend that ambiguous 'Fire!' cannot be a nominal warning.
        """
        found = {}
        for sentence in doc.sents:
            if any(t.pos_ in ("VERB", "AUX") for t in sentence):
                continue
            words = [t for t in sentence if not t.is_punct and not t.is_space]
            if not words:
                continue
            candidate = words[1] if len(words) > 1 and words[0].lower_ == "please" else words[0]
            if candidate.pos_ in ("DET", "PRON", "ADP", "CCONJ", "SCONJ"):
                continue
            if self.verbs.paradigms(candidate.lower_) or candidate.lower_ in self.wordnet["verb"]:
                found[candidate.i] = candidate.lower_
        return found


def compile_record(doc, source, audit):
    observation = extract(doc)
    recovered = audit.recovery(doc)
    token_slots = {}
    slots, defaults, parts = {}, {}, []
    tokens, attributes = [], []
    cursor = 0
    counts = Counter()
    for token in doc:
        alternatives = audit.alternatives(token.text)
        kind = ("verb" if token.pos_ in ("VERB", "AUX") or token.i in recovered
                else "adjective" if token.pos_ == "ADJ" or token.tag_ in ("JJ", "JJR", "JJS")
                else "nominal" if token.pos_ in ("NOUN", "PROPN", "PRON") else None)
        lemma = recovered.get(token.i, token.lemma_.lower() or token.lower_)
        item = {"i": token.i, "text": token.text, "span": [token.idx, token.idx + len(token)],
                "lemma": lemma, "parser_lemma": token.lemma_.lower() or token.lower_,
                "pos": token.pos_, "tag": token.tag_, "dep": token.dep_,
                "head": token.head.i, "morph": str(token.morph), "kind": kind,
                "analysis": "imperative_alternative" if token.i in recovered else "parser",
                **alternatives}
        tokens.append(item)
        if kind is None:
            continue
        key = kind[0] + str(counts[kind])
        counts[kind] += 1
        token_slots[token.i] = key
        descriptor = {"kind": kind, "pos": token.pos_, "tag": token.tag_,
                      "morph": str(token.morph), "dependency": token.dep_,
                      "analysis": item["analysis"]}
        if kind == "verb":
            descriptor["lemma"] = lemma
        slots[key] = descriptor
        defaults[key] = token.text
        if token.idx > cursor:
            parts.append({"literal": doc.text[cursor:token.idx]})
        parts.append({"slot": key})
        cursor = token.idx + len(token)
    if cursor < len(doc.text):
        parts.append({"literal": doc.text[cursor:]})
    if not parts:
        parts = [{"literal": doc.text}]

    for token in doc:
        key = token_slots.get(token.i)
        if key:
            slots[key]["head_slot"] = token_slots.get(token.head.i)
            if token.dep_ == "pobj" and token.head.pos_ == "ADP":
                # ADPs remain literal text. Preserve their governor separately
                # so consumers can distinguish verb-attached by from noun-
                # attached by without inferring attachment from passive voice.
                slots[key]["preposition"] = {
                    "text": token.head.lower_,
                    "head_slot": token_slots.get(token.head.head.i),
                }
        if key and (token.pos_ == "ADJ" or token.tag_ in ("JJ", "JJR", "JJS")):
            target = token.head if token.dep_ == "amod" else next(
                (t for t in token.head.children if t.dep_ in ("nsubj", "nsubjpass")), None)
            attributes.append({"slot": key, "target_slot": token_slots.get(target.i) if target else None,
                               "degree": token.morph.get("Degree"),
                               "modifiers": [{"text": t.text, "span": [t.idx, t.idx + len(t)]}
                                             for t in token.children if t.dep_ in ("advmod", "neg")]})

    groups = []
    for predicate in observation["predicates"]:
        head = next((t for t in doc if t.idx == predicate["span"][0]), None)
        if head is None:
            raise AssertionError("predicate head disappeared")
        groups.append({"head_slot": token_slots.get(head.i), "lemma": predicate["lexical_unit"],
                       "constructions": predicate["constructions"]})
    for index, lemma in recovered.items():
        groups.append({"head_slot": token_slots[index], "lemma": lemma,
                       "constructions": [], "analysis": "imperative_alternative"})
    # Markers are source offsets in the audit. Runtime groups use template parts
    # and token slots, so replacement lengths never invalidate their positions.
    runtime_groups = []
    for group in groups:
        constructions = []
        for c in group["constructions"]:
            markers = []
            for start, end in c["markers"]:
                token = next(t for t in doc if t.idx == start and t.idx + len(t) == end)
                slot = token_slots.get(token.i)
                if slot:
                    markers.append({"slot": slot})
                else:
                    # A marker may be inside a literal containing whitespace or
                    # punctuation. Preserve its range within that literal part.
                    position = 0
                    for part_index, part in enumerate(parts):
                        value = part.get("literal", defaults.get(part.get("slot"), ""))
                        if position <= start and end <= position + len(value):
                            markers.append({"part": part_index, "start": start-position, "end": end-position})
                            break
                        position += len(value)
                    else:
                        raise AssertionError("cannot locate predicate marker in template")
            constructions.append({"lemma": c["lemma"], "kind": c["kind"],
                                  "separated": c["separated"], "markers": markers})
        runtime_groups.append({"head_slot": group["head_slot"], "lemma": group["lemma"],
                               "constructions": constructions})
    verbs = sorted({t["lemma"] for t in tokens if t["kind"] == "verb"})
    constructions = sorted({c["lemma"] for group in groups for c in group["constructions"]})
    # Attribute offsets vary with bound text and live in the occurrence ledger;
    # only stable target/degree metadata belongs to the template shape.
    shape = {"parts": parts, "slots": slots, "verbs": verbs, "constructions": constructions,
             "predicate_groups": runtime_groups,
             "attributes": [{k: v for k, v in a.items() if k != "modifiers"} for a in attributes]}
    record = {"template_id": fingerprint(shape), **shape, "defaults": defaults,
              "adjective_lemmas": sorted({t["parser_lemma"] for t in tokens if t["pos"] == "ADJ" or t["tag"] in ("JJ", "JJR", "JJS")})}
    if CorpusTemplate(record).render()["text"] != doc.text:
        raise AssertionError(f"record {source}: template roundtrip lost source text")
    status = ("imperative_alternative_added" if recovered else "parsed_verbal" if verbs
              else "lexical_alternatives_recorded" if any(t["verb_lemmas"] for t in tokens)
              else "no_verbal_analysis")
    ledger = {**source, "text": doc.text, "template_id": record["template_id"],
              "bindings": defaults, "tokens": tokens, "attributes": attributes,
              "predicate_groups": groups, "analysis_status": status,
              "sentence_template": observation["sentence_template"],
              "predicate_templates": [{"lemma": p["lexical_unit"], "template": p["template"]}
                                      for p in observation["predicates"]]}
    return record, ledger


class WordInventory:
    def __init__(self):
        self.rows = {}

    def add(self, lemma, token, source, analysis):
        row = self.rows.setdefault(lemma, {"lemma": lemma, "forms": {}, "analyses": {}, "morphology": {}, "dependencies": {}, "examples": []})
        row["forms"][token["text"]] = row["forms"].get(token["text"], 0) + 1
        row["analyses"][analysis] = row["analyses"].get(analysis, 0) + 1
        for field, value in (("morphology", token["morph"]), ("dependencies", token["dep"])):
            row[field][value] = row[field].get(value, 0) + 1
        if len(row["examples"]) < 3:
            row["examples"].append({**source, "span": token["span"]})

    def write(self, path, lexicon=None):
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for lemma, row in sorted(self.rows.items()):
                if lexicon is not None:
                    row["paradigms"] = [p.to_json() for p in lexicon.paradigms(lemma)]
                    row["inflection_source"] = "dictionary" if row["paradigms"] else "observed_forms_only"
                _write(stream, row)


def build_bank(source, output, work, nlp, *, wiktionary, wordnet=None, previous=None, field="example", limit=None,
               batch_size=512, progress=None):
    source, output, work = Path(source), Path(output), Path(work)
    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    (output / "templates").mkdir()
    (output / "occurrences").mkdir()
    started = time.perf_counter()
    counts, statuses = Counter(), Counter()
    digest = hashlib.sha256()
    audit = LexicalAudit(wiktionary, wordnet)
    verbs, adjectives, verb_alternatives, adjective_alternatives = (WordInventory() for _ in range(4))
    seen_sentences, seen_predicates = set(), set()
    constructions = {}
    manifest = {"format": FORMAT, "status": "building", "source": str(source.resolve()),
                "full_input": limit is None, "model": nlp.meta, "spacy_version": version("spacy"),
                "review_status": "UNREVIEWED", "field": field,
                "note": "Executable surface patterns; automatic syntax is not verified semantic role annotation."}
    if wordnet is not None:
        manifest["wordnet_sha256"] = hashlib.sha256(Path(wordnet).read_bytes()).hexdigest()
        with ZipFile(wordnet) as archive:
            (output / "WORDNET-LICENSE.txt").write_bytes(archive.read("wordnet/LICENSE"))

    def save(status):
        manifest.update(status=status, counts=dict(counts), analyses=dict(statuses),
                        elapsed_seconds=round(time.perf_counter() - started, 3))
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    save("building")
    try:
        if progress:
            progress("Compiling templates, verbs and adjectives", 0)

        def texts():
            with source.open("rb") as stream:
                for number, line in enumerate(stream, 1):
                    if limit is not None and number > limit:
                        break
                    counts["input_records"] += 1
                    digest.update(line)
                    row = json.loads(line)
                    text = row[field]
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"line {number}: expected nonempty text")
                    yield text, {"line": number, "idx": row.get("idx")}

        with ExitStack() as stack:
            shards = {}
            ledger_stream = None
            for doc, provenance in nlp.pipe(texts(), as_tuples=True, batch_size=batch_size):
                if counts["compiled_records"] % 10000 == 0:
                    if ledger_stream is not None:
                        ledger_stream.close()
                    ledger_stream = stack.enter_context(gzip.open(
                        output / "occurrences" / f"{counts['compiled_records']//10000:05d}.jsonl.gz",
                        "wt", encoding="utf-8", compresslevel=3))
                template, ledger = compile_record(doc, provenance, audit)
                key = template["template_id"][:2]
                if key not in shards:
                    shards[key] = stack.enter_context(gzip.open(work / f"{key}.jsonl.gz", "wt", encoding="utf-8", compresslevel=1))
                _write(shards[key], {"template": template, "source": provenance})
                _write(ledger_stream, ledger)
                counts["compiled_records"] += 1
                if progress and counts["compiled_records"] % batch_size == 0:
                    progress("Compiling templates, verbs and adjectives", counts["compiled_records"])
                counts["roundtrips_verified"] += 1
                counts["tokens_audited"] += len(ledger["tokens"])
                statuses[ledger["analysis_status"]] += 1
                seen_sentences.add(fingerprint(ledger["sentence_template"]))
                seen_predicates.update(fingerprint(p) for p in ledger["predicate_templates"])
                for group in ledger["predicate_groups"]:
                    for construction in group["constructions"]:
                        key = (construction["kind"], construction["lemma"])
                        row = constructions.setdefault(key, {"kind": key[0], "lemma": key[1], "count": 0, "examples": []})
                        row["count"] += 1
                        if len(row["examples"]) < 3:
                            row["examples"].append({**provenance, "template_id": template["template_id"]})
                example = {**provenance, "text": doc.text}
                for token in ledger["tokens"]:
                    if token["kind"] == "verb":
                        counts["verb_occurrences"] += 1
                        verbs.add(token["lemma"], token, example, token["analysis"])
                    is_adjective = token["pos"] == "ADJ" or token["tag"] in ("JJ", "JJR", "JJS")
                    if is_adjective:
                        counts["adjective_occurrences"] += 1
                        adjectives.add(token["parser_lemma"], token, example, "parser")
                    for lemma in token["verb_lemmas"]:
                        if token["kind"] != "verb" or lemma != token["lemma"]:
                            verb_alternatives.add(lemma, token, example, "dictionary_homograph")
                            counts["verb_alternative_analyses"] += 1
                    for lemma in token["adjective_lemmas"]:
                        if not is_adjective or lemma != token["parser_lemma"]:
                            adjective_alternatives.add(lemma, token, example, "dictionary_homograph")
                            counts["adjective_alternative_analyses"] += 1
                    if token["tag"] in ("VBN", "VBG"):
                        adjective_alternatives.add(token["text"].lower(), token, example, "participle_reading")
                        counts["participle_analyses"] += 1
                if counts["compiled_records"] % 10000 == 0:
                    save("building")
                    print(f"Compiled {counts['compiled_records']:,} records / {counts['tokens_audited']:,} tokens", flush=True)

        # Consolidate one partition at a time: memory does not scale with all
        # distinct templates or all original sentences at once.
        if progress:
            progress("Consolidating mined inventories", counts["compiled_records"])
        for shard in sorted(work.glob("*.jsonl.gz")):
            rows = {}
            for item in read_rows(shard):
                template = item["template"]
                key = template["template_id"]
                if key not in rows:
                    rows[key] = {**template, "count": 0, "examples": [], "fillers": {
                        name: set() for name in template["slots"]}, "attested_bindings": {},
                        "adjective_lemmas": set()}
                row = rows[key]
                row["count"] += 1
                row["adjective_lemmas"].update(template["adjective_lemmas"])
                for name, value in template["defaults"].items():
                    row["fillers"][name].add(value)
                row["attested_bindings"][fingerprint(template["defaults"])] = template["defaults"]
                if len(row["examples"]) < 3:
                    row["examples"].append(item["source"])
            with gzip.open(output / "templates" / shard.name, "wt", encoding="utf-8") as sink:
                for _, row in sorted(rows.items()):
                    row["fillers"] = {name: sorted(values) for name, values in row["fillers"].items()}
                    row["attested_bindings"] = list(row["attested_bindings"].values())
                    row["adjective_lemmas"] = sorted(row["adjective_lemmas"])
                    _write(sink, row)
                    counts["templates"] += 1
                    counts["template_occurrences"] += row["count"]
                    counts["templates_with_adjectives"] += bool(row["attributes"])
            save("building")
        verbs.write(output / "verbs.jsonl.gz", audit.verbs)
        adjectives.write(output / "adjectives.jsonl.gz")
        verb_alternatives.write(output / "verb_alternatives.jsonl.gz", audit.verbs)
        adjective_alternatives.write(output / "adjective_alternatives.jsonl.gz")
        with gzip.open(output / "constructions.jsonl.gz", "wt", encoding="utf-8") as stream:
            for _, row in sorted(constructions.items()):
                _write(stream, row)
        counts.update(verb_lemmas=len(verbs.rows), adjective_lemmas=len(adjectives.rows),
                      verb_alternative_lemmas=len(verb_alternatives.rows),
                      adjective_alternative_forms=len(adjective_alternatives.rows),
                      sentence_templates_covered=len(seen_sentences),
                      predicate_templates_covered=len(seen_predicates), constructions=len(constructions))
        if counts["compiled_records"] != counts["input_records"] or counts["template_occurrences"] != counts["compiled_records"]:
            raise AssertionError("record coverage is incomplete")
        manifest["source_sha256"] = digest.hexdigest()
        if previous:
            previous = Path(previous)
            old = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
            if old["input_sha256"] != manifest["source_sha256"]:
                raise ValueError("previous analysis used different input bytes")
            missing = {"verbs": 0, "sentence_templates": 0, "predicate_templates": 0, "constructions": 0}
            for row in read_rows(previous / "verbs.jsonl"):
                missing["verbs"] += row["lemma"] not in verbs.rows
            for row in read_rows(previous / "sentence_templates.jsonl"):
                missing["sentence_templates"] += fingerprint(row["template"]) not in seen_sentences
            for row in read_rows(previous / "predicate_templates.jsonl"):
                missing["predicate_templates"] += fingerprint({"lemma": row["lemma"], "template": row["template"]}) not in seen_predicates
            for row in read_rows(previous / "multiword_verbs.jsonl"):
                missing["constructions"] += (row["kind"], row["lemma"]) not in constructions
            manifest["previous_catalogue_missing"] = missing
            if any(missing.values()):
                raise AssertionError(f"previous catalogue entries are missing: {missing}")
        save("complete")
    except BaseException:
        save("failed")
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--wiktionary", type=Path, default=Path("data/wiktionary"))
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--wordnet", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    import spacy
    nlp = spacy.load(args.model, disable=["ner"])
    if nlp.lang != "en" or not {"parser", "lemmatizer"} <= set(nlp.pipe_names):
        parser.error("an English dependency parser and lemmatizer are required")
    result = build_bank(args.source, args.output, args.work, nlp, wiktionary=args.wiktionary,
                        wordnet=args.wordnet, previous=args.previous, limit=args.limit)
    print(json.dumps({k: result[k] for k in ("status", "counts", "analyses", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
