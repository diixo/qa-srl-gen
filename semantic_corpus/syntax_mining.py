"""Mine observed syntax, not verified semantic frames, from a JSONL corpus.

Optional dependency: spaCy and an English dependency-parser model. Run with
``python -m semantic_corpus.syntax_mining INPUT --output DIR --model MODEL``.
The occurrence ledger retains every parsed record, including verbless texts;
catalogues aggregate observations without imposing a frequency threshold.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

EXTRACTOR_VERSION = "2"


def enrich(result):
    """Keep adverbial candidates such as look forward to, without calling
    ordinary adverbs particles. Works on the saved dependency ledger too.
    """
    for predicate in result["predicates"]:
        base = [c for c in predicate["constructions"] if not c["kind"].startswith("adverb")]
        predicate["constructions"] = base.copy()
        for argument in predicate["arguments"]:
            if argument["dependency"] != "advmod" or argument["head_span"][0] < predicate["span"][1]:
                continue
            adverb = argument["head"].lower()
            unit = predicate["lexical_unit"] + " " + adverb
            marker = argument["head_span"]
            predicate["constructions"].append({
                "lemma": unit, "kind": "adverb_candidate", "separated": False,
                "markers": [marker],
            })
            for construction in base:
                if (construction["kind"] == "preposition_candidate"
                        and marker[1] <= construction["markers"][0][0]):
                    tail = construction["lemma"][len(predicate["lexical_unit"]):]
                    predicate["constructions"].append({
                        "lemma": unit + tail, "kind": "adverb_preposition_candidate",
                        "separated": False, "markers": [marker, *construction["markers"]],
                    })
    return result


def _span(token):
    return [token.idx, token.idx + len(token.text)]


def _prep_template(token):
    parts = [token.lower_]
    for child in token.children:
        if child.dep_ == "prep":
            parts.append(_prep_template(child))
        elif not child.is_punct:
            parts.append("{" + child.dep_.upper() + "}")
    return " ".join(parts)


def extract(doc):
    """Extract dependency observations; ``prt`` is distinct from ``prep``.

    No dictionary guessing: ambiguous prepositional attachments remain
    candidates, and semantic idiomaticity is never inferred from adjacency.
    Auxiliaries appear in the verb inventory too, tagged with their POS.
    """
    verbs, predicates = [], []
    for token in doc:
        if token.pos_ not in ("VERB", "AUX"):
            continue
        lemma = token.lemma_.lower() or token.lower_
        verbs.append({"lemma": lemma, "surface": token.text, "pos": token.pos_,
                      "span": _span(token)})
        if token.dep_ in ("aux", "auxpass", "cop"):
            continue
        children = sorted(token.children, key=lambda t: t.i)
        particles = [t for t in children if t.dep_ == "prt"]
        preps = [t for t in children if t.dep_ == "prep"]
        objects = [t for t in children if t.dep_ in ("dobj", "obj")]
        unit = " ".join([lemma, *(t.lower_ for t in particles)])
        ordered = [(token.i, "{VERB}")]
        arguments = []
        for child in children:
            dep = child.dep_
            if child.is_punct:
                continue
            if dep in ("aux", "auxpass", "neg", "prt"):
                wording = child.lower_
            elif dep == "prep":
                wording = _prep_template(child)
            else:
                wording = "{" + dep.upper() + "}"
            ordered.append((child.i, wording))
            arguments.append({"dependency": dep, "head": child.text,
                              "head_span": _span(child)})
        constructions = []
        if particles:
            separated = any(token.i < obj.i < part.i
                            for obj in objects for part in particles)
            constructions.append({"lemma": unit, "kind": "particle",
                                  "separated": separated,
                                  "markers": [_span(t) for t in particles]})
        for prep in preps:
            # Include chained prepositions (get out of, put up with).
            chain = [prep]
            while True:
                nested = [c for c in chain[-1].children if c.dep_ == "prep"]
                if len(nested) != 1:
                    break
                chain.append(nested[0])
            constructions.append({
                "lemma": " ".join([unit, *(t.lower_ for t in chain)]),
                "kind": "particle_preposition" if particles else "preposition_candidate",
                "separated": False,
                "markers": [_span(t) for t in (*particles, *chain)],
            })
        predicates.append({
            "lemma": lemma, "lexical_unit": unit, "surface": token.text,
            "span": _span(token), "dependency": token.dep_,
            "morphology": str(token.morph),
            "template": " ".join(word for _, word in sorted(ordered)),
            "arguments": arguments, "constructions": constructions,
        })

    # These placeholders describe syntax, never ontology labels or QA roles.
    chunks = {chunk.start: chunk for chunk in doc.noun_chunks}
    words, index = [], 0
    while index < len(doc):
        chunk = chunks.get(index)
        if chunk is not None:
            words.append("{NP:" + chunk.root.dep_.upper() + "}")
            index = chunk.end
            continue
        token = doc[index]
        words.append("{VERB}" if token.pos_ == "VERB" else token.lower_)
        index += 1
    return enrich({"verbs": verbs, "predicates": predicates,
                   "sentence_template": " ".join(words)})


def _dump(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


class Catalogue:
    def __init__(self):
        self.rows = {}

    def add(self, key, fields, source):
        row = self.rows.setdefault(key, {**fields, "count": 0, "examples": []})
        row["count"] += 1
        if len(row["examples"]) < 3 and source not in row["examples"]:
            row["examples"].append(source)
        return row

    def write(self, path):
        _dump(path, (row for _, row in sorted(
            self.rows.items(), key=lambda pair: (-pair[1]["count"], pair[0]))))


class Inventory:
    def __init__(self, counts):
        self.counts = counts
        self.verbs, self.patterns, self.phrases, self.sentences = (Catalogue() for _ in range(4))
        self.review_queue = []

    def observe(self, result):
        self.counts["parsed"] += 1
        self.counts["verb_occurrences"] += len(result["verbs"])
        self.counts["predicate_occurrences"] += len(result["predicates"])
        if not result["verbs"]:
            self.counts["verbless_records"] += 1
            self.review_queue.append({key: result[key] for key in ("line", "idx", "text")} | {
                "reason": "no_verb_detected", "review_status": "UNREVIEWED"})
        example = {key: result[key] for key in ("line", "idx", "text")}
        for verb in result["verbs"]:
            row = self.verbs.add(verb["lemma"], {"lemma": verb["lemma"], "forms": {}, "pos": {}}, example)
            for key, value in (("forms", verb["surface"]), ("pos", verb["pos"])):
                row[key][value] = row[key].get(value, 0) + 1
        for predicate in result["predicates"]:
            self.patterns.add((predicate["lexical_unit"], predicate["template"]),
                              {"lemma": predicate["lexical_unit"], "template": predicate["template"]}, example)
            for construction in predicate["constructions"]:
                self.counts["construction_occurrences"] += 1
                row = self.phrases.add((construction["kind"], construction["lemma"]),
                                      {"lemma": construction["lemma"], "kind": construction["kind"],
                                       "separated_count": 0, "separated_examples": []}, example)
                if construction["separated"]:
                    row["separated_count"] += 1
                    if len(row["separated_examples"]) < 3 and example not in row["separated_examples"]:
                        row["separated_examples"].append(example)
        self.sentences.add(result["sentence_template"], {"template": result["sentence_template"]}, example)

    def write(self, output):
        _dump(output / "review_queue.jsonl", self.review_queue)
        for name, catalogue in (("verbs", self.verbs), ("predicate_templates", self.patterns),
                                ("multiword_verbs", self.phrases), ("sentence_templates", self.sentences)):
            catalogue.write(output / f"{name}.jsonl")
            self.counts[name] = len(catalogue.rows)


def mine(path, output, nlp, *, field="example", batch_size=512, workers=1, limit=None):
    """Stream the whole input; refuse to replace a previous audit directory."""
    path, output = Path(path), Path(output)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    digest = hashlib.sha256()
    counts = Counter()
    inventory = Inventory(counts)
    manifest = {"format": "syntax-candidates-v1", "extractor_version": EXTRACTOR_VERSION,
                "status": "running",
                "input": str(path.resolve()), "field": field, "limit": limit,
                "model": nlp.meta, "pipeline": nlp.pipe_names,
                "spacy_version": version("spacy"),
                "batch_size": batch_size, "workers": workers,
                "review_status": "UNREVIEWED",
                "note": "Statistical parses; not semantic frames or verified QA labels."}
    manifest_path = output / "manifest.json"

    def save(status):
        manifest.update(status=status, counts=dict(counts),
                        elapsed_seconds=round(time.perf_counter() - started, 3))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")

    save("running")
    try:
        with (output / "rejected.jsonl").open("w", encoding="utf-8") as rejected:
            def inputs():
                with path.open("rb") as source:
                    for line_no, raw in enumerate(source, 1):
                        if limit is not None and counts["lines"] >= limit:
                            break
                        digest.update(raw)
                        counts["lines"] += 1
                        try:
                            record = json.loads(raw)
                            text = record[field]
                            if not isinstance(text, str) or not text.strip():
                                raise ValueError("text must be a nonempty string")
                            yield text, {"line": line_no, "idx": record.get("idx")}
                        except (ValueError, KeyError, TypeError) as error:
                            counts["rejected"] += 1
                            rejected.write(json.dumps({"line": line_no, "error": str(error)}) + "\n")

            with gzip.open(output / "occurrences.jsonl.gz", "wt", encoding="utf-8") as ledger:
                for doc, source in nlp.pipe(inputs(), as_tuples=True,
                                           batch_size=batch_size, n_process=workers):
                    result = {**source, "text": doc.text, **extract(doc)}
                    inventory.observe(result)
                    ledger.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if counts["parsed"] % 10000 == 0:
                        save("running")
                        print(f"Parsed {counts['parsed']:,} records in {time.perf_counter()-started:.1f}s", flush=True)
        inventory.write(output)
        manifest["input_sha256"] = digest.hexdigest()
        manifest["full_input"] = limit is None
        save("complete_with_rejections" if counts["rejected"] else "complete")
    except BaseException:
        save("failed")
        raise
    return manifest


def reindex(source, output):
    """Rebuild catalogues from a completed dependency ledger, without spaCy."""
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "syntax-candidates-v1" or manifest.get("status") not in (
            "complete", "complete_with_rejections"):
        raise ValueError("reindex requires a completed syntax-candidates-v1 ledger")
    expected = manifest["counts"]["parsed"]
    counts = Counter({key: manifest["counts"].get(key, 0) for key in ("lines", "rejected")})
    inventory = Inventory(counts)
    started = time.perf_counter()
    previous_elapsed = manifest["elapsed_seconds"]
    output.mkdir(parents=True, exist_ok=False)
    manifest.update(extractor_version=EXTRACTOR_VERSION, reindexed_from=str(source.resolve()))

    def save(status):
        elapsed = round(time.perf_counter() - started, 3)
        manifest.update(status=status, counts=dict(counts), reindex_seconds=elapsed,
                        elapsed_seconds=round(previous_elapsed + elapsed, 3))
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")

    save("running")
    try:
        shutil.copyfile(source / "rejected.jsonl", output / "rejected.jsonl")
        with gzip.open(source / "occurrences.jsonl.gz", "rt", encoding="utf-8") as rows:
            with gzip.open(output / "occurrences.jsonl.gz", "wt", encoding="utf-8") as ledger:
                for line in rows:
                    result = enrich(json.loads(line))
                    inventory.observe(result)
                    ledger.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if counts["parsed"] % 100000 == 0:
                        save("running")
                        print(f"Reindexed {counts['parsed']:,} records", flush=True)
        if counts["parsed"] != expected or counts["parsed"] + counts["rejected"] != counts["lines"]:
            raise ValueError("ledger record count does not match its manifest")
        inventory.write(output)
        save("complete_with_rejections" if counts["rejected"] else "complete")
    except BaseException:
        save("failed")
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--field", default="example")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Debug sample only; omit for full corpus")
    parser.add_argument("--reindex", action="store_true", help="Input is a completed analysis directory; reuse its parses")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.workers <= 0:
        parser.error("batch-size and workers must be positive")
    if args.reindex:
        if args.limit is not None:
            parser.error("--limit is not supported with --reindex")
        result = reindex(args.input, args.output)
    else:
        import spacy
        nlp = spacy.load(args.model, disable=["ner"])
        if nlp.lang != "en" or not {"parser", "lemmatizer"} <= set(nlp.pipe_names):
            parser.error("an English dependency parser with a lemmatizer is required")
        result = mine(args.input, args.output, nlp, field=args.field,
                      batch_size=args.batch_size, workers=args.workers, limit=args.limit)
    print(json.dumps({key: result[key] for key in ("status", "counts", "elapsed_seconds")}, indent=2))
    return 1 if result["counts"].get("rejected") else 0


if __name__ == "__main__":
    raise SystemExit(main())
