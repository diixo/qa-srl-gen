"""Parser-independent accounting, plus controlled dependency trees when spaCy is installed."""

import gzip
import hashlib
import json
from collections import Counter

import pytest

from semantic_corpus.syntax_mining import Inventory, enrich, extract, mine, reindex


def parsed(words, heads, deps, pos, lemmas):
    spacy = pytest.importorskip("spacy")
    from spacy.tokens import Doc
    return Doc(spacy.blank("en").vocab, words=words, heads=heads, deps=deps, pos=pos, lemmas=lemmas)


def sample():
    return parsed(["Anna", "picked", "the", "book", "up", "."],
                  [1, 1, 3, 1, 1, 1], ["nsubj", "ROOT", "det", "dobj", "prt", "punct"],
                  ["PROPN", "VERB", "DET", "NOUN", "ADP", "PUNCT"],
                  ["Anna", "pick", "the", "book", "up", "."])


def test_particle_can_be_separated_and_has_offsets():
    doc = sample()
    item, = extract(doc)["predicates"]
    assert item["lemma"] == "pick" and item["lexical_unit"] == "pick up"
    assert item["template"] == "{NSUBJ} {VERB} {DOBJ} up"
    phrase, = item["constructions"]
    assert phrase["kind"] == "particle" and phrase["separated"]
    assert [doc.text[a:b] for a, b in phrase["markers"]] == ["up"]


def test_preposition_is_candidate_and_noun_attachment_is_not_a_phrasal_verb():
    doc = parsed(["Anna", "looked", "at", "the", "book", "on", "the", "table"],
                 [1, 1, 1, 4, 2, 4, 7, 5],
                 ["nsubj", "ROOT", "prep", "det", "pobj", "prep", "det", "pobj"],
                 ["PROPN", "VERB", "ADP", "DET", "NOUN", "ADP", "DET", "NOUN"],
                 ["Anna", "look", "at", "the", "book", "on", "the", "table"])
    item, = extract(doc)["predicates"]
    assert item["lexical_unit"] == "look"
    phrase, = item["constructions"]
    assert phrase["lemma"] == "look at" and phrase["kind"] == "preposition_candidate"


def test_particle_with_preposition_and_auxiliary():
    doc = parsed(["Anna", "has", "put", "up", "with", "him"], [2, 2, 2, 2, 2, 4],
                 ["nsubj", "aux", "ROOT", "prt", "prep", "pobj"],
                 ["PROPN", "AUX", "VERB", "ADP", "ADP", "PRON"],
                 ["Anna", "have", "put", "up", "with", "he"])
    extracted = extract(doc)
    assert [v["lemma"] for v in extracted["verbs"]] == ["have", "put"]
    item, = extracted["predicates"]
    assert {(c["lemma"], c["kind"]) for c in item["constructions"]} == {
        ("put up", "particle"), ("put up with", "particle_preposition")}


def test_adverbial_phrasal_candidate_is_not_lost_or_called_a_particle():
    doc = parsed(["I", "look", "forward", "to", "it"], [1, 1, 1, 1, 3],
                 ["nsubj", "ROOT", "advmod", "prep", "pobj"],
                 ["PRON", "VERB", "ADV", "ADP", "PRON"],
                 ["I", "look", "forward", "to", "it"])
    result = extract(doc)
    item, = result["predicates"]
    assert ("look forward to", "adverb_preposition_candidate") in {
        (c["lemma"], c["kind"]) for c in item["constructions"]}
    assert not any(c["kind"] == "particle" for c in item["constructions"])
    before = json.dumps(result)
    assert json.dumps(enrich(result)) == before


def test_no_verb_observations_stay_visible_for_review(tmp_path):
    counts = Counter()
    inventory = Inventory(counts)
    # A parser can miss an imperative; do not interpret missing tags as gold.
    inventory.observe({"line": 12, "idx": 9, "text": "Jump!", "verbs": [],
                       "predicates": [], "sentence_template": "{NP:ROOT} !"})
    inventory.write(tmp_path)
    row = json.loads((tmp_path / "review_queue.jsonl").read_text())
    assert row == {"line": 12, "idx": 9, "text": "Jump!", "reason": "no_verb_detected",
                   "review_status": "UNREVIEWED"}
    assert counts["verbless_records"] == 1 and counts["parsed"] == 1


def test_full_accounting_hash_ledger_and_refusal_to_overwrite(tmp_path):
    doc = sample()
    class Pipeline:
        meta = {"name": "controlled-test-tree"}
        pipe_names = ["parser", "lemmatizer"]

        def pipe(self, inputs, **kwargs):
            for text, metadata in inputs:
                assert text == doc.text
                yield doc, metadata

    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"idx": 8, "example": doc.text}) + "\n" +
                      '{"example": ""}\n' + 'not json\n', encoding="utf-8")
    output = tmp_path / "output"
    result = mine(source, output, Pipeline())
    assert result["status"] == "complete_with_rejections"
    assert result["full_input"]
    assert result["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["counts"]["parsed"] == 1 and result["counts"]["rejected"] == 2
    with gzip.open(output / "occurrences.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = list(map(json.loads, stream))
    assert len(rows) == 1 and rows[0]["line"] == 1 and rows[0]["idx"] == 8
    assert json.loads((output / "multiword_verbs.jsonl").read_text())["separated_count"] == 1
    with pytest.raises(FileExistsError):
        mine(source, output, Pipeline())
    rebuilt = tmp_path / "rebuilt"
    summary = reindex(output, rebuilt)
    assert summary["counts"] == result["counts"]
    for name in ("verbs", "predicate_templates", "multiword_verbs", "sentence_templates", "rejected"):
        assert (output / f"{name}.jsonl").read_bytes() == (rebuilt / f"{name}.jsonl").read_bytes()
    # Truncating a ledger must not yield a successful catalogue.
    with gzip.open(output / "occurrences.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write("")
    with pytest.raises(ValueError, match="record count"):
        reindex(output, tmp_path / "truncated")
    assert json.loads((tmp_path / "truncated" / "manifest.json").read_text())["status"] == "failed"
