"""Lossless compilation and executable selection, without downloaded NLP models."""

import gzip
import hashlib
import json
from zipfile import ZipFile

import pytest

from semantic_corpus.cli import main
from semantic_corpus.template_bank import CorpusTemplate, TemplateBank, generate_file, read_rows
from semantic_corpus.template_compiler import LexicalAudit, build_bank, compile_record


def parsed(words, heads, deps, pos, tags, lemmas, morphs=None):
    spacy = pytest.importorskip("spacy")
    from spacy.tokens import Doc
    return Doc(spacy.blank("en").vocab, words=words, heads=heads, deps=deps,
               pos=pos, tags=tags, lemmas=lemmas, morphs=morphs)


def sentence(adjective="beautiful", subject="Anna", noun="book"):
    return parsed([subject, "picked", "a", adjective, noun, "up", "."],
        [1, 1, 4, 4, 1, 1, 1], ["nsubj", "ROOT", "det", "amod", "dobj", "prt", "punct"],
        ["PROPN", "VERB", "DET", "ADJ", "NOUN", "ADP", "PUNCT"],
        ["NNP", "VBD", "DT", "JJ", "NN", "RP", "."],
        [subject.lower(), "pick", "a", adjective, noun, "up", "."],
        ["", "Tense=Past|VerbForm=Fin", "", "Degree=Pos", "Number=Sing", "", ""])


@pytest.fixture
def resources(tmp_path):
    root = tmp_path / "lexicon"
    root.mkdir()
    (root / "en_verb_inflections.txt").write_text(
        "pick\tpicks\tpicking\tpicked\tpicked\n"
        "jump\tjumps\tjumping\tjumped\tjumped\n", encoding="utf-8")
    (root / "en_postags_withverb.txt").write_text("pick\tnoun verb\n", encoding="utf-8")
    wordnet = tmp_path / "wordnet.zip"
    with ZipFile(wordnet, "w") as archive:
        archive.writestr("wordnet/index.verb", "pick v\njump v\n")
        archive.writestr("wordnet/verb.exc", "picked pick\n")
        archive.writestr("wordnet/index.adj", "beautiful a\ngood a\nkind a\n")
        archive.writestr("wordnet/adj.exc", "better good\nbest good\n")
        archive.writestr("wordnet/LICENSE", "Test license only\n")
    return root, wordnet


@pytest.fixture
def bank(tmp_path, resources):
    docs = [sentence(), sentence("wonderful", "Beth", "stone"), sentence(),
        parsed(["Jump", "!"], [0, 0], ["ROOT", "punct"], ["NOUN", "PUNCT"], ["NN", "."], ["jump", "!"]),
        parsed(["Hello", "!"], [0, 0], ["ROOT", "punct"], ["INTJ", "PUNCT"], ["UH", "."], ["hello", "!"])]
    class Pipeline:
        meta = {"name": "controlled-trees"}
        def pipe(self, inputs, **kwargs):
            by_text = {doc.text: doc for doc in docs}
            for text, source in inputs:
                yield by_text[text], source
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"idx": i, "example": doc.text}) + "\n"
                              for i, doc in enumerate(docs)), encoding="utf-8")
    directory, wordnet = resources
    output = tmp_path / "bank"
    build_bank(source, output, tmp_path / "work", Pipeline(), wiktionary=directory, wordnet=wordnet)
    return TemplateBank(output)


@pytest.mark.parametrize('by_dependency,by_head', [('agent', 3), ('prep', 3), ('prep', 6)])
def test_compiler_preserves_by_attachment_for_dialogue_admission(resources, by_dependency, by_head):
    from semantic_corpus.dialogue_generator.mined_realization import active_phrase, lexicon
    from dataclasses import asdict

    doc = parsed(['The', 'book', 'was', 'sold', 'in', 'the', 'shop', 'by', 'the', 'river', '.'],
                 [1, 3, 3, 3, 3, 6, 4, by_head, 9, 7, 3],
                 ['det', 'nsubjpass', 'auxpass', 'ROOT', 'prep', 'det', 'pobj', by_dependency, 'det', 'pobj', 'punct'],
                 ['DET', 'NOUN', 'AUX', 'VERB', 'ADP', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'PUNCT'],
                 ['DT', 'NN', 'VBD', 'VBN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN', '.'],
                 ['the', 'book', 'be', 'sell', 'in', 'the', 'shop', 'by', 'the', 'river', '.'])
    directory, wordnet = resources
    row, _ = compile_record(doc, {'line': 1}, LexicalAudit(directory, wordnet))
    row['attested_bindings'] = [row['defaults']]
    assert CorpusTemplate(row).render()['text'] == doc.text
    assert row['slots']['n2']['preposition'] == {'text': 'by', 'head_slot': 'v1' if by_head == 3 else 'n1'}
    forms = [asdict(p) for p in lexicon().paradigms('sell')]
    spec, reason = active_phrase(row, 'v1', 'sell', forms, set())
    if by_head == 3:
        assert reason is None and spec['voice'] == 'passive'
        assert {arg['role'] for arg in spec['arguments']} == {'oblique_in', 'by_phrase'}
    else:
        assert spec is None and reason == 'unresolved_prepositional_attachment'


def test_full_coverage_includes_duplicates_and_verbless_records(bank):
    counts = bank.manifest["counts"]
    assert counts["input_records"] == counts["compiled_records"] == counts["roundtrips_verified"] == 5
    assert counts["templates"] == 3 and counts["template_occurrences"] == 5
    ledger = [row for f in sorted((bank.path / "occurrences").glob("*.gz")) for row in read_rows(f)]
    assert [row["line"] for row in ledger] == [1, 2, 3, 4, 5]
    assert ledger[3]["analysis_status"] == "imperative_alternative_added"
    assert ledger[4]["analysis_status"] == "no_verbal_analysis"
    templates = {t.template_id: t for t in bank.templates()}
    for row in ledger:
        assert templates[row["template_id"]].render(row["bindings"])["text"] == row["text"]
        for token in row["tokens"]:
            assert row["text"][slice(*token["span"])] == token["text"]
    source = bank.path.parent / "source.jsonl"
    assert bank.manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_replacements_shift_separated_particles_and_adjective_target(bank):
    template, = bank.templates(lemma="pick up")
    row = template.render({"n0": "Elizabeth", "n1": "telescope", "a0": "wonderful"})
    assert "Elizabeth picked a wonderful telescope up" in row["text"]
    attribute, = row["attributes"]
    assert row["text"][slice(*attribute["span"])] == "wonderful"
    assert row["text"][slice(*attribute["target_span"])] == "telescope"
    group, = row["predicate_groups"]
    assert row["text"][slice(*group["head_span"])] == "picked"
    construction, = group["constructions"]
    assert construction["separated"] and construction["lemma"] == "pick up"
    marker, = construction["markers"]
    assert row["text"][slice(*marker["span"])] == marker["text"] == "up"
    with pytest.raises(ValueError, match="unknown"):
        template.render({"missing": "x"})
    with pytest.raises(ValueError, match="nonempty"):
        template.render({"n0": ""})


@pytest.mark.parametrize("recombine", [False, True])
def test_adjective_selection_filters_bindings_not_just_shared_shape(bank, recombine):
    # beautiful and wonderful share a shape; selecting one must not output the other.
    for seed in range(12):
        row, = bank.generate(adjective="beautiful", seed=seed, recombine=recombine)
        assert "beautiful" in row["text"] and "wonderful" not in row["text"]
    with pytest.raises(ValueError, match="no adjective"):
        list(bank.generate(adjective="nonexistent"))


def test_quota_is_per_template_and_attested_bindings_are_not_lost_to_random_collisions(bank):
    assert len(list(bank.generate())) == bank.manifest["counts"]["templates"]
    for seed in range(10):
        rows = list(bank.generate(lemma="pick", count=2, seed=seed))
        assert len({row["text"] for row in rows}) == 2
    assert list(bank.generate(seed=12)) == list(bank.generate(seed=12))
    with pytest.raises(ValueError, match="attested bindings"):
        list(bank.generate(lemma="pick", count=3))
    with pytest.raises(ValueError, match="positive"):
        list(bank.generate(count=0))
    with pytest.raises(ValueError, match="no templates"):
        list(bank.generate(lemma="unknown"))


def test_cli_renders_full_bank_and_refuses_overwrite(bank, tmp_path):
    out = tmp_path / "generated.jsonl.gz"
    assert main(["template-generate", "--bank", str(bank.path), "--out", str(out)]) == 0
    assert len(list(read_rows(out))) == 3
    with pytest.raises(FileExistsError):
        generate_file(bank, out)
    template, = bank.templates(lemma="jump")
    assert len(list(bank.generate(template_id=template.template_id))) == 1
    with pytest.raises(ValueError, match="SHA-256"):
        list(bank.templates(template_id="../../malicious"))


def test_wordnet_adjectives_and_comparison_are_audited(resources):
    audit = LexicalAudit(*resources)
    assert audit.alternatives("Better")["adjective_lemmas"] == ["good"]
    assert audit.alternatives("kind")["adjective_lemmas"] == ["kind"]
    doc = parsed(["She", "is", "better"], [1, 1, 1], ["nsubj", "ROOT", "acomp"],
        ["PRON", "AUX", "ADJ"], ["PRP", "VBZ", "JJR"], ["she", "be", "good"], ["", "", "Degree=Cmp"])
    template, ledger = compile_record(doc, {"line": 1}, audit)
    adjective, = ledger["attributes"]
    assert adjective["degree"] == ["Cmp"] and adjective["target_slot"] == "n0"
    assert template["adjective_lemmas"] == ["good"]
    assert ledger["tokens"][2]["adjective_lemmas"] == ["good"]


def test_literal_braces_and_unicode_are_preserved(resources):
    audit = LexicalAudit(*resources)
    doc = parsed(["{beautiful}", "…"], [0, 0], ["ROOT", "punct"], ["X", "PUNCT"], ["FW", "."], ["{beautiful}", "…"])
    template, _ = compile_record(doc, {"line": 1}, audit)
    assert CorpusTemplate(template).render()["text"] == doc.text


def test_failed_build_is_not_usable_and_never_silently_skips_bad_rows(tmp_path, resources):
    pytest.importorskip("spacy")
    source = tmp_path / "invalid.jsonl"
    source.write_text('{"example": ""}\n', encoding="utf-8")
    class Pipeline:
        meta = {}
        def pipe(self, inputs, **kwargs):
            yield from inputs
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="nonempty"):
        build_bank(source, output, tmp_path / "work", Pipeline(), wiktionary=resources[0])
    with pytest.raises(ValueError, match="complete"):
        TemplateBank(output)


def test_participle_reading_and_dictionary_homographs_are_not_gold(tmp_path, resources):
    doc = parsed(["Kind", "people", "have", "jumped"], [1, 3, 3, 3], ["amod", "nsubj", "aux", "ROOT"],
        ["NOUN", "NOUN", "AUX", "VERB"], ["NN", "NNS", "VBP", "VBN"], ["kind", "people", "have", "jump"])
    class Pipeline:
        meta = {}
        def pipe(self, inputs, **kwargs):
            for text, source in inputs:
                yield doc, source
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"example": doc.text}) + "\n", encoding="utf-8")
    out = tmp_path / "bank"
    build_bank(source, out, tmp_path / "work", Pipeline(), wiktionary=resources[0], wordnet=resources[1])
    assert list(read_rows(out / "adjectives.jsonl.gz")) == []
    alternatives = {r['lemma']: r for r in read_rows(out / "adjective_alternatives.jsonl.gz")}
    assert alternatives['kind']['analyses'] == {'dictionary_homograph': 1}
    assert alternatives['jumped']['analyses'] == {'participle_reading': 1}


def test_previous_catalogue_coverage_is_enforced(tmp_path, resources):
    from semantic_corpus.syntax_mining import mine
    doc = sentence()
    class Pipeline:
        meta = {}
        pipe_names = ["parser", "lemmatizer"]
        def pipe(self, inputs, **kwargs):
            for text, source in inputs:
                yield doc, source
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"example": doc.text}) + "\n", encoding="utf-8")
    previous = tmp_path / "previous"
    mine(source, previous, Pipeline())
    result = build_bank(source, tmp_path / "bank", tmp_path / "work", Pipeline(),
                        wiktionary=resources[0], previous=previous)
    assert not any(result['previous_catalogue_missing'].values())
    with (previous / 'verbs.jsonl').open('a', encoding='utf-8') as stream:
        stream.write('{"lemma": "invented"}\n')
    with pytest.raises(AssertionError, match="missing"):
        build_bank(source, tmp_path / "bad-bank", tmp_path / "bad-work", Pipeline(),
                   wiktionary=resources[0], previous=previous)


def test_independent_full_validator_detects_lost_adjectives(bank, tmp_path):
    from semantic_corpus.template_validation import validate_bank
    source = bank.path.parent / 'source.jsonl'
    rendered = tmp_path / 'every-template.jsonl.gz'
    result = validate_bank(bank, source, rendered=rendered)
    assert result['status'] == 'passed' and result['source_records'] == 5
    assert result['adjectives_occurrences'] == 3
    assert len(list(read_rows(rendered))) == 3
    with gzip.open(bank.path / 'adjectives.jsonl.gz', 'wt', encoding='utf-8') as stream:
        stream.write('')
    with pytest.raises(ValueError, match='adjectives inventory'):
        validate_bank(bank, source)
