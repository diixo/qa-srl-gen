"""Published mining jobs become immutable components of one resource pool."""

import gzip
import hashlib
import json

import pytest

from semantic_corpus.resource_pool import BuiltInPool
from semantic_corpus.semantic_generator.substitutions import EntityPool
from semantic_corpus.template_bank import FORMAT, TemplateBank, fingerprint
from semantic_corpus.template_registry import register_bank, registered_banks
from semantic_corpus.template_validation import validate_bank


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value) + "\n")


def make_bank(directory, *, subject="Ada", verb="run", adjective="bright"):
    """A small complete bank checked by the real, NLP-independent validator."""
    bindings = {"n0": subject, "v0": verb, "a0": adjective}
    text = f"{subject} {verb} {adjective}."
    source = directory.parent / f"{directory.name}-source.jsonl"
    write_json(source, {"idx": 1, "example": text})
    row = {"parts": [{"slot": "n0"}, {"literal": " "}, {"slot": "v0"},
                     {"literal": " "}, {"slot": "a0"}, {"literal": "."}],
           "slots": {"n0": {"kind": "nominal", "pos": "PROPN", "tag": "NNP"},
                     "v0": {"kind": "verb", "pos": "VERB", "tag": "VB", "lemma": verb},
                     "a0": {"kind": "adjective", "pos": "ADJ", "tag": "JJ"}},
           "verbs": [verb], "constructions": [], "predicate_groups": [],
           "attributes": [{"slot": "a0", "target_slot": "n0", "degree": ["Pos"]}]}
    row["template_id"] = fingerprint(row)
    row.update(defaults=bindings, attested_bindings=[bindings], count=1,
               adjective_lemmas=[adjective], fillers={k: [v] for k, v in bindings.items()})
    write_rows(directory / "templates" / f"{row['template_id'][:2]}.jsonl.gz", [row])
    write_rows(directory / "verbs.jsonl.gz", [{"lemma": verb, "forms": {verb: 1}}])
    write_rows(directory / "adjectives.jsonl.gz", [{"lemma": adjective, "forms": {adjective: 1}}])
    for name in ("constructions", "verb_alternatives", "adjective_alternatives"):
        write_rows(directory / f"{name}.jsonl.gz", [])
    tokens, cursor = [], 0
    for kind, pos, word in (("nominal", "PROPN", subject), ("verb", "VERB", verb),
                            ("adjective", "ADJ", adjective)):
        tokens.append({"kind": kind, "pos": pos, "tag": "JJ" if pos == "ADJ" else "",
                       "lemma": word, "parser_lemma": word,
                       "span": [cursor, cursor + len(word)], "text": word})
        cursor += len(word) + 1
    write_rows(directory / "occurrences" / "00000.jsonl.gz",
               [{"line": 1, "idx": 1, "text": text, "template_id": row["template_id"],
                 "bindings": bindings, "tokens": tokens}])
    manifest = {"format": FORMAT, "status": "complete", "field": "example",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "counts": {"input_records": 1, "compiled_records": 1, "templates": 1,
                           "tokens_audited": 3, "verb_lemmas": 1, "adjective_lemmas": 1,
                           "constructions": 0, "verb_alternative_lemmas": 0,
                           "adjective_alternative_forms": 0}}
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "validation.json", validate_bank(TemplateBank(directory), source))
    return directory, source, text


def publish(repo, directory, **kwargs):
    bank, source, text = make_bank(directory, **kwargs)
    target = register_bank(bank, source, repo_root=repo, bank_id=directory.name)
    return target, text


def all_documents(pool, **kwargs):
    return [doc for split in ("train", "dev", "test")
            for doc in pool.mined_documents(split=split, **kwargs)]


def test_publication_preserves_validated_bank_and_source_and_is_idempotent(tmp_path):
    repo = tmp_path / "repo"
    bank, source, _ = make_bank(tmp_path / "job-one")
    target = register_bank(bank, source, repo_root=repo, bank_id="job-one")
    assert target == repo / "data" / "mined-templates" / "job-one"
    assert (target / "source.jsonl").read_bytes() == source.read_bytes()
    assert validate_bank(TemplateBank(target), target / "source.jsonl")["status"] == "passed"
    assert [bank.path for bank in registered_banks(repo)] == [target]
    before = {path.relative_to(target): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    assert register_bank(bank, source, repo_root=repo, bank_id="job-one") == target
    assert before == {path.relative_to(target): path.read_bytes() for path in target.rglob("*") if path.is_file()}


@pytest.mark.parametrize("bank_id", ["../outside", "/outside", "..", ".staging", "Bank", ""])
def test_bank_ids_cannot_escape_registry(tmp_path, bank_id):
    bank, source, _ = make_bank(tmp_path / "job-one")
    with pytest.raises(ValueError, match="bank_id"):
        register_bank(bank, source, repo_root=tmp_path / "repo", bank_id=bank_id)
    assert not (tmp_path / "repo").exists()


@pytest.mark.parametrize("change", ["status", "source_sha256", "source_records"])
def test_invalid_validation_result_cannot_be_published(tmp_path, change, monkeypatch):
    from semantic_corpus import template_registry

    bank, source, _ = make_bank(tmp_path / "job-one")
    report_path = bank / "validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[change] = {"status": "failed", "source_sha256": "0" * 64, "source_records": 2}[change]
    monkeypatch.setattr(template_registry, "validate_bank", lambda *args: report)
    with pytest.raises(ValueError, match="validation report"):
        register_bank(bank, source, repo_root=tmp_path / "repo", bank_id="job-one")
    assert registered_banks(tmp_path / "repo") == ()


def test_mutated_source_is_rejected_before_publication(tmp_path):
    bank, source, _ = make_bank(tmp_path / "job-one")
    source.write_text('{"example": "different"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        register_bank(bank, source, repo_root=tmp_path / "repo", bank_id="job-one")
    assert registered_banks(tmp_path / "repo") == ()


def test_publication_runs_full_validation_and_replaces_any_stale_report(tmp_path):
    bank, source, _ = make_bank(tmp_path / "job-one")
    write_json(bank / "validation.json", {"status": "failed"})
    target = register_bank(bank, source, repo_root=tmp_path / "repo", bank_id="job-one")
    assert json.loads((target / "validation.json").read_text())["status"] == "passed"


def test_incomplete_bank_is_not_published_despite_old_passed_report(tmp_path):
    bank, source, _ = make_bank(tmp_path / "job-one")
    for path in (bank / "templates").glob("*.gz"):
        path.unlink()
    with pytest.raises(ValueError, match="no executable attested binding"):
        register_bank(bank, source, repo_root=tmp_path / "repo", bank_id="job-one")
    assert registered_banks(tmp_path / "repo") == ()


def test_conflicting_publication_never_overwrites_an_existing_bank(tmp_path):
    repo = tmp_path / "repo"
    target, _ = publish(repo, tmp_path / "job-one")
    original = (target / "source.jsonl").read_bytes()
    other, source, _ = make_bank(tmp_path / "job-two", subject="Beth")
    with pytest.raises(FileExistsError, match="different template bank"):
        register_bank(other, source, repo_root=repo, bank_id="job-one")
    assert (target / "source.jsonl").read_bytes() == original


def test_failed_copy_rolls_back_and_staging_is_never_discovered(tmp_path, monkeypatch):
    from semantic_corpus import template_registry

    repo = tmp_path / "repo"
    bank, source, _ = make_bank(tmp_path / "job-one")
    staging = repo / "data" / "mined-templates" / ".interrupted.staging"
    write_json(staging / "manifest.json", {"status": "incomplete"})
    assert registered_banks(repo) == ()

    def failed_copy(*args, **kwargs):
        raise OSError("injected copy failure")

    monkeypatch.setattr(template_registry.shutil, "copyfile", failed_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        register_bank(bank, source, repo_root=repo, bank_id="job-one")
    assert list(staging.parent.iterdir()) == [staging]
    assert registered_banks(repo) == ()


def test_cancellation_after_validation_prevents_publication_and_preserves_existing_banks(tmp_path):
    repo = tmp_path / "repo"
    original, _ = publish(repo, tmp_path / "job-one")
    before = {path.relative_to(original): path.read_bytes()
              for path in original.rglob("*") if path.is_file()}
    bank, source, _ = make_bank(tmp_path / "job-two", subject="Beth")
    callbacks = []

    def cancelled():
        staging, = original.parent.glob(".job-two-*.staging")
        assert json.loads((staging / "validation.json").read_text())["status"] == "passed"
        assert (staging / "publication.json").is_file()
        assert [bank.path for bank in registered_banks(repo)] == [original]
        callbacks.append(True)
        raise RuntimeError("job cancelled")

    with pytest.raises(RuntimeError, match="job cancelled"):
        register_bank(bank, source, repo_root=repo, bank_id="job-two", before_publish=cancelled)
    assert callbacks == [True]
    assert list(original.parent.iterdir()) == [original]
    assert before == {path.relative_to(original): path.read_bytes()
                      for path in original.rglob("*") if path.is_file()}
    assert [bank.path for bank in registered_banks(repo)] == [original]


def test_published_component_without_eng_base_is_a_complete_pool(tmp_path):
    repo = tmp_path / "repo"
    target, text = publish(repo, tmp_path / "job-one")
    pool = BuiltInPool(repo_root=repo, entities=EntityPool())
    assert pool.bank.path == target
    assert pool.summary()["available"] and pool.summary()["source_banks"] == 1
    assert [doc.text for doc in all_documents(pool)] == [text]
    assert pool.metadata()["bank_components"][0]["id"] == "job-one"


def test_duplicate_template_shapes_keep_every_sources_fillers_and_provenance(tmp_path):
    repo = tmp_path / "repo"
    base, _, original_text = make_bank(repo / "data" / "eng-base-templates")
    before = {path.relative_to(base): path.read_bytes() for path in base.rglob("*") if path.is_file()}
    old_pool = BuiltInPool(repo_root=repo, entities=EntityPool())
    old_identity = old_pool.metadata()
    original_document, = all_documents(old_pool)
    _, new_text = publish(repo, tmp_path / "job-one", subject="Beth", adjective="beautiful")
    pool = BuiltInPool(repo_root=repo, entities=EntityPool())
    templates = list(pool.templates())
    assert len(templates) == 2 and templates[0].template_id == templates[1].template_id
    documents = all_documents(pool)
    assert {doc.text for doc in documents} == {original_text, new_text}
    assert next(doc for doc in documents if doc.text == original_text) == original_document
    assert {doc.source for doc in documents} == {"eng-base-templates", "mined-templates/job-one"}
    assert all(doc.metadata["review_status"] == "UNREVIEWED" for doc in documents)
    assert len({doc.document_id for doc in documents}) == 2
    summary = pool.summary()
    assert summary["templates"] == summary["verbs"] == summary["adjectives"] == 2
    assert summary["source_banks"] == 2 and summary["counts_are_component_entries"]
    assert len(list(pool.occurrences())) == 2
    assert old_identity != pool.metadata()
    assert before == {path.relative_to(base): path.read_bytes() for path in base.rglob("*") if path.is_file()}


def test_filter_matches_across_components_without_failing_on_nonmatching_banks(tmp_path):
    repo = tmp_path / "repo"
    make_bank(repo / "data" / "eng-base-templates")
    _, text = publish(repo, tmp_path / "job-one", subject="Beth", verb="jump", adjective="beautiful")
    pool = BuiltInPool(repo_root=repo, entities=EntityPool())
    for filters in ({"lemma": "jump"}, {"adjective": "beautiful"},
                    {"lemma": "jump", "adjective": "beautiful"}):
        assert [doc.text for doc in all_documents(pool, **filters)] == [text]
    with pytest.raises(ValueError, match="no templates match"):
        list(pool.mined_documents(lemma="missing"))


def test_modified_published_manifest_is_visible_as_invalid_instead_of_omitted(tmp_path):
    repo = tmp_path / "repo"
    make_bank(repo / "data" / "eng-base-templates")
    target, _ = publish(repo, tmp_path / "job-one")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    manifest["model"] = {"name": "changed after publication"}
    write_json(target / "manifest.json", manifest)
    pool = BuiltInPool(repo_root=repo, entities=EntityPool())
    assert not pool.summary()["available"] and pool.summary()["status"] == "invalid"
    with pytest.raises(ValueError, match="publication"):
        pool.metadata()


def test_missing_publication_record_is_reported_as_corrupt_component(tmp_path):
    repo = tmp_path / "repo"
    target, _ = publish(repo, tmp_path / "job-one")
    (target / "publication.json").unlink()
    pool = BuiltInPool(repo_root=repo, entities=EntityPool())
    assert pool.summary()["status"] == "invalid"
    assert "job-one" in pool.summary()["error"]


def test_monitoring_does_not_read_source_or_decompress_inventories(tmp_path, monkeypatch):
    from semantic_corpus import resource_pool, template_registry

    repo = tmp_path / "repo"
    publish(repo, tmp_path / "job-one")

    def unexpected_read(*args, **kwargs):
        raise AssertionError("monitoring must only read small metadata files")

    monkeypatch.setattr(resource_pool, "read_rows", unexpected_read)
    monkeypatch.setattr(template_registry, "validate_bank", unexpected_read)
    pool = BuiltInPool(repo_root=repo, entities=EntityPool())
    assert pool.summary()["templates"] == 1
    assert pool.metadata()["bank_components"]
