"""One resource pool exposes mined data without promoting syntax to gold labels."""

import gzip
import json

import pytest

from semantic_corpus.documents import Document
from semantic_corpus.ontology import Label, LabelSet
from semantic_corpus.resource_pool import BuiltInPool, mined_split
from semantic_corpus.semantic_generator.frames import DEFAULT_FRAMES
from semantic_corpus.semantic_generator.substitutions import Entity, EntityPool
from semantic_corpus.template_bank import CorpusTemplate, FORMAT, fingerprint


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


@pytest.fixture
def resources(tmp_path):
    directory = tmp_path / "data" / "eng-base-templates"
    # One known held-out form lets the test distinguish source-text hashing
    # from the stronger guarantee already promised by synthetic name splits.
    candidates = EntityPool(Entity(f"Reserved{i}", LabelSet.of(Label.PERSON)) for i in range(100))
    held_out = candidates.split("test").entities[0]
    entities = EntityPool([held_out])
    bindings = [{"n0": held_out.text if i == 0 else f"Person{i:03d}",
                 "v0": "picked", "a0": "beautiful", "n1": "book"} for i in range(120)]
    template = {
        "parts": [{"slot": "n0"}, {"literal": " "}, {"slot": "v0"},
                  {"literal": " a "}, {"slot": "a0"}, {"literal": " "},
                  {"slot": "n1"}, {"literal": " up."}],
        "slots": {
            "n0": {"kind": "nominal", "pos": "PROPN", "tag": "NNP"},
            "v0": {"kind": "verb", "pos": "VERB", "tag": "VBD", "lemma": "pick"},
            "a0": {"kind": "adjective", "pos": "ADJ", "tag": "JJ"},
            "n1": {"kind": "nominal", "pos": "NOUN", "tag": "NN"}},
        "verbs": ["pick"], "constructions": ["pick up"],
        "predicate_groups": [{"head_slot": "v0", "constructions": [{
            "kind": "particle", "lemma": "pick up", "separated": True,
            "markers": [{"part": 7, "start": 1, "end": 3}]}]}],
        "attributes": [{"slot": "a0", "target_slot": "n1", "degree": ["Pos"]}],
    }
    template["template_id"] = fingerprint(template)
    template.update(defaults=bindings[0], attested_bindings=bindings,
                    adjective_lemmas=["beautiful"], count=len(bindings),
                    fillers={slot: sorted({b[slot] for b in bindings}) for slot in bindings[0]})
    literal = {"parts": [{"literal": "Hello!"}], "slots": {}, "verbs": [],
               "constructions": [], "predicate_groups": [], "attributes": []}
    literal["template_id"] = fingerprint(literal)
    literal.update(defaults={}, attested_bindings=[{}], fillers={}, count=1,
                   adjective_lemmas=[])
    shards = {}
    for row in [template, literal]:
        shards.setdefault(row["template_id"][:2], []).append(row)
    for name, rows in shards.items():
        write_rows(directory / "templates" / f"{name}.jsonl.gz", rows)
    inventories = {
        "verbs": [{"lemma": "pick", "forms": {"picked": 120}}],
        "adjectives": [{"lemma": "beautiful", "forms": {"beautiful": 120}}],
        "constructions": [{"lemma": "pick up", "kind": "particle", "count": 120}],
        "verb_alternatives": [{"lemma": "book", "analyses": {"dictionary_homograph": 120}}],
        "adjective_alternatives": [{"lemma": "picked", "analyses": {"participle_reading": 120}}],
    }
    for name, rows in inventories.items():
        write_rows(directory / f"{name}.jsonl.gz", rows)
    occurrences = [{"line": i + 1, "template_id": template["template_id"],
                    "bindings": b, "text": CorpusTemplate(template).render(b)["text"]}
                   for i, b in enumerate(bindings)]
    occurrences.append({"line": 121, "template_id": literal["template_id"],
                        "bindings": {}, "text": "Hello!"})
    write_rows(directory / "occurrences" / "00000.jsonl.gz", occurrences[:60])
    write_rows(directory / "occurrences" / "00001.jsonl.gz", occurrences[60:])
    (directory / "occurrences" / "README.md").write_text("Not a data shard.", encoding="utf-8")
    manifest = {"format": FORMAT, "status": "complete", "source_sha256": "a" * 64,
                "counts": {"input_records": 121, "compiled_records": 121,
                           "templates": 2, "verb_lemmas": 1, "adjective_lemmas": 1,
                           "constructions": 1, "verb_alternative_lemmas": 1,
                           "adjective_alternative_forms": 1}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return BuiltInPool(repo_root=tmp_path, entities=entities), inventories, occurrences, held_out.text


def test_all_resource_types_are_visible_in_one_pool(resources):
    pool, inventories, occurrences, _ = resources
    assert pool.frames == DEFAULT_FRAMES
    assert len(pool.entities) == 1
    for name, expected in inventories.items():
        assert list(getattr(pool, name)()) == expected
    assert list(pool.occurrences()) == occurrences
    assert len(list(pool.templates())) == 2
    assert len(list(pool.templates(lemma="pick up", adjective="beautiful"))) == 1
    summary = pool.summary()
    assert summary["available"]
    assert summary["entities"] == 1 and summary["frames"] == len(DEFAULT_FRAMES)
    assert summary["templates"] == 2 and summary["records"] == 121
    for name in inventories:
        assert summary[name] == 1


def test_missing_bank_is_visible_and_does_not_silently_fall_back(tmp_path):
    pool = BuiltInPool(repo_root=tmp_path, entities=EntityPool())
    summary = pool.summary()
    assert not summary["available"] and summary["error"]
    assert summary["entities"] == 0
    assert list(pool.mined_documents(count=0)) == []
    # Curated generation can explicitly opt out even on a fresh checkout.
    assert pool.metadata(include_mined=False)
    with pytest.raises((FileNotFoundError, ValueError)):
        list(pool.mined_documents(count=1))


def test_metadata_identity_tracks_bank_and_entity_changes(resources, tmp_path):
    pool, _, _, _ = resources
    identity = pool.metadata()
    assert identity == pool.metadata()
    json.dumps(identity)  # The identity is persisted in build configuration.
    manifest_path = pool.bank.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    changed = BuiltInPool(repo_root=tmp_path, entities=pool.entities)
    assert identity != changed.metadata()
    different_entities = BuiltInPool(repo_root=tmp_path, entities=EntityPool())
    assert changed.metadata() != different_entities.metadata()


@pytest.mark.parametrize("manifest", [[], {"format": FORMAT, "status": "complete", "counts": []}])
def test_invalid_manifest_is_visible_in_monitor_instead_of_crashing(tmp_path, manifest):
    directory = tmp_path / "data" / "eng-base-templates"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pool = BuiltInPool(repo_root=tmp_path, entities=EntityPool())
    assert pool.summary()["status"] == "invalid"
    assert not pool.summary()["available"]
    with pytest.raises(ValueError):
        pool.metadata()


def test_generation_and_metadata_do_not_load_optional_audit_inventories(resources):
    pool, _, _, _ = resources
    # These resources remain available to inspection, but are not dependencies
    # of ordinary rendering or a small dashboard/identity request.
    for name in ("verbs", "verb_alternatives", "adjective_alternatives", "constructions"):
        (pool.bank.path / f"{name}.jsonl.gz").unlink()
    assert pool.metadata()["bank_source_sha256"] == "a" * 64
    assert pool.summary()["available"]
    assert list(pool.mined_documents(count=120, lemma="pick", split="train"))
    with pytest.raises(FileNotFoundError):
        list(pool.verb_alternatives())


def test_mined_outputs_preserve_phrasal_and_adjective_spans_without_gold_runs(resources):
    pool, _, _, _ = resources
    document = next(pool.mined_documents(count=120, lemma="pick up", split="test"))
    assert isinstance(document, Document)
    assert document.metadata["review_status"] == "UNREVIEWED"
    group, = document.metadata["predicate_groups"]
    assert document.text[slice(*group["head_span"])] == "picked"
    construction, = group["constructions"]
    assert construction["separated"] and construction["lemma"] == "pick up"
    marker, = construction["markers"]
    assert document.text[slice(*marker["span"])] == marker["text"] == "up"
    attribute, = document.metadata["attributes"]
    assert document.text[slice(*attribute["span"])] == "beautiful"
    assert document.text[slice(*attribute["target_span"])] == "book"


def test_splits_are_disjoint_and_held_out_names_never_enter_training(resources):
    pool, _, _, held_out = resources
    partitions = {}
    for split in ("train", "dev", "test"):
        stats = {}
        docs = list(pool.mined_documents(count=120, lemma="pick", split=split, stats=stats))
        partitions[split] = {doc.text: doc for doc in docs}
        assert docs and all(doc.split == split for doc in docs)
        assert stats["rendered"] == 120
        assert stats["eligible"] == len(docs)
        assert stats["split_excluded"] + stats["eligible"] == stats["rendered"]
    assert sum(map(len, partitions.values())) == 120
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not partitions[left].keys() & partitions[right].keys()
    assert all(held_out not in text for text in partitions["train"])
    assert any(text.startswith(held_out + " ") for text in partitions["test"])


def test_identity_and_assignment_do_not_change_with_seed_filters_or_order(resources):
    pool, _, _, _ = resources
    for split in ("train", "dev", "test"):
        baseline = {d.text: d.document_id for d in pool.mined_documents(
            count=120, seed=1, lemma="pick", split=split)}
        filtered = {d.text: d.document_id for d in pool.mined_documents(
            count=120, seed=987, lemma="pick up", adjective="beautiful", split=split)}
        assert baseline == filtered


def test_quota_is_enforced_before_split_filter_and_zero_disables_bank(resources):
    pool, _, _, _ = resources
    stats = {}
    list(pool.mined_documents(count=1, split="train", stats=stats))
    assert stats["rendered"] == 2  # Includes the nonverbal template.
    assert stats["eligible"] + stats["split_excluded"] == 2
    with pytest.raises(ValueError, match="attested bindings"):
        list(pool.mined_documents(count=121, lemma="pick", split="train"))
    assert list(pool.mined_documents(count=0)) == []
    with pytest.raises(ValueError):
        list(pool.mined_documents(count=-1))
    with pytest.raises(ValueError, match="split"):
        list(pool.mined_documents(split="invalid"))


def test_held_out_matching_observes_case_and_word_boundaries():
    # First find naturally training texts, so a failure to match is visible.
    cases = [("Rex", "Rex"), ("North Paris", "North Paris")]
    for text, name in cases:
        candidate = next(f"{text} visits shop {i}." for i in range(100)
                         if mined_split(f"{text} visits shop {i}.") == "train")
        assert mined_split(candidate, held_out_names=[name]) == "test"
    for text in ("rex visits shop", "Rexford visits shop", "a Northwest Parisian visits shop"):
        candidate = next(f"{text} {i}." for i in range(100)
                         if mined_split(f"{text} {i}.") == "train")
        assert mined_split(candidate, held_out_names=["Rex", "North Paris"]) == "train"


def test_cli_build_includes_mined_pool_by_default_without_fake_gold_exports(resources, tmp_path, monkeypatch):
    from semantic_corpus import resource_pool
    from semantic_corpus.cli import main
    from semantic_corpus.semantic_generator import Generator
    from semantic_corpus.storage import CorpusStore
    from semantic_corpus.storage.repository import StoreError

    pool, _, _, _ = resources
    monkeypatch.setattr(resource_pool, "BuiltInPool", lambda: pool)
    # Existing quota/generator tests cover curated generation. Isolate the
    # mined part here so an empty BIO export cannot hide fake all-O rows.
    monkeypatch.setattr(Generator, "generate_per_frame", lambda self, count: iter(()))
    seed = next(seed for seed in range(100) if list(pool.mined_documents(seed=seed)))
    expected = list(pool.mined_documents(seed=seed))
    root, out = tmp_path / "store", tmp_path / "export"
    args = ["build", "--store", str(root), "--out", str(out), "--count", "1", "--seed", str(seed)]
    assert main(args) == 0  # No extra template operation or opt-in flag.
    store = CorpusStore(root, create=False)
    docs = list(store.documents())
    assert {d.document_id for d in docs} == {d.document_id for d in expected}
    assert docs and all(d.metadata["review_status"] == "UNREVIEWED" for d in docs)
    assert store.count("annotation_runs") == store.count("qa_examples") == 0
    assert not (out / "bio.jsonl").read_text(encoding="utf-8").strip()
    assert not (out / "sft.jsonl").read_text(encoding="utf-8").strip()
    version, = store.versions()
    assert version["build_config"]["template_count"] == 1
    assert version["build_config"]["resource_pool"]["bank_source_sha256"] == "a" * 64
    summary = json.loads((out / "generation-summary.json").read_text(encoding="utf-8"))
    assert summary["mined"]["rendered"] == 2
    assert summary["mined"]["stored"] == summary["pending_annotation"] == len(expected)
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    assert main(args) == 0
    assert before == {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    with pytest.raises(StoreError, match="version"):
        main(args + ["--seed", str(seed + 100)])
    assert before == {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}


def test_cli_mined_late_quota_failure_precedes_any_curated_write(resources, tmp_path, monkeypatch):
    from semantic_corpus import resource_pool
    from semantic_corpus.cli import main
    from semantic_corpus.semantic_generator import Generator
    from semantic_corpus.storage import CorpusStore

    pool, _, _, _ = resources
    # A valid curated sentence is available, but may only be written after the
    # entire mined selection has passed its own quota validation.
    realized = next(Generator(seed=42).generate_per_frame(1))
    monkeypatch.setattr(resource_pool, "BuiltInPool", lambda: pool)
    monkeypatch.setattr(Generator, "generate_per_frame", lambda self, count: iter([realized]))
    original_templates = pool.bank.templates
    monkeypatch.setattr(pool.bank, "templates", lambda **filters: iter(sorted(
        original_templates(**filters), key=lambda t: len(t.record["attested_bindings"]), reverse=True)))
    original_generate = pool.bank.generate
    rendered = []
    def observed_generate(**kwargs):
        for row in original_generate(**kwargs):
            rendered.append(row)
            yield row
    monkeypatch.setattr(pool.bank, "generate", observed_generate)
    root = tmp_path / "store"
    with pytest.raises(ValueError, match="attested bindings"):
        main(["build", "--store", str(root), "--out", str(tmp_path / "export"),
              "--count", "1", "--template-count", "2"])
    assert len(rendered) == 2  # The second template, not the first, rejected the quota.
    store = CorpusStore(root, create=False)
    assert store.count("documents") == store.count("annotation_runs") == store.count("qa_examples") == 0
    assert not store.versions()


@pytest.mark.parametrize("already_annotated", [False, True])
def test_cli_raw_first_collision_is_checked_before_writing_curated_results(
        resources, tmp_path, monkeypatch, already_annotated):
    from itertools import islice

    from semantic_corpus import resource_pool
    from semantic_corpus.cli import main
    from semantic_corpus.documents import AnnotationRun
    from semantic_corpus.semantic_generator import Generator
    from semantic_corpus.storage import CorpusStore
    from semantic_corpus.storage.repository import StoreError

    pool, _, _, _ = resources
    realized = list(islice(Generator(seed=42).generate_per_frame(1), 2))
    monkeypatch.setattr(resource_pool, "BuiltInPool", lambda: pool)
    monkeypatch.setattr(Generator, "generate_per_frame", lambda self, count: iter(realized))
    root, out = tmp_path / "store", tmp_path / "out"
    store = CorpusStore(root)
    raw = Document("raw-first", realized[1].text, source="eng-base-templates", split="train",
                   metadata={"resource_pool": "built-in", "review_status": "UNREVIEWED"})
    store.add_document(raw)
    if already_annotated:
        # An empty but existing annotation run is distinct from no annotation
        # run. AnnotationRun has __len__, so a truthiness check gets this wrong.
        store.add_run(AnnotationRun("existing-empty-run"), raw)
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    args = ["build", "--store", str(root), "--out", str(out), "--count", "1",
            "--template-count", "0"]
    if not already_annotated:
        with pytest.raises(StoreError, match="unannotated document.*fresh store"):
            main(args)
        assert before == {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
        assert store.count("documents") == 1 and store.count("annotation_runs") == 0
        assert not store.versions()
    else:
        assert main(args) == 0
        assert store.count("documents") == store.count("annotation_runs") == 2
        assert store.document("raw-first") == raw
        summary = json.loads((out / "generation-summary.json").read_text(encoding="utf-8"))
        assert summary["curated_stored"] == 1 and summary["curated_duplicates"] == 1
        assert summary["pending_annotation"] == 0
