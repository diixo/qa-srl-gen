"""Normalization must be reversible and mined names must be reviewed and auditable."""

import hashlib
import json

import pytest

from semantic_corpus.dailydialog_mining import (
    APOSTROPHES, Normalizer, archive_originals, load_curation, normalized_inputs,
    prepare_source, restore_original, update_inputs, write_rows,
)
from semantic_corpus.semantic_generator.substitutions import default_pool
from semantic_corpus.template_bank import TemplateBank, fingerprint
from semantic_corpus.template_registry import register_bank
from semantic_corpus.template_validation import validate_bank


@pytest.mark.parametrize('apostrophe', APOSTROPHES)
def test_apostrophes_and_contractions_are_reversible(apostrophe):
    original = f"i{apostrophe}Ve said: {apostrophe}go{apostrophe}. O{apostrophe}Neil didn{apostrophe}T."
    text, edits = Normalizer({})(original)
    assert text == "I've said: 'go'. O'Neil didn't."
    assert restore_original(text, edits) == original


def test_only_reviewed_spellings_change_and_case_does_not_change_common_words():
    text = "i may want a bill and a jack in london. Jim has comerials and a methone."
    normalized, edits = Normalizer(load_curation())(text)
    assert normalized == "I may want a bill and a jack in London. Jim has commercials and a methone."
    assert restore_original(normalized, edits) == text
    assert Normalizer(load_curation())('sandy beaches, terry cloth and keeping bach.')[0] == 'sandy beaches, terry cloth and keeping bach.'
    assert Normalizer(load_curation())("I'II go. I'v seen it. They arn't here.")[0] == "I'll go. I've seen it. They aren't here."


def test_contraction_spacing_does_not_destroy_quotation_or_name():
    text, edits = Normalizer({})("I ' m sure O'Neil said 'hello'. We ' ve agreed.")
    assert text == "I'm sure O'Neil said 'hello'. We've agreed."
    assert restore_original(text, edits) == "I ' m sure O'Neil said 'hello'. We ' ve agreed."
    assert Normalizer({})("He said 's' and then 'd'.")[0] == "He said 's' and then 'd'."


def test_truecase_does_not_change_addresses_and_dictated_lowercase():
    n = Normalizer(load_curation())
    text = "Go to www.google.com, or email jim@example.com."
    assert n(text)[0] == text
    assert n('My email is lower case t, smith.')[0] == 'My email is lower case t, smith.'
    assert n('a china vase, sandy beaches, the mike and beat around the bush.')[0] == 'a china vase, sandy beaches, the mike and beat around the bush.'


def test_sentence_truecase_preserves_acronyms_brands_and_abbreviations():
    spacy = pytest.importorskip('spacy')
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    original = "hello! is the NASA iPhone ready? eBay works. Dr. smith agrees."
    edits = []
    text = Normalizer({}).sentence_case(nlp(original), edits)
    assert text == "Hello! Is the NASA iPhone ready? eBay works. Dr. smith agrees."
    assert restore_original(text, edits) == original


def test_source_preserves_all_splits_duplicates_and_original_turns(tmp_path):
    for split in ('train', 'val', 'test'):
        (tmp_path / f'dailydialog-{split}.jsonl').write_text(
            json.dumps(["i\u2019m here", "i\u2019m here"]) + '\n', encoding='utf-8')
    path = tmp_path / 'source.jsonl'
    report = prepare_source(tmp_path, path, {}, splits=('train', 'val', 'test'))
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert report['counts']['turns'] == 6
    assert [r['idx'] for r in rows] == list(range(6))
    assert {r['origin']['split'] for r in rows} == {'train', 'val', 'test'}
    assert [r['origin']['turn'] for r in rows] == [1, 2, 1, 2, 1, 2]
    assert all(restore_original(r['example'], r['edits']) == r['original'] for r in rows)


def test_original_files_are_preserved_before_in_place_correction(tmp_path):
    from zipfile import ZipFile

    inputs, compiled, bank = tmp_path / 'input', tmp_path / 'compiled', tmp_path / 'bank'
    for directory in (inputs, compiled, bank / 'sources'):
        directory.mkdir(parents=True)
    original = b'["i\\u2019m here", "hello"]\r\n'
    path = inputs / 'dailydialog-train.jsonl'
    path.write_bytes(original)
    source = tmp_path / 'source.jsonl'
    report = prepare_source(inputs, source, {}, splits=('train',))
    archive_originals(inputs, compiled, report)
    with ZipFile(bank / 'sources/daily.zip', 'w') as archive:
        archive.write(compiled / 'originals' / path.name, 'originals/' + path.name)
    expected = normalized_inputs(source)
    update_inputs(inputs, source, report, published_bank=bank, bank_id='daily')
    assert path.read_bytes() == expected[path.name]
    assert json.loads(path.read_text()) == ["I'm here", 'hello']
    assert (compiled / 'originals' / path.name).read_bytes() == original
    path.write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='original source'):
        update_inputs(inputs, source, report, published_bank=bank, bank_id='daily')
    assert path.read_bytes() == original + b'\n'


def enriched_fixture(tmp_path):
    from test_template_registry import make_bank

    bank, source, text = make_bank(tmp_path / 'source-bank', subject='Zorvian')
    row = dict(entity=dict(text='Zorvian', labels=['PERSON', 'NAMED_ENTITY'], named=True, type='human'),
               review_status='REVIEWED', count=1,
               examples=[dict(line=1, idx=1, text=text, span=[0, 7])])
    write_rows(bank / 'entities.jsonl.gz', [row])
    manifest = json.loads((bank / 'manifest.json').read_text())
    manifest['counts']['entities'] = 1
    manifest['resource_inventories'] = {'entities': hashlib.sha256(bytes.fromhex(fingerprint(row))).hexdigest()}
    (bank / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return bank, source, row


def test_reviewed_entities_publish_with_templates_and_load_from_committed_ranges(tmp_path):
    bank, source, row = enriched_fixture(tmp_path)
    repo = tmp_path / 'repo'
    root = register_bank(bank, source, repo_root=repo, bank_id='daily-test')
    assert validate_bank(TemplateBank(root))['entities'] == 1
    assert default_pool(mined_bank=root).readings('Zorvian')
    # An interrupted append must not leak an uncommitted entity into generation.
    row['entity']['text'] = 'UncommittedName'
    write_rows(root / 'tail.gz', [row])
    with (root / 'entities.jsonl.gz').open('ab') as sink:
        sink.write((root / 'tail.gz').read_bytes())
    assert not default_pool(mined_bank=root).readings('UncommittedName')
    assert len(default_pool(mined_bank=root).readings('Zorvian')) == 1
    from semantic_corpus.resource_pool import BuiltInPool
    pool = BuiltInPool(repo_root=repo)
    assert pool.entities.readings('Zorvian')
    assert list(pool.proposals()) == []
    with pytest.raises(ValueError, match='unknown proposal'):
        list(pool.proposals('entities'))


@pytest.mark.parametrize('damage', ['review', 'span', 'type', 'hash', 'count'])
def test_invalid_enrichment_cannot_be_published(tmp_path, damage):
    bank, source, row = enriched_fixture(tmp_path)
    if damage == 'review':
        row['review_status'] = 'UNREVIEWED'
    elif damage == 'span':
        row['examples'][0]['span'] = [1, 7]
    elif damage == 'type':
        row['entity']['type'] = 'city'
    elif damage == 'count':
        row['count'] = 0
    else:
        row['count'] = 2
    write_rows(bank / 'entities.jsonl.gz', [row])
    with pytest.raises(ValueError):
        register_bank(bank, source, repo_root=tmp_path / 'repo', bank_id='invalid')
    assert not (tmp_path / 'repo/data/mined-templates/manifest.json').exists()
