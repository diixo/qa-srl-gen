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


@pytest.mark.parametrize('coordinator', ['and', 'or'])
def test_coordinated_particles_keep_both_constructions_and_the_connector(coordinator):
    doc=parsed(['Anna','turned','it','on',coordinator,'off'],[1,1,1,1,3,3],
        ['nsubj','ROOT','dobj','prt','cc','conj'],['PROPN','VERB','PRON','ADP','CCONJ','ADP'],
        ['anna','turn','it','on',coordinator,'off'])
    item,=extract(doc)['predicates']
    assert item['lexical_unit']=='turn'
    assert [(c['lemma'],c['kind'],c['separated']) for c in item['constructions']]==[
        ('turn on','particle',True),('turn off','particle',True)]
    assert [[doc.text[a:b] for a,b in c['markers']] for c in item['constructions']]==[['on'],['off']]
    assert item['template']==f'{{NSUBJ}} {{VERB}} {{DOBJ}} on {coordinator} off'


def test_coordinated_particle_with_its_own_complement_is_not_guessed():
    doc=parsed(['They','got','in','and','out','of','trouble'],[1,1,1,2,2,4,5],
        ['nsubj','ROOT','prt','cc','conj','prep','pobj'],['PRON','VERB','ADP','CCONJ','ADV','ADP','NOUN'],
        ['they','get','in','and','out','of','trouble'])
    item,=extract(doc)['predicates']
    assert [c['lemma'] for c in item['constructions']]==['get in']


def test_coordinated_particles_preserve_local_negation():
    doc=parsed(['They','turned','it','on','but','not','off'],[1,1,1,1,3,6,3],
        ['nsubj','ROOT','dobj','prt','cc','neg','conj'],
        ['PRON','VERB','PRON','ADP','CCONJ','PART','ADP'],
        ['they','turn','it','on','but','not','off'])
    item,=extract(doc)['predicates']
    assert item['template']=='{NSUBJ} {VERB} {DOBJ} on but not off'
    assert [c['lemma'] for c in item['constructions']]==['turn on','turn off']


@pytest.mark.parametrize('prep,words,heads,deps', [
    ('on',['Tuesday','night'],[5,3],['compound','pobj']),
    ('in',['the','morning'],[5,3],['det','pobj']),
    ('at',['night'],[3],['pobj']),
])
def test_time_phrase_is_kept_separately_from_particle(prep,words,heads,deps):
    doc=parsed(['They','turned','in',prep,*words],[1,1,1,1,*heads],
        ['nsubj','ROOT','prt','prep',*deps],['PRON','VERB','ADP','ADP',*['NOUN']*len(words)],
        ['they','turn','in',prep,*[w.lower() for w in words]])
    item,=extract(doc)['predicates']
    assert [c['lemma'] for c in item['constructions']]==['turn in']
    temporal,=item['temporal_adjunct_candidates']
    assert doc.text[slice(*temporal['span'])].strip()==' '.join([prep,*words])
    assert temporal['status']=='UNREVIEWED'
    assert any(a['dependency']=='prep' and a['head']==prep for a in item['arguments'])


def test_non_temporal_head_and_governed_preposition_remain_candidates():
    doc=parsed(['They','looked','up','at','the','night','sky'],[1,1,1,1,6,6,3],
        ['nsubj','ROOT','prt','prep','det','compound','pobj'],['PRON','VERB','ADP','ADP','DET','NOUN','NOUN'],
        ['they','look','up','at','the','night','sky'])
    item,=extract(doc)['predicates']
    assert {c['lemma'] for c in item['constructions']}=={'look up','look up at'}
    assert 'temporal_adjunct_candidates' not in item
    doc=parsed(['They','put','up','with','winter'],[1,1,1,1,3],
        ['nsubj','ROOT','prt','prep','pobj'],['PRON','VERB','ADP','ADP','NOUN'],
        ['they','put','up','with','winter'])
    assert {c['lemma'] for c in extract(doc)['predicates'][0]['constructions']}=={'put up','put up with'}


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


@pytest.mark.parametrize('original_version', ['1', '2'])
def test_reindex_does_not_claim_old_parses_used_new_extraction_rules(tmp_path, original_version):
    from semantic_corpus.syntax_mining import EXTRACTOR_VERSION
    source=tmp_path/'source';source.mkdir()
    manifest=dict(format='syntax-candidates-v1',status='complete',extractor_version=original_version,
                  counts=dict(parsed=1,lines=1,rejected=0),elapsed_seconds=1)
    (source/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    (source/'rejected.jsonl').write_text('',encoding='utf8')
    record=dict(line=1,idx=0,text=sample().text,**extract(sample()))
    with gzip.open(source/'occurrences.jsonl.gz','wt',encoding='utf8') as f:
        f.write(json.dumps(record)+'\n')
    result=reindex(source,tmp_path/'result')
    assert result['extractor_version']==original_version
    assert result['reindex_version']==EXTRACTOR_VERSION
