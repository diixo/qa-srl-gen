"""Locate possible analogues of reviewed QC errors without changing analyses.

Matches are review candidates, never automatic corrections or semantic approval.
All occurrences are scanned; existing reviews and manual reuses are protected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.import_dailydialog_sentences import digest
from semantic_corpus.template_bank import read_rows


def diagnose(tokens):
    """Return structural triggers, retaining token positions for review."""
    hits = []
    children = {i: [] for i in range(len(tokens))}
    for i, token in enumerate(tokens):
        if token['head'] != i:
            children[token['head']].append(i)

    def add(family, *positions):
        hits.append(dict(family=family, tokens=list(positions)))

    def word(i):
        return tokens[i]['text'].lower()

    for i, t in enumerate(tokens):
        w, head = word(i), tokens[t['head']]
        lemma = t.get('parser_lemma', '').lower()
        if w == 'dear' and t['pos'] == 'ADJ' and 'oh' in [word(j) for j in range(max(0,i-2),i)]:
            add('oh_dear_interjection', i)
        if lemma == 'seem' and (t['pos'] == 'AUX' or t['dep'] in {'aux','auxpass'}):
            add('seem_lexical_linking_verb', i)
        if w in {'ok','okay'} and t['pos'] == 'PROPN' and t['dep'] in {'acomp','attr'}:
            add('okay_evaluation', i)
        if w in {'chequing','checking'} and t['pos'] == 'VERB' and head['text'].lower() in {'account','accounts'}:
            add('bank_account_nominal', i)
        if t['pos'] == 'PROPN' and t['dep'] == 'npadvmod' and i and word(i-1) == ',':
            add('possible_vocative', i)
        if w == 'ground' and t['pos'] == 'NOUN' and head['text'].lower() == 'beef':
            add('ground_beef_modifier', i)
        if w == 'nice' and t['dep'] == 'amod' and head['pos'] == 'INTJ':
            add('greeting_adjective', i)
        if w == 'on' and t['dep'] == 'prep' and head.get('parser_lemma') == 'try' and not children[i]:
            add('try_on_particle', i)
        if w == 'down' and t['dep'] == 'prep' and any(word(j)=='payment' for j in children[i]):
            add('down_payment_compound', i)
        if w == 'in' and head.get('parser_lemma') == 'move' and t['dep'] == 'advmod':
            add('move_in_particle', i)
        if lemma == 'be':
            adjectives = [j for j in children[i] if word(j) in {'hard','easy','difficult'} and tokens[j]['dep']=='acomp']
            infinitives = [j for j in children[i] if tokens[j]['dep']=='xcomp' and tokens[j]['pos']=='VERB']
            if adjectives and infinitives:
                add('evaluative_infinitive_attachment', i, *adjectives, *infinitives)
        if t['dep']=='xcomp' and head.get('parser_lemma')=='stop' and any(word(j)=='to' for j in children[i]):
            add('stop_infinitive_purpose', i)
        if t['pos'] in {'VERB','AUX'} and t['dep']=='conj' and not any(tokens[j]['dep'] in {'nsubj','nsubjpass','expl'} for j in children[i]):
            parent = t['head']
            for j in range(max(0,i-6),i):
                noun=tokens[j];obj=tokens[noun['head']]
                if (noun['pos'] in {'NOUN','PROPN','PRON'} and noun['dep']=='conj'
                        and obj['dep'] in {'dobj','pobj'} and obj['head']==parent):
                    add('coordinated_subject_as_object', j, i)
        if w=='birdseed' and t['pos']=='ADJ':
            add('birdseed_material_noun', i)
        if t['dep']=='conj' and any(word(j)=='if' for j in children[i]):
            # A broad diagnostic only: coordination with a clause can be valid.
            if head['dep']=='relcl':
                add('conditional_coordination_scope', i)
    return hits


def run(candidate, out):
    candidate, out = candidate.resolve(), out.resolve()
    if out.exists():
        raise ValueError('Choose a new diagnostic directory')
    report=json.loads((candidate/'report.json').read_text(encoding='utf8'))
    if report['validation']['status']!='passed':
        raise ValueError('Source candidate must be validated')
    out.mkdir(parents=True)
    files=sorted((candidate/'bank/occurrences').glob('*.jsonl.gz'))
    frozen={str(p):digest(p) for p in [candidate/'report.json',candidate/'contexts.jsonl.gz',*files]}
    contexts={r['id']:r['turns'] for r in read_rows(candidate/'contexts.jsonl.gz')}
    counts=Counter();families=Counter();protected_families=Counter();ids=set()
    with gzip.open(out/'candidates.jsonl.gz','xt',encoding='utf8',newline='\n') as sink:
        for path in files:
            for row in read_rows(path):
                counts['scanned_occurrences']+=1
                if any('head' not in t or 'dep' not in t for t in row['tokens']):
                    counts['without_dependency_analysis']+=1
                    continue
                hits=diagnose(row['tokens'])
                if not hits:continue
                protected=bool(row.get('context_review') or row.get('reused_from') or
                    row.get('review_status')!='UNREVIEWED' or row.get('method')!='local_spacy_syntax')
                counts['protected_matches' if protected else 'unreviewed_matches']+=1
                (protected_families if protected else families).update({h['family'] for h in hits})
                assert row['idx'] not in ids
                ids.add(row['idx'])
                prefix=contexts[row['context_id']][:row['origin']['turn']]
                result=dict(id=row['idx'],text=row['text'],origin=row['origin'],tokens=row['tokens'],
                    triggers=hits,context_prefix=prefix,protected=protected,
                    status='PROTECTED_EXISTING_REVIEW' if protected else 'NEEDS_REVIEW',
                    semantic_approval=False,publication=False)
                sink.write(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    assert counts['scanned_occurrences']==report['validation']['source_records']
    for path,sha in frozen.items():
        if digest(Path(path))!=sha:raise ValueError('Input changed: '+path)
    summary=dict(version='v36-error-analogue-audit-v1',candidate=str(candidate),counts=dict(counts),
        families=dict(families),protected_families=dict(protected_families),inputs=frozen,
        script_sha256=digest(Path(__file__)),automatic_corrections=0,api_calls=0,publication=False,
        note='Diagnostic candidates only. Scope does not include every possible parse/meaning error. Existing reviews are never overwritten by a trigger.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:summary[k] for k in ('counts','families','protected_families')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(**vars(parser.parse_args()))
