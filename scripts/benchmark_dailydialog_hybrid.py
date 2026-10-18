"""Offline comparison of local NLP with the published DailyDialog reviews.

This is an experiment, not a publisher. Outputs deliberately have no native
bank manifest, and every automatic annotation remains UNREVIEWED.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantic_corpus.template_bank import CorpusTemplate, TemplateBank, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from scripts.import_dailydialog_sentences import check_turn, digest, turn_id

VERSION = 'dailydialog-hybrid-benchmark-v1'
CLAUSES = {'ccomp', 'xcomp', 'advcl', 'relcl', 'acl', 'csubj'}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write(stream, obj):
    stream.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':'))+'\n')


def load_reference(path):
    """Read the committed, self-contained bank; review-parts is not required."""
    bank = TemplateBank(path)
    rows, identities = [], set()
    for component in bank.components():
        info = component.manifest.get('manual_import', {})
        if info.get('method') != 'model_reviewed':
            continue
        templates = {t.template_id: t.record for t in component.templates()}
        occurrences = {o['idx']: o for o in component.occurrences()}
        for row in component.rows('sentence_review.jsonl.gz'):
            identity = turn_id(row)
            if identity in identities or identity not in info['reviewed_turns']:
                raise ValueError('duplicated or undeclared published review: '+identity)
            identities.add(identity)
            for segment in row['segments']:
                if segment['status'] != 'sentence':
                    continue
                idx = identity+':'+str(segment['origin']['segment'])
                observation = occurrences.pop(idx)
                record = deepcopy(templates[observation['template_id']])
                record['defaults'] = observation['bindings']
                segment['record'] = record
                if observation['text'] != segment['text'] or observation['origin'] != segment['origin']:
                    raise ValueError('published review/occurrence mismatch')
            check_turn(row, row['original'])
            rows.append(row)
        if occurrences:
            raise ValueError('published sentences have no review entry')
    if not rows:
        raise ValueError('no published model-reviewed DailyDialog turns')
    return sorted(rows, key=lambda r: (r['split'], r['dialogue'], r['turn']))


def corpus_index(directory, references, out):
    """Account for every input dialogue; never reuse a context-dependent turn."""
    indexed = {turn_id(r): r for r in references}
    seen, counts, cohorts, contexts = {}, Counter(), {}, {}
    remaining_seen, reviewed_signatures = {}, set()
    with (out/'coverage.jsonl').open('x', encoding='utf-8') as stream:
        for split in ('train', 'val', 'test'):
            for number, turns in enumerate(read_rows(directory/f'dailydialog-{split}.jsonl'), 1):
                if not isinstance(turns, list) or not turns or any(not isinstance(t, str) or not t.strip() for t in turns):
                    raise ValueError('invalid source dialogue')
                identity = f'{split}:{number}'
                signature = fingerprint(turns)
                # Whole duplicate dialogues always belong to the same cohort.
                cohort = 'evaluation' if int(signature[:8], 16) % 5 == 0 else 'calibration'
                reviewed = all(f'{identity}:{t}' in indexed for t in range(1, len(turns)+1))
                found = [r for t in range(1, len(turns)+1) if (r := indexed.get(f'{identity}:{t}')) is not None]
                if found and not reviewed:
                    raise ValueError('benchmark requires complete reviewed dialogues')
                for row in found:
                    if row['original'] != turns[row['turn']-1]:
                        raise ValueError('published review differs from current source: '+turn_id(row))
                    cohorts[turn_id(row)] = cohort
                    contexts[turn_id(row)] = turns
                counts['dialogues'] += 1
                counts['turns'] += len(turns)
                counts['reviewed_dialogues' if reviewed else 'remaining_dialogues'] += 1
                if reviewed:
                    counts[cohort+'_dialogues'] += 1
                    reviewed_signatures.add(signature)
                if signature in seen:
                    counts['exact_whole_dialogue_duplicates'] += 1
                if not reviewed:
                    remaining_seen.setdefault(signature, identity)
                write(stream, dict(dialogue=identity, turns=len(turns), source_sha256=signature,
                                   reviewed=reviewed, cohort=cohort if reviewed else None,
                                   exact_dialogue_duplicate_of=seen.get(signature)))
                seen.setdefault(signature, identity)
    if set(cohorts) != set(indexed):
        raise ValueError('published reference contains unknown source turns')
    counts['unique_remaining_dialogue_texts'] = len(remaining_seen)
    counts['remaining_novel_dialogue_texts'] = len(set(remaining_seen)-reviewed_signatures)
    counts['remaining_whole_dialogue_reuse_candidates'] = (
        counts['remaining_dialogues']-counts['remaining_novel_dialogue_texts'])
    return dict(counts), cohorts, contexts


def sentence_spans(doc):
    """Trim boundary whitespace only; retain every other source character."""
    result, cursor = [], 0
    for sentence in doc.sents:
        start, end = sentence.start_char, sentence.end_char
        while start < end and doc.text[start].isspace():
            start += 1
        while end > start and doc.text[end-1].isspace():
            end -= 1
        if start == end:
            continue
        if start < cursor or doc.text[cursor:start].strip():
            raise ValueError('sentence segmentation lost source text')
        result.append([start, end])
        cursor = end
    if not result or doc.text[cursor:].strip():
        raise ValueError('sentence segmentation left an unaccounted remainder')
    return result


def trimmed_span(text, span):
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end-1].isspace():
        end -= 1
    return [start, end]


def features(record):
    """Observable signatures, not semantic gold or a linguistic error rate."""
    rendered = CorpusTemplate(record).render()
    spans, slots = rendered['spans'], record['slots']
    result = dict(verbs=[], adjectives=[], verb_scope=[], nominal_roles=[], attributes=[], constructions=[])
    for name, slot in slots.items():
        span = spans[name]
        head = slot.get('preposition', {}).get('head_slot', slot.get('head_slot'))
        parent = spans.get(head)
        dep = {'obj': 'dobj'}.get(slot['dependency'], slot['dependency'])
        if slot['kind'] == 'verb':
            result['verbs'].append([span, slot.get('lemma'), slot['tag']])
            result['verb_scope'].append([span, dep, parent])
        elif slot['kind'] == 'adjective':
            # Earlier manual annotations do not consistently store ADJ lemmas.
            result['adjectives'].append([span, slot['tag']])
        elif slot['kind'] == 'nominal':
            result['nominal_roles'].append([span, dep, parent, slot.get('preposition', {}).get('text')])
    for attribute in rendered['attributes']:
        result['attributes'].append([attribute['span'], attribute['target_span']])
    for group in rendered['predicate_groups']:
        for construction in group['constructions']:
            result['constructions'].append([group['head_span'], construction['lemma'], construction['kind'],
                construction['separated'], [m['span'] for m in construction['markers']]])
    return {k: sorted(v, key=lambda x: json.dumps(x)) for k, v in result.items()}


def comparable_span(left, right):
    # Phrase-sized manual nominals can contain a single automatic noun head.
    return left is not None and right is not None and (
        left[0] <= right[0] and right[1] <= left[1] or right[0] <= left[0] and left[1] <= right[1])


def aligned_nominals(left, right):
    """Relax nominal span width only when alignment is unambiguous."""
    if len(left) != len(right):
        return False
    used = set()
    for span, dep, parent, prep in left:
        matches = [i for i, r in enumerate(right) if comparable_span(span, r[0])]
        if len(matches) != 1 or matches[0] in used:
            return False
        index = matches[0]
        if [dep, parent, prep] != right[index][1:]:
            return False
        used.add(index)
    return True


def compare(reference, automatic):
    a, b = features(reference), features(automatic)
    equal = {k: a[k] == b[k] for k in a}
    equal['nominal_roles_containment'] = aligned_nominals(a['nominal_roles'], b['nominal_roles'])
    return equal


def risk_flags(doc, record, ledger):
    """Frozen triage heuristics; absence of flags is NOT approval."""
    flags = set()
    slots = list(record['slots'].values())
    verbs = [s for s in slots if s['kind'] == 'verb']
    if not verbs:
        flags.add('no_verb_or_fragment')
    if ledger['analysis_status'] == 'imperative_alternative_added':
        flags.add('dictionary_recovery')
    if ledger.get('imperative_candidates'):
        flags.add('dictionary_imperative_candidate')
    if any(s['dependency'] in CLAUSES for s in slots):
        flags.add('embedded_clause')
    if any(s['dependency'] == 'conj' for s in verbs):
        flags.add('coordinated_verbs')
    if any(s['tag'] == 'MD' or s.get('lemma') in {'want', 'need', 'hope', 'wish', 'intend', 'plan', 'expect'} for s in verbs):
        flags.add('modality_or_intention')
    if any(s['dependency'] == 'ROOT' and s['tag'] in {'VBG', 'VBN'} for s in verbs):
        flags.add('nonfinite_root')
    if re.search(r"\b(?:not|never|no|nobody|nothing|neither)\b|n['’]t\b", doc.text, re.I):
        flags.add('negation')
    if re.search(r'\b(?:if|unless|whether)\b', doc.text, re.I):
        flags.add('conditional_or_embedded_question')
    if '"' in doc.text or any(c in doc.text for c in '“”'):
        flags.add('quotation')
    if any(s.get('lemma') in {'say', 'tell', 'report', 'claim', 'reply', 'ask'} for s in verbs):
        flags.add('reported_speech')
    if record['constructions']:
        flags.add('particle_or_prepositional_unit')
    if '?' in doc.text:
        flags.add('question')
    if any(s['dependency'] == 'ROOT' and s['tag'] == 'VB' for s in verbs) and not any(s['dependency'] == 'nsubj' for s in slots):
        flags.add('imperative_or_missing_subject')
    if any(s.get('pos') == 'PROPN' for s in slots):
        flags.add('proper_name')
    if any(s['dependency'] == 'ROOT' and s.get('lemma') in {'do', 'be', 'have'} for s in verbs):
        flags.add('controlled_or_elliptical_predicate')
    return sorted(flags)


def select_review(cases, seed='hybrid-v1'):
    """Deterministic stratified sample; deliberate, not a population estimator."""
    ordered = sorted(cases, key=lambda c: fingerprint([seed, c['id']]))
    selected = {}
    def take(predicate, limit, reason):
        candidates = [c for c in ordered if c['id'] not in selected and predicate(c)]
        for case in candidates[:limit]:
            selected[case['id']] = dict(case, selection_reason=reason)
    take(lambda c: c['kind'] == 'boundary' and not c['equal'], 10, 'boundary_difference')
    take(lambda c: c['kind'] == 'syntax' and not c['flags'] and all(c['equal'].values()), 10, 'unflagged_agreement')
    take(lambda c: c['kind'] == 'syntax' and not c['flags'] and not all(c['equal'].values()), 10, 'unflagged_difference')
    take(lambda c: c['kind'] == 'syntax' and bool(c['flags']) and all(c['equal'].values()), 10, 'flagged_agreement')
    take(lambda c: c['kind'] == 'syntax' and bool(c['flags']) and not all(c['equal'].values()), 20, 'flagged_difference')
    return list(selected.values())


def run(repo, out, model):
    started = time.monotonic()
    bank_path = repo/'data/mined-templates'
    # Refuse any output nested in production resources, including symlink aliases.
    if out.resolve().is_relative_to((repo/'data').resolve()):
        raise ValueError('benchmark output must be outside data/')
    if out.exists():
        raise ValueError('use a new output directory')
    bank = TemplateBank(bank_path)
    protected = {bank_path/name: digest(bank_path/name) for c in bank.components()
                 if c.manifest.get('manual_import') for name in c.files}
    protected[bank_path/'manifest.json'] = digest(bank_path/'manifest.json')
    references = load_reference(bank_path)
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'data/mined-templates'], cwd=repo).returncode != 0:
        raise ValueError('main bank has uncommitted changes; benchmark requires a committed baseline')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    out.mkdir(parents=True)
    corpus, cohorts, contexts = corpus_index(repo/'data/dailydialog/filtered', references, out)
    import spacy
    nlp = spacy.load(model, disable=['ner'])
    if nlp.lang != 'en' or not {'parser', 'lemmatizer'} <= set(nlp.pipe_names):
        raise ValueError('English dependency parser and lemmatizer required')
    plan = dict(format=VERSION, status='running', git_revision=revision, model=nlp.meta,
        spacy=spacy.__version__, model_location=str(model), script_sha256=digest(Path(__file__)),
        compiler_sha256=digest(repo/'semantic_corpus/template_compiler.py'),
        wiktionary={name:digest(repo/'data/wiktionary'/name) for name in ('en_verb_inflections.txt','en_postags_withverb.txt')},
        source_hashes={s:digest(repo/f'data/dailydialog/filtered/dailydialog-{s}.jsonl') for s in ('train','val','test')},
        reference_turns=len(references), cohort_policy='whole-dialogue text hash modulo 5; 0 evaluation',
        admission='none; all outputs UNREVIEWED; no automatic publication', corpus=corpus)
    dump(out/'plan.json', plan)
    for split in ('train','val','test'):
        path = repo/f'data/dailydialog/filtered/dailydialog-{split}.jsonl'
        protected[path] = plan['source_hashes'][split]
    audit = LexicalAudit(repo/'data/wiktionary')
    statistics = defaultdict(Counter)
    boundary_rows, syntax_rows = [], []
    jobs = []
    print(f"Reference: {corpus['reviewed_dialogues']} dialogues, {len(references)} turns", flush=True)
    # Parsing raw turns determines boundaries without consulting reference spans.
    for doc, row in nlp.pipe(((r['original'], r) for r in references), as_tuples=True, batch_size=64):
        identity = turn_id(row)
        spans = sentence_spans(doc)
        expected = [trimmed_span(row['original'], s['origin']['span']) for s in row['segments']]
        cohort = cohorts[identity]
        statistics[cohort]['turns'] += 1
        statistics[cohort]['boundary_exact_turns'] += spans == expected
        actual_set, expected_set = set(map(tuple, spans)), set(map(tuple, expected))
        statistics[cohort]['automatic_segments'] += len(spans)
        statistics[cohort]['reference_segments'] += len(expected)
        statistics[cohort]['exact_segment_spans'] += len(actual_set & expected_set)
        boundary_rows.append(dict(id=identity, kind='boundary', cohort=cohort, original=row['original'],
            equal=spans == expected, reference_spans=expected, automatic_spans=spans,
            reference=[dict(status=s['status'], text=s['text']) for s in row['segments']],
            automatic=[row['original'][a:b] for a,b in spans]))
        for i, (a,b) in enumerate(spans, 1):
            ref = next((s for s in row['segments'] if trimmed_span(row['original'],s['origin']['span'])==[a,b]), None)
            jobs.append((row['original'][a:b], dict(id=f'{identity}:{i}', turn_id=identity, cohort=cohort,
                source_span=[a,b], reference=ref)))
    with (out/'boundary-comparison.jsonl').open('x',encoding='utf-8') as stream:
        for row in boundary_rows:
            write(stream,row)
    print(f"Raw sentence candidates: {len(jobs)}", flush=True)
    # Reparse each full sentence in isolation, preserving its dialogue address.
    with (out/'automatic.jsonl').open('x',encoding='utf-8') as sink:
        for doc, job in nlp.pipe(jobs, as_tuples=True, batch_size=64):
            record, ledger = compile_record(doc, {'idx':job['id']}, audit)
            flags = risk_flags(doc, record, ledger)
            ref = job.pop('reference')
            equal = compare(ref['record'],record) if ref and ref['status']=='sentence' and ref['text']==doc.text else None
            status = ref['status'] if ref else 'boundary_unmatched'
            counts = statistics[job['cohort']]
            counts['raw_roundtrips'] += 1
            counts['unflagged_raw_candidates'] += not flags
            counts['raw_reference_'+status] += 1
            counts['raw_aligned_uncorrected_sentences'] += equal is not None
            counts['raw_all_features_equal'] += equal is not None and all(equal.values())
            counts['raw_non_sentence_with_verbs'] += status in ('fragment','pending') and bool(record['verbs'])
            for flag in flags:
                counts['flag_'+flag] += 1
            write(sink,dict(**job,review_status='UNREVIEWED',flags=flags,reference_status=status,
                comparable_features=equal,record=record))
    # Diagnostic only: reviewed sentence boundaries and corrections are supplied
    # here to measure syntax separately. These scores are NOT end-to-end scores.
    sentence_jobs = []
    for row in references:
        for segment in row['segments']:
            if segment['status']=='sentence':
                sentence_jobs.append((segment['text'], (row,segment)))
    with (out/'syntax-comparison.jsonl').open('x',encoding='utf-8') as sink:
        for doc, (row, segment) in nlp.pipe(sentence_jobs, as_tuples=True, batch_size=64):
            identity = turn_id(row)+':'+str(segment['origin']['segment'])
            automatic, ledger = compile_record(doc, {'idx':identity}, audit)
            reference = segment['record']
            equal = compare(reference, automatic)
            flags = risk_flags(doc, automatic, ledger)
            counts = statistics[cohorts[turn_id(row)]]
            counts['diagnostic_sentences'] += 1
            counts['diagnostic_roundtrips'] += 1
            counts['diagnostic_all_features_equal'] += all(equal.values())
            counts['diagnostic_unflagged'] += not flags
            counts['diagnostic_unflagged_all_equal'] += not flags and all(equal.values())
            for field, match in equal.items():
                counts['diagnostic_equal_'+field] += match
            case = dict(id=identity,kind='syntax',cohort=cohorts[turn_id(row)],text=doc.text,
                correction=segment.get('correction'),reference_note=segment.get('note'),flags=flags,equal=equal,
                reference_pattern=CorpusTemplate(reference).text,automatic_pattern=CorpusTemplate(automatic).text,
                reference=features(reference),automatic=features(automatic))
            syntax_rows.append(case)
            write(sink,dict(**case,review_status='UNREVIEWED',automatic_record=automatic))
    sample = select_review(boundary_rows+syntax_rows)
    with (out/'review-sample.jsonl').open('x',encoding='utf-8') as stream:
        for row in sample:
            identity = row['id'] if row['kind']=='boundary' else row['id'].rsplit(':',1)[0]
            turn = int(identity.rsplit(':',1)[1])
            dialogue = contexts[identity]
            write(stream,dict(**row,context_before=dialogue[max(0,turn-3):turn-1],
                context_after=dialogue[turn:turn+1],human_decision='pending'))
    unchanged = all(digest(path)==before for path,before in protected.items())
    if not unchanged:
        raise ValueError('protected bank changed during the experiment')
    report = dict(format=VERSION,status='complete',corpus=corpus,cohorts=dict(statistics),
        structural_roundtrips=sum(s['raw_roundtrips']+s['diagnostic_roundtrips'] for s in statistics.values()),
        review_sample=len(sample),quality_decision='pending_manual_review',published=False,
        protected_bank_unchanged=unchanged,protected_files=len(protected),
        elapsed_seconds=round(time.monotonic()-started,2),
        interpretation=['Feature agreement is not semantic accuracy.',
            'Nominal phrase width differences are also reported with conservative containment alignment.',
            'Diagnostic syntax scores use reviewed boundaries/corrections; raw scores do not.',
            'The stratified review sample cannot estimate an overall error rate without sampling weights.',
            'No new annotation is approved by these heuristics.'],
        next_step='Review sampled disagreements AND agreements; fix or reject risky classes before publication.')
    dump(out/'report.json',report)
    plan['status']='complete'
    dump(out/'plan.json',plan)
    print(json.dumps(report,ensure_ascii=True,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=ROOT)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',default=None)
    args=parser.parse_args()
    model=args.model
    if model is None:
        models=sorted((args.repo/'artifacts/models').glob('spacy-en-*/en_core_web_sm/en_core_web_sm-*'))
        model=str(models[-1]) if models else 'en_core_web_sm'
    run(args.repo,args.out,model)


if __name__=='__main__':
    main()
