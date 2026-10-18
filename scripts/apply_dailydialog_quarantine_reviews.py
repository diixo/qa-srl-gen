"""Apply explicit local repairs and partial readings without publishing them.

Each bundle binds a complete preimage and context. Retokenization is limited
to one unchanged occurrence; ambiguous readings retain context-only usage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import process_dailydialog_hybrid as full
from scripts.apply_dailydialog_local_syntax import apply as syntax
from scripts.audit_dailydialog_hybrid_bank import open_overlay, WholeBankAudit
from scripts.benchmark_dailydialog_hybrid import dump
from scripts.import_dailydialog_sentences import digest
from scripts.resegment_dailydialog_hybrid_source import compile_replacement
from scripts.revise_dailydialog_hybrid_context import apply_review
from scripts.revise_dailydialog_hybrid_lexical import cached_doc
from semantic_corpus.template_bank import CorpusTemplate, fingerprint, read_rows
from semantic_corpus.template_compiler import LexicalAudit, compile_record
from semantic_corpus.syntax_mining import EXTRACTOR_VERSION

VERSION = 'explicit_quarantine_review_v1'


def refresh_constructions(body, payload, context, review, audit, vocab):
    """Recompile only explicitly inspected groups, keeping syntax and scope.

    An earlier context review is retained verbatim, including its restrictions.
    Compiler improvements do not confer semantic approval or publication.
    """
    original=dict(body=deepcopy(body),payload=deepcopy(payload))
    origin=body['provenance']['origin']
    if (review['method']!='direct_assistant_construction_review' or review['id']!=body['idx']
            or review['preimage']!=fingerprint(original) or review['context_sha256']!=fingerprint(context)
            or review['extractor_version']!=EXTRACTOR_VERSION or not review['note'].strip()
            or review['semantic_approval'] is not False or review['publication'] is not False
            or payload.get('method')!='local_spacy_syntax' or payload.get('review_status')!='UNREVIEWED'
            or context[origin['turn']-1][slice(*origin['span'])]!=body['example']):
        raise ValueError('Stale or overbroad construction review')
    policy=payload['record'].get('usage_constraints')
    if body['provenance'].get('context_review') and (not policy or policy['mode']!='context_only'):
        raise ValueError('Construction refresh cannot alter an admitted reference')
    doc=cached_doc(payload['ledger'],vocab)
    compiled,ledger=compile_record(doc,{},audit)
    stable=('i','text','span','parser_lemma','pos','tag','dep','head','morph','kind','verb_lemmas','adjective_lemmas')
    if len(ledger['tokens'])!=len(payload['ledger']['tokens']) or any(
            any(a[k]!=b[k] for k in stable) for a,b in zip(ledger['tokens'],payload['ledger']['tokens'])):
        raise ValueError('Construction refresh changed token evidence')
    for k in ('parts','defaults','verbs'):
        if compiled[k]!=payload['record'][k]:raise ValueError('Construction refresh changed surface or lexical slots')
    if ({g['head_slot']:[c['lemma'] for c in g['constructions']] for g in compiled['predicate_groups']}
            !=review['expected_constructions']
            or ledger.get('temporal_adjunct_candidates',[])!=review['expected_temporal_adjuncts']):
        raise ValueError('Compiled constructions differ from explicit decision')
    if all(compiled[k]==payload['record'][k] for k in ('constructions','predicate_groups')):
        raise ValueError('Construction review has no structural change')
    result=deepcopy(payload)
    for k in ('constructions','predicate_groups'):result['record'][k]=compiled[k]
    shape=('parts','slots','verbs','constructions','predicate_groups','attributes')
    if policy is not None:shape+=('usage_constraints',)
    result['record']['template_id']=fingerprint({k:result['record'][k] for k in shape})
    result['ledger']['template_id']=result['record']['template_id']
    for k in ('predicate_groups','predicate_templates'):result['ledger'][k]=ledger[k]
    if 'temporal_adjunct_candidates' in ledger:
        result['ledger']['temporal_adjunct_candidates']=ledger['temporal_adjunct_candidates']
    else:result['ledger'].pop('temporal_adjunct_candidates',None)
    result['ledger']['construction_revision']=dict(version=EXTRACTOR_VERSION,review=deepcopy(review),preimage=original)
    result['variant']=full.variant_signature(result['record'],result['ledger'])
    updated=deepcopy(body)
    updated['provenance'].update(cache_key=fingerprint([VERSION,'constructions',review]),
        previous_cache_key=body['provenance']['cache_key'])
    if CorpusTemplate(result['record']).render()['text']!=body['example']:
        raise ValueError('Construction refresh changed source text')
    return updated,result


def apply_bundle(body, payload, context, review, audit, vocab):
    original = dict(body=deepcopy(body), payload=deepcopy(payload))
    if (review['id'] != body['idx'] or review['method'] != VERSION
            or review['preimage'] != fingerprint(original)
            or review['context_sha256'] != fingerprint(context)
            or review['semantic_approval'] is not False or review['publication'] is not False
            or not review['steps'] or not review['note'].strip()
            or (body['provenance'].get('context_review')
                and [s['method'] for s in review['steps']]!=['direct_assistant_construction_review'])):
        raise ValueError('Stale or overbroad quarantine decision')
    for step in review['steps']:
        if step['method'] == 'direct_assistant_local_syntax_review':
            body, payload = syntax(body, payload, context, step, audit, vocab)
        elif step['method'] == 'direct_assistant_construction_review':
            body, payload = refresh_constructions(body,payload,context,step,audit,vocab)
        elif step['method'] == 'direct_assistant_source_review':
            if (step['before_turn'] != step['after_turn'] or len(step['outputs']) != 1
                    or step['outputs'][0]['id'] != body['idx']
                    or step['outputs'][0]['span'] != body['provenance']['origin']['span']
                    or step['outputs'][0]['usage']['mode'] != 'context_only'):
                raise ValueError('Quarantine retokenization must preserve the source and restrictions')
            outputs, revised_context = compile_replacement(body, payload, context, step, audit, vocab)
            if revised_context != context:
                raise ValueError('Quarantine review changed context')
            body, payload, _ = outputs[0]
        elif step['method'] == 'direct_assistant_review':
            if step['token_edits'] or step['usage']['mode'] != 'context_only':
                raise ValueError('Partial readings must remain context-only')
            body, payload = apply_review(body, payload, context, step, audit, vocab)
        else:
            raise ValueError('Unsupported quarantine step')
    if (body['idx'] != original['body']['idx'] or body['example'] != original['body']['example']
            or body['provenance']['origin'] != original['body']['provenance']['origin']):
        raise ValueError('Quarantine bundle changed source identity')
    payload = deepcopy(payload)
    payload['ledger']['quarantine_revision'] = dict(
        version=VERSION, preimage=original, review=deepcopy(review), semantic_approval=False)
    return body, payload


def conserve(source, out, decisions, expected):
    def rows(root):
        for path in sorted((root/'bank/occurrences').glob('*.jsonl.gz')):
            yield from read_rows(path)
    targets = {r['id']: r for r in decisions}
    counts = Counter()
    derived = {'template_id', 'group_id', 'syntax_revision', 'tokens', 'attributes', 'bindings',
               'predicate_templates', 'predicate_groups', 'sentence_template', 'analysis_status',
               'imperative_candidates', 'local_syntax_revision', 'cache_key', 'previous_cache_key',
               'candidate_kind', 'quarantine_revision', 'source_revision', 'context_review',
               'previous_original', 'boundary_flags', 'correction',
               'construction_revision', 'temporal_adjunct_candidates'}
    seen = set()
    for a, b in zip_longest(rows(source), rows(out)):
        if a is None or b is None or a['idx'] != b['idx']:
            raise ValueError('Lost or reordered occurrence')
        identity = a['idx']; counts['occurrences'] += 1
        if identity not in targets:
            if a != b:
                raise ValueError('Unaddressed occurrence changed: '+identity)
            counts['unchanged_occurrences'] += 1
            continue
        seen.add(identity)
        rev = b['quarantine_revision']
        image = rev['preimage']
        body, payload = expected[identity]
        if (rev['review'] != targets[identity] or rev['semantic_approval'] is not False
                or fingerprint(image) != targets[identity]['preimage']
                or image['body']['example'] != a['text']
                or image['payload']['ledger']['tokens'] != a['tokens']
                or payload['ledger']['tokens'] != b['tokens']
                or body['provenance']['cache_key'] != b['cache_key']
                or payload['record']['template_id'] != b['template_id']):
            raise ValueError('Quarantine output lost review or preimage')
        allowed = derived
        if any(step['method'] == 'direct_assistant_source_review' for step in targets[identity]['steps']):
            # Retokenization produces a new ledger. Earlier source bindings
            # remain in its complete preimage rather than describing new tokens.
            historical = {'candidate_source_revision', 'candidate_context_revision', 'automatic_revision',
                          'attribute_revision', 'attribute_target_revision'}
            prior = {**image['payload']['ledger'], **image['body']['provenance']}
            if any(name in a and prior.get(name) != a[name] for name in historical):
                raise ValueError('Prior repair history missing from retokenization preimage: '+identity)
            allowed = derived | historical
        if {k:v for k,v in a.items() if k not in allowed} != {k:v for k,v in b.items() if k not in allowed}:
            raise ValueError('Prior source or repair metadata changed: '+identity)
        counts['revised_occurrences'] += 1
    if seen != targets.keys():
        raise ValueError('Missing quarantine decision')
    for name in ('contexts.jsonl.gz', 'coverage.jsonl.gz', 'reviews.jsonl'):
        if digest(source/name) != digest(out/name):
            raise ValueError('Original contexts or reference reviews changed')
    return dict(status='passed', counts=dict(counts), source_unchanged=True,
                prior_decisions_preserved_in_preimages=True, all_unaddressed_occurrences_identical=True)


def run(source, reviews, out, repo=ROOT):
    import spacy
    source, reviews, out, repo = (Path(p).resolve() for p in (source, reviews, out, repo))
    if out.exists() or out.is_relative_to(repo/'data') or out.is_relative_to(source):
        raise ValueError('Use a fresh candidate outside data/')
    baseline = json.loads((source/'report.json').read_text(encoding='utf-8'))
    if baseline['validation']['status'] != 'passed' or baseline['conservation']['status'] != 'passed':
        raise ValueError('Source candidate must pass validation and conservation')
    decisions = list(read_rows(reviews))
    if not decisions or len({r['id'] for r in decisions}) != len(decisions):
        raise ValueError('Empty or duplicate decisions')
    protected = {str(p):digest(p) for root in (repo/'data/mined-templates', repo/'data/dailydialog', source)
                 for p in root.rglob('*') if p.is_file()}
    protected[str(reviews)] = digest(reviews)
    out.mkdir(parents=True)
    db, _, original, lexplan = open_overlay(source, out)
    try:
        if digest(original) != lexplan['source_cache_sha256']:
            raise ValueError('Source cache changed')
        for name, value in lexplan['lexical_resources'].items():
            if digest(repo/name) != value:
                raise ValueError('Lexical resources changed')
        contexts = {r['id']:r['turns'] for r in read_rows(source/'contexts.jsonl.gz')}
        audit, vocab = LexicalAudit(repo/'data/wiktionary'), spacy.blank('en').vocab
        dump(out/'plan.json', dict(version=VERSION, source=str(source), reviews=str(reviews),
            review_sha256=digest(reviews), code_sha256=digest(Path(__file__)), protected=protected,
            api_calls=0, parser_inference_calls=0, published=False))
        expected = {}
        for review in decisions:
            row = db.execute('SELECT o.body,p.payload FROM occurrences o JOIN parses p ON o.key=p.key WHERE o.id=?', (review['id'],)).fetchone()
            if not row: raise ValueError('Missing reviewed occurrence')
            b, p = map(json.loads,row)
            b, p = apply_bundle(b,p,contexts[b['provenance']['context_id']],review,audit,vocab)
            expected[b['idx']] = b,p
            db.execute('INSERT OR REPLACE INTO corrections VALUES (?,?,?,?,?,?)',
                (b['idx'],b['provenance']['cache_key'],b['example'],json.dumps(b),json.dumps(p),json.dumps(review)))
        db.commit()
        for name in ('contexts.jsonl.gz','coverage.jsonl.gz','reviews.jsonl'):
            shutil.copyfile(source/name,out/name)
        checker = WholeBankAudit(source,out)
        try:
            counts = full.export(db,out,observer=checker.observe); checked = checker.finish()
        finally: checker.stream.close()
        meta = json.loads((source/'bank/manifest.json').read_text(encoding='utf-8'))['model']
        manifest, validation = full.materialize(db,out,meta,repo/'data/wiktionary')
        conservation = conserve(source,out,decisions,expected)
        for name, value in protected.items():
            if digest(Path(name)) != value: raise ValueError('Protected input changed: '+name)
        shutil.copytree(source/'bank/review',out/'bank/review')
        shutil.copyfile(reviews,out/'bank/review/quarantine-reviews.jsonl')
        manifest['hybrid_import']['quarantine_review'] = dict(version=VERSION,occurrences=len(decisions),semantic_approval=False)
        dump(out/'bank/manifest.json',manifest)
        report = dict(status='explicit_quarantine_reviews_complete',validation=validation,conservation=conservation,
            counts=counts,dialogues=checked['dialogues'],turns=checked['turns'],protected_files_unchanged=len(protected),
            api_calls=0,parser_inference_calls=0,published=False)
        dump(out/'report.json',report)
        return report
    finally: db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','reviews','out'): parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
