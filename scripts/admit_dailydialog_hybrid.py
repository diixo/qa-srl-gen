"""Compile only explicitly admitted attested sentences from a hybrid revision.

No inference, automatic family approval, or source ZIP duplication. Compilation
is separate from publication so the selected component can be tested first.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_dailydialog_hybrid_bank import open_overlay
from scripts.benchmark_dailydialog_hybrid import dump, write
from scripts.import_dailydialog_sentences import digest
from scripts.process_dailydialog_hybrid import CachedInputs
from semantic_corpus.template_bank import CorpusTemplate, TemplateBank, fingerprint, read_rows
from semantic_corpus.template_compiler import build_bank
from semantic_corpus.template_registry import _check_hybrid_admission, register_bank
from semantic_corpus.template_validation import validate_bank
from semantic_corpus.dailydialog_quarantine import check_admitted_ids, pool_holds


def apply_admission(body, payload, context, decision):
    """Permit an explicit attested-only promotion, retaining the earlier review.

    Repairing a token or closing a warning does not call this operation. An
    independent admission decision must bind the current restricted record;
    it may change neither its meaning nor any syntax or source evidence.
    """
    from scripts.revise_dailydialog_hybrid_context import apply_review
    if decision.get('method') != 'direct_assistant_attestation_admission':
        return apply_review(body,payload,context,decision,None,None)
    previous=body['provenance'].get('context_review')
    policy=payload['record'].get('usage_constraints',{})
    expected={k:deepcopy(v) for k,v in policy.items() if k in {
        'mode','reasons','fixed_expressions','alternative_properties','property_bindings'}}
    expected['mode']='attested_sentence'
    if (not previous or previous.get('unit')!='sentence'
            or policy.get('mode')!='context_only'
            or payload.get('method')!='local_spacy_syntax'
            or payload.get('review_status')!='UNREVIEWED'
            or decision.get('previous_context_review_sha256')!=fingerprint(previous)
            or decision.get('semantic_approval') is not False or decision.get('publication') is not False
            or decision.get('unit')!='sentence' or decision.get('token_edits')!=[]
            or decision.get('meaning')!=policy.get('meaning') or decision.get('usage')!=expected
            or previous.get('meaning')!=policy.get('meaning')
            or policy.get('context')!=context or policy.get('text')!=body['example']
            or policy.get('source_id')!=body['idx']
            or decision.get('evidence_turns')!=previous.get('evidence_turns')
            or not decision.get('note','').strip()):
        raise ValueError('Attestation admission cannot rewrite or bypass the previous context review')
    CorpusTemplate(payload['record']).render()  # Validate the existing restrictions before promotion.
    preimage=dict(body=deepcopy(body),payload=deepcopy(payload))
    staged=deepcopy(body)
    staged['provenance'].setdefault('context_review_history',[]).append(
        staged['provenance'].pop('context_review'))
    # The existing validator checks source, payload and complete dialogue hashes.
    updated,result=apply_review(staged,payload,context,
        {**decision,'method':'direct_assistant_review'},None,None)
    result['ledger']['attestation_admission']=dict(
        review=deepcopy(decision),preimage=preimage,semantic_approval=False)
    return updated,result


def compile_admission(source, reviews, out, repo=ROOT, *, quarantine=None):
    source, out = source.resolve(), out.resolve()
    if out.exists() or out.is_relative_to(repo.resolve()/'data') or out.is_relative_to(source):
        raise ValueError('use a fresh quarantine output outside data/')
    review_paths = [reviews] if isinstance(reviews, Path) else list(reviews)
    decisions = [row for path in review_paths for row in read_rows(path)]
    if not decisions or len({r['id'] for r in decisions}) != len(decisions):
        raise ValueError('empty or duplicate review decisions')
    accepted = {r['id']:r for r in decisions if r.get('usage',{}).get('mode')=='attested_sentence'}
    if not accepted:
        raise ValueError('no explicitly admitted sentences')
    check_admitted_ids(accepted, pool_holds(repo))
    frozen = {}
    blocked, contexts = set(), {}
    if quarantine is not None:
        quarantine = Path(quarantine).resolve()
        summary = json.loads((quarantine/'summary.json').read_text(encoding='utf-8'))
        baseline = json.loads((source/'report.json').read_text(encoding='utf-8'))
        if (Path(summary['candidate']).resolve() != source
                or baseline['validation']['status'] != 'passed'
                or baseline['conservation']['status'] != 'passed'):
            raise ValueError('quarantine must describe the validated source candidate')
        cases = list(read_rows(quarantine/'quarantine-cases.jsonl.gz'))
        blocked = {r['id'] for r in cases}
        if len(blocked) != len(cases) or len(blocked) != summary['quarantined_current_occurrences']:
            raise ValueError('quarantine coverage mismatch')
        for case in cases:
            if (case['image']['body']['idx'] != case['id']
                    or fingerprint(case['image']) != case['image_sha256']):
                raise ValueError('quarantine image mismatch')
        if blocked & accepted.keys():
            raise ValueError('open quarantine occurrence cannot be admitted: '+str(sorted(blocked & accepted.keys())))
        if any(r.get('token_edits') != [] for r in decisions):
            raise ValueError('admission must not silently repair syntax; apply repairs separately')
        contexts = {r['id']:r['turns'] for r in read_rows(source/'contexts.jsonl.gz')}
        paths = [source/name for name in ('plan.json','report.json','revision.sqlite3','contexts.jsonl.gz')]
        paths += review_paths + [quarantine/'summary.json',quarantine/'quarantine-cases.jsonl.gz']
        frozen = {str(p.resolve()):digest(p) for p in paths}
    out.mkdir(parents=True)
    records = {}
    snapshots = []
    with tempfile.TemporaryDirectory(prefix='read-overlay-', dir=out) as temporary:
        db, *_ = open_overlay(source,Path(temporary))
        try:
            with (out/'sentences.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
                selected = {r['id']:r for r in decisions} if quarantine is not None else accepted
                for identity, decision in selected.items():
                    if quarantine is not None:
                        found = db.execute('SELECT o.body,p.payload FROM occurrences o '
                            'JOIN parses p ON p.key=o.key WHERE o.id=?', (identity,)).fetchone()
                        if not found:
                            raise ValueError('unknown contextual occurrence: '+identity)
                        body, payload = map(json.loads,found)
                        context = contexts[body['provenance']['context_id']]
                        preimage = dict(body=deepcopy(body),payload=deepcopy(payload))
                        body, payload = apply_admission(body,payload,context,decision)
                        snapshots.append(dict(id=identity,preimage=preimage,
                            preimage_sha256=fingerprint(preimage),context=context,review=decision,
                            body=body,payload=payload))
                        if identity not in accepted:
                            continue
                        records[identity] = payload
                        write(stream,body)
                        continue
                    found = db.execute('SELECT o.body,p.payload,c.review FROM occurrences o '
                        'JOIN parses p ON p.key=o.key JOIN corrections c ON c.id=o.id WHERE o.id=?',
                        (identity,)).fetchone()
                    if not found:
                        raise ValueError('review was not applied: '+identity)
                    body,payload,stored = map(json.loads,found)
                    if decision!=stored or decision['unit']!='sentence':
                        raise ValueError('admission differs from the applied contextual decision')
                    record = payload['record']
                    usage=record.get('usage_constraints',{})
                    if usage.get('mode')!='attested_sentence' or usage.get('source_id')!=identity:
                        raise ValueError('record lost its attested-only restrictions')
                    if CorpusTemplate(record).render()['text']!=body['example']:
                        raise ValueError('admission text mismatch')
                    records[identity]=payload
                    write(stream,body)
        finally:
            db.close()

    def cached(doc, provenance, audit):
        payload=deepcopy(records[provenance['idx']])
        if payload['ledger']['text']!=doc.text:
            raise ValueError('admission source mismatch')
        return payload['record'],dict(**payload['ledger'],**provenance)

    manifest=build_bank(out/'sentences.jsonl',out/'bank',out/'shards',
        CachedInputs({'name':'explicit contextual attestation reviews','parser':None}),
        wiktionary=repo/'data/wiktionary',record_compiler=cached)
    manifest['hybrid_import']=dict(review_status='REVIEWED_ATTESTATIONS',
        admitted_occurrences=list(accepted), semantic_approval=False,
        source_revision=str(source),
        review_inputs=[dict(path=str(path),sha256=digest(path)) for path in review_paths],
        note='Exact attested sentences only. Context and limited meanings are embedded; '
             'dependency QA and recombination are not licensed.')
    if len(review_paths) == 1:
        manifest['hybrid_import'].update(review_file=str(review_paths[0]),review_sha256=digest(review_paths[0]))
    manifest['note']=manifest['hybrid_import']['note']
    # The original file may be deleted after publication. Every record is self-contained.
    manifest.pop('source',None)
    dump(out/'bank/manifest.json',manifest)
    bank=TemplateBank(out/'bank')
    _check_hybrid_admission(bank, repo_root=repo)
    validation=validate_bank(bank,out/'sentences.jsonl')
    dump(out/'bank/validation.json',validation)
    report=dict(admitted_occurrences=list(accepted),
                held_back=[r['id'] for r in decisions if r['id'] not in accepted],
                validation=validation,api_calls=0,parser_inference_calls=0,published=False)
    if quarantine is not None:
        with gzip.open(out/'contextual-reviews.jsonl.gz','wt',encoding='utf-8',newline='\n') as stream:
            for snapshot in snapshots:
                write(stream,snapshot)
        for name,sha in frozen.items():
            if digest(Path(name)) != sha:
                raise ValueError('protected admission input changed: '+name)
        report.update(contextual_reviews=len(snapshots),quarantine=str(quarantine),
            open_quarantine_occurrences=len(blocked),protected_inputs=frozen,
            candidate_unchanged=True,review_snapshot_sha256=digest(out/'contextual-reviews.jsonl.gz'))
    dump(out/'report.json',report)
    return report


def publish(out, bank_id, repo=ROOT):
    """Append and verify every pre-existing gzip byte range is still identical."""
    pool=repo/'data/mined-templates'
    before={p.relative_to(pool).as_posix():[p.stat().st_size,digest(p)]
            for p in pool.rglob('*') if p.is_file() and p.name!='manifest.json'}
    old=json.loads((pool/'manifest.json').read_text(encoding='utf-8')) if (pool/'manifest.json').exists() else None
    register_bank(out/'bank',out/'sentences.jsonl',repo_root=repo,bank_id=bank_id,archive_source=False)
    for name,(length,sha) in before.items():
        if digest(pool/name,length)!=sha:
            raise ValueError('previous pool bytes changed: '+name)
    current=TemplateBank(pool)
    if old and 'sources' in old and current.manifest['sources'][:len(old['sources'])]!=old['sources']:
        raise ValueError('previous committed components changed')
    report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    report.update(published=True,bank_id=bank_id,previous_files_preserved=len(before))
    dump(out/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--reviews',type=Path,nargs='+')
    parser.add_argument('--quarantine',type=Path,
                        help='Apply new context-only/attested reviews to a subset, blocking open quarantine IDs; no full-candidate rebuild')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--bank-id',default='dailydialog-hybrid-attested-v1')
    args=parser.parse_args()
    if args.publish:
        result=publish(args.out,args.bank_id)
    else:
        if args.source is None or args.reviews is None:
            parser.error('--source and --reviews are required for compilation')
        result=compile_admission(args.source,args.reviews,args.out,quarantine=args.quarantine)
    print(json.dumps(result,indent=2))
