"""Audit deferred DailyDialog sources against the currently published pool.

Read-only for source data, candidates and the pool. The report distinguishes
unresolved warnings from usable fragment adapters and complete source variants; it never marks
the candidate reviewed or treats deferral as an annotation correction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.finalize_dailydialog_quarantine import current_ids, key
from scripts.import_dailydialog_sentences import digest
from semantic_corpus.dailydialog_quarantine import (
    REGISTER_NAME, check_template, load_holds,
)
from semantic_corpus.dialogue_generator.fragments import load_fragment_patterns, fragment_episode
from semantic_corpus.dialogue_generator.export import extraction_row, response_row
from semantic_corpus.dialogue_generator.extraction_text import decode_update, parse_prompt
from semantic_corpus.template_bank import TemplateBank, fingerprint, read_rows
from semantic_corpus.template_registry import _check_hybrid_admission


def verify_evidence(repo, register):
    """Keep evidence immutable while allowing an unchanged catalog prefix.

    A catalog snapshot is bound to the original *file* hash, including its
    metadata and order. Only appended patterns may differ; a current-catalog
    hash must never replace the historical evidence hash to silence the audit.
    """
    catalog_name = 'data/mined-templates/reviewed/dialogue-fragments.json'
    for name, expected in register['evidence'].items():
        path = repo/name
        if digest(path) == expected:
            continue
        snapshot = register.get('catalog_evidence', {})
        count = snapshot.get('patterns')
        if (name != catalog_name or snapshot.get('path') != name
                or snapshot.get('serialization') != 'utf8-json-indent2-lf'
                or type(count) is not int or count < 0):
            raise ValueError('Review evidence changed: '+name)
        catalog = json.loads(path.read_text(encoding='utf-8'))
        patterns = catalog.get('patterns')
        if not isinstance(patterns, list) or len(patterns) < count:
            raise ValueError('Review evidence changed: '+name)
        baseline = {**catalog, 'patterns': patterns[:count]}
        encoded = (json.dumps(baseline, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
        if sha256(encoded).hexdigest() != expected:
            raise ValueError('Review evidence changed: '+name)


def audit_inputs(repo, register, ledger=None):
    """Use the imported pool evidence after the one-time source workspace is gone."""
    state_name = register.get('import_state')
    if state_name:
        state_path = (repo/state_name).resolve()
        if not state_path.is_relative_to((repo/'data/mined-templates').resolve()):
            raise ValueError('Import state must live in the shared pool')
        state = json.loads(state_path.read_text(encoding='utf8'))
        central_ledger = (repo/state['ledger']).resolve()
        if ledger is None or Path(ledger).resolve() == central_ledger:
            if (state.get('format') != 'dailydialog-import-state-v1'
                    or state.get('review_parts_required') is not False
                    or state.get('candidate_published') is not False):
                raise ValueError('Invalid one-time import state')
            for name, proof in state['files'].items():
                path = (repo/name).resolve()
                if (not path.is_relative_to((repo/'data/mined-templates').resolve())
                        or digest(path) != proof['sha256']):
                    raise ValueError('Imported evidence changed: '+name)
            candidate = repo/state['candidate']
            if state.get('candidate_source_id'):
                from semantic_corpus.template_pending import verify_pending
                entries = {e['id']: e for e in verify_pending(candidate)}
                for proof in state['shared_ranges'].values():
                    entry = entries[proof['source_id']]
                    if (entry['file_sha256'][proof['file']] != proof['sha256']
                            or entry['files'][proof['file']][1] != proof['size']):
                        raise ValueError('Shared import range changed')
                archive_path = repo/state['checkpoint']/'manifest.json'
                if digest(archive_path) != state['checkpoint_manifest_sha256']:
                    raise ValueError('Import checkpoint changed')
                archived = {f['path']: f for f in json.loads(archive_path.read_text(encoding='utf8'))['files']}
                for name, proof in state['archived_files'].items():
                    if any(archived[name][field] != proof[field] for field in ('size', 'sha256')):
                        raise ValueError('Missing archived import evidence: '+name)
                candidate = TemplateBank(candidate, _entry=entries[state['candidate_source_id']])
            return dict(ledger=central_ledger, source=repo/state['source'],
                        annotations=repo/state['annotations'], candidate=candidate,
                        parts=None, centralized=True)
    if ledger is None:
        raise ValueError('Specify the historical ledger or consolidate the one-time import')
    return dict(ledger=Path(ledger).resolve(), source=repo/'data/dailydialog/filtered',
                annotations=repo/'data/dailydialog/review-parts/annotations',
                parts=repo/'data/dailydialog/review-parts', candidate=None, centralized=False)


def audit(repo, ledger=None):
    repo = Path(repo).resolve()
    pool = repo/'data/mined-templates'
    register_path = pool/'reviewed'/REGISTER_NAME
    catalog_path = pool/'reviewed/dialogue-fragments.json'
    holds = load_holds(register_path)
    register = json.loads(register_path.read_text(encoding='utf-8'))
    inputs = audit_inputs(repo, register, ledger)
    ledger = inputs['ledger']
    summary = json.loads((ledger/'summary.json').read_text(encoding='utf-8'))
    warnings = list(read_rows(ledger/'quarantined-warnings.jsonl.gz'))
    cases = {r['id']:r for r in read_rows(ledger/'quarantine-cases.jsonl.gz')}
    indexed = {row['id']:row for row in holds}
    unresolved_keys = {tuple(r['warning_key']) for r in holds
                       if r['status'] not in {'source_variant_only', 'excluded'}}
    excluded_keys = {tuple(r['warning_key']) for r in holds if r['status'] == 'excluded'} - unresolved_keys
    resolved_keys = {tuple(r['warning_key']) for r in holds} - unresolved_keys - excluded_keys
    if (set(indexed) != set(cases)
            or len(warnings) != register.get('source_warnings', register['unresolved_warnings'])
            or len(unresolved_keys) != register['unresolved_warnings']
            or len(excluded_keys) != register.get('excluded_warnings', 0)
            or len(resolved_keys) != register.get('source_variant_resolved_warnings', 0)):
        raise ValueError('Register does not cover the current quarantine')
    verify_evidence(repo, register)
    for warning in warnings:
        for identity in current_ids(warning):
            entry = indexed[identity]
            if (entry['warning_key'] != list(key(warning))
                    or entry['warning_sha256'] != fingerprint(warning)
                    or entry['candidate_image_sha256'] != fingerprint(cases[identity]['image'])):
                raise ValueError('Stale quarantine disposition: '+identity)
    raw = {split:list(read_rows(inputs['source']/f'dailydialog-{split}.jsonl'))
           for split in ('train','val','test')}
    for entry in holds:
        s = entry['source']
        if raw[s['split']][s['dialogue']-1][:s['turn']] != entry['original_prefix']:
            raise ValueError('Original prefix changed: '+entry['id'])

    protected_paths = list(p for p in pool.rglob('*') if p.is_file())
    protected_paths += list(inputs['source'].glob('*.jsonl'))
    if inputs['parts'] is not None:
        protected_paths += list(inputs['parts'].glob('part-*.jsonl'))
    protected_paths += list(inputs['annotations'].glob('part-*.jsonl'))
    protected_paths += list(p for p in ledger.iterdir() if p.is_file())
    before = {p:digest(p) for p in protected_paths}

    patterns = load_fragment_patterns(catalog_path)
    by_pattern = {p['id']:p for p in patterns}
    allowed = {identity:sha for entry in holds for identity,sha in entry['allowed_adapters'].items()}
    variants = {e['source_variant']['pattern_id']:e['source_variant']['pattern_sha256']
                for e in holds if e['status']=='source_variant_only'}
    if any(identity not in by_pattern or fingerprint(by_pattern[identity]) != sha
           for identity,sha in allowed.items()):
        raise ValueError('A reviewed adjacent adapter was lost or changed')
    for identity in variants:
        source = by_pattern[identity]['source']
        if raw[source['split']][source['dialogue']-1] != source['context']:
            raise ValueError('Reviewed source variant differs from preserved source: '+identity)
    bank = TemplateBank(pool)
    inspected = []
    rendered = 0
    for component in bank.components():
        if not component.source_id.startswith('dailydialog-'):
            continue
        _check_hybrid_admission(component, repo_root=repo)
        count = 0
        for template in component.templates():
            check_template(template.record, holds)
            template.render()
            count += 1
        rendered += count
        inspected.append(dict(source=component.source_id, templates=count))

    # Negative checks use changed in-memory records, never mutate a candidate.
    rejected = 0
    for case in cases.values():
        record = deepcopy(case['image']['payload']['record'])
        record.setdefault('usage_constraints', {})['mode'] = 'attested_sentence'
        try:
            check_template(record, holds)
        except ValueError as error:
            if 'quarantine blocks' not in str(error):
                raise
            rejected += 1
        else:
            raise ValueError('Quarantine could be admitted: '+case['id'])
    candidate_bank = inputs['candidate'] or Path(summary['candidate'])/'bank'
    try:
        _check_hybrid_admission(candidate_bank if isinstance(candidate_bank, TemplateBank)
                                else TemplateBank(candidate_bank), repo_root=repo)
    except ValueError as error:
        if 'unreviewed hybrid quarantine' not in str(error):
            raise
    else:
        raise ValueError('Full unreviewed candidate could be published')

    episodes = extraction = responses = 0
    for identity in {*allowed, *variants}:
        for user_only in (True,False):
            context = ('the kitchen','tired','tea') if user_only else ('Anna','Rome')
            episode = fragment_episode(by_pattern[identity], context, split='train', user_only=user_only)
            episode.validate()
            if episode.memory.events:
                raise ValueError('Adjacent interpretation invented a completed event')
            for transition in episode.transitions:
                row = extraction_row(episode, transition)
                prompt, target = (m['content'] for m in row['messages'][1:])
                memory, turn, previous = parse_prompt(prompt)
                if decode_update(target,transition.before,transition.turn,
                                 transition.previous_assistant) != transition.delta:
                    raise ValueError('Adjacent adapter extraction failed')
                # The text prompt intentionally omits the original turn IDs;
                # parse_prompt creates local IDs. Validate that path too, but
                # compare its evidence against those local IDs, not u2 vs u1.
                decoded = decode_update(target,memory,turn,previous)
                local_ids = {turn.id} | ({previous.id} if previous else set())
                if any(e.turn_id not in local_ids for fact in decoded.add for e in fact.evidence):
                    raise ValueError('Decoded adapter evidence is outside the visible pair')
                extraction += 1
            for view in ('history','memory'):
                response_row(episode,view)
                responses += 1
            episodes += 1
    for path,sha in before.items():
        if digest(path) != sha:
            raise ValueError('Protected file changed: '+str(path))
    manual = {(r['split'],r['dialogue']) for p in inputs['annotations'].glob('part-*.jsonl')
              for r in read_rows(p)}
    return dict(status='passed',register=register_path.relative_to(repo).as_posix(),register_sha256=digest(register_path),
        source_warnings=len(warnings),unresolved_warnings=len(unresolved_keys),active_automatic_retries=0,
        excluded_warnings=len(excluded_keys),
        excluded_occurrences=sum(r['status']=='excluded' for r in holds),
        source_variant_resolved_warnings=len(resolved_keys),source_variant_adapters=len(variants),
        input_layout='shared_pool' if inputs['centralized'] else 'historical_source_workspace',
        review_parts_required=not inputs['centralized'],
        deferred_occurrences=sum(r['status']=='deferred_missing_evidence' for r in holds),
        blocked_template_occurrences=len(holds),allowed_adjacent_adapters=len(allowed),
        disposition_categories=dict(Counter(r['category'] for r in holds)),
        published_dailydialog_components=inspected,published_templates_checked=rendered,
        published_quarantined_templates=0,fragment_patterns_checked=len(patterns),
        rejected_admission_probes=rejected,full_candidate_publication_blocked=True,
        generation=dict(episodes=episodes,extraction=extraction,history_and_memory=responses),
        source_dialogues_preserved=sum(map(len,raw.values())),manual_dialogues_preserved=len(manual),
        protected_files_unchanged=len(before),original_text_changed=False,candidate_rebuilt=False,
        paid_evidence_sha256=digest(ledger/'received-evidence.jsonl.gz'),
        source_ambiguities_resolved=0,full_candidate_published=False,api_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('choose a new audit report path')
    result = audit(args.repo,args.ledger)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='published_dailydialog_components'},indent=2))
