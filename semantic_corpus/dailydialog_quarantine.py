"""Occurrence-level holds; reviewed fragments are explicit exceptions.

The register lives beside the reviewed catalog. Runtime checks need neither the
original DailyDialog files nor candidate artifacts. A hold is lifted by a new
review of its evidence, never by changing a template's admission label.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from .template_bank import fingerprint

REGISTER_NAME = 'dailydialog-quarantine.json'
REPO_ROOT = Path(__file__).resolve().parents[1]
_ID = re.compile(r'(train|val|test):[1-9][0-9]*:[1-9][0-9]*:[1-9][0-9]*\Z')


def load_holds(path):
    path = Path(path)
    if not path.exists():
        return ()  # Other repositories/catalogs need not contain DailyDialog.
    register = json.loads(path.read_text(encoding='utf-8'))
    if register.get('format') != 'dailydialog-quarantine-v1':
        raise ValueError('Unsupported DailyDialog quarantine register')
    entries = register.get('occurrences')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Empty DailyDialog quarantine register')
    seen = set()
    for entry in entries:
        identity = entry.get('id', '')
        source = entry.get('source', {})
        span = source.get('span')
        prefix = entry.get('original_prefix')
        if (not isinstance(identity, str) or not _ID.fullmatch(identity) or identity in seen
                or entry.get('template_admission') is not False
                or entry.get('automatic_retry') is not False
                or entry.get('status') not in {'deferred_missing_evidence', 'adapter_only', 'source_variant_only', 'excluded'}
                or not entry.get('reason') or not entry.get('unknown')):
            raise ValueError('Invalid DailyDialog quarantine decision')
        if entry['status'] == 'excluded':
            if entry.get('review_closed') is not True or entry.get('reopen_when') is not None:
                raise ValueError('An excluded occurrence must be closed without a retry condition')
        elif not entry.get('reopen_when'):
            raise ValueError('An open quarantine decision needs its evidence condition')
        seen.add(identity)
        split, dialogue, turn, segment = identity.split(':')
        if (source.get('split') != split or source.get('dialogue') != int(dialogue)
                or source.get('turn') != int(turn) or source.get('segment') != int(segment)
                or not isinstance(prefix, list) or len(prefix) != int(turn)
                or any(not isinstance(t, str) for t in prefix)
                or not isinstance(span, list) or len(span) != 2
                or any(type(n) is not int for n in span)
                or not 0 <= span[0] < span[1] <= len(prefix[-1])
                or prefix[-1][slice(*span)] != source.get('text')
                or fingerprint(prefix) != entry.get('original_prefix_sha256')):
            raise ValueError('Invalid DailyDialog quarantine source evidence')
        allowed = entry.get('allowed_adapters')
        if not isinstance(allowed, dict) or any(
                not isinstance(k, str) or not k or not isinstance(v, str)
                or re.fullmatch('[0-9a-f]{64}', v) is None for k, v in allowed.items()):
            raise ValueError('Invalid DailyDialog quarantine adapter exceptions')
        if entry['status'] != 'excluded' and bool(allowed) != (entry['status'] == 'adapter_only'):
            raise ValueError('Only reviewed adjacent fragments can have adapter exceptions')
        variant = entry.get('source_variant')
        if entry['status'] == 'source_variant_only':
            if (not isinstance(variant, dict) or not isinstance(variant.get('pattern_id'), str)
                    or not variant['pattern_id']
                    or not isinstance(variant.get('pattern_sha256'), str)
                    or re.fullmatch('[0-9a-f]{64}', variant['pattern_sha256']) is None
                    or not isinstance(variant.get('reason'), str) or not variant['reason']):
                raise ValueError('A source variant needs a reviewed, pinned adapter')
        elif variant is not None:
            raise ValueError('Source variant attached to an incompatible quarantine status')
    return tuple(entries)


def pool_holds(repo_root=None):
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    return load_holds(root / 'data/mined-templates/reviewed' / REGISTER_NAME)


def matches_source(entry, source):
    """Match the address or an overlapping span, including renumbered segments."""
    old = entry['source']
    if any(source.get(k) != old[k] for k in ('split', 'dialogue', 'turn')):
        return False
    if source.get('segment') == old['segment']:
        return True
    span = source.get('span')
    return (isinstance(span, (list, tuple)) and len(span) == 2
            and all(type(n) is int for n in span)
            and span[0] < old['span'][1] and old['span'][0] < span[1])


def check_admitted_ids(identities, entries):
    blocked = set(identities) & {entry['id'] for entry in entries}
    if blocked:
        raise ValueError('DailyDialog quarantine blocks template admission: ' + ', '.join(sorted(blocked)))


def check_template(record, entries):
    check_admitted_ids([record.get('usage_constraints', {}).get('source_id')], entries)
    for example in record.get('examples', []):
        for entry in entries:
            if matches_source(entry, example.get('origin', {})):
                raise ValueError('DailyDialog quarantine blocks source span: ' + entry['id'])


def check_fragment_catalog(patterns, catalog):
    entries = load_holds(Path(catalog).parent / REGISTER_NAME)
    by_id = {p['id']: p for p in patterns}
    for entry in entries:
        if entry['status'] != 'source_variant_only':
            continue
        variant = entry['source_variant']
        pattern = by_id.get(variant['pattern_id'])
        if pattern is None or fingerprint(pattern) != variant['pattern_sha256']:
            raise ValueError('Reviewed source variant adapter was lost or changed: ' + entry['id'])
        source = pattern.get('source', {})
        context, turn = source.get('context'), source.get('turn')
        text = source.get('text', '')
        old_turn, old_text = entry['original_prefix'][-1], entry['source']['text']
        if (pattern.get('review', {}).get('method') != 'direct_assistant_fragment_adapter'
                or matches_source(entry, source)
                or not isinstance(context, list) or type(turn) is not int
                or not 1 <= turn <= len(context) or not isinstance(context[turn-1], str)
                or not isinstance(text, str) or len(text) <= len(old_text)
                or not context[turn-1].casefold().startswith(old_turn.casefold())
                or not text.casefold().startswith(old_text.casefold())):
            raise ValueError('Source variant does not preserve and extend the original turn: ' + entry['id'])
    for pattern in patterns:
        for entry in entries:
            if matches_source(entry, pattern.get('source', {})):
                if entry['allowed_adapters'].get(pattern['id']) != fingerprint(pattern):
                    raise ValueError('DailyDialog quarantine blocks fragment adapter: ' + entry['id'])
