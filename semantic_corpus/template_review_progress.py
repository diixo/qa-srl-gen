"""Count occurrence decisions without treating a reviewed example as a family approval."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from .template_bank import TemplateBank, fingerprint
from .template_pending import load_pending


def count_coverage(occurrences, decisions, published, closed):
    totals, remaining = Counter(), Counter()
    outcomes = Counter()
    seen = set()
    for row in occurrences:
        identity, template = row['idx'], row['template_id']
        if identity in seen:
            raise ValueError('Duplicate candidate occurrence: ' + identity)
        seen.add(identity)
        totals[template] += 1
        if identity in closed:
            outcome = 'closed_restricted'
        elif identity in decisions:
            outcome = decisions[identity]
        elif row['text'] in published.get(identity, set()):
            outcome = 'published_attestation'
        else:
            outcome = 'remaining'
            remaining[template] += 1
        outcomes[outcome] += 1
    untouched = sum(remaining[t] == n for t, n in totals.items())
    return dict(
        candidate_templates=len(totals), candidate_occurrences=len(seen),
        occurrence_outcomes=dict(sorted(outcomes.items())),
        remaining_occurrences=outcomes['remaining'],
        templates_with_remaining_occurrences=len(remaining),
        templates_without_decisions=untouched,
        templates_partially_resolved=len(remaining)-untouched,
        templates_all_occurrences_resolved=len(totals)-len(remaining),
        note='Resolved includes restricted adapters and closed exclusions; it does not license arbitrary substitutions or approve every meaning of a template.')


def review_progress(root):
    root = Path(root)
    entries = load_pending(root)
    candidate = next(e for e in entries if e['id'] == 'dailydialog')
    decisions, pins = {}, {}
    for key, outcome in [('sentence_attestations', 'attested_sentence'),
                         ('occurrence_adaptations', 'context_adapter')]:
        for row in candidate.get('review', {}).get(key, {}).get('decisions', []):
            identity = row['occurrence_id']
            if identity in decisions:
                raise ValueError('Conflicting occurrence decisions: ' + identity)
            decisions[identity] = outcome
            pins[identity] = row['candidate_occurrence_sha256']
    published = defaultdict(set)
    for component in TemplateBank(root).components():
        if not component.manifest.get('hybrid_import'):
            continue
        for template in component.templates():
            policy = template.record.get('usage_constraints', {})
            if policy.get('mode') == 'attested_sentence':
                published[policy['source_id']].add(policy['text'])
    quarantine = json.loads((root/'reviewed/dailydialog-quarantine.json').read_text(encoding='utf8'))
    closed = {r['id'] for r in quarantine['occurrences']
              if r.get('automatic_retry') is False and not r.get('template_admission')}
    def checked_occurrences():
        for row in TemplateBank(root, _entry=candidate).occurrences():
            if row['idx'] in pins and fingerprint(row) != pins[row['idx']]:
                raise ValueError('Stale candidate occurrence decision: ' + row['idx'])
            yield row
    result = count_coverage(checked_occurrences(), decisions, published, closed)
    result['unresolved_quarantine_warnings'] = quarantine['unresolved_warnings']
    result['lexical_remaining'] = {
        name: review['counts'].get('remaining_records', 0)
        for e in entries if e['kind'] == 'inventories'
        for name, review in e.get('review', {}).items()}
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default='data/mined-templates')
    print(json.dumps(review_progress(parser.parse_args().root), indent=2))
