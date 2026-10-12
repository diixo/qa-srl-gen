"""Audit step-3 roles/voice/multiple events and all earlier export checks.

python scripts/review_dialogue_roles.py artifacts/dialogue-mined-roles-step3 \
    --baseline artifacts/dialogue-mined-scopes-step2 \
    --out artifacts/reviews/dialogue-roles-step3-validation.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from review_dialogue_scopes import review_scopes
from semantic_corpus.dialogue_generator.events import Event, EventArgument
from semantic_corpus.dialogue_generator.mined_subjects import MinedSubject
from semantic_corpus.dialogue_generator.extraction_text import _event_payload, _fields, parse_commands


def read_event(data):
    return Event(**{**data, 'arguments': tuple(EventArgument(**a) for a in data['arguments']),
                    'operators': tuple(data['operators']),
                    'missing_roles': tuple(data.get('missing_roles', ())),
                    'subject': MinedSubject(**data['subject']) if data['subject'] else None,
                    'permitter': EventArgument(**data['permitter']) if data['permitter'] else None})


def review_roles(dataset, baseline, max_length):
    report = review_scopes(dataset, baseline, max_length)
    report['roles_and_voice'] = {}
    for split in ('train', 'validation', 'test'):
        voices, roles, examples = Counter(), Counter(), {}
        multiple = withdrawals = 0
        with (dataset / (split + '.episodes.jsonl')).open(encoding='utf-8') as stream:
            for line in stream:
                episode = json.loads(line)
                source = episode.get('mined_source')
                if not source or source['realization'] != 'compositional':
                    continue
                initial = episode['transitions'][1]['delta']['event_add']
                primary = read_event(initial[0]['event'])
                if primary.voice != source['voice']:
                    raise ValueError('Source voice changed')
                if len(initial) != len(source['companion_heads']) + 1:
                    raise ValueError('A companion clause was dropped or invented')
                for claim in initial:
                    event = read_event(claim['event'])
                    kind, fields = _event_payload(event)
                    if parse_commands(kind + ' ' + _fields(fields))[0].event != event:
                        raise ValueError('Event roles or voice changed in text roundtrip')
                    voices[event.voice] += 1
                    roles.update(a.role for a in event.arguments if a.role != 'complement')
                    examples.setdefault(event.voice, {'source': source['source_text'], 'turns': episode['turns'],
                                                       'event': claim['event']})
                if len(initial) > 1:
                    multiple += 1
                    examples.setdefault('multiple', {'source': source['source_text'], 'turns': episode['turns'],
                                                      'events': initial})
                    for companion in initial[1:]:
                        if not any(c['event'] == companion['event'] for c in episode['memory']['events']):
                            raise ValueError('An unrelated companion event was changed or withdrawn')
                    selective = any('Withdraw only this report:' in t['text'] for t in episode['turns'])
                    withdrawals += selective
                    if episode['condition'].endswith((':no', ':unknown')) and not selective:
                        # Only initially asserted primary events need withdrawal.
                        if primary.phase == 'reported' and primary.operators == ():
                            raise ValueError('Missing selective withdrawal contrast')
        report['roles_and_voice'][split] = {'voices': dict(voices), 'roles': dict(roles),
                                            'multi_event_episodes': multiple,
                                            'selective_withdrawals': withdrawals, 'examples': examples}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=768)
    args = parser.parse_args()
    report = review_roles(args.dataset, args.baseline, args.max_length)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
