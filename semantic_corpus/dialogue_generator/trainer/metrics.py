"""Strict answer matching plus a small, explicit diagnostic answer taxonomy."""

from ..extraction_text import decode_update, parse_commands, parse_prompt
from ..models import apply_delta


def answer_kind(text):
    text = text.strip().casefold()
    if text.startswith('yes,'):
        return 'yes'
    if text.startswith('no,'):
        return 'no'
    if text.startswith("i don't know"):
        return 'unknown'
    return 'other'


class AnswerMetrics:
    def __init__(self):
        self.by_answer = {}
        self.by_task = {}
        self.by_condition = {}
        self.groups = {}

    def add(self, reference, prediction, *, task, condition=None, group=None):
        expected, actual = answer_kind(reference), answer_kind(prediction)
        exact = reference.strip() == prediction.strip()
        for table, key in ((self.by_answer, expected), (self.by_task, task),
                           (self.by_condition, condition or 'unspecified')):
            counts = table.setdefault(key, {'count': 0, 'exact_matches': 0, 'confusion': {}})
            counts['count'] += 1
            counts['exact_matches'] += int(exact)
            row = counts['confusion'].setdefault(expected, {})
            row[actual] = row.get(actual, 0) + 1
        if group:
            self.groups.setdefault(group, []).append((expected, exact))

    def report(self):
        complete = [rows for rows in self.groups.values()
                    if len(rows) == 3 and {kind for kind, _ in rows} == {'yes', 'no', 'unknown'}]
        correct = sum(all(exact for _, exact in rows) for rows in complete)
        unknown = self.by_answer.get('unknown', {})
        false_yes = unknown.get('confusion', {}).get('unknown', {}).get('yes', 0)
        return {'by_answer_kind': self.by_answer, 'task_answer_confusion': self.by_task,
                'by_condition': self.by_condition, 'unknown_false_yes': false_yes,
                'unknown_count': unknown.get('count', 0),
                'contrast_groups': {'complete': len(complete), 'all_correct': correct,
                                    'partial_or_duplicate': len(self.groups) - len(complete),
                                    'accuracy': correct / len(complete) if complete else None},
                'answer_kind_note': 'Prefix-based diagnostic labels; exact match remains the primary metric.'}


class ExtractionMetrics:
    """Replay predictions on supplied prior state; do not grade by prose prefixes."""
    def __init__(self):
        self.count = self.valid = self.updates = self.states = self.false_changes = 0
        self.by_operation = {}

    def add(self, reference, prediction, *, prompt):
        memory, user, previous = parse_prompt(prompt)
        turns = ((previous,) if previous else ()) + (user,)
        expected = decode_update(reference, memory, user, previous)
        wanted = apply_delta(memory, expected, turns)
        operations = ({name for name, values in (('add', expected.add), ('replace', expected.replace),
                                                 ('remove', expected.invalidate), ('event_add', expected.event_add),
                                                 ('event_replace', expected.event_replace),
                                                 ('event_remove', expected.event_invalidate)) if values} or {'no_change'})
        valid = update_match = state_match = False
        try:
            delta = decode_update(prediction, memory, user, previous)
            result = apply_delta(memory, delta, turns)
            valid = True
            # Order/new IDs are implementation details. Source selection remains
            # part of update equality (SET differs from SET_CONTEXT).
            update_match = set(parse_commands(reference)) == set(parse_commands(prediction))
            def facts(state):
                return ({(f.subject, f.relation, f.modality, f.polarity, f.value) for f in state.facts},
                        {claim.event for claim in state.events})
            state_match = facts(wanted) == facts(result)
        except ValueError:
            pass
        self.count += 1
        self.valid += valid
        self.updates += update_match
        self.states += state_match
        self.false_changes += operations == {'no_change'} and prediction.strip() != 'NO_CHANGE'
        for operation in operations:
            counts = self.by_operation.setdefault(operation, {'count': 0, 'valid': 0, 'update_matches': 0, 'state_matches': 0})
            counts['count'] += 1
            counts['valid'] += valid
            counts['update_matches'] += update_match
            counts['state_matches'] += state_match

    def report(self):
        return {'extraction': {
            'count': self.count, 'valid_predictions': self.valid, 'invalid_predictions': self.count - self.valid,
            'update_matches': self.updates, 'state_matches': self.states,
            'valid_rate': self.valid / self.count if self.count else None,
            'update_match_rate': self.updates / self.count if self.count else None,
            'state_match_rate': self.states / self.count if self.count else None,
            'false_changes_on_no_change': self.false_changes, 'by_operation': self.by_operation,
            'note': 'Single-turn replay from supplied prior state; source-mode matching is separate from active-fact matching. '
                    'This is not a rolling-dialogue or independent semantic-grounding evaluation.',
        }}
