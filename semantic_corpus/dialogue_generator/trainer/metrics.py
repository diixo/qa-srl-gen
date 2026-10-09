"""Strict answer matching plus a small, explicit diagnostic answer taxonomy."""


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
