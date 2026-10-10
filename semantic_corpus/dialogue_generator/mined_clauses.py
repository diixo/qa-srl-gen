"""Split only independent, explicitly subject-bearing finite clauses.

Shared-subject coordination, disjunction, conditions, quoted speech and cross-
clause control are deliberately not flattened into asserted events.
"""
import re


def independent_clauses(record):
    slots = record['slots']
    roots = {k for k, s in slots.items() if s.get('kind') == 'verb' and s.get('dependency') in ('ROOT', 'conj')}
    if len(roots) < 2:
        return None, None
    # Diagnostics must choose the same first failure across Python hash seeds.
    for key in sorted(roots):
        slot = slots[key]
        if slot.get('dependency') == 'conj' and slot.get('head_slot') not in roots:
            return None, 'coordinated_embedded_scope'
        subjects = [k for k, s in slots.items() if s.get('head_slot') == key
                    and s.get('dependency') in ('nsubj', 'nsubjpass', 'nsubj:pass')]
        if len(subjects) != 1:
            return None, 'shared_or_unresolved_coordination_subject'
    segments = [[]]
    boundary = re.compile(r'(?:,?\s+and\s+|,?\s+but\s+|[.;]\s+)')
    for index, original in enumerate(record['parts']):
        part = {**original, '_source_part': index}
        if 'slot' in part:
            segments[-1].append(part)
            continue
        literal = part['literal']
        start = 0
        for match in boundary.finditer(literal):
            if match.start() > start:
                segments[-1].append({'literal': literal[start:match.start()], '_source_part': index})
            segments.append([])
            start = match.end()
        if start < len(literal):
            segments[-1].append({'literal': literal[start:], '_source_part': index})
    if len(segments) != len(roots) or len(segments) > 4:
        return None, 'unresolved_clause_boundary'
    clauses = []
    for parts in segments:
        keys = {p['slot'] for p in parts if 'slot' in p}
        heads = keys & roots
        if len(heads) != 1:
            return None, 'unresolved_clause_boundary'
        root, = heads
        if any(s.get('head_slot') in keys and k not in keys for k, s in slots.items() if k not in roots):
            return None, 'cross_clause_dependency'
        if any(slots[k].get('head_slot') not in keys | {None} for k in keys - {root}):
            return None, 'cross_clause_dependency'
        subset = {k: dict(slot) for k, slot in slots.items() if k in keys}
        subset[root].update(dependency='ROOT', head_slot=root)
        clauses.append({**record, 'parts': parts, 'slots': subset,
                        'predicate_groups': [g for g in record['predicate_groups'] if g['head_slot'] in keys]})
    return clauses, None
