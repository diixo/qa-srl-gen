"""Explicit, finite lexical substitutions reviewed together with their meanings.

Source attestations remain immutable. These rules license synthetic variants,
not arbitrary slot fillers or new dependency-derived factual annotations.
"""
from copy import deepcopy
from difflib import SequenceMatcher
from functools import lru_cache
import json
from pathlib import Path
import re

FIELDS = ('assistant', 'reply', 'value', 'question')


def substitute(text, edits):
    # One simultaneous pass: a replacement is never processed as another input.
    if not edits:
        return text
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(k) for k in sorted(edits, key=len, reverse=True)) + r')(?!\w)')
    return pattern.sub(lambda match: edits[match[0]], text)


def validate_rules(rules, patterns):
    from .template_bank import fingerprint
    indexed = {p['id']: p for p in patterns}
    seen = set()
    adapters, templates = set(), set()
    for rule in rules:
        if (rule.get('version') != 'reviewed-substitutions-v1' or not rule.get('reason')
                or rule.get('id') in seen or not rule.get('id')
                or not rule.get('adapters') or not rule.get('templates')
                or not isinstance(rule.get('choices'), list) or not rule['choices']):
            raise ValueError('Invalid reviewed substitution rule')
        if (adapters.intersection(rule['adapters']) or templates.intersection(rule['templates'])):
            raise ValueError('Overlapping reviewed substitution rules')
        adapters.update(rule['adapters'])
        templates.update(rule['templates'])
        seen.add(rule['id'])
        for choice in rule['choices']:
            if (not isinstance(choice, dict) or not choice or
                    any(not isinstance(k, str) or not k.strip() or not isinstance(v, str)
                        or not v.strip() or k == v or any(c in k+v for c in '{}\r\n')
                        for k, v in choice.items())):
                raise ValueError('Invalid reviewed substitution choice')
        if len({fingerprint(c) for c in rule['choices']}) != len(rule['choices']):
            raise ValueError('Duplicate reviewed substitution choice')
        for key, digest in rule['adapters'].items():
            p = indexed.get(key)
            if p is None or fingerprint(p) != digest or p['kind'] != 'interpretation':
                raise ValueError('Stale reviewed substitution adapter: ' + key)
            if p['source']['split'] != rule['split'] or p['source']['dialogue'] != rule['dialogue']:
                raise ValueError('Substitution rule crosses source dialogues')
        for key, digest in rule['templates'].items():
            if not all(isinstance(x, str) and re.fullmatch('[0-9a-f]{64}', x) for x in (key, digest)):
                raise ValueError('Invalid substitution template identity')
        for key, revision in rule.get('adapter_revisions', {}).items():
            if (key not in rule['adapters'] or not revision.get('reason')
                    or revision.get('source_sha256') != rule['adapters'][key]
                    or not revision.get('fields')):
                raise ValueError('Invalid reviewed adapter revision')
            for field, edit in revision['fields'].items():
                if (field not in FIELDS or set(edit) != {'before', 'after'}
                        or indexed[key][field] != edit['before']
                        or not isinstance(edit['after'], str) or not edit['after'].strip()):
                    raise ValueError('Stale reviewed adapter revision')
        for key, licenses in rule.get('adapter_choices', {}).items():
            if key not in rule['adapters'] or not isinstance(licenses, list):
                raise ValueError('Invalid reviewed adapter choices')
            indices = set()
            for item in licenses:
                index = item.get('choice')
                if (type(index) is not int or not 0 <= index < len(rule['choices'])
                        or index in indices or not item.get('reason')
                        or item.get('choice_sha256') != fingerprint(rule['choices'][index])
                        or not isinstance(item.get('fields'), dict) or not item['fields']
                        or any(f not in FIELDS or not isinstance(v, str) or not v.strip()
                               for f, v in item['fields'].items())):
                    raise ValueError('Invalid reviewed adapter choice')
                indices.add(index)
        for key, licenses in rule.get('phrase_choices', {}).items():
            if key not in rule['templates'] or not isinstance(licenses, list):
                raise ValueError('Invalid reviewed phrase template')
            seen_choices = set()
            for item in licenses:
                index = item.get('choice')
                if (type(index) is not int or not 0 <= index < len(rule['choices'])
                        or index in seen_choices or not item.get('reason')
                        or item.get('choice_sha256') != fingerprint(rule['choices'][index])
                        or not isinstance(item.get('before'), str) or not item['before'].strip()
                        or not isinstance(item.get('after'), str) or not item['after'].strip()
                        or item['before'] == item['after']
                        or substitute(item['before'], rule['choices'][index]) != item['after']):
                    raise ValueError('Invalid reviewed phrase choice')
                seen_choices.add(index)
        literal_choices = rule.get('literal_choices', {})
        if not isinstance(literal_choices, dict):
            raise ValueError('Invalid reviewed literal templates')
        for key, choices in literal_choices.items():
            if key not in rule['templates'] or not isinstance(choices, list):
                raise ValueError('Invalid reviewed literal template')
            indices = set()
            for item in choices:
                index = item.get('choice')
                if (type(index) is not int or not 0 <= index < len(rule['choices'])
                        or index in indices or not item.get('reason')
                        or item.get('choice_sha256') != fingerprint(rule['choices'][index])
                        or not item.get('parts')):
                    raise ValueError('Invalid reviewed literal choice')
                indices.add(index)
                parts = set()
                for edit in item['parts']:
                    if (type(edit.get('part')) is not int or edit['part'] < 0
                            or edit['part'] in parts
                            or not isinstance(edit.get('before'), str)
                            or not isinstance(edit.get('after'), str)
                            or edit['before'] == edit['after']):
                        raise ValueError('Invalid reviewed literal part')
                    parts.add(edit['part'])
        slot_choices = rule.get('slot_choices', {})
        if not isinstance(slot_choices, dict):
            raise ValueError('Invalid reviewed slot templates')
        for key, choices in slot_choices.items():
            if key not in rule['templates'] or not isinstance(choices, list):
                raise ValueError('Invalid reviewed slot template')
            indices = set()
            for item in choices:
                index = item.get('choice')
                if (type(index) is not int or not 0 <= index < len(rule['choices'])
                        or index in indices or not item.get('reason')
                        or item.get('choice_sha256') != fingerprint(rule['choices'][index])
                        or not isinstance(item.get('bindings'), dict) or not item['bindings']):
                    raise ValueError('Invalid reviewed slot choice')
                indices.add(index)
                for slot, edit in item['bindings'].items():
                    if (not isinstance(slot, str) or not slot
                            or not isinstance(edit, dict) or set(edit) != {'before', 'after'}
                            or any(not isinstance(edit[f], str) or not edit[f].strip()
                                   or any(c in edit[f] for c in '{}\r\n') for f in ('before', 'after'))
                            or edit['before'] == edit['after']):
                        raise ValueError('Invalid reviewed slot binding')
        verbs = rule.get('verb_choices', {})
        if not isinstance(verbs, dict) or set(verbs) - set(rule['templates']):
            raise ValueError('Invalid reviewed verb templates')
        for licenses in verbs.values():
            indices = set()
            if not isinstance(licenses, list):
                raise ValueError('Invalid reviewed verb choices')
            for item in licenses:
                index = item.get('choice')
                if (type(index) is not int or not 0 <= index < len(rule['choices'])
                        or index in indices or not item.get('reason')
                        or item.get('choice_sha256') != fingerprint(rule['choices'][index])
                        or not isinstance(item.get('bindings'), dict) or not item['bindings']):
                    raise ValueError('Invalid reviewed verb choice')
                indices.add(index)
                for key, edit in item['bindings'].items():
                    if (not isinstance(key, str) or not key or not isinstance(edit, dict)
                            or set(edit) != {'before', 'after', 'lemma', 'tag', 'slot_sha256'}
                            or any(not isinstance(edit[f], str) or not re.fullmatch('[A-Za-z]+', edit[f])
                                   for f in ('before', 'after', 'lemma'))
                            or edit['before'] == edit['after']
                            or edit['tag'] not in {'VB', 'VBP', 'VBZ', 'VBD', 'VBG', 'VBN'}
                            or not isinstance(edit['slot_sha256'], str)
                            or not re.fullmatch('[0-9a-f]{64}', edit['slot_sha256'])):
                        raise ValueError('Invalid reviewed verb binding')
        # Early admitted sentences have a reviewed interpretation but predate
        # explicit adapter pins. Bind them without rewriting source evidence.
        links = rule.get('template_adapters', {})
        if not isinstance(links, dict) or set(links) - set(rule['templates']):
            raise ValueError('Invalid substitution template adapter links')
        for key, ref in links.items():
            if (not isinstance(ref, dict) or set(ref) != {'id', 'sha256', 'interpretation'} or
                    ref['id'] not in indexed or
                    rule['adapters'].get(ref['id']) != ref['sha256'] or
                    indexed[ref['id']].get('review', {}).get('template_sha256') != rule['templates'][key] or
                    indexed[ref['id']].get('value') != ref['interpretation']):
                raise ValueError('Unreviewed substitution template adapter link')


@lru_cache(maxsize=4)
def _read(path, stamp, size):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rules = data.get('substitutions', [])
    validate_rules(rules, data['patterns'])
    return {key: rule for rule in rules for key in rule['templates']}


def template_rules(root):
    path = Path(root) / 'reviewed/dialogue-fragments.json'
    if not path.is_file():
        return {}
    stat = path.stat()
    return _read(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def template_adapter_revisions(root):
    """Export corrected meanings alongside immutable attested template records."""
    rules = {r['id']: r for r in template_rules(root).values()}
    return {key: revision for r in rules.values()
            for key, revision in r.get('adapter_revisions', {}).items()}


def revise_pattern(pattern, rule):
    """Apply exact reviewed corrections without rewriting archived evidence."""
    from .template_bank import fingerprint
    revision = rule.get('adapter_revisions', {}).get(pattern['id'])
    if not revision:
        return pattern
    if fingerprint(pattern) != revision['source_sha256']:
        raise ValueError('Stale reviewed adapter revision')
    result = deepcopy(pattern)
    for field, edit in revision['fields'].items():
        if field not in FIELDS or result[field] != edit['before']:
            raise ValueError('Stale reviewed adapter revision')
        result[field] = edit['after']
    return result


def expand_patterns(patterns, rules):
    """Keep original catalog rows/pins and add opt-in synthetic pattern families."""
    validate_rules(rules, patterns)
    indexed = {p['id']: p for p in patterns}
    expanded = list(patterns)
    for rule in rules:
        for key in rule['adapters']:
            base = indexed[key]
            effective = revise_pattern(base, rule)
            if (not rule.get('adapter_choices', {}).get(key) and
                    not any(any(substitute(effective[f], c) != effective[f] for f in FIELDS) for c in rule['choices'])):
                continue
            row = deepcopy(base)
            row['id'] = key + '_substitution'
            if row['id'] in indexed:
                raise ValueError('Synthetic pattern ID conflicts with catalog')
            row['substitution_rule'] = deepcopy(rule)
            row['substitution_base_id'] = key
            expanded.append(row)
        for key in rule.get('adapter_revisions', {}):
            index = next(i for i, p in enumerate(patterns) if p['id'] == key)
            expanded[index] = revise_pattern(patterns[index], rule)
    return expanded


def realize_pattern(pattern, variant):
    from .template_bank import fingerprint
    if 'substitution_rule' not in pattern:
        return pattern
    rule = pattern['substitution_rule']
    base = {k: v for k, v in pattern.items() if k not in {'substitution_rule', 'substitution_base_id'}}
    base['id'] = pattern['substitution_base_id']
    if fingerprint(base) != rule['adapters'].get(base['id']):
        raise ValueError('Stale synthetic pattern base')
    index = variant % len(rule['choices'])
    result = deepcopy(pattern)
    effective = revise_pattern(base, rule)
    for field in FIELDS:
        result[field] = substitute(effective[field], rule['choices'][index])
    for item in rule.get('adapter_choices', {}).get(base['id'], []):
        if item['choice'] == index:
            if item['choice_sha256'] != fingerprint(rule['choices'][index]):
                raise ValueError('Stale reviewed adapter choice')
            result.update(item['fields'])
    result['realized_substitution'] = {'rule': rule['id'], 'choice': index}
    return result


def render_template(template, rule, choice):
    from .template_bank import CorpusTemplate, fingerprint
    from .template_usage import validate_usage
    record = template.record
    if fingerprint(record) != rule['templates'].get(template.template_id):
        raise ValueError('Stale reviewed substitution template')
    policy = record['usage_constraints']
    if policy['mode'] != 'attested_sentence':
        raise ValueError('Substitution requires an admitted sentence')
    ref = policy['meaning'].get('extraction_adapter') or rule.get('template_adapters', {}).get(template.template_id)
    if not ref:
        raise ValueError('Substitution requires a reviewed template adapter link')
    if rule['adapters'].get(ref['id']) != ref['sha256']:
        raise ValueError('Substitution does not match template interpretation')
    edits = rule['choices'][choice]
    values = dict(record['defaults'])
    original = template.render()
    text = original['text']
    target = substitute(text, edits)
    raw = deepcopy(record)
    raw.pop('usage_constraints')
    phrase = next((x for x in rule.get('phrase_choices', {}).get(template.template_id, [])
                   if x['choice'] == choice), None)
    if phrase:
        if (phrase['choice_sha256'] != fingerprint(edits) or phrase['before'] != text
                or phrase['after'] != target or not phrase.get('reason') or target == text):
            raise ValueError('Stale reviewed phrase choice')
        # A whole-phrase paraphrase licenses exactly one surface, not a newly
        # inferred dependency tree or free slots. The reviewed interpretation
        # remains the only semantic adapter for this context-only realization.
        raw.update(parts=[{'literal': target}], slots={}, defaults={},
                   verbs=[], constructions=[], predicate_groups=[], attributes=[])
        values = {}
    # Individually pinned finite edits can cover numbers/vocatives retained by
    # the compiler as literals. They do not turn fixed text into free slots.
    for license in ([] if phrase else rule.get('literal_choices', {}).get(template.template_id, [])):
        if license['choice'] != choice:
            continue
        if license['choice_sha256'] != fingerprint(edits):
            raise ValueError('Stale reviewed literal choice')
        for edit in license['parts']:
            index = edit['part']
            if (index >= len(raw['parts']) or
                    raw['parts'][index].get('literal') != edit['before']):
                raise ValueError('Stale reviewed literal part')
            operations = SequenceMatcher(None, edit['before'], edit['after'], autojunk=False).get_opcodes()
            for group in raw['predicate_groups']:
                for construction in group['constructions']:
                    for marker in construction['markers']:
                        if marker.get('part') != index:
                            continue
                        start, end = marker['start'], marker['end']
                        # A particle/preposition must survive wholly inside one
                        # unchanged block, even when a preceding number grows.
                        block = next((op for op in operations if op[0] == 'equal'
                                      and op[1] <= start <= end <= op[2]), None)
                        if block is None:
                            return None
                        marker['start'] = block[3] + start - block[1]
                        marker['end'] = block[3] + end - block[1]
            raw['parts'][index]['literal'] = edit['after']
    # An explicitly reviewed alignment can disambiguate a diff spanning several
    # adjacent noun/adjective slots. It cannot change syntax, verbs or the target.
    for license in ([] if phrase else rule.get('slot_choices', {}).get(template.template_id, [])):
        if license['choice'] != choice:
            continue
        if license['choice_sha256'] != fingerprint(edits):
            raise ValueError('Stale reviewed slot choice')
        for key, edit in license['bindings'].items():
            if (template.slots.get(key, {}).get('kind') not in {'nominal', 'adjective'}
                    or values.get(key) != edit['before']
                    or not isinstance(edit['after'], str) or not edit['after'].strip()):
                raise ValueError('Invalid or stale reviewed slot binding')
            values[key] = edit['after']
    # Lexical verbs need an additional model-reviewed surface/form license.
    # Keep auxiliaries, tense/aspect, negation, dependencies and particles fixed.
    for license in ([] if phrase else rule.get('verb_choices', {}).get(template.template_id, [])):
        if license['choice'] != choice:
            continue
        if license['choice_sha256'] != fingerprint(edits):
            raise ValueError('Stale reviewed verb choice')
        for key, edit in license['bindings'].items():
            slot = raw['slots'].get(key, {})
            if (slot.get('kind') != 'verb' or slot.get('pos') != 'VERB'
                    or slot.get('dependency') in {'aux', 'auxpass', 'cop'}
                    or slot.get('tag') != edit['tag']
                    or fingerprint(slot) != edit['slot_sha256']
                    or values.get(key) != edit['before']
                    or not all(isinstance(edit[f], str) and re.fullmatch('[A-Za-z]+', edit[f])
                               for f in ('after', 'lemma'))):
                raise ValueError('Invalid or stale reviewed verb binding')
            groups = [g for g in raw['predicate_groups'] if g['head_slot'] == key]
            if (any(g['constructions'] for g in groups)
                    or any(m.get('slot') == key for g in raw['predicate_groups']
                           for c in g['constructions'] for m in c['markers'])):
                return None  # Phrasal/prepositional constructions need their own review.
            values[key] = edit['after']
            slot['lemma'] = edit['lemma']
            for group in groups:
                group['lemma'] = edit['lemma']
    if rule.get('verb_choices', {}).get(template.template_id):
        raw['verbs'] = sorted({s['lemma'] for s in raw['slots'].values() if s['kind'] == 'verb'})
    surface = CorpusTemplate(raw).render(values)
    changes = {}
    for tag, a, b, c, d in SequenceMatcher(None, surface['text'], target, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        owners = [(key, span) for key, span in surface['spans'].items()
                  if span[0] <= a <= b <= span[1]
                  and template.slots[key]['kind'] in {'nominal', 'adjective'}]
        if len(owners) != 1:
            return None  # A change to fixed syntax or a verb is not licensed.
        key, span = owners[0]
        changes.setdefault(key, []).append((a-span[0], b-span[0], target[c:d]))
    for key, pieces in changes.items():
        for a, b, replacement in reversed(pieces):
            values[key] = values[key][:a] + replacement + values[key][b:]
    if values == record['defaults'] and surface['text'] == text:
        return None
    result = CorpusTemplate(raw).render(values)
    if result['text'] != target:
        return None  # Never change only half of a repeated/literal reference.
    updated = deepcopy(policy)
    context = policy['context']
    start, end = policy['span']
    prefix = substitute(context[policy['turn']-1][:start], edits)
    updated.update(mode='context_only', text=result['text'],
                   context=[substitute(t, edits) for t in context],
                   span=[len(prefix), len(prefix)+len(result['text'])],
                   reasons=['Synthetic lexical variant licensed by ' + rule['id'], rule['reason']])
    if phrase:
        # Structural and expression annotations describe the old surface only.
        for field in ('fixed_expressions', 'property_bindings', 'alternative_properties'):
            updated.pop(field, None)
    for expression in updated.get('fixed_expressions', []):
        a, b = expression['span']
        if substitute(text[a:b], edits) != text[a:b]:
            return None
        new_start = len(substitute(text[:a], edits))
        expression['span'] = [new_start, new_start + b - a]
    for field in ('interpretation', 'enclosing_turn_text'):
        if field in policy['meaning']:
            updated['meaning'][field] = substitute(policy['meaning'][field], edits)
    if 'interpretation' in ref:
        updated['meaning']['interpretation'] = substitute(ref['interpretation'], edits)
    revision = rule.get('adapter_revisions', {}).get(ref['id'], {})
    if 'value' in revision.get('fields', {}):
        updated['meaning']['interpretation'] = substitute(revision['fields']['value']['after'], edits)
    for item in rule.get('adapter_choices', {}).get(ref['id'], []):
        if item['choice'] == choice and 'value' in item['fields']:
            if item['choice_sha256'] != fingerprint(edits):
                raise ValueError('Stale reviewed adapter choice')
            updated['meaning']['interpretation'] = item['fields']['value']
    updated['meaning']['extraction_adapter'] = {k: ref[k] for k in ('id', 'sha256')}
    proof = dict(rule=rule['id'], rule_sha256=fingerprint(rule), choice=choice,
                 source_template_sha256=fingerprint(record),
                 adapter_id=ref['id'], source_text=policy['text'])
    updated['meaning']['reviewed_substitution'] = proof
    validate_usage(updated, result['text'], result['slots'])
    result.update(usage_constraints=updated, reviewed_substitution=proof)
    return result
