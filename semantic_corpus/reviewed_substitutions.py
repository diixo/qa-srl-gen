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


def expand_patterns(patterns, rules):
    """Keep original catalog rows/pins and add opt-in synthetic pattern families."""
    validate_rules(rules, patterns)
    indexed = {p['id']: p for p in patterns}
    expanded = list(patterns)
    for rule in rules:
        for key in rule['adapters']:
            base = indexed[key]
            if not any(any(substitute(base[f], c) != base[f] for f in FIELDS) for c in rule['choices']):
                continue
            row = deepcopy(base)
            row['id'] = key + '_substitution'
            if row['id'] in indexed:
                raise ValueError('Synthetic pattern ID conflicts with catalog')
            row['substitution_rule'] = deepcopy(rule)
            row['substitution_base_id'] = key
            expanded.append(row)
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
    for field in FIELDS:
        result[field] = substitute(pattern[field], rule['choices'][index])
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
    changes = {}
    for tag, a, b, c, d in SequenceMatcher(None, text, target, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        owners = [(key, span) for key, span in original['spans'].items()
                  if span[0] <= a <= b <= span[1]
                  and template.slots[key]['kind'] in {'nominal', 'adjective'}]
        if len(owners) != 1:
            return None  # A change to fixed syntax or a verb is not licensed.
        key, span = owners[0]
        changes.setdefault(key, []).append((a-span[0], b-span[0], target[c:d]))
    for key, pieces in changes.items():
        for a, b, replacement in reversed(pieces):
            values[key] = values[key][:a] + replacement + values[key][b:]
    if values == record['defaults']:
        return None
    raw = deepcopy(record)
    raw.pop('usage_constraints')
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
    updated['meaning']['extraction_adapter'] = {k: ref[k] for k in ('id', 'sha256')}
    proof = dict(rule=rule['id'], rule_sha256=fingerprint(rule), choice=choice,
                 source_template_sha256=fingerprint(record),
                 adapter_id=ref['id'], source_text=policy['text'])
    updated['meaning']['reviewed_substitution'] = proof
    validate_usage(updated, result['text'], result['slots'])
    result.update(usage_constraints=updated, reviewed_substitution=proof)
    return result
