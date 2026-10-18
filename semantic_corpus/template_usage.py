"""Occurrence-local restrictions for reviewed, context-dependent sentences.

An attested sentence can be admitted as a surface pattern without licensing
new slot combinations or dependency-derived factual QA. Keep the actual
context and the limited interpretation together, even without the source file.
"""
from copy import deepcopy


VERSION = 'contextual-attestation-v1'


def validate_usage(usage, text, slots):
    if usage.get('version') != VERSION or usage.get('mode') not in {
            'attested_sentence', 'context_only', 'pending'}:
        raise ValueError('invalid template usage policy')
    if usage.get('text') != text or not usage.get('source_id'):
        raise ValueError('usage policy does not describe this attestation')
    context, turn, span = usage.get('context'), usage.get('turn'), usage.get('span')
    if (not isinstance(context, list) or not context or
            any(not isinstance(t, str) for t in context) or type(turn) is not int or
            not 1 <= turn <= len(context) or not isinstance(span, list) or len(span) != 2 or
            any(type(n) is not int for n in span) or
            not 0 <= span[0] < span[1] <= len(context[turn-1]) or
            context[turn-1][span[0]:span[1]] != text):
        raise ValueError('usage policy has no matching embedded context')
    if not isinstance(usage.get('meaning'), dict) or not usage['meaning']:
        raise ValueError('usage policy requires an explicit contextual interpretation')
    if not isinstance(usage.get('reasons'), list) or not usage['reasons']:
        raise ValueError('usage policy requires restrictions with reasons')
    for expression in usage.get('fixed_expressions', []):
        start, end = expression['span']
        if (not 0 <= start < end <= len(text) or text[start:end] != expression['text']
                or not expression.get('meaning')):
            raise ValueError('invalid fixed expression')
    for alternative in usage.get('alternative_properties', []):
        names, target = alternative['slots'], alternative['target_slot']
        if (alternative.get('operator') != 'or' or alternative.get('scope') != 'question'
                or len(set(names)) != len(names) or len(names) < 2 or
                target not in slots or slots[target].get('kind') != 'nominal' or
                any(n not in slots or slots[n].get('kind') != 'adjective' for n in names)):
            raise ValueError('invalid questioned property alternatives')
    seen = set()
    for binding in usage.get('property_bindings', []):
        if set(binding) != {'slot', 'target_slot', 'target_kind', 'scope'}:
            raise ValueError('invalid reviewed property binding fields')
        name, target, kind = binding['slot'], binding['target_slot'], binding['target_kind']
        if (name in seen or name not in slots or slots[name].get('kind') != 'adjective'
                or not isinstance(binding['scope'], str) or not binding['scope'].strip()):
            raise ValueError('invalid reviewed property binding')
        seen.add(name)
        expected = {'nominal': 'nominal', 'free_relative': 'nominal', 'clause': 'verb'}
        if kind == 'implicit_participant':
            if target is not None:
                raise ValueError('an implicit property participant has no observed slot')
        elif (kind not in expected or target not in slots
                or slots[target].get('kind') != expected[kind]):
            raise ValueError('reviewed property target kind does not match its slot')
    if seen & {name for a in usage.get('alternative_properties', []) for name in a['slots']}:
        raise ValueError('conflicting property decisions')


def constrain_record(record, body, context, review):
    """Apply only explicitly reviewed alternatives; never infer them from text."""
    from .template_bank import CorpusTemplate, fingerprint
    policy = deepcopy(review['usage'])
    if set(policy) - {'mode', 'reasons', 'fixed_expressions', 'alternative_properties', 'property_bindings'}:
        raise ValueError('unsupported usage policy field')
    if policy.get('mode') == 'attested_sentence' and review['unit'] != 'sentence':
        raise ValueError('only a reviewed full sentence may be admitted')
    origin = body['provenance']['origin']
    policy.update(version=VERSION, source_id=body['idx'], text=body['example'],
                  context=deepcopy(context), turn=origin['turn'], span=origin['span'],
                  meaning=deepcopy(review['meaning']))
    validate_usage(policy, body['example'], record['slots'])
    for binding in policy.get('property_bindings', []):
        attributes = [a for a in record['attributes'] if a['slot'] == binding['slot']]
        if len(attributes) != 1:
            raise ValueError('reviewed property has no unique attribute')
        attributes[0].update(binding)
    for alternative in policy.get('alternative_properties', []):
        for name in alternative['slots']:
            attributes = [a for a in record['attributes'] if a['slot'] == name]
            if len(attributes) != 1:
                raise ValueError('alternative property has no unique attribute')
            attribute = attributes[0]
            attribute['target_slot'] = alternative['target_slot']
            attribute['scope'] = 'question_alternative'
    record['usage_constraints'] = policy
    shape = ('parts', 'slots', 'verbs', 'constructions', 'predicate_groups', 'attributes', 'usage_constraints')
    record['template_id'] = fingerprint({k: record[k] for k in shape})
    return CorpusTemplate(record).render()
