"""Whole-document training targets derived from saved structure, without parsing.

Keep semantic annotations and mined syntax distinct. A dependency tree, a
successful round trip, or a contextual interpretation is not a complete gold
semantic graph. Pending records retain everything needed for further review.
"""

from copy import deepcopy
import json

from .documents import AnnotationRun, Document, Utterance
from .storage.schema import decode_run, decode_span, encode_run_rows
from .template_usage import validate_usage


VERSION = 'sentence-analysis-v1'
PROMPT_PREFIX = 'Analyze the complete text:\n'
SYSTEM = (
    'Return a sentence-analysis-v1 JSON object describing the entire input. '
    'Include every known mention, predicate, property, relation and dialogue act. '
    'Use local IDs and exact half-open character spans into text. Preserve tense, '
    'aspect, voice, polarity, modality, omitted roles and uncertainty. '
    'Keep syntax and contextual interpretations separate from asserted facts. '
    'An empty field is not proof of absence. Do not invent missing information.'
)
STRUCTURE_KEYS = ('slots', 'bindings', 'spans', 'parts', 'part_spans', 'verbs',
                  'predicate_groups', 'attributes')


def _dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _check_surface(text, structure):
    """Check the current rendering, including changed-length substitutions."""
    slots = structure.get('slots', {})
    bindings, spans = structure.get('bindings', {}), structure.get('spans', {})
    if set(slots) != set(bindings) or set(slots) != set(spans):
        raise ValueError('Saved slots, bindings and spans disagree')
    for key in slots:
        start, end = spans[key]
        if (type(start) is not int or type(end) is not int or
                not 0 <= start < end <= len(text) or text[start:end] != bindings[key]):
            raise ValueError(f'Stale template binding/span: {key}')
        head = slots[key].get('head_slot')
        if head is not None and head not in slots:
            raise ValueError(f'Unknown dependency head: {head}')
    if 'parts' in structure:
        pieces = [part['literal'] if 'literal' in part else bindings[part['slot']]
                  for part in structure['parts']]
        if ''.join(pieces) != text:
            raise ValueError('Template parts no longer reconstruct the text')
        cursor, expected = 0, []
        for piece in pieces:
            expected.append([cursor, cursor + len(piece)])
            cursor += len(piece)
        if structure.get('part_spans') != expected:
            raise ValueError('Stale template part spans')
    for group in structure.get('predicate_groups', []):
        head = group['head_slot']
        if head not in slots or group.get('head_span') != spans[head]:
            raise ValueError('Stale predicate group head')
        for construction in group.get('constructions', []):
            for marker in construction.get('markers', []):
                start, end = marker['span']
                if not 0 <= start < end <= len(text) or text[start:end] != marker['text']:
                    raise ValueError('Stale construction marker')
    for attribute in structure.get('attributes', []):
        if (attribute['span'] != spans.get(attribute['slot']) or
                attribute.get('target_span') != spans.get(attribute.get('target_slot'))):
            raise ValueError('Stale property binding')


def analysis_record(document, run=None):
    """One complete saved analysis, with a separately computed admission gate.

    Corpus IDs, source snapshots and review evidence are outside the target.
    Annotation fields use the canonical codec, so new supported fields cannot
    silently disappear in a parallel handwritten export schema.
    """
    if run is not None:
        errors = run.validate_against(document)
        if errors:
            raise ValueError('; '.join(errors))
    rows = encode_run_rows(run or AnnotationRun('empty'))
    ids = {node['id']: f'n{i}' for i, node in enumerate(rows['spans'])}
    utterance_ids = {u.utterance_id: f'u{i}' for i, u in enumerate(document.utterances)}

    def clean(row):
        result = {k: deepcopy(v) for k, v in row.items()
                  if k not in {'run_id', 'document_id', 'source', 'confidence'}}
        for field in ('id', 'source_id', 'target_id'):
            if result.get(field) is not None:
                result[field] = ids[result[field]]
        if 'utterance_id' in result:
            result['utterance_id'] = utterance_ids[result['utterance_id']]
        return result

    syntax = deepcopy({key: document.metadata[key] for key in STRUCTURE_KEYS if key in document.metadata})
    if syntax:
        _check_surface(document.text, syntax)
    policy = deepcopy(document.metadata.get('usage_constraints'))
    if policy is not None:
        validate_usage(policy, document.text, syntax.get('slots', {}))
    context = ({'turns': policy['context'], 'turn': policy['turn'], 'span': policy['span']}
               if policy else None)
    utterances = [{'id': utterance_ids[u.utterance_id], 'speaker': u.speaker,
                   'turn_index': u.turn_index, 'span': u.span.to_json()} for u in document.utterances]
    annotations = [*rows['spans'], *rows['dialogue']]
    verified = bool(annotations) and all(a.get('review_status') == 'VERIFIED' for a in annotations)
    # Only the generator knows that its whole situation was realized. Checking
    # offsets or seeing VERIFIED on a few spans does not establish completeness.
    constructed = bool(run and run.synthetic and run.model_name == 'semantic_generator')
    gaps = []
    if not verified:
        gaps.append('annotations_not_fully_verified')
    if syntax:
        gaps.append('saved_syntax_requires_semantic_review')
    if not constructed:
        gaps.append('whole_text_semantics_not_confirmed')
    if policy:
        gaps.append('contextual_meaning_has_limited_usage')
    coverage = {'saved_annotations_preserved': True,
                'semantic_completeness': 'by_construction' if constructed and not syntax and not policy else 'unassessed',
                'gaps': gaps}
    if 'template_semantics' in document.metadata:
        coverage['annotation_coverage'] = deepcopy(document.metadata['template_semantics'])
    target = {'version': VERSION, 'nodes': [clean(n) for n in rows['spans']],
              'relations': [clean(e) for e in rows['relations']],
              'dialogue': [clean(a) for a in rows['dialogue']],
              'structure': syntax,
              'interpretation': ({k: deepcopy(policy[k]) for k in (
                  'mode', 'meaning', 'reasons', 'fixed_expressions', 'property_bindings', 'alternative_properties')
                  if k in policy} if policy else None),
              'coverage': coverage}
    # Interpretation adapter IDs/revisions remain evidence, not labels to learn.
    if target['interpretation']:
        meaning = target['interpretation']['meaning']
        for key in ('extraction_adapter', 'adapter_revision', 'reviewed_substitution'):
            meaning.pop(key, None)
    prompt = {'text': document.text, 'utterances': utterances, 'context': context}
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': PROMPT_PREFIX + _dumps(prompt)},
                {'role': 'assistant', 'content': _dumps(target)}]
    validate_target(messages[1]['content'], messages[2]['content'])
    return {'messages': messages, 'task': 'sentence_analysis',
            'document_id': document.document_id, 'run_id': run.run_id if run else None,
            'split': document.split, 'scenario_key': document.metadata.get('template_id'),
            'training_eligible': not gaps,
            'provenance': {'model': run.model_name if run else None,
                           'ontology_version': run.ontology_version if run else None,
                           'node_ids': ids, 'utterance_ids': utterance_ids,
                           'usage_constraints': policy}}


def validate_target(prompt, answer, *, require_training=False):
    """Validate structure/evidence, never infer semantic truth from validity."""
    if not isinstance(prompt, str) or not prompt.startswith(PROMPT_PREFIX):
        raise ValueError('Expected a sentence-analysis prompt')
    try:
        source, target = json.loads(prompt[len(PROMPT_PREFIX):]), json.loads(answer)
        if (not isinstance(target, dict) or set(target) != {
                'version', 'nodes', 'relations', 'dialogue', 'structure', 'interpretation', 'coverage'}
                or target['version'] != VERSION):
            raise ValueError('Invalid sentence-analysis target schema')
        if (not isinstance(source, dict) or set(source) != {'text', 'utterances', 'context'}
                or not isinstance(source['utterances'], list)
                or any(not isinstance(target[key], list) for key in ('nodes', 'relations', 'dialogue'))
                or not isinstance(target['structure'], dict)):
            raise ValueError('Invalid sentence-analysis containers')
        coverage = target['coverage']
        if (not isinstance(coverage, dict)
                or type(coverage.get('saved_annotations_preserved')) is not bool
                or coverage.get('semantic_completeness') not in ('by_construction', 'unassessed')
                or not isinstance(coverage.get('gaps'), list)
                or any(not isinstance(reason, str) or not reason for reason in coverage['gaps'])):
            raise ValueError('Invalid sentence-analysis coverage')
        text = source['text']
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Expected nonempty source text')
        doc = Document('target', text, utterances=tuple(
            Utterance(u['id'], 'target', u['turn_index'], u['speaker'], decode_span(u['span']))
            for u in source['utterances']))
        for u in doc.utterances:
            if u.span.end_char > len(text):
                raise ValueError('Utterance outside the text')
        structure = target['structure']
        if structure:
            _check_surface(text, structure)
        if target['interpretation'] is not None:
            context = source['context']
            validate_usage({**target['interpretation'], 'version': 'contextual-attestation-v1',
                            'source_id': 'target', 'text': text, 'context': context['turns'],
                            'turn': context['turn'], 'span': context['span']},
                           text, structure.get('slots', {}))
        run = decode_run({'run_id': 'target'},
                         [{**n, 'document_id': 'target'} for n in target['nodes']],
                         target['relations'], target['dialogue'])
        errors = run.validate_against(doc)
        if errors:
            raise ValueError('; '.join(errors))
        if require_training:
            if (coverage['semantic_completeness'] != 'by_construction' or coverage['gaps'] or
                    structure or target['interpretation'] is not None or not target['nodes'] or
                    any(n.get('review_status') != 'VERIFIED'
                        for n in [*target['nodes'], *target['dialogue']])):
                raise ValueError('Sentence analysis is pending review, not a training target')
        return target
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f'Malformed sentence analysis: {error}') from error
