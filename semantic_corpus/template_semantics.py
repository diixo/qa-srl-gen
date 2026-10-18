"""Dependency-derived semantic candidates and extractive QA for mined templates.

Works on saved slot dependencies, without running an NLP model again. These
are explicitly UNREVIEWED annotations: syntax cannot settle every entity type,
predicate sense or participant role. Unsupported clauses stay pending and carry
machine-readable reasons instead of guessed questions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import re

from .documents import AnnotationRun, Document, EntityMention, Predicate, PredicatePart, Property, RelationEdge, TextSpan
from .ontology import Label, LabelSet, Polarity, Relation, ReviewStatus, load_default_hierarchy
from .qasrl_core.frame import ArgStructure, Frame
from .question_generator.answers import QAExample, QAKind, phrase_answer
from .question_generator.templates import ParadigmResolver, native_questions, _without_property
from .agreement import verb_words
from .template_syntax import nominal_members

VERSION = 'mined-dependencies-v2'
SOURCE = 'template_semantics'
_SUBJECT = {'nsubj', 'nsubjpass', 'nsubj:pass'}
_OBJECT = {'dobj', 'obj'}
_MODALS = {'can', 'will', 'might', 'should', 'would'}
_DETERMINER = re.compile(
    r"\b(?:(?:a|an|the|this|that|these|those|my|your|his|her|its|our|their|some|any|each|every|no|all|both|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|[0-9]+(?:,[0-9]+)*)(?:\s+|-))+$", re.I)
_PRONOUNS = {'i', 'me', 'you', 'he', 'him', 'she', 'her', 'we', 'us', 'they', 'them',
             'someone', 'somebody', 'everyone', 'everybody', 'who'}
_TRANSFER = {'give', 'send', 'tell', 'show', 'offer', 'sell', 'lend', 'teach'}
_BENEFACTIVE = {'buy', 'build', 'make', 'cook', 'get', 'find', 'order', 'save'}
_ADJECTIVE_MODIFIERS = re.compile(r"\b(?:(?:not|very|too|quite|so|less|more|most|least|rather|[a-z]+ly)\s+)+$", re.I)
_REFLEXIVES = {'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'}
_NON_PROPERTIES = {'other', 'many', 'few', 'several', 'same', 'such', 'own', 'only', 'former', 'alleged',
                   'possible', 'next', 'last', 'previous', 'following', 'mere'}
_STATES = {'know', 'believe', 'like', 'love', 'hate', 'want', 'need', 'prefer', 'remember', 'understand'}


class Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class SemanticResult:
    document: Document
    run: AnnotationRun | None


class TemplateAnnotator:
    def __init__(self, *, entities, frames=(), paradigms=None, lexicon=None):
        self.entities = entities
        self.frames = {frame.lemma: frame for frame in frames}
        self.resolver = ParadigmResolver(known=paradigms or {}, lexicon=lexicon)
        self.hierarchy = load_default_hierarchy()

    def annotate(self, document, *, seed=0):
        syntax = document.metadata
        slots, spans = syntax.get('slots', {}), syntax.get('spans', {})
        groups = syntax.get('predicate_groups', [])
        mentions, predicates, relations, properties, skipped = {}, [], [], [], []
        if syntax.get('usage_constraints') is not None:
            # The embedded interpretation is specific to an observed context.
            # It is not a license to turn questions, hearsay or idioms into facts.
            coverage = {'version': VERSION, 'predicates': len(groups), 'annotated_predicates': 0,
                        'properties': 0, 'complete': False,
                        'skipped': [{'head_slot': None, 'reason': 'reviewed_context_only'}],
                        'review_status': str(ReviewStatus.UNREVIEWED)}
            return SemanticResult(replace(document, metadata={**syntax, 'template_semantics': coverage}), None)
        for group in groups:
            head = group.get('head_slot')
            try:
                pred, args = self._predicate(document, slots, spans, group)
                edges = self._questions(document, pred, args)
                if not edges:
                    raise Unsupported('no_supported_argument_questions')
            except Unsupported as error:
                skipped.append({'head_slot': head, 'reason': str(error)})
                continue
            predicates.append(pred)
            for _, mention in args:
                mentions[mention.mention_id] = mention
            relations.extend(edges)

        for attribute in syntax.get('attributes', []):
            key, target = attribute.get('slot'), attribute.get('target_slot')
            if (key not in slots or target not in slots or slots[key].get('dependency') != 'amod'
                    or slots[key].get('tag') != 'JJ' or target not in spans):
                continue
            try:
                mention = self._mention(document, slots, spans, target)
            except Unsupported:
                continue
            span = TextSpan(*spans[key])
            if document.slice(span).lower() in _NON_PROPERTIES:
                continue
            # Comparative/scope/negation questions need more than an adjective
            # head; only an unmodified attributive property is supported here.
            before = document.text[:span.start_char]
            if _ADJECTIVE_MODIFIERS.search(before):
                continue
            mentions[mention.mention_id] = mention
            prop = Property(f'{document.document_id}#property-{key}', document.document_id,
                            span, document.slice(span), document.slice(span),
                            target_id=mention.mention_id, source=SOURCE)
            properties.append(prop)
            relations.append(RelationEdge(prop.property_id, Relation.PROPERTY_OF, mention.mention_id, source=SOURCE))

        if not groups:
            skipped.append({'head_slot': None, 'reason': 'no_verbal_predicates'})
        coverage = {'version': VERSION, 'predicates': len(groups), 'annotated_predicates': len(predicates),
                    'properties': len(properties), 'complete': bool(groups) and not skipped,
                    'skipped': skipped, 'review_status': str(ReviewStatus.UNREVIEWED)}
        annotated_document = replace(document, metadata={**syntax, 'template_semantics': coverage})
        run = None
        if predicates or properties:
            run = AnnotationRun(f'{document.document_id}#{VERSION}', model_name=SOURCE,
                                prompt_version=VERSION, random_seed=seed, synthetic=False,
                                mentions=tuple(mentions.values()), predicates=tuple(predicates),
                                properties=tuple(properties), relations=tuple(relations))
            errors = run.validate_against(document)
            if errors:
                raise ValueError(f'Invalid mined annotation for {document.document_id}: {errors}')
        return SemanticResult(annotated_document, run)

    def _mention(self, doc, slots, spans, key):
        if key not in spans or slots[key].get('kind') != 'nominal':
            raise Unsupported('missing_nominal_span')
        members = nominal_members(slots, key)
        if any(slot.get('head_slot') in members and k not in members
               and slot.get('dependency') in {'conj', 'appos', 'relcl', 'acl', 'acl:relcl'}
               for k, slot in slots.items()):
            raise Unsupported('complex_nominal_argument')
        if any(k not in spans for k in members):
            raise Unsupported('missing_nominal_span')
        start = min(spans[k][0] for k in members)
        end = max(spans[k][1] for k in members)
        # Do not bridge another predicate or unrelated slot into an answer.
        if any(k not in members and start <= span[0] < end for k, span in spans.items()):
            raise Unsupported('noncontiguous_nominal_argument')
        # A literal determiner/modifier may precede the noun phrase, but a
        # neighbouring slot is a separate argument ("gave her a book").
        prefix_start = max((span[1] for k, span in spans.items()
                            if k not in members and span[1] <= start), default=0)
        if any(spans[k][0] == start and slots[k].get('dependency') == 'amod' for k in members):
            modifier = _ADJECTIVE_MODIFIERS.search(doc.text[prefix_start:start])
            if modifier:
                start = prefix_start + modifier.start()
        det = _DETERMINER.search(doc.text[prefix_start:start])
        if det:
            start = prefix_start + det.start()
        span = TextSpan(start, end)
        surface, head = doc.slice(span), doc.text[slice(*spans[key])]
        labels = self._labels(surface, head, slots[key])
        morph = slots[key].get('morph', '')
        plural = ('Number=Plur' in morph or slots[key].get('tag') in ('NNS', 'NNPS')
                  or head.lower() in {'i', 'you', 'we', 'they', 'us', 'them'})
        return EntityMention(f'{doc.document_id}#mention-{key}', doc.document_id, span, surface,
                             labels, source=SOURCE, type_grounded=False,
                             number='plural' if plural else 'singular')

    def _labels(self, text, head, slot):
        if slot.get('pos') == 'PRON' and head.lower() in _PRONOUNS:
            return LabelSet.of(Label.PERSON)
        candidates = []
        for surface in (text, head, 'the ' + head.lower(), 'a ' + head.lower(), 'an ' + head.lower()):
            candidates.extend(entity.labels for entity in self.entities.readings(surface))
        candidates.extend(self.hierarchy.candidate_labels(head.lower()))
        types = {labels.entity_labels for labels in candidates if labels.entity_labels}
        if len(types) == 1:
            return LabelSet.of(*next(iter(types)))
        return LabelSet()  # Unknown and ambiguous names do not become PERSON by capitalization.

    def _predicate(self, doc, slots, spans, group):
        key = group.get('head_slot')
        slot = slots.get(key, {})
        if not slot.get('dependency') or key not in spans:
            raise Unsupported('missing_slot_dependencies')
        if slot.get('analysis') == 'imperative_alternative':
            raise Unsupported('ambiguous_imperative')
        if slot['dependency'] not in {'ROOT', 'root', 'conj'}:
            raise Unsupported('embedded_or_nonfinite_clause')
        # A conjunct inherits the scope of its coordination head. In
        # "She says Anna bought a book and Ben sold it", sold is still reported
        # content, not a second independent assertion by the speaker.
        ancestor, visited = key, set()
        while slots[ancestor].get('dependency') == 'conj':
            visited.add(ancestor)
            ancestor = slots[ancestor].get('head_slot')
            if ancestor not in slots or ancestor in visited:
                raise Unsupported('missing_slot_dependencies')
        if slots[ancestor].get('dependency') not in {'ROOT', 'root'}:
            raise Unsupported('embedded_or_nonfinite_clause')
        children = {k: s for k, s in slots.items() if k != key and s.get('head_slot') == key}
        if any(s.get('dependency') in {'ccomp', 'xcomp', 'csubj', 'csubjpass', 'attr', 'acomp', 'oprd', 'expl'}
               for s in children.values()):
            raise Unsupported('clausal_or_copular_complement')
        head = slot.get('lemma', '')
        if not head or head in {'be', 'do'}:
            raise Unsupported('copula_or_light_verb')
        subjects = [k for k, s in children.items() if s.get('dependency') in _SUBJECT]
        objects = [k for k, s in children.items() if s.get('dependency') in _OBJECT]
        recipients = [k for k, s in children.items() if s.get('dependency') == 'dative' and s.get('kind') == 'nominal']
        if len(subjects) != 1 or len(objects) > 1 or len(recipients) > 1:
            raise Unsupported('missing_or_multiple_core_arguments')
        subject = self._mention(doc, slots, spans, subjects[0])
        # Questions about questions, wishes and conditionals require scoped
        # semantics. Dependency heads alone do not justify asserting them.
        left = max((m.end() for m in re.finditer(r'[.!?;\n]', doc.text[:spans[key][0]])), default=0)
        tail = re.search(r'[.!?;\n]', doc.text[spans[key][1]:])
        right = spans[key][1] + tail.end() if tail else len(doc.text)
        clause = doc.text[left:right]
        if '?' in clause or re.search(r'\b(if|unless|wish|whether|as long as|provided that)\b', clause, re.I):
            raise Unsupported('question_or_conditional_scope')
        if re.search(r'\b(no|nobody|nothing|none|neither|nowhere|without)\b', clause, re.I):
            raise Unsupported('negative_quantifier_scope')
        if re.search(r'\b(never|seldom|rarely|hardly|scarcely|barely)\b', clause, re.I):
            raise Unsupported('negative_adverb_scope')
        if re.search(r'\b(perhaps|maybe|probably|possibly|apparently|reportedly|except)\b', clause, re.I):
            raise Unsupported('uncertainty_or_exception_scope')
        passive = (slots[subjects[0]]['dependency'] != 'nsubj'
                   or any(s.get('dependency') in {'auxpass', 'aux:pass'} for s in children.values()))
        auxiliaries = sorted((k for k, s in children.items() if s.get('dependency') in {'aux', 'auxpass', 'aux:pass'}),
                             key=lambda k: spans[k][0])
        modal = next((slots[k].get('lemma') for k in auxiliaries if slots[k].get('tag') == 'MD'), None)
        if modal and modal not in _MODALS:
            raise Unsupported('unsupported_modality')
        finite = slots[auxiliaries[0]] if auxiliaries else slot
        if not modal and finite.get('tag') not in {'VBD', 'VBP', 'VBZ'}:
            raise Unsupported('missing_finite_tense')
        tense = 'past' if finite.get('tag') == 'VBD' else 'present'
        perfect = any(slots[k].get('lemma') == 'have' for k in auxiliaries)
        progressive = (slot.get('tag') == 'VBG' or any(slots[k].get('tag') == 'VBG' for k in auxiliaries))
        aspect = '+'.join(name for name, yes in [('perfect', perfect), ('progressive', progressive)] if yes) or 'simple'
        verb_start = min([spans[key][0], *(spans[k][0] for k in auxiliaries)])
        negated = bool(re.search(r"\b(?:not|never)\b|n['’]t", doc.text[verb_start:spans[key][1]], re.I))
        forms = self.resolver(head)
        # Avoid silently accepting the regular fallback for an irregular or
        # contradictory surface reading, including recombined verbal slots.
        expected = {'VBD': forms.past, 'VBN': forms.past_participle, 'VBG': forms.present_participle,
                    'VBZ': forms.present_singular_3rd, 'VBP': forms.stem, 'VB': forms.stem}.get(slot.get('tag'))
        if doc.text[slice(*spans[key])].lower() != expected:
            raise Unsupported('unresolved_verb_inflection')
        parts = []
        constructions = group.get('constructions', [])
        particles = [c for c in constructions if c['kind'] == 'particle']
        if len(particles) > 1:
            raise Unsupported('ambiguous_particle_construction')
        if particles:
            parts = [PredicatePart(m['text'], TextSpan(*m['span'])) for m in particles[0]['markers']]
        lemma = ' '.join([head, *(p.text.lower() for p in parts)])
        governed = None
        for candidate in constructions:
            frame = self.frames.get(candidate['lemma'])
            if frame and frame.object_preposition:
                governed = frame.object_preposition
                parts = [PredicatePart(m['text'], TextSpan(*m['span']),
                                       'preposition' if m['text'].lower() == governed else 'particle')
                         for m in candidate['markers']]
                lemma = candidate['lemma']
                break
        if not governed and any(c['kind'] == 'adverb_preposition_candidate' for c in constructions):
            raise Unsupported('unresolved_adverb_preposition')
        if not governed and not particles and (head == 'put' or not objects and
                any(c['kind'] == 'preposition_candidate' for c in constructions)):
            raise Unsupported('unresolved_prepositional_complement')
        frame = self.frames.get(lemma)
        pred = Predicate(f'{doc.document_id}#predicate-{key}', doc.document_id, TextSpan(*spans[key]),
                         doc.text[slice(*spans[key])], lemma,
                         predicate_type=frame.predicate_type if frame else None,
                         head_lemma=head, parts=tuple(parts), tense=tense, modality=modal, aspect=aspect,
                         voice='passive' if passive else 'active',
                         polarity=Polarity.NEGATIVE if negated else Polarity.POSITIVE, source=SOURCE,
                         allows_passive=frame.allows_passive if frame else True)
        args = []
        # Construction markers retain verb-attached prepositions even in old
        # banks. A by-agent is absent from that inventory and needs a narrower
        # single-predicate fallback.
        pp = []
        local_predicates = sum(left <= spans[g['head_slot']][0] < right
                               for g in doc.metadata.get('predicate_groups', []) if g.get('head_slot') in spans)
        for k, s in slots.items():
            if (s.get('dependency') != 'pobj' or s.get('kind') != 'nominal'
                    or k not in spans or not left <= spans[k][0] < right):
                continue
            mention = self._mention(doc, slots, spans, k)
            before = doc.text[left:mention.span.start_char].rstrip().split()
            prep = before[-1].lower() if before else ''
            attached = any(c['kind'] in {'preposition_candidate', 'particle_preposition'}
                           and c['markers'] and c['markers'][-1]['text'].lower() == prep
                           and c['markers'][-1]['span'][1] <= mention.span.start_char
                           and not doc.text[c['markers'][-1]['span'][1]:mention.span.start_char].strip()
                           for c in constructions)
            if (prep == 'by' and passive and local_predicates == 1
                    and mention.span.start_char > spans[key][1]):
                pp.append((prep, mention))
            elif attached and (prep == governed or prep == 'to' and head in _TRANSFER):
                pp.append((prep, mention))
        agent_role = Relation.EXPERIENCER_OF if head in _STATES or head in {'feel', 'see'} else Relation.AGENT_OF
        theme_role = Relation.STATE_OF if head == 'feel' else Relation.THEME_OF
        if passive:
            if objects:
                raise Unsupported('passive_with_multiple_objects')
            args.append((theme_role, subject))
            agents = [m for prep, m in pp if prep == 'by']
            if len(agents) > 1:
                raise Unsupported('ambiguous_passive_agent')
            if agents:
                args.append((agent_role, agents[0]))
        else:
            args.append((agent_role, subject))
            if objects:
                args.append((theme_role, self._mention(doc, slots, spans, objects[0])))
            elif governed:
                themes = [m for prep, m in pp if prep == governed]
                if len(themes) != 1:
                    raise Unsupported('unresolved_governed_object')
                args.append((theme_role, themes[0]))
        if recipients:
            if head not in _TRANSFER | _BENEFACTIVE | {'ask'}:
                raise Unsupported('unresolved_recipient_construction')
            args.append((Relation.RECIPIENT_OF, self._mention(doc, slots, spans, recipients[0])))
        for prep, mention in pp:
            if prep == 'to' and not governed:
                args.append((Relation.RECIPIENT_OF, mention))
        if sum(role == Relation.RECIPIENT_OF for role, _ in args) > 1:
            raise Unsupported('ambiguous_recipient')
        if negated:
            # Dropping an adjunct can reverse a negative claim: "didn't buy a
            # book in Paris" does not imply "didn't buy a book". Keep only
            # negative clauses whose full scope is represented in the question.
            covered = [spans[key], *(spans[k] for k in auxiliaries),
                       *([m.span.start_char, m.span.end_char] for _, m in args),
                       *([p.span.start_char, p.span.end_char] for p in parts)]
            residue = list(doc.text[left:right])
            for start, end in covered:
                for index in range(max(left, start), min(right, end)):
                    residue[index-left] = ' '
            remainder = re.sub(r"\bnot\b|n['\u2019]t", '', ''.join(residue), flags=re.I)
            if re.search(r'\w', remainder):
                raise Unsupported('negative_complement_scope')
        return pred, args

    def _questions(self, doc, predicate, args):
        by_role = dict(args)
        agent = by_role.get(Relation.AGENT_OF) or by_role.get(Relation.EXPERIENCER_OF)
        theme = by_role.get(Relation.THEME_OF) or by_role.get(Relation.STATE_OF)
        recipient = by_role.get(Relation.RECIPIENT_OF)
        passive = predicate.voice == 'passive' and agent is None
        subject = theme if passive else agent
        if subject is None:
            return []

        def inline(mention):
            if mention is None:
                return ''
            text = mention.exact_text
            if text.split()[0].lower() in {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'my', 'your', 'our', 'their'}:
                text = text[:1].lower() + text[1:]
            if text.lower() in _PRONOUNS | {'it'}:
                text = 'I' if text.lower() == 'i' else text.lower()
            if mention is subject:
                text = {'me': 'I', 'him': 'he', 'her': 'she', 'us': 'we', 'them': 'they'}.get(text.lower(), text)
            else:
                text = {'i': 'me', 'he': 'him', 'she': 'her', 'we': 'us', 'they': 'them'}.get(text.lower(), text)
            return text

        frame = Frame(self.resolver(predicate.head_lemma), ArgStructure({}, is_passive=passive),
                      tense=predicate.modality or predicate.tense,
                      is_perfect='perfect' in predicate.aspect, is_progressive='progressive' in predicate.aspect,
                      is_negated=predicate.polarity == Polarity.NEGATIVE)
        edges = []
        recipient_prep = ('to' if predicate.head_lemma in _TRANSFER else
                          'for' if predicate.head_lemma in _BENEFACTIVE else '') if recipient else ''
        for role, mention in args:
            gap_subject = mention is subject
            if gap_subject and any(m is not subject and m.exact_text.lower() in _REFLEXIVES for _, m in args):
                continue  # "Who injured myself?" requires reflexive rebinding.
            words = verb_words(frame, question=not gap_subject,
                               plural=not gap_subject and subject.number == 'plural')
            if not gap_subject and inline(subject) == 'I':
                words[0] = {'are': 'am', 'were': 'was', "weren't": "wasn't"}.get(words[0], words[0])
            if gap_subject:
                prefix = [self._wh(mention), *words]
            else:
                prefix = [self._wh(mention), words[0], inline(subject), *words[1:]]
            tail = [p.text.lower() for p in predicate.parts]
            object_text = inline(theme) if not passive and theme is not mention else ''
            if recipient and not recipient_prep and recipient is not mention:
                prefix.append(inline(recipient))
            if (object_text.lower() in {'it', 'him', 'her', 'them', 'me', 'us', 'you'}
                    and tail and all(p.kind == 'particle' for p in predicate.parts)):
                prefix += [object_text, *tail]
            else:
                prefix += [*tail, object_text]
            if recipient_prep:
                prefix += [recipient_prep, '' if recipient is mention else inline(recipient)]
            question = ' '.join(word for word in prefix if word).strip()
            question = question[:1].upper() + question[1:] + '?'
            edges.append(RelationEdge(mention.mention_id, role, predicate.predicate_id,
                                      question=question, source=SOURCE))
        return edges

    @staticmethod
    def _wh(mention):
        if Label.GROUP in mention.labels:
            return 'Which group'
        if mention.labels.labels & {Label.PERSON, Label.ANIMAL, Label.ORGANIZATION}:
            return 'Who'
        if mention.labels.entity_labels or mention.exact_text.lower() == 'it':
            return 'What'
        return 'Who or what'


def questions_for_run(document, run, *, context=None):
    """Only questions supported by dependency candidates, with provenance.

    Do not infer negative examples, complete events or verified entity types
    from a partial dependency analysis. Re-running QA uses the stored wording.
    """
    untyped = replace(run, predicates=tuple(replace(p, predicate_type=None) for p in run.predicates))
    rows = native_questions(document, untyped, context=context)
    mentions = {m.mention_id: m for m in run.mentions}
    for prop in run.properties:
        target = mentions.get(prop.target_id)
        if target is None:
            continue
        wording = _without_property(target.exact_text, prop.head, prop.degree)
        if wording.split()[0].lower() in {'a', 'an', 'the'}:
            wording = wording[:1].lower() + wording[1:]
        rows.append(QAExample(
            context=context if context is not None else document.text,
            question=f'Which property is attributed to {wording}?', answer=phrase_answer(prop.exact_text),
            kind=QAKind.PROPERTY, document_id=document.document_id, run_id=run.run_id,
            evidence=(prop.span,), metadata={'head': prop.head, 'target_id': prop.target_id}))
    return [replace(row, metadata={**row.metadata, 'annotation_source': SOURCE,
                                  'review_status': str(ReviewStatus.UNREVIEWED),
                                  'semantic_method': VERSION, 'template_id': document.metadata.get('template_id')})
            for row in rows]


def accumulate_coverage(stats, document, run, qa_count=0):
    coverage = document.metadata['template_semantics']
    for key, value in [('annotated_documents', bool(run)), ('complete_documents', coverage['complete']),
                       ('pending_documents', not coverage['complete']), ('qa_examples', qa_count)]:
        stats[key] = stats.get(key, 0) + int(value)
    reasons = stats.setdefault('skip_reasons', {})
    for reason, count in Counter(item['reason'] for item in coverage['skipped']).items():
        reasons[reason] = reasons.get(reason, 0) + count


def needs_annotation(document, runs):
    """Partial automatically annotated records remain eligible for a teacher."""
    latest = None
    for run in runs:
        latest = run
    return latest is None or (latest.model_name == SOURCE and
                             not document.metadata.get('template_semantics', {}).get('complete', False))
