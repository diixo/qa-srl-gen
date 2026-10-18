"""Complete, versioned index of the working mined predicate bank for dialogues.

Nothing here certifies parser output as semantic truth. Inflections normalize
observations; full attested clauses keep argument order, negation and embedded
operators. Unsupported morphology is retained explicitly, never regularized by
guessing. The cache is disposable and source banks are read only.
"""
from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import tempfile

from ..controlled_verbs import VERSION as CONTROLLED_VERSION, controlled_head, excluded_predicates
from ..resource_pool import BuiltInPool
from ..template_bank import CorpusTemplate, fingerprint
from ..ontology import Label
from .mined_realization import active_phrase, source_scope, lexicon, lexicon_fingerprint, refresh_resources, DICTIONARY_PATH
from .mined_arguments import realize_arguments
from .mined_subjects import realize_subject
from .mined_review import POLICY_PATH, REASONS, normalize_inventory, review_template, reason_category, policy, policy_fingerprint

VERSION = 'dialogue-mined-bank-v9'


def normalize_verb(row, lexicon):
    """Keep all matching paradigms; reverse lookup repairs parser lemmas only
    when the observed forms agree on a single dictionary head.
    """
    return normalize_inventory(row, lexicon)


class DialogueMinedBank:
    def __init__(self, repo_root=None, *, cache=True, progress=None):
        self.pool = BuiltInPool(repo_root)
        self.root = self.pool.repo_root
        self.banks = self.pool.banks
        self.dictionary = self.root / 'data/wiktionary/en_verb_inflections.txt'
        refresh_resources(self.dictionary)
        self._morphology_policy = deepcopy(policy())
        self.person_names = {e.text for e in self.pool.entities if e.is_named and Label.PERSON in e.labels}
        # Fingerprint all bank files, not just the list of names. Published banks
        # are immutable; stat changes invalidate a disposable cache even if an
        # external edit forgot to update its manifest.
        self.sources = {'version': VERSION, 'controlled_policy': CONTROLLED_VERSION,
                        'person_names_sha256': fingerprint(sorted(self.person_names)),
                        'entity_types_sha256': fingerprint([(e.text, str(e.labels)) for e in self.pool.entities]),
                        'compiler_sha256': sha256(Path(__file__).read_bytes() +
                                                 Path(__file__).with_name('mined_realization.py').read_bytes() +
                                                 Path(__file__).with_name('mined_scope.py').read_bytes() +
                                                 Path(__file__).with_name('verb_operators.py').read_bytes() +
                                                 Path(__file__).with_name('mined_arguments.py').read_bytes() +
                                                 Path(__file__).with_name('mined_clauses.py').read_bytes() +
                                                 Path(__file__).parents[1].joinpath('template_syntax.py').read_bytes() +
                                                 Path(__file__).with_name('mined_subjects.py').read_bytes()).hexdigest(),
                        'dictionary_sha256': lexicon_fingerprint(self.dictionary),
                        'renderer_dictionary_sha256': lexicon_fingerprint(),
                        'lexical_review_sha256': sha256(Path(__file__).with_name('mined_review.py').read_bytes()).hexdigest(),
                        'morphology_policy_sha256': policy_fingerprint(),
                        'banks': [{'path': str(b.path.resolve()), 'manifest': fingerprint(b.manifest),
                                   'files': [(str(p.relative_to(b.path)), p.stat().st_size, p.stat().st_mtime_ns)
                                             for p in sorted(b.path.rglob('*.gz'))]} for b in self.banks]}
        self.identity = fingerprint(self.sources)
        self.cache_path = self.root / 'artifacts/dialogue-resource-cache' / (self.identity + '.json.gz')
        self._shards = OrderedDict()
        self._selected = {}
        if cache and self.cache_path.exists():
            with gzip.open(self.cache_path, 'rt', encoding='utf-8') as stream:
                data = json.load(stream)
            if data.get('identity') != self.identity:
                raise ValueError('Mined dialogue cache identity mismatch')
        else:
            data = self._build(progress)
            self.validate_resources()
            if cache:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=self.cache_path.parent, suffix='.partial', delete=False) as tmp:
                    temporary = Path(tmp.name)
                try:
                    with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
                        json.dump(data, stream, ensure_ascii=False, separators=(',', ':'))
                    try:
                        os.replace(temporary, self.cache_path)
                    except PermissionError:
                        # Another Windows worker may already be reading the
                        # identical, atomically published cache.
                        with gzip.open(self.cache_path, 'rt', encoding='utf-8') as stream:
                            existing = json.load(stream)
                        if existing.get('identity') != self.identity:
                            raise
                finally:
                    temporary.unlink(missing_ok=True)
        self.entries = data['entries']
        self.report = data['report']
        self.lemmas = tuple(sorted(key for key, entry in self.entries.items() if entry['references']))
        self.validate_resources()

    def validate_resources(self):
        """Do not publish/cache an index or export under another rule snapshot."""
        for path, key in ((self.dictionary, 'dictionary_sha256'),
                          (DICTIONARY_PATH, 'renderer_dictionary_sha256'),
                          (POLICY_PATH, 'morphology_policy_sha256')):
            if sha256(path.read_bytes()).hexdigest() != self.sources[key]:
                raise ValueError(f'Mined resource changed during generation: {path}; create a new bank and retry')
        if policy_fingerprint() != self.sources['morphology_policy_sha256']:
            raise ValueError('Mined policy snapshot changed during generation; create a new bank and retry')

    def _build(self, progress):
        dictionary = lexicon(self.dictionary)
        entries = {}
        alternatives = Counter()
        for bank_index, bank in enumerate(self.banks):
            for row in bank.verbs():
                if controlled_head(row['lemma']):
                    continue
                info = normalize_verb(row, dictionary)
                if row['lemma'] in entries:
                    forms = Counter(entries[row['lemma']]['forms'])
                    forms.update(info['forms'])
                    entries[row['lemma']]['forms'] = dict(forms)
                entry = entries.setdefault(row['lemma'], {**info, 'references': [], 'native_references': [],
                                                          'scoped_references': [],
                                                          'passive_references': [], 'argument_references': [], 'multi_references': [],
                                                          'native_rejections': Counter(), 'source_banks': [],
                                                          'lexical_decisions': Counter(), 'review_examples': {},
                                                          'resolved_heads': Counter()})
                entry['source_banks'].append(bank_index)
            if 'verb_alternatives.jsonl.gz' in bank.files:
                for row in bank.rows('verb_alternatives.jsonl.gz'):
                    if not controlled_head(row['lemma']):
                        alternatives[row['lemma']] += sum(row.get('forms', {}).values())
        for lemma, entry in entries.items():
            entry.update(normalize_verb({'lemma': lemma, 'forms': entry['forms']}, dictionary))
        templates = predicates = excluded = 0
        for bank_index, bank in enumerate(self.banks):
            for template in bank.templates():
                templates += 1
                reviewed = review_template(template.record, dictionary)
                if excluded_predicates(template.record):
                    excluded += 1
                # Source quotations may retain controlled predicates as context.
                # They are never selected or realized as an unreviewed head.
                for head_slot, slot in template.slots.items():
                    if slot.get('kind') != 'verb':
                        continue
                    lemma = slot['lemma']
                    if lemma not in entries:
                        continue
                    entry = entries[lemma]
                    if entry['controlled']:
                        continue  # A recovered alias must not bypass manual routing.
                    reference = [bank_index, template.template_id, head_slot]
                    entry['references'].append(reference)
                    decision = reviewed[1][head_slot]
                    entry['lexical_decisions'][decision['reason']] += 1
                    if decision['status'] != 'quotation_only':
                        entry['resolved_heads'][decision['canonical']] += 1
                    spec, reason = active_phrase(template.record, head_slot, entry['canonical'],
                                                 entry['paradigms'], self.person_names, reviewed=reviewed)
                    if spec:
                        entry['native_references'].append(reference)
                        if spec['has_source_scope']:
                            entry['scoped_references'].append(reference)
                        if spec.get('voice') == 'passive':
                            entry['passive_references'].append(reference)
                        if spec.get('arguments'):
                            entry['argument_references'].append(reference)
                        if spec.get('companions'):
                            entry['multi_references'].append(reference)
                    else:
                        entry['native_rejections'][reason] += 1
                    # One reproducible example per decision/restriction per raw
                    # head; counts still cover EVERY inspected occurrence.
                    for issue in sorted({decision['reason'], reason} - {None, 'consistent'}):
                        if issue not in entry['review_examples']:
                            entry['review_examples'][issue] = {
                                'reference': reference, 'category': reason_category(issue),
                                'lexical_review': decision,
                                'compositional': bool(spec), 'restriction': reason,
                                'source_text': ''.join(p.get('literal', template.record['defaults'].get(p.get('slot'), ''))
                                                       for p in template.record['parts'])}
                    predicates += 1
                if progress and templates % 10000 == 0:
                    progress(templates)
        dispositions = Counter()
        for entry in entries.values():
            entry['status'] = ('controlled_alias' if entry['controlled'] else
                               'compositional' if entry['native_references'] else
                               'attested_only' if entry['references'] else 'no_observed_verb_slot')
            dispositions[entry['status']] += 1
        report = {'identity': self.identity, 'sources': self.sources, 'templates_scanned': templates,
                  'excluded_templates': excluded, 'eligible_templates': templates - excluded,
                  'predicate_occurrences': predicates, 'inventory_lemmas': len(entries),
                  'dispositions': dict(dispositions), 'inflections': dict(Counter(e['inflection_source'] for e in entries.values())),
                  'source_scope_lemmas': sum(bool(e['scoped_references']) for e in entries.values()),
                  'source_scope_occurrences': sum(len(e['scoped_references']) for e in entries.values()),
                  'passive_occurrences': sum(len(e['passive_references']) for e in entries.values()),
                  'argument_occurrences': sum(len(e['argument_references']) for e in entries.values()),
                  'multi_event_occurrences': sum(len(e['multi_references']) for e in entries.values()),
                  'alternative_lemmas': len(alternatives),
                  'alternative_only_lemmas': sorted(set(alternatives) - set(entries)),
                  'unavailable': {k: e['status'] for k, e in entries.items() if not e['references']},
                  'review_status': 'UNREVIEWED',
                  'lexical_decisions': dict(sum((e['lexical_decisions'] for e in entries.values()), Counter())),
                  'review_reasons': REASONS,
                  'note': 'Every archived template is inspected. Raw inventory entries are not certified verb senses. Lexical projection never overwrites the source; ambiguous forms and unsupported syntax remain whole quotations.'}
        return {'identity': self.identity, 'entries': entries, 'report': report}

    def template(self, reference):
        bank_index, template_id, head = reference
        if (bank_index, template_id) in self._selected:
            return CorpusTemplate(self._selected[bank_index, template_id]), head
        key = (bank_index, template_id[:2])
        if key not in self._shards:
            bank = self.banks[bank_index]
            self._shards[key] = {r['template_id']: r for r in bank.rows('templates/' + template_id[:2] + '.jsonl.gz')}
            # Bounded cache: no loading the complete bank into RAM.
            if len(self._shards) > 4:
                self._shards.popitem(last=False)
        self._shards.move_to_end(key)
        return CorpusTemplate(self._shards[key][template_id]), head

    def reference(self, lemma, rng, *, native=True):
        entry = self.entries[lemma]
        references = entry['native_references'] if native and entry['native_references'] else entry['references']
        featured = [entry[key] for key in ('scoped_references', 'passive_references', 'multi_references') if entry[key]]
        if native and featured and rng.randrange(2) == 0:
            references = rng.choice(featured)
        return rng.choice(references)

    def preload(self, references):
        """Read each needed gzip shard once for a full vocabulary cycle."""
        by_shard = {}
        self._selected = {}
        for bank_index, template_id, _ in references:
            by_shard.setdefault((bank_index, template_id[:2]), set()).add(template_id)
        for (bank_index, prefix), wanted in sorted(by_shard.items()):
            for row in self.banks[bank_index].rows('templates/' + prefix + '.jsonl.gz'):
                if row['template_id'] in wanted:
                    self._selected[bank_index, row['template_id']] = row
        self._shards.clear()

    def sample(self, lemma, rng, *, native=True):
        entry = self.entries[lemma]
        reference = self.reference(lemma, rng, native=native)
        template, head = self.template(reference)
        row = template.sample(seed=rng.getrandbits(64))
        reviewed = review_template(template.record, lexicon(self.dictionary))
        spec, reason = active_phrase(template.record, head, entry['canonical'], entry['paradigms'], self.person_names,
                                     reviewed=reviewed)
        decision = reviewed[1][head]
        info = {'bank': str(self.banks[reference[0]].path),
                           'source_id': self.banks[reference[0]].source_id, 'template_id': template.template_id,
                           'raw_lemma': lemma, 'canonical_lemma': decision['canonical'],
                           'lexical_review': decision, 'restriction': reason,
                           'inflection_source': entry['inflection_source'], 'review_status': 'UNREVIEWED',
                           'source_text': row['text'], 'head_slot': head,
                           'slots': row['slots'], 'spans': row['spans'], 'predicate_groups': row['predicate_groups']}
        if 'usage_constraints' in row:
            info['usage_constraints'] = row['usage_constraints']
        if spec:
            def realize(candidate):
                subject, reason = realize_subject(template.record, candidate, row['bindings'], self.person_names, self.pool.entities)
                if reason:
                    raise ValueError(f'Indexed subject no longer validates: {reason}')
                values, reason = realize_arguments(template.record, candidate, row['bindings'], self.person_names, self.pool.entities)
                if reason:
                    raise ValueError(f'Indexed arguments no longer validate: {reason}')
                tail, arguments = values
                return {**candidate, 'tail': tail, 'argument_values': arguments, 'subject': subject,
                        'source_scope': source_scope(candidate, row['bindings'], candidate['source_paradigms'])}
            companions = [realize(c) for c in spec.get('companions', ())]
            spec = {**realize(spec), 'companions': companions}
            info['subject'] = asdict(spec['subject'])
            info['source_scope'] = asdict(spec['source_scope'])
            info['voice'] = spec['voice']
            info['arguments'] = spec['argument_values']
            info['companion_heads'] = [c['head_slot'] for c in companions]
            info['companion_lexical_reviews'] = [c['lexical_review'] for c in companions]
        return row, spec, info

    def coverage(self, selected):
        selected = set(selected)
        return {**self.report, 'available_lemmas': len(self.lemmas), 'selected_lemmas': sorted(selected),
                'unselected_lemmas': sorted(set(self.lemmas) - selected)}

    def inventory_report(self):
        return {lemma: {key: entry[key] for key in ('canonical', 'inflection_source', 'status', 'native_rejections',
                                                   'lexical_decisions', 'resolved_heads')}
                | {'source_occurrences': len(entry['references']), 'compositional_occurrences': len(entry['native_references']),
                   'scoped_occurrences': len(entry['scoped_references']),
                   'passive_occurrences': len(entry['passive_references']),
                   'argument_occurrences': len(entry['argument_references']),
                   'multi_event_occurrences': len(entry['multi_references'])}
                for lemma, entry in self.entries.items()}

    def review_report(self):
        self.validate_resources()
        return {'identity': self.identity, 'policy': deepcopy(self._morphology_policy),
                'reason_descriptions': REASONS, 'lexical_decisions': self.report['lexical_decisions'],
                'note': 'Examples are bounded to one per reason and raw head; counts cover all occurrences. '
                        'Use reference [bank index, template ID, head slot] to inspect the immutable source. '
                        'A passed morphology check does not resolve word sense or certify parser roles.',
                'predicates': {lemma: {**row, 'examples': self.entries[lemma]['review_examples']}
                               for lemma, row in self.inventory_report().items()}}
