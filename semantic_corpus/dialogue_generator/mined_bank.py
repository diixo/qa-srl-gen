"""Complete, versioned index of the working mined predicate bank for dialogues.

Nothing here certifies parser output as semantic truth. Inflections normalize
observations; full attested clauses keep argument order, negation and embedded
operators. Unsupported morphology is retained explicitly, never regularized by
guessing. The cache is disposable and source banks are read only.
"""
from collections import Counter, OrderedDict
from dataclasses import asdict
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import tempfile

from ..controlled_verbs import VERSION as CONTROLLED_VERSION, controlled_head, excluded_predicates
from ..qasrl_core.inflections import load_inflections
from ..resource_pool import BuiltInPool
from ..template_bank import CorpusTemplate, fingerprint, read_rows
from ..ontology import Label
from .mined_realization import active_phrase, phrase_tail
from .mined_subjects import realize_subject

VERSION = 'dialogue-mined-bank-v4'


def normalize_verb(row, lexicon):
    """Keep all matching paradigms; reverse lookup repairs parser lemmas only
    when the observed forms agree on a single dictionary head.
    """
    lemma = row['lemma']
    observed = Counter({word.lower(): count for word, count in row.get('forms', {}).items()})
    candidates = lexicon.paradigms(lemma)
    source = 'dictionary'
    if not candidates:
        # The same surface can be three forms of one paradigm (read/read/read).
        # That is not three independent votes for a lexical interpretation.
        matching = [{p.stem for p, _ in lexicon.analyses(surface)} for surface in observed]
        matching = [heads for heads in matching if heads]
        heads = set.intersection(*matching) if matching else set()
        candidates = lexicon.paradigms(next(iter(heads))) if len(heads) == 1 else ()
        source = 'recovered_from_observed_forms' if candidates else 'observed_only'
    canonical = candidates[0].stem if candidates else lemma
    return {'lemma': lemma, 'canonical': canonical, 'forms': dict(observed),
            'paradigms': [asdict(p) for p in candidates], 'inflection_source': source,
            'controlled': controlled_head(canonical)}


class DialogueMinedBank:
    def __init__(self, repo_root=None, *, cache=True, progress=None):
        self.pool = BuiltInPool(repo_root)
        self.root = self.pool.repo_root
        self.banks = self.pool.banks
        self.dictionary = self.root / 'data/wiktionary/en_verb_inflections.txt'
        self.person_names = {e.text for e in self.pool.entities if e.is_named and Label.PERSON in e.labels}
        # Fingerprint all bank files, not just the list of names. Published banks
        # are immutable; stat changes invalidate a disposable cache even if an
        # external edit forgot to update its manifest.
        self.sources = {'version': VERSION, 'controlled_policy': CONTROLLED_VERSION,
                        'person_names_sha256': fingerprint(sorted(self.person_names)),
                        'entity_types_sha256': fingerprint([(e.text, str(e.labels)) for e in self.pool.entities]),
                        'compiler_sha256': sha256(Path(__file__).read_bytes() +
                                                 Path(__file__).with_name('mined_realization.py').read_bytes() +
                                                 Path(__file__).with_name('mined_subjects.py').read_bytes()).hexdigest(),
                        'dictionary_sha256': sha256(self.dictionary.read_bytes()).hexdigest(),
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

    def _build(self, progress):
        lexicon = load_inflections(self.dictionary)
        entries = {}
        alternatives = Counter()
        for bank_index, bank in enumerate(self.banks):
            for row in bank.verbs():
                if controlled_head(row['lemma']):
                    continue
                info = normalize_verb(row, lexicon)
                if row['lemma'] in entries:
                    forms = Counter(entries[row['lemma']]['forms'])
                    forms.update(info['forms'])
                    entries[row['lemma']]['forms'] = dict(forms)
                entry = entries.setdefault(row['lemma'], {**info, 'references': [], 'native_references': [],
                                                          'native_rejections': Counter(), 'source_banks': []})
                entry['source_banks'].append(bank_index)
            alt = bank.path / 'verb_alternatives.jsonl.gz'
            if alt.is_file():
                for row in read_rows(alt):
                    if not controlled_head(row['lemma']):
                        alternatives[row['lemma']] += sum(row.get('forms', {}).values())
        for lemma, entry in entries.items():
            entry.update(normalize_verb({'lemma': lemma, 'forms': entry['forms']}, lexicon))
        templates = predicates = excluded = 0
        for bank_index, bank in enumerate(self.banks):
            for template in bank.templates():
                templates += 1
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
                    spec, reason = active_phrase(template.record, head_slot, entry['canonical'],
                                                 entry['paradigms'], self.person_names)
                    if spec:
                        entry['native_references'].append(reference)
                    else:
                        entry['native_rejections'][reason] += 1
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
                  'alternative_lemmas': len(alternatives),
                  'alternative_only_lemmas': sorted(set(alternatives) - set(entries)),
                  'unavailable': {k: e['status'] for k, e in entries.items() if not e['references']},
                  'review_status': 'UNREVIEWED',
                  'note': 'Every archived template is inspected. Only uncontrolled heads are selected. Whole quotations retain source context; dictionary homographs are not promoted to asserted predicates.'}
        return {'identity': self.identity, 'entries': entries, 'report': report}

    def template(self, reference):
        bank_index, template_id, head = reference
        if (bank_index, template_id) in self._selected:
            return CorpusTemplate(self._selected[bank_index, template_id]), head
        key = (bank_index, template_id[:2])
        if key not in self._shards:
            path = self.banks[bank_index].path / 'templates' / (template_id[:2] + '.jsonl.gz')
            self._shards[key] = {r['template_id']: r for r in read_rows(path)}
            # Bounded cache: no loading the complete bank into RAM.
            if len(self._shards) > 4:
                self._shards.popitem(last=False)
        self._shards.move_to_end(key)
        return CorpusTemplate(self._shards[key][template_id]), head

    def reference(self, lemma, rng, *, native=True):
        entry = self.entries[lemma]
        references = entry['native_references'] if native and entry['native_references'] else entry['references']
        return rng.choice(references)

    def preload(self, references):
        """Read each needed gzip shard once for a full vocabulary cycle."""
        by_shard = {}
        self._selected = {}
        for bank_index, template_id, _ in references:
            by_shard.setdefault((bank_index, template_id[:2]), set()).add(template_id)
        for (bank_index, prefix), wanted in sorted(by_shard.items()):
            for row in read_rows(self.banks[bank_index].path / 'templates' / (prefix + '.jsonl.gz')):
                if row['template_id'] in wanted:
                    self._selected[bank_index, row['template_id']] = row
        self._shards.clear()

    def sample(self, lemma, rng, *, native=True):
        entry = self.entries[lemma]
        reference = self.reference(lemma, rng, native=native)
        template, head = self.template(reference)
        row = template.sample(seed=rng.getrandbits(64))
        spec, _ = active_phrase(template.record, head, entry['canonical'], entry['paradigms'], self.person_names)
        info = {'bank': str(self.banks[reference[0]].path), 'template_id': template.template_id,
                           'raw_lemma': lemma, 'canonical_lemma': entry['canonical'],
                           'inflection_source': entry['inflection_source'], 'review_status': 'UNREVIEWED',
                           'source_text': row['text'], 'head_slot': head,
                           'slots': row['slots'], 'spans': row['spans'], 'predicate_groups': row['predicate_groups']}
        if spec:
            subject, reason = realize_subject(template.record, spec, row['bindings'], self.person_names, self.pool.entities)
            if reason:
                raise ValueError(f'Indexed subject no longer validates: {reason}')
            spec = {**spec, 'tail': phrase_tail(spec, row['bindings']), 'subject': subject}
            info['subject'] = asdict(subject)
        return row, spec, info

    def coverage(self, selected):
        selected = set(selected)
        return {**self.report, 'available_lemmas': len(self.lemmas), 'selected_lemmas': sorted(selected),
                'unselected_lemmas': sorted(set(self.lemmas) - selected)}

    def inventory_report(self):
        return {lemma: {key: entry[key] for key in ('canonical', 'inflection_source', 'status', 'native_rejections')}
                | {'source_occurrences': len(entry['references']), 'compositional_occurrences': len(entry['native_references'])}
                for lemma, entry in self.entries.items()}
