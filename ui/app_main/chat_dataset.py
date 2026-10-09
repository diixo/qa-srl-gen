"""Inputs and worker adapter for the dedicated chat-dataset page."""

from zipfile import ZIP_DEFLATED, ZipFile

from django import forms

from semantic_corpus.dialogue_generator import TASKS, USER_ASSISTANT_TASKS, build_dialogue_dataset, dialogue_tasks

TASK_LABELS = {
    'recall': 'Recall facts',
    'correction': 'Correct earlier information',
    'different_people': 'Distinguish different people',
    'negation': 'Understand negation',
    'desire_vs_done': 'Distinguish wishes and plans from completed actions',
    'option_reference': 'Resolve first / second option',
    'ambiguous_reference': 'Clarify ambiguous references',
    'unknown': 'Recognize missing information',
    'suggestion_vs_done': 'Distinguish suggestions from completed actions',
    'topic_return': 'Return to an earlier topic',
    'retraction': 'Withdraw an earlier fact',
    'capability': 'Respect the assistant\'s capabilities',
    'event_scope': 'Distinguish scoped intentions, needs and phrasal actions',
    'event_progress': 'Distinguish attempts, progress and completion',
    'event_updates': 'Correct and retract events without losing other actions',
    'event_states': 'Distinguish location and possession from wishes and past states',
    'mined_predicates': 'Use all mined predicates: compositional events and scoped source quotations',
}


class ChatDatasetForm(forms.Form):
    user_assistant_only = forms.BooleanField(
        label='User and assistant only', initial=True, required=False,
        help_text='Conversations about the user. Includes 16 tasks; uncheck to include named participants (17 tasks). '
                  'Mined source sentences may contain names in either profile.',
    )
    train = forms.IntegerField(
        label='Training dialogues per task', initial=200, min_value=1, max_value=1000000,
        help_text='Examples used to train the model.',
    )
    mined_train = forms.IntegerField(
        label='Mined training dialogues', initial=0, required=False, min_value=0, max_value=1000000,
        help_text='0 covers every mined predicate with yes/no/unknown examples. A positive value sets a separate quota; '
                  'blank uses Training dialogues per task. Validation and test keep their per-task quotas.',
    )
    validation = forms.IntegerField(
        label='Validation dialogues per task', initial=20, min_value=1, max_value=1000000,
        help_text='Examples used to compare training settings.',
    )
    test = forms.IntegerField(
        label='Test dialogues per task', initial=20, min_value=1, max_value=1000000,
        help_text='Held-out examples for the final evaluation.',
    )
    seed = forms.IntegerField(
        label='Seed', initial=42, min_value=0, max_value=2147483647,
        help_text='The same seed and resources reproduce the same dataset.',
    )
    max_length = forms.IntegerField(
        label='Maximum example tokens (GPT-2)', initial=768, required=False, min_value=1, max_value=1024,
        help_text='Total ChatML length: instructions, context and answer. Applies to all three datasets. '
                  'Long dialogues are replaced, never truncated. Blank disables the check. '
                  'Requires the optional tokenizer packages and a cached GPT-2 tokenizer; no model is loaded.',
    )
    include_metadata = forms.BooleanField(
        label='Include metadata', initial=True, required=False,
        help_text='Include IDs, task labels and contrast-group details in training JSON records. '
                  'Unchecked: export only messages. This does not change the message content.',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = ('form-check-input' if isinstance(field.widget, forms.CheckboxInput)
                                           else 'form-control')

    @classmethod
    def from_config(cls, config):
        # Preserve the behavior of jobs saved before these checkboxes existed.
        return cls({**config, 'include_metadata': config.get('include_metadata', True),
                    'user_assistant_only': config.get('user_assistant_only', False)})

    def tasks(self):
        return dialogue_tasks(self.fields['user_assistant_only'].clean(self['user_assistant_only'].value()))

    def clean(self):
        data = super().clean()
        for name in ('mined_train', 'max_length'):
            if data.get(name) is None:
                data.pop(name, None)
        return data

    def preview(self):
        totals = {}
        for name in ('train', 'validation', 'test'):
            try:
                value = self.fields[name].clean(self[name].value())
            except forms.ValidationError:
                totals[name] = None
            else:
                totals[name] = value * len(self.tasks())
                if name == 'train':
                    try:
                        mined = self.fields['mined_train'].clean(self['mined_train'].value())
                        if mined == 0:
                            mined = 3 * self.mined_inventory_count()
                        if mined is not None:
                            totals[name] += mined - value
                    except (forms.ValidationError, OSError, ValueError):
                        totals[name] = None
        totals['total'] = (sum(totals.values()) if all(value is not None for value in totals.values()) else None)
        return totals

    def mined_inventory_count(self):
        # Small inventories only. Never decompress the full template bank on GET.
        from semantic_corpus.resource_pool import BuiltInPool
        if not hasattr(self, '_mined_inventory_count'):
            self._mined_inventory_count = len({row['lemma'] for row in BuiltInPool().verbs()})
        return self._mined_inventory_count


def execute_chat_dataset(job, out, config, progress):
    from .models import Job

    form = ChatDatasetForm.from_config(config)
    if not form.is_valid():
        raise ValueError(f'Invalid chat dataset settings: {form.errors.as_text()}')

    def update(phase, processed, total):
        if Job.objects.filter(pk=job.pk, cancel_requested=True).exists():
            raise ValueError('Chat dataset generation was cancelled.')
        progress(phase, processed, total)

    # The existing artifact browser excludes work/, including interrupted builds.
    staged = out / 'work' / 'dataset'
    manifest = build_dialogue_dataset(staged, **form.cleaned_data, progress=update)
    total = sum(row['episodes'] for row in manifest['splits'].values())
    update('Preparing dataset download', total, total)
    dataset = out / 'dataset'
    staged.rename(dataset)
    pending_archive = out / 'chat-dataset.zip.partial'
    with ZipFile(pending_archive, 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(dataset.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(dataset).as_posix())
    update('Finalizing chat dataset', total, total)
    pending_archive.rename(out / 'chat-dataset.zip')
    for split, row in manifest['splits'].items():
        print(f"{split}: {row['episodes']} paired response examples; "
              f"{row['extraction_rows']} memory updates", flush=True)
    progress('Chat dataset ready', total, total,
             **{split: row['episodes'] for split, row in manifest['splits'].items()},
             memory_updates=sum(row['extraction_rows'] for row in manifest['splits'].values()))
