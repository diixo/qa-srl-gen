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
}


class ChatDatasetForm(forms.Form):
    user_assistant_only = forms.BooleanField(
        label='User and assistant only', initial=True, required=False,
        help_text='Conversations about the user: no personal names, named places or facts about other people. '
                  'Includes 15 tasks. Uncheck to include named people and places (16 tasks).',
    )
    train = forms.IntegerField(
        label='Training dialogues per task', initial=200, min_value=1, max_value=1000000,
        help_text='Examples used to train the model.',
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

    def preview(self):
        totals = {}
        for name in ('train', 'validation', 'test'):
            try:
                value = self.fields[name].clean(self[name].value())
            except forms.ValidationError:
                totals[name] = None
            else:
                totals[name] = value * len(self.tasks())
        totals['total'] = (sum(totals.values()) if all(value is not None for value in totals.values()) else None)
        return totals


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
