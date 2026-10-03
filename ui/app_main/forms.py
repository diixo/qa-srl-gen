from django import forms

from . import catalog
from .models import OPERATIONS


class JobForm(forms.Form):
    operation = forms.ChoiceField(label='Operation', choices=OPERATIONS)
    source = forms.ChoiceField(label='Source file', required=False)
    store = forms.ChoiceField(label='Source corpus', required=False)
    input_format = forms.ChoiceField(label='Text format', choices=[
        ('text', 'TXT / JSONL with a text field'), ('dialogue', 'JSONL: list of dialogue turns')])
    count = forms.IntegerField(label='Number of situations', min_value=1, max_value=1000000, initial=100)
    limit = forms.IntegerField(label='Input document limit', min_value=1, required=False,
                              help_text='Leave blank to process the entire input corpus.')
    seed = forms.IntegerField(label='Seed', initial=42, min_value=0, max_value=2147483647)
    split = forms.ChoiceField(label='Split', choices=[(s, s) for s in ('train', 'dev', 'test')])
    pool = forms.ChoiceField(label='Entity pool', required=False)
    context_turns = forms.IntegerField(label='Context: preceding turns', initial=2, min_value=0, max_value=100)
    scheme = forms.ChoiceField(label='Export tagging scheme', choices=[('BIO', 'BIO'), ('BILOU', 'BILOU')])
    max_per_kind = forms.IntegerField(label='Maximum QA per kind per document', min_value=1, required=False)
    no_answer_share = forms.FloatField(label='Maximum unanswerable share', min_value=0, max_value=1, required=False)
    teacher = forms.ChoiceField(label='Teacher', choices=[('rules', 'Rules (offline)'), ('http', 'HTTP teacher')])
    teacher_url = forms.URLField(label='HTTP teacher URL', required=False, assume_scheme='http')
    model = forms.CharField(label='Model name', initial='http-teacher', required=False, max_length=100)
    text = forms.CharField(label='Text / verbs', widget=forms.Textarea(attrs={'rows': 5}), required=False, max_length=50000)

    def __init__(self, *args, operations=None, **kwargs):
        super().__init__(*args, **kwargs)
        if operations:
            self.fields['operation'].choices = [(k, v) for k, v in OPERATIONS if k in operations]
        self.fields['source'].choices = [('', 'Select a file')] + [(s['path'], s['path']) for s in catalog.sources()]
        self.fields['store'].choices = [('', 'Select a corpus')] + [(s['path'], s['path']) for s in catalog.stores()]
        self.fields['pool'].choices = catalog.pool_choices()
        # Defaults apply to options hidden for the selected operation as well.
        for name in ('count', 'seed', 'split', 'input_format', 'context_turns', 'scheme', 'teacher'):
            self.fields[name].required = False
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-select' if isinstance(field.widget, forms.Select) else 'form-control'

    def clean(self):
        data = super().clean()
        defaults = dict(count=100, seed=42, split='train', input_format='text', context_turns=2,
                        scheme='BIO', teacher='rules', model='http-teacher')
        for key, value in defaults.items():
            if data.get(key) in (None, ''):
                data[key] = value
        operation = data.get('operation')
        if operation in ('ingest', 'bank', 'dailydialog', 'validate', 'roundtrip', 'mining'):
            if not data.get('source'):
                self.add_error('source', 'Select a source file.')
            else:
                try:
                    catalog.data_path(data['source'])
                except ValueError as error:
                    self.add_error('source', str(error))
        if operation in ('annotate', 'questions', 'export', 'report'):
            try:
                if not data.get('store'):
                    raise ValueError('Select a source corpus.')
                catalog.store_path(data['store'])
            except ValueError as error:
                self.add_error('store', str(error))
        if operation == 'annotate' and data.get('teacher') == 'http':
            if not data.get('teacher_url'):
                self.add_error('teacher_url', 'Enter the HTTP teacher URL.')
            elif not data['teacher_url'].startswith(('http://', 'https://')):
                self.add_error('teacher_url', 'Only HTTP and HTTPS are supported.')
        if operation in ('lookup', 'candidates') and not data.get('text', '').strip():
            self.add_error('text', 'Enter text to process.')
        return data
