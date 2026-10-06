from django import forms
from django.conf import settings

from semantic_corpus.semantic_generator import DEFAULT_FRAMES

from . import catalog
from .operation_registry import OperationRegistry
from .operations import validate_operation


class JobForm(forms.Form):
    operation = forms.ChoiceField(label='Operation')
    source = forms.ChoiceField(label='Source file', required=False)
    store = forms.ChoiceField(label='Source corpus', required=False)
    input_format = forms.ChoiceField(label='Text format', choices=[
        ('text', 'TXT / JSONL with a text field'), ('dialogue', 'JSONL: list of dialogue turns')])
    count = forms.IntegerField(label='Situations per frame', min_value=1, max_value=1000000, initial=100)
    template_count = forms.IntegerField(label='Examples per template', min_value=1, max_value=1000000,
        initial=1, required=False, help_text='Every matching template is included. Start with 1; many templates have only one attested example.')
    template_verb = forms.CharField(label='Verb or phrasal verb', required=False, max_length=200,
        help_text='Optional lemma, for example buy or pick up. Leave blank for all verbs.')
    template_adjective = forms.CharField(label='Adjective', required=False, max_length=200,
        help_text='Optional lemma, for example beautiful. Leave blank for all adjectives.')
    recombine = forms.BooleanField(label='Recombine observed slot fillers', required=False,
        help_text='Creates new combinations that need grammatical and semantic review. Unchecked: use attested combinations.')
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
        self.fields['operation'].choices = OperationRegistry.default().choices(operations)
        self.fields['source'].choices = [('', 'Select a file')] + [(s['path'], s['path']) for s in catalog.sources()]
        self.fields['store'].choices = [('', 'Select a corpus')] + [(s['path'], s['path']) for s in catalog.stores()]
        self.fields['pool'].choices = catalog.pool_choices()
        self.fields['count'].widget.attrs['data-frame-count'] = len(DEFAULT_FRAMES)
        self.fields['count'].help_text = (
            f'Distinct situations for each of {len(DEFAULT_FRAMES)} frames. '
            'Total = this value × number of frames. Each situation produces multiple QA examples. '
            'If the selected pool is too small, reduce this value or expand the pool.')
        # Defaults apply to options hidden for the selected operation as well.
        for name in ('count', 'seed', 'split', 'input_format', 'context_turns', 'scheme', 'teacher'):
            self.fields[name].required = False
        for field in self.fields.values():
            field.widget.attrs['class'] = ('form-select' if isinstance(field.widget, forms.Select) else
                'form-check-input' if isinstance(field.widget, forms.CheckboxInput) else 'form-control')

    def clean_operation(self):
        code = self.cleaned_data['operation']
        try:
            validate_operation(code)
        except ValueError as error:
            raise forms.ValidationError(str(error)) from error
        return code

    def clean(self):
        data = super().clean()
        defaults = dict(count=100, seed=42, split='train', input_format='text', context_turns=2,
                        scheme='BIO', teacher='rules', model='http-teacher', template_count=1)
        for key, value in defaults.items():
            if data.get(key) in (None, ''):
                data[key] = value
        operation = data.get('operation')
        if operation == 'build':
            data['count_scope'] = 'per_frame'
        if operation == 'templates':
            from semantic_corpus.template_bank import TemplateBank
            try:
                TemplateBank(settings.REPO_ROOT / 'data/eng-base-templates')
            except (OSError, ValueError) as error:
                self.add_error('operation', f'The compiled template bank is not available: {error}')
            for name in ('template_verb', 'template_adjective'):
                data[name] = data.get(name, '').strip().lower()
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
