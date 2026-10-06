from django import forms
from django.conf import settings

from semantic_corpus.template_bank import TemplateBank

from . import catalog
from .operation_registry import OperationRegistry
from .operations import validate_operation


class JobForm(forms.Form):
    operation = forms.ChoiceField(label='Operation')
    source = forms.ChoiceField(label='Source file', required=False)
    store = forms.ChoiceField(label='Source corpus', required=False)
    input_format = forms.ChoiceField(label='Text format', choices=[
        ('text', 'TXT / JSONL with a text field'), ('dialogue', 'JSONL: list of dialogue turns')])
    count = forms.IntegerField(label='Situations per frame', min_value=1, max_value=1000000, initial=20)
    template_count = forms.IntegerField(label='Examples per mined template', min_value=0, max_value=1000000,
        initial=1, required=False,
        help_text='Every mined template is included. Set 0 to skip mined templates. '
                  'Quota is before split filtering and deduplication. '
                  'Many templates have only one attested example.')
    recombine = forms.BooleanField(label='Recombine observed slot fillers', required=False,
        help_text='Creates new combinations that need grammatical and semantic review. Unchecked: use attested combinations.')
    limit = forms.IntegerField(label='Input document limit', min_value=1, required=False,
                              help_text='Leave blank to process the entire input corpus.')
    seed = forms.IntegerField(label='Seed', initial=42, min_value=0, max_value=2147483647)
    split = forms.ChoiceField(label='Split', choices=[(s, s) for s in ('train', 'dev', 'test')])
    pool = forms.ChoiceField(label='Entity component', required=False,
        help_text='Built-in entities or an uploaded replacement. Frames and mined resources remain in the shared Built-in pool.')
    context_turns = forms.IntegerField(label='Context: preceding turns', initial=2, min_value=0, max_value=100)
    scheme = forms.ChoiceField(label='Export tagging scheme', choices=[('BIO', 'BIO'), ('BILOU', 'BILOU')])
    max_per_kind = forms.IntegerField(label='Maximum QA per kind per document', min_value=1, required=False)
    no_answer_share = forms.FloatField(label='Maximum unanswerable share', min_value=0, max_value=1, required=False)
    teacher = forms.ChoiceField(label='Teacher', choices=[('rules', 'Rules (offline)'), ('http', 'HTTP teacher')])
    pending_only = forms.BooleanField(label='Only documents awaiting annotation', initial=True, required=False,
        help_text='Annotate documents without a run and incomplete mined annotations. Preserve completed runs.')
    teacher_url = forms.URLField(label='HTTP teacher URL', required=False, assume_scheme='http')
    model = forms.CharField(label='Model name', initial='http-teacher', required=False, max_length=100)
    text = forms.CharField(label='Text / verbs', widget=forms.Textarea(attrs={'rows': 5}), required=False, max_length=50000)

    def __init__(self, *args, operations=None, resource_scope='built_in_v1', **kwargs):
        super().__init__(*args, **kwargs)
        self.resource_scope = resource_scope
        choices = OperationRegistry.default().choices(operations)
        # Old template jobs remain executable and retryable, but new generation
        # always starts from Synthetic corpus and the shared resource pool.
        if operations is None:
            choices = [(code, label) for code, label in choices if code not in ('templates', 'template_mining')]
        self.fields['operation'].choices = choices
        self.fields['source'].choices = [('', 'Select a file')] + [(s['path'], s['path']) for s in catalog.sources()]
        self.fields['store'].choices = [('', 'Select a corpus')] + [(s['path'], s['path']) for s in catalog.stores()]
        self.fields['pool'].choices = catalog.pool_choices()
        self.resources = catalog.resource_summary()
        frame_count = self.resources['frames']
        self.fields['count'].widget.attrs['data-frame-count'] = frame_count
        self.fields['template_count'].widget.attrs['data-template-count'] = self.resources['templates']
        self.fields['count'].help_text = (
            f'Distinct situations for each of {frame_count} curated frames. '
            'Total = this value times the number of frames. Each situation produces multiple QA examples. '
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
            if self.resource_scope is not None:
                data['resource_scope'] = self.resource_scope
        uses_mined_templates = (operation == 'templates' or
                                (operation == 'build' and self.resource_scope == 'built_in_v1'))
        if uses_mined_templates:
            template_count = data['template_count']
            if operation == 'templates' and template_count == 0:
                self.add_error('template_count', 'Historical template jobs require at least one example per template.')
            try:
                if template_count > 0:
                    if operation == 'templates':
                        # Historical template jobs name the original bank only.
                        TemplateBank(settings.REPO_ROOT / 'data/eng-base-templates')
                    else:
                        catalog.get_resources().banks
            except (OSError, ValueError) as error:
                remedy = ('Restore them or set Examples per mined template to 0.' if operation == 'build'
                          else 'Restore them to retry this historical job.')
                self.add_error('template_count',
                    f'The mined resources in Built-in pool are not available: {error}. {remedy}')
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
