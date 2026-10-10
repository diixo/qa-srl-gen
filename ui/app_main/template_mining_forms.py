"""Inputs for the dedicated template-mining resource page."""

from django import forms

from .template_mining import resolve_source_path, source_file_suffix


class TemplateMiningForm(forms.Form):
    text = forms.CharField(
        label='',
        required=False,
        strip=False,
        max_length=500000,
        widget=forms.Textarea(attrs={
            'rows': 16,
            'class': 'form-control',
            'aria-label': 'English text',
            'placeholder': 'She picked up the book.\nThey visited a beautiful city.',
        }),
        help_text='Each non-empty line becomes one source record. '
                  'For larger inputs, select a file below.',
    )
    source_file = forms.FileField(
        label='Source file',
        required=False,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.txt,.jsonl,.jsonl.gz',
        }),
        help_text='Choose a file from your computer. Supports UTF-8 TXT '
                  '(one record per non-empty line), JSONL and JSONL.GZ '
                  'with an example or text field. The file is uploaded when you queue the job.',
    )

    def __init__(self, *args, source_path='', **kwargs):
        super().__init__(*args, **kwargs)
        # A retry reuses a saved upload; browser requests cannot supply a path.
        self.saved_source_path = source_path

    def clean_text(self):
        text = self.cleaned_data['text']
        return text if text.strip() else ''

    def clean_source_file(self):
        uploaded = self.cleaned_data['source_file']
        if uploaded is None:
            return None
        try:
            source_file_suffix(uploaded.name)
        except ValueError as error:
            raise forms.ValidationError(str(error)) from error
        return uploaded

    def clean(self):
        data = super().clean()
        data['source_path'] = ''
        if self.saved_source_path:
            try:
                data['source_path'] = str(resolve_source_path(self.saved_source_path))
            except (OSError, ValueError) as error:
                raise forms.ValidationError(str(error)) from error
        if not self.errors and not any(data.get(key) for key in ('text', 'source_file', 'source_path')):
            raise forms.ValidationError('Enter English text or choose a source file. You can also use both.')
        return data
