from django import forms
from .models import Prediction

ALLOWED_TYPES = ('image/jpeg', 'image/png', 'image/jpg')
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class UploadForm(forms.Form):
    image = forms.ImageField(
        label='Fundus Image',
        help_text='JPG or PNG, max 10 MB',
    )

    def clean_image(self):
        img = self.cleaned_data['image']
        if img.content_type not in ALLOWED_TYPES:
            raise forms.ValidationError('Only JPG and PNG images are accepted.')
        if img.size > MAX_UPLOAD_BYTES:
            raise forms.ValidationError('Image must be smaller than 10 MB.')
        return img


class NotesForm(forms.ModelForm):
    class Meta:
        model = Prediction
        fields = ('notes',)
        widgets = {
            'notes': forms.Textarea(attrs={
                'rows': 4,
                'placeholder': 'Add clinical notes…',
                'class': 'notes-textarea',
            }),
        }


class HistoryFilterForm(forms.Form):
    GRADE_CHOICES = [('', 'All Grades')] + [(str(i), lbl) for i, lbl in [
        (0, 'No DR'), (1, 'Mild NPDR'), (2, 'Moderate NPDR'),
        (3, 'Severe NPDR'), (4, 'Proliferative DR'),
    ]]
    grade = forms.ChoiceField(choices=GRADE_CHOICES, required=False)
    date_from = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    date_to = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
