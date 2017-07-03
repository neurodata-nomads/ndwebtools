from django import forms
from django.core import validators

class CutoutForm(forms.Form):
    def __init__(self,*args,**kwargs):
        CHOICES = kwargs.pop('channels')
        super(CutoutForm, self).__init__(*args, **kwargs)
        self.fields['channels'] = forms.MultipleChoiceField(
            label='Channels:',choices=[(c, c) for c in CHOICES], 
            widget=forms.CheckboxSelectMultiple,
            initial= [c for c in CHOICES])
        # self.fields['channels'] = forms.ChoiceField(label='Channels',
        #     choices=[(c, c) for c in CHOICES])
        
    x_min = forms.IntegerField(label='x_min')
    x_max = forms.IntegerField(label='x_max')

    y_min = forms.IntegerField(label='y_min')
    y_max = forms.IntegerField(label='y_max')
    
    z_min = forms.IntegerField(label='z_min')
    z_max = forms.IntegerField(label='z_max')

    ENDPOINTS = (
        ('sgram','Synaptogram'),
        ('ndviz','Neurodata Viz links per channel'),
        ('cut_urls','Cut URLS per channel'),
        ('tiff_stack','Download TIFF stack per channel'),
        )
    endpoint = forms.ChoiceField(label='Return:',choices=ENDPOINTS, 
        widget=forms.RadioSelect())