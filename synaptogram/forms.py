from django import forms
from django.core import validators

class CutoutForm(forms.Form):
    def __init__(self,*args,**kwargs):
        CHOICES = kwargs.pop('channels')
        LIMITS = kwargs.pop('limits')
        super(CutoutForm, self).__init__(*args, **kwargs)
        self.fields['channels'] = forms.MultipleChoiceField(
            label='Channels:',choices=[(c, c) for c in CHOICES], 
            widget=forms.CheckboxSelectMultiple,
            initial= [c for c in CHOICES])
        # self.fields['channels'] = forms.ChoiceField(label='Channels',
        #     choices=[(c, c) for c in CHOICES])
        
        self.fields['x_min'] = forms.IntegerField(label='x_min',min_value=LIMITS['x_start'],max_value=LIMITS['x_stop'])
        self.fields['x_max'] = forms.IntegerField(label='x_max',min_value=LIMITS['x_start'],max_value=LIMITS['x_stop'])

        self.fields['y_min'] = forms.IntegerField(label='y_min',min_value=LIMITS['y_start'],max_value=LIMITS['y_stop'])
        self.fields['y_max'] = forms.IntegerField(label='y_max',min_value=LIMITS['y_start'],max_value=LIMITS['y_stop'])
    
        self.fields['z_min'] = forms.IntegerField(label='z_min',min_value=LIMITS['z_start'],max_value=LIMITS['z_stop'])
        self.fields['z_max'] = forms.IntegerField(label='z_max',min_value=LIMITS['z_start'],max_value=LIMITS['z_stop'])

    ENDPOINTS = (
        ('sgram','Synaptogram'),
        ('ndviz','Neurodata Viz links per channel'),
        ('tiff_stack','Download TIFF stack per channel'),
        ('cut_urls','Cut URLS per channel'),
        )
    endpoint = forms.ChoiceField(label='Return:',choices=ENDPOINTS, 
        widget=forms.RadioSelect())



# add validations -
#   x_min < x_max