from django.conf.urls import url

from . import views

app_name = 'synaptogram'

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^coll_list$', views.coll_list, name='coll_list'),
    url(r'^exp_list/(?P<coll>[-\w]+)/$', views.exp_list, name='exp_list'),
    url(r'^cutout/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/$',
        views.cutout, name='cutout'),
    url(r'^sgram/$', views.sgram, name='sgram'),
    url(r'^ndviz_url/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/(?P<channel>[-\w]+)/$',
        views.get_ndviz_url, name='ndviz_url'),
    url(r'^ndviz_url/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/$',
        views.get_ndviz_url, {'channel': None}, name='ndviz_url'),
    url(r'^ndviz_url_list/$', views.ndviz_url_list, name='ndviz_url_list'),
    url(r'^tiff_stack/$', views.tiff_stack, name='tiff_stack'),
    url(r'^tiff_stack_channel/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/(?P<x>[-\w:]+)/(?P<y>[-\w:]+)/(?P<z>[-\w:]+)/(?P<channel>[-\w:]+)/(?P<res>[-\w:]+)/$',
        views.tiff_stack_channel, name='tiff_stack_channel'),
    url(r'^zip_tiff_stacks/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/(?P<x>[-\w:]+)/(?P<y>[-\w:]+)/(?P<z>[-\w:]+)/(?P<channels>[-\w,]+)/(?P<res>[-\w:]+)/$',
        views.zip_tiff_stacks, name='zip_tiff_stacks'),
    url(r'^sgram_from_ndviz/$', views.sgram_from_ndviz, name='sgram_from_ndviz'),
    url(r'^start_downsample/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/(?P<channel>[-\w]+)/$',
        views.start_downsample, name='start_downsample'),
    url(r'^stop_downsample/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/(?P<channel>[-\w]+)/$',
        views.stop_downsample, name='stop_downsample'),
    url(r'^channel_detail/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/(?P<channel>[-\w]+)/$',
        views.channel_detail, name='channel_detail'),
    url(r'^avatr_pull/$',
        views.avatr_pull, name='avatr_pull'),
    url(r'^avatr_push/$',
        views.avatr_push, name='avatr_push'),
]
