from django.conf.urls import url

from . import views

app_name = 'synaptogram'

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^sign_up$', views.sign_up, name='sign_up'),
    url(r'^login$', views.login, name='login'),
    url(r'^logout$', views.logout, name='logout'),
    url(r'^coll_list$', views.coll_list, name='coll_list'),
    url(r'^exp_list/(?P<coll>[-\w]+)/$', views.exp_list, name='exp_list'),
    url(r'^cutout/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/$', views.cutout, name='cutout'),
    url(r'^sgram/$',views.sgram, name='sgram'),
    url(r'^cut_url_list/$', views.cut_url_list, name='cut_url_list'),
    url(r'^ndviz_url_list/$', views.ndviz_url_list, name='ndviz_url_list'),
    url(r'^tiff_stack/$',views.tiff_stack, name='tiff_stack'),
    url(r'^tiff_stack_channel/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/(?P<x>[-\w:]+)/(?P<y>[-\w:]+)/(?P<z>[-\w:]+)/(?P<channel>[-\w:]+)/$', 
            views.tiff_stack_channel, name='tiff_stack_channel'),
    url(r'^zip_tiff_stacks/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/(?P<x>[-\w:]+)/(?P<y>[-\w:]+)/(?P<z>[-\w:]+)/(?P<channels>[-\w,]+)/$',
            views.zip_tiff_stacks, name='zip_tiff_stacks'),
    url(r'^sgram_from_ndviz/$', views.sgram_from_ndviz, name='sgram_from_ndviz'),
]
