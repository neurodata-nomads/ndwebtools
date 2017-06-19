from django.conf.urls import url

from . import views

app_name = 'synaptogram'

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^sign_up$', views.sign_up, name='sign_up'),
    url(r'^login$', views.login, name='login'),
    url(r'^coll_list$', views.coll_list, name='coll_list'),
    url(r'^exp_list/(?P<coll>[-\w]+)/$', views.exp_list, name='exp_list'),
    url(r'^cutout/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/$', views.cutout, name='cutout'),
    url(r'^sgram/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/(?P<x>[-\w:]+)/(?P<y>[-\w:]+)/(?P<z>[-\w:]+)/$', 
            views.sgram, name='sgram'),
    url(r'^cut_url_list/(?P<coll>[-\w]+)/(?P<exp>[-\w]+)/(?P<x>[-\w:]+)/(?P<y>[-\w:]+)/(?P<z>[-\w:]+)/$', 
            views.cut_url_list, name='cut_url_list'),
]
