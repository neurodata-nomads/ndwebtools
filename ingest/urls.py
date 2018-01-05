from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('list_jobs', views.list_jobs, name='list_jobs'),
]
