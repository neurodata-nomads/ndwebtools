from django.shortcuts import render
from django.http import HttpResponse

from .models import IngestJob
# Create your views here.


def index(request):
    return HttpResponse('ello gov\'nor')


def list_jobs(request):
    jobs = IngestJob.objects.order_by('-start_time').values()
    # output = ', '.join([q.command_args for q in jobs])
    # return HttpResponse(output)
    context = {'jobs': jobs}
    return render(request, 'ingest/job_list.html', context)


def ingest_job_start(request, data):
    # takes command arguments POSTed from ingest_large_vol
    # unpickles them
    # creates the IngestJob entry in the database
    pass


def ingest_job_update(request, data):
    # when ingest_large_vol completes a portion of an ingest, it updates the
    # entry here by posting it's progress
    pass


def ingest_job_stop(request):
    # when ingest_large_vol finishes, it updates the entry
    pass
