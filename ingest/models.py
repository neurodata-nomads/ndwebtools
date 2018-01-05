from django.db import models


class IngestJob(models.Model):
    # hash of the command arguments
    unique_id = models.CharField(max_length=200, unique=True)
    boss_user_id = models.IntegerField()
    command_args = models.TextField()
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True)
    post_errors = models.IntegerField(default=0)
    load_errors = models.IntegerField(default=0)
    z_start = models.IntegerField()
    z_end = models.IntegerField()
    z_current = models.IntegerField(null=True)

    # add progress field that could be updated by ingest program?
    # add keys for each command line argument?
