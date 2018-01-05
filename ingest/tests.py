from django.test import TestCase
from django.utils import timezone

from .models import IngestJob


class IngestJobTest(TestCase):

    def test_create_IngestJob(self):
        unique_id = 100
        boss_user_id = 1
        command_args = {'datasource': 'local'}
        z_start = 0
        z_end = 20
        ingest_job = IngestJob.objects.create(unique_id=unique_id, boss_user_id=boss_user_id,
                                              command_args=command_args, z_start=z_start,
                                              z_end=z_end)

        pkey = ingest_job.id
        q = IngestJob.objects.get(pk=pkey)

        self.assertEqual(z_start, q.z_start)
        self.assertEqual(None, q.end_time)
        self.assertEqual(boss_user_id, q.boss_user_id)

    def test_create_IngestJob_same_id(self):
        unique_id = 100
        boss_user_id = 1
        command_args = {'datasource': 'local'}
        z_start = 0
        z_end = 20
        ingest_job = IngestJob.objects.create(unique_id=unique_id, boss_user_id=boss_user_id,
                                              command_args=command_args, z_start=z_start,
                                              z_end=z_end)

        pkey = ingest_job.id
        q = IngestJob.objects.get(pk=pkey)

        self.assertEqual(z_start, q.z_start)
        self.assertEqual(None, q.end_time)
        self.assertEqual(boss_user_id, q.boss_user_id)
