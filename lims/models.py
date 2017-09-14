from django.db import models

# Create your models here.


class Channel(models.Model):
    # - foreign key (boss ID) for each channel

    # ### Description of data:
    # - Voxel size

    # - Orientation
    orientation = models.CharField(max_length=3)

    # - Modality (explicit) - 2p, colm, lavision, ca imaging
    modality = models.CharField(max_length=200)

    # - Channel descriptions
    description = models.CharField(max_length=200)

    # - Scientific question
    scientific_question = models.CharField(max_length=200)

    # - Data owner (e.g. Brian, Matt, etc).
    # a link to keycloak user...?
    user = models.ForeignKey(User)

    # - Project grouping

    # - Anatomy Quirks ("missing olfactory")

    # - Sample ID (foreign key)
    sample_ID = models.CharField(max_length=100)

    # - Data quirks (missing sections, etc.)
    data_quirks = models.CharField(max_length=200)

    # - Other notes
    notes = models.CharField(max_length=200)

    # Annotations:
    # - RAMON IDs


class User(models.Model):


class Collection(models.Model):


class Experiment(models.Model):


class Protocol(models.Model):
    # - Method/protocol descriptions (if applicable)
    experiment = models.ForeignKey(Experiment)
    description = models.CharField(max_length=200)


class Annotation(models.Model):
    channel = models.ForeignKey(Channel)
