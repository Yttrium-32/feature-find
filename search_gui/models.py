from django.db import models

class ImageFeatures(models.Model):
    image_id = models.AutoField(primary_key=True)
    image_hash = models.CharField(max_length=64, unique=True)
    image_name = models.CharField(max_length=255)
    features = models.JSONField()
