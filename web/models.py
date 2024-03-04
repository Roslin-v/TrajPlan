from django.db import models


class Food(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=50, blank=True, null=True)
    score = models.FloatField(blank=True, null=True)
    comment = models.IntegerField(blank=True, null=True)
    category_id = models.IntegerField(blank=True, null=True)
    category = models.CharField(max_length=10, blank=True, null=True)
    price = models.IntegerField(blank=True, null=True)
    distance_from_center_km = models.FloatField(blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)
    latitude = models.FloatField(blank=True, null=True)
    position = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'food'


class Spot(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=50, blank=True, null=True)
    category = models.IntegerField(blank=True, null=True)
    score = models.FloatField(blank=True, null=True)
    comment = models.IntegerField(blank=True, null=True)
    price = models.IntegerField(blank=True, null=True)
    latitude = models.FloatField(blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)
    recommend_time = models.FloatField(blank=True, null=True)
    night_visit = models.IntegerField(blank=True, null=True)
    belong = models.IntegerField(blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    pic = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'spot'


class Traj(models.Model):
    user = models.IntegerField(primary_key=True)  # The composite primary key (user, day, seq) found, that is not supported. The first column is selected.
    day = models.IntegerField()
    seq = models.IntegerField()
    name = models.CharField(max_length=50, blank=True, null=True)
    poi = models.IntegerField(blank=True, null=True)
    time = models.FloatField(blank=True, null=True)
    norm = models.FloatField(blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'traj'
        unique_together = (('user', 'day', 'seq'),)


class Transportation(models.Model):
    trans_id = models.IntegerField(primary_key=True)  # The composite primary key (trans_id, seq) found, that is not supported. The first column is selected.
    trans = models.CharField(max_length=50, blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)
    latitude = models.FloatField(blank=True, null=True)
    seq = models.IntegerField()
    id = models.IntegerField(blank=True, null=True)
    name = models.CharField(max_length=50, blank=True, null=True)
    category = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'transportation'
        unique_together = (('trans_id', 'seq'),)


class User(models.Model):
    name = models.CharField(max_length=20)
    email = models.CharField(max_length=255)
    password = models.CharField(max_length=256)

    class Meta:
        managed = True
        db_table = 'user'


class Collection(models.Model):
    user_id = models.IntegerField(primary_key=True)
    plan_id = models.IntegerField()
    plan = models.TextField(blank=True, null=True)
    trans = models.TextField(blank=True, null=True)
    score = models.FloatField(blank=True, null=True)
    days = models.IntegerField(blank=True, null=True)
    budget = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'collection'
        unique_together = (('user_id', 'plan_id'),)


class Comment(models.Model):
    user_id = models.IntegerField()
    author = models.CharField(max_length=50)
    type = models.CharField(max_length=20)
    comment = models.TextField(blank=False, null=False)

    class Meta:
        managed = True
        db_table = 'comment'
