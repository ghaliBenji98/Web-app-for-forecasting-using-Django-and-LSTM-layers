
from django.urls import path
from . import views

urlpatterns = [
    path('forecast', views.forecast, name='forecast'),

]
