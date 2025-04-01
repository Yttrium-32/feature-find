from django.urls import path
from . import views

urlpatterns = [
        path('', views.search_gui, name='search_gui')
]

