from django.urls import path
from . import views

urlpatterns = [
    path('', views.search_gui, name='root'),  # new root
    path('search_gui/', views.search_gui, name="search_gui")
]
