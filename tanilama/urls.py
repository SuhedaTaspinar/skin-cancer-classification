from django.urls import path
from . import views

urlpatterns = [
    path('', views.tanila, name='tanila'),  # Ana sayfa, tanila view fonksiyonuna yönlendirir
]
