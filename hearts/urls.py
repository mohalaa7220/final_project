from django.urls import path
from .views import ProcessInput

urlpatterns = [
    path('process_text', ProcessInput.as_view(), name='process_input'),
]
