"""app URL Configuration"""

from django.urls import path

from app import views
from app.service import TrainigService

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('predict/<str:image_url>', views.get_prediction)
    # path('predict', views.post_prediction, name = 'post_prediction'),
]

TrainigService().train()
