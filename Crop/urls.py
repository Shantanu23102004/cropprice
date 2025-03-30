from django.urls import path
from .views import predict_price,predict_price_page

urlpatterns = [
    path('', predict_price_page, name='predict_page'),
    path('predict/', predict_price, name='predict_price'),
]

