from django.urls import path
from . import views
from datasets.views import predict_bemba_api
from .views import (
    prediction_page,
    english_predictions,
    create_sms,
    read_all_sms,
    read_one_sms,
    update_sms,
    delete_sms,
)

app_name = 'action'

urlpatterns = [

    path('', prediction_page, name='prediction_page'),

    path('api/action/v1/english-predidctions/', english_predictions, name='english_predictions'),
    path('api/action/v1/predict-bemba/', predict_bemba_api, name='predict_bemba_api'),

    # CRUD API endpoints (v1)
    path('api/action/v1/sms/create/', create_sms, name='create_sms'),
    path('api/action/v1/sms/', read_all_sms, name='read_all_sms'),
    path('api/action/v1/sms/<int:pk>/', read_one_sms, name='read_one_sms'),
    path('api/action/v1/sms/update/<int:pk>/', update_sms, name='update_sms'),
    path('api/action/v1/sms/delete/<int:pk>/', delete_sms, name='delete_sms'),
]
